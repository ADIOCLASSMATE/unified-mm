#!/usr/bin/env python3
"""Direct download -> 512px view -> Qwen pair -> Codex/sol judgement + fallback.

Input JSONL: source, source_id, url (or local_path), split=train, optional
capabilities, required_boxes (EXIF-normalized xyxy), parent_id. One controller
owns each output directory; independent manifest partitions scale across hosts.
"""

from __future__ import annotations

import argparse
import asyncio
from concurrent.futures import ProcessPoolExecutor
import fcntl
import hashlib
import io
import json
import multiprocessing
import os
from pathlib import Path
import random
import shutil
import sqlite3
import struct
import tarfile
import tempfile
import time
import uuid
from urllib.parse import urlparse

import httpx

from utils.image_shard_io import read_image_bytes
from utils.direct_network import check_direct_routes, direct_ssl_context
from utils.image_text_preprocessing import VIEW_VERSION, prepare_view
from utils.image_text_teacher import (
    EFFORT, PRIMARY_MODEL, FINAL_MODEL, PROMPT, PROMPT_VERSION, SCHEMA, PAIR_TEMPLATE,
    JUDGE_PROMPT, JUDGE_SCHEMA, ROUTING_POLICY, QwenGenerator, CodexFinalTeacher,
    load_qwen_settings, parse_judgement,
)
from utils.imagenet_synthetic_text_index import INDEX_SCHEMA


def digest_file(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            result.update(chunk)
    return result.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


class DirectDownloader:
    def __init__(self, connections=64, per_host=8, max_bytes=20 << 20):
        # Explicit transport disables environment HTTP(S)/ALL/SOCKS proxies.
        self.client = httpx.AsyncClient(
            trust_env=False, proxy=None, follow_redirects=False, http2=True,
            verify=direct_ssl_context(),
            timeout=httpx.Timeout(20, connect=5),
            limits=httpx.Limits(max_connections=connections, max_keepalive_connections=connections),
        )
        self.per_host, self.max_bytes = per_host, max_bytes
        self.hosts = {}
        self.host_failures, self.host_retry_at = {}, {}

    async def get(self, row):
        if row.get("local_path"):
            return await asyncio.to_thread(read_image_bytes, row["local_path"], max_bytes=self.max_bytes)
        initial_url = str(row["url"])
        origin = urlparse(initial_url).hostname
        if time.monotonic() < self.host_retry_at.get(origin, 0):
            raise ValueError("direct image host temporarily unavailable; retry later")
        for attempt in range(3):
            try:
                url = initial_url
                for _ in range(8):
                    parsed = urlparse(url)
                    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username:
                        raise ValueError("image URL must be an unauthenticated HTTP(S) URL")
                    semaphore = self.hosts.setdefault(parsed.hostname, asyncio.Semaphore(self.per_host))
                    async with semaphore, self.client.stream("GET", url) as response:
                        if response.status_code in {301, 302, 303, 307, 308}:
                            url = str(response.url.join(response.headers["location"]))
                            continue
                        response.raise_for_status()
                        if int(response.headers.get("content-length", 0)) > self.max_bytes:
                            raise ValueError("image exceeds byte limit")
                        data = bytearray()
                        async for chunk in response.aiter_bytes():
                            data.extend(chunk)
                            if len(data) > self.max_bytes:
                                raise ValueError("image exceeds byte limit")
                        self.host_failures[origin] = 0
                        self.host_retry_at.pop(origin, None)
                        return bytes(data)
                raise ValueError("image redirect limit exceeded")
            except httpx.HTTPStatusError as exc:
                code = exc.response.status_code
                if code != 429 and code < 500:
                    raise ValueError(f"image HTTP {code}") from None
                if attempt == 2:
                    self.mark_host_failure(origin)
                    raise ValueError(f"image HTTP {code} after retries") from None
            except httpx.TransportError:
                if attempt == 2:
                    self.mark_host_failure(origin)
                    raise ValueError("direct image transport failed after retries") from None
            await asyncio.sleep(2 ** attempt + random.random())

    def mark_host_failure(self, host):
        self.host_failures[host] = self.host_failures.get(host, 0) + 1
        if self.host_failures[host] >= 2:
            # Do not let a dead origin occupy all image workers indefinitely.
            # These tasks stay retryable; no proxy or alternate source is used.
            self.host_retry_at[host] = time.monotonic() + 120

    async def close(self):
        await self.client.aclose()


class ImageArchives:
    """Sequential tar writes; byte-range reads do not scan tar member tables."""
    def __init__(self, root, max_bytes=512 << 20):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        existing = list(self.root.glob("images-*.tar"))
        self.index = max((int(p.stem.split("-")[-1]) for p in existing), default=-1) + 1
        self.archive = self.handle = None
        self.max_bytes = max_bytes

    def add(self, name, data):
        if self.handle is None or self.handle.tell() + len(data) > self.max_bytes:
            self.close()
            self.path = (self.root / f"images-{self.index:06d}.tar").resolve()
            self.index += 1
            self.handle = self.path.open("xb")
            self.archive = tarfile.open(fileobj=self.handle, mode="w", format=tarfile.USTAR_FORMAT)
        offset = self.handle.tell() + 512
        info = tarfile.TarInfo(name)
        info.size = len(data)
        self.archive.addfile(info, io.BytesIO(data))
        self.handle.flush()
        return f"tar:{self.path}::{offset}:{len(data)}/{name}"

    def sync(self):
        if self.handle is not None:
            self.handle.flush()
            os.fsync(self.handle.fileno())

    def close(self):
        if self.archive is not None:
            self.archive.close()
            self.sync()
            self.handle.close()
            self.archive = self.handle = None


class State:
    def __init__(self, root, archives=None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = (self.root / ".controller.lock").open("a")
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.db = sqlite3.connect(self.root / "state.sqlite3")
        # One writer per partition; rollback journal also works on shared storage.
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("CREATE TABLE IF NOT EXISTS tasks (key TEXT PRIMARY KEY, row TEXT NOT NULL, status TEXT NOT NULL, view TEXT, raw TEXT, result TEXT, error TEXT)")
        columns = {row[1] for row in self.db.execute("PRAGMA table_info(tasks)")}
        if "judge_raw" not in columns:
            self.db.execute("ALTER TABLE tasks ADD COLUMN judge_raw TEXT")
        self.db.execute("CREATE TABLE IF NOT EXISTS judge_batches (id TEXT PRIMARY KEY, raw TEXT NOT NULL, image_ids TEXT NOT NULL)")
        self.db.execute("CREATE TABLE IF NOT EXISTS images (sha TEXT PRIMARY KEY, key TEXT NOT NULL)")
        # Disk-backed uniqueness within this invocation prevents duplicate input
        # rows from issuing concurrent requests for the same image.
        self.db.execute("PRAGMA temp_store=FILE")
        self.db.execute("CREATE TEMP TABLE enqueued (key TEXT PRIMARY KEY)")
        self.archives = archives
        self.last_commit = time.monotonic()

    def commit(self, force=False):
        if force or time.monotonic() - self.last_commit > 1:
            if self.archives is not None:
                self.archives.sync()
            self.db.commit()
            self.last_commit = time.monotonic()

    def update(self, key, status, **fields):
        if not fields.keys() <= {"view", "raw", "judge_raw", "result", "error"}:
            raise ValueError("unknown task state field")
        assignments = ",".join(f"{field}=?" for field in fields)
        self.db.execute(f"UPDATE tasks SET status=?{',' + assignments if assignments else ''} WHERE key=?",
                        (status, *fields.values(), key))
        self.commit()

    def close(self):
        self.commit(force=True)
        self.db.close()
        self.lock.close()


def validate_pair(raw, key, tokenizer):
    if raw.get("status") != "completed":
        raise ValueError("teacher response is incomplete")
    text = raw["output_text"].strip()
    # SII may wrap otherwise valid JSON in a single Markdown fence. Removing
    # only that wrapper avoids spending a sol rewrite on good Qwen labels.
    lines = text.splitlines()
    if len(lines) >= 3 and lines[0].strip() in {"```", "```json"} and lines[-1].strip() == "```":
        text = "\n".join(lines[1:-1])
    value = json.loads(text)
    if value.get("image_id") != key:
        raise ValueError("teacher image_id mismatch")
    for task in ("i2t", "t2i"):
        text = value.get(task)
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"empty {task}")
        # Reserve 61 tokens for task prefix; loader has 1021 total text tokens.
        if len(tokenizer.encode(text, add_special_tokens=False)) > 960:
            raise ValueError(f"{task} exceeds frozen training token budget")
        if type(value.get("usable", {}).get(task)) is not bool:
            raise ValueError("invalid usable flag")
    if not isinstance(value.get("capabilities"), list) or not isinstance(value.get("uncertainties"), list):
        raise ValueError("invalid capabilities/uncertainties")
    if not all(isinstance(item, str) for item in value["capabilities"] + value["uncertainties"]):
        raise ValueError("non-string capability/uncertainty")
    if not isinstance(value.get("observations"), dict):
        raise ValueError("missing candidate observations")
    observations = value["observations"]
    for field in ("relations", "visible_text"):
        if not isinstance(observations.get(field), list) or not all(isinstance(v, str) for v in observations[field]):
            raise ValueError(f"invalid {field} observations")
    if not isinstance(observations.get("counts"), list):
        raise ValueError("invalid count observations")
    for count in observations["counts"]:
        if (not isinstance(count, dict) or not isinstance(count.get("entity"), str)
                or type(count.get("count")) is not int or count["count"] < 0):
            raise ValueError("invalid count observation")
    return value


def excluded(row, exclusions):
    identities = {f"{row['source']}:{row['source_id']}"}
    if row.get("parent_id"):
        identities.add(f"{row['source']}:{row['parent_id']}")
    return bool(identities & exclusions) or row.get("split", "train") != "train"


def reused_candidate(row, key):
    """Reuse only synthetic labels; the final teacher must recheck frozen pixels."""
    reuse = row["reuse_pair"]
    if (reuse.get("i2t_source") == "original" or reuse.get("t2i_style") != "faithful_photo"
            or not all(isinstance(reuse.get(k), str) and reuse[k] for k in
                       ("i2t", "t2i", "i2t_model", "t2i_model", "source_dataset", "i2t_source"))):
        raise ValueError("reuse requires synthetic I2T and faithful-photo T2I with original provenance")
    value = {"image_id": key, "i2t": reuse["i2t"], "t2i": reuse["t2i"],
             "observations": {"counts": [], "relations": [], "visible_text": []},
             "capabilities": [], "uncertainties": [], "usable": {"i2t": True, "t2i": True}}
    return {"status": "completed", "output_text": json.dumps(value),
            "reuse": {k: v for k, v in reuse.items() if k not in {"i2t", "t2i"}}}


def resolve_final_pair(candidate, primary_errors, verdict, key, tokenizer, batch_id, reuse=None):
    decision = verdict["decision"]
    provenance = {
        "primary_model": PRIMARY_MODEL, "judge_model": FINAL_MODEL,
        "judge_effort": EFFORT, "judge_backend": "codex_cli",
        "judge_batch_id": batch_id, "decision": decision,
        "primary_errors": primary_errors,
    }
    if decision == "reject":
        return "review", {"provenance": {**provenance, "generator_model": None},
                          "issues": verdict["issues"]}
    if decision == "accept":
        if candidate is None or primary_errors or not all(candidate["usable"].values()):
            raise ValueError("final teacher cannot accept a mechanically invalid/unusable primary pair")
        result, author = candidate, reuse["i2t_model"] if reuse else PRIMARY_MODEL
    else:
        result = validate_pair({"status": "completed", "output_text": json.dumps(verdict["replacement"])},
                               key, tokenizer)
        author = FINAL_MODEL
    status = "ready" if all(result["usable"].values()) else "review"
    authors = {"i2t": author, "t2i": reuse["t2i_model"] if reuse and decision == "accept" else author}
    return status, {**result, "provenance": {**provenance, "generator_model": author},
                    "generator_models": authors,
                    "reuse_candidate": reuse, "reused_unchanged": bool(reuse and decision == "accept"),
                    "judge_issues": verdict["issues"]}


async def run_pipeline(args, teacher=None, tokenizer=None, generator=None):
    if args.partition < 0 or args.partition >= args.partitions:
        raise ValueError("partition must be in [0, partitions)")
    if min(args.download_workers, args.cpu_workers, args.qwen_workers,
           args.teacher_workers, args.per_host, args.judge_batch_size) < 1:
        raise ValueError("worker/batch counts must be positive")
    if args.judge_flush_seconds <= 0:
        raise ValueError("judge flush interval must be positive")
    routes = check_direct_routes()
    qwen_settings = load_qwen_settings(args.qwen_example, require_api_key=not args.prepare_only)
    exclusions = {line.strip() for line in Path(args.exclude).read_text().splitlines() if line.strip()}
    fingerprint = {
        "manifest_sha256": digest_file(args.manifest), "exclude_sha256": digest_file(args.exclude),
        "partition": args.partition, "partitions": args.partitions,
        "routing_policy": ROUTING_POLICY, "primary": qwen_settings.public_contract(),
        "final_teacher": {"model": FINAL_MODEL, "reasoning_effort": EFFORT, "backend": "codex_cli",
                          "batch_size": args.judge_batch_size},
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
        "schema_sha256": hashlib.sha256(json.dumps(SCHEMA, sort_keys=True).encode()).hexdigest(),
        "primary_template_sha256": hashlib.sha256(json.dumps(PAIR_TEMPLATE, sort_keys=True).encode()).hexdigest(),
        "judge_prompt_sha256": hashlib.sha256(JUDGE_PROMPT.encode()).hexdigest(),
        "judge_schema_sha256": hashlib.sha256(json.dumps(JUDGE_SCHEMA, sort_keys=True).encode()).hexdigest(),
        "view_version": VIEW_VERSION, "image_size": 512, "min_short_side": args.min_short_side,
        "tokenizer": str(Path(args.tokenizer).resolve()),
    }
    near = None
    if getattr(args, "near_exclude_index", None):
        from utils.image_near_duplicates import NearDuplicateIndex
        near = NearDuplicateIndex(Path(args.near_exclude_index).resolve())
        fingerprint["near_exclusion_index"] = {
            "path": str(near.root),
            "sha256": {name: digest_file(near.root / name) for name in
                       ("index.json", "hashes.npy", "source_ids.npy", "orders.npy", "offsets.npy", "sources.json")},
        }
    root = Path(args.output).resolve()
    image_root = Path(getattr(args, "image_root", None) or root / "images").resolve()
    fingerprint["image_root"] = str(image_root)
    archives = ImageArchives(image_root)
    state = State(root, archives)
    downloader = pool = None
    try:
        run_path = root / "run.json"
        if run_path.exists() and json.loads(run_path.read_text())["contract"] != fingerprint:
            raise ValueError("run contract changed; use a new output directory")
        if not run_path.exists():
            atomic_json(run_path, {"contract": fingerprint, "direct_route_devices": routes})
        if not args.prepare_only and tokenizer is None:
            os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
        if not args.prepare_only:
            if generator is None:
                generator = QwenGenerator(qwen_settings, args.rpm, args.tpm, args.qwen_workers)
            if teacher is None:
                teacher = CodexFinalTeacher(args.codex_bin, args.codex_timeout)
        downloader = DirectDownloader(args.download_workers, args.per_host)
        pool = ProcessPoolExecutor(max_workers=args.cpu_workers,
                                   mp_context=multiprocessing.get_context("spawn"))
        queue_in = asyncio.Queue(maxsize=args.download_workers * 2)
        queue_primary = asyncio.Queue(maxsize=args.qwen_workers * 2)
        queue_cached = asyncio.Queue(maxsize=args.teacher_workers * args.judge_batch_size * 2)
        queue_judge = asyncio.Queue(maxsize=args.teacher_workers * args.judge_batch_size * 2)
        queue_judge_batches = asyncio.Queue(maxsize=args.teacher_workers * 2)
        started, last_report = time.monotonic(), time.monotonic()
        completed = 0
        failure_streak = {"qwen": 0, "sol": 0}

        def report():
            nonlocal last_report, completed
            completed += 1
            if time.monotonic() - last_report > 30:
                print(json.dumps({"processed_this_run": completed,
                                  "elapsed_seconds": round(time.monotonic() - started, 1)}), flush=True)
                last_report = time.monotonic()

        def fail(key, phase, exc):
            detail = str(exc) if isinstance(exc, ValueError) else f"status={getattr(exc, 'status_code', None)}"
            cause = exc.__cause__
            if isinstance(cause, httpx.HTTPError):
                detail += f"; transport={type(cause).__name__}"
                # Only our numeric curl diagnostic is safe to log verbatim.
                if str(cause).startswith("SII direct transport exited with code "):
                    detail += "; " + str(cause)
            state.update(key, "failed", error=f"{phase}: {type(exc).__name__}; {detail}")
            failure_streak[phase] += 1
            return (getattr(exc, "status_code", None) in {401, 403, 404}
                    or getattr(exc, "code", None) in {"insufficient_quota", "model_not_found", "invalid_api_key"}
                    or failure_streak[phase] >= 32)

        def finalize(sample, verdict, batch_id):
            status, result = resolve_final_pair(sample["candidate"], sample["primary_errors"], verdict,
                                                sample["key"], tokenizer, batch_id, sample.get("reuse"))
            state.update(sample["key"], status, result=json.dumps(result), error=None)
            failure_streak["sol"] = 0
            report()

        async def producer():
            with Path(args.manifest).open() as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    identity = f"{row['source']}:{row['source_id']}"
                    key = hashlib.sha256(identity.encode()).hexdigest()
                    if int(key[:16], 16) % args.partitions != args.partition:
                        continue
                    previous = state.db.execute(
                        "SELECT status,view,raw,judge_raw,row,error FROM tasks WHERE key=?", (key,)).fetchone()
                    serialized = json.dumps(row, sort_keys=True)
                    if previous:
                        status, view, raw, judge_raw, old_row, error = previous
                        if old_row != serialized:
                            raise ValueError(f"conflicting source identity: {identity}")
                        if status in {"ready", "review", "excluded", "duplicate"} or (status == "failed" and not args.retry_failed):
                            continue
                        retry_phase = getattr(args, "retry_phase", "all")
                        if status == "failed" and retry_phase != "all" and not (error or "").startswith(retry_phase + ":"):
                            continue
                        if (status == "failed" and error and error.startswith("prepare:")
                                and not any(part in error for part in (
                                    "transport failed", "temporarily unavailable", "HTTP 429", "HTTP 5"))):
                            continue
                        if args.prepare_only and status == "prepared":
                            continue
                        if status == "failed" and args.regenerate_failed:
                            # A timed-out teacher has no judge_raw, but its
                            # completed primary response must still be reused.
                            if judge_raw is not None or (error or "").startswith("sol:"):
                                if raw is None:
                                    raise ValueError("final-teacher retry requires a saved primary response")
                                judge_raw = None
                                state.update(key, "generated", judge_raw=None, result=None, error=None)
                            else:
                                raw = None
                                state.update(key, "prepared", raw=None, result=None, error=None)
                    else:
                        status, view, raw, judge_raw = "pending", None, None, None
                        state.db.execute("INSERT INTO tasks(key,row,status) VALUES(?,?,?)", (key, serialized, status))
                    if not state.db.execute("INSERT OR IGNORE INTO enqueued VALUES(?)", (key,)).rowcount:
                        continue
                    if excluded(row, exclusions):
                        state.update(key, "excluded", error="held-out source/image")
                        continue
                    await queue_in.put((key, row, json.loads(view) if view else None,
                                        json.loads(raw) if raw else None, json.loads(judge_raw) if judge_raw else None))
                    state.commit()
            for _ in range(args.download_workers):
                await queue_in.put(None)

        async def prepare_worker():
            while (item := await queue_in.get()) is not None:
                key, row, view, raw, judge_raw = item
                try:
                    if view is None:
                        data = await downloader.get(row)
                        source_sha = hashlib.sha256(data).hexdigest()
                        if row.get("expected_source_sha256") and row["expected_source_sha256"] != source_sha:
                            raise ValueError("source bytes differ from the reused caption's image")
                        if source_sha in exclusions:
                            state.update(key, "excluded", error="held-out source SHA256")
                            continue
                        pixels, view = await asyncio.get_running_loop().run_in_executor(
                            pool, prepare_view, data, row, args.min_short_side, near is not None)
                        if view["view_sha256"] in exclusions:
                            state.update(key, "excluded", error="held-out processed SHA256")
                            continue
                        if near and (hit := near.lookup(view["perceptual_hashes"])):
                            state.update(key, "excluded", error="held-out perceptual match: " + json.dumps(hit))
                            continue
                        existing = state.db.execute("SELECT key FROM images WHERE sha=?", (view["view_sha256"],)).fetchone()
                        if existing and existing[0] != key:
                            state.update(key, "duplicate", error=f"same frozen pixels as {existing[0]}")
                            continue
                        state.db.execute("INSERT OR IGNORE INTO images VALUES(?,?)", (view["view_sha256"], key))
                        view["original_ref"] = archives.add(key + ".original", data)
                        view["source_path"] = archives.add(key + "." + view["extension"], pixels)
                        state.update(key, "prepared", view=json.dumps(view))
                    else:
                        pixels = await asyncio.to_thread(read_image_bytes, view["source_path"])
                        if hashlib.sha256(pixels).hexdigest() != view["view_sha256"]:
                            raise ValueError("frozen image bytes changed")
                    if not args.prepare_only:
                        target = queue_cached if raw is not None or row.get("reuse_pair") else queue_primary
                        await target.put((key, row, view, pixels, raw, judge_raw))
                    else:
                        report()
                except (ValueError, OSError, KeyError, httpx.HTTPError) as exc:
                    state.update(key, "failed", error=f"prepare: {type(exc).__name__}: {exc}")

        async def primary_worker(queue):
            while (item := await queue.get()) is not None:
                key, row, view, pixels, raw, judge_raw = item
                phase = "qwen"
                try:
                    if raw is None:
                        if row.get("reuse_pair"):
                            raw = reused_candidate(row, key)
                        else:
                            for attempt in range(4):
                                try:
                                    raw = await generator.generate(key, pixels, view["extension"])
                                    # Reusing saved text says nothing about SII
                                    # availability; only a live success resets it.
                                    failure_streak["qwen"] = 0
                                    break
                                except Exception as exc:
                                    code = getattr(exc, "status_code", None)
                                    connect_failure = isinstance(exc.__cause__, (httpx.ConnectError, httpx.ConnectTimeout))
                                    if (code not in {429, 500, 502, 503, 504} and not connect_failure) or attempt == 3:
                                        raise
                                    await asyncio.sleep(2 ** attempt + random.random())
                        state.update(key, "generated", raw=json.dumps(raw))
                        state.commit(force=True)
                    # Quality/format failures go to sol; service failures above do
                    # not silently turn into an expensive all-sol generation run.
                    candidate, errors = None, []
                    try:
                        candidate = validate_pair(raw, key, tokenizer)
                        if not all(candidate["usable"].values()):
                            errors.append("primary pair has an unusable task")
                    except (ValueError, KeyError, TypeError, AttributeError) as exc:
                        errors.append(f"primary format/length failure: {type(exc).__name__}")
                    sample = {"key": key, "pixels": pixels, "extension": view["extension"],
                              "candidate": candidate, "primary_errors": errors, "reuse": raw.get("reuse")}
                    if judge_raw:
                        phase = "sol"
                        batch_raw, image_ids = state.db.execute(
                            "SELECT raw,image_ids FROM judge_batches WHERE id=?", (judge_raw["batch_id"],)).fetchone()
                        decisions = parse_judgement(json.loads(batch_raw), json.loads(image_ids))
                        finalize(sample, decisions[key], judge_raw["batch_id"])
                    else:
                        await queue_judge.put(sample)
                except Exception as exc:
                    if fail(key, phase, exc):
                        raise RuntimeError(f"{phase} service/configuration failed; progress saved") from None

        async def judge_batcher():
            # A single assembler prevents concurrent teachers from splitting
            # staggered arrivals into separate, mostly empty batches.
            ended = False
            while (first := await queue_judge.get()) is not None:
                batch = [first]
                deadline = time.monotonic() + args.judge_flush_seconds
                while len(batch) < args.judge_batch_size:
                    try:
                        item = queue_judge.get_nowait()
                    except asyncio.QueueEmpty:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            break
                        try:
                            item = await asyncio.wait_for(queue_judge.get(), remaining)
                        except asyncio.TimeoutError:
                            break
                    if item is None:
                        ended = True
                        break
                    batch.append(item)
                await queue_judge_batches.put(batch)
                if ended:
                    break
            for _ in range(args.teacher_workers):
                await queue_judge_batches.put(None)

        def save_judgement(batch, raw):
            batch_id = uuid.uuid4().hex
            ids = [sample["key"] for sample in batch]
            state.db.execute("INSERT INTO judge_batches VALUES(?,?,?)",
                             (batch_id, json.dumps(raw), json.dumps(ids)))
            for sample in batch:
                state.update(sample["key"], "judged", judge_raw=json.dumps({"batch_id": batch_id}))
            # Persist the entire received batch once, before parsing it.
            state.commit(force=True)
            return batch_id, ids

        async def judge_worker():
            while (batch := await queue_judge_batches.get()) is not None:
                raw = None
                try:
                    raw = await teacher.evaluate_batch(batch)
                    batch_id, ids = save_judgement(batch, raw)
                    decisions = parse_judgement(raw, ids)
                except Exception as exc:
                    fatal = False
                    for sample in batch:
                        fatal |= fail(sample["key"], "sol", exc)
                    if fatal or (raw is not None and raw.get("status") != "completed"):
                        raise RuntimeError("sol final teacher failed; received responses/progress saved") from None
                    continue
                for sample in batch:
                    try:
                        finalize(sample, decisions[sample["key"]], batch_id)
                    except (ValueError, KeyError, TypeError, AttributeError):
                        # A valid batch can still contain a malformed replacement
                        # (for example, its inner image_id is wrong). Rejudge only
                        # that image once; keep Qwen and all other decisions.
                        retry_raw = None
                        try:
                            retry_raw = await teacher.evaluate_batch([sample])
                            retry_raw["retry_of_batch_id"] = batch_id
                            retry_id, ids = save_judgement([sample], retry_raw)
                            retry_decisions = parse_judgement(retry_raw, ids)
                            finalize(sample, retry_decisions[sample["key"]], retry_id)
                        except Exception as exc:
                            fatal = fail(sample["key"], "sol", exc)
                            if fatal or (retry_raw is not None and retry_raw.get("status") != "completed"):
                                raise RuntimeError("sol final teacher retry failed; progress saved") from None

        async def preparation():
            async with asyncio.TaskGroup() as group:
                group.create_task(producer())
                for _ in range(args.download_workers):
                    group.create_task(prepare_worker())
            for _ in range(args.qwen_workers):
                await queue_primary.put(None)
            await queue_cached.put(None)

        async def primary_stage():
            async with asyncio.TaskGroup() as group:
                for _ in range(args.qwen_workers):
                    group.create_task(primary_worker(queue_primary))
                group.create_task(primary_worker(queue_cached))
            await queue_judge.put(None)

        async with asyncio.TaskGroup() as group:
            group.create_task(preparation())
            group.create_task(primary_stage())
            group.create_task(judge_batcher())
            for _ in range(args.teacher_workers):
                group.create_task(judge_worker())
        summary = dict(state.db.execute("SELECT status, COUNT(*) FROM tasks GROUP BY status"))
        authors = dict(state.db.execute(
            "SELECT json_extract(result,'$.provenance.generator_model'),COUNT(*) FROM tasks WHERE status='ready' GROUP BY 1"))
        atomic_json(root / "summary.json", {
            "counts": summary, "accepted_by_generator": authors,
            "judge_batches": state.db.execute("SELECT COUNT(*) FROM judge_batches").fetchone()[0],
            "elapsed_seconds": time.monotonic() - started,
        })
        print(json.dumps({"counts": summary, "accepted_by_generator": authors}, sort_keys=True), flush=True)
        return summary
    finally:
        if downloader is not None:
            await downloader.close()
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
        for client in (generator, teacher):
            if client is not None:
                await client.close()
        archives.close()
        state.close()


def export_training(root, destination, near_exclude_index=None):
    """Globally deduplicate and number approved rows from completed partitions."""
    roots = [Path(p).resolve() for p in (root if isinstance(root, (list, tuple)) else [root])]
    destination = Path(destination).resolve()
    contracts = [json.loads((p / "run.json").read_text())["contract"] for p in roots]
    if not contracts or any(c.get("routing_policy") != ROUTING_POLICY for c in contracts):
        raise ValueError("publication requires the Qwen + final-sol routing contract")
    common = ("image_size", "view_version", "tokenizer", "prompt_version", "prompt_sha256",
              "schema_sha256", "primary_template_sha256", "judge_prompt_sha256", "judge_schema_sha256")
    if any(any(c.get(key) != contracts[0].get(key) for key in common) for c in contracts[1:]):
        raise ValueError("source partitions have incompatible view/text contracts")
    if destination.exists():
        raise FileExistsError("training publication is immutable; choose a new directory")
    near = None
    if near_exclude_index:
        from utils.image_near_duplicates import NearDuplicateIndex, perceptual_hashes
        near = NearDuplicateIndex(near_exclude_index)
    states, temporary, dedup = [], None, None
    try:
        for path in roots:
            states.append(State(path))
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=destination.name + ".tmp-", dir=destination.parent))
        dedup_path = temporary / "dedup.sqlite3"
        dedup = sqlite3.connect(dedup_path)
        dedup.execute("PRAGMA synchronous=OFF")
        dedup.execute("PRAGMA journal_mode=OFF")
        dedup.execute("CREATE TABLE seen (identity TEXT PRIMARY KEY)")
        count, duplicates, near_hits = 0, 0, []
        with (temporary / "manifest.jsonl").open("w") as manifest, \
             (temporary / "captions.jsonl").open("wb") as captions, \
             (temporary / "captions.offsets.u64").open("wb") as caption_offsets, \
             (temporary / "t2i.jsonl").open("wb") as prompts, \
             (temporary / "t2i.offsets.u64").open("wb") as prompt_offsets, \
             (temporary / "t2i_mapping.bi").open("wb") as mapping:
            caption_offsets.write(struct.pack("<Q", 0))
            prompt_offsets.write(struct.pack("<Q", 0))
            rows = ((state.root, *row) for state in states for row in state.db.execute(
                "SELECT key,row,view,result FROM tasks WHERE status='ready' ORDER BY key"))
            for source_run, key, row, view, result in rows:
                row, view, result = json.loads(row), json.loads(view), json.loads(result)
                identities = ["key:" + key, "source:" + view["source_sha256"], "view:" + view["view_sha256"]]
                if dedup.execute("SELECT 1 FROM seen WHERE identity IN (?,?,?) LIMIT 1", identities).fetchone():
                    duplicates += 1
                    continue
                if near is not None:
                    values = set(perceptual_hashes(read_image_bytes(view["source_path"])))
                    values.update(perceptual_hashes(read_image_bytes(view["original_ref"])))
                    match = near.lookup(values)
                    if match:
                        near_hits.append({"key": key, "source": row["source"], "source_id": row["source_id"], **match})
                        continue
                provenance = result.get("provenance", {})
                reuse = result.get("reuse_candidate")
                expected_i2t_model = (reuse["i2t_model"] if result.get("reused_unchanged") and reuse else
                                      PRIMARY_MODEL if provenance.get("decision") == "accept" else FINAL_MODEL)
                if (provenance.get("judge_model") != FINAL_MODEL
                        or provenance.get("judge_effort") != EFFORT
                        or provenance.get("judge_backend") != "codex_cli"
                        or provenance.get("decision") not in {"accept", "replace"}
                        or provenance.get("generator_model") != expected_i2t_model
                        or (result.get("reused_unchanged") and provenance.get("decision") != "accept")):
                    raise ValueError("ready row lacks final-teacher approval and generator provenance")
                provenance = {**provenance, "generator_models": result["generator_models"],
                              "reused_unchanged": result["reused_unchanged"], "reuse_candidate": reuse,
                              "source_run": str(source_run)}
                image_id = count + 1
                manifest.write(json.dumps({**view, "img_id": image_id, "split": "train", "source": row["source"],
                    "source_id": row["source_id"], "key": key,
                    "selection_bucket": row.get("selection_bucket", row["source"]),
                    "capabilities": result["capabilities"], "observations": result["observations"],
                    "uncertainties": result["uncertainties"]}) + "\n")
                caption = {"manifest_index": count, "img_id": image_id, "provenance": provenance,
                           "captions": [{"source": provenance["generator_model"], "text": result["i2t"]}]}
                prompt = {"image_id": f"train/{key}", "provenance": provenance,
                          "model_result": {"prompts": [{"prompt": result["t2i"]}]}}
                for output, offsets, value in ((captions, caption_offsets, caption), (prompts, prompt_offsets, prompt)):
                    output.write((json.dumps(value, ensure_ascii=False) + "\n").encode())
                    offsets.write(struct.pack("<Q", output.tell()))
                mapping.write(struct.pack("<BI", 0, count))
                dedup.executemany("INSERT INTO seen VALUES(?)", [(identity,) for identity in identities])
                count += 1
        if not count:
            raise ValueError("no approved paired images to publish")
        atomic_json(temporary / "text_index.json", {
            "schema": INDEX_SCHEMA, "split": "train", "records": count,
            "caption": {"path": "captions.jsonl", "offsets_path": "captions.offsets.u64"},
            "t2i": {"shards": [{"shard_index": 0, "records": count, "path": "t2i.jsonl", "offsets_path": "t2i.offsets.u64"}]},
            "mapping": {"path": "t2i_mapping.bi"},
        })
        atomic_json(temporary / "publication.json", {"records": count, "image_size": 512,
                    "duplicate_images_dropped": duplicates,
                    "near_duplicate_images_dropped": len(near_hits),
                    "near_duplicate_rejections": near_hits,
                    "near_exclusion_index": {"path": str(near.root),
                        "hashes_sha256": digest_file(near.root / "hashes.npy")} if near else None,
                    "source_runs": [{"path": str(path), "contract": c} for path, c in zip(roots, contracts)],
                    "contract": {key: contracts[0][key] for key in common if key in contracts[0]}})
        dedup.close()
        dedup = None
        dedup_path.unlink()
        temporary.replace(destination)
        return count
    finally:
        for state in states:
            state.close()
        if dedup is not None:
            dedup.close()
        if temporary is not None and temporary.exists():
            shutil.rmtree(temporary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--manifest", required=True)
    run.add_argument("--exclude", required=True, help="One source:id or image SHA256 per line, covering benchmark parent images.")
    run.add_argument("--near-exclude-index", help="Screen frozen benchmark perceptual hashes before model calls.")
    run.add_argument("--output", required=True)
    run.add_argument("--image-root", help="Separate public image archive directory; keep published text image-free.")
    run.add_argument("--tokenizer", default="public/models/Qwen--Qwen3-0.6B-Base")
    run.add_argument("--qwen-example", default="test_api.py",
                     help="Read Anthropic settings statically; never execute this file.")
    run.add_argument("--codex-bin", default="codex")
    run.add_argument("--codex-timeout", type=float, default=300)
    run.add_argument("--judge-flush-seconds", type=float, default=10.0,
                     help="Maximum batch assembly wait from the first image; one assembler feeds all teachers.")
    run.add_argument("--prepare-only", action="store_true")
    run.add_argument("--retry-failed", action="store_true")
    run.add_argument("--retry-phase", choices=("all", "qwen", "sol", "prepare"), default="all")
    run.add_argument("--regenerate-failed", action="store_true",
                     help="With --retry-failed, regenerate failed sol judgement; retain a completed Qwen result.")
    for name, default in (("download-workers", 64), ("per-host", 8), ("cpu-workers", 8),
                          ("qwen-workers", 4), ("teacher-workers", 2), ("judge-batch-size", 4),
                          ("rpm", 120), ("tpm", 600000),
                          ("min-short-side", 512), ("partition", 0), ("partitions", 1)):
        run.add_argument("--" + name, type=int, default=default)
    publish = sub.add_parser("export")
    publish.add_argument("--run", required=True, action="append", help="Repeat for globally deduplicated publication across partitions.")
    publish.add_argument("--output", required=True)
    publish.add_argument("--near-exclude-index", help="Frozen benchmark pHash index for conservative overlap screening.")
    args = parser.parse_args()
    if args.command == "run":
        asyncio.run(run_pipeline(args))
    else:
        print(json.dumps({"published": export_training(args.run, args.output, args.near_exclude_index)}))


if __name__ == "__main__":
    main()
