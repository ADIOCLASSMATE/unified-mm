"""Import frozen views and source captions without rewriting historical releases."""
from __future__ import annotations

import io
import json
from pathlib import Path
import sqlite3
import tempfile
import time
import unicodedata

import numpy as np
from PIL import Image

from data_synthesis.clients import frozen_pixels
from data_synthesis.io import atomic_json, dumps, file_sha, sha, training_image_id
from data_synthesis.state import State
from data_synthesis.integrity import hashing_enabled, view_reference
from utils.image_near_duplicates import NearDuplicateIndex, perceptual_hashes
from utils.imagenet_synthetic_text_index import ImageNetSyntheticTextIndex


def identity(row):
    parent = row.get("parent_id")
    return str(parent) if isinstance(parent, str) and ":" in parent else f"{row['source']}:{row['source_id']}"


def caption_candidate(row, text, *, author, kind, provenance, **extra):
    return {"image_identity": identity(row), "text": text, "author": author,
            "kind": kind, "provenance": provenance, **extra}


def prepared_rows(root):
    root = Path(root).resolve()
    db = sqlite3.connect(f"file:{root / 'state.sqlite3'}?mode=ro", uri=True)
    try:
        for row, view in db.execute("SELECT row,view FROM tasks WHERE view IS NOT NULL AND status IN ('prepared','ready','generated','judged') ORDER BY key"):
            yield json.loads(row), json.loads(view)
    finally:
        db.close()


def release_rows(root):
    root = Path(root).resolve()
    index = ImageNetSyntheticTextIndex(root / "text_index.json")
    if index.split != "train":
        index.close()
        raise ValueError("only training releases can be reused")
    try:
        with (root / "manifest.jsonl").open() as handle:
            for offset, line in enumerate(handle):
                view = json.loads(line)
                row = {k: view[k] for k in ("source", "source_id", "split", "capabilities", "selection_bucket") if k in view}
                caption, prompt = index.read_caption(offset), index.read_t2i(offset)
                if (caption["manifest_index"] != offset or caption["img_id"] != view["img_id"]
                        or prompt["image_id"] != training_image_id(view)):
                    raise ValueError("historical release text/image mapping mismatch")
                provenance = caption.get("provenance", {})
                if provenance != prompt.get("provenance"):
                    raise ValueError("historical caption/prompt provenance differs")
                if len(caption["captions"]) != 1 or len(prompt["model_result"]["prompts"]) != 1:
                    raise ValueError("use a faithful paired release, not old multi-style prompts")
                # Only already accepted, image-grounded releases qualify for exact-view reuse.
                accepted = (provenance.get("decision") in {"accept", "replace", "generate", "reuse"}
                            and (provenance.get("judge_backend") == "codex_cli"
                                 or provenance.get("finalizer_backend") == "codex_cli"))
                if provenance.get("pipeline") == "b512-reuse-sii-fallback-v1":
                    accepted = provenance.get("route") in {"reuse", "normalize", "annotation", "sii", "codex_fallback"}
                if not accepted:
                    raise ValueError("historical pair has no accepted grounded provenance")
                models = provenance.get("generator_models", {})
                row["caption_candidates"] = [caption_candidate(
                    row, caption["captions"][0]["text"], author=models.get("i2t", caption["captions"][0]["source"]),
                    kind="accepted_pair", provenance={"release": str(root), "row": offset, "original": provenance},
                    i2t=caption["captions"][0]["text"], t2i=prompt["model_result"]["prompts"][0]["prompt"],
                    view_sha256=view["view_sha256"], source_sha256=view["source_sha256"],
                    view_id=view_reference(view),
                    generator_models=models, observations=view.get("observations"))]
                yield row, view
    finally:
        index.close()


def ingest_candidates(manifest, supply_root, prefix, *, compute_hashes=True):
    """Publish immutable bounded input batches for independent download/preparation."""
    if not prefix or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in prefix):
        raise ValueError("candidate prefix must be a simple unique name")
    root = Path(supply_root).resolve()
    if (root / "candidates.closed.json").exists() or (root / "download.closed.json").exists():
        raise ValueError("download scope is closed; start a new supply cohort")
    source = Path(manifest).resolve()
    digest = file_sha(source) if compute_hashes else None
    output = root / "candidates"
    output.mkdir(parents=True, exist_ok=True)
    part, pending, count, unique = 0, [], 0, 0

    def flush():
        nonlocal part
        path = output / f"{prefix}-{part:05d}.jsonl"
        data = "".join(dumps(row) + "\n" for row in pending)
        if path.exists() and path.read_text() != data:
            raise ValueError("candidate cohort changed on resume")
        if not path.exists():
            tmp = path.with_suffix(".tmp")
            tmp.write_text(data)
            tmp.replace(path)
        part += 1
        pending.clear()

    # Points / OCR metadata often contain many annotation rows for one image.
    # Aggregate on disk before publishing any batches; keep memory bounded.
    with tempfile.TemporaryDirectory(prefix="candidate-join-", dir=root) as work:
        db = sqlite3.connect(Path(work) / "join.sqlite3")
        try:
            db.execute("CREATE TABLE candidates(key TEXT PRIMARY KEY,row TEXT NOT NULL)")
            with source.open() as handle:
                for line in handle:
                    row = json.loads(line)
                    if row.get("split") != "train" or not row.get("source") or not str(row.get("source_id", "")):
                        raise ValueError("candidate needs explicit original train/source/source_id")
                    if not row.get("url") and not row.get("local_path"):
                        raise ValueError("candidate requires a source image URL or local reference")
                    row["source_id"] = str(row["source_id"])
                    row.setdefault("view_policy", "fit_pad")
                    row.update(supply_source_file=str(source), supply_source_sha256=digest)
                    key = f"{row['source']}:{row['source_id']}"
                    old = db.execute("SELECT row FROM candidates WHERE key=?", (key,)).fetchone()
                    if old:
                        row = merge_candidate_rows(json.loads(old[0]), row)
                    else:
                        unique += 1
                    db.execute("INSERT OR REPLACE INTO candidates VALUES (?,?)", (key, dumps(row)))
                    count += 1
                    if count % 4096 == 0:
                        db.commit()
            db.commit()
            for (encoded,) in db.execute("SELECT row FROM candidates ORDER BY key"):
                pending.append(json.loads(encoded))
                if len(pending) == 256:
                    flush()
            if pending:
                flush()
        finally:
            db.close()
    atomic_json(root / f"{prefix}.candidate_status.json", {"state": "completed", "parts": part,
                "records": unique, "input_rows": count, "input_sha256": digest,
                "input_bytes": source.stat().st_size, "compute_hashes": compute_hashes,
                "source_targets_are_caps": False})
    return {"records": unique, "input_rows": count, "parts": part}


def merge_candidate_rows(old, row):
    merged = dict(old)
    join_fields = {"caption_candidates", "verified_facts", "annotations", "capabilities", "identity_aliases"}
    for key, value in row.items():
        if key in join_fields:
            if not isinstance(value, list) or not isinstance(merged.get(key, []), list):
                raise ValueError(f"{key} must be a list to preserve joined annotations")
            merged[key] = list({dumps(x): x for x in merged.get(key, []) + value}.values())
        elif key in merged and merged[key] != value:
            raise ValueError(f"conflicting {key} for one source image; normalize before ingestion")
        else:
            merged[key] = value
    return merged


def seal_candidates(supply_root, *, compute_hashes=True):
    root = Path(supply_root).resolve()
    paths = sorted((root / "candidates").glob("*.jsonl"))
    if not paths:
        raise ValueError("no candidate batches to seal")
    files = {p.name: file_sha(p) if compute_hashes else None for p in paths}
    sizes = {p.name: p.stat().st_size for p in paths}
    marker = root / "candidates.closed.json"
    if marker.exists():
        previous = json.loads(marker.read_text())
        if ((compute_hashes and previous.get("file_sha256") != files)
                or set(previous.get("file_sha256", {})) != set(files)
                or (previous.get("file_sizes") is not None and previous["file_sizes"] != sizes)):
            raise ValueError("sealed candidate files changed")
        return previous
    # Duplicate identities across independent intakes must be joined explicitly;
    # a download worker must never silently discard their extra annotations.
    with tempfile.TemporaryDirectory(prefix="candidate-seal-", dir=root) as work:
        db = sqlite3.connect(Path(work) / "seen.sqlite3")
        records = 0
        try:
            db.execute("CREATE TABLE seen(key TEXT PRIMARY KEY)")
            for path in paths:
                with path.open() as handle:
                    for line in handle:
                        row = json.loads(line)
                        try:
                            db.execute("INSERT INTO seen VALUES (?)", (f"{row['source']}:{row['source_id']}",))
                        except sqlite3.IntegrityError as exc:
                            raise ValueError("duplicate source image across intakes; join those rows in one manifest/cohort") from exc
                        records += 1
                db.commit()
        finally:
            db.close()
    result = {"files": len(files), "file_sha256": files, "file_sizes": sizes,
              "compute_hashes": compute_hashes, "records": records, "closed_at": time.time()}
    atomic_json(marker, result)
    return result


def prompt_hash(text, *, compute_hashes=True):
    normalized = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
    return sha(normalized.encode()) if compute_hashes else normalized


def excluded_prompts(path, *, compute_hashes=True):
    result = set()
    if path:
        with Path(path).open() as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                value = json.loads(line) if line.startswith(('"', '{')) else line
                text = value.get("prompt") if isinstance(value, dict) else value
                if not isinstance(text, str) or not text.strip():
                    raise ValueError("excluded prompts must be text lines or JSON strings / {prompt: text}")
                result.add(prompt_hash(text, compute_hashes=compute_hashes))
    return result


def _near_corpus(db, values, pixels):
    # Hash screening is only a candidate search. Require near-identical RGB pixels
    # as well, so equally shaped but differently colored counting scenes survive.
    possible = set()
    for value in values:
        for part, shift in enumerate((0, 16, 32, 48)):
            rows = db.execute("SELECT value,item_key FROM phashes WHERE part=? AND bucket=? LIMIT 256",
                              (part, (int(value) >> shift) & 65535))
            possible.update(key for old, key in rows if (int(value) ^ int(old)).bit_count() <= 3)
    if not possible:
        return None
    with Image.open(io.BytesIO(pixels)) as image:
        current = np.asarray(image, dtype=np.int16)
    for key in sorted(possible):
        row = db.execute("SELECT view_json FROM items WHERE key=?", (key,)).fetchone()
        if row is None:
            continue
        view = json.loads(row[0])
        from utils.image_shard_io import read_image_bytes
        with Image.open(io.BytesIO(read_image_bytes(view["source_path"]))) as image:
            prior = np.asarray(image.convert("RGB"), dtype=np.int16)
        if prior.shape != current.shape:
            continue
        delta = np.abs(prior - current)
        if float(delta.mean()) <= 1.0 and float((delta.max(axis=2) > 12).mean()) <= 0.001:
            return key
    return None


def freeze(root, config, *, prepared=(), releases=(), inbox=None, exclude=None, near_index=None, exclude_prompts=None, pilot=False):
    if not hashing_enabled(config):
        from data_synthesis.sources_no_hash import freeze_no_hash
        return freeze_no_hash(root, config, prepared=prepared, releases=releases, inbox=inbox,
                              exclude=exclude, exclude_prompts=exclude_prompts, pilot=pilot)
    sources = [("prepared", Path(p).resolve()) for p in prepared] + [("release", Path(p).resolve()) for p in releases]
    inboxes = ([inbox] if isinstance(inbox, (str, Path)) else inbox) or []
    for entry in inboxes:
        directory = Path(entry).resolve()
        if not (directory / "closed.json").is_file():
            raise ValueError("download/preparation inbox is not closed; bulk synthesis cannot start")
        closed = json.loads((directory / "closed.json").read_text())
        descriptors = [p for p in sorted(directory.glob("*.json")) if p.name != "closed.json"]
        if closed.get("batches") != len(descriptors):
            raise ValueError("closed preparation inbox has missing or additional batch descriptors")
        for path in descriptors:
            descriptor = json.loads(path.read_text())
            source = Path(descriptor["source_run"])
            if file_sha(source / "state.sqlite3") != descriptor["state_sha256"]:
                raise ValueError("prepared batch changed after sealing")
            sources.append(("prepared", source.resolve()))
    if not sources:
        raise ValueError("provide a closed prepared inbox, prepared run or accepted release")
    if not pilot and (not exclude or not near_index or not exclude_prompts):
        raise ValueError("production freeze requires identity exclusions, benchmark near index and test-prompt exclusions")
    prompt_exclusions = excluded_prompts(exclude_prompts)
    exclusions = set(Path(exclude).read_text().splitlines()) if exclude else set()
    near = NearDuplicateIndex(near_index) if near_index else None
    with State(root, config) as state:
        if state.meta("frozen"):
            raise ValueError("image pool is immutable after freezing; use another cohort/root")
        exclusion_contract = {"identities": file_sha(exclude) if exclude else None,
                              "prompts": file_sha(exclude_prompts) if exclude_prompts else None,
                              "near": {p.name: file_sha(p) for p in sorted(Path(near_index).iterdir()) if p.is_file()} if near_index else None}
        prior_exclusions = state.meta("exclusion_contract")
        if prior_exclusions is not None and prior_exclusions != exclusion_contract:
            raise ValueError("evaluation exclusions changed during pool admission; start a new root")
        state.set_meta("exclusion_contract", exclusion_contract)
        state.set_meta("excluded_prompt_hashes", sorted(prompt_exclusions))
        state.db.commit()
        for kind, source in sources:
            if kind == "prepared" and not pilot:
                marker = source / "batch.json"
                if not marker.is_file() or json.loads(marker.read_text()).get("state_sha256") != file_sha(source / "state.sqlite3"):
                    raise ValueError("production prepared inputs must have a valid closed batch receipt")
            primary = source / ("state.sqlite3" if kind == "prepared" else "manifest.jsonl")
            digest = file_sha(primary)
            admission_id = sha((kind + ":" + str(source)).encode())
            prior = state.db.execute("SELECT sha256 FROM admissions WHERE id=?", (admission_id,)).fetchone()
            if prior:
                if prior[0] != digest:
                    raise ValueError("already imported source changed")
                continue
            rows = prepared_rows(source) if kind == "prepared" else release_rows(source)
            visited = 0
            for row, view in rows:
                ident = identity(row)
                reason = None
                if row.get("split") != "train":
                    reason = "non_train"
                elif row["source"] == "imagenet" and not config.get("include_imagenet", False):
                    reason = "imagenet_deferred_not_part_of_non_imagenet_target"
                aliases = set(row.get("identity_aliases", [])) | {ident, f"{row['source']}:{row['source_id']}"}
                if aliases & exclusions or view["source_sha256"] in exclusions:
                    reason = "evaluation_or_imagenet_identity_exclusion"
                texts = [c.get(field, "") for c in row.get("caption_candidates", []) for field in ("text", "i2t", "t2i")]
                if any(isinstance(t, str) and t and prompt_hash(t) in prompt_exclusions for t in texts):
                    reason = "evaluation_test_prompt_overlap"
                item = {"key": view["source_sha256"], "identity": ident, "row": row, "view": view}
                if reason is None:
                    try:
                        pixels, _ = frozen_pixels(item)
                        values = sorted(set(view.get("perceptual_hashes", [])) | set(perceptual_hashes(pixels)))
                        if near and near.lookup(values):
                            reason = "benchmark_near_duplicate"
                    except (ValueError, OSError) as exc:
                        reason = f"invalid_frozen_view:{type(exc).__name__}"
                if reason:
                    state.db.execute("INSERT OR REPLACE INTO exclusions VALUES (?,?,?)", (ident, reason, dumps(row)))
                    continue
                old = state.db.execute("SELECT key,identity,row_json,source_sha256,view_sha256 FROM items WHERE identity=? OR source_sha256=? OR view_sha256=?",
                                       (ident, view["source_sha256"], view["view_sha256"])).fetchall()
                if old:
                    if len(old) != 1 or (old[0]["source_sha256"] != view["source_sha256"] and old[0]["view_sha256"] != view["view_sha256"]):
                        state.db.execute("INSERT OR REPLACE INTO exclusions VALUES (?,?,?)", (ident, "original_identity_content_conflict", dumps(row)))
                        continue
                    old = old[0]
                    merged = json.loads(old["row_json"])
                    merged["identity_aliases"] = sorted(set(merged.get("identity_aliases", [])) | aliases | {old["identity"]})
                    for field in ("caption_candidates", "verified_facts"):
                        records = {dumps(x): x for x in merged.get(field, []) + row.get(field, [])}
                        merged[field] = list(records.values())
                    merged["capabilities"] = sorted(set(merged.get("capabilities", [])) | set(row.get("capabilities", [])))
                    state.db.execute("UPDATE items SET row_json=? WHERE key=?", (dumps(merged), old["key"]))
                    continue
                match = _near_corpus(state.db, values, pixels)
                if match:
                    state.db.execute("INSERT OR REPLACE INTO exclusions VALUES (?,?,?)", (ident, "corpus_near_duplicate:" + match, dumps(row)))
                    continue
                # Caption identity is explicit; different near-duplicate images never share annotations.
                row["identity_aliases"] = sorted(aliases)
                state.db.execute("INSERT INTO items(key,identity,source_sha256,view_sha256,row_json,view_json,status,created_at) VALUES (?,?,?,?,?,?,'pending',?)",
                                 (item["key"], ident, view["source_sha256"], view["view_sha256"], dumps(row), dumps(view), time.time()))
                state.db.executemany("INSERT OR IGNORE INTO phashes VALUES (?,?,?,?)",
                                     [(part, (int(value) >> shift) & 65535, str(value), item["key"])
                                      for value in values for part, shift in enumerate((0, 16, 32, 48))])
                visited += 1
                if visited % 512 == 0:
                    state.db.commit()
            state.db.execute("INSERT INTO admissions VALUES (?,?,?,?)", (admission_id, str(source), digest, visited))
            state.db.commit()
        count = state.db.execute("SELECT count(*) FROM items WHERE json_extract(row_json,'$.source') != 'imagenet'").fetchone()[0]
        total = state.db.execute("SELECT count(*) FROM items").fetchone()[0]
        report = {"state": "frozen" if pilot or count >= config["minimum_images"] else "needs_more_images",
                  "records": total, "non_imagenet_records": count, "minimum_images": config["minimum_images"],
                  "source_targets_are_caps": False, "pilot": bool(pilot), "frozen_at": time.time(),
                  "exclude_sha256": file_sha(exclude) if exclude else None,
                  "exclude_prompts_sha256": file_sha(exclude_prompts) if exclude_prompts else None,
                  "prompt_exclusion_scope": "normalized exact text; benchmark image near exclusion is separate",
                  "near_index_sha256": file_sha(Path(near_index) / "hashes.npy") if near_index else None,
                  "exclusions": dict(state.db.execute("SELECT reason,count(*) FROM exclusions GROUP BY reason")),
                  "dedup_method": "identity/source/view SHA256 + conservative pHash<=3 and strict RGB similarity"}
        if total == 0:
            raise ValueError("no eligible frozen images")
        state.set_meta("frozen", report if report["state"] == "frozen" else None)
        state.db.commit()
        atomic_json(state.root / "frozen_pool.json", report)
        return report
