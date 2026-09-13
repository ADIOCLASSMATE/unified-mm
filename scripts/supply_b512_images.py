"""Independent direct-download and 512px preparation services feeding frozen synthesis cohorts."""

import argparse
import asyncio
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import fcntl
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sqlite3
import time
from urllib.parse import urlparse

from data_synthesis.io import DirectDownloader, ImageArchives
from data_synthesis.io import atomic_json, cohort_id, dumps, file_sha, sha
from utils.direct_network import check_direct_routes
from utils.image_shard_io import read_image_bytes
from utils.image_text_preprocessing import prepare_view


def prepare_one(reference, row):
    data = read_image_bytes(reference)
    if row.get("expected_source_sha256") and sha(data) != row["expected_source_sha256"]:
        raise ValueError("source image hash mismatch")
    # Keep all countable entities and text in the frame. Relation boxes still
    # constrain the crop; unusually wide relation scenes use full-frame padding.
    selected = dict(row)
    if {"counting", "ocr", "text"} & set(row.get("capabilities", [])):
        selected["view_policy"] = "fit_pad"
    try:
        return prepare_view(data, selected, include_phashes=True)
    except ValueError as exc:
        if row.get("required_boxes_normalized") and "square crop cannot retain" in str(exc):
            selected["view_policy"] = "fit_pad"
            return prepare_view(data, selected, include_phashes=True)
        raise


def candidates(args):
    from scripts.prepare_b512_candidates import openimages_candidates, pixmo_candidates, wikiart_candidates
    root = Path(args.root).resolve()
    output = root / "candidates"
    output.mkdir(parents=True, exist_ok=True)
    source = Path(args.input).resolve()
    # Wait for an independently downloading frozen metadata shard, not synthesis.
    while not source.exists():
        atomic_json(root / (args.prefix + ".candidate_status.json"), {
            "state": "waiting_for_source", "pid": os.getpid(), "path": str(source), "updated_at": time.time()})
        time.sleep(5)
    excluded = set(Path(args.exclude).read_text().splitlines())
    if args.kind == "openimages":
        rows = openimages_candidates(source, args.limit, 20260915, excluded)
    elif args.kind == "pixmo":
        rows = pixmo_candidates(source, args.limit, args.revision, 20260915, excluded_ids=excluded)
    elif args.kind == "wikiart":
        # The upstream shard contains fewer rows than a global quota. Select all
        # eligible rows from this shard without treating source exhaustion as bad.
        import pyarrow.parquet as pq
        count = pq.ParquetFile(source).metadata.num_rows
        def wiki_rows():
            try:
                yield from wikiart_candidates(source, Path(args.image_root) / "source_extracts" / args.prefix,
                                               count, args.revision, 20260915, excluded)
            except ValueError as exc:
                if "WikiArt" not in str(exc) or "available" not in str(exc):
                    raise
        rows = wiki_rows()
    else:
        payload = json.loads(source.read_text())
        text_source_digest = file_sha(source)
        def text_rows():
            annotations = payload["anns"]
            for identity, info in payload["imgs"].items():
                if info.get("set") != "train" or "openimages:" + identity in excluded:
                    continue
                w, h = info["width"], info["height"]
                if min(w, h) < 512 or max(w, h) / min(w, h) > 2.5:
                    continue
                visible = []
                for ann_id in payload["imgToAnns"][identity]:
                    ann = annotations[ann_id]
                    text = ann.get("utf8_string", "")
                    x, y, bw, bh = ann["bbox"]
                    if text.strip() not in {"", "."} and bh * 512 / max(w, h) >= 10 and bw > 0:
                        visible.append({"text": text, "box": [x / w, y / h, (x + bw) / w, (y + bh) / h]})
                if len(visible) < 2:
                    continue
                yield {"source": "openimages", "source_id": identity, "split": "train",
                       "url": f"https://open-images-dataset.s3.amazonaws.com/train/{identity}.jpg",
                       "min_short_side": 512, "capabilities": ["ocr"], "selection_bucket": "textocr_readable",
                       "metadata_source": "TextOCR_0.1_train", "metadata_sha256": text_source_digest,
                       "selection_annotation": {"readable_words": visible[:32], "annotation_size": [w, h]}}
        rows = text_rows()
    part, buffered, counts = 0, [], Counter()
    source_digest = file_sha(source)

    def flush():
        nonlocal part, buffered
        if not buffered:
            return
        path = output / f"{args.prefix}-{part:05d}.jsonl"
        data = "".join(dumps(row) + "\n" for row in buffered)
        if path.exists():
            if path.read_text() != data:
                raise ValueError("candidate batch changed on resume")
        else:
            temporary = path.with_suffix(".tmp")
            temporary.write_text(data)
            temporary.replace(path)
        part += 1
        buffered = []

    for row in rows:
        bucket = row.get("selection_bucket", "")
        if args.kind == "pixmo" and bucket not in {"relation", "counting", "ocr", "style"}:
            continue
        row["supply_source_file"] = str(source)
        row["supply_source_sha256"] = source_digest
        buffered.append(row)
        counts[bucket] += 1
        if len(buffered) == 256:
            flush()
    flush()
    atomic_json(root / (args.prefix + ".candidate_status.json"), {"state": "completed", "pid": os.getpid(),
                "parts": part, "counts": dict(counts), "updated_at": time.time()})
    print(dumps({"prefix": args.prefix, "parts": part, "counts": dict(counts)}), flush=True)


def connection(root, name):
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / (name + ".lock")).open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    db = sqlite3.connect(root / (name + ".sqlite3"))
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=FULL")
    return db, lock


def download_host(row):
    return "local" if row.get("local_path") else (urlparse(row.get("url", "")).hostname or "invalid")


def retryable_download_error(error):
    text = str(error).lower()
    return any(s in text for s in ("transport", "temporarily", "http 429", "http 5"))


def validate_recovered_images(db, path, image_root):
    """Bind downloaded mirror bytes to existing candidates before publication."""
    value = json.loads(Path(path).read_text())
    if value.get("schema") != "b512.image_recovery.v1":
        raise ValueError("unknown image recovery schema")
    result, seen = [], set()
    for item in value["rows"]:
        if item["key"] in seen:
            raise ValueError("duplicate recovered candidate")
        seen.add(item["key"])
        current = db.execute("SELECT row_json,status,attempts FROM tasks WHERE key=?", (item["key"],)).fetchone()
        if current is None:
            raise ValueError("recovery is outside the frozen download candidates")
        row = json.loads(current[0])
        if (row["source"] != "pixmo_cap" or row["source_id"] != item["source_id"]
                or row["url"] != item["original_url"] or sha(dumps(row).encode()) != item["candidate_sha256"]):
            raise ValueError("recovered candidate identity changed")
        if current[1] == "downloaded":
            continue
        reference = item["reference"]
        physical = reference[4:].rsplit("::", 1)[0] if reference.startswith("tar:") else reference
        if not Path(physical).resolve().is_relative_to(Path(image_root).resolve()):
            raise ValueError("recovery image is outside the public image pool")
        # Verified mirror renditions can be larger than the direct-download
        # default. Match the recovery CLI's explicit maximum byte allowance.
        data = read_image_bytes(reference, max_bytes=128 << 20)
        if sha(data) != item["sha256"] or (row.get("expected_source_sha256") and sha(data) != row["expected_source_sha256"]):
            raise ValueError("recovered image hash mismatch")
        mirror = item["mirror"]
        if mirror.get("proxy") is not False or mirror.get("rendition") != "huggingface_viewer_full_size_jpeg_or_png":
            raise ValueError("unrecognized mirror transport provenance")
        row.update(download_transport="direct_huggingface_dataset_mirror", mirror_provenance=mirror)
        result.append({"key": item["key"], "row": row, "reference": reference, "sha256": item["sha256"],
                       "recovered_from_status": current[1], "previous_attempts": current[2]})
    return result


async def direct_dns_download(client, row):
    """Resolve via HTTPS, then fetch image bytes directly; no image proxy."""
    url = str(row["url"])
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username:
        raise ValueError("image URL must be an unauthenticated HTTP(S) URL")
    env = {k: v for k, v in os.environ.items()
           if k.lower() not in {"http_proxy", "https_proxy", "all_proxy", "ftp_proxy", "socks_proxy", "no_proxy"}
           and k not in {"SII_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"}}
    env.update(NO_PROXY="*", no_proxy="*")
    marker = b"\n__B512_DIRECT_IMAGE_STATUS__"
    process = await asyncio.create_subprocess_exec(
        "curl", "--disable", "--proxy", "", "--noproxy", "*",
        "--doh-url", "https://dns.alidns.com/dns-query", "--silent", "--show-error",
        "--connect-timeout", "5", "--max-time", "30", "--max-filesize", str(client.max_bytes),
        "--proto", "=http,https", "--proto-redir", "=http,https", "--location", "--max-redirs", "8",
        "--write-out", marker.decode() + "%{http_code}", url,
        stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE, env=env)
    try:
        stdout, _ = await process.communicate()
    except BaseException:
        if process.returncode is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass
        await process.communicate()
        raise
    if process.returncode:
        if process.returncode == 63:
            raise ValueError("image exceeds byte limit")
        client.mark_host_failure(parsed.hostname)
        raise ValueError(f"direct image DNS transport failed: curl exit {process.returncode}")
    data, separator, code = stdout.rpartition(marker)
    if not separator or not code.isdigit() or len(code) != 3:
        raise ValueError("direct image transport returned no HTTP status")
    if int(code) >= 400:
        if int(code) == 429 or int(code) >= 500:
            client.mark_host_failure(parsed.hostname)
        raise ValueError(f"image HTTP {code.decode()}")
    if int(code) != 200 or len(data) > client.max_bytes:
        raise ValueError("invalid image HTTP response or byte limit")
    client.host_failures[parsed.hostname] = 0
    client.host_retry_at.pop(parsed.hostname, None)
    return data


async def bounded_download(client, row, timeout, direct_dns_hosts=()):
    try:
        request = direct_dns_download(client, row) if download_host(row) in direct_dns_hosts else client.get(row)
        return await asyncio.wait_for(request, timeout=timeout)
    except TimeoutError:
        client.mark_host_failure(download_host(row))
        raise ValueError("direct image transport exceeded the request deadline") from None


async def download_service(args):
    check_direct_routes()
    root, images = Path(args.root).resolve(), Path(args.image_root).resolve()
    cohort = cohort_id(root)
    db, lock = connection(root, "download")
    db.executescript("""CREATE TABLE IF NOT EXISTS files(path TEXT PRIMARY KEY,sha TEXT);
        CREATE TABLE IF NOT EXISTS tasks(key TEXT PRIMARY KEY,row_json TEXT,status TEXT DEFAULT 'pending',
            attempts INTEGER DEFAULT 0,retry_at REAL DEFAULT 0,error TEXT,reference TEXT,batch_id TEXT,family TEXT);
        CREATE TABLE IF NOT EXISTS batches(id TEXT PRIMARY KEY,manifest_json TEXT,published INTEGER DEFAULT 0);
        CREATE TABLE IF NOT EXISTS recovery_imports(path TEXT PRIMARY KEY,sha256 TEXT,records INTEGER,imported_at REAL);
    """)
    if "family" not in {r[1] for r in db.execute("PRAGMA table_info(tasks)")}:
        db.execute("ALTER TABLE tasks ADD COLUMN family TEXT")
    if "host" not in {r[1] for r in db.execute("PRAGMA table_info(tasks)")}:
        db.execute("ALTER TABLE tasks ADD COLUMN host TEXT")
    if "attempt_base" not in {r[1] for r in db.execute("PRAGMA table_info(tasks)")}:
        db.execute("ALTER TABLE tasks ADD COLUMN attempt_base INTEGER NOT NULL DEFAULT 0")
    db.create_function("download_host", 1, lambda text: download_host(json.loads(text)))
    db.execute("UPDATE tasks SET host=download_host(row_json) WHERE host IS NULL")
    db.execute("""UPDATE tasks SET family=CASE
        WHEN json_extract(row_json,'$.capabilities[0]')='counting' THEN 'counting'
        WHEN json_extract(row_json,'$.capabilities[0]') IN ('ocr','text') THEN 'ocr'
        WHEN json_extract(row_json,'$.capabilities[0]')='style' THEN 'style'
        ELSE 'relation' END WHERE family IS NULL""")
    db.execute("CREATE INDEX IF NOT EXISTS download_ready ON tasks(status,family,retry_at,key)")
    db.execute("CREATE INDEX IF NOT EXISTS download_host_ready ON tasks(status,family,retry_at,host,key)")
    db.execute("UPDATE tasks SET status='pending' WHERE status='running'")
    db.commit()
    client = DirectDownloader(args.workers, args.per_host)
    direct_dns_hosts = set(getattr(args, "direct_dns_host", []))
    direct_dns_rps = getattr(args, "direct_dns_rps", 2.0)
    host_next_start = {}
    host_active = Counter()
    active, last_scan, last_status, serial = {}, 0, 0, db.execute("SELECT count(*) FROM batches").fetchone()[0]
    family_tick, families = 0, ("relation", "counting", "ocr", "style")
    raw_root = root / "raw_batches"
    raw_root.mkdir(exist_ok=True)
    archives, batch_rows, batch_started = None, [], time.time()
    imported_recoveries = {r[0]: r[1] for r in db.execute("SELECT path,sha256 FROM recovery_imports")}

    def publish_pending():
        for batch_id, text in db.execute("SELECT id,manifest_json FROM batches WHERE published=0").fetchall():
            atomic_json(raw_root / (batch_id + ".json"), json.loads(text))
            db.execute("UPDATE batches SET published=1 WHERE id=?", (batch_id,))
        db.commit()

    def flush():
        nonlocal archives, batch_rows, batch_started, serial
        if not batch_rows:
            return
        if archives:
            archives.close()
        batch_id = f"supply-{cohort}-{serial:06d}"
        manifest = {"batch_id": batch_id, "rows": batch_rows, "records": len(batch_rows), "closed_at": time.time()}
        db.execute("INSERT INTO batches(id,manifest_json) VALUES (?,?)", (batch_id, dumps(manifest)))
        for row in batch_rows:
            db.execute("UPDATE tasks SET status='downloaded',reference=?,batch_id=? WHERE key=?", (row["reference"], batch_id, row["key"]))
        db.commit()
        publish_pending()
        serial += 1
        archives, batch_rows, batch_started = None, [], time.time()

    async def import_recoveries():
        # Commit any successful original requests before considering mirrors.
        flush()
        processed = []
        for path in sorted((root / "recovery_inbox").glob("*.json")):
            if str(path) in imported_recoveries:
                continue
            if len(list(raw_root.glob("*.json"))) - len(list((root / "prepared_batches").glob("*/batch.json"))) >= 32:
                break
            digest = file_sha(path)
            rows = validate_recovered_images(db, path, images)
            keys = {row["key"] for row in rows}
            for task, (key, _, _, host, _) in list(active.items()):
                if key in keys:
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
                    del active[task]
                    host_active[host] -= 1
                    if not host_active[host]:
                        del host_active[host]
            for row in rows:
                batch_rows.append(row)
                if len(batch_rows) >= 256:
                    flush()
            processed.append((str(path), digest, len(rows), time.time()))
        flush()
        if processed:
            db.executemany("INSERT INTO recovery_imports VALUES (?,?,?,?)", processed)
            db.commit()
            imported_recoveries.update((p, digest) for p, digest, _, _ in processed)

    try:
        publish_pending()
        while True:
            now = time.time()
            if now - last_scan >= 5:
                for path in sorted((root / "candidates").glob("*.jsonl")):
                    known = db.execute("SELECT sha FROM files WHERE path=?", (str(path),)).fetchone()
                    if known:
                        continue
                    digest = file_sha(path)
                    for line in path.open():
                        row = json.loads(line)
                        key = sha((row["source"] + ":" + row["source_id"]).encode())
                        capability = (row.get("capabilities") or ["general"])[0]
                        family = capability if capability in families else "ocr" if capability == "text" else "relation"
                        db.execute("INSERT OR IGNORE INTO tasks(key,row_json,family,host) VALUES (?,?,?,?)",
                                   (key, dumps(row), family, download_host(row)))
                    db.execute("INSERT INTO files VALUES (?,?)", (str(path), digest))
                db.commit()
                await import_recoveries()
                last_scan = now
            pending_raw = len(list(raw_root.glob("*.json"))) - len(list((root / "prepared_batches").glob("*/batch.json")))
            while len(active) < args.workers and pending_raw < 32:
                row = None
                blocked = {host for host, count in host_active.items() if count >= args.per_host}
                blocked.update(host for host, until in client.host_retry_at.items() if until > time.monotonic())
                blocked.update(host for host, until in host_next_start.items() if until > time.monotonic())
                host_filter = " AND host NOT IN (" + ",".join("?" for _ in blocked) + ")" if blocked else ""
                blocked = sorted(blocked)
                for _ in families:
                    family = families[family_tick % len(families)]
                    family_tick += 1
                    row = db.execute("SELECT key,row_json,attempts,host,attempt_base FROM tasks INDEXED BY download_host_ready "
                                     "WHERE status='pending' AND family=? AND retry_at<=?" + host_filter + " LIMIT 1",
                                     (family, now, *blocked)).fetchone()
                    if row:
                        break
                if row is None:
                    break
                key, text, attempts, host, attempt_base = row
                if host in direct_dns_hosts:
                    host_next_start[host] = time.monotonic() + 1 / direct_dns_rps
                db.execute("UPDATE tasks SET status='running',attempts=attempts+1 WHERE key=?", (key,))
                db.commit()
                task = asyncio.create_task(bounded_download(client, json.loads(text), getattr(args, "request_timeout", 90), direct_dns_hosts))
                active[task] = (key, text, attempts, host, attempt_base)
                host_active[host] += 1
                await asyncio.sleep(0)
            wait_seconds = min([1.0] + [max(0.01, until - time.monotonic())
                for until in host_next_start.values() if until > time.monotonic()])
            if active:
                finished, _ = await asyncio.wait(active, timeout=wait_seconds, return_when=asyncio.FIRST_COMPLETED)
            else:
                finished = set()
                await asyncio.sleep(wait_seconds)
            for task in finished:
                key, text, attempts, host, attempt_base = active.pop(task)
                host_active[host] -= 1
                if not host_active[host]:
                    del host_active[host]
                row = json.loads(text)
                try:
                    data = task.result()
                    if host in direct_dns_hosts:
                        row.update(download_transport="curl_direct_doh", dns_resolver="https://dns.alidns.com/dns-query")
                    if row.get("expected_source_sha256") and sha(data) != row["expected_source_sha256"]:
                        raise ValueError("source SHA mismatch")
                    if row.get("local_path"):
                        reference = row["local_path"]
                    else:
                        if archives is None:
                            archives = ImageArchives(images / "raw_supply" / cohort / f"supply-{serial:06d}")
                        reference = archives.add(key + ".original", data)
                    batch_rows.append({"key": key, "row": row, "reference": reference, "sha256": sha(data)})
                except Exception as exc:
                    wave_attempts = attempts - attempt_base
                    retry = retryable_download_error(exc) and wave_attempts < 4
                    db.execute("UPDATE tasks SET status=?,retry_at=?,error=? WHERE key=?", (
                        "pending" if retry else "failed", now + min(300, 30 * 2 ** min(wave_attempts, 4)), str(exc), key))
                    db.commit()
                if len(batch_rows) >= 256:
                    flush()
            if batch_rows and (time.time() - batch_started >= 30 or not active):
                flush()
            if now - last_status >= 10:
                counts = dict(db.execute("SELECT status,count(*) FROM tasks GROUP BY status"))
                atomic_json(root / "download_status.json", {"state": "running", "pid": os.getpid(), "counts": counts,
                            "workers": args.workers, "per_host": args.per_host, "active": len(active),
                            "active_hosts": dict(host_active), "raw_batches": serial, "direct_proxy": False,
                            "direct_dns_hosts": sorted(direct_dns_hosts), "direct_dns_rps": direct_dns_rps, "updated_at": now})
                last_status = now
            if (root / "candidates.closed.json").exists() and not active and not batch_rows:
                closed = json.loads((root / "candidates.closed.json").read_text())
                recovery_open = any(json.loads(p.read_text()).get("state") != "completed"
                                    for p in (root / "recovery_producers").glob("*.json"))
                if (db.execute("SELECT count(*) FROM files").fetchone()[0] == closed["files"]
                        and not recovery_open
                        and all(str(p) in imported_recoveries for p in (root / "recovery_inbox").glob("*.json"))
                        and not db.execute("SELECT 1 FROM tasks WHERE status IN ('pending','running') LIMIT 1").fetchone()):
                    atomic_json(root / "download.closed.json", {"batches": serial, "completed_at": time.time()})
                    atomic_json(root / "download_status.json", {"state": "completed", "pid": os.getpid(),
                        "counts": dict(db.execute("SELECT status,count(*) FROM tasks GROUP BY status")),
                        "workers": args.workers, "per_host": args.per_host, "active": 0, "raw_batches": serial,
                        "direct_proxy": False, "updated_at": time.time()})
                    break
    finally:
        flush()
        await client.close()
        db.close()
        lock.close()


async def prepare_service(args):
    from utils.image_near_duplicates import NearDuplicateIndex
    root, images, farm_root = (Path(v).resolve() for v in (args.root, args.image_root, args.farm_root))
    cohort = cohort_id(root)
    farm_root.mkdir(parents=True, exist_ok=True)
    owner = farm_root / "prepare_owner.json"
    with owner.with_suffix(".lock").open("a") as ownership:
        fcntl.flock(ownership, fcntl.LOCK_EX)
        if owner.exists() and json.loads(owner.read_text())["source_supply"] != str(root):
            raise ValueError("use a separate preparation inbox root for each source cohort")
        atomic_json(owner, {"source_supply": str(root)})
    db, lock = connection(root, "prepare")
    db.execute("CREATE TABLE IF NOT EXISTS batches(id TEXT PRIMARY KEY,status TEXT,records INTEGER,excluded INTEGER)")
    near = NearDuplicateIndex(args.near_exclude_index)
    exclude = set(Path(args.exclude).read_text().splitlines())
    pool = ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"))
    inbox = farm_root / "input_queue"
    inbox.mkdir(exist_ok=True)
    loop = asyncio.get_running_loop()
    try:
        while True:
            worked = False
            for raw_path in sorted((root / "raw_batches").glob("*.json")):
                batch_id = raw_path.stem
                if db.execute("SELECT 1 FROM batches WHERE id=? AND status='completed'", (batch_id,)).fetchone():
                    continue
                worked = True
                batch = json.loads(raw_path.read_text())
                output = root / "prepared_batches" / batch_id
                output.mkdir(parents=True, exist_ok=True)
                marker = output / "batch.json"
                if marker.exists():
                    manifest = json.loads(marker.read_text())
                else:
                    archives = ImageArchives(images / "prepared_supply" / cohort / batch_id)
                    tasks = sqlite3.connect(output / "state.sqlite3")
                    tasks.execute("CREATE TABLE IF NOT EXISTS tasks(key TEXT PRIMARY KEY,row TEXT,status TEXT,view TEXT,raw TEXT,result TEXT,error TEXT)")
                    accepted, rejected = 0, []
                    futures = [loop.run_in_executor(pool, prepare_one, row["reference"], row["row"]) for row in batch["rows"]]
                    for entry, future in zip(batch["rows"], futures):
                        row, key = entry["row"], entry["key"]
                        try:
                            pixels, view = await future
                            if f"{row['source']}:{row['source_id']}" in exclude or view["source_sha256"] in exclude:
                                raise ValueError("benchmark identity/hash exclusion")
                            match = near.lookup(view["perceptual_hashes"])
                            if match:
                                raise ValueError("benchmark perceptual overlap: " + dumps(match))
                            view.update(original_ref=entry["reference"], source_path=archives.add(key + "." + view["extension"], pixels))
                            tasks.execute("INSERT OR REPLACE INTO tasks VALUES (?,?,'prepared',?,NULL,NULL,NULL)", (key, dumps(row), dumps(view)))
                            accepted += 1
                        except Exception as exc:
                            rejected.append({"key": key, "error": str(exc)})
                            tasks.execute("INSERT OR REPLACE INTO tasks VALUES (?,?,'failed',NULL,NULL,NULL,?)", (key, dumps(row), str(exc)))
                    archives.close()
                    tasks.commit()
                    tasks.close()
                    atomic_json(output / "run.json", {"contract": {"routing_policy": "prepare_only_before_reuse_sii_fallback", "image_size": 512,
                        "source_batch": str(raw_path), "source_batch_sha256": file_sha(raw_path), "benchmark_index": str(near.root)}})
                    manifest = {"batch_id": batch_id, "source_run": str(output), "records": accepted,
                        "candidate_records": len(batch["rows"]), "rejections": rejected,
                        "state_sha256": file_sha(output / "state.sqlite3"), "closed_at": time.time()}
                    atomic_json(marker, manifest)
                atomic_json(inbox / (batch_id + ".json"), {"batch_id": batch_id, "source_run": str(output),
                    "records": manifest["records"], "state_sha256": manifest["state_sha256"], "batch_manifest": str(marker),
                    "batch_manifest_sha256": file_sha(marker)})
                db.execute("INSERT OR REPLACE INTO batches VALUES (?,'completed',?,?)", (
                    batch_id, manifest["records"], manifest["candidate_records"] - manifest["records"]))
                db.commit()
                totals = db.execute("SELECT count(*),coalesce(sum(records),0),coalesce(sum(excluded),0) FROM batches").fetchone()
                atomic_json(root / "prepare_status.json", {"state": "running", "pid": os.getpid(),
                    "batches": totals[0], "prepared": totals[1], "rejected": totals[2], "workers": args.workers,
                    "last_batch": batch_id, "updated_at": time.time()})
                print(dumps({"batch_id": batch_id, "prepared": manifest["records"], "total": totals[1]}), flush=True)
            if (root / "download.closed.json").exists() and not worked:
                expected = json.loads((root / "download.closed.json").read_text())["batches"]
                if db.execute("SELECT count(*) FROM batches WHERE status='completed'").fetchone()[0] == expected:
                    atomic_json(inbox / "closed.json", {"source_supply": str(root), "batches": expected, "completed_at": time.time()})
                    totals = db.execute("SELECT count(*),coalesce(sum(records),0),coalesce(sum(excluded),0) FROM batches").fetchone()
                    atomic_json(root / "prepare_status.json", {"state": "completed", "pid": os.getpid(),
                        "batches": totals[0], "prepared": totals[1], "rejected": totals[2],
                        "workers": args.workers, "updated_at": time.time()})
                    break
            if not worked:
                await asyncio.sleep(2)
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
        db.close()
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    choose = sub.add_parser("candidates")
    choose.add_argument("--kind", choices=["openimages", "pixmo", "wikiart", "textocr"], required=True)
    choose.add_argument("--input", required=True)
    choose.add_argument("--prefix", required=True)
    choose.add_argument("--revision", default="frozen-local")
    choose.add_argument("--limit", type=int, default=32768)
    choose.add_argument("--exclude", required=True)
    down = sub.add_parser("download")
    down.add_argument("--workers", type=int, default=64)
    down.add_argument("--per-host", type=int, default=16)
    down.add_argument("--request-timeout", type=float, default=90)
    down.add_argument("--direct-dns-host", action="append", default=[],
                      help="Resolve this origin via HTTPS while fetching image bytes directly")
    down.add_argument("--direct-dns-rps", type=float, default=2.0,
                      help="Pace direct-DNS origins to avoid repeated HTTP 429 bursts")
    prep = sub.add_parser("prepare")
    prep.add_argument("--workers", type=int, default=2)
    prep.add_argument("--farm-root", required=True)
    prep.add_argument("--near-exclude-index", required=True)
    prep.add_argument("--exclude", required=True)
    for command in (choose, down, prep):
        command.add_argument("--root", required=True)
        command.add_argument("--image-root", required=True)
    args = parser.parse_args()
    if args.command == "download" and (args.workers < 1 or args.per_host < 1 or args.request_timeout <= 0 or args.direct_dns_rps <= 0):
        parser.error("download workers, per-host limit and timeout must be positive")
    if args.command == "candidates":
        candidates(args)
    elif args.command == "download":
        asyncio.run(download_service(args))
    else:
        asyncio.run(prepare_service(args))


if __name__ == "__main__":
    main()
