#!/usr/bin/env python3
"""Download an immutable full-source catalogue directly, with resumable ranges.

This stage never calls an inference service. A download receipt is emitted only
when every declared object has passed the configured checks (lengths always).
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import sqlite3
import ssl
import time

import anyio
import httpx

from utils.direct_network import check_direct_routes, direct_ssl_context
from data_synthesis.config import load_config
from data_synthesis.integrity import hashing_enabled

REPO = Path(__file__).resolve().parents[1]
PUBLIC = (REPO / "public").resolve()
ROOT = Path(load_config()["preparation_root"]) / "downloads"
POOL = Path(load_config()["image_root"])


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def digest(path, algorithm="sha256"):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, algorithm).hexdigest()


def init_catalogue(root, catalogue_path=None):
    """Load an explicitly selected, immutable archive scope, without source expansion."""
    root.mkdir(parents=True, exist_ok=True)
    destination = root / "archive_catalogue.json"
    if catalogue_path:
        value = json.loads(Path(catalogue_path).read_text())
        if destination.exists() and json.loads(destination.read_text()) != value:
            raise ValueError("archive catalogue changed; use a new cohort root")
    elif destination.exists():
        value = json.loads(destination.read_text())
    else:
        raise ValueError("provide --catalogue with selected pinned source objects; full-library expansion is retired")
    seen = set()
    if not value.get("files"):
        raise ValueError("archive catalogue must contain explicit objects")
    for row in value["files"]:
        if (row["id"] in seen or not Path(row["path"]).is_absolute() or row["bytes"] < 1
                or not row.get("revision")):
            raise ValueError("archive objects need unique IDs, absolute paths, positive lengths and pinned revisions")
        seen.add(row["id"])
    if value.get("declared_bytes") != sum(row["bytes"] for row in value["files"]):
        raise ValueError("archive catalogue byte total differs from its declared scope")
    if not destination.exists():
        atomic_json(destination, value)
    return value


async def fetch_file(client, row, range_workers, progress, compute_hashes=True):
    path = Path(row["path"])
    receipt_path = path.with_name(path.name + ".v3-verified.json")
    if path.exists():
        if path.stat().st_size != row["bytes"]:
            raise ValueError("existing object length differs from frozen catalogue")
        if receipt_path.exists():
            previous = json.loads(receipt_path.read_text())
            stat = path.stat()
            if (previous.get("size") == stat.st_size and previous.get("mtime_ns") == stat.st_mtime_ns
                    and previous.get("id") == row["id"] and previous.get("revision") == row["revision"]
                    and (not compute_hashes or previous.get("sha256")
                         and previous.get("compute_hashes", True))):
                return previous
        actual = await asyncio.to_thread(digest, path) if compute_hashes else None
        if compute_hashes and row.get("sha256") and actual != row["sha256"]:
            raise ValueError("existing object failed SHA256 verification; preserved for investigation")
    else:
        if not row.get("url"):
            raise ValueError("required frozen local source disappeared")
        path.parent.mkdir(parents=True, exist_ok=True)
        parts = path.with_name(path.name + ".v3-parts")
        parts.mkdir(exist_ok=True)
        chunk_bytes = 32 << 20
        contract = {"id": row["id"], "revision": row["revision"], "bytes": row["bytes"], "chunk_bytes": chunk_bytes}
        marker = parts / "contract.json"
        if marker.exists() and json.loads(marker.read_text()) != contract:
            raise ValueError("range download scope changed")
        atomic_json(marker, contract)
        queue = asyncio.Queue()
        for start in range(0, row["bytes"], chunk_bytes):
            queue.put_nowait((start, min(start + chunk_bytes, row["bytes"]) - 1))
        for _ in range(range_workers):
            queue.put_nowait(None)

        async def worker():
            while (item := await queue.get()) is not None:
                start, end = item
                piece = parts / f"{start:012d}.part"
                target = end - start + 1
                for attempt in range(6):
                    available = piece.stat().st_size if piece.exists() else 0
                    if available == target:
                        break
                    if available > target:
                        raise ValueError("stored range exceeds expected bytes")
                    try:
                        headers = {"Range": f"bytes={start + available}-{end}", "Accept-Encoding": "identity"}
                        async with client.stream("GET", row["url"], headers=headers) as response:
                            response.raise_for_status()
                            expected = f"bytes {start + available}-{end}/{row['bytes']}"
                            full_small = response.status_code == 200 and start == available == 0 and end + 1 == row["bytes"]
                            if not full_small and (response.status_code != 206 or response.headers.get("content-range") != expected):
                                raise ValueError("server did not honor exact byte range")
                            with piece.open("ab") as handle:
                                async for chunk in response.aiter_raw():
                                    if available + len(chunk) > target:
                                        raise ValueError("response exceeded declared range")
                                    handle.write(chunk)
                                    available += len(chunk)
                                    progress["network_bytes"] += len(chunk)
                            if available != target:
                                raise OSError("truncated range")
                        break
                    except (httpx.HTTPError, OSError, anyio.EndOfStream,
                            anyio.BrokenResourceError, anyio.ClosedResourceError):
                        if attempt == 5:
                            raise
                        await asyncio.sleep(min(2 ** attempt, 30) * random.uniform(0.5, 1.5))
        async with asyncio.TaskGroup() as group:
            for _ in range(range_workers):
                group.create_task(worker())

        def assemble():
            temporary = path.with_name(path.name + ".assembling")
            checksum = hashlib.sha256() if compute_hashes else None
            with temporary.open("wb") as output:
                for start in range(0, row["bytes"], chunk_bytes):
                    with (parts / f"{start:012d}.part").open("rb") as part:
                        for chunk in iter(lambda: part.read(4 << 20), b""):
                            if checksum is not None:
                                checksum.update(chunk)
                            output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
            actual = checksum.hexdigest() if checksum is not None else None
            if temporary.stat().st_size != row["bytes"] or (compute_hashes and row.get("sha256") and actual != row["sha256"]):
                raise ValueError("assembled object failed length or SHA256 verification")
            if compute_hashes and row.get("md5_base64") and base64.b64encode(bytes.fromhex(digest(temporary, "md5"))).decode() != row["md5_base64"]:
                raise ValueError("assembled GCS object failed MD5 verification")
            temporary.replace(path)
            # Only disposable range files belonging to this completed object.
            for piece in parts.glob("*.part"):
                piece.unlink()
            marker.unlink()
            parts.rmdir()
            return actual
        actual = await asyncio.to_thread(assemble)
    if compute_hashes and row.get("md5_base64") and base64.b64encode(bytes.fromhex(await asyncio.to_thread(digest, path, "md5"))).decode() != row["md5_base64"]:
        raise ValueError("existing GCS object failed MD5 verification")
    stat = path.stat()
    receipt = {"id": row["id"], "path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
               "sha256": actual, "upstream_sha256": row.get("sha256"), "upstream_md5_base64": row.get("md5_base64"),
               "revision": row["revision"], "verified_at": time.time(), "proxy": False,
               "compute_hashes": compute_hashes, "checksums_verified": compute_hashes}
    atomic_json(receipt_path, receipt)
    return receipt


async def run(args):
    check_direct_routes()
    compute_hashes = hashing_enabled()
    root = args.root.resolve()
    catalogue = init_catalogue(root, args.catalogue)
    if args.command == "init":
        print(json.dumps({"objects": len(catalogue["files"]), "declared_bytes": catalogue["declared_bytes"]}))
        return
    lock = (root / "archive-download.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    db = sqlite3.connect(root / "archive_download.sqlite3")
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("CREATE TABLE IF NOT EXISTS objects (id TEXT PRIMARY KEY, status TEXT NOT NULL, receipt TEXT, error TEXT)")
    for row in catalogue["files"]:
        db.execute("INSERT OR IGNORE INTO objects VALUES (?, 'pending', NULL, NULL)", (row["id"],))
    retry_ids = {r[0] for r in db.execute("SELECT id FROM objects WHERE status='failed'")}
    db.execute("UPDATE objects SET status='pending' WHERE status IN ('failed','downloading')")
    db.commit()
    previous_status = root / "archive_download_status.json"
    previous = json.loads(previous_status.read_text()) if previous_status.exists() else {}
    progress = {"network_bytes": previous.get("network_bytes", 0), "started_at": time.time()}
    queue = asyncio.Queue()
    # Metadata is ready early, so URL discovery/download can run independently.
    for row in sorted(catalogue["files"], key=lambda row: (
            not row["reuse_existing"], row["id"] not in retry_ids, row["bytes"])):
        queue.put_nowait(row)
    for _ in range(args.files):
        queue.put_nowait(None)
    done = asyncio.Event()

    def status():
        counts = dict(db.execute("SELECT status, COUNT(*) FROM objects GROUP BY status"))
        verified_bytes = sum(json.loads(v)["size"] for (v,) in db.execute("SELECT receipt FROM objects WHERE status='verified'"))
        return {"state": "running" if not done.is_set() else ("completed" if counts.get("verified") == len(catalogue["files"]) else "incomplete"),
                "pid": os.getpid(), "updated_at": time.time(), "objects": len(catalogue["files"]),
                "counts": counts, "verified_bytes": verified_bytes, "declared_bytes": catalogue["declared_bytes"],
                "file_workers": args.files, "ranges_per_file": args.ranges, "proxy": False,
                "http2": False, "tls_version": "TLSv1.2", "compute_hashes": compute_hashes, **progress,
                "image_url_downloads_are_separate": True, "bulk_synthesis_started": False}

    async def monitor():
        while not done.is_set():
            atomic_json(root / "archive_download_status.json", status())
            try:
                await asyncio.wait_for(done.wait(), timeout=10)
            except TimeoutError:
                pass

    # Separate HTTP/1.1 connections avoid the upstream HTTP/2 stream resets and
    # shared-connection state errors seen during concurrent large range reads.
    context = direct_ssl_context()
    context.minimum_version = context.maximum_version = ssl.TLSVersion.TLSv1_2
    async with httpx.AsyncClient(trust_env=False, proxy=None, verify=context, http2=False,
                                follow_redirects=True, timeout=httpx.Timeout(90, connect=15),
                                limits=httpx.Limits(max_connections=args.files * args.ranges + 4,
                                                   max_keepalive_connections=args.files * args.ranges + 4,
                                                   keepalive_expiry=60)) as client:
        async def worker():
            while (row := await queue.get()) is not None:
                db.execute("UPDATE objects SET status='downloading',error=NULL WHERE id=?", (row["id"],))
                db.commit()
                try:
                    receipt = await fetch_file(client, row, args.ranges, progress, compute_hashes=compute_hashes)
                    db.execute("UPDATE objects SET status='verified',receipt=?,error=NULL WHERE id=?", (json.dumps(receipt), row["id"]))
                    print(json.dumps({"id": row["id"], "status": "verified", "bytes": row["bytes"]}), flush=True)
                except Exception as exc:
                    message = repr(exc)[:2000]
                    db.execute("UPDATE objects SET status='failed',error=? WHERE id=?", (message, row["id"]))
                    print(json.dumps({"id": row["id"], "status": "failed", "error": message}), flush=True)
                db.commit()
        watcher = asyncio.create_task(monitor())
        try:
            await asyncio.gather(*(worker() for _ in range(args.files)))
        finally:
            done.set()
            await watcher
    final = status()
    atomic_json(root / "archive_download_status.json", final)
    if final["state"] == "completed":
        atomic_json(root / "archive_download_complete.json", {
            **final, "catalogue_sha256": digest(root / "archive_catalogue.json") if compute_hashes else None,
            "receipts": [json.loads(v) for (v,) in db.execute("SELECT receipt FROM objects ORDER BY id")],
            "all_dataset_images_ready": False,
        })
    db.close()
    lock.close()
    if final["state"] != "completed":
        raise SystemExit("Some declared objects remain incomplete; restart resumes verified ranges. No completion receipt issued.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("init", "download"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--catalogue", type=Path, help="Explicit selected archive catalogue; no automatic full-source expansion")
    parser.add_argument("--files", type=int, default=8)
    parser.add_argument("--ranges", type=int, default=4)
    args = parser.parse_args()
    if args.files < 1 or args.ranges < 1:
        parser.error("worker counts must be positive")
    asyncio.run(run(args))
