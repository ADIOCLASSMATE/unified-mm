#!/usr/bin/env python3
"""Download an immutable full-source catalogue directly, with resumable ranges.

This stage never calls an inference service. A download receipt is emitted only
when every declared object has passed its length and upstream digest checks.
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
import sqlite3
import time
from urllib.parse import quote, urlencode

import httpx

from utils.direct_network import check_direct_routes, direct_ssl_context

REPO = Path(__file__).resolve().parents[1]
PUBLIC = (REPO / "public").resolve()
ROOT = PUBLIC / "data_preparation/unified_b_corners_api_v3"
POOL = PUBLIC / "datasets/unified_image_pool_512_v3"


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def digest(path, algorithm="sha256"):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, algorithm).hexdigest()


def init_catalogue(root):
    destination = root / "archive_catalogue.json"
    if destination.exists():
        return json.loads(destination.read_text())
    discovery = root / "source_discovery"
    files = []
    for repo, family, include in (
        ("allenai/pixmo-cap", "pixmo_cap", lambda name: name.startswith("data/train-")),
        ("allenai/pixmo-points", "pixmo_points", lambda name: name.startswith("data/train-")),
        ("huggan/wikiart", "wikiart", lambda name: name.startswith("data/train-") or name == "dataset_infos.json"),
        ("ahmed-masry/ChartQA", "chartqa", lambda name: name == "ChartQA Dataset.zip"),
    ):
        info = json.loads((discovery / (repo.replace("/", "_") + ".json")).read_text())
        for entry in info["siblings"]:
            name = entry["rfilename"]
            if name != "README.md" and not include(name):
                continue
            path = POOL / "source_archives" / family / Path(name).name
            # Reuse verified original bytes; do not duplicate large old archives.
            candidates = [PUBLIC / "datasets/unified_image_pool_512_v1/source_archives" /
                          ("pixmo" if family == "pixmo_cap" else family) / Path(name).name]
            if family == "pixmo_cap" and name.endswith("00000-of-00004.parquet"):
                candidates.append(PUBLIC / "data_preparation/unified_b_512_v2/sources/pixmo_train_00000.parquet")
            if family == "wikiart" and name.endswith("00000-of-00072.parquet"):
                candidates.append(PUBLIC / "data_preparation/unified_b_512_v2/sources/wikiart_train_00000.parquet")
            existing = next((p for p in candidates if p.is_file() and p.stat().st_size == entry["size"]), None)
            files.append({"id": family + ":" + name, "source": family, "upstream_path": name,
                          "url": f"https://huggingface.co/datasets/{repo}/resolve/{info['sha']}/{quote(name)}",
                          "revision": info["sha"], "bytes": entry["size"],
                          "sha256": entry.get("lfs", {}).get("sha256"),
                          "path": str(existing or path), "reuse_existing": bool(existing)})
    anyword = json.loads((discovery / "anyword.json").read_text())
    for entry in anyword["Data"]["Files"]:
        if entry["Type"] != "blob" or entry["Path"].startswith("."):
            continue
        name = entry["Path"]
        files.append({"id": "anyword3m:" + name, "source": "anyword3m", "upstream_path": name,
                      "url": "https://modelscope.cn/api/v1/datasets/iic/AnyWord-3M/repo?" +
                             urlencode({"Revision": entry["Revision"], "FilePath": name}),
                      "revision": entry["Revision"], "bytes": entry["Size"], "sha256": entry["Sha256"],
                      "path": str(POOL / "source_archives/anyword3m" / name), "reuse_existing": False})
    docci = json.loads((discovery / "docci_files.json").read_text())
    if {r["filename"] for r in docci} != {"docci_images.tar.gz", "docci_descriptions.jsonlines", "docci_metadata.jsonlines"}:
        raise ValueError("DOCCI object discovery is incomplete")
    for entry in docci:
        if not entry["revision"]:
            raise ValueError("GCS object generation must be frozen")
        md5 = next((part.strip()[4:] for part in entry.get("md5", "").split(",")
                    if part.strip().startswith("md5=")), None)
        files.append({"id": "docci:" + entry["filename"], "source": "docci",
                      "upstream_path": entry["filename"], "url": entry["url"] + "?generation=" + entry["revision"],
                      "revision": entry["revision"], "bytes": entry["bytes"], "md5_base64": md5,
                      "path": str(POOL / "source_archives/docci" / entry["filename"]), "reuse_existing": False})
    for source, path in (
        ("openimages_relationships", PUBLIC / "data_preparation/unified_b_512_v2/sources/openimages_train_relationships.csv"),
        ("textocr", PUBLIC / "datasets/unified_image_pool_512_v1/source_archives/textocr/TextOCR_0.1_train.json"),
    ):
        files.append({"id": source + ":" + path.name, "source": source, "upstream_path": path.name,
                      "path": str(path), "bytes": path.stat().st_size, "sha256": digest(path),
                      "revision": "frozen_local_sha256", "reuse_existing": True, "url": None})
    value = {"version": "b512-corners-download-first-v3", "created_at": time.time(),
             "imagenet_included": False, "random_sample_cap": None,
             "proxy": "disabled_per_process", "files": files,
             "declared_bytes": sum(r["bytes"] for r in files),
             "scope_note": "Complete declared train sources; DOCCI/ChartQA distribution archives also contain held-out splits, excluded at indexing. URL-backed PixMo/OpenImages/TextOCR images require the separate image download receipt."}
    atomic_json(destination, value)
    return value


async def fetch_file(client, row, range_workers, progress):
    path = Path(row["path"])
    receipt_path = path.with_name(path.name + ".v3-verified.json")
    if path.exists():
        if path.stat().st_size != row["bytes"]:
            raise ValueError("existing object length differs from frozen catalogue")
        if receipt_path.exists():
            previous = json.loads(receipt_path.read_text())
            stat = path.stat()
            if previous.get("size") == stat.st_size and previous.get("mtime_ns") == stat.st_mtime_ns and previous.get("id") == row["id"]:
                return previous
        actual = await asyncio.to_thread(digest, path)
        if row.get("sha256") and actual != row["sha256"]:
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
                    except (httpx.HTTPError, OSError):
                        if attempt == 5:
                            raise
                        await asyncio.sleep(min(2 ** attempt, 30))
        async with asyncio.TaskGroup() as group:
            for _ in range(range_workers):
                group.create_task(worker())

        def assemble():
            temporary = path.with_name(path.name + ".assembling")
            checksum = hashlib.sha256()
            with temporary.open("wb") as output:
                for start in range(0, row["bytes"], chunk_bytes):
                    with (parts / f"{start:012d}.part").open("rb") as part:
                        for chunk in iter(lambda: part.read(4 << 20), b""):
                            checksum.update(chunk)
                            output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
            actual = checksum.hexdigest()
            if temporary.stat().st_size != row["bytes"] or (row.get("sha256") and actual != row["sha256"]):
                raise ValueError("assembled object failed length or SHA256 verification")
            if row.get("md5_base64") and base64.b64encode(bytes.fromhex(digest(temporary, "md5"))).decode() != row["md5_base64"]:
                raise ValueError("assembled GCS object failed MD5 verification")
            temporary.replace(path)
            # Only disposable range files belonging to this completed object.
            for piece in parts.glob("*.part"):
                piece.unlink()
            marker.unlink()
            parts.rmdir()
            return actual
        actual = await asyncio.to_thread(assemble)
    if row.get("md5_base64") and base64.b64encode(bytes.fromhex(await asyncio.to_thread(digest, path, "md5"))).decode() != row["md5_base64"]:
        raise ValueError("existing GCS object failed MD5 verification")
    stat = path.stat()
    receipt = {"id": row["id"], "path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
               "sha256": actual, "upstream_sha256": row.get("sha256"), "upstream_md5_base64": row.get("md5_base64"),
               "revision": row["revision"], "verified_at": time.time(), "proxy": False}
    atomic_json(receipt_path, receipt)
    return receipt


async def run(args):
    check_direct_routes()
    for key in list(os.environ):
        if key.lower() in {"http_proxy", "https_proxy", "all_proxy", "ftp_proxy", "socks_proxy", "no_proxy"}:
            os.environ.pop(key)
    os.environ.update(NO_PROXY="*", no_proxy="*")
    root = args.root.resolve()
    catalogue = init_catalogue(root)
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
    db.commit()
    progress = {"network_bytes": 0, "started_at": time.time()}
    queue = asyncio.Queue()
    # Metadata is ready early, so URL discovery/download can run independently.
    for row in sorted(catalogue["files"], key=lambda row: (not row["reuse_existing"], row["bytes"])):
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
                "file_workers": args.files, "ranges_per_file": args.ranges, "proxy": False, **progress,
                "image_url_downloads_are_separate": True, "bulk_synthesis_started": False}

    async def monitor():
        while not done.is_set():
            atomic_json(root / "archive_download_status.json", status())
            try:
                await asyncio.wait_for(done.wait(), timeout=10)
            except TimeoutError:
                pass

    async with httpx.AsyncClient(trust_env=False, proxy=None, verify=direct_ssl_context(), http2=True,
                                follow_redirects=True, timeout=httpx.Timeout(90, connect=15),
                                limits=httpx.Limits(max_connections=args.files * args.ranges + 4,
                                                   max_keepalive_connections=args.files * args.ranges)) as client:
        async def worker():
            while (row := await queue.get()) is not None:
                db.execute("UPDATE objects SET status='downloading',error=NULL WHERE id=?", (row["id"],))
                db.commit()
                try:
                    receipt = await fetch_file(client, row, args.ranges, progress)
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
            **final, "catalogue_sha256": digest(root / "archive_catalogue.json"),
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
    parser.add_argument("--files", type=int, default=8)
    parser.add_argument("--ranges", type=int, default=4)
    args = parser.parse_args()
    if args.files < 1 or args.ranges < 1:
        parser.error("worker counts must be positive")
    asyncio.run(run(args))
