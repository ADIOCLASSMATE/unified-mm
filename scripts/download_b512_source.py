#!/usr/bin/env python3
"""Download frozen source metadata in parallel, validated direct byte ranges."""

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import shutil

import httpx

from utils.direct_network import check_direct_routes, direct_ssl_context


async def download(url, destination, size, workers=8, chunk_bytes=8 << 20, sha256=None):
    check_direct_routes()
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.stat().st_size != size:
            raise ValueError("existing source has the wrong length")
        if sha256 and hashlib.sha256(destination.read_bytes()).hexdigest() != sha256:
            raise ValueError("existing source has the wrong digest")
        return
    ranges = destination.with_name(destination.name + ".ranges")
    ranges.mkdir(exist_ok=True)
    contract = {"url": url, "size": size, "chunk_bytes": chunk_bytes, "sha256": sha256}
    marker = ranges / "contract.json"
    if marker.exists() and json.loads(marker.read_text()) != contract:
        raise ValueError("range download contract changed")
    marker.write_text(json.dumps(contract, indent=2) + "\n")
    queue = asyncio.Queue()
    tasks = [(start, min(start + chunk_bytes, size) - 1) for start in range(0, size, chunk_bytes)]
    for task in tasks:
        queue.put_nowait(task)
    for _ in range(workers):
        queue.put_nowait(None)
    # Reuse a stopped single-stream download's prefix, including a partial chunk.
    previous = destination.with_suffix(destination.suffix + ".part")
    if previous.exists():
        with previous.open("rb") as source:
            for start, end in tasks:
                path = ranges / f"{start:012d}.part"
                available = min(end + 1, previous.stat().st_size) - start
                if available > 0 and not path.exists():
                    source.seek(start)
                    path.write_bytes(source.read(available))

    completed = 0
    async def worker():
        nonlocal completed
        async with httpx.AsyncClient(trust_env=False, proxy=None, http2=True, follow_redirects=True,
                                     verify=direct_ssl_context(), timeout=httpx.Timeout(90, connect=15)) as client:
            while (task := await queue.get()) is not None:
                start, end = task
                path = ranges / f"{start:012d}.part"
                target = end - start + 1
                for attempt in range(5):
                    available = path.stat().st_size if path.exists() else 0
                    if available > target:
                        raise ValueError("range prefix exceeds expected length")
                    if available == target:
                        break
                    try:
                        headers = {"Range": f"bytes={start + available}-{end}", "Accept-Encoding": "identity"}
                        async with client.stream("GET", url, headers=headers) as response:
                            response.raise_for_status()
                            expected = f"bytes {start + available}-{end}/{size}"
                            if response.status_code != 206 or response.headers.get("Content-Range") != expected:
                                raise ValueError("server did not honor the exact requested range")
                            with path.open("ab") as handle:
                                async for data in response.aiter_raw():
                                    if available + len(data) > target:
                                        raise ValueError("server exceeded requested range")
                                    handle.write(data)
                                    available += len(data)
                            if available != target:
                                raise OSError("incomplete range")
                        break
                    except (httpx.HTTPError, OSError):
                        if attempt == 4:
                            raise
                        await asyncio.sleep(min(2 ** attempt, 8))
                completed += 1
                print(json.dumps({"chunks_complete": completed, "chunks": len(tasks), "proxy": False}), flush=True)
    async with asyncio.TaskGroup() as group:
        for _ in range(workers):
            group.create_task(worker())
    assembled = destination.with_suffix(destination.suffix + ".assembled")
    digest = hashlib.sha256()
    with assembled.open("wb") as handle:
        for start, end in tasks:
            path = ranges / f"{start:012d}.part"
            if path.stat().st_size != end - start + 1:
                raise ValueError("range size changed before assembly")
            with path.open("rb") as source:
                for data in iter(lambda: source.read(1 << 20), b""):
                    digest.update(data)
                    handle.write(data)
    if sha256 and digest.hexdigest() != sha256:
        raise ValueError("assembled source SHA256 does not match")
    assembled.replace(destination)
    (destination.with_suffix(destination.suffix + ".download.json")).write_text(json.dumps({
        **contract, "actual_sha256": digest.hexdigest(), "proxy": False,
    }, indent=2) + "\n")
    shutil.rmtree(ranges)
    previous.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bytes", type=int, required=True)
    parser.add_argument("--sha256")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.bytes <= 0 or args.workers <= 0:
        parser.error("bytes and workers must be positive")
    asyncio.run(download(args.url, args.output, args.bytes, args.workers, sha256=args.sha256))


if __name__ == "__main__":
    main()
