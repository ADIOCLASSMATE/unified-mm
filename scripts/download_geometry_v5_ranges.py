"""Resume pinned official assets using validated, proxy-free HTTP byte ranges."""

import argparse
import concurrent.futures
import json
import os
import shutil
import time
import urllib.error
import urllib.request
from pathlib import Path

from scripts.prepare_geometry_v5_assets import RUN, emit, fetch, write_json


def chunk_path(entry, start, end):
    path = Path(entry["path"])
    return path.parent / (path.name + ".ranges") / f"{start:012d}-{end:012d}.part"


def get_chunk(task):
    entry, start, end = task
    destination = chunk_path(entry, start, end)
    destination.parent.mkdir(parents=True, exist_ok=True)
    target_size = end - start + 1
    for attempt in range(5):
        available = destination.stat().st_size if destination.exists() else 0
        assert available <= target_size
        if available == target_size:
            return target_size
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        request = urllib.request.Request(
            entry["url"],
            headers={
                "Range": f"bytes={start + available}-{end}",
                "User-Agent": "unified-mm-geometry-v5",
                "Accept-Encoding": "identity",
            },
        )
        try:
            with opener.open(request, timeout=90) as response:
                expected = f"bytes {start + available}-{end}/{entry['bytes']}"
                if (
                    response.status != 206
                    or response.headers.get("Content-Range") != expected
                ):
                    raise ValueError(
                        f"Server did not honor the exact byte range: {entry['name']} {response.status} {response.headers.get('Content-Range')}"
                    )
                with destination.open("ab") as output:
                    remaining = target_size - available
                    while remaining:
                        block = response.read(min(1024 * 1024, remaining))
                        if not block:
                            raise EOFError("Premature end of HTTP range")
                        output.write(block)
                        remaining -= len(block)
                    if response.read(1):
                        raise ValueError("Server exceeded requested range")
            assert destination.stat().st_size == target_size
            return target_size
        except (OSError, EOFError, urllib.error.HTTPError) as error:
            emit(
                "range_retry",
                file=entry["name"],
                offset=start,
                attempt=attempt,
                error=str(error),
            )
            if attempt == 4:
                raise
            time.sleep(min(2**attempt, 8))
    raise RuntimeError("Unreachable retry state")


def prepare_chunks(entry, chunk_size):
    # Reuse the bytes from the explicitly stopped single-stream downloader.
    path = Path(entry["path"])
    previous = path.with_suffix(path.suffix + ".part")
    previous_size = previous.stat().st_size if previous.exists() else 0
    assert previous_size <= entry["bytes"]
    tasks = []
    for start in range(0, entry["bytes"], chunk_size):
        end = min(start + chunk_size, entry["bytes"]) - 1
        chunk = chunk_path(entry, start, end)
        if not chunk.exists() and previous_size > start:
            chunk.parent.mkdir(parents=True, exist_ok=True)
            with previous.open("rb") as source, chunk.open("wb") as destination:
                source.seek(start)
                remaining = min(end + 1, previous_size) - start
                while remaining:
                    data = source.read(min(1024 * 1024, remaining))
                    assert data
                    destination.write(data)
                    remaining -= len(data)
        tasks.append((entry, start, end))
    return tasks


def assemble(entry, tasks):
    path = Path(entry["path"])
    if path.exists():
        assert path.stat().st_size == entry["bytes"]
        return
    temp = path.with_suffix(path.suffix + ".assembled")
    with temp.open("wb") as destination:
        for _, start, end in tasks:
            source_path = chunk_path(entry, start, end)
            assert source_path.stat().st_size == end - start + 1
            with source_path.open("rb") as source:
                shutil.copyfileobj(source, destination, length=4 * 1024 * 1024)
    assert temp.stat().st_size == entry["bytes"]
    temp.replace(path)
    # Keep range chunks and legacy partials until model-load verification succeeds.
    emit(
        "ranged_asset_complete",
        repo=entry["repo"],
        name=entry["name"],
        bytes=entry["bytes"],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=RUN)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--chunk-mib", type=int, default=32)
    args = parser.parse_args()
    manifest = json.loads((args.output_dir / "asset-manifest.json").read_text())
    small, large = [], []
    for entry in manifest["files"]:
        (
            large if entry["bytes"] and entry["bytes"] > 32 * 1024 * 1024 else small
        ).append(entry)
    # Official source trees are independent small downloads, available while weights stream.
    with (
        concurrent.futures.ThreadPoolExecutor(max_workers=8) as small_pool,
        concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool,
    ):
        small_futures = {small_pool.submit(fetch, row): row for row in small}
        tasks_by_file, all_tasks = {}, []
        for entry in large:
            path = Path(entry["path"])
            if path.exists():
                assert path.stat().st_size == entry["bytes"]
                continue
            tasks = prepare_chunks(entry, args.chunk_mib * 1024 * 1024)
            tasks_by_file[entry["path"]] = tasks
            all_tasks.extend(tasks)
        # Interleave models, so a slow large model doesn't starve the smaller encoders.
        all_tasks.sort(key=lambda task: task[1])
        futures = {pool.submit(get_chunk, task): task for task in all_tasks}
        completed_by_file = {name: 0 for name in tasks_by_file}
        completed_bytes = 0
        for done, future in enumerate(concurrent.futures.as_completed(futures), 1):
            task = futures[future]
            completed_bytes += future.result()
            name = task[0]["path"]
            completed_by_file[name] += 1
            if completed_by_file[name] == len(tasks_by_file[name]):
                assemble(task[0], tasks_by_file[name])
            if done % 8 == 0 or done == len(futures):
                state = {
                    "pid": os.getpid(),
                    "chunks_done": done,
                    "chunks_total": len(futures),
                    "range_bytes_done": completed_bytes,
                    "complete": False,
                    "proxy_disabled": True,
                }
                write_json(args.output_dir / "asset-range-status.json", state)
                emit("range_progress", **state)
        for future in concurrent.futures.as_completed(small_futures):
            future.result()
    files = []
    for entry in manifest["files"]:
        path = Path(entry["path"])
        assert path.is_file()
        if entry["bytes"]:
            assert path.stat().st_size == entry["bytes"]
        files.append(
            {
                "repo": entry["repo"],
                "name": entry["name"],
                "bytes": path.stat().st_size,
                "path": str(path),
                "proxy_disabled": True,
            }
        )
    write_json(
        args.output_dir / "asset-download-status.json",
        {
            "complete": True,
            "pid": os.getpid(),
            "completed": files,
            "failures": [],
            "expected_files": len(files),
            "transport": "validated HTTP Range, no proxy",
        },
    )
    write_json(
        args.output_dir / "asset-range-status.json",
        {"complete": True, "pid": os.getpid(), "files": len(files)},
    )
    emit("all_assets_complete", files=len(files))


if __name__ == "__main__":
    main()
