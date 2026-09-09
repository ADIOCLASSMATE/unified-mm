"""Download pinned official comparison assets without any HTTP/SOCKS proxy.

Only explicitly selected model files are downloaded. Files are resumed to .part,
checked against the published byte length, and atomically committed. No data or
weight hashes are computed. Git revisions are public provenance identifiers.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.research.geometry_v5_assets import (
    ROOT, RUN, MODELS, REPOSITORIES, SOURCES, emit, write_json,
    direct_json, allowed, freeze_manifest, fetch,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=RUN)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    manifest = freeze_manifest(args.output_dir)
    completed, failures = [], []
    # The model-specific contract is frozen before any weight bytes are requested.
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch, entry): entry for entry in manifest["files"]}
        for future in concurrent.futures.as_completed(futures):
            entry = futures[future]
            try:
                completed.append(future.result())
            except Exception as error:  # noqa: BLE001 - persist every worker failure, then fail the whole run
                failures.append(
                    {"repo": entry["repo"], "name": entry["name"], "error": str(error)}
                )
                emit("download_failed", **failures[-1])
            write_json(
                args.output_dir / "asset-download-status.json",
                {
                    "complete": len(completed) == len(manifest["files"])
                    and not failures,
                    "pid": os.getpid(),
                    "completed": completed,
                    "failures": failures,
                    "expected_files": len(manifest["files"]),
                },
            )
    if failures:
        raise RuntimeError(f"{len(failures)} assets failed; resume this same manifest")
    emit(
        "all_assets_complete",
        files=len(completed),
        bytes=sum(r["bytes"] for r in completed),
    )


if __name__ == "__main__":
    main()
