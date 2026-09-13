#!/usr/bin/env python3
"""Build the reusable benchmark screening index without touching benchmark data."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

from utils.image_near_duplicates import perceptual_hashes, write_index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public", default="public")
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    public = Path(args.public).resolve()
    paths = set()
    for manifest in (public / "benchmarks").glob("*/image_manifest.jsonl"):
        for line in manifest.open():
            path = Path(json.loads(line)["source_path"])
            if str(path).startswith("/inspire/dataset/"):
                path = public / "dataset" / path.relative_to("/inspire/dataset")
            if not path.is_file():
                raise FileNotFoundError(f"benchmark image unavailable: {path}")
            paths.add(path)
    paths = sorted(paths)
    def calculate(path):
        return perceptual_hashes(path.read_bytes())
    hashes, source_ids = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for index, values in enumerate(pool.map(calculate, paths)):
            hashes.extend(values)
            source_ids.extend([index] * len(values))
            if (index + 1) % 10000 == 0:
                print(json.dumps({"images": index + 1, "total": len(paths)}), flush=True)
    write_index(args.output, hashes, source_ids, [str(p) for p in paths])
    print(json.dumps({"images": len(paths), "hashes": len(hashes), "output": args.output}), flush=True)


if __name__ == "__main__":
    main()
