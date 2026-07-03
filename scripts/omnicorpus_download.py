"""
Download OmniCorpus-CC parquet shards from Hugging Face.

Examples:
    python scripts/omnicorpus_download.py \
        --include "data/CC-MAIN-2016-26/shard_0.parquet" \
        --local_dir public/datasets/omnicorpus/raw

    python scripts/omnicorpus_download.py \
        --max_shards 4 \
        --local_dir public/datasets/omnicorpus/raw

    python scripts/omnicorpus_download.py \
        --local_dir public/datasets/omnicorpus/raw \
        --max_workers 2 \
        --retries 20 \
        --download_timeout 30
"""

import argparse
import concurrent.futures as futures
import fnmatch
import time
from pathlib import Path

from huggingface_hub import HfApi, constants, hf_hub_download
from tqdm import tqdm


REPO_ID = "OpenGVLab/OmniCorpus-CC-210M"


def list_parquet_shards(repo_id: str, include, max_shards):
    api = HfApi()
    info = api.repo_info(repo_id, repo_type="dataset", files_metadata=True)
    shards = [
        (s.size or 0, s.rfilename)
        for s in info.siblings
        if s.rfilename.endswith(".parquet")
    ]
    shards.sort()

    if include:
        selected = []
        shard_names = [name for _, name in shards]
        shard_sizes = dict((name, size) for size, name in shards)
        for pattern in include:
            matches = [name for name in shard_names if fnmatch.fnmatch(name, pattern)]
            if pattern in shard_sizes and pattern not in matches:
                matches.append(pattern)
            if not matches:
                raise FileNotFoundError(f"No parquet shards matched --include {pattern!r}")
            selected.extend(matches)
        shards = [(shard_sizes[name], name) for name in dict.fromkeys(selected)]

    if max_shards is not None:
        shards = shards[:max_shards]
    return shards


def is_complete(local_dir: Path, filename: str, expected_size: int) -> bool:
    path = local_dir / filename
    return path.exists() and (expected_size <= 0 or path.stat().st_size == expected_size)


def download_one(args, filename: str, expected_size: int) -> str:
    local_dir = Path(args.local_dir)
    if is_complete(local_dir, filename, expected_size):
        return f"skip {filename}"

    last_error = None
    for attempt in range(1, args.retries + 1):
        try:
            hf_hub_download(
                repo_id=args.repo_id,
                filename=filename,
                repo_type="dataset",
                local_dir=str(local_dir),
                etag_timeout=args.etag_timeout,
                token=args.token,
            )
            if not is_complete(local_dir, filename, expected_size):
                path = local_dir / filename
                actual = path.stat().st_size if path.exists() else 0
                raise IOError(
                    f"incomplete file after download: {filename} "
                    f"({actual} != {expected_size} bytes)"
                )
            return f"done {filename}"
        except Exception as exc:
            last_error = exc
            if attempt < args.retries:
                sleep_s = min(args.retry_sleep * attempt, args.max_retry_sleep)
                print(
                    f"[retry {attempt}/{args.retries}] {filename}: {exc}; "
                    f"sleep {sleep_s:.1f}s",
                    flush=True,
                )
                time.sleep(sleep_s)

    raise RuntimeError(f"failed after {args.retries} attempts: {filename}") from last_error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", default=REPO_ID)
    parser.add_argument("--local_dir", default="public/datasets/omnicorpus/raw")
    parser.add_argument("--include", action="append", default=None,
                        help="Shard pattern/path to include. Can be passed multiple times.")
    parser.add_argument("--max_shards", type=int, default=None,
                        help="Download the smallest N parquet shards when --include is omitted.")
    parser.add_argument("--max_workers", type=int, default=1,
                        help="Number of shard downloads to run concurrently.")
    parser.add_argument("--retries", type=int, default=20,
                        help="Retries per shard. Hugging Face resumes incomplete files automatically.")
    parser.add_argument("--retry_sleep", type=float, default=5.0,
                        help="Initial retry sleep in seconds; grows linearly with attempts.")
    parser.add_argument("--max_retry_sleep", type=float, default=120.0)
    parser.add_argument("--etag_timeout", type=float, default=30.0)
    parser.add_argument("--download_timeout", type=float, default=30.0,
                        help="Read timeout in seconds. Low values force slow/stalled transfers to retry sooner.")
    parser.add_argument("--token", default=None,
                        help="Optional Hugging Face token for gated/private access.")
    args = parser.parse_args()

    local_dir = Path(args.local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    constants.HF_HUB_DOWNLOAD_TIMEOUT = args.download_timeout
    constants.HF_HUB_ETAG_TIMEOUT = args.etag_timeout

    shards = list_parquet_shards(args.repo_id, args.include, args.max_shards)
    done = sum(is_complete(local_dir, name, size) for size, name in shards)
    print(f"Downloading {len(shards)} parquet shard(s) to {local_dir}")
    print(f"Already complete: {done}/{len(shards)}")
    print(
        f"Workers={args.max_workers}, retries={args.retries}, "
        f"download_timeout={args.download_timeout}s"
    )

    failures = []
    with futures.ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        future_to_shard = {
            ex.submit(download_one, args, name, size): name
            for size, name in shards
            if not is_complete(local_dir, name, size)
        }
        for fut in tqdm(
            futures.as_completed(future_to_shard),
            total=len(future_to_shard),
            desc="OmniCorpus shards",
        ):
            name = future_to_shard[fut]
            try:
                fut.result()
            except Exception as exc:
                failures.append((name, exc))

    if failures:
        print("Failed shards:")
        for name, exc in failures:
            print(f"  {name}: {exc}")
        raise SystemExit(1)

    print(f"Done: {local_dir}")


if __name__ == "__main__":
    main()
