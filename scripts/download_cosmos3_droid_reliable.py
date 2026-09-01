#!/usr/bin/env python3
"""Reliably download Cosmos3-DROID without using environment proxies.

The stock ``hf download`` command aborts the whole snapshot when one Xet/CAS
file reconstruction fails.  This downloader handles every repository file as
an independent job, uses the regular HTTP path, and retries only the failed
file.  Files completed by previous runs are reused by ``hf_hub_download``.
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


PROXY_ENV_NAMES = (
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "no_proxy",
    "NO_PROXY",
)

# These values must be set before importing huggingface_hub.  Its HTTP client
# and feature flags are initialized lazily from the process environment.
for _name in PROXY_ENV_NAMES:
    os.environ.pop(_name, None)
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")

from huggingface_hub import HfApi, hf_hub_download  # noqa: E402


PRINT_LOCK = threading.Lock()
URL_RE = re.compile(r"https?://\S+")


def log(message: str) -> None:
    with PRINT_LOCK:
        print(message, flush=True)


def sanitized_error(exc: BaseException) -> str:
    first_line = str(exc).splitlines()[0] if str(exc) else repr(exc)
    return URL_RE.sub("<url>", first_line)[:500]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="nvidia/Cosmos3-DROID")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--local-dir", type=Path, required=True)
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        help="Optional fnmatch pattern. Repeat to select multiple groups.",
    )
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--base-retry-seconds", type=float, default=5.0)
    parser.add_argument("--max-retry-seconds", type=float, default=300.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def selected_files(all_files: list[str], patterns: list[str]) -> list[str]:
    if not patterns:
        return sorted(all_files)
    return sorted(
        path
        for path in all_files
        if any(fnmatch.fnmatchcase(path, pattern) for pattern in patterns)
    )


def main() -> int:
    args = parse_args()
    if args.workers <= 0:
        raise ValueError("--workers must be positive")

    args.local_dir.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    files = selected_files(
        api.list_repo_files(
            repo_id=args.repo_id,
            repo_type="dataset",
            revision=args.revision,
        ),
        args.include,
    )
    if not files:
        raise RuntimeError(f"no repository files matched: {args.include!r}")

    log(
        "download_start "
        f"repo={args.repo_id} revision={args.revision} files={len(files)} "
        f"workers={args.workers} proxy_env=cleared xet=disabled "
        f"local_dir={args.local_dir}"
    )
    if args.dry_run:
        for path in files:
            log(path)
        return 0

    attempts: dict[str, int] = {}

    def download_one(path: str) -> str:
        while True:
            try:
                hf_hub_download(
                    repo_id=args.repo_id,
                    filename=path,
                    repo_type="dataset",
                    revision=args.revision,
                    local_dir=args.local_dir,
                )
                return path
            except KeyboardInterrupt:
                raise
            except Exception as exc:  # retry each file independently
                attempt = attempts.get(path, 0) + 1
                attempts[path] = attempt
                cap = min(
                    args.max_retry_seconds,
                    args.base_retry_seconds * (2 ** min(attempt - 1, 6)),
                )
                delay = max(1.0, cap * random.uniform(0.75, 1.25))
                log(
                    f"retry file={path} attempt={attempt} delay_s={delay:.1f} "
                    f"error={type(exc).__name__}: {sanitized_error(exc)}"
                )
                time.sleep(delay)

    completed = 0
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_path = {executor.submit(download_one, path): path for path in files}
        try:
            for future in as_completed(future_to_path):
                future.result()
                completed += 1
                if completed == 1 or completed % 25 == 0 or completed == len(files):
                    elapsed = max(time.monotonic() - started, 1e-6)
                    log(
                        f"progress completed={completed}/{len(files)} "
                        f"files_per_hour={completed * 3600.0 / elapsed:.2f}"
                    )
        except KeyboardInterrupt:
            log("interrupted")
            executor.shutdown(wait=False, cancel_futures=True)
            return 130

    log(f"download_complete files={len(files)} elapsed_s={time.monotonic() - started:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
