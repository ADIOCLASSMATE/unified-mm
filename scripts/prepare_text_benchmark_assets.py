#!/usr/bin/env python3
"""Download the public text benchmark assets without content hashing."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import tempfile
import urllib.request
import zipfile

import pyarrow.parquet as pq


DEFAULT_ROOT = Path("public/benchmarks/selfless_text_v1")

PARQUET_ASSETS = {
    "arc_easy_validation": {
        "url": (
            "https://huggingface.co/api/datasets/allenai/ai2_arc/parquet/"
            "ARC-Easy/validation/0.parquet"
        ),
        "path": "arc_easy/validation.parquet",
        "records": 570,
    },
    "arc_challenge_validation": {
        "url": (
            "https://huggingface.co/api/datasets/allenai/ai2_arc/parquet/"
            "ARC-Challenge/validation/0.parquet"
        ),
        "path": "arc_challenge/validation.parquet",
        "records": 299,
    },
    "hellaswag_validation": {
        "url": (
            "https://huggingface.co/api/datasets/Rowan/hellaswag/parquet/"
            "default/validation/0.parquet"
        ),
        "path": "hellaswag/validation.parquet",
        "records": 10042,
    },
    "winogrande_validation": {
        "url": (
            "https://huggingface.co/api/datasets/allenai/winogrande/parquet/"
            "winogrande_xl/validation/0.parquet"
        ),
        "path": "winogrande/validation.parquet",
        "records": 1267,
    },
    "boolq_validation": {
        "url": (
            "https://huggingface.co/api/datasets/google/boolq/parquet/"
            "default/validation/0.parquet"
        ),
        "path": "boolq/validation.parquet",
        "records": 3270,
    },
    "openbookqa_validation": {
        "url": (
            "https://huggingface.co/api/datasets/allenai/openbookqa/parquet/"
            "main/validation/0.parquet"
        ),
        "path": "openbookqa/validation.parquet",
        "records": 500,
    },
    "mmlu_dev": {
        "url": (
            "https://huggingface.co/api/datasets/cais/mmlu/parquet/"
            "all/dev/0.parquet"
        ),
        "path": "mmlu/dev.parquet",
        "records": 285,
    },
    "mmlu_test": {
        "url": (
            "https://huggingface.co/api/datasets/cais/mmlu/parquet/"
            "all/test/0.parquet"
        ),
        "path": "mmlu/test.parquet",
        "records": 14042,
    },
}

PIQA_ARCHIVE_URL = (
    "https://storage.googleapis.com/ai2-mosaic/public/physicaliqa/"
    "physicaliqa-train-dev.zip"
)
PIQA_RECORDS = 1838


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def atomic_write_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".partial",
            delete=False,
        ) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def download(url: str, destination: Path, force: bool) -> None:
    if destination.is_file() and not force:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial: Path | None = None
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "unified-mm-evaluation/1.0"},
    )
    try:
        with tempfile.NamedTemporaryFile(
            "wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".partial",
            delete=False,
        ) as output:
            partial = Path(output.name)
            with urllib.request.urlopen(request, timeout=120) as response:
                shutil.copyfileobj(response, output, length=8 * 1024 * 1024)
                output.flush()
                os.fsync(output.fileno())
        os.replace(partial, destination)
        partial = None
    finally:
        if partial is not None:
            partial.unlink(missing_ok=True)


def parquet_records(path: Path) -> int:
    return int(pq.ParquetFile(path).metadata.num_rows)


def text_records(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def prepare_piqa(root: Path, force: bool) -> dict[str, object]:
    archive = root / "piqa" / "physicaliqa-train-dev.zip"
    dev_jsonl = root / "piqa" / "validation.jsonl"
    dev_labels = root / "piqa" / "validation-labels.lst"
    download(PIQA_ARCHIVE_URL, archive, force=force)
    if force or not (dev_jsonl.is_file() and dev_labels.is_file()):
        with zipfile.ZipFile(archive) as source:
            json_bytes = source.read("physicaliqa-train-dev/dev.jsonl")
            label_bytes = source.read("physicaliqa-train-dev/dev-labels.lst")
        atomic_write_text(dev_jsonl, json_bytes.decode("utf-8"))
        atomic_write_text(dev_labels, label_bytes.decode("utf-8"))
    records = text_records(dev_jsonl)
    label_records = text_records(dev_labels)
    if records != PIQA_RECORDS or label_records != PIQA_RECORDS:
        raise ValueError(
            "PIQA validation row mismatch: "
            f"examples={records}, labels={label_records}, expected={PIQA_RECORDS}"
        )
    return {
        "source_url": PIQA_ARCHIVE_URL,
        "archive": str(archive.resolve()),
        "archive_size_bytes": int(archive.stat().st_size),
        "examples": str(dev_jsonl.resolve()),
        "labels": str(dev_labels.resolve()),
        "records": records,
    }


def main() -> None:
    args = parse_args()
    root = args.output_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {
        "schema": "selfless_text_benchmark_assets_v1",
        "complete": False,
        "runtime_hashing_enabled": False,
        "root": str(root),
        "assets": {},
    }
    for name, spec in PARQUET_ASSETS.items():
        path = root / str(spec["path"])
        print(f"Preparing {name}: {path}", flush=True)
        download(str(spec["url"]), path, force=bool(args.force))
        records = parquet_records(path)
        if records != int(spec["records"]):
            raise ValueError(
                f"{name} row mismatch: {records} != {int(spec['records'])}"
            )
        report["assets"][name] = {
            "source_url": str(spec["url"]),
            "path": str(path.resolve()),
            "size_bytes": int(path.stat().st_size),
            "records": records,
        }
    print("Preparing piqa_validation", flush=True)
    report["assets"]["piqa_validation"] = prepare_piqa(
        root, force=bool(args.force)
    )
    report["complete"] = True
    report["completed_at"] = datetime.now(timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )
    manifest = root / "manifest.json"
    atomic_write_text(
        manifest,
        json.dumps(report, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
