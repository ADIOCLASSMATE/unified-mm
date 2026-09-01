#!/usr/bin/env python3
"""Extract only the Karpathy test images from image-bearing Parquet shards.

The downloaded Parquet files are split-only transport containers.  Stanford's
Karpathy JSON remains the authority for membership, filenames, and captions.
This script validates those fields exactly (ignoring surrounding whitespace),
writes the original encoded image bytes, and never calculates content hashes.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable

import pyarrow.parquet as pq


DATASET_CONTRACTS = {
    "mscoco": {
        "expected_images": 5_000,
        "expected_captions": 25_010,
        "expected_caption_count_distribution": {5: 4_990, 6: 10},
        "source_id_field": "image_id",
        "source_caption_field": "captions",
    },
    "flickr30k": {
        "expected_images": 1_000,
        "expected_captions": 5_000,
        "expected_caption_count_distribution": {5: 1_000},
        "source_id_field": "filename",
        "source_caption_field": "original_alt_text",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(DATASET_CONTRACTS), required=True)
    parser.add_argument("--karpathy_json", type=Path, required=True)
    parser.add_argument("--parquet", type=Path, nargs="+", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--batch_size", type=int, default=32)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def normalized_captions(values: Iterable[Any]) -> tuple[str, ...]:
    captions = tuple(str(value).strip() for value in values)
    if not captions or any(not caption for caption in captions):
        raise ValueError("Karpathy test captions must be non-empty")
    return captions


def canonical_test_rows(
    dataset: str, karpathy_json: Path
) -> dict[str, dict[str, Any]]:
    payload = json.loads(karpathy_json.read_text(encoding="utf-8"))
    images = payload.get("images")
    if not isinstance(images, list):
        raise ValueError(f"Karpathy JSON has no images list: {karpathy_json}")
    rows: dict[str, dict[str, Any]] = {}
    for row in images:
        if str(row.get("split")) != "test":
            continue
        filename = str(row.get("filename", "")).strip()
        if not filename:
            raise ValueError("Karpathy test row has no filename")
        source_id = (
            str(row.get("cocoid")) if dataset == "mscoco" else filename
        )
        if source_id in rows:
            raise ValueError(f"duplicate Karpathy test source ID: {source_id}")
        rows[source_id] = {
            "filename": filename,
            "captions": normalized_captions(
                sentence.get("raw", "") for sentence in row.get("sentences", [])
            ),
        }
    expected = int(DATASET_CONTRACTS[dataset]["expected_images"])
    if len(rows) != expected:
        raise ValueError(
            f"{dataset} Karpathy test split has {len(rows)} images, expected {expected}"
        )
    caption_count_distribution = Counter(
        len(row["captions"]) for row in rows.values()
    )
    expected_distribution = Counter(
        DATASET_CONTRACTS[dataset]["expected_caption_count_distribution"]
    )
    if caption_count_distribution != expected_distribution:
        raise ValueError(
            f"{dataset} Karpathy test caption-count distribution is "
            f"{dict(sorted(caption_count_distribution.items()))}, expected "
            f"{dict(sorted(expected_distribution.items()))}"
        )
    captions = sum(
        count * images for count, images in caption_count_distribution.items()
    )
    expected_captions = int(DATASET_CONTRACTS[dataset]["expected_captions"])
    if captions != expected_captions:
        raise ValueError(
            f"{dataset} Karpathy test split has {captions} captions, "
            f"expected {expected_captions}"
        )
    return rows


def image_bytes(value: Any) -> bytes:
    if isinstance(value, dict):
        value = value.get("bytes")
    if isinstance(value, memoryview):
        value = value.tobytes()
    if not isinstance(value, bytes) or not value:
        raise ValueError("Parquet image field does not contain encoded image bytes")
    return value


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def atomic_write_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, delete=False
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


def main() -> None:
    args = parse_args()
    if int(args.batch_size) <= 0:
        raise ValueError("--batch_size must be positive")
    for source in (args.karpathy_json, *args.parquet):
        if not source.is_file():
            raise FileNotFoundError(source)
    manifest_path = args.output_dir / "extraction_manifest.json"
    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"test images already extracted; pass --overwrite: {manifest_path}"
        )

    contract = DATASET_CONTRACTS[args.dataset]
    canonical = canonical_test_rows(args.dataset, args.karpathy_json)
    source_id_field = str(contract["source_id_field"])
    source_caption_field = str(contract["source_caption_field"])
    seen: set[str] = set()
    total_bytes = 0

    columns = [source_id_field, source_caption_field, "image"]
    for parquet_path in args.parquet:
        parquet_file = pq.ParquetFile(parquet_path)
        for batch in parquet_file.iter_batches(
            batch_size=int(args.batch_size), columns=columns
        ):
            for row in batch.to_pylist():
                source_id = str(row[source_id_field]).strip()
                expected = canonical.get(source_id)
                if expected is None:
                    raise ValueError(
                        f"Parquet row is not in the canonical Karpathy test split: {source_id}"
                    )
                if source_id in seen:
                    raise ValueError(f"duplicate Parquet test source ID: {source_id}")
                captions = normalized_captions(row[source_caption_field])
                if captions != expected["captions"]:
                    raise ValueError(
                        f"captions disagree with the canonical Karpathy JSON: {source_id}"
                    )
                encoded = image_bytes(row["image"])
                target = args.output_dir / str(expected["filename"])
                if target.exists() and not args.overwrite:
                    raise FileExistsError(target)
                atomic_write_bytes(target, encoded)
                total_bytes += len(encoded)
                seen.add(source_id)

    missing = sorted(set(canonical) - seen)
    if missing:
        raise ValueError(
            f"Parquet input is missing {len(missing)} Karpathy test images: {missing[:8]}"
        )
    expected_images = int(contract["expected_images"])
    if len(seen) != expected_images:
        raise ValueError(f"extracted {len(seen)} images, expected {expected_images}")
    caption_count_distribution = Counter(
        len(row["captions"]) for row in canonical.values()
    )
    captions = sum(
        count * images for count, images in caption_count_distribution.items()
    )

    manifest = {
        "schema": "selfless_cross_dataset_test_image_extraction_v1",
        "complete": True,
        "created_at": utc_now(),
        "runtime_hashing_enabled": False,
        "dataset": args.dataset,
        "split": "karpathy_test",
        "images": len(seen),
        "captions": captions,
        "caption_count_distribution": {
            str(count): images
            for count, images in sorted(caption_count_distribution.items())
        },
        "encoded_image_bytes": total_bytes,
        "karpathy_json": str(args.karpathy_json.resolve()),
        "source_parquet": [str(path.resolve()) for path in args.parquet],
        "image_root": str(args.output_dir.resolve()),
    }
    atomic_write_text(
        manifest_path,
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
