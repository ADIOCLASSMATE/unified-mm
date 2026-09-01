#!/usr/bin/env python3
"""Normalize official Karpathy test splits for bidirectional retrieval.

The script never downloads data and never calculates content hashes.  Point it
at a Karpathy ``dataset_*.json`` file and the corresponding image directory.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable


DATASET_CONTRACTS = {
    "mscoco": {
        "task": "mscoco_karpathy_test_5k",
        "expected_images": 5_000,
        "expected_captions": 25_010,
        "expected_caption_count_distribution": {5: 4_990, 6: 10},
        "id_base": 8_000_000_000,
    },
    "flickr30k": {
        "task": "flickr30k_karpathy_test_1k",
        "expected_images": 1_000,
        "expected_captions": 5_000,
        "expected_caption_count_distribution": {5: 1_000},
        "id_base": 8_100_000_000,
    },
}


@dataclass(frozen=True)
class RetrievalAssetRecord:
    image_index: int
    img_id: int
    source_image_id: str
    source_path: str
    captions: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(DATASET_CONTRACTS), required=True)
    parser.add_argument("--karpathy_json", type=Path, required=True)
    parser.add_argument("--image_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def resolve_image_path(image_root: Path, row: dict[str, Any]) -> Path:
    filename = str(row.get("filename", "")).strip()
    if not filename:
        raise ValueError("Karpathy image row has no filename")
    filepath = str(row.get("filepath", "")).strip()
    candidates = []
    if filepath:
        candidates.append(image_root / filepath / filename)
    candidates.append(image_root / filename)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"image {filename!r} was not found under {image_root}; tried {candidates}"
    )


def caption_texts(row: dict[str, Any]) -> tuple[str, ...]:
    captions = tuple(
        str(sentence.get("raw", "")).strip()
        for sentence in row.get("sentences", [])
    )
    if not captions or any(not value for value in captions):
        raise ValueError(
            f"Karpathy test image {row.get('filename')} has empty captions"
        )
    return captions


def load_karpathy_test_records(
    *,
    dataset: str,
    karpathy_json: Path,
    image_root: Path,
    expected_images: int | None = None,
) -> list[RetrievalAssetRecord]:
    payload = json.loads(karpathy_json.read_text(encoding="utf-8"))
    rows = payload.get("images")
    if not isinstance(rows, list):
        raise ValueError("Karpathy JSON has no images list")
    test_rows = [row for row in rows if str(row.get("split")) == "test"]
    contract = DATASET_CONTRACTS[dataset]
    target = int(contract["expected_images"] if expected_images is None else expected_images)
    if len(test_rows) != target:
        raise ValueError(
            f"{dataset} Karpathy test split has {len(test_rows)} images, expected {target}"
        )

    records: list[RetrievalAssetRecord] = []
    filenames: set[str] = set()
    for image_index, row in enumerate(test_rows):
        source_path = resolve_image_path(image_root, row)
        filename = str(row["filename"])
        if filename in filenames:
            raise ValueError(f"duplicate Karpathy test filename: {filename}")
        filenames.add(filename)
        source_image_id = row.get("cocoid", row.get("imgid", filename))
        records.append(
            RetrievalAssetRecord(
                image_index=image_index,
                img_id=int(contract["id_base"]) + image_index,
                source_image_id=str(source_image_id),
                source_path=str(source_path),
                captions=caption_texts(row),
            )
        )
    if expected_images is None:
        caption_count_distribution = Counter(
            len(record.captions) for record in records
        )
        expected_distribution = Counter(
            contract["expected_caption_count_distribution"]
        )
        if caption_count_distribution != expected_distribution:
            raise ValueError(
                f"{dataset} Karpathy test caption-count distribution is "
                f"{dict(sorted(caption_count_distribution.items()))}, expected "
                f"{dict(sorted(expected_distribution.items()))}"
            )
        captions = sum(len(record.captions) for record in records)
        if captions != int(contract["expected_captions"]):
            raise ValueError(
                f"{dataset} Karpathy test split has {captions} captions, "
                f"expected {contract['expected_captions']}"
            )
    return records


def atomic_write(path: Path, payload: str) -> None:
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


def jsonl(rows: Iterable[dict[str, Any]]) -> str:
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    )


def main() -> None:
    args = parse_args()
    output_files = (
        args.output_dir / "manifest.json",
        args.output_dir / "retrieval.jsonl",
        args.output_dir / "image_manifest.jsonl",
    )
    existing = [path for path in output_files if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            f"retrieval assets already exist; pass --overwrite: {existing}"
        )
    records = load_karpathy_test_records(
        dataset=args.dataset,
        karpathy_json=args.karpathy_json,
        image_root=args.image_root,
    )
    contract = DATASET_CONTRACTS[args.dataset]
    caption_count_distribution = Counter(len(record.captions) for record in records)
    captions = sum(len(record.captions) for record in records)
    atomic_write(
        args.output_dir / "retrieval.jsonl",
        jsonl(
            {
                **asdict(record),
                "captions": list(record.captions),
                "dataset": args.dataset,
                "split": "karpathy_test",
            }
            for record in records
        ),
    )
    atomic_write(
        args.output_dir / "image_manifest.jsonl",
        jsonl(
            {
                "img_id": record.img_id,
                "source_path": record.source_path,
                "synset": None,
            }
            for record in records
        ),
    )
    manifest = {
        "schema": "selfless_cross_dataset_retrieval_assets_v1",
        "complete": True,
        "created_at": utc_now(),
        "runtime_hashing_enabled": False,
        "dataset": args.dataset,
        "task": contract["task"],
        "split": "karpathy_test",
        "images": len(records),
        "captions": captions,
        "caption_count_distribution": {
            str(count): images
            for count, images in sorted(caption_count_distribution.items())
        },
        "protocol": {
            "bidirectional": True,
            "recall_at": [1, 5, 10],
            "coco_five_fold_1k_average": False,
        },
        "sources": {
            "karpathy_json": str(args.karpathy_json.resolve()),
            "image_root": str(args.image_root.resolve()),
        },
        "files": {
            "retrieval": "retrieval.jsonl",
            "image_manifest": "image_manifest.jsonl",
        },
    }
    atomic_write(
        args.output_dir / "manifest.json",
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
