#!/usr/bin/env python3
"""Prepare no-hash ImageNet-1K val manifests and seekable T2I/I2T text."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path

from utils.imagenet_synthetic_text_index import (
    INDEX_SCHEMA,
    MAPPING_STRUCT,
    OFFSET_STRUCT,
)


EXPECTED_RECORDS = 50_000
EXPECTED_CLASSES = 1_000
EXPECTED_IMAGES_PER_CLASS = 50
EXPECTED_PROMPTS = 12
T2I_SCHEMA = "imagenet1k_codex_t2i_v1"


def _json_line(value: dict) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def _relative(path: Path, base: Path) -> str:
    return os.path.relpath(path, base)


def _extract_t2i_shards(dataset_root: Path, index_root: Path):
    sources = sorted((dataset_root / "t2i" / "shards" / "val").glob("*.jsonl.gz"))
    if not sources:
        raise FileNotFoundError("no official ImageNet val T2I shards found")
    if len(sources) > 255:
        raise ValueError("T2I mapping uses a uint8 shard index")

    indexed_shards = []
    shard_rows: dict[str, list[dict]] = {}
    output_root = index_root / "t2i"
    output_root.mkdir(parents=True, exist_ok=True)
    for shard_index, source in enumerate(sources):
        destination = output_root / source.name.removesuffix(".gz")
        offsets_path = destination.with_suffix(
            destination.suffix + ".offsets.u64"
        )
        rows = []
        prompts = 0
        position = 0
        with gzip.open(source, "rb") as input_handle, destination.open(
            "wb"
        ) as output, offsets_path.open("wb") as offsets:
            offsets.write(OFFSET_STRUCT.pack(0))
            for row_index, encoded in enumerate(input_handle):
                if not encoded.strip():
                    raise ValueError(f"blank row in {source}:{row_index + 1}")
                row = json.loads(encoded)
                if row.get("schema") != T2I_SCHEMA or row.get("split") != "val":
                    raise ValueError(
                        f"invalid val T2I contract in {source}:{row_index + 1}"
                    )
                prompt_rows = row.get("model_result", {}).get("prompts")
                visual_description = str(
                    row.get("model_result", {}).get("visual_description", "")
                ).strip()
                prompt_texts = [
                    str(item.get("prompt", "")).strip()
                    for item in prompt_rows or []
                ]
                if (
                    len(prompt_texts) != EXPECTED_PROMPTS
                    or any(not value for value in prompt_texts)
                    or len(set(prompt_texts)) != EXPECTED_PROMPTS
                    or not visual_description
                ):
                    raise ValueError(
                        f"incomplete val text in {source}:{row_index + 1}"
                    )
                output.write(encoded)
                position += len(encoded)
                offsets.write(OFFSET_STRUCT.pack(position))
                prompts += len(prompt_texts)
                rows.append(
                    {
                        "image_id": str(row["image_id"]),
                        "image_path": str(row["image_path"]),
                        "synset": str(row["synset"]),
                        "visual_description": visual_description,
                    }
                )
        source_key = source.relative_to(dataset_root).as_posix()
        shard_rows[source_key] = rows
        indexed_shards.append(
            {
                "shard_index": shard_index,
                "records": len(rows),
                "prompts": prompts,
                "bytes": position,
                "path": _relative(destination, index_root),
                "offsets_path": _relative(offsets_path, index_root),
                "source_path": source_key,
            }
        )
    return indexed_shards, shard_rows


def _write_aligned_assets(
    *,
    alignment_path: Path,
    image_root: Path,
    manifest_path: Path,
    caption_path: Path,
    caption_offsets_path: Path,
    mapping_path: Path,
    shard_rows: dict[str, list[dict]],
) -> dict:
    shard_lookup = {
        source: shard_index
        for shard_index, source in enumerate(sorted(shard_rows))
    }
    class_counts: Counter[str] = Counter()
    seen_image_ids: set[str] = set()
    seen_image_paths: set[str] = set()
    referenced_t2i_rows: set[tuple[str, int]] = set()
    caption_position = 0
    records = 0

    with gzip.open(alignment_path, "rt", encoding="utf-8") as alignment, (
        manifest_path.open("wb")
    ) as manifest, caption_path.open("wb") as captions, (
        caption_offsets_path.open("wb")
    ) as caption_offsets, mapping_path.open("wb") as mapping:
        caption_offsets.write(OFFSET_STRUCT.pack(0))
        for manifest_index, line in enumerate(alignment):
            row = json.loads(line)
            if row.get("split") != "val":
                raise ValueError(
                    f"alignment row {manifest_index} is not ImageNet val"
                )
            image_id = str(row["image_id"])
            relative_image_path = str(row["image_path"])
            synset = str(row["synset"])
            pointer = row.get("t2i") or {}
            shard_key = str(pointer.get("shard", ""))
            shard_row = int(pointer.get("row", -1))
            if shard_key not in shard_rows:
                raise ValueError(f"unknown T2I shard at alignment row {manifest_index}")
            if shard_row < 0 or shard_row >= len(shard_rows[shard_key]):
                raise ValueError(f"invalid T2I row at alignment row {manifest_index}")
            t2i = shard_rows[shard_key][shard_row]
            expected = (image_id, relative_image_path, synset)
            observed = (t2i["image_id"], t2i["image_path"], t2i["synset"])
            if observed != expected:
                raise ValueError(
                    f"alignment/T2I identity mismatch at row {manifest_index}: "
                    f"{observed!r} != {expected!r}"
                )
            source_path = image_root / synset / Path(relative_image_path).name
            if not source_path.is_file():
                raise FileNotFoundError(source_path)
            if image_id in seen_image_ids or relative_image_path in seen_image_paths:
                raise ValueError(f"duplicate val image at row {manifest_index}")
            pointer_identity = (shard_key, shard_row)
            if pointer_identity in referenced_t2i_rows:
                raise ValueError(f"duplicate T2I pointer at row {manifest_index}")
            seen_image_ids.add(image_id)
            seen_image_paths.add(relative_image_path)
            referenced_t2i_rows.add(pointer_identity)
            class_counts[synset] += 1

            numeric_id = manifest_index + 1
            manifest.write(
                _json_line(
                    {
                        "img_id": numeric_id,
                        "source_path": str(source_path),
                        "synset": synset,
                        "split": "val",
                        "image_id": image_id,
                        "manifest_index": manifest_index,
                    }
                )
            )
            caption_record = {
                "schema": "imagenet1k_val_visual_caption_v1",
                "manifest_index": manifest_index,
                "img_id": numeric_id,
                "id": Path(relative_image_path).stem,
                "path": relative_image_path,
                "split": "val",
                "synset": synset,
                "captions": [
                    {
                        "source": "val_visual_description",
                        "caption_slot": 0,
                        "text": t2i["visual_description"],
                    }
                ],
            }
            encoded_caption = _json_line(caption_record)
            captions.write(encoded_caption)
            caption_position += len(encoded_caption)
            caption_offsets.write(OFFSET_STRUCT.pack(caption_position))
            mapping.write(
                MAPPING_STRUCT.pack(shard_lookup[shard_key], shard_row)
            )
            records += 1

    expected_t2i_rows = sum(len(rows) for rows in shard_rows.values())
    if records != EXPECTED_RECORDS or len(referenced_t2i_rows) != expected_t2i_rows:
        raise ValueError(
            "official val record coverage mismatch: "
            f"alignment={records}, t2i={expected_t2i_rows}, "
            f"referenced={len(referenced_t2i_rows)}"
        )
    if len(class_counts) != EXPECTED_CLASSES or set(class_counts.values()) != {
        EXPECTED_IMAGES_PER_CLASS
    }:
        raise ValueError(
            "official val class balance mismatch: "
            f"classes={len(class_counts)}, counts={sorted(set(class_counts.values()))}"
        )
    return {
        "records": records,
        "caption_bytes": caption_position,
        "classes": len(class_counts),
    }


def prepare(args: argparse.Namespace) -> dict:
    dataset_root = Path(args.dataset_root)
    image_root = Path(args.image_root)
    alignment_path = dataset_root / "alignment" / "val.jsonl.gz"
    manifest_output = Path(args.manifest_output)
    caption_output = Path(args.caption_output)
    index_output = Path(args.index_output)
    for required in (dataset_root, image_root, alignment_path):
        if not required.exists():
            raise FileNotFoundError(required)
    existing = (manifest_output, caption_output, index_output / "manifest.json")
    if any(path.exists() for path in existing) and not args.force:
        raise FileExistsError(
            "val assets already exist; inspect them or pass --force: "
            + ", ".join(str(path) for path in existing if path.exists())
        )

    index_output.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(prefix=".imagenet-val-assets-", dir=index_output.parent)
    )
    try:
        temporary_index = temporary_root / "index"
        temporary_index.mkdir()
        temporary_manifest = temporary_root / "manifest_val.jsonl"
        temporary_caption = temporary_root / "captions_val.jsonl"
        caption_offsets = temporary_index / "captions.offsets.u64"
        mapping_path = temporary_index / "t2i_mapping.bi"

        indexed_shards, shard_rows = _extract_t2i_shards(
            dataset_root, temporary_index
        )
        aligned = _write_aligned_assets(
            alignment_path=alignment_path,
            image_root=image_root,
            manifest_path=temporary_manifest,
            caption_path=temporary_caption,
            caption_offsets_path=caption_offsets,
            mapping_path=mapping_path,
            shard_rows=shard_rows,
        )
        index_manifest = {
            "schema": INDEX_SCHEMA,
            "split": "val",
            "records": aligned["records"],
            "runtime_hashing_enabled": False,
            "caption": {
                "path": _relative(caption_output, index_output),
                "offsets_path": "captions.offsets.u64",
                "records": aligned["records"],
                "bytes": aligned["caption_bytes"],
                "captions_per_image": 1,
                "source": "model_result.visual_description",
            },
            "t2i": {
                "prompts_per_image": EXPECTED_PROMPTS,
                "shards": indexed_shards,
            },
            "mapping": {
                "path": "t2i_mapping.bi",
                "records": aligned["records"],
                "bytes": aligned["records"] * MAPPING_STRUCT.size,
                "record_struct": MAPPING_STRUCT.format,
            },
        }
        (temporary_index / "manifest.json").write_text(
            json.dumps(index_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        manifest_output.parent.mkdir(parents=True, exist_ok=True)
        caption_output.parent.mkdir(parents=True, exist_ok=True)
        if args.force:
            manifest_output.unlink(missing_ok=True)
            caption_output.unlink(missing_ok=True)
            if index_output.exists():
                shutil.rmtree(index_output)
        os.replace(temporary_manifest, manifest_output)
        os.replace(temporary_caption, caption_output)
        os.replace(temporary_index, index_output)
        return {
            "status": "ok",
            "runtime_hashing_enabled": False,
            "split": "val",
            "records": aligned["records"],
            "classes": aligned["classes"],
            "manifest": str(manifest_output),
            "caption": str(caption_output),
            "index": str(index_output / "manifest.json"),
        }
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset_root",
        default="public/datasets/imagenet1k_synthetic_v1",
    )
    parser.add_argument(
        "--image_root",
        default="public/dataset/imagenet/v1/ILSVRC/Data/CLS-LOC/val",
    )
    parser.add_argument(
        "--manifest_output",
        default="public/datasets/imagenet_full/manifest_val.jsonl",
    )
    parser.add_argument(
        "--caption_output",
        default=(
            "public/datasets/imagenet1k_synthetic_v1/captions/"
            "imagenet1k_val_visual_descriptions.jsonl"
        ),
    )
    parser.add_argument(
        "--index_output",
        default="public/datasets/imagenet1k_synthetic_v1/indexed/val",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(prepare(parse_args()), indent=2, sort_keys=True))
