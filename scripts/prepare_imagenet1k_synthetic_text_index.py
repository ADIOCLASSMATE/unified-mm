#!/usr/bin/env python3
"""Build seekable training indexes for ImageNet-1K synthetic text v1."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sqlite3
import sys
import tempfile
from pathlib import Path
from typing import Iterator

from utils.imagenet_synthetic_text_index import (
    INDEX_SCHEMA,
    MAPPING_STRUCT,
    OFFSET_STRUCT,
)


DATASET_SCHEMA = "imagenet1k_synthetic_dataset_v1"
T2I_SCHEMA = "imagenet1k_codex_t2i_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _relative_to(path: Path, base: Path) -> str:
    return os.path.relpath(path, base)


def _open_atomic_pair(data_path: Path, offsets_path: Path):
    token = f"{os.getpid()}"
    data_temporary = data_path.with_name(f".{data_path.name}.{token}.tmp")
    offsets_temporary = offsets_path.with_name(
        f".{offsets_path.name}.{token}.tmp"
    )
    data_path.parent.mkdir(parents=True, exist_ok=True)
    return data_temporary, offsets_temporary


def _write_offsets_for_uncompressed_jsonl(
    source: Path,
    offsets_path: Path,
    *,
    expected_rows: int,
) -> dict[str, object]:
    token = f"{os.getpid()}"
    temporary = offsets_path.with_name(f".{offsets_path.name}.{token}.tmp")
    digest = hashlib.sha256()
    rows = 0
    position = 0
    try:
        with source.open("rb") as input_handle, temporary.open("wb") as offsets:
            offsets.write(OFFSET_STRUCT.pack(0))
            for line in input_handle:
                if not line.strip():
                    raise ValueError(f"blank JSONL row in {source} at row {rows}")
                digest.update(line)
                position += len(line)
                offsets.write(OFFSET_STRUCT.pack(position))
                rows += 1
            offsets.flush()
            os.fsync(offsets.fileno())
        if rows != expected_rows:
            raise ValueError(
                f"caption row count mismatch: {rows} != {expected_rows}"
            )
        os.replace(temporary, offsets_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return {
        "rows": rows,
        "bytes": position,
        "sha256": digest.hexdigest(),
    }


def _iter_gzip_lines(path: Path) -> Iterator[bytes]:
    with gzip.open(path, "rb") as handle:
        yield from handle


def _extract_t2i_shard(
    source: Path,
    destination: Path,
    offsets_path: Path,
    *,
    expected_rows: int,
    expected_prompts: int,
    expected_split: str,
) -> dict[str, object]:
    data_temporary, offsets_temporary = _open_atomic_pair(
        destination, offsets_path
    )
    digest = hashlib.sha256()
    rows = 0
    prompts = 0
    position = 0
    try:
        with data_temporary.open("wb") as output, offsets_temporary.open(
            "wb"
        ) as offsets:
            offsets.write(OFFSET_STRUCT.pack(0))
            for line in _iter_gzip_lines(source):
                if not line.strip():
                    raise ValueError(f"blank JSONL row in {source} at row {rows}")
                row = json.loads(line)
                if row.get("schema") != T2I_SCHEMA:
                    raise ValueError(
                        f"T2I schema mismatch in {source} row {rows}: "
                        f"{row.get('schema')!r}"
                    )
                if row.get("split") != expected_split:
                    raise ValueError(
                        f"T2I split mismatch in {source} row {rows}: "
                        f"{row.get('split')!r}"
                    )
                prompt_rows = row.get("model_result", {}).get("prompts")
                if not isinstance(prompt_rows, list) or len(prompt_rows) != 12:
                    raise ValueError(
                        f"T2I prompt count mismatch in {source} row {rows}"
                    )
                if any(not str(item.get("prompt", "")).strip() for item in prompt_rows):
                    raise ValueError(f"empty T2I prompt in {source} row {rows}")
                output.write(line)
                digest.update(line)
                position += len(line)
                offsets.write(OFFSET_STRUCT.pack(position))
                rows += 1
                prompts += len(prompt_rows)
            output.flush()
            offsets.flush()
            os.fsync(output.fileno())
            os.fsync(offsets.fileno())
        if rows != expected_rows:
            raise ValueError(
                f"T2I shard row count mismatch for {source}: "
                f"{rows} != {expected_rows}"
            )
        if prompts != expected_prompts:
            raise ValueError(
                f"T2I shard prompt count mismatch for {source}: "
                f"{prompts} != {expected_prompts}"
            )
        os.replace(data_temporary, destination)
        os.replace(offsets_temporary, offsets_path)
    except BaseException:
        data_temporary.unlink(missing_ok=True)
        offsets_temporary.unlink(missing_ok=True)
        raise
    return {
        "records": rows,
        "prompts": prompts,
        "bytes": position,
        "sha256": digest.hexdigest(),
    }


def _build_mapping(
    alignment_database: Path,
    output_path: Path,
    *,
    expected_rows: int,
    shard_lookup: dict[str, int],
) -> dict[str, object]:
    token = f"{os.getpid()}"
    temporary = output_path.with_name(f".{output_path.name}.{token}.tmp")
    digest = hashlib.sha256()
    rows = 0
    connection = sqlite3.connect(
        f"file:{alignment_database.resolve()}?mode=ro", uri=True
    )
    try:
        query = """
            SELECT c.img_id, c.caption_row, t.shard, t.shard_row
            FROM caption_samples AS c
            JOIN t2i_samples AS t USING (image_id)
            WHERE t.split = 'train'
            ORDER BY c.img_id
        """
        with temporary.open("wb") as output:
            for img_id, caption_row, shard, shard_row in connection.execute(query):
                expected_img_id = rows + 1
                if int(img_id) != expected_img_id or int(caption_row) != rows:
                    raise ValueError(
                        "alignment is not cache-row ordered at row "
                        f"{rows}: img_id={img_id}, caption_row={caption_row}"
                    )
                if shard not in shard_lookup:
                    raise ValueError(f"alignment references unknown shard {shard!r}")
                encoded = MAPPING_STRUCT.pack(
                    int(shard_lookup[shard]), int(shard_row)
                )
                output.write(encoded)
                digest.update(encoded)
                rows += 1
            output.flush()
            os.fsync(output.fileno())
        if rows != expected_rows:
            raise ValueError(f"alignment rows mismatch: {rows} != {expected_rows}")
        os.replace(temporary, output_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    finally:
        connection.close()
    return {
        "records": rows,
        "bytes": rows * MAPPING_STRUCT.size,
        "record_struct": MAPPING_STRUCT.format,
        "sha256": digest.hexdigest(),
    }


def _validate_existing(manifest_path: Path) -> dict | None:
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != INDEX_SCHEMA:
        return None
    base = manifest_path.parent
    paths = [
        base / manifest["caption"]["offsets_path"],
        base / manifest["mapping"]["path"],
    ]
    for shard in manifest["t2i"]["shards"]:
        paths.extend([base / shard["path"], base / shard["offsets_path"]])
    if not all(path.is_file() for path in paths):
        return None
    return manifest


def build_index(dataset_root: Path, output_dir: Path, *, force: bool) -> dict:
    dataset_root = dataset_root.resolve()
    output_dir = output_dir.resolve()
    manifest_path = output_dir / "manifest.json"
    existing = _validate_existing(manifest_path)
    if existing is not None and not force:
        return existing
    if manifest_path.exists() and not force:
        raise RuntimeError(
            f"incomplete or incompatible index exists at {manifest_path}; "
            "rerun with --force after inspecting it"
        )

    dataset_manifest_path = dataset_root / "dataset_manifest.json"
    dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    if dataset_manifest.get("schema") != DATASET_SCHEMA:
        raise ValueError(f"unexpected dataset schema: {dataset_manifest.get('schema')}")
    expected_rows = int(dataset_manifest["captions"]["split_records"]["train"])

    caption_manifest_path = dataset_root / dataset_manifest["captions"]["manifest"]
    caption_manifest = json.loads(
        caption_manifest_path.read_text(encoding="utf-8")
    )
    caption_path = caption_manifest_path.parent / caption_manifest["file"]
    output_dir.mkdir(parents=True, exist_ok=True)
    caption_offsets_path = output_dir / "captions.offsets.u64"
    print(f"indexing captions: {caption_path}", file=sys.stderr, flush=True)
    caption_result = _write_offsets_for_uncompressed_jsonl(
        caption_path,
        caption_offsets_path,
        expected_rows=expected_rows,
    )
    if caption_result["sha256"] != caption_manifest["file_sha256"]:
        raise ValueError("caption SHA256 differs from the published manifest")

    t2i_manifest_path = dataset_root / dataset_manifest["t2i"]["manifest"]
    t2i_manifest = json.loads(t2i_manifest_path.read_text(encoding="utf-8"))
    train_shards = [
        shard for shard in t2i_manifest["shards"] if shard["split"] == "train"
    ]
    train_shards.sort(key=lambda value: int(value["shard_index"]))
    if len(train_shards) > 255:
        raise ValueError("uint8 T2I mapping supports at most 255 shards")
    shard_lookup: dict[str, int] = {}
    indexed_shards: list[dict[str, object]] = []
    t2i_output_dir = output_dir / "t2i"
    t2i_output_dir.mkdir(parents=True, exist_ok=True)
    for expected_shard_index, shard in enumerate(train_shards):
        shard_index = int(shard["shard_index"])
        if shard_index != expected_shard_index:
            raise ValueError(
                f"non-contiguous train shard index: {shard_index} != "
                f"{expected_shard_index}"
            )
        source = dataset_root / "t2i" / shard["path"]
        print(
            f"indexing T2I shard {shard_index + 1}/{len(train_shards)}: "
            f"{source.name}",
            file=sys.stderr,
            flush=True,
        )
        if sha256_file(source) != shard["sha256"]:
            raise ValueError(f"compressed T2I shard SHA256 mismatch: {source}")
        destination = t2i_output_dir / source.name.removesuffix(".gz")
        offsets_path = destination.with_suffix(destination.suffix + ".offsets.u64")
        result = _extract_t2i_shard(
            source,
            destination,
            offsets_path,
            expected_rows=int(shard["records"]),
            expected_prompts=int(shard["prompts"]),
            expected_split="train",
        )
        shard_lookup[str(Path("t2i") / shard["path"])] = shard_index
        indexed_shards.append(
            {
                "shard_index": shard_index,
                "records": result["records"],
                "prompts": result["prompts"],
                "bytes": result["bytes"],
                "sha256": result["sha256"],
                "path": _relative_to(destination, output_dir),
                "offsets_path": _relative_to(offsets_path, output_dir),
                "source_path": _relative_to(source, dataset_root),
                "source_sha256": shard["sha256"],
            }
        )

    mapping_path = output_dir / "t2i_mapping.bi"
    print("building caption/T2I alignment mapping", file=sys.stderr, flush=True)
    mapping_result = _build_mapping(
        dataset_root / "alignment" / "index.sqlite3",
        mapping_path,
        expected_rows=expected_rows,
        shard_lookup=shard_lookup,
    )
    manifest = {
        "schema": INDEX_SCHEMA,
        "split": "train",
        "records": expected_rows,
        "source_dataset_manifest": _relative_to(dataset_manifest_path, output_dir),
        "source_dataset_manifest_sha256": sha256_file(dataset_manifest_path),
        "caption": {
            "path": _relative_to(caption_path, output_dir),
            "offsets_path": _relative_to(caption_offsets_path, output_dir),
            "records": caption_result["rows"],
            "bytes": caption_result["bytes"],
            "sha256": caption_result["sha256"],
            "captions_per_image": int(caption_manifest["captions_per_image"]),
        },
        "t2i": {
            "source_manifest": _relative_to(t2i_manifest_path, output_dir),
            "source_manifest_sha256": sha256_file(t2i_manifest_path),
            "prompts_per_image": int(t2i_manifest["prompts_per_image"]),
            "shards": indexed_shards,
        },
        "mapping": {
            "path": _relative_to(mapping_path, output_dir),
            **mapping_result,
        },
    }
    atomic_json(manifest_path, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset_root",
        default="public/datasets/imagenet1k_synthetic_v1",
    )
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else dataset_root / "indexed" / "train"
    )
    manifest = build_index(dataset_root, output_dir, force=args.force)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
