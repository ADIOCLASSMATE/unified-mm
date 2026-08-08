from __future__ import annotations

import hashlib
import json
import os
import struct
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from .io import atomic_write_json, fsync_directory, sha256_file, sha256_text, utc_now


CANONICAL_MANIFEST_SCHEMA = "imagenet_caption_farm_manifest_v1"


def _relative_image_path(path: str) -> str:
    parts = Path(path).parts
    if len(parts) < 2:
        raise ValueError(f"invalid ImageNet image path: {path!r}")
    return "/".join(parts[-2:])


def _iter_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"non-object JSON at {path}:{line_number}")
            yield line_number, value


@dataclass(frozen=True)
class ManifestInfo:
    path: Path
    offsets_path: Path
    metadata_path: Path
    row_count: int
    sha256: str
    source_manifest_sha256: str
    original_captions_sha256: str


def build_canonical_manifest(
    source_manifest: Path,
    original_captions: Path,
    output_path: Path,
    *,
    dataset_mount: Path,
) -> ManifestInfo:
    offsets_path = output_path.with_suffix(output_path.suffix + ".offsets")
    metadata_path = output_path.with_suffix(output_path.suffix + ".metadata.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    token = f"{os.getpid()}.{uuid.uuid4().hex}"
    temporary = output_path.parent / f".{output_path.name}.{token}.tmp"
    offsets_temporary = output_path.parent / f".{offsets_path.name}.{token}.tmp"
    digest = hashlib.sha256()
    row_count = 0

    manifest_iter = _iter_jsonl(source_manifest)
    captions_iter = _iter_jsonl(original_captions)
    try:
        with temporary.open("wb") as output, offsets_temporary.open("wb") as offsets:
            while True:
                try:
                    manifest_item = next(manifest_iter)
                except StopIteration:
                    manifest_item = None
                try:
                    caption_item = next(captions_iter)
                except StopIteration:
                    caption_item = None
                if manifest_item is None and caption_item is None:
                    break
                if manifest_item is None or caption_item is None:
                    raise ValueError("manifest and original-caption row counts differ")
                manifest_line, manifest = manifest_item
                caption_line, caption = caption_item
                img_id = int(manifest["img_id"])
                if int(caption["img_id"]) != img_id:
                    raise ValueError(
                        f"img_id mismatch at manifest line {manifest_line} and caption line {caption_line}"
                    )
                source_path = str(manifest["source_path"])
                path = _relative_image_path(source_path)
                synset = str(manifest.get("synset") or "")
                image_id = Path(path).stem
                if not synset or Path(path).parent.name != synset:
                    raise ValueError(f"path/synset mismatch for img_id={img_id}: {path!r}, {synset!r}")
                if str(caption.get("id")) != image_id:
                    raise ValueError(f"image id mismatch for img_id={img_id}")
                if str(caption.get("path")) != path or str(caption.get("synset")) != synset:
                    raise ValueError(f"caption identity mismatch for img_id={img_id}")
                original = " ".join(str(caption.get("recaption_short") or "").split())
                if not original:
                    raise ValueError(f"empty original caption for img_id={img_id}")
                imagenet_relative_path = f"ILSVRC/Data/CLS-LOC/train/{path}"
                canonical_source_path = str(dataset_mount / imagenet_relative_path)
                row = {
                    "schema": CANONICAL_MANIFEST_SCHEMA,
                    "manifest_index": row_count,
                    "img_id": img_id,
                    "image_id": image_id,
                    "id": image_id,
                    "path": path,
                    "imagenet_relative_path": imagenet_relative_path,
                    "source_path": canonical_source_path,
                    "synset": synset,
                    "class": synset,
                    "original_caption_key": image_id,
                    "original_caption": original,
                    "original_caption_sha256": sha256_text(original),
                }
                encoded = (json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
                offsets.write(struct.pack("<Q", output.tell()))
                output.write(encoded)
                digest.update(encoded)
                row_count += 1
            output.flush()
            offsets.flush()
            os.fsync(output.fileno())
            os.fsync(offsets.fileno())
        os.replace(temporary, output_path)
        os.replace(offsets_temporary, offsets_path)
        fsync_directory(output_path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        offsets_temporary.unlink(missing_ok=True)
        raise

    metadata = {
        "schema": CANONICAL_MANIFEST_SCHEMA,
        "created_at": utc_now(),
        "row_count": row_count,
        "sha256": digest.hexdigest(),
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": sha256_file(source_manifest),
        "original_captions": str(original_captions),
        "original_captions_sha256": sha256_file(original_captions),
        "dataset_mount": str(dataset_mount),
        "offsets_path": str(offsets_path),
        "offsets_bytes": offsets_path.stat().st_size,
    }
    atomic_write_json(metadata_path, metadata)
    return ManifestInfo(
        path=output_path,
        offsets_path=offsets_path,
        metadata_path=metadata_path,
        row_count=row_count,
        sha256=digest.hexdigest(),
        source_manifest_sha256=metadata["source_manifest_sha256"],
        original_captions_sha256=metadata["original_captions_sha256"],
    )


class ManifestReader:
    def __init__(self, path: Path, offsets_path: Path) -> None:
        self.path = path
        self.offsets_path = offsets_path
        size = offsets_path.stat().st_size
        if size % 8:
            raise ValueError(f"corrupt manifest offset table: {offsets_path}")
        self.row_count = size // 8
        self._manifest_fd: int | None = None
        self._offsets_fd: int | None = None

    def _open(self) -> tuple[int, int]:
        if self._manifest_fd is None:
            self._manifest_fd = os.open(self.path, os.O_RDONLY)
        if self._offsets_fd is None:
            self._offsets_fd = os.open(self.offsets_path, os.O_RDONLY)
        return self._manifest_fd, self._offsets_fd

    def read(self, index: int) -> dict[str, Any]:
        if index < 0 or index >= self.row_count:
            raise IndexError(index)
        manifest_fd, offsets_fd = self._open()
        offset_bytes = os.pread(offsets_fd, 8, index * 8)
        if len(offset_bytes) != 8:
            raise ValueError(f"missing manifest offset for row {index}")
        offset = struct.unpack("<Q", offset_bytes)[0]
        chunks: list[bytes] = []
        position = offset
        while True:
            chunk = os.pread(manifest_fd, 64 * 1024, position)
            if not chunk:
                raise ValueError(f"unterminated manifest row {index}")
            newline = chunk.find(b"\n")
            if newline >= 0:
                chunks.append(chunk[:newline])
                break
            chunks.append(chunk)
            position += len(chunk)
        value = json.loads(b"".join(chunks))
        if int(value["manifest_index"]) != index:
            raise ValueError(f"manifest index corruption at row {index}")
        return value

    def close(self) -> None:
        for descriptor in (self._manifest_fd, self._offsets_fd):
            if descriptor is not None:
                os.close(descriptor)
        self._manifest_fd = None
        self._offsets_fd = None

    def __enter__(self) -> "ManifestReader":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

