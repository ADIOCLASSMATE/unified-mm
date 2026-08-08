from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any, Iterator

from .io import atomic_write_json, fsync_directory, load_json, sha256_file, utc_now
from .manifest import ManifestReader
from .queue import RESULT_SCHEMA, TaskStore


PUBLISHED_SCHEMA = "imagenet_local_qwen_multicap_v1"


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"non-object row at {path}:{line_number}")
            yield value


def audit_run(run_dir: Path, *, verify_images: bool = False) -> dict[str, Any]:
    store = TaskStore(run_dir)
    try:
        snapshot = store.snapshot()
        errors: list[str] = []
        if snapshot["PENDING"] or snapshot["LEASED"] or snapshot["FAILED"]:
            errors.append(
                "queue is not drained: "
                f"PENDING={snapshot['PENDING']} LEASED={snapshot['LEASED']} FAILED={snapshot['FAILED']}"
            )
        seen_keys: set[str] = set()
        result_count = 0
        missing_samples: list[str] = []
        for ordinal in range(store.total_tasks):
            task = store._task_from_ordinal(ordinal)
            path = store.result_path(task)
            if not path.is_file():
                if len(missing_samples) < 20:
                    missing_samples.append(task["task_key"])
                continue
            record = load_json(path)
            store._validate_visible_result(task, record)
            if record.get("schema") != RESULT_SCHEMA:
                raise ValueError(f"unexpected result schema in {path}")
            key = str(record["task_key"])
            if key in seen_keys:
                raise ValueError(f"duplicate visible key {key}")
            seen_keys.add(key)
            result_count += 1
            if verify_images and not Path(task["source_path"]).is_file():
                raise ValueError(f"ImageNet mapping is not readable: {task['source_path']}")
        visible_files = sum(1 for _ in store.results_dir.glob("*/*.json"))
        if visible_files != result_count:
            errors.append(
                f"result tree contains {visible_files} files but {result_count} canonical keys"
            )
        if missing_samples:
            errors.append(
                f"missing {store.total_tasks - result_count} canonical results; samples={missing_samples}"
            )

        subset_reports: list[dict[str, Any]] = []
        full_by_img_id: dict[int, tuple[str, str]] | None = None
        for raw_path in store.run["dataset"].get("compatibility_manifests") or []:
            subset_path = Path(raw_path)
            if not subset_path.is_file():
                errors.append(f"compatibility manifest is missing: {subset_path}")
                continue
            if full_by_img_id is None:
                full_by_img_id = {}
                for index in range(store.manifest.row_count):
                    row = store.manifest.read(index)
                    full_by_img_id[int(row["img_id"])] = (str(row["path"]), str(row["synset"]))
            checked = 0
            mismatch_samples: list[dict[str, Any]] = []
            for row in _iter_jsonl(subset_path):
                img_id = int(row["img_id"])
                expected = full_by_img_id.get(img_id)
                observed = (str(row.get("path") or ""), str(row.get("synset") or ""))
                if expected is None or (observed[0] and observed != expected):
                    if len(mismatch_samples) < 20:
                        mismatch_samples.append(
                            {"img_id": img_id, "expected": expected, "observed": observed}
                        )
                checked += 1
            subset_reports.append(
                {
                    "path": str(subset_path),
                    "sha256": sha256_file(subset_path),
                    "rows": checked,
                    "mismatches": len(mismatch_samples),
                    "mismatch_samples": mismatch_samples,
                }
            )
            if mismatch_samples:
                errors.append(f"compatibility mapping mismatch in {subset_path}")

        report = {
            "schema": "imagenet_caption_farm_audit_v1",
            "timestamp": utc_now(),
            "run_fingerprint": store.run["run_fingerprint"],
            "snapshot": snapshot,
            "canonical_result_count": result_count,
            "visible_file_count": visible_files,
            "duplicate_keys": 0,
            "missing_samples": missing_samples,
            "verify_images": verify_images,
            "compatibility": subset_reports,
            "status": "passed" if not errors else "failed",
            "errors": errors,
        }
        atomic_write_json(run_dir / "audit.json", report)
        if errors:
            raise RuntimeError("; ".join(errors))
        return report
    finally:
        store.close()


def publish_run(run_dir: Path, *, verify_images: bool = False) -> dict[str, Any]:
    audit = audit_run(run_dir, verify_images=verify_images)
    store = TaskStore(run_dir)
    try:
        destination = Path(store.run["output"]["published_jsonl"])
        metadata_path = Path(
            store.run["output"].get(
                "published_metadata",
                str(destination.with_suffix(destination.suffix + ".metadata.json")),
            )
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.parent / f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        digest = hashlib.sha256()
        with temporary.open("wb") as output:
            for manifest_index in range(store.manifest.row_count):
                identity = store.manifest.read(manifest_index)
                captions: list[dict[str, Any]] = [
                    {
                        "caption_slot": -1,
                        "source": "original",
                        "model": "original",
                        "text": identity["original_caption"],
                        "text_sha256": identity["original_caption_sha256"],
                    }
                ]
                for slot in range(store.captions_per_image):
                    ordinal = manifest_index * store.captions_per_image + slot
                    task = store._task_from_ordinal(ordinal)
                    result = load_json(store.result_path(task))
                    captions.append(
                        {
                            "caption_slot": slot,
                            "source": "local_qwen",
                            "model": store.run["model"]["repository"],
                            "model_fingerprint": store.run["model"]["fingerprint"],
                            "prompt_fingerprint": store.run["caption"]["prompt_fingerprint"],
                            "prompt_version": result["prompt_version"],
                            "text": result["caption"],
                            "word_count": result["word_count"],
                            "text_sha256": result["caption_sha256"],
                            "task_key": result["task_key"],
                        }
                    )
                row = {
                    "schema": PUBLISHED_SCHEMA,
                    "run_fingerprint": store.run["run_fingerprint"],
                    "manifest_sha256": store.run["manifest"]["sha256"],
                    "manifest_index": manifest_index,
                    "img_id": identity["img_id"],
                    "image_id": identity["image_id"],
                    "id": identity["id"],
                    "path": identity["path"],
                    "imagenet_relative_path": identity["imagenet_relative_path"],
                    "source_path": identity["source_path"],
                    "synset": identity["synset"],
                    "class": identity["class"],
                    "original_caption_key": identity["original_caption_key"],
                    "caption_count": len(captions),
                    "captions": captions,
                }
                encoded = (
                    json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                    + "\n"
                ).encode("utf-8")
                output.write(encoded)
                digest.update(encoded)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
        fsync_directory(destination.parent)
        metadata = {
            "schema": PUBLISHED_SCHEMA,
            "published_at": utc_now(),
            "path": str(destination),
            "sha256": digest.hexdigest(),
            "rows": store.manifest.row_count,
            "captions_per_image": store.captions_per_image,
            "captions_including_original_per_row": store.captions_per_image + 1,
            "run_fingerprint": store.run["run_fingerprint"],
            "manifest_sha256": store.run["manifest"]["sha256"],
            "model_repository": store.run["model"]["repository"],
            "model_fingerprint": store.run["model"]["fingerprint"],
            "prompt_fingerprint": store.run["caption"]["prompt_fingerprint"],
            "audit": audit,
        }
        atomic_write_json(metadata_path, metadata)
        return metadata
    finally:
        store.close()

