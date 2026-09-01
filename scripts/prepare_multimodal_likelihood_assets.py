#!/usr/bin/env python3
"""Normalize multimodal likelihood benchmarks without calculating file hashes.

The produced task manifests contain only image-conditioned text likelihood
requests.  They deliberately do not require free-form answer generation:

* MMBench-dev is converted to semantic candidate-text ranking.  All official
  circular option rows are retained and grouped by original question.
* SugarCrepe is converted to positive-vs-hard-negative caption ranking.
* COCO val2017 captions, when present, are converted to caption perplexity.
* Optional SEED-Bench, POPE, ARO VG-Relation/VG-Attribution and normalized
  Winoground assets can be added through their corresponding command-line
  paths.
* SVO-Probes is represented as shared-caption positive-vs-negative image
  pairs; What’sUp Controlled A/B retains its official individual, pair, and
  four-image-set grouping.

Images are assigned deterministic arithmetic IDs.  No content or file digest
is evaluated anywhere in this script.
"""

from __future__ import annotations

import argparse
import base64
import csv
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Sequence

from PIL import Image


ASSET_SCHEMA = "selfless_multimodal_likelihood_assets_v1"
TASK_SCHEMA = "selfless_multimodal_likelihood_task_v1"
DEFAULT_ROOT = Path("public/benchmarks/selfless_multimodal_likelihood_v1")
DEFAULT_I2T_PREFIX = "Describe this image in one detailed caption:"
SUGARCREPE_SPLITS = (
    "add_att",
    "add_obj",
    "replace_att",
    "replace_obj",
    "replace_rel",
    "swap_att",
    "swap_obj",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


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


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "wb",
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


def materialize_aro_crop(job: tuple[str, str, int, int, int, int]) -> None:
    """Create one lossless official ARO crop in a worker process."""

    source_value, crop_value, x, y, width, height = job
    crop_path = Path(crop_value)
    if crop_path.is_file():
        return
    with Image.open(source_value) as image:
        crop = image.convert("RGB").crop((x, y, x + width, y + height))
        buffer = io.BytesIO()
        crop.save(buffer, format="PNG")
    atomic_write_bytes(crop_path, buffer.getvalue())


def jsonl_text(rows: Iterable[dict[str, Any]]) -> str:
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    )


def source_stat(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def normalize_text(value: Any) -> str:
    # Preserve official line breaks (notably in code/OCR choices); only remove
    # surrounding whitespace introduced by tabular serialization.
    return str(value or "").strip()


def image_format_and_extension(payload: bytes) -> tuple[str, str]:
    with Image.open(io.BytesIO(payload)) as image:
        image.verify()
        image_format = str(image.format or "").upper()
    extensions = {
        "JPEG": ".jpg",
        "JPG": ".jpg",
        "PNG": ".png",
        "WEBP": ".webp",
        "GIF": ".gif",
        "BMP": ".bmp",
    }
    if image_format not in extensions:
        raise ValueError(f"unsupported embedded image format: {image_format!r}")
    return image_format, extensions[image_format]


@dataclass
class ImageRecord:
    img_id: int
    source_path: Path
    benchmark_sources: set[str] = field(default_factory=set)


class ImageRegistry:
    def __init__(self) -> None:
        self._by_id: dict[int, ImageRecord] = {}
        self._id_by_path: dict[Path, int] = {}

    def add(self, img_id: int, source_path: Path, benchmark: str) -> int:
        img_id = int(img_id)
        source_path = source_path.resolve()
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        existing = self._by_id.get(img_id)
        if existing is not None and existing.source_path != source_path:
            raise ValueError(
                f"img_id={img_id} maps to both {existing.source_path} and {source_path}"
            )
        existing_id = self._id_by_path.get(source_path)
        if existing_id is not None and existing_id != img_id:
            raise ValueError(
                f"image path {source_path} maps to both {existing_id} and {img_id}"
            )
        if existing is None:
            existing = ImageRecord(img_id=img_id, source_path=source_path)
            self._by_id[img_id] = existing
            self._id_by_path[source_path] = img_id
        existing.benchmark_sources.add(str(benchmark))
        return img_id

    def rows(self) -> list[dict[str, Any]]:
        return [
            {
                "img_id": int(record.img_id),
                "source_path": str(record.source_path),
                "benchmark_sources": sorted(record.benchmark_sources),
                "synset": None,
            }
            for record in sorted(self._by_id.values(), key=lambda item: item.img_id)
        ]

    def __len__(self) -> int:
        return len(self._by_id)


def task_record(
    *,
    task: str,
    kind: str,
    item_id: str,
    image_id: int,
    prompt: str,
    candidates: Sequence[str],
    label: int | None,
    category: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    normalized_candidates = [normalize_text(value) for value in candidates]
    if not normalized_candidates or any(not value for value in normalized_candidates):
        raise ValueError(f"{task}/{item_id} contains an empty candidate")
    # A few official MMBench questions contain text-identical distractors.
    # Preserve them exactly; the evaluator resolves a likelihood tie by the
    # original candidate index instead of silently changing the benchmark.
    if label is not None and not 0 <= int(label) < len(normalized_candidates):
        raise ValueError(f"{task}/{item_id} has invalid label={label}")
    row = {
        "schema": TASK_SCHEMA,
        "task": str(task),
        "kind": str(kind),
        "item_id": str(item_id),
        "image_id": int(image_id),
        "prompt": str(prompt).strip(),
        "candidates": normalized_candidates,
        "label": int(label) if label is not None else None,
        "category": str(category) if category else None,
    }
    if metadata:
        row["metadata"] = metadata
    return row


def prepare_mmbench(
    tsv_path: Path,
    image_dir: Path,
    registry: ImageRegistry,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    with tsv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    required = {
        "index",
        "question",
        "hint",
        "A",
        "B",
        "C",
        "D",
        "answer",
        "category",
        "image",
    }
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"MMBench TSV is missing required fields: {tsv_path}")

    embedded_paths: dict[int, Path] = {}
    for row in rows:
        index = int(row["index"])
        if index >= 1_000_000:
            continue
        payload = base64.b64decode(row["image"], validate=True)
        _, extension = image_format_and_extension(payload)
        path = image_dir / f"{index:07d}{extension}"
        if not path.is_file() or path.stat().st_size != len(payload):
            atomic_write_bytes(path, payload)
        embedded_paths[index] = path
    if len(embedded_paths) != 1164:
        raise ValueError(
            f"MMBench-dev must contain 1,164 original images, got {len(embedded_paths)}"
        )

    examples: list[dict[str, Any]] = []
    letters = ("A", "B", "C", "D")
    for row in rows:
        index = int(row["index"])
        original_index = index % 1_000_000
        if original_index not in embedded_paths:
            raise ValueError(
                f"MMBench circular row index={index} has no original image row"
            )
        choices = [(letter, row[letter].strip()) for letter in letters if row[letter].strip()]
        choice_letters = [letter for letter, _ in choices]
        answer = row["answer"].strip().upper()
        if answer not in choice_letters:
            raise ValueError(f"MMBench index={index} has answer={answer!r} outside choices")
        hint = str(row.get("hint", "")).strip()
        question = str(row["question"]).strip()
        prompt_parts = ["Use the image to answer the question."]
        if hint:
            prompt_parts.append(hint)
        prompt_parts.append(f"Question: {question}")
        prompt_parts.append("Answer:")
        img_id = registry.add(
            1_000_000_000 + original_index,
            embedded_paths[original_index],
            "mmbench_dev_en",
        )
        examples.append(
            task_record(
                task="mmbench_dev_en",
                kind="mmbench_circular_multiple_choice",
                item_id=str(index),
                image_id=img_id,
                prompt="\n".join(prompt_parts),
                candidates=[choice for _, choice in choices],
                label=choice_letters.index(answer),
                category=str(row.get("l2-category") or row.get("category") or ""),
                metadata={
                    "fine_category": str(row.get("category", "")),
                    "source": str(row.get("source", "")),
                    "answer_letter": answer,
                    "choice_letters": choice_letters,
                    "official_index": index,
                    "original_index": original_index,
                    "circular_variant": index // 1_000_000,
                    "is_original": index < 1_000_000,
                    "protocol_variant": "semantic_candidate_normalized_nll",
                },
            )
        )
    return examples, {
        "source": source_stat(tsv_path),
        "official_original_questions": 1164,
        "official_circular_rows": len(rows),
        "retained_rows": len(examples),
        "circular_rows_retained": True,
        "evaluation": (
            "all official circular rows are scored; strict circular accuracy "
            "requires every option rotation for an original question to pass"
        ),
    }


def prepare_sugarcrepe(
    data_dir: Path,
    coco_val2017_dir: Path,
    registry: ImageRegistry,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for split in SUGARCREPE_SPLITS:
        path = data_dir / f"{split}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"SugarCrepe split must be a JSON object: {path}")
        sources.append(source_stat(path))
        for key in sorted(payload, key=lambda value: int(value)):
            row = payload[key]
            filename = str(row["filename"])
            stem = Path(filename).stem
            if not stem.isdigit():
                raise ValueError(f"unexpected COCO filename in {path}: {filename}")
            image_path = coco_val2017_dir / filename
            img_id = registry.add(
                2_000_000_000 + int(stem), image_path, "sugarcrepe"
            )
            examples.append(
                task_record(
                    task="sugarcrepe",
                    kind="pairwise_caption_ranking",
                    item_id=f"{split}/{key}",
                    image_id=img_id,
                    prompt=DEFAULT_I2T_PREFIX,
                    candidates=[row["caption"], row["negative_caption"]],
                    label=0,
                    category=split,
                    metadata={"filename": filename},
                )
            )
    return examples, {
        "sources": sources,
        "coco_image_root": str(coco_val2017_dir.resolve()),
        "retained_rows": len(examples),
    }


def prepare_coco_caption_ppl(
    annotations_path: Path,
    coco_val2017_dir: Path,
    registry: ImageRegistry,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = json.loads(annotations_path.read_text(encoding="utf-8"))
    images = {int(row["id"]): str(row["file_name"]) for row in payload["images"]}
    examples: list[dict[str, Any]] = []
    for row in sorted(payload["annotations"], key=lambda value: int(value["id"])):
        image_id = int(row["image_id"])
        filename = images[image_id]
        img_id = registry.add(
            2_000_000_000 + image_id,
            coco_val2017_dir / filename,
            "coco_caption_ppl",
        )
        examples.append(
            task_record(
                task="coco_caption_ppl",
                kind="caption_perplexity",
                item_id=str(row["id"]),
                image_id=img_id,
                prompt=DEFAULT_I2T_PREFIX,
                candidates=[row["caption"]],
                label=0,
                category="caption",
                metadata={
                    "coco_image_id": image_id,
                    "filename": filename,
                },
            )
        )
    return examples, {
        "source": source_stat(annotations_path),
        "coco_image_root": str(coco_val2017_dir.resolve()),
        "images": len(images),
        "retained_rows": len(examples),
    }


def find_image(root: Path, value: str) -> Path:
    raw = Path(str(value))
    candidates = [root / raw]
    if raw.suffix == "":
        candidates.extend(
            [root / f"{raw}.jpg", root / f"{raw}.png", root / f"{raw}.jpeg"]
        )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    matches = list(root.rglob(raw.name if raw.suffix else f"{raw.name}.*"))
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"could not resolve image {value!r} under {root}")


def prepare_seed_bench_image(
    questions_path: Path,
    image_root: Path,
    registry: ImageRegistry,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = json.loads(questions_path.read_text(encoding="utf-8"))
    questions = payload["questions"] if isinstance(payload, dict) else payload
    question_type = payload.get("question_type", {}) if isinstance(payload, dict) else {}
    question_type_by_id = {
        int(value): str(name) for name, value in question_type.items()
    }
    examples: list[dict[str, Any]] = []
    skipped_non_single_image = 0
    image_ids_by_path: dict[Path, int] = {}
    for source_index, row in enumerate(questions):
        data_id = row.get("data_id")
        data_type = str(row.get("data_type", "image")).strip().lower()
        if isinstance(data_id, list) or data_type not in {"image", "single image"}:
            skipped_non_single_image += 1
            continue
        image_path = find_image(image_root, str(data_id))
        resolved_image = image_path.resolve()
        img_id = image_ids_by_path.get(resolved_image)
        if img_id is None:
            img_id = 3_000_000_000 + len(image_ids_by_path)
            image_ids_by_path[resolved_image] = img_id
        img_id = registry.add(img_id, resolved_image, "seed_bench_image")
        choices = [row[f"choice_{letter}"] for letter in "abcd"]
        answer = str(row["answer"]).strip().upper()
        question_type_id = int(row["question_type_id"])
        category = question_type_by_id.get(
            question_type_id, str(question_type_id)
        )
        examples.append(
            task_record(
                task="seed_bench_image",
                kind="multiple_choice",
                item_id=str(row.get("question_id", source_index)),
                image_id=img_id,
                prompt=(
                    "Use the image to answer the question.\n"
                    f"Question: {str(row['question']).strip()}\nAnswer:"
                ),
                candidates=choices,
                label="ABCD".index(answer),
                category=category,
                metadata={
                    "data_id": str(data_id),
                    "question_type_id": question_type_id,
                    "question_type": category,
                },
            )
        )
    return examples, {
        "source": source_stat(questions_path),
        "image_root": str(image_root.resolve()),
        "retained_rows": len(examples),
        "unique_images": len(image_ids_by_path),
        "question_types": {
            str(key): value for key, value in sorted(question_type_by_id.items())
        },
        "skipped_non_single_image": skipped_non_single_image,
    }


def prepare_pope(
    annotation_dir: Path,
    coco_val2014_dir: Path,
    registry: ImageRegistry,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for strategy in ("random", "popular", "adversarial"):
        path = annotation_dir / f"coco_pope_{strategy}.json"
        sources.append(source_stat(path))
        with path.open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        for source_index, row in enumerate(rows):
            filename = str(row["image"])
            digits = "".join(character for character in Path(filename).stem if character.isdigit())
            if not digits:
                raise ValueError(f"cannot obtain COCO image id from {filename!r}")
            image_id = int(digits[-12:])
            img_id = registry.add(
                4_000_000_000 + image_id,
                coco_val2014_dir / filename,
                "pope_coco",
            )
            label_text = str(row["label"]).strip().lower()
            if label_text not in {"yes", "no"}:
                raise ValueError(f"unexpected POPE label: {label_text!r}")
            examples.append(
                task_record(
                    task="pope_coco",
                    kind="binary_classification",
                    item_id=f"{strategy}/{row.get('question_id', source_index)}",
                    image_id=img_id,
                    prompt=f"{str(row['text']).strip()}\nAnswer:",
                    candidates=["yes", "no"],
                    label=0 if label_text == "yes" else 1,
                    category=strategy,
                    metadata={"filename": filename},
                )
            )
    return examples, {
        "sources": sources,
        "coco_image_root": str(coco_val2014_dir.resolve()),
        "retained_rows": len(examples),
    }


def prepare_aro_vg_task(
    *,
    task: str,
    annotation_path: Path,
    image_root: Path,
    crop_root: Path,
    id_base: int,
    registry: ImageRegistry,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Normalize the official cropped ARO VG relation/attribution protocol."""

    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"ARO annotation must be a JSON list: {annotation_path}")
    normalized: list[tuple[int, dict[str, Any], Path, Path, int, int, int, int]] = []
    crop_jobs: list[tuple[str, str, int, int, int, int]] = []
    for source_index, row in enumerate(payload):
        source_path = find_image(image_root, str(row["image_path"]))
        x = int(row["bbox_x"])
        y = int(row["bbox_y"])
        width = int(row["bbox_w"])
        height = int(row["bbox_h"])
        if width <= 0 or height <= 0:
            raise ValueError(f"invalid ARO crop at row {source_index}")
        crop_path = crop_root / f"{source_index:07d}.png"
        if not crop_path.is_file():
            crop_jobs.append(
                (str(source_path), str(crop_path), x, y, width, height)
            )
        normalized.append(
            (source_index, row, source_path, crop_path, x, y, width, height)
        )

    crop_workers = int(
        os.environ.get("ARO_CROP_WORKERS", str(min(32, os.cpu_count() or 1)))
    )
    if crop_workers <= 0:
        raise ValueError("ARO_CROP_WORKERS must be positive")
    if crop_jobs:
        with ProcessPoolExecutor(max_workers=crop_workers) as executor:
            for _ in executor.map(materialize_aro_crop, crop_jobs, chunksize=8):
                pass

    examples: list[dict[str, Any]] = []
    for source_index, row, source_path, crop_path, x, y, width, height in normalized:
        img_id = registry.add(id_base + source_index, crop_path, task)
        if task == "aro_vg_relation":
            category = str(row.get("relation_name", ""))
        else:
            attributes = row.get("attributes") or []
            category = "_".join(str(value) for value in attributes)
        examples.append(
            task_record(
                task=task,
                kind="pairwise_caption_ranking",
                item_id=str(source_index),
                image_id=img_id,
                prompt=DEFAULT_I2T_PREFIX,
                candidates=[row["true_caption"], row["false_caption"]],
                label=0,
                category=category,
                metadata={
                    "source_image": str(source_path.resolve()),
                    "crop_xywh": [x, y, width, height],
                },
            )
        )
    return examples, {
        "source": source_stat(annotation_path),
        "image_root": str(image_root.resolve()),
        "cropped_image_root": str(crop_root.resolve()),
        "crop_workers": crop_workers,
        "retained_rows": len(examples),
    }


def prepare_winoground_normalized(
    jsonl_path: Path,
    image_root: Path,
    registry: ImageRegistry,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read a user-authorized Winoground export without handling credentials.

    Each JSONL row must provide ``id``, ``caption_0``, ``caption_1``,
    ``image_0`` and ``image_1``.  Two ranking rows are emitted per official
    group, one for each image.
    """

    examples: list[dict[str, Any]] = []
    with jsonl_path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    for source_index, row in enumerate(rows):
        group_id = str(row.get("id", source_index))
        captions = [row["caption_0"], row["caption_1"]]
        for image_slot in (0, 1):
            image_path = find_image(image_root, str(row[f"image_{image_slot}"]))
            img_id = registry.add(
                5_000_000_000 + 2 * source_index + image_slot,
                image_path,
                "winoground",
            )
            examples.append(
                task_record(
                    task="winoground",
                    kind="winoground_pair",
                    item_id=f"{group_id}/image_{image_slot}",
                    image_id=img_id,
                    prompt=DEFAULT_I2T_PREFIX,
                    candidates=captions,
                    label=image_slot,
                    category=str(row.get("collapsed_tag", "")),
                    metadata={
                        "group_id": group_id,
                        "image_slot": image_slot,
                        "tags": row.get("tags", []),
                    },
                )
            )
    return examples, {
        "source": source_stat(jsonl_path),
        "image_root": str(image_root.resolve()),
        "groups": len(rows),
        "retained_rows": len(examples),
    }


def parse_csv_bool(value: Any) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"invalid CSV boolean: {value!r}")


def prepare_svo_probes(
    csv_path: Path,
    image_root: Path,
    registry: ImageRegistry,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Normalize the official SVO-Probes image-pair protocol.

    Images are deliberately local-only.  The official URLs are retained as
    readable metadata, but this normalizer never downloads them implicitly.
    """

    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "sentence",
        "pos_triplet",
        "neg_triplet",
        "pos_url",
        "neg_url",
        "pos_image_id",
        "neg_image_id",
        "subj_neg",
        "verb_neg",
        "obj_neg",
    }
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"SVO-Probes CSV is missing required fields: {csv_path}")

    examples: list[dict[str, Any]] = []
    negative_type_counts: dict[str, int] = defaultdict(int)
    for source_index, row in enumerate(rows):
        active = [
            name
            for name, column in (
                ("subject", "subj_neg"),
                ("verb", "verb_neg"),
                ("object", "obj_neg"),
            )
            if parse_csv_bool(row[column])
        ]
        if len(active) != 1:
            raise ValueError(
                f"SVO-Probes row {source_index} must have one negative type: {active}"
            )
        negative_type = active[0]
        negative_type_counts[negative_type] += 1
        sentence = normalize_text(row["sentence"])
        pair_id = str(source_index)
        shared_metadata = {
            "pair_id": pair_id,
            "negative_type": negative_type,
            "positive_triplet": normalize_text(row["pos_triplet"]),
            "negative_triplet": normalize_text(row["neg_triplet"]),
        }
        for role, prefix in (("positive", "pos"), ("negative", "neg")):
            official_image_id = int(row[f"{prefix}_image_id"])
            image_path = find_image(image_root, str(official_image_id))
            img_id = registry.add(
                7_000_000_000 + official_image_id,
                image_path,
                "svo_probes",
            )
            examples.append(
                task_record(
                    task="svo_probes",
                    kind="svo_image_pair",
                    item_id=f"{pair_id}/{role}",
                    image_id=img_id,
                    prompt=DEFAULT_I2T_PREFIX,
                    candidates=[sentence],
                    label=0,
                    category=negative_type,
                    metadata={
                        **shared_metadata,
                        "image_role": role,
                        "official_image_id": official_image_id,
                        "official_url": str(row[f"{prefix}_url"]),
                    },
                )
            )
    return examples, {
        "source": source_stat(csv_path),
        "image_root": str(image_root.resolve()),
        "official_pairs": len(rows),
        "retained_rows": len(examples),
        "negative_type_counts": dict(sorted(negative_type_counts.items())),
        "evaluation": (
            "the shared sentence must receive higher conditional likelihood "
            "on the official positive image than on the negative image"
        ),
    }


def whatsup_relation(image_path: str, subset: str) -> str:
    name = Path(image_path).name
    if "left_of" in name:
        return "left"
    if "right_of" in name:
        return "right"
    if subset == "A":
        if "_on_" in name:
            return "on"
        if "under" in name:
            return "under"
    else:
        if "in-front_of" in name:
            return "in-front"
        if "behind" in name:
            return "behind"
    raise ValueError(f"cannot infer What’sUp relation from {image_path!r}")


def prepare_whatsup_controlled(
    annotation_a: Path,
    image_root_a: Path,
    annotation_b: Path,
    image_root_b: Path,
    registry: ImageRegistry,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Normalize What’sUp Controlled A/B with official grouping semantics."""

    examples: list[dict[str, Any]] = []
    sources: dict[str, Any] = {}
    grouped_relations: dict[tuple[str, str], set[str]] = defaultdict(set)
    next_image_id = 7_500_000_000
    for subset, annotation_path, image_root in (
        ("A", annotation_a, image_root_a),
        ("B", annotation_b, image_root_b),
    ):
        payload = json.loads(annotation_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(
                f"What’sUp Controlled-{subset} annotation must be a list"
            )
        sources[subset] = {
            "annotation": source_stat(annotation_path),
            "image_root": str(image_root.resolve()),
            "records": len(payload),
        }
        for source_index, row in enumerate(payload):
            original_image_path = str(row["image_path"])
            image_path = find_image(image_root, Path(original_image_path).name)
            candidates = list(row["caption_options"])
            if len(candidates) != 4:
                raise ValueError(
                    f"What’sUp Controlled-{subset} row {source_index} needs four captions"
                )
            filename_parts = image_path.stem.split("_")
            if len(filename_parts) < 3:
                raise ValueError(f"unexpected What’sUp filename: {image_path.name}")
            # This is the exact grouping used by the official evaluator: first
            # and last filename components identify the fixed object pair.
            set_id = f"{filename_parts[0]}::{filename_parts[-1]}"
            relation = whatsup_relation(image_path.name, subset)
            grouped_relations[(subset, set_id)].add(relation)
            img_id = registry.add(next_image_id, image_path, "whatsup_controlled")
            next_image_id += 1
            examples.append(
                task_record(
                    task="whatsup_controlled",
                    kind="whatsup_controlled_spatial",
                    item_id=f"{subset}/{source_index}",
                    image_id=img_id,
                    prompt=DEFAULT_I2T_PREFIX,
                    candidates=candidates,
                    label=0,
                    category=f"{subset}/{relation}",
                    metadata={
                        "subset": subset,
                        "set_id": set_id,
                        "relation": relation,
                        "original_image_path": original_image_path,
                    },
                )
            )

    expected = {
        "A": {"left", "right", "on", "under"},
        "B": {"left", "right", "in-front", "behind"},
    }
    incomplete = {
        f"{subset}/{set_id}": sorted(relations)
        for (subset, set_id), relations in grouped_relations.items()
        if relations != expected[subset]
    }
    if incomplete:
        preview = dict(list(sorted(incomplete.items()))[:8])
        raise ValueError(f"incomplete What’sUp four-image sets: {preview}")
    return examples, {
        "sources": sources,
        "sets": len(grouped_relations),
        "retained_rows": len(examples),
        "correct_candidate_index": 0,
        "metrics": ["individual_accuracy", "pair_accuracy", "set_accuracy"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--mmbench_tsv", type=Path, default=None)
    parser.add_argument("--sugarcrepe_dir", type=Path, default=None)
    parser.add_argument("--coco_val2017_dir", type=Path, default=None)
    parser.add_argument("--coco_captions_json", type=Path, default=None)
    parser.add_argument("--seed_questions_json", type=Path, default=None)
    parser.add_argument("--seed_image_root", type=Path, default=None)
    parser.add_argument("--pope_annotation_dir", type=Path, default=None)
    parser.add_argument("--coco_val2014_dir", type=Path, default=None)
    parser.add_argument("--aro_root", type=Path, default=None)
    parser.add_argument("--winoground_jsonl", type=Path, default=None)
    parser.add_argument("--winoground_image_root", type=Path, default=None)
    parser.add_argument("--svo_csv", type=Path, default=None)
    parser.add_argument("--svo_image_root", type=Path, default=None)
    parser.add_argument("--whatsup_annotation_a", type=Path, default=None)
    parser.add_argument("--whatsup_image_root_a", type=Path, default=None)
    parser.add_argument("--whatsup_annotation_b", type=Path, default=None)
    parser.add_argument("--whatsup_image_root_b", type=Path, default=None)
    parser.add_argument(
        "--required_tasks",
        default=(
            "mmbench_dev_en,seed_bench_image,sugarcrepe,"
            "aro_vg_relation,aro_vg_attribution"
        ),
        help="Comma-separated tasks that must be available after normalization.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root
    raw = root / "raw"
    mmbench_tsv = args.mmbench_tsv or raw / "mmbench" / "MMBench_DEV_EN.tsv"
    sugarcrepe_dir = args.sugarcrepe_dir or raw / "sugarcrepe"
    coco_val2017_dir = args.coco_val2017_dir or raw / "coco" / "val2017"
    coco_captions_json = (
        args.coco_captions_json
        or raw / "coco" / "annotations" / "captions_val2017.json"
    )

    registry = ImageRegistry()
    tasks: dict[str, dict[str, Any]] = {}
    task_rows: dict[str, list[dict[str, Any]]] = {}
    unavailable: dict[str, str] = {}

    def prepare(
        name: str,
        prerequisites: Sequence[Path | None],
        callback,
    ) -> None:
        missing = [str(path) for path in prerequisites if path is None or not path.exists()]
        if missing:
            unavailable[name] = "missing required asset(s): " + ", ".join(missing)
            return
        try:
            rows, details = callback()
        except (FileNotFoundError, ValueError, KeyError) as exc:
            unavailable[name] = f"normalization failed: {type(exc).__name__}: {exc}"
            return
        if not rows:
            unavailable[name] = "normalization produced zero records"
            return
        task_rows[name] = rows
        task_path = root / "tasks" / f"{name}.jsonl"
        atomic_write_text(task_path, jsonl_text(rows))
        tasks[name] = {
            "status": "ready",
            "kind": rows[0]["kind"],
            "records": len(rows),
            "candidate_requests": sum(len(row["candidates"]) for row in rows),
            "manifest_jsonl": str(task_path.resolve()),
            "details": details,
        }

    prepare(
        "mmbench_dev_en",
        [mmbench_tsv],
        lambda: prepare_mmbench(
            mmbench_tsv,
            root / "images" / "mmbench",
            registry,
        ),
    )
    sugar_paths = [sugarcrepe_dir / f"{split}.json" for split in SUGARCREPE_SPLITS]
    prepare(
        "sugarcrepe",
        [*sugar_paths, coco_val2017_dir],
        lambda: prepare_sugarcrepe(sugarcrepe_dir, coco_val2017_dir, registry),
    )
    prepare(
        "coco_caption_ppl",
        [coco_captions_json, coco_val2017_dir],
        lambda: prepare_coco_caption_ppl(
            coco_captions_json, coco_val2017_dir, registry
        ),
    )

    if args.seed_questions_json is not None or args.seed_image_root is not None:
        prepare(
            "seed_bench_image",
            [args.seed_questions_json, args.seed_image_root],
            lambda: prepare_seed_bench_image(
                args.seed_questions_json, args.seed_image_root, registry
            ),
        )
    else:
        unavailable["seed_bench_image"] = (
            "official SEED-Bench image assets were not supplied; video and "
            "multi-image subsets are outside this single-image likelihood protocol"
        )

    if args.pope_annotation_dir is not None or args.coco_val2014_dir is not None:
        prepare(
            "pope_coco",
            [args.pope_annotation_dir, args.coco_val2014_dir],
            lambda: prepare_pope(
                args.pope_annotation_dir, args.coco_val2014_dir, registry
            ),
        )
    else:
        unavailable["pope_coco"] = (
            "official POPE annotations and COCO val2014 images were not supplied"
        )

    if args.aro_root is not None:
        prepare(
            "aro_vg_relation",
            [args.aro_root / "visual_genome_relation.json", args.aro_root / "images"],
            lambda: prepare_aro_vg_task(
                task="aro_vg_relation",
                annotation_path=args.aro_root / "visual_genome_relation.json",
                image_root=args.aro_root / "images",
                crop_root=root / "images" / "aro_vg_relation",
                id_base=6_000_000_000,
                registry=registry,
            ),
        )
        prepare(
            "aro_vg_attribution",
            [
                args.aro_root / "visual_genome_attribution.json",
                args.aro_root / "images",
            ],
            lambda: prepare_aro_vg_task(
                task="aro_vg_attribution",
                annotation_path=args.aro_root / "visual_genome_attribution.json",
                image_root=args.aro_root / "images",
                crop_root=root / "images" / "aro_vg_attribution",
                id_base=6_100_000_000,
                registry=registry,
            ),
        )
    else:
        unavailable["aro_vg_relation"] = "official ARO VG assets were not supplied"
        unavailable["aro_vg_attribution"] = "official ARO VG assets were not supplied"
    unavailable["aro_coco_order"] = (
        "official COCO-2014 Karpathy images and POS-tokenizer assets were not supplied"
    )
    unavailable["aro_flickr30k_order"] = (
        "Flickr30K requires the dataset owner's manual access flow"
    )

    if args.winoground_jsonl is not None or args.winoground_image_root is not None:
        prepare(
            "winoground",
            [args.winoground_jsonl, args.winoground_image_root],
            lambda: prepare_winoground_normalized(
                args.winoground_jsonl, args.winoground_image_root, registry
            ),
        )
    else:
        unavailable["winoground"] = (
            "gated Winoground files were not supplied; accept the official "
            "research terms and provide a local normalized export"
        )

    if args.svo_csv is not None or args.svo_image_root is not None:
        prepare(
            "svo_probes",
            [args.svo_csv, args.svo_image_root],
            lambda: prepare_svo_probes(
                args.svo_csv, args.svo_image_root, registry
            ),
        )
    else:
        unavailable["svo_probes"] = (
            "official SVO-Probes CSV and locally materialized image files were "
            "not supplied; the normalizer never downloads the source URLs implicitly"
        )

    whatsup_values = (
        args.whatsup_annotation_a,
        args.whatsup_image_root_a,
        args.whatsup_annotation_b,
        args.whatsup_image_root_b,
    )
    if any(value is not None for value in whatsup_values):
        prepare(
            "whatsup_controlled",
            list(whatsup_values),
            lambda: prepare_whatsup_controlled(
                args.whatsup_annotation_a,
                args.whatsup_image_root_a,
                args.whatsup_annotation_b,
                args.whatsup_image_root_b,
                registry,
            ),
        )
    else:
        unavailable["whatsup_controlled"] = (
            "official What’sUp Controlled-A/B annotations and image roots were not supplied"
        )

    image_manifest = root / "image_manifest.jsonl"
    atomic_write_text(image_manifest, jsonl_text(registry.rows()))
    required = {
        value.strip()
        for value in str(args.required_tasks).split(",")
        if value.strip()
    }
    missing_required = sorted(required - set(tasks))
    manifest = {
        "schema": ASSET_SCHEMA,
        "created_at": utc_now(),
        "complete": not missing_required,
        "runtime_hashing_enabled": False,
        "image_manifest_jsonl": str(image_manifest.resolve()),
        "images": len(registry),
        "tasks": tasks,
        "unavailable": unavailable,
        "required_tasks": sorted(required),
        "missing_required_tasks": missing_required,
        "scoring_contract": {
            "model_family": "selfless_dual_stream",
            "prediction_position": "same_position_query_stream",
            "primary_candidate_score": "mean_token_loglikelihood",
            "free_form_generation_required": False,
            "runtime_hashing_enabled": False,
        },
    }
    atomic_write_text(
        root / "manifest.json",
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    if missing_required:
        raise RuntimeError(f"missing required tasks: {missing_required}")
    print(
        json.dumps(
            {
                "manifest": str((root / "manifest.json").resolve()),
                "images": len(registry),
                "tasks": {name: value["records"] for name, value in tasks.items()},
                "unavailable": unavailable,
                "runtime_hashing_enabled": False,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
