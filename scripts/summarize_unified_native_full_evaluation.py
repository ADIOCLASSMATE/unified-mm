#!/usr/bin/env python3
"""Combine the stable core suite with ImageNet-val native understanding metrics."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

from utils.evaluation_model_source import resolve_evaluation_model_source


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--core_output_root", type=Path, required=True)
    parser.add_argument("--native_output_root", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--profile", choices=("smoke", "formal"), required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


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


def forbidden_audit_fields(value: Any, prefix: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            lowered = str(key).lower()
            if "sha256" in lowered or "digest" in lowered:
                found.append(path)
            found.extend(forbidden_audit_fields(item, path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(forbidden_audit_fields(item, f"{prefix}[{index}]"))
    return found


def require_finite(value: Any, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite metric: {label}={number}")
    return number


def validate_formal_native(native: dict[str, Any]) -> None:
    retrieval_1k = native["retrieval_1k"]
    retrieval_5k = native["retrieval_5k"]
    benchmarks = native["paper_compositional_benchmarks"]
    internal = native["internal_ablation_diagnostics"]
    if retrieval_1k.get("complete_formal_target") is not True:
        raise ValueError("formal native retrieval-1K is incomplete")
    if retrieval_5k.get("complete_formal_target") is not True:
        raise ValueError("formal native retrieval-5K is incomplete")
    standard = native["standard_cross_dataset_retrieval"]
    for task in (
        "mscoco_karpathy_test_5k",
        "flickr30k_karpathy_test_1k",
    ):
        if standard[task].get("complete_formal_target") is not True:
            raise ValueError(f"formal standard retrieval is incomplete: {task}")
    expected_records = {
        "sugarcrepe": 7_511,
        "aro_vg_relation": 23_937,
        "aro_vg_attribution": 28_748,
    }
    for task, expected in expected_records.items():
        metrics = benchmarks[task]["metrics"]
        if int(metrics.get("records", 0)) != expected:
            raise ValueError(
                f"formal retained benchmark is incomplete: {task} "
                f"({metrics.get('records')} != {expected})"
            )
    expected_internal_records = {
        "mmbench_dev_en": 4_329,
        "seed_bench_image": 14_233,
    }
    for task, summary in internal.items():
        expected = expected_internal_records[task]
        metrics = summary["metrics"]
        if int(metrics.get("records", 0)) != expected:
            raise ValueError(
                f"formal internal diagnostic is incomplete: {task} "
                f"({metrics.get('records')} != {expected})"
            )


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    source = resolve_evaluation_model_source(checkpoint)
    step = source.global_step
    core_path = args.core_output_root / "full_evaluation_summary.json"
    native_path = (
        args.native_output_root / "pretraining_native_understanding_summary.json"
    )
    core = read_json(core_path)
    native = read_json(native_path)

    if core.get("complete") is not True or native.get("complete") is not True:
        raise ValueError("core or pretraining-native evaluation is incomplete")
    if core.get("runtime_hashing_enabled", True) is not False:
        raise ValueError("core evaluation violates the no-hash contract")
    if native.get("runtime_hashing_enabled", True) is not False:
        raise ValueError("native evaluation violates the no-hash contract")
    if Path(core["checkpoint"]).resolve() != checkpoint:
        raise ValueError("core evaluation belongs to another checkpoint")
    if Path(native["checkpoint"]).resolve() != checkpoint:
        raise ValueError("native evaluation belongs to another checkpoint")
    for name, payload in (("core", core), ("native", native)):
        weight_source = payload.get("weight_source")
        if source.is_hf_final_ema and weight_source != source.kind:
            raise ValueError(f"{name} evaluation lacks final-HF source identity")
        if weight_source is not None and weight_source != source.kind:
            raise ValueError(f"{name} evaluation weight source mismatch")
    if int(core["global_step"]) != step or int(native["global_step"]) != step:
        raise ValueError("evaluation component checkpoint steps disagree")
    core_contract = core.get("dataset_contract", {})
    native_contract = native.get("dataset_contract", {})
    if core_contract.get("training_split") != "imagenet_train":
        raise ValueError("core training split is not ImageNet train")
    if core_contract.get("evaluation_split") != "imagenet_val":
        raise ValueError("core evaluation split is not ImageNet val")
    if native_contract.get("image_training_split") != "imagenet_train":
        raise ValueError("native training split is not ImageNet train")
    if native_contract.get("image_evaluation_split") != "imagenet_val":
        raise ValueError("native evaluation split is not ImageNet val")
    if native_contract.get("train_validation_overlap_allowed") is not False:
        raise ValueError("native evaluation permits ImageNet train/val overlap")
    if args.profile == "formal":
        validate_formal_native(native)

    for task, value in native["primary_metrics"].items():
        require_finite(value, f"native.{task}")

    report = {
        "schema": "unified_native_full_checkpoint_evaluation_summary_v3",
        "complete": True,
        "profile": args.profile,
        "runtime_hashing_enabled": False,
        "checkpoint": str(checkpoint),
        "global_step": step,
        "weight_source": source.kind,
        "model_source": source.report(),
        "dataset_contract": {
            "training_split": "imagenet_train",
            "evaluation_split": "imagenet_val",
            "standard_retrieval_splits": [
                "mscoco_karpathy_test_5k",
                "flickr30k_karpathy_test_1k",
            ],
            "train_validation_overlap_allowed": False,
        },
        "generation": core["generation"],
        "understanding": {
            "domains": {
                "custom_in_domain": "imagenet_val",
                "compositional": [
                    "sugarcrepe",
                    "aro_vg_relation",
                    "aro_vg_attribution",
                ],
                "standard_retrieval": [
                    "mscoco_karpathy_test_5k",
                    "flickr30k_karpathy_test_1k",
                ],
                "internal_ablation_only": [
                    "mmbench_dev_en",
                    "seed_bench_image",
                ],
            },
            "pretraining_native": native,
            "heldout_validation": core["understanding"]["heldout_validation"],
            "qualitative_captions": core["understanding"][
                "qualitative_captions"
            ],
            "out_of_domain_general_vlm_benchmarks_in_paper_summary": False,
        },
        "pure_text": core["pure_text"],
        "coverage": {
            "primary_image_understanding_tasks": [
                "imagenet_val_retrieval_1k",
                "imagenet_val_retrieval_5k",
                "mscoco_karpathy_test_5k",
                "flickr30k_karpathy_test_1k",
                "sugarcrepe",
                "aro_vg_relation",
                "aro_vg_attribution",
            ],
            "internal_ablation_only_tasks": [
                "mmbench_dev_en",
                "seed_bench_image",
            ],
            "all_primary_image_understanding_tasks_complete": True,
            "removed_from_protocol": [
                "pope_coco",
                "coco_caption_ppl",
                "winoground",
                "svo_probes",
                "whatsup_controlled",
                "imagenet_caption_random_negative",
                "imagenet_caption_same_class_negative",
                "imagenet_caption_near_class_negative",
                "visual_calibration",
                "imagenet_classification_top1_top5",
                "imagenet_real_top1_top5",
            ],
        },
        "components": {
            "core_summary": str(core_path.resolve()),
            "pretraining_native_summary": str(native_path.resolve()),
        },
        "completed_at": datetime.now(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        ),
    }
    forbidden = forbidden_audit_fields(report)
    if forbidden:
        raise ValueError(f"no-hash report contains forbidden fields: {forbidden}")
    destination = args.output_root / "native_full_evaluation_summary.json"
    atomic_write_text(
        destination,
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
