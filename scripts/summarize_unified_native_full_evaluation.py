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

from scripts.summarize_pretraining_native_understanding import (
    FORMAL_BENCHMARK_RECORDS,
    INTERNAL_ABLATION_TASKS,
    PAPER_BENCHMARK_TASKS,
    STANDARD_RETRIEVAL_TASKS,
    nested_primary,
    require_benchmark_metric_contract,
    require_calibrated_protocol,
    require_imagenet_metric_contract,
    require_retrieval_metric_contract,
    require_unit_interval,
)
from utils.evaluation_model_source import resolve_evaluation_model_source


GENERATION_PROTOCOL = "imagenet_val_fid50k_torch_fidelity_stratified_is"
GENERATION_REFERENCE = "imagenet_val_50000"
GENERATION_SCOPE = "same_protocol_only"
GENERATION_NON_COMPARABILITY_REASON = (
    "validation_reference_and_pytorch_torch_fidelity_extractor"
)


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


def validate_formal_generation(generation: dict[str, Any]) -> None:
    expected = {
        "project_formal_protocol": True,
        "leaderboard_comparable_to_adm_dit": False,
        "protocol_name": GENERATION_PROTOCOL,
        "reference_distribution": GENERATION_REFERENCE,
        "comparison_scope": GENERATION_SCOPE,
        "not_adm_dit_reason": GENERATION_NON_COMPARABILITY_REASON,
        "fid_reducer": "symmetric_eigendecomposition",
        "samples": 50_000,
        "is_split_assignment": "stratified_by_synset",
    }
    mismatches = {
        key: {"expected": value, "actual": generation.get(key)}
        for key, value in expected.items()
        if generation.get(key) != value
    }
    if mismatches:
        raise ValueError(f"formal generation protocol is invalid: {mismatches}")
    if not str(generation.get("strategy", "")).strip():
        raise ValueError("formal generation strategy is missing")
    fid = require_finite(generation.get("fid"), "generation.fid")
    is_mean = require_finite(
        generation.get("inception_score_mean"), "generation.is_mean"
    )
    is_std = require_finite(
        generation.get("inception_score_std"), "generation.is_std"
    )
    is_values = [
        require_finite(value, f"generation.is_splits[{index}]")
        for index, value in enumerate(
            generation.get("inception_score_splits", [])
        )
    ]
    if fid < 0.0 or is_mean < 1.0 - 1.0e-9 or is_std < 0.0:
        raise ValueError("formal generation metric range is invalid")
    if len(is_values) != 10 or any(
        value < 1.0 - 1.0e-9 for value in is_values
    ):
        raise ValueError("formal generation IS split metrics are invalid")
    expected_mean = sum(is_values) / len(is_values)
    expected_std = math.sqrt(
        sum((value - expected_mean) ** 2 for value in is_values)
        / len(is_values)
    )
    if not math.isclose(is_mean, expected_mean, rel_tol=0.0, abs_tol=1.0e-10):
        raise ValueError("formal generation IS mean is inconsistent")
    if not math.isclose(is_std, expected_std, rel_tol=0.0, abs_tol=1.0e-10):
        raise ValueError("formal generation IS std is inconsistent")


def validate_formal_native(native: dict[str, Any]) -> None:
    classification = native["imagenet1k_zeroshot_classification"]
    benchmarks = native["paper_compositional_benchmarks"]
    internal = native["internal_ablation_diagnostics"]
    require_calibrated_protocol(classification, "ImageNet classification")
    require_imagenet_metric_contract(classification)

    standard = native["standard_cross_dataset_retrieval"]
    if set(standard) != set(STANDARD_RETRIEVAL_TASKS):
        raise ValueError("formal standard retrieval task set is invalid")
    for task in STANDARD_RETRIEVAL_TASKS:
        require_calibrated_protocol(standard[task], f"standard retrieval {task}")
        require_retrieval_metric_contract(standard[task], task)

    if set(benchmarks) != set(PAPER_BENCHMARK_TASKS):
        raise ValueError("formal paper benchmark task set is invalid")
    if set(internal) != set(INTERNAL_ABLATION_TASKS):
        raise ValueError("formal internal diagnostic task set is invalid")
    for task, summary in {**benchmarks, **internal}.items():
        require_benchmark_metric_contract(task, summary)

    selection = native.get("selection_contract") or {}
    expected_selection = {
        "imagenet_validation_images": 50_000,
        "image_text_matching_score_variant": (
            "language_prior_debiased_mean_token_loglikelihood_only"
        ),
        "language_prior_alpha": 1.0,
        "dense_retrieval_language_prior_estimator": (
            "candidate_image_logmeanexp"
        ),
        "hard_negative_language_prior_estimator": (
            "content_free_gaussian_image_logmeanexp"
        ),
        "hard_negative_language_prior_null_images": 3,
    }
    mismatches = {
        key: {"expected": value, "actual": selection.get(key)}
        for key, value in expected_selection.items()
        if selection.get(key) != value
    }
    if mismatches:
        raise ValueError(f"formal native selection contract is invalid: {mismatches}")

    expected_primary = {
        "imagenet1k_zeroshot_top_1_accuracy",
        "imagenet1k_zeroshot_top_5_accuracy",
        *STANDARD_RETRIEVAL_TASKS,
        *PAPER_BENCHMARK_TASKS,
    }
    if set(native.get("primary_metrics", {})) != expected_primary:
        raise ValueError("formal native primary metric set is invalid")
    top1 = require_unit_interval(
        classification["top_1_accuracy"], "ImageNet Top-1"
    )
    top5 = require_unit_interval(
        classification["top_5_accuracy"], "ImageNet Top-5"
    )
    expected_values = {
        "imagenet1k_zeroshot_top_1_accuracy": top1,
        "imagenet1k_zeroshot_top_5_accuracy": top5,
        **{
            task: nested_primary(standard[task])
            for task in STANDARD_RETRIEVAL_TASKS
        },
        **{
            task: nested_primary(benchmarks[task]["metrics"])
            for task in PAPER_BENCHMARK_TASKS
        },
    }
    for key, expected in expected_values.items():
        actual = require_unit_interval(native["primary_metrics"][key], key)
        if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError(f"formal native primary metric is inconsistent: {key}")

    if set(FORMAL_BENCHMARK_RECORDS) != set(benchmarks) | set(internal):
        raise AssertionError("internal benchmark contract constants disagree")


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
    if core.get("schema") != "unified_full_checkpoint_evaluation_summary_v3":
        raise ValueError("core evaluation uses an obsolete text protocol")
    if native.get("schema") != "pretraining_native_understanding_summary_v5":
        raise ValueError("native evaluation uses an obsolete metric protocol")
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
        validate_formal_generation(core["generation"]["imagenet_val_t2i"])
        validate_formal_native(native)

    for task, value in native["primary_metrics"].items():
        require_finite(value, f"native.{task}")

    report = {
        "schema": "unified_native_full_checkpoint_evaluation_summary_v5",
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
                "zero_shot_classification": "imagenet_val_50k",
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
                "imagenet1k_val_50k_zeroshot_classification",
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
                "imagenet_val_retrieval_1k",
                "imagenet_val_retrieval_5k",
                "uncalibrated_retrieval_loglikelihood",
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
