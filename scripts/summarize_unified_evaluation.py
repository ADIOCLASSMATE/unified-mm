#!/usr/bin/env python3
"""Validate unified evaluation artifacts and write one machine-readable summary."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from utils.evaluation.model_contracts import validate_image_generation_report
from utils.evaluation_model_source import (
    EvaluationModelSource,
    resolve_evaluation_model_source,
)

IS_SPLIT_ASSIGNMENT = "stratified_by_synset"
GENERATION_PROTOCOL_NAME = (
    "imagenet_val_fid50k_torch_fidelity_stratified_is"
)
GENERATION_REFERENCE_DISTRIBUTION = "imagenet_val_50000"
GENERATION_COMPARISON_SCOPE = "same_protocol_only"
GENERATION_NON_COMPARABILITY_REASON = (
    "validation_reference_and_pytorch_torch_fidelity_extractor"
)
GENERATION_FID_REDUCER = "symmetric_eigendecomposition"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--profile", choices=("smoke", "formal"), required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def require_finite(value: Any, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite evaluation metric: {label}={number}")
    return number


def validate_t2i_is_protocol(
    t2i: dict[str, Any], *, profile: str
) -> dict[str, Any]:
    metric_protocol = t2i.get("metric_protocol", {})
    if metric_protocol.get("is_split_assignment") != IS_SPLIT_ASSIGNMENT:
        raise ValueError(
            "T2I Inception Score does not use synset-stratified splits"
        )
    plan = metric_protocol.get("is_split_plan")
    if not isinstance(plan, dict):
        raise ValueError("T2I Inception Score split plan is missing")
    if plan.get("assignment") != IS_SPLIT_ASSIGNMENT:
        raise ValueError("T2I Inception Score split plan is inconsistent")
    if plan.get("source_dataset_split") != "val":
        raise ValueError("T2I Inception Score source is not ImageNet val")
    if profile == "formal":
        expected = {
            "samples": 50000,
            "splits": 10,
            "class_count": 1000,
            "samples_per_class_min": 50,
            "samples_per_class_max": 50,
            "samples_per_class_per_split_min": 5,
            "samples_per_class_per_split_max": 5,
        }
        mismatches = {
            key: {"expected": value, "actual": plan.get(key)}
            for key, value in expected.items()
            if plan.get(key) != value
        }
        if list(plan.get("samples_per_split", [])) != [5000] * 10:
            mismatches["samples_per_split"] = {
                "expected": [5000] * 10,
                "actual": plan.get("samples_per_split"),
            }
        if list(plan.get("classes_per_split", [])) != [1000] * 10:
            mismatches["classes_per_split"] = {
                "expected": [1000] * 10,
                "actual": plan.get("classes_per_split"),
            }
        if mismatches:
            raise ValueError(
                "formal T2I Inception Score split balance is invalid: "
                f"{mismatches}"
            )
    return plan


def validate_t2i_generation_protocol(
    t2i: dict[str, Any], *, profile: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate and select the only reportable project FID/IS variant."""

    if t2i.get("schema") != "selfless_imagenet_val_t2i_fid_is_v2":
        raise ValueError("T2I FID/IS result uses an obsolete protocol")
    if t2i.get("runtime_hashing_enabled") is not False:
        raise ValueError("T2I FID/IS did not explicitly disable runtime hashing")
    if t2i.get("leaderboard_comparable_to_adm_dit") is not False:
        raise ValueError("ImageNet-val FID/IS comparability label is invalid")
    if str(t2i.get("split")) != "val":
        raise ValueError("T2I evaluation did not use ImageNet val")
    if t2i.get("real_source") != "cached_original_imagenet_val":
        raise ValueError("T2I FID uses an unexpected real-image source")

    metric_protocol = t2i.get("metric_protocol")
    if not isinstance(metric_protocol, dict):
        raise ValueError("T2I metric protocol is missing")
    expected_protocol = {
        "protocol_name": GENERATION_PROTOCOL_NAME,
        "reference_distribution": GENERATION_REFERENCE_DISTRIBUTION,
        "comparison_scope": GENERATION_COMPARISON_SCOPE,
        "not_adm_dit_reason": GENERATION_NON_COMPARABILITY_REASON,
        "fid_reducer": GENERATION_FID_REDUCER,
        "is_split_assignment": IS_SPLIT_ASSIGNMENT,
        "is_std": "population",
    }
    mismatches = {
        key: {"expected": expected, "actual": metric_protocol.get(key)}
        for key, expected in expected_protocol.items()
        if metric_protocol.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"T2I metric protocol metadata is invalid: {mismatches}")
    is_splits = int(metric_protocol.get("is_splits", -1))
    if is_splits <= 0:
        raise ValueError("T2I Inception Score split count is invalid")
    plan = validate_t2i_is_protocol(t2i, profile=profile)
    if int(plan.get("splits", -1)) != is_splits:
        raise ValueError("T2I Inception Score split count is inconsistent")

    samples = int(t2i.get("samples_evaluated", -1))
    if samples <= 0 or int(t2i.get("samples_requested", -2)) != samples:
        raise ValueError("T2I requested/evaluated sample counts are inconsistent")
    strategies = t2i.get("strategies")
    if not isinstance(strategies, dict) or len(strategies) != 1:
        raise ValueError("T2I result must contain exactly one generation strategy")
    strategy_name, strategy = next(iter(strategies.items()))
    if not strategy_name or not isinstance(strategy, dict):
        raise ValueError("T2I generation strategy is invalid")
    if int(strategy.get("count", -1)) != samples:
        raise ValueError("T2I strategy sample count is inconsistent")

    fid_raw = strategy.get("fid")
    fid = None if fid_raw is None else require_finite(fid_raw, "t2i.fid")
    if fid is not None and fid < 0.0:
        raise ValueError(f"T2I FID must be non-negative, got {fid}")
    is_mean = require_finite(
        strategy.get("inception_score_mean"), "t2i.inception_score_mean"
    )
    is_std = require_finite(
        strategy.get("inception_score_std"), "t2i.inception_score_std"
    )
    is_values = [
        require_finite(value, f"t2i.inception_score_splits[{index}]")
        for index, value in enumerate(
            strategy.get("inception_score_splits", [])
        )
    ]
    if len(is_values) != is_splits:
        raise ValueError("T2I Inception Score split metrics are incomplete")
    if is_mean < 1.0 - 1.0e-9 or any(
        value < 1.0 - 1.0e-9 for value in is_values
    ):
        raise ValueError("T2I Inception Score is below its theoretical minimum")
    if is_std < 0.0:
        raise ValueError("T2I Inception Score standard deviation is negative")
    expected_mean = sum(is_values) / len(is_values)
    expected_std = math.sqrt(
        sum((value - expected_mean) ** 2 for value in is_values)
        / len(is_values)
    )
    if not math.isclose(is_mean, expected_mean, rel_tol=0.0, abs_tol=1.0e-10):
        raise ValueError("T2I Inception Score mean is inconsistent with splits")
    if not math.isclose(is_std, expected_std, rel_tol=0.0, abs_tol=1.0e-10):
        raise ValueError("T2I Inception Score std is inconsistent with splits")
    throughput = require_finite(
        strategy.get("generation_samples_per_second"),
        "t2i.generation_samples_per_second",
    )
    if throughput <= 0.0:
        raise ValueError("T2I generation throughput must be positive")

    if profile == "formal":
        if t2i.get("project_formal_protocol") is not True:
            raise ValueError("formal T2I evaluation is not marked project-formal")
        if samples != 50_000:
            raise ValueError("formal T2I evaluation must contain 50,000 samples")
        if metric_protocol.get("fid_computed") is not True or fid is None:
            raise ValueError("formal T2I evaluation did not compute FID")
        if is_splits != 10:
            raise ValueError("formal T2I evaluation must use 10 IS splits")

    return plan, {
        "strategy": strategy_name,
        "fid": fid,
        "inception_score_mean": is_mean,
        "inception_score_std": is_std,
        "inception_score_splits": is_values,
        "generation_samples_per_second": throughput,
    }


def validate_t2i_model_source(
    t2i: dict[str, Any], source: EvaluationModelSource
) -> None:
    """Require the evaluator's unified source report to match the request."""
    if t2i.get("weight_source") != source.kind:
        raise ValueError("T2I weight source differs from requested source")

    for label, report in (
        ("model-source identity", t2i.get("evaluation_model_source")),
        ("model-source load", t2i.get("model_source_load")),
    ):
        if not isinstance(report, dict):
            raise ValueError(f"T2I {label} is missing")
        path = report.get("path")
        if not path or Path(path).resolve() != source.path:
            raise ValueError(f"T2I {label} path is inconsistent")
        if report.get("kind") != source.kind:
            raise ValueError(f"T2I {label} kind is inconsistent")
        if int(report.get("global_step", -1)) != source.global_step:
            raise ValueError(f"T2I {label} step is inconsistent")

    load_report = t2i["model_source_load"]
    if source.is_hf_final_ema:
        if Path(t2i.get("model_path", "")).resolve() != source.path:
            raise ValueError("T2I evaluation used a different model source")
        if load_report.get("ema_checkpoint") is not None:
            raise ValueError("T2I final-HF evaluation overlaid a sharded EMA")
        return

    overlay = load_report.get("ema_checkpoint")
    if overlay is None or Path(overlay).resolve() != source.path:
        raise ValueError("T2I sharded-EMA load report is inconsistent")


def main() -> None:
    args = parse_args()
    source = resolve_evaluation_model_source(args.checkpoint)
    step = source.global_step
    validation_root = args.output_root / "validation"
    validation_run = load_json(validation_root / "evaluation_run.json")
    validation = load_json(
        validation_root / f"validation_metrics_step_{step}.json"
    )
    t2i = load_json(args.output_root / "t2i-fid-is" / "metrics.json")

    if not validation_run.get("complete"):
        raise ValueError("held-out validation run is not marked complete")
    if int(validation_run["global_step"]) != step:
        raise ValueError("validation/checkpoint step mismatch")
    if Path(validation_run["checkpoint"]).resolve() != source.path:
        raise ValueError("validation used a different model source")
    if validation_run.get("weight_source") != source.kind:
        raise ValueError("validation weight source differs from requested source")
    if validation_run.get("imagenet_split") != "val":
        raise ValueError("held-out validation did not use ImageNet val")
    for name, payload in (("validation", validation_run), ("t2i", t2i)):
        if payload.get("runtime_hashing_enabled") is not False:
            raise ValueError(f"{name} did not explicitly disable runtime hashing")
    validate_t2i_model_source(t2i, source)
    is_split_plan, t2i_metrics = validate_t2i_generation_protocol(
        t2i, profile=args.profile
    )

    val_metrics = validation.get("metrics", {})
    required_validation = (
        "val/loss",
        "val/loss_i2t",
        "val/loss_t2i",
        "val/ppl_text",
        "val/weighted_contribution_i2t",
        "val/weighted_contribution_t2i",
        "val/weighted_contribution_total",
    )
    selected_validation = {
        key: require_finite(val_metrics[key], key) for key in required_validation
    }
    strategy_name = str(t2i_metrics["strategy"])
    caption_dir = validation_root / "validation_i2t_captions" / f"step-{step:08d}"
    caption_path = caption_dir / "captions.jsonl"
    if not caption_path.is_file():
        raise FileNotFoundError(caption_path)
    caption_count = sum(
        1 for line in caption_path.read_text(encoding="utf-8").splitlines() if line.strip()
    )
    generation_path = (
        validation_root / f"validation_generation_step_{step}.json"
    )
    generation = load_json(generation_path)
    # Joint DiT has no image reveal order. Its held-out report uses the
    # architecture's "joint" trace even when the shared FID runner retains
    # its spatial_halton result key.
    heldout_strategy = (
        "joint"
        if generation.get("architecture_variant") == "selfless_joint_dit"
        else strategy_name
    )
    generation_cache_enabled = validate_image_generation_report(
        generation,
        validation_run.get("dual_stream_attention_contract", "selfless_strict"),
        strategy=heldout_strategy,
    )
    image_paths = sorted(
        (validation_root / "validation_flow_images").glob(
            f"step-{step:08d}-*.png"
        )
    )
    if caption_count <= 0 or not image_paths:
        raise ValueError("qualitative validation artifacts are incomplete")

    summary = {
        "schema": "unified_checkpoint_evaluation_summary_v3",
        "complete": True,
        "profile": args.profile,
        "runtime_hashing_enabled": False,
        "checkpoint": str(args.checkpoint.resolve()),
        "global_step": step,
        "weight_source": source.kind,
        "model_source": source.report(),
        "dataset_contract": {
            "training_split": "imagenet_train",
            "evaluation_split": "imagenet_val",
        },
        "heldout_validation": selected_validation,
        "t2i_fid_is": {
            "project_formal_protocol": bool(t2i["project_formal_protocol"]),
            "leaderboard_comparable_to_adm_dit": False,
            "protocol_name": GENERATION_PROTOCOL_NAME,
            "reference_distribution": GENERATION_REFERENCE_DISTRIBUTION,
            "comparison_scope": GENERATION_COMPARISON_SCOPE,
            "not_adm_dit_reason": GENERATION_NON_COMPARABILITY_REASON,
            "fid_reducer": GENERATION_FID_REDUCER,
            "samples": int(t2i["samples_evaluated"]),
            "is_split_assignment": IS_SPLIT_ASSIGNMENT,
            "is_split_plan": is_split_plan,
            **t2i_metrics,
        },
        "qualitative_artifacts": {
            "caption_rows": caption_count,
            "caption_jsonl": str(caption_path.resolve()),
            "generation_report": str(generation_path.resolve()),
            "generation_entry": "model.generate",
            "backbone_kv_cache_enabled": generation_cache_enabled,
            "flow_image_count": len(image_paths),
            "flow_images": [str(path.resolve()) for path in image_paths],
        },
        "components": {
            "validation": str((validation_root / "evaluation_run.json").resolve()),
            "validation_generation": str(generation_path.resolve()),
            "t2i": str((args.output_root / "t2i-fid-is" / "metrics.json").resolve()),
        },
        "completed_at": datetime.now(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        ),
    }
    destination = args.output_root / "evaluation_summary.json"
    destination.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
