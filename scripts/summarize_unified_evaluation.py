#!/usr/bin/env python3
"""Validate unified evaluation artifacts and write one machine-readable summary."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from utils.evaluation_model_source import (
    EvaluationModelSource,
    resolve_evaluation_model_source,
)

IS_SPLIT_ASSIGNMENT = "stratified_by_synset"


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
    if str(t2i.get("split")) != "val":
        raise ValueError("T2I evaluation did not use ImageNet val")
    validate_t2i_model_source(t2i, source)
    is_split_plan = validate_t2i_is_protocol(t2i, profile=args.profile)
    if args.profile == "formal" and t2i.get("official_protocol") is not True:
        raise ValueError("formal T2I evaluation is not marked official")
    if args.profile == "formal" and t2i.get("metric_protocol", {}).get(
        "fid_computed"
    ) is not True:
        raise ValueError("formal T2I evaluation did not compute FID")

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
    strategy_name = "spatial_halton"
    strategy = t2i.get("strategies", {}).get(strategy_name)
    if not isinstance(strategy, dict):
        raise ValueError(f"missing T2I strategy metrics: {strategy_name}")
    fid_value = strategy.get("fid")
    t2i_metrics = {
        "fid": (
            require_finite(fid_value, "t2i.fid")
            if fid_value is not None
            else None
        ),
        "inception_score_mean": require_finite(
            strategy["inception_score_mean"], "t2i.inception_score_mean"
        ),
        "inception_score_std": require_finite(
            strategy["inception_score_std"], "t2i.inception_score_std"
        ),
        "inception_score_splits": [
            require_finite(value, f"t2i.inception_score_splits[{index}]")
            for index, value in enumerate(
                strategy["inception_score_splits"]
            )
        ],
        "generation_samples_per_second": require_finite(
            strategy["generation_samples_per_second"],
            "t2i.generation_samples_per_second",
        ),
    }
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
    strategy_generation = generation.get("strategies", {}).get(
        strategy_name,
        {},
    )
    if (
        generation.get("generation_entry") != "model.generate"
        or generation.get("use_cache") is not True
        or strategy_generation.get("backbone_kv_cache_enabled") is not True
    ):
        raise ValueError(
            "held-out T2I generation did not use the unified cached entry"
        )
    image_paths = sorted(
        (validation_root / "validation_flow_images").glob(
            f"step-{step:08d}-*.png"
        )
    )
    if caption_count <= 0 or not image_paths:
        raise ValueError("qualitative validation artifacts are incomplete")

    summary = {
        "schema": "unified_checkpoint_evaluation_summary_v2",
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
            "official_protocol": bool(t2i["official_protocol"]),
            "samples": int(t2i["samples_evaluated"]),
            "strategy": strategy_name,
            "is_split_assignment": IS_SPLIT_ASSIGNMENT,
            "is_split_plan": is_split_plan,
            **t2i_metrics,
        },
        "qualitative_artifacts": {
            "caption_rows": caption_count,
            "caption_jsonl": str(caption_path.resolve()),
            "generation_report": str(generation_path.resolve()),
            "generation_entry": "model.generate",
            "backbone_kv_cache_enabled": True,
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
