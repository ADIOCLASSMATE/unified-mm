#!/usr/bin/env python3
"""Combine image and pure-text metrics into one unified evaluation report."""

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


TEXT_TASK_PROTOCOLS = {
    "arc_easy": (2_376, "accuracy_normalized"),
    "arc_challenge": (1_172, "accuracy_normalized"),
    "hellaswag": (10_042, "accuracy_normalized"),
    "piqa": (1_838, "accuracy_normalized"),
    "winogrande": (1_267, "accuracy"),
    "boolq": (3_270, "accuracy"),
    "openbookqa": (500, "accuracy_normalized"),
    "mmlu": (14_042, "accuracy_macro"),
}
LM_EVAL_REFERENCE_COMMIT = "b954108c9baaaa934b4ad842033b31a97ee30816"
GENERATION_PROTOCOL = "imagenet_val_fid50k_torch_fidelity_stratified_is"
GENERATION_REFERENCE = "imagenet_val_50000"
GENERATION_SCOPE = "same_protocol_only"
GENERATION_NON_COMPARABILITY_REASON = (
    "validation_reference_and_pytorch_torch_fidelity_extractor"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--profile", choices=("smoke", "formal"), required=True)
    return parser.parse_args()


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


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


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


def validate_text_summary(text: dict[str, Any], *, formal: bool) -> None:
    if text.get("schema") != "selfless_text_benchmark_summary_v3":
        raise ValueError("pure-text evaluation uses an obsolete task protocol")
    protocol = text.get("protocol", {})
    if protocol.get("protocol_schema") != "selfless_text_benchmark_v2":
        raise ValueError("pure-text protocol schema is invalid")
    reference = protocol.get("lm_eval_reference", {})
    if reference.get("commit") != LM_EVAL_REFERENCE_COMMIT:
        raise ValueError("pure-text lm-eval reference is not frozen")
    if text.get("accuracy_unit") != "unit_interval":
        raise ValueError("pure-text accuracy unit is ambiguous")
    if set(text.get("tasks", {})) != set(TEXT_TASK_PROTOCOLS):
        raise ValueError("pure-text task coverage is incomplete")
    if set(text.get("primary_metrics", {})) != set(TEXT_TASK_PROTOCOLS):
        raise ValueError("pure-text primary metric coverage is incomplete")
    primary_values: list[float] = []
    for task, (records, metric) in TEXT_TASK_PROTOCOLS.items():
        task_metrics = text["tasks"][task]
        if task_metrics.get("schema") != "selfless_text_multiple_choice_metrics_v1":
            raise ValueError(f"pure-text task schema is invalid: {task}")
        if task_metrics.get("complete") is not True:
            raise ValueError(f"pure-text task is incomplete: {task}")
        if task_metrics.get("runtime_hashing_enabled", True) is not False:
            raise ValueError(f"pure-text task violates the no-hash contract: {task}")
        if task_metrics.get("task") != task:
            raise ValueError(f"pure-text task identity mismatch: {task}")
        if formal and int(task_metrics.get("samples", -1)) != records:
            raise ValueError(f"formal pure-text task is incomplete: {task}")
        if formal and int(task_metrics.get("truncated_context_samples", -1)) != 0:
            raise ValueError(f"formal pure-text task truncated prompts: {task}")
        expected_value = task_metrics.get(metric)
        value = float(expected_value)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"pure-text primary metric is invalid: {task}={value}")
        if float(text["primary_metrics"][task]) != value:
            raise ValueError(f"pure-text primary metric mismatch: {task}")
        primary_values.append(value)
        if task == "mmlu" and formal:
            categories = task_metrics.get("by_category") or {}
            if len(categories) != 57:
                raise ValueError("formal MMLU requires all 57 subjects")
            if sum(int(row.get("samples", -1)) for row in categories.values()) != int(
                task_metrics.get("samples", -2)
            ):
                raise ValueError("MMLU subject sample counts are inconsistent")
            category_values = [float(row["accuracy"]) for row in categories.values()]
            if any(
                not math.isfinite(category_value)
                or not 0.0 <= category_value <= 1.0
                for category_value in category_values
            ):
                raise ValueError("MMLU subject accuracy is invalid")
            expected_macro = sum(category_values) / len(category_values)
            if not math.isclose(
                value, expected_macro, rel_tol=0.0, abs_tol=1.0e-12
            ):
                raise ValueError("MMLU primary is not the 57-subject macro accuracy")
    macro = float(text.get("macro_average_primary", math.nan))
    if not math.isfinite(macro) or not math.isclose(
        macro,
        sum(primary_values) / len(primary_values),
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("pure-text cross-task macro is inconsistent")
    if text.get("macro_average_role") != "internal_cross_task_summary_only":
        raise ValueError("pure-text cross-task macro has an invalid reporting role")


def validate_generation_summary(generation: dict[str, Any], *, formal: bool) -> None:
    if generation.get("leaderboard_comparable_to_adm_dit") is not False:
        raise ValueError("generation comparability label is invalid")
    expected = {
        "protocol_name": GENERATION_PROTOCOL,
        "reference_distribution": GENERATION_REFERENCE,
        "comparison_scope": GENERATION_SCOPE,
        "not_adm_dit_reason": GENERATION_NON_COMPARABILITY_REASON,
        "fid_reducer": "symmetric_eigendecomposition",
        "is_split_assignment": "stratified_by_synset",
    }
    mismatches = {
        key: {"expected": value, "actual": generation.get(key)}
        for key, value in expected.items()
        if generation.get(key) != value
    }
    if mismatches:
        raise ValueError(f"generation summary protocol is invalid: {mismatches}")
    if not str(generation.get("strategy", "")).strip():
        raise ValueError("generation summary strategy is missing")
    samples = int(generation.get("samples", -1))
    if samples <= 0:
        raise ValueError("generation summary has no samples")
    if formal and (
        generation.get("project_formal_protocol") is not True
        or samples != 50_000
    ):
        raise ValueError("formal generation summary is not a complete FID50K run")
    fid_raw = generation.get("fid")
    fid = None if fid_raw is None else float(fid_raw)
    is_mean = float(generation.get("inception_score_mean", math.nan))
    is_std = float(generation.get("inception_score_std", math.nan))
    is_values = [
        float(value) for value in generation.get("inception_score_splits", [])
    ]
    if formal and fid is None:
        raise ValueError("formal generation FID is missing")
    if fid is not None and (not math.isfinite(fid) or fid < 0.0):
        raise ValueError("generation FID is invalid")
    if not math.isfinite(is_mean) or is_mean < 1.0 - 1.0e-9:
        raise ValueError("generation Inception Score mean is invalid")
    if not math.isfinite(is_std) or is_std < 0.0:
        raise ValueError("generation Inception Score std is invalid")
    split_plan = generation.get("is_split_plan") or {}
    expected_splits = int(split_plan.get("splits", -1))
    if expected_splits <= 0 or len(is_values) != expected_splits:
        raise ValueError("generation Inception Score splits are incomplete")
    if any(not math.isfinite(value) or value < 1.0 - 1.0e-9 for value in is_values):
        raise ValueError("generation Inception Score split value is invalid")
    split_mean = sum(is_values) / len(is_values)
    split_std = math.sqrt(
        sum((value - split_mean) ** 2 for value in is_values) / len(is_values)
    )
    if not math.isclose(is_mean, split_mean, rel_tol=0.0, abs_tol=1.0e-10):
        raise ValueError("generation Inception Score mean is inconsistent")
    if not math.isclose(is_std, split_std, rel_tol=0.0, abs_tol=1.0e-10):
        raise ValueError("generation Inception Score std is inconsistent")


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    source = resolve_evaluation_model_source(checkpoint)
    image = read_json(args.output_root / "evaluation_summary.json")
    text = read_json(args.output_root / "text" / "summary.json")
    if image.get("schema") != "unified_checkpoint_evaluation_summary_v3":
        raise ValueError("image evaluation uses an obsolete generation protocol")
    if not bool(image.get("complete", False)) or not bool(text.get("complete", False)):
        raise ValueError("image or text evaluation is incomplete")
    validate_generation_summary(
        image.get("t2i_fid_is") or {}, formal=args.profile == "formal"
    )
    validate_text_summary(text, formal=args.profile == "formal")
    if Path(image["checkpoint"]).resolve() != checkpoint:
        raise ValueError("image evaluation checkpoint differs from requested checkpoint")
    if Path(text["checkpoint"]).resolve() != checkpoint:
        raise ValueError("text evaluation checkpoint differs from requested checkpoint")
    for name, payload in (("image", image), ("text", text)):
        weight_source = payload.get("weight_source")
        if source.is_hf_final_ema and weight_source != source.kind:
            raise ValueError(f"{name} evaluation lacks final-HF source identity")
        if weight_source is not None and weight_source != source.kind:
            raise ValueError(f"{name} evaluation weight source differs")
    if int(image["global_step"]) != int(text["checkpoint_step"]):
        raise ValueError("image and text evaluation checkpoint steps differ")
    dataset_contract = image.get("dataset_contract", {})
    if dataset_contract.get("training_split") != "imagenet_train":
        raise ValueError("image training split is not ImageNet train")
    if dataset_contract.get("evaluation_split") != "imagenet_val":
        raise ValueError("image evaluation split is not ImageNet val")
    report = {
        "schema": "unified_full_checkpoint_evaluation_summary_v3",
        "complete": True,
        "profile": args.profile,
        "runtime_hashing_enabled": False,
        "checkpoint": str(checkpoint),
        "global_step": int(image["global_step"]),
        "weight_source": source.kind,
        "model_source": source.report(),
        "dataset_contract": image["dataset_contract"],
        "generation": {
            "imagenet_val_t2i": image["t2i_fid_is"],
            "qualitative_images": image["qualitative_artifacts"]["flow_images"],
        },
        "understanding": {
            "heldout_validation": image["heldout_validation"],
            "qualitative_captions": image["qualitative_artifacts"][
                "caption_jsonl"
            ],
        },
        "pure_text": {
            "macro_average_primary": text["macro_average_primary"],
            "macro_average_role": text["macro_average_role"],
            "primary_metrics": text["primary_metrics"],
            "tasks": text["tasks"],
            "protocol": text["protocol"],
        },
        "components": {
            "image_summary": str(
                (args.output_root / "evaluation_summary.json").resolve()
            ),
            "text_summary": str(
                (args.output_root / "text" / "summary.json").resolve()
            ),
        },
        "completed_at": datetime.now(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        ),
    }
    forbidden = forbidden_audit_fields(report)
    if forbidden:
        raise ValueError(f"no-hash report contains forbidden audit fields: {forbidden}")
    destination = args.output_root / "full_evaluation_summary.json"
    atomic_write_text(
        destination,
        json.dumps(report, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
