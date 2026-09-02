#!/usr/bin/env python3
"""Select a complete 1B LR sweep using readable, unhashed evidence."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing required sweep artifact: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _resolve_repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _require_finite_positive(value: Any, label: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{label} must be finite and positive, got {value!r}")
    return number


def _validate_rank_files(checkpoint: Path, world_size: int) -> None:
    expected = set(range(world_size))
    patterns = {
        "data state": ("data_state_rank_*.pt", "data_state_rank_", ".pt"),
        "RNG state": ("random_states_*.pkl", "random_states_", ".pkl"),
        "EMA shard": (
            "ema_shard_rank_*.safetensors",
            "ema_shard_rank_",
            ".safetensors",
        ),
    }
    for label, (pattern, prefix, suffix) in patterns.items():
        observed = {
            int(path.name[len(prefix) : -len(suffix)])
            for path in checkpoint.glob(pattern)
        }
        if observed != expected:
            missing = sorted(expected - observed)
            extra = sorted(observed - expected)
            raise ValueError(
                f"{checkpoint}: incomplete {label} files; "
                f"missing={missing}, extra={extra}"
            )


def _comparable_contract(contract: dict[str, Any]) -> dict[str, Any]:
    """Canonicalize identities plus swept and directly derived LR values."""

    comparable = copy.deepcopy(contract)
    experiment = comparable.setdefault("experiment", {})
    for key in ("project", "name", "output_dir"):
        experiment[key] = "<sweep-arm-identity>"
    model = comparable.setdefault("model", {})
    model.setdefault("training_objective", "selfless_dual_stream")
    model.setdefault(
        "dual_stream_attention_contract", "xlnet_content_diagonal"
    )
    model.setdefault("showo_mask_schedule", "cosine")
    model.setdefault("showo_min_masking_rate", 0.0)
    optimizer = comparable.setdefault("optimizer", {}).setdefault(
        "params", {}
    )
    for key in (
        "learning_rate",
        "backbone_learning_rate",
        "special_token_learning_rate",
        "flow_learning_rate",
        "projector_learning_rate",
    ):
        optimizer[key] = "<swept-learning-rate>"
    scheduler = comparable.setdefault("lr_scheduler", {}).setdefault(
        "params", {}
    )
    scheduler["learning_rate"] = "<derived-backbone-learning-rate>"
    return comparable


def _read_arm(
    arm: dict[str, Any],
    *,
    output_root: Path,
    stop_step: int,
    world_size: int,
    training_seed: int,
    validation_seed: int,
) -> dict[str, Any]:
    arm_id = str(arm["id"])
    run_root = output_root / arm_id
    checkpoint = run_root / f"checkpoint-{stop_step}"

    complete = _read_json(checkpoint / "checkpoint_complete.json")
    if complete != {
        "schema": "selfless_caption_checkpoint_complete_v1",
        "global_step": stop_step,
    }:
        raise ValueError(f"invalid checkpoint completion marker: {checkpoint}")

    metadata = _read_json(checkpoint / "metadata.json")
    required_metadata = {
        "schema": "selfless_caption_training_checkpoint_v3",
        "global_step": stop_step,
        "world_size": world_size,
        "gradient_accumulation_steps": 4,
        "config_contract_version": 1,
        "mixed_data_state_schema": "unified_mixed_data_state_v1",
    }
    for key, expected in required_metadata.items():
        if metadata.get(key) != expected:
            raise ValueError(
                f"{checkpoint}: metadata {key}={metadata.get(key)!r}, "
                f"expected {expected!r}"
            )
    contract = metadata.get("config_contract")
    if not isinstance(contract, dict):
        raise ValueError(f"{checkpoint}: missing readable resume contract")
    training = contract.get("training") or {}
    model = contract.get("model") or {}
    optimizer = (contract.get("optimizer") or {}).get("params") or {}
    if training.get("runtime_hashing_enabled") is not False:
        raise ValueError(f"{checkpoint}: runtime hashing was not disabled")
    if str(model.get("training_objective", "selfless_dual_stream")) != (
        "selfless_dual_stream"
    ):
        raise ValueError(f"{checkpoint}: sweep arm is not baseline b")
    if str(
        model.get(
            "dual_stream_attention_contract", "xlnet_content_diagonal"
        )
    ) != (
        "xlnet_content_diagonal"
    ):
        raise ValueError(f"{checkpoint}: sweep arm is not baseline b")
    expected_backbone_lr = _require_finite_positive(
        arm["backbone_lr"], f"{arm_id}.backbone_lr"
    )
    expected_flow_lr = _require_finite_positive(
        arm["flow_lr"], f"{arm_id}.flow_lr"
    )
    for key in (
        "learning_rate",
        "backbone_learning_rate",
        "special_token_learning_rate",
    ):
        if float(optimizer.get(key, -1.0)) != expected_backbone_lr:
            raise ValueError(f"{checkpoint}: optimizer {key} does not match arm")
    for key in ("flow_learning_rate", "projector_learning_rate"):
        if float(optimizer.get(key, -1.0)) != expected_flow_lr:
            raise ValueError(f"{checkpoint}: optimizer {key} does not match arm")

    _validate_rank_files(checkpoint, world_size)

    metrics_payload = _read_json(
        run_root / f"validation_metrics_step_{stop_step}.json"
    )
    if metrics_payload.get("schema") != "selfless_flow_validation_metrics_v1":
        raise ValueError(f"{run_root}: invalid validation metrics schema")
    if int(metrics_payload.get("global_step", -1)) != stop_step:
        raise ValueError(f"{run_root}: validation step mismatch")
    if int(metrics_payload.get("training_seed", -1)) != training_seed:
        raise ValueError(f"{run_root}: training seed mismatch")
    if int(metrics_payload.get("validation_seed", -1)) != validation_seed:
        raise ValueError(f"{run_root}: validation seed mismatch")
    metrics = metrics_payload.get("metrics") or {}
    text_loss = _require_finite_positive(
        metrics.get("val/loss_text"), f"{arm_id}.val/loss_text"
    )
    image_loss = _require_finite_positive(
        metrics.get("val/loss_image_flow"),
        f"{arm_id}.val/loss_image_flow",
    )
    _require_finite_positive(
        metrics.get("val/text_target_tokens"),
        f"{arm_id}.val/text_target_tokens",
    )
    _require_finite_positive(
        metrics.get("val/image_target_tokens"),
        f"{arm_id}.val/image_target_tokens",
    )

    runtime = _read_json(run_root / "training_runtime_metrics.json")
    if runtime.get("schema") != "selfless_training_runtime_metrics_v1":
        raise ValueError(f"{run_root}: invalid runtime metrics schema")
    if int(runtime.get("global_step", -1)) != stop_step:
        raise ValueError(f"{run_root}: runtime global step mismatch")
    if int(runtime.get("world_size", -1)) != world_size:
        raise ValueError(f"{run_root}: runtime world size mismatch")
    if int(runtime.get("finite_loss_microbatches_checked", 0)) <= 0:
        raise ValueError(f"{run_root}: no finite-loss checks recorded")

    return {
        "arm_id": arm_id,
        "job_name": str(arm["job_name"]),
        "backbone_lr": expected_backbone_lr,
        "flow_lr": expected_flow_lr,
        "checkpoint": _display_path(checkpoint),
        "validation": {
            "loss_text": text_loss,
            "loss_image_flow": image_loss,
        },
        "training_wall_seconds": _require_finite_positive(
            runtime.get("cumulative_training_wall_seconds"),
            f"{arm_id}.cumulative_training_wall_seconds",
        ),
        "_comparison_contract": _comparable_contract(contract),
    }


def _ordinal_ranks(rows: list[dict[str, Any]], metric: str) -> dict[str, int]:
    ordered = sorted(
        rows,
        key=lambda row: (float(row["validation"][metric]), row["arm_id"]),
    )
    return {row["arm_id"]: index + 1 for index, row in enumerate(ordered)}


def select(manifest_path: Path) -> dict[str, Any]:
    manifest = OmegaConf.to_container(
        OmegaConf.load(manifest_path), resolve=True
    )
    if not isinstance(manifest, dict) or manifest.get("schema") != "unified_lr_sweep_v1":
        raise ValueError("unsupported LR sweep manifest")
    selection = manifest.get("selection") or {}
    if selection.get("runtime_hashing_enabled") is not False:
        raise ValueError("sweep selection must disable runtime hashing")
    arms = manifest.get("arms") or []
    required_arms = int(selection.get("required_arms", -1))
    if len(arms) != required_arms or required_arms != 9:
        raise ValueError(f"expected exactly 9 sweep arms, got {len(arms)}")
    arm_ids = [str(arm["id"]) for arm in arms]
    if len(set(arm_ids)) != len(arm_ids):
        raise ValueError("sweep arm IDs must be unique")

    output_root = _resolve_repo_path(str(manifest["output_root"]))
    stop_step = int(manifest["stop_after_steps"])
    world_size = int(manifest["world_size"])
    training_seed = int(manifest["seed"])
    validation_seed = int(manifest["validation_seed"])
    rows = [
        _read_arm(
            arm,
            output_root=output_root,
            stop_step=stop_step,
            world_size=world_size,
            training_seed=training_seed,
            validation_seed=validation_seed,
        )
        for arm in arms
    ]
    reference_contract = rows[0].pop("_comparison_contract")
    for row in rows[1:]:
        comparable = row.pop("_comparison_contract")
        if comparable != reference_contract:
            raise ValueError(
                "sweep arms differ outside the selected or directly derived "
                "learning-rate fields: "
                f"{rows[0]['arm_id']} vs {row['arm_id']}"
            )

    text_ranks = _ordinal_ranks(rows, "loss_text")
    image_ranks = _ordinal_ranks(rows, "loss_image_flow")
    min_text = min(row["validation"]["loss_text"] for row in rows)
    min_image = min(row["validation"]["loss_image_flow"] for row in rows)
    for row in rows:
        arm_id = row["arm_id"]
        text_rank = text_ranks[arm_id]
        image_rank = image_ranks[arm_id]
        row["selection"] = {
            "text_rank": text_rank,
            "image_rank": image_rank,
            "mean_modality_rank": (text_rank + image_rank) / 2.0,
            "worst_modality_rank": max(text_rank, image_rank),
            "relative_loss_sum": (
                row["validation"]["loss_text"] / min_text
                + row["validation"]["loss_image_flow"] / min_image
            ),
        }
    rows.sort(
        key=lambda row: (
            row["selection"]["mean_modality_rank"],
            row["selection"]["worst_modality_rank"],
            row["selection"]["relative_loss_sum"],
            row["arm_id"],
        )
    )
    winner = rows[0]
    return {
        "schema": "unified_lr_selection_v1",
        "manifest": _display_path(manifest_path),
        "runtime_hashing_enabled": False,
        "all_arms_complete": True,
        "selection_rule": {
            "primary": "mean_modality_rank",
            "tie_breakers": [
                "worst_modality_rank",
                "relative_loss_sum",
                "arm_id",
            ],
            "lower_is_better": True,
        },
        "winner": winner,
        "ranking": rows,
        "formal_continuation": {
            "resume_from_checkpoint": winner["checkpoint"],
            "max_train_steps": int(
                (manifest.get("formal_continuation") or {})["max_train_steps"]
            ),
            "restart_from_step_zero": False,
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="configs/protocols/unified_baseline_lr_sweep_1b_ascend64.yaml",
    )
    parser.add_argument("--output")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest_path = _resolve_repo_path(args.manifest)
    report = select(manifest_path)
    output_path = (
        _resolve_repo_path(args.output)
        if args.output
        else _resolve_repo_path(
            str(OmegaConf.load(manifest_path).output_root)
        )
        / "lr_selection.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output_path)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
