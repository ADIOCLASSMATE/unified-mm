#!/usr/bin/env python3
"""Validate and rank the formal ImageNet-100 class LR sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return payload


def _rank(rows: list[dict[str, Any]], key: str, *, reverse: bool = False) -> None:
    complete = [row for row in rows if row["complete"] and row[key] is not None]
    for rank, row in enumerate(
        sorted(complete, key=lambda item: float(item[key]), reverse=reverse),
        start=1,
    ):
        row[f"{key}_rank"] = rank


def _validate_training(
    candidate: dict[str, Any], training_contract: dict[str, Any]
) -> list[str]:
    run_dir = Path(candidate["run_dir"])
    errors: list[str] = []
    runtime_path = run_dir / "training_runtime_metrics.json"
    config_path = run_dir / "config.yaml"
    checkpoint = run_dir / "hf_model-final-ema"
    if not runtime_path.is_file():
        errors.append(f"missing {runtime_path}")
    else:
        runtime = _load_json(runtime_path)
        expected_runtime = {
            "global_step": int(training_contract["max_train_steps"]),
            "world_size": int(training_contract["world_size"]),
            "total_batch_size": int(training_contract["global_batch_size"]),
        }
        for field, expected in expected_runtime.items():
            actual = int(runtime.get(field, -1))
            if actual != expected:
                errors.append(f"training {field}={actual}, expected {expected}")
    if not config_path.is_file():
        errors.append(f"missing {config_path}")
    else:
        config = OmegaConf.load(config_path)
        checks = {
            "conditioning_mode": (
                str(config.dataset.params.conditioning_mode),
                str(training_contract["conditioning_mode"]),
            ),
            "batch_size_per_gpu": (
                int(config.training.batch_size),
                int(training_contract["batch_size_per_gpu"]),
            ),
            "global_batch_size": (
                int(config.training.total_batch_size),
                int(training_contract["global_batch_size"]),
            ),
            "gradient_checkpointing": (
                bool(config.training.use_gradient_checkpointing),
                bool(training_contract["gradient_checkpointing"]),
            ),
            "gradient_accumulation_steps": (
                int(config.training.total_batch_size)
                // (
                    int(config.training.batch_size)
                    * int(training_contract["world_size"])
                ),
                int(training_contract["gradient_accumulation_steps"]),
            ),
            "backbone_lr": (
                float(config.optimizer.params.backbone_learning_rate),
                float(candidate["backbone_lr"]),
            ),
            "flow_head_lr": (
                float(config.optimizer.params.flow_learning_rate),
                float(candidate["flow_head_lr"]),
            ),
            "projector_lr": (
                float(config.optimizer.params.projector_learning_rate),
                float(candidate["flow_head_lr"]),
            ),
            "special_token_lr": (
                float(config.optimizer.params.special_token_learning_rate),
                float(candidate["backbone_lr"]),
            ),
        }
        for field, (actual, expected) in checks.items():
            if actual != expected:
                errors.append(f"training {field}={actual!r}, expected {expected!r}")
    for filename in ("config.json", "model.safetensors", "tokenizer.json"):
        if not (checkpoint / filename).is_file():
            errors.append(f"missing {checkpoint / filename}")
    return errors


def _validate_evaluation(
    candidate: dict[str, Any], evaluation_contract: dict[str, Any]
) -> tuple[list[str], dict[str, float | None]]:
    metrics_path = Path(candidate["eval_dir"]) / "metrics.json"
    values: dict[str, float | None] = {
        "fid": None,
        "inception_score_mean": None,
        "inception_score_std": None,
    }
    if not metrics_path.is_file():
        return [f"missing {metrics_path}"], values
    payload = _load_json(metrics_path)
    errors: list[str] = []
    strategy_name = str(evaluation_contract["strategy"])
    distributed = payload.get("distributed", {})
    checks = {
        "official_protocol": (bool(payload.get("official_protocol")), True),
        "samples_evaluated": (
            int(payload.get("samples_evaluated", -1)),
            int(evaluation_contract["samples"]),
        ),
        "world_size": (
            int(distributed.get("world_size", -1)),
            int(evaluation_contract["world_size"]),
        ),
        "batch_size_per_gpu": (
            int(distributed.get("batch_size_per_rank", -1)),
            int(evaluation_contract["batch_size_per_gpu"]),
        ),
        "sampling_steps": (
            int(payload.get("sampling_steps", -1)),
            int(evaluation_contract["sampling_steps"]),
        ),
        "cfg": (float(payload.get("cfg", -1.0)), float(evaluation_contract["cfg"])),
        "flow_solver": (
            str(payload.get("flow_solver")),
            str(evaluation_contract["flow_solver"]),
        ),
    }
    for field, (actual, expected) in checks.items():
        if actual != expected:
            errors.append(f"evaluation {field}={actual!r}, expected {expected!r}")
    strategy = payload.get("strategies", {}).get(strategy_name)
    if not isinstance(strategy, dict):
        errors.append(f"missing evaluation strategy {strategy_name!r}")
    else:
        count = int(strategy.get("count", -1))
        if count != int(evaluation_contract["samples"]):
            errors.append(
                f"strategy count={count}, expected {evaluation_contract['samples']}"
            )
        for field in values:
            raw = strategy.get(field)
            values[field] = float(raw) if raw is not None else None
            if values[field] is None:
                errors.append(f"missing strategy metric {field}")
    return errors, values


def summarize(manifest_path: Path) -> dict[str, Any]:
    manifest = _load_json(manifest_path)
    rows: list[dict[str, Any]] = []
    for candidate in manifest["candidates"]:
        training_errors = _validate_training(candidate, manifest["training"])
        evaluation_errors, metrics = _validate_evaluation(
            candidate, manifest["evaluation"]
        )
        errors = training_errors + evaluation_errors
        rows.append(
            {
                "id": str(candidate["id"]),
                "backbone_lr": float(candidate["backbone_lr"]),
                "flow_head_lr": float(candidate["flow_head_lr"]),
                **metrics,
                "complete": not errors,
                "validation_errors": errors,
                "run_dir": str(candidate["run_dir"]),
                "eval_dir": str(candidate["eval_dir"]),
            }
        )

    _rank(rows, "fid")
    _rank(rows, "inception_score_mean", reverse=True)
    for row in rows:
        ranks = (row.get("fid_rank"), row.get("inception_score_mean_rank"))
        row["mean_rank"] = (
            (float(ranks[0]) + float(ranks[1])) / 2.0
            if row["complete"] and all(rank is not None for rank in ranks)
            else None
        )
    complete = [row for row in rows if row["mean_rank"] is not None]
    winner = (
        min(complete, key=lambda row: (float(row["mean_rank"]), float(row["fid"])))
        if complete
        else None
    )
    return {
        "schema": "selfless_flow_imagenet100_class_lr_sweep_summary_v1",
        "manifest": str(manifest_path),
        "selection_rule": manifest["selection"]["rule"],
        "complete_candidates": len(complete),
        "total_candidates": len(rows),
        "winner": winner,
        "candidates": sorted(
            rows,
            key=lambda row: (
                row["mean_rank"] is None,
                float(row["mean_rank"]) if row["mean_rank"] is not None else 0.0,
                row["id"],
            ),
        ),
    }


def _format(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.7g}"
    return str(value)


def _print_table(summary: dict[str, Any]) -> None:
    print("id\tbackbone_lr\tflow_head_lr\tFID\tIS\tIS_std\tmean_rank\tcomplete")
    for row in summary["candidates"]:
        print(
            "\t".join(
                _format(value)
                for value in (
                    row["id"],
                    row["backbone_lr"],
                    row["flow_head_lr"],
                    row["fid"],
                    row["inception_score_mean"],
                    row["inception_score_std"],
                    row["mean_rank"],
                    row["complete"],
                )
            )
        )
    if summary["winner"] is not None:
        winner = summary["winner"]
        print(
            "Winner: "
            f"{winner['id']} (backbone_lr={winner['backbone_lr']:g}, "
            f"flow_head_lr={winner['flow_head_lr']:g}, FID={winner['fid']:.6f}, "
            f"IS={winner['inception_score_mean']:.6f})"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("configs/selfless/imagenet100_class_lr_sweep_80ep.json"),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    summary = summarize(args.manifest)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    _print_table(summary)
    if args.require_complete and summary["complete_candidates"] != summary["total_candidates"]:
        raise SystemExit(
            f"incomplete sweep: {summary['complete_candidates']}/{summary['total_candidates']} "
            "candidates passed training and formal-evaluation validation"
        )


if __name__ == "__main__":
    main()
