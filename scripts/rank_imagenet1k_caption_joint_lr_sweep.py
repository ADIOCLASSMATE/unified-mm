#!/usr/bin/env python3
"""Rank completed ImageNet-1K caption/T2I LR-sweep candidates."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


DEFAULT_SWEEP = Path(
    "configs/selfless/imagenet1k_caption_joint_lr_sweep_10ep.json"
)
EXPECTED_SCHEMA = "selfless_flow_validation_metrics_v1"


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _finite_metric(payload: dict[str, Any], key: str, path: Path) -> float:
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict) or key not in metrics:
        raise ValueError(f"missing metric {key!r}: {path}")
    value = float(metrics[key])
    if not math.isfinite(value):
        raise ValueError(f"non-finite metric {key!r}={value}: {path}")
    return value


def _average_tie_ranks(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    ordered = sorted(rows, key=lambda row: (float(row[key]), str(row["id"])))
    result: dict[str, float] = {}
    cursor = 0
    while cursor < len(ordered):
        end = cursor + 1
        value = float(ordered[cursor][key])
        while end < len(ordered) and math.isclose(
            float(ordered[end][key]), value, rel_tol=0.0, abs_tol=1.0e-12
        ):
            end += 1
        average_rank = ((cursor + 1) + end) / 2.0
        for row in ordered[cursor:end]:
            result[str(row["id"])] = average_rank
        cursor = end
    return result


def collect(
    sweep_path: Path,
    output_root: Path,
    *,
    require_complete: bool,
) -> dict[str, Any]:
    sweep = _read_json(sweep_path)
    candidates = sweep.get("candidates")
    selection = sweep.get("selection")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError(f"sweep has no candidates: {sweep_path}")
    if not isinstance(selection, dict):
        raise ValueError(f"sweep has no selection contract: {sweep_path}")
    validation_steps = [int(step) for step in selection["validation_steps"]]
    final_step = validation_steps[-1]

    rows: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for candidate in candidates:
        run_id = str(candidate["id"])
        run_root = output_root / (
            f"selfless-flow-imagenet1k-caption-joint-sweep-{run_id}"
        )
        checkpoints: list[dict[str, float | int]] = []
        missing_steps: list[int] = []
        for step in validation_steps:
            path = run_root / f"validation_metrics_step_{step}.json"
            if not path.is_file():
                missing_steps.append(step)
                continue
            payload = _read_json(path)
            if payload.get("schema") != EXPECTED_SCHEMA:
                raise ValueError(
                    f"validation schema mismatch at {path}: "
                    f"{payload.get('schema')!r}"
                )
            if int(payload.get("global_step", -1)) != step:
                raise ValueError(f"validation step mismatch: {path}")
            checkpoints.append(
                {
                    "step": step,
                    "text_loss": _finite_metric(payload, "val/loss_text", path),
                    "image_loss": _finite_metric(
                        payload, "val/loss_image_flow", path
                    ),
                }
            )
        if missing_steps:
            missing.append(
                {
                    "id": run_id,
                    "run_root": str(run_root),
                    "missing_validation_steps": missing_steps,
                }
            )
            continue
        final = next(item for item in checkpoints if item["step"] == final_step)
        best_text = min(float(item["text_loss"]) for item in checkpoints)
        best_image = min(float(item["image_loss"]) for item in checkpoints)
        final_text = float(final["text_loss"])
        final_image = float(final["image_loss"])
        rows.append(
            {
                "id": run_id,
                "backbone_lr": float(candidate["backbone_lr"]),
                "flow_lr": float(candidate["flow_lr"]),
                "final_step": final_step,
                "final_text_loss": final_text,
                "final_image_loss": final_image,
                "best_text_loss": best_text,
                "best_image_loss": best_image,
                "mean_normalized_final_regression": 0.5
                * (
                    (final_text - best_text) / max(abs(best_text), 1.0e-12)
                    + (final_image - best_image) / max(abs(best_image), 1.0e-12)
                ),
                "checkpoints": checkpoints,
            }
        )

    if require_complete and missing:
        details = ", ".join(
            f"{item['id']}:{item['missing_validation_steps']}" for item in missing
        )
        raise FileNotFoundError(f"sweep is incomplete: {details}")
    if not rows:
        return {
            "schema": "selfless_imagenet1k_caption_joint_lr_ranking_v1",
            "status": "incomplete",
            "sweep": str(sweep_path),
            "completed_candidates": 0,
            "expected_candidates": len(candidates),
            "missing": missing,
            "ranking": [],
            "top_k": [],
            "winner": None,
        }

    text_ranks = _average_tie_ranks(rows, "final_text_loss")
    image_ranks = _average_tie_ranks(rows, "final_image_loss")
    for row in rows:
        row["text_rank"] = text_ranks[str(row["id"])]
        row["image_rank"] = image_ranks[str(row["id"])]
        row["mean_rank"] = 0.5 * (
            float(row["text_rank"]) + float(row["image_rank"])
        )
    rows.sort(
        key=lambda row: (
            float(row["mean_rank"]),
            float(row["mean_normalized_final_regression"]),
            str(row["id"]),
        )
    )
    for rank, row in enumerate(rows, start=1):
        row["overall_rank"] = rank
    top_k_count = int(selection.get("top_k_for_generation_evaluation", 3))
    status = "complete" if not missing else "incomplete"
    return {
        "schema": "selfless_imagenet1k_caption_joint_lr_ranking_v1",
        "status": status,
        "sweep": str(sweep_path),
        "completed_candidates": len(rows),
        "expected_candidates": len(candidates),
        "final_step": final_step,
        "selection_rule": selection["rule"],
        "missing": missing,
        "ranking": rows,
        "top_k": [row["id"] for row in rows[:top_k_count]],
        "winner": rows[0]["id"] if status == "complete" else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", type=Path, default=DEFAULT_SWEEP)
    parser.add_argument("--output_root", type=Path, default=Path("output"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--require_complete", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = collect(
        args.sweep,
        args.output_root,
        require_complete=bool(args.require_complete),
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
