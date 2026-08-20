#!/usr/bin/env python3
"""Select candidates at a staged LR-sweep boundary with gradient evidence."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from scripts.rank_imagenet1k_caption_joint_lr_sweep import _average_tie_ranks
from utils.joint_sweep_health import training_health


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _finite(value: Any, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite {label}: {number}")
    return number


def collect_stage(
    sweep_path: Path,
    output_root: Path,
    *,
    step: int,
    lambda_text: float,
    require_complete: bool,
    candidate_selection_path: Path | None = None,
) -> dict[str, Any]:
    sweep = _read_json(sweep_path)
    stages = sweep["staged_training"]
    if int(step) not in map(int, stages["stage_boundaries"]):
        raise ValueError(f"unsupported stage step {step}")
    top_k = int(stages["top_k_to_continue"][str(step)])
    candidates = sweep["candidates"]
    upstream_selection: dict[str, Any] | None = None
    if candidate_selection_path is not None:
        upstream_selection = _read_json(candidate_selection_path)
        if (
            upstream_selection.get("schema")
            != "selfless_caption_t2i_stage_selection_v1"
        ):
            raise ValueError(
                f"candidate selection schema mismatch: {candidate_selection_path}"
            )
        if upstream_selection.get("status") != "complete":
            raise ValueError("candidate selection must be complete")
        upstream_step = int(upstream_selection.get("stage_step", -1))
        if upstream_step >= int(step):
            raise ValueError(
                f"candidate selection step {upstream_step} must precede {step}"
            )
        selected_ids = [
            str(value) for value in upstream_selection.get("selected", [])
        ]
        if not selected_ids:
            raise ValueError("candidate selection has no selected IDs")
        by_id = {str(candidate["id"]): candidate for candidate in candidates}
        unknown = sorted(set(selected_ids).difference(by_id))
        if unknown:
            raise ValueError(
                f"candidate selection contains IDs absent from sweep: {unknown}"
            )
        candidates = [by_id[run_id] for run_id in selected_ids]
    rows: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for candidate in candidates:
        run_id = str(candidate["id"])
        run_root = output_root / f"selfless-flow-imagenet1k-caption-joint-sweep-{run_id}"
        metrics_path = run_root / f"validation_metrics_step_{step}.json"
        probe_path = run_root / "gradient_probe" / f"checkpoint-{step}" / "probe.json"
        missing_paths = [
            str(path) for path in (metrics_path, probe_path) if not path.is_file()
        ]
        if missing_paths:
            missing.append({"id": run_id, "missing": missing_paths})
            continue
        metrics = _read_json(metrics_path)
        probe = _read_json(probe_path)
        if metrics.get("schema") != "selfless_flow_validation_metrics_v1":
            raise ValueError(f"validation schema mismatch: {metrics_path}")
        if probe.get("schema") != "selfless_caption_t2i_gradient_probe_v1":
            raise ValueError(f"probe schema mismatch: {probe_path}")
        if probe.get("status") != "complete":
            missing.append({"id": run_id, "missing": [f"incomplete:{probe_path}"]})
            continue
        health = training_health(run_root)
        summary = probe["summary"]
        ratio = _finite(
            summary["ratio_g_image_over_g_text"]["median"], "gradient ratio"
        )
        cosine = _finite(summary["cosine"]["median"], "gradient cosine")
        task_conflict = summary["task_conflict"]
        persistent_negative = bool(task_conflict.get("persistent_negative", False))
        negative_fraction = _finite(
            task_conflict.get(
                "negative_fraction",
                1.0 if persistent_negative else 0.0,
            ),
            "negative cosine fraction",
        )
        if not 0.0 <= negative_fraction <= 1.0:
            raise ValueError(
                f"negative cosine fraction must be in [0, 1], got {negative_fraction}"
            )
        row = {
            "id": run_id,
            "backbone_lr": float(candidate["backbone_lr"]),
            "flow_lr": float(candidate["flow_lr"]),
            "step": int(step),
            "text_loss": _finite(metrics["metrics"]["val/loss_text"], "text loss"),
            "image_loss": _finite(
                metrics["metrics"]["val/loss_image_flow"], "image loss"
            ),
            "gradient_ratio": ratio,
            "gradient_cosine": cosine,
            "gradient_balance_log_error": abs(math.log(float(lambda_text) / ratio)),
            "gradient_conflict_score": negative_fraction + max(0.0, -cosine),
            "task_conflict": task_conflict,
            "health": health,
            "metrics_path": str(metrics_path),
            "probe_path": str(probe_path),
        }
        if not health["passed"]:
            excluded.append({"id": run_id, "reason": "training instability", "row": row})
            continue
        rows.append(row)
    if require_complete and missing:
        raise FileNotFoundError(f"stage {step} incomplete: {missing}")
    if rows:
        text_ranks = _average_tie_ranks(rows, "text_loss")
        image_ranks = _average_tie_ranks(rows, "image_loss")
        balance_ranks = _average_tie_ranks(rows, "gradient_balance_log_error")
        conflict_ranks = _average_tie_ranks(rows, "gradient_conflict_score")
        for row in rows:
            row["text_rank"] = text_ranks[row["id"]]
            row["image_rank"] = image_ranks[row["id"]]
            row["gradient_balance_rank"] = balance_ranks[row["id"]]
            row["gradient_conflict_rank"] = conflict_ranks[row["id"]]
            row["mean_rank"] = (
                row["text_rank"]
                + row["image_rank"]
                + row["gradient_balance_rank"]
                + row["gradient_conflict_rank"]
            ) / 4.0
        rows.sort(
            key=lambda row: (
                row["mean_rank"],
                row["text_rank"] + row["image_rank"],
                row["id"],
            )
        )
        for rank, row in enumerate(rows, start=1):
            row["overall_rank"] = rank
    return {
        "schema": "selfless_caption_t2i_stage_selection_v1",
        "status": "complete" if not missing else "incomplete",
        "stage_step": int(step),
        "lambda_text": float(lambda_text),
        "candidate_selection": (
            {
                "path": str(candidate_selection_path),
                "stage_step": int(upstream_selection["stage_step"]),
                "selected": [
                    str(value) for value in upstream_selection["selected"]
                ],
            }
            if upstream_selection is not None
            and candidate_selection_path is not None
            else None
        ),
        "selection_rule": (
            "exclude instability, then mean-rank text loss, image-flow loss, "
            "|log(lambda_text / median(g_image/g_text))|, and shared-backbone "
            "negative-cosine conflict"
        ),
        "expected_candidates": len(candidates),
        "eligible_candidates": len(rows),
        "missing": missing,
        "excluded": excluded,
        "ranking": rows,
        "continue_top_k": top_k,
        "selected": [row["id"] for row in rows[:top_k]],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sweep",
        type=Path,
        default=Path("configs/selfless/imagenet1k_caption_joint_lr_sweep_10ep.json"),
    )
    parser.add_argument("--output_root", type=Path, default=Path("output"))
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--lambda_text", type=float, default=0.2)
    parser.add_argument(
        "--candidate_selection",
        type=Path,
        default=None,
        help="Completed prior-stage selection; only its selected IDs are ranked.",
    )
    parser.add_argument("--require_complete", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = collect_stage(
        args.sweep,
        args.output_root,
        step=args.step,
        lambda_text=args.lambda_text,
        require_complete=args.require_complete,
        candidate_selection_path=args.candidate_selection,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
