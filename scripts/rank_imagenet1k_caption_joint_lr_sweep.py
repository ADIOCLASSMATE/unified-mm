#!/usr/bin/env python3
"""Rank completed ImageNet-1K caption/T2I LR-sweep candidates."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from utils.joint_sweep_health import training_health


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
    candidate_selection_path: Path | None = None,
) -> dict[str, Any]:
    sweep = _read_json(sweep_path)
    candidates = sweep.get("candidates")
    selection = sweep.get("selection")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError(f"sweep has no candidates: {sweep_path}")
    if not isinstance(selection, dict):
        raise ValueError(f"sweep has no selection contract: {sweep_path}")
    candidate_selection: dict[str, Any] | None = None
    if candidate_selection_path is not None:
        candidate_selection = _read_json(candidate_selection_path)
        selected_ids = [str(value) for value in candidate_selection.get("selected", [])]
        if not selected_ids:
            raise ValueError(
                f"candidate selection has no selected IDs: {candidate_selection_path}"
            )
        by_id = {str(candidate["id"]): candidate for candidate in candidates}
        unknown = sorted(set(selected_ids).difference(by_id))
        if unknown:
            raise ValueError(
                f"candidate selection contains IDs absent from sweep: {unknown}"
            )
        candidates = [by_id[run_id] for run_id in selected_ids]
    validation_steps = [int(step) for step in selection["validation_steps"]]
    final_step = validation_steps[-1]
    require_gradient_probe = bool(selection.get("require_gradient_probe", False))
    probe_contract = sweep.get("gradient_probe", {})
    if not isinstance(probe_contract, dict):
        raise ValueError(f"invalid gradient_probe contract: {sweep_path}")
    default_lambda_text = float(probe_contract.get("training_lambda_text", 0.2))
    default_lambda_image = float(probe_contract.get("training_lambda_image", 1.0))
    if default_lambda_text <= 0.0 or default_lambda_image <= 0.0:
        raise ValueError("training loss weights must be positive")

    rows: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for candidate in candidates:
        run_id = str(candidate["id"])
        run_project = str(
            candidate.get(
                "run_project",
                f"selfless-flow-imagenet1k-caption-joint-sweep-{run_id}",
            )
        )
        run_root = output_root / run_project
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
        probe_path = (
            run_root
            / "gradient_probe"
            / f"checkpoint-{final_step}"
            / "probe.json"
        )
        missing_probe = require_gradient_probe and not probe_path.is_file()
        if missing_steps or missing_probe:
            missing_row: dict[str, Any] = {
                "id": run_id,
                "run_root": str(run_root),
                "missing_validation_steps": missing_steps,
            }
            if missing_probe:
                missing_row["missing_gradient_probe"] = str(probe_path)
            missing.append(missing_row)
            continue
        probe_summary: dict[str, Any] | None = None
        if require_gradient_probe:
            probe = _read_json(probe_path)
            if probe.get("schema") != "selfless_caption_t2i_gradient_probe_v1":
                raise ValueError(f"gradient probe schema mismatch: {probe_path}")
            if probe.get("status") != "complete":
                raise ValueError(f"gradient probe is incomplete: {probe_path}")
            probe_summary = probe["summary"]
            ratio_median = float(
                probe_summary["ratio_g_image_over_g_text"]["median"]
            )
            cosine_median = float(probe_summary["cosine"]["median"])
            if (
                not math.isfinite(ratio_median)
                or ratio_median <= 0.0
                or not math.isfinite(cosine_median)
                or not -1.0 <= cosine_median <= 1.0
            ):
                excluded.append(
                    {
                        "id": run_id,
                        "run_project": run_project,
                        "reason": "invalid gradient probe",
                        "probe_path": str(probe_path),
                    }
                )
                continue
        health = training_health(run_root)
        if not health["passed"]:
            excluded.append(
                {
                    "id": run_id,
                    "run_project": run_project,
                    "reason": "training instability",
                    "health": health,
                }
            )
            continue
        final = next(item for item in checkpoints if item["step"] == final_step)
        best_text = min(float(item["text_loss"]) for item in checkpoints)
        best_image = min(float(item["image_loss"]) for item in checkpoints)
        final_text = float(final["text_loss"])
        final_image = float(final["image_loss"])
        gradient_diagnostics: dict[str, float] = {}
        if probe_summary is not None:
            candidate_lambda_text = float(
                candidate.get("lambda_text", default_lambda_text)
            )
            target_ratio = candidate_lambda_text / default_lambda_image
            task_conflict = probe_summary["task_conflict"]
            negative_fraction = float(
                task_conflict.get(
                    "negative_fraction",
                    float(bool(task_conflict.get("persistent_negative", False))),
                )
            )
            pre_clip_norms = health.get("pre_clip_grad_norms", [])
            clipped_events = sum(
                float(item["pre_clip"]) > 1.0 for item in pre_clip_norms
            )
            gradient_diagnostics = {
                "gradient_balance_target_ratio": target_ratio,
                "gradient_balance_log_error": abs(
                    math.log(ratio_median / target_ratio)
                ),
                "negative_cosine_fraction": negative_fraction,
                "observed_clip_event_fraction": (
                    clipped_events / len(pre_clip_norms) if pre_clip_norms else 0.0
                ),
            }
        rows.append(
            {
                "id": run_id,
                "run_project": run_project,
                "backbone_lr": float(candidate["backbone_lr"]),
                "flow_lr": float(candidate["flow_lr"]),
                **(
                    {"lambda_text": float(candidate["lambda_text"])}
                    if "lambda_text" in candidate
                    else {}
                ),
                **(
                    {
                        "evaluation_model_subdir": str(
                            candidate["evaluation_model_subdir"]
                        )
                    }
                    if "evaluation_model_subdir" in candidate
                    else {}
                ),
                **(
                    {
                        "generation_evaluation_subdir": str(
                            candidate["generation_evaluation_subdir"]
                        )
                    }
                    if "generation_evaluation_subdir" in candidate
                    else {}
                ),
                "final_step": final_step,
                "final_text_loss": final_text,
                "final_image_loss": final_image,
                "best_text_loss": best_text,
                "best_image_loss": best_image,
                "health": health,
                **gradient_diagnostics,
                **(
                    {
                        "gradient_probe": {
                            "path": str(probe_path),
                            "ratio_median": ratio_median,
                            "cosine_median": cosine_median,
                            "task_conflict": probe_summary["task_conflict"],
                        }
                    }
                    if probe_summary is not None
                    else {}
                ),
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
            "candidate_selection": (
                str(candidate_selection_path)
                if candidate_selection_path is not None
                else None
            ),
            "completed_candidates": 0,
            "expected_candidates": len(candidates),
            "missing": missing,
            "excluded": excluded,
            "ranking": [],
            "top_k": [],
            "validation_leader": None,
            "winner": None,
        }

    final_text_ranks = _average_tie_ranks(rows, "final_text_loss")
    final_image_ranks = _average_tie_ranks(rows, "final_image_loss")
    best_text_ranks = _average_tie_ranks(rows, "best_text_loss")
    best_image_ranks = _average_tie_ranks(rows, "best_image_loss")
    if require_gradient_probe:
        gradient_balance_ranks = _average_tie_ranks(
            rows, "gradient_balance_log_error"
        )
        conflict_ranks = _average_tie_ranks(rows, "negative_cosine_fraction")
        clip_ranks = _average_tie_ranks(rows, "observed_clip_event_fraction")
    for row in rows:
        run_id = str(row["id"])
        row["final_text_rank"] = final_text_ranks[run_id]
        row["final_image_rank"] = final_image_ranks[run_id]
        row["best_text_rank"] = best_text_ranks[run_id]
        row["best_image_rank"] = best_image_ranks[run_id]
        loss_rank_sum = (
            float(row["final_text_rank"])
            + float(row["final_image_rank"])
            + float(row["best_text_rank"])
            + float(row["best_image_rank"])
        )
        row["validation_loss_mean_rank"] = loss_rank_sum / 4.0
        if require_gradient_probe:
            row["gradient_balance_rank"] = gradient_balance_ranks[run_id]
            row["negative_cosine_conflict_rank"] = conflict_ranks[run_id]
            row["clip_stability_rank"] = clip_ranks[run_id]
            row["mean_rank"] = (
                loss_rank_sum
                + gradient_balance_ranks[run_id]
                + conflict_ranks[run_id]
                + clip_ranks[run_id]
            ) / 7.0
        else:
            row["mean_rank"] = row["validation_loss_mean_rank"]
    rows.sort(
        key=lambda row: (
            float(row["mean_rank"]),
            float(row.get("gradient_balance_log_error", 0.0)),
            float(row.get("negative_cosine_fraction", 0.0)),
            float(row.get("observed_clip_event_fraction", 0.0)),
            float(row["mean_normalized_final_regression"]),
            str(row["id"]),
        )
    )
    for rank, row in enumerate(rows, start=1):
        row["overall_rank"] = rank
    top_k_count = min(
        int(selection.get("top_k_for_generation_evaluation", 3)),
        len(rows),
    )
    status = "complete" if not missing else "incomplete"
    return {
        "schema": "selfless_imagenet1k_caption_joint_lr_ranking_v1",
        "status": status,
        "sweep": str(sweep_path),
        "candidate_selection": (
            str(candidate_selection_path)
            if candidate_selection_path is not None
            else None
        ),
        "completed_candidates": len(rows),
        "expected_candidates": len(candidates),
        "final_step": final_step,
        "selection_rule": selection["rule"],
        "missing": missing,
        "excluded": excluded,
        "ranking": rows,
        "top_k": [row["id"] for row in rows[:top_k_count]],
        "validation_leader": rows[0]["id"] if status == "complete" else None,
        "winner": None,
        "selection_status": (
            "validation_only_not_final"
            if status == "complete"
            else "incomplete"
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", type=Path, default=DEFAULT_SWEEP)
    parser.add_argument("--output_root", type=Path, default=Path("output"))
    parser.add_argument(
        "--candidate_selection",
        type=Path,
        default=None,
        help=(
            "Optional staged-selection JSON. Only its selected IDs must have "
            "the full validation history."
        ),
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--require_complete", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = collect(
        args.sweep,
        args.output_root,
        require_complete=bool(args.require_complete),
        candidate_selection_path=args.candidate_selection,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
