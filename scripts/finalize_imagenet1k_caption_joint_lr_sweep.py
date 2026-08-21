#!/usr/bin/env python3
"""Select the final joint-training LR winner from top-k generation metrics."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


VALIDATION_SCHEMA = "selfless_imagenet1k_caption_joint_lr_ranking_v1"
CAPTION_SCHEMA = "selfless_imagenet1k_i2t_clip_metrics_v1"
DEFAULT_INITIALIZATION_METRICS = Path(
    "output/selfless-flow-imagenet1k-class-ascend64-b1024-800ep-fid-is/"
    "metrics.json"
)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _finite(value: Any, *, label: str, path: Path) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite {label}={result}: {path}")
    return result


def _average_tie_ranks(
    rows: list[dict[str, Any]],
    key: str,
    *,
    maximize: bool,
) -> dict[str, float]:
    direction = -1.0 if maximize else 1.0
    ordered = sorted(
        rows,
        key=lambda row: (direction * float(row[key]), str(row["id"])),
    )
    ranks: dict[str, float] = {}
    cursor = 0
    while cursor < len(ordered):
        end = cursor + 1
        value = float(ordered[cursor][key])
        while end < len(ordered) and math.isclose(
            float(ordered[end][key]),
            value,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            end += 1
        rank = ((cursor + 1) + end) / 2.0
        for row in ordered[cursor:end]:
            ranks[str(row["id"])] = rank
        cursor = end
    return ranks


def _direction(delta: float, *, lower_is_better: bool) -> str:
    if math.isclose(float(delta), 0.0, rel_tol=0.0, abs_tol=1.0e-12):
        return "unchanged"
    improved = delta < 0.0 if lower_is_better else delta > 0.0
    return "improved" if improved else "degraded"


def _safe_relative_subdir(value: Any, *, label: str) -> Path:
    path = Path(str(value))
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError(f"{label} must be a safe relative path: {value!r}")
    return path


def collect_final(
    validation_ranking_path: Path,
    output_root: Path,
    *,
    require_complete: bool,
    strategy: str = "spatial_halton",
    initialization_metrics_path: Path = DEFAULT_INITIALIZATION_METRICS,
) -> dict[str, Any]:
    validation = _read_json(validation_ranking_path)
    if validation.get("schema") != VALIDATION_SCHEMA:
        raise ValueError(
            f"validation ranking schema mismatch: {validation_ranking_path}"
        )
    if validation.get("status") != "complete":
        raise ValueError("validation ranking must be complete before final selection")
    top_k = [str(value) for value in validation.get("top_k", [])]
    if not top_k:
        raise ValueError("validation ranking has no top-k candidates")
    validation_rows = {
        str(row["id"]): row for row in validation.get("ranking", [])
    }
    initialization = _read_json(initialization_metrics_path)
    if initialization.get("official_protocol") is not True:
        raise ValueError(
            "initialization T2I metrics are not official-protocol: "
            f"{initialization_metrics_path}"
        )
    if int(initialization.get("samples_evaluated", -1)) != 50_000:
        raise ValueError("initialization T2I comparison requires 50,000 samples")
    initialization_strategy = initialization.get("strategies", {}).get(strategy)
    if not isinstance(initialization_strategy, dict):
        raise ValueError(
            f"initialization metrics are missing strategy={strategy!r}: "
            f"{initialization_metrics_path}"
        )
    initialization_fid = _finite(
        initialization_strategy["fid"],
        label="initialization fid",
        path=initialization_metrics_path,
    )
    initialization_is = _finite(
        initialization_strategy["inception_score_mean"],
        label="initialization inception score",
        path=initialization_metrics_path,
    )
    initialization_is_std = _finite(
        initialization_strategy["inception_score_std"],
        label="initialization inception score std",
        path=initialization_metrics_path,
    )
    clear_fid_regression = max(1.0, 0.05 * initialization_fid)
    clear_is_regression = max(initialization_is_std, 0.01 * initialization_is)

    rows: list[dict[str, Any]] = []
    missing: list[dict[str, str]] = []
    excluded: list[dict[str, Any]] = []
    for run_id in top_k:
        validation_row = validation_rows[run_id]
        run_project = str(
            validation_row.get(
                "run_project",
                f"selfless-flow-imagenet1k-caption-joint-sweep-{run_id}",
            )
        )
        run_root = output_root / run_project
        evaluation_subdir = _safe_relative_subdir(
            validation_row.get(
                "generation_evaluation_subdir", "generation-evaluation"
            ),
            label="generation_evaluation_subdir",
        )
        caption_path = run_root / evaluation_subdir / "i2t-clip/metrics.json"
        image_path = run_root / evaluation_subdir / "t2i-fid-is/metrics.json"
        absent = [
            str(path)
            for path in (caption_path, image_path)
            if not path.is_file()
        ]
        if absent:
            missing.append({"id": run_id, "missing": ",".join(absent)})
            continue
        caption = _read_json(caption_path)
        if caption.get("schema") != CAPTION_SCHEMA:
            raise ValueError(f"caption metric schema mismatch: {caption_path}")
        if int(caption.get("samples", -1)) != 1_000:
            raise ValueError(f"Caption CLIP requires 1,000 samples: {caption_path}")
        class_balance = caption.get("class_balance", {})
        expected_balance = {
            "class_count": 1_000,
            "min_samples_per_class": 1,
            "max_samples_per_class": 1,
        }
        for key, expected in expected_balance.items():
            if int(class_balance.get(key, -1)) != expected:
                raise ValueError(
                    f"Caption CLIP is not one-sample-per-class balanced "
                    f"({key}): {caption_path}"
                )
        image = _read_json(image_path)
        if image.get("official_protocol") is not True:
            raise ValueError(f"T2I metrics are not official-protocol: {image_path}")
        if int(image.get("samples_evaluated", -1)) != 50_000:
            raise ValueError(f"T2I FID/IS requires 50,000 samples: {image_path}")
        strategy_metrics = image.get("strategies", {}).get(strategy)
        if not isinstance(strategy_metrics, dict):
            raise ValueError(
                f"missing T2I strategy={strategy!r}: {image_path}"
            )
        if int(strategy_metrics.get("count", -1)) != 50_000:
            raise ValueError(
                f"T2I strategy count must be 50,000: {image_path}"
            )
        fid = _finite(strategy_metrics["fid"], label="fid", path=image_path)
        inception_score = _finite(
            strategy_metrics["inception_score_mean"],
            label="inception_score_mean",
            path=image_path,
        )
        fid_delta = fid - initialization_fid
        inception_delta = inception_score - initialization_is
        fid_direction = _direction(fid_delta, lower_is_better=True)
        is_direction = _direction(inception_delta, lower_is_better=False)
        result_row = {
            "id": run_id,
            "run_project": run_project,
            "generation_evaluation_subdir": str(evaluation_subdir),
            "backbone_lr": float(validation_row["backbone_lr"]),
            "flow_lr": float(validation_row["flow_lr"]),
            **(
                {"lambda_text": float(validation_row["lambda_text"])}
                if "lambda_text" in validation_row
                else {}
            ),
            "validation_rank": int(validation_row["overall_rank"]),
            "caption_clip_score": _finite(
                caption["clip"]["caption_clip_score"],
                label="caption_clip_score",
                path=caption_path,
            ),
            "fid": fid,
            "inception_score_mean": inception_score,
            "t2i_vs_initialization": {
                "fid_delta": fid_delta,
                "fid_status": fid_direction,
                "inception_score_delta": inception_delta,
                "inception_score_status": is_direction,
                "joint_status": (
                    "improved_both"
                    if fid_direction == is_direction == "improved"
                    else (
                        "degraded_both"
                        if fid_direction == is_direction == "degraded"
                        else "mixed_or_unchanged"
                    )
                ),
            },
            "caption_samples": int(caption["samples"]),
            "image_samples": int(image["samples_evaluated"]),
            "caption_metrics_path": str(caption_path),
            "image_metrics_path": str(image_path),
        }
        if (
            fid_delta > clear_fid_regression
            and inception_delta < -clear_is_regression
        ):
            excluded.append(
                {
                    "id": run_id,
                    "reason": "clear T2I regression versus initialization",
                    "thresholds": {
                        "fid_increase": clear_fid_regression,
                        "inception_score_decrease": clear_is_regression,
                    },
                    "metrics": result_row,
                }
            )
            continue
        rows.append(result_row)

    if require_complete and missing:
        raise FileNotFoundError(
            "generation evaluation is incomplete: "
            + "; ".join(f"{row['id']}:{row['missing']}" for row in missing)
        )
    if rows:
        caption_ranks = _average_tie_ranks(
            rows, "caption_clip_score", maximize=True
        )
        fid_ranks = _average_tie_ranks(rows, "fid", maximize=False)
        is_ranks = _average_tie_ranks(
            rows, "inception_score_mean", maximize=True
        )
        for row in rows:
            run_id = str(row["id"])
            row["caption_rank"] = caption_ranks[run_id]
            row["fid_rank"] = fid_ranks[run_id]
            row["inception_score_rank"] = is_ranks[run_id]
            row["generation_mean_rank"] = (
                caption_ranks[run_id]
                + fid_ranks[run_id]
                + is_ranks[run_id]
            ) / 3.0
        rows.sort(
            key=lambda row: (
                float(row["generation_mean_rank"]),
                int(row["validation_rank"]),
                str(row["id"]),
            )
        )
        for index, row in enumerate(rows, start=1):
            row["final_rank"] = index

    complete = len(rows) + len(excluded) == len(top_k) and not missing
    return {
        "schema": "selfless_imagenet1k_caption_joint_final_selection_v1",
        "status": "complete" if complete else "incomplete",
        "validation_ranking": str(validation_ranking_path),
        "validation_leader": (
            validation.get("validation_leader")
            or validation.get("winner")
        ),
        "top_k": top_k,
        "generation_strategy": strategy,
        "initialization_t2i_baseline": {
            "path": str(initialization_metrics_path),
            "samples": 50_000,
            "fid": initialization_fid,
            "inception_score_mean": initialization_is,
            "inception_score_std": initialization_is_std,
        },
        "clear_t2i_regression_rule": {
            "requires_both": True,
            "fid_increase_threshold": clear_fid_regression,
            "inception_score_decrease_threshold": clear_is_regression,
        },
        "selection_rule": (
            "lowest mean rank across caption CLIP (higher is better), FID "
            "(lower), and Inception Score (higher); validation rank breaks ties"
        ),
        "missing": missing,
        "excluded": excluded,
        "ranking": rows,
        "winner": rows[0]["id"] if complete and rows else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation_ranking", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, default=Path("output"))
    parser.add_argument("--strategy", default="spatial_halton")
    parser.add_argument(
        "--initialization_metrics",
        type=Path,
        default=DEFAULT_INITIALIZATION_METRICS,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require_complete", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = collect_final(
        args.validation_ranking,
        args.output_root,
        require_complete=bool(args.require_complete),
        strategy=str(args.strategy),
        initialization_metrics_path=args.initialization_metrics,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
