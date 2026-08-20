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


def collect_final(
    validation_ranking_path: Path,
    output_root: Path,
    *,
    require_complete: bool,
    strategy: str = "spatial_halton",
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

    rows: list[dict[str, Any]] = []
    missing: list[dict[str, str]] = []
    for run_id in top_k:
        run_root = output_root / (
            f"selfless-flow-imagenet1k-caption-joint-sweep-{run_id}"
        )
        caption_path = run_root / "generation-evaluation/i2t-clip/metrics.json"
        image_path = run_root / "generation-evaluation/t2i-fid-is/metrics.json"
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
        image = _read_json(image_path)
        if image.get("official_protocol") is not True:
            raise ValueError(f"T2I metrics are not official-protocol: {image_path}")
        strategy_metrics = image.get("strategies", {}).get(strategy)
        if not isinstance(strategy_metrics, dict):
            raise ValueError(
                f"missing T2I strategy={strategy!r}: {image_path}"
            )
        validation_row = validation_rows[run_id]
        rows.append(
            {
                "id": run_id,
                "backbone_lr": float(validation_row["backbone_lr"]),
                "flow_lr": float(validation_row["flow_lr"]),
                "validation_rank": int(validation_row["overall_rank"]),
                "caption_clip_score": _finite(
                    caption["clip"]["caption_clip_score"],
                    label="caption_clip_score",
                    path=caption_path,
                ),
                "fid": _finite(
                    strategy_metrics["fid"],
                    label="fid",
                    path=image_path,
                ),
                "inception_score_mean": _finite(
                    strategy_metrics["inception_score_mean"],
                    label="inception_score_mean",
                    path=image_path,
                ),
                "caption_samples": int(caption["samples"]),
                "image_samples": int(image["samples_evaluated"]),
                "caption_metrics_path": str(caption_path),
                "image_metrics_path": str(image_path),
            }
        )

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

    complete = len(rows) == len(top_k) and not missing
    return {
        "schema": "selfless_imagenet1k_caption_joint_final_selection_v1",
        "status": "complete" if complete else "incomplete",
        "validation_ranking": str(validation_ranking_path),
        "validation_winner": validation.get("winner"),
        "top_k": top_k,
        "generation_strategy": strategy,
        "selection_rule": (
            "lowest mean rank across caption CLIP (higher is better), FID "
            "(lower), and Inception Score (higher); validation rank breaks ties"
        ),
        "missing": missing,
        "ranking": rows,
        "winner": rows[0]["id"] if complete else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation_ranking", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, default=Path("output"))
    parser.add_argument("--strategy", default="spatial_halton")
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
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
