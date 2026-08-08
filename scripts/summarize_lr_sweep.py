#!/usr/bin/env python3
"""Summarize validation and FID/IS results for the 40-epoch LR sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def _validation_rows(run_dir: Path) -> list[tuple[int, float]]:
    rows: list[tuple[int, float]] = []
    for path in run_dir.glob("validation_metrics_step_*.json"):
        payload = _load_json(path)
        step = int(payload["global_step"])
        loss = float(payload["metrics"]["val/loss"])
        rows.append((step, loss))
    return sorted(rows)


def _metric_rank(rows: list[dict[str, Any]], key: str, reverse: bool = False) -> None:
    complete = [row for row in rows if row[key] is not None]
    for rank, row in enumerate(
        sorted(complete, key=lambda item: float(item[key]), reverse=reverse),
        start=1,
    ):
        row[f"{key}_rank"] = rank


def summarize(manifest_path: Path) -> dict[str, Any]:
    manifest = _load_json(manifest_path)
    validation_step = int(manifest["selection"]["validation_step"])
    strategy = str(manifest["selection"]["strategy"])
    rows: list[dict[str, Any]] = []

    for candidate in manifest["candidates"]:
        run_dir = Path(candidate["run_dir"])
        eval_dir = Path(candidate["eval_dir"])
        validation = _validation_rows(run_dir)
        by_step = dict(validation)
        final_val_loss = by_step.get(validation_step)
        best_step, best_val_loss = (
            min(validation, key=lambda item: item[1]) if validation else (None, None)
        )
        metrics_path = eval_dir / "metrics.json"
        fid = inception_score = inception_score_std = None
        if metrics_path.is_file():
            metrics = _load_json(metrics_path)["strategies"][strategy]
            fid = float(metrics["fid"])
            inception_score = float(metrics["inception_score_mean"])
            inception_score_std = float(metrics["inception_score_std"])
        rows.append(
            {
                "id": candidate["id"],
                "backbone_lr": float(candidate["backbone_lr"]),
                "flow_lr": float(candidate["flow_lr"]),
                "final_val_loss": final_val_loss,
                "best_val_loss": best_val_loss,
                "best_val_step": best_step,
                "overfit_gap": (
                    None
                    if final_val_loss is None or best_val_loss is None
                    else final_val_loss - best_val_loss
                ),
                "fid": fid,
                "inception_score_mean": inception_score,
                "inception_score_std": inception_score_std,
                "run_dir": str(run_dir),
                "eval_dir": str(eval_dir),
            }
        )

    _metric_rank(rows, "final_val_loss")
    _metric_rank(rows, "fid")
    _metric_rank(rows, "inception_score_mean", reverse=True)
    for row in rows:
        rank_keys = (
            "final_val_loss_rank",
            "fid_rank",
            "inception_score_mean_rank",
        )
        ranks = [row.get(key) for key in rank_keys]
        row["mean_rank"] = (
            sum(float(rank) for rank in ranks) / len(ranks)
            if all(rank is not None for rank in ranks)
            else None
        )

    complete = [row for row in rows if row["mean_rank"] is not None]
    winner = (
        min(
            complete,
            key=lambda row: (
                float(row["mean_rank"]),
                float(row["overfit_gap"]),
                float(row["final_val_loss"]),
            ),
        )
        if complete
        else None
    )
    return {
        "schema": "selfless_flow_lr_sweep_summary_v1",
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


def _format_value(value: Any, digits: int = 6) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


def _print_table(summary: dict[str, Any]) -> None:
    headers = (
        "id",
        "backbone_lr",
        "flow_lr",
        "final_val_loss",
        "best_val_loss",
        "overfit_gap",
        "fid",
        "IS",
        "mean_rank",
    )
    print("\t".join(headers))
    for row in summary["candidates"]:
        print(
            "\t".join(
                (
                    row["id"],
                    _format_value(row["backbone_lr"]),
                    _format_value(row["flow_lr"]),
                    _format_value(row["final_val_loss"]),
                    _format_value(row["best_val_loss"]),
                    _format_value(row["overfit_gap"]),
                    _format_value(row["fid"]),
                    _format_value(row["inception_score_mean"]),
                    _format_value(row["mean_rank"]),
                )
            )
        )
    if summary["winner"] is not None:
        winner = summary["winner"]
        print(
            "Winner: "
            f"{winner['id']} (backbone_lr={winner['backbone_lr']:g}, "
            f"flow_lr={winner['flow_lr']:g}, mean_rank={winner['mean_rank']:.3f})"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("configs/selfless/imagenet100_caption_lr_sweep_40ep.json"),
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
    if (
        args.require_complete
        and summary["complete_candidates"] != summary["total_candidates"]
    ):
        raise SystemExit(
            "incomplete sweep: "
            f"{summary['complete_candidates']}/{summary['total_candidates']} candidates "
            "have final validation and FID/IS metrics"
        )


if __name__ == "__main__":
    main()
