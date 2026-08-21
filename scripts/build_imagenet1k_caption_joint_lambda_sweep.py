#!/usr/bin/env python3
"""Materialize the short lambda_text sweep after staged LR selection."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from decimal import Decimal
from pathlib import Path
from typing import Any


DEFAULT_LR_SWEEP = Path(
    "configs/selfless/imagenet1k_caption_joint_lr_sweep_10ep.json"
)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def lambda_tag(value: float) -> str:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"lambda_text must be finite and positive, got {value}")
    rendered = format(Decimal(str(value)).normalize(), "f")
    return rendered.replace("-", "m").replace(".", "p")


def build_manifest(
    probe_path: Path,
    lr_ranking_path: Path,
    lr_sweep_path: Path,
    *,
    top_lr_id: str | None = None,
) -> dict[str, Any]:
    probe = _read_json(probe_path)
    if probe.get("schema") != "selfless_caption_t2i_gradient_probe_v1":
        raise ValueError(f"gradient probe schema mismatch: {probe_path}")
    if probe.get("status") != "complete":
        raise ValueError("gradient probe must be complete")
    batches = probe.get("batches", [])
    if not isinstance(batches, list) or not 16 <= len(batches) <= 32:
        raise ValueError("lambda sweep requires a formal 16-32 batch probe")
    summary = probe["summary"]
    lambda_summary = summary["lambda_text"]
    values = sorted({float(value) for value in lambda_summary["candidates"]})
    lower, upper = map(float, lambda_summary["bounds"])
    if not values or any(
        not math.isfinite(value) or not lower <= value <= upper for value in values
    ):
        raise ValueError("probe emitted invalid lambda_text candidates")

    ranking = _read_json(lr_ranking_path)
    if ranking.get("status") != "complete":
        raise ValueError("LR validation ranking must be complete")
    selected_lr = str(
        top_lr_id
        or ranking.get("validation_leader")
        or ranking.get("winner", "")
    )
    ranking_rows = {
        str(row["id"]): row for row in ranking.get("ranking", [])
    }
    if selected_lr not in ranking_rows:
        raise ValueError(f"top LR ID is absent from ranking: {selected_lr!r}")
    if len(ranking_rows[selected_lr].get("checkpoints", [])) < 2:
        raise ValueError("top LR candidate requires at least two validation points")

    lr_sweep = _read_json(lr_sweep_path)
    lr_candidates = {
        str(candidate["id"]): candidate for candidate in lr_sweep["candidates"]
    }
    if selected_lr not in lr_candidates:
        raise ValueError(f"top LR ID is absent from sweep: {selected_lr!r}")
    lr_candidate = lr_candidates[selected_lr]

    candidates = []
    for value in values:
        tag = lambda_tag(value)
        candidate_id = f"{selected_lr}-lt{tag}"
        reuses_lr_baseline = math.isclose(value, 0.2, rel_tol=0.0, abs_tol=1e-12)
        if reuses_lr_baseline:
            run_project = f"selfless-flow-imagenet1k-caption-joint-sweep-{selected_lr}"
            source = "lr_sweep_baseline"
        else:
            run_project = (
                "selfless-flow-imagenet1k-caption-joint-lambda-"
                f"{selected_lr}-lt{tag}"
            )
            source = "lambda_sweep"
        candidates.append(
            {
                "id": candidate_id,
                "run_project": run_project,
                "source": source,
                "reuse_existing_run": reuses_lr_baseline,
                "top_lr_id": selected_lr,
                "backbone_lr": float(lr_candidate["backbone_lr"]),
                "flow_lr": float(lr_candidate["flow_lr"]),
                "lambda_text": value,
                "lambda_image": 1.0,
            }
        )

    return {
        "schema": "selfless_imagenet1k_caption_joint_lambda_sweep_v1",
        "source_probe": {
            "path": str(probe_path),
            "sha256": _sha256(probe_path),
            "checkpoint": probe["checkpoint"],
            "batches": len(batches),
            "gradient_ratio_median": float(
                summary["ratio_g_image_over_g_text"]["median"]
            ),
            "gradient_cosine": summary["cosine"],
            "task_conflict": summary["task_conflict"],
        },
        "source_lr_selection": {
            "path": str(lr_ranking_path),
            "top_lr_id": selected_lr,
            "validation_points": len(
                ranking_rows[selected_lr]["checkpoints"]
            ),
        },
        "initialization": probe["checkpoint"],
        "lambda_policy": {
            "raw_center": float(lambda_summary["raw_center"]),
            "bounded_center": float(lambda_summary["center"]),
            "bounds": [lower, upper],
            "scaled": lambda_summary["scaled"],
            "reference_points": lambda_summary["reference_points"],
            "lambda_image": 1.0,
            "cartesian_product_with_lr": False,
        },
        "selection": {
            "validation_steps": [2404, 4808],
            "require_gradient_probe": True,
            "rule": (
                "exclude instability, mean-rank final/best text and image "
                "losses, and use best-to-final regression as tie-break; "
                "generation metrics make the final decision"
            ),
            "top_k_for_generation_evaluation": min(3, len(candidates)),
        },
        "candidates": candidates,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--lr_ranking", type=Path, required=True)
    parser.add_argument("--lr_sweep", type=Path, default=DEFAULT_LR_SWEEP)
    parser.add_argument("--top_lr_id", default=None)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_manifest(
        args.probe,
        args.lr_ranking,
        args.lr_sweep,
        top_lr_id=args.top_lr_id,
    )
    rendered = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
