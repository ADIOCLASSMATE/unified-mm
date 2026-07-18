#!/usr/bin/env python3
"""Validate and summarize formal Qwen-Show-o CFG sweep points."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.validate_showo_cfg_metrics import (  # noqa: E402
    EXPECTED_CHECKPOINT,
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_CONFIG,
    EXPECTED_SAMPLES_SHA256,
    nested,
    validate_metrics,
)


SCHEMA = "qwen_showo_cfg_sweep_summary_v1"


class SummaryError(ValueError):
    pass


def parse_assignment(spec: str, *, option: str) -> tuple[float, str]:
    if "=" not in spec:
        raise SummaryError(f"{option} must use COMMON_CFG=VALUE: {spec!r}")
    raw_cfg, value = spec.split("=", 1)
    try:
        common_cfg = float(raw_cfg)
    except ValueError as error:
        raise SummaryError(f"invalid common CFG value: {raw_cfg!r}") from error
    if not math.isfinite(common_cfg) or common_cfg < 1.0:
        raise SummaryError(
            f"common CFG must be finite and >=1.0, got {common_cfg}"
        )
    if not value:
        raise SummaryError(f"{option} value must not be empty")
    return common_cfg, value


def build_summary(
    point_specs: Sequence[str],
    job_specs: Sequence[str],
    *,
    require_images: bool,
    expected_checkpoint: str = EXPECTED_CHECKPOINT,
    expected_checkpoint_sha256: str | None = EXPECTED_CHECKPOINT_SHA256,
    expected_samples_sha256: str | None = EXPECTED_SAMPLES_SHA256,
    expected_config: str = EXPECTED_CONFIG,
) -> dict[str, Any]:
    if not point_specs:
        raise SummaryError("at least one --point COMMON_CFG=METRICS is required")

    points: dict[float, Path] = {}
    for spec in point_specs:
        common_cfg, raw_path = parse_assignment(spec, option="--point")
        if common_cfg in points:
            raise SummaryError(f"duplicate common CFG point: {common_cfg}")
        points[common_cfg] = Path(raw_path)

    jobs: dict[float, str] = {}
    for spec in job_specs:
        common_cfg, job_name = parse_assignment(spec, option="--job")
        if common_cfg in jobs:
            raise SummaryError(f"duplicate common CFG job: {common_cfg}")
        jobs[common_cfg] = job_name
    unknown_jobs = sorted(set(jobs) - set(points))
    if unknown_jobs:
        raise SummaryError(f"jobs without matching points: {unknown_jobs}")

    rows = []
    for common_cfg in sorted(points):
        guidance_scale = common_cfg - 1.0
        metrics_path = points[common_cfg]
        payload, errors, image_count = validate_metrics(
            metrics_path,
            expected_guidance_scale=guidance_scale,
            require_images=require_images,
            expected_checkpoint=expected_checkpoint,
            expected_checkpoint_sha256=expected_checkpoint_sha256,
            expected_samples_sha256=expected_samples_sha256,
            expected_config=expected_config,
        )
        if errors:
            raise SummaryError(
                f"{metrics_path}: validation failed: {', '.join(errors)}"
            )
        rows.append(
            {
                "common_cfg_scale": common_cfg,
                "showo_guidance_scale": guidance_scale,
                "fid": nested(payload, "metrics", "fid"),
                "inception_score_mean": nested(
                    payload, "metrics", "inception_score_mean"
                ),
                "inception_score_std": nested(
                    payload, "metrics", "inception_score_std"
                ),
                "metrics_path": str(metrics_path.resolve()),
                "output_path": str(metrics_path.resolve().parent),
                "image_count": image_count,
                "job_name": jobs.get(common_cfg),
            }
        )

    best_fid = min(rows, key=lambda row: (row["fid"], row["common_cfg_scale"]))
    best_is = max(
        rows,
        key=lambda row: (
            row["inception_score_mean"],
            -row["common_cfg_scale"],
        ),
    )
    return {
        "schema": SCHEMA,
        "protocol": {
            "name": "imagenet100-balanced-val100-per-class-class-name-v1",
            "checkpoint": str(
                (REPO_ROOT / expected_checkpoint).resolve()
                if not Path(expected_checkpoint).is_absolute()
                else Path(expected_checkpoint).resolve()
            ),
            "checkpoint_sha256": expected_checkpoint_sha256,
            "config": str(
                (REPO_ROOT / expected_config).resolve()
                if not Path(expected_config).is_absolute()
                else Path(expected_config).resolve()
            ),
            "guidance_formula": "(1+s)*conditional-s*unconditional",
            "common_cfg_mapping": "w=1+s",
            "temperature_schedule": (
                "official_showo_cumulative_one_minus_ratio"
            ),
            "timesteps": 12,
            "temperature": 1.0,
            "seed": 42,
            "samples_per_point": 10_000,
            "world_size": 8,
            "local_batch_size": 8,
        },
        "points": rows,
        "best_by_fid": {
            key: best_fid[key]
            for key in (
                "common_cfg_scale",
                "showo_guidance_scale",
                "fid",
                "metrics_path",
                "job_name",
            )
        },
        "best_by_is": {
            key: best_is[key]
            for key in (
                "common_cfg_scale",
                "showo_guidance_scale",
                "inception_score_mean",
                "inception_score_std",
                "metrics_path",
                "job_name",
            )
        },
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "common_cfg_scale",
        "showo_guidance_scale",
        "fid",
        "inception_score_mean",
        "inception_score_std",
        "image_count",
        "job_name",
        "metrics_path",
        "output_path",
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in rows)
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize a formal Qwen-Show-o CFG sweep."
    )
    parser.add_argument("--point", action="append", default=[])
    parser.add_argument("--job", action="append", default=[])
    parser.add_argument("--require-images", action="store_true")
    parser.add_argument("--expected-checkpoint", default=EXPECTED_CHECKPOINT)
    parser.add_argument(
        "--expected-checkpoint-sha256",
        default=EXPECTED_CHECKPOINT_SHA256,
    )
    parser.add_argument(
        "--expected-samples-sha256",
        default=EXPECTED_SAMPLES_SHA256,
    )
    parser.add_argument("--expected-config", default=EXPECTED_CONFIG)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_summary(
        args.point,
        args.job,
        require_images=bool(args.require_images),
        expected_checkpoint=args.expected_checkpoint,
        expected_checkpoint_sha256=args.expected_checkpoint_sha256,
        expected_samples_sha256=args.expected_samples_sha256,
        expected_config=args.expected_config,
    )
    write_json(args.output_json, summary)
    write_csv(args.output_csv, summary["points"])
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
