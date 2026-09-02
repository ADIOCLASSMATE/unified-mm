#!/usr/bin/env python3
"""Aggregate complete selected-protocol evaluations across checkpoints."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any


RETAINED_BENCHMARKS = (
    "mmbench_dev_en",
    "seed_bench_image",
    "sugarcrepe",
    "aro_vg_relation",
    "aro_vg_attribution",
)
GENERATION_PROTOCOL = "imagenet_val_fid50k_torch_fidelity_stratified_is"
GENERATION_REFERENCE = "imagenet_val_50000"
GENERATION_SCOPE = "same_protocol_only"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evaluation_root",
        type=Path,
        action="append",
        required=True,
        help="Completed native-full root; repeat once per checkpoint.",
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def finite(value: Any, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite trend metric: {label}={number}")
    return number


def metric(metrics: dict[str, Any], key: str, label: str) -> float:
    return finite(metrics[key], label)


def standard_retrieval_row(
    summary: dict[str, Any], label: str
) -> dict[str, float]:
    row: dict[str, float] = {}
    for direction, prefix in (("image_to_text", "i2t"), ("text_to_image", "t2i")):
        direction_metrics = summary[direction]
        for k in (1, 5, 10):
            row[f"{prefix}_r{k}"] = metric(
                direction_metrics, f"recall_at_{k}", f"{label}.{prefix}.r{k}"
            )
    return row


def trend_row(root: Path) -> dict[str, Any]:
    path = root / "native_full_evaluation_summary.json"
    summary = read_json(path)
    if summary.get("complete") is not True or summary.get("profile") != "formal":
        raise ValueError(f"input is not a complete formal evaluation: {path}")
    if summary.get("runtime_hashing_enabled", True) is not False:
        raise ValueError(f"input violates the no-hash contract: {path}")
    if summary.get("schema") != (
        "unified_native_full_checkpoint_evaluation_summary_v5"
    ):
        raise ValueError(f"input uses an obsolete evaluation protocol: {path}")
    contract = summary.get("dataset_contract", {})
    if contract.get("training_split") != "imagenet_train":
        raise ValueError(f"training split is not ImageNet train: {path}")
    if contract.get("evaluation_split") != "imagenet_val":
        raise ValueError(f"evaluation split is not ImageNet val: {path}")
    if contract.get("train_validation_overlap_allowed") is not False:
        raise ValueError(f"ImageNet train/val overlap is permitted: {path}")

    generation = summary["generation"]["imagenet_val_t2i"]
    expected_generation = {
        "project_formal_protocol": True,
        "leaderboard_comparable_to_adm_dit": False,
        "protocol_name": GENERATION_PROTOCOL,
        "reference_distribution": GENERATION_REFERENCE,
        "comparison_scope": GENERATION_SCOPE,
        "samples": 50_000,
        "is_split_assignment": "stratified_by_synset",
    }
    mismatches = {
        key: {"expected": value, "actual": generation.get(key)}
        for key, value in expected_generation.items()
        if generation.get(key) != value
    }
    if mismatches:
        raise ValueError(f"generation protocol is invalid in {path}: {mismatches}")
    if not str(generation.get("strategy", "")).strip():
        raise ValueError(f"generation strategy is missing in {path}")

    understanding = summary["understanding"]
    native = understanding["pretraining_native"]
    validation = understanding["heldout_validation"]
    benchmarks = {
        **native["paper_compositional_benchmarks"],
        **native["internal_ablation_diagnostics"],
    }
    missing = sorted(set(RETAINED_BENCHMARKS) - set(benchmarks))
    if missing:
        raise ValueError(f"retained benchmarks are missing from {path}: {missing}")

    benchmark_metrics = {
        task: benchmarks[task]["metrics"] for task in RETAINED_BENCHMARKS
    }
    mmbench = benchmark_metrics["mmbench_dev_en"]
    sugarcrepe = benchmark_metrics["sugarcrepe"]
    fid = finite(generation["fid"], "fid")
    is_mean = finite(
        generation["inception_score_mean"], "inception_score_mean"
    )
    is_std = finite(
        generation["inception_score_std"], "inception_score_std"
    )
    if fid < 0.0 or is_mean < 1.0 - 1.0e-9 or is_std < 0.0:
        raise ValueError(f"generation metric range is invalid: {path}")
    return {
        "global_step": int(summary["global_step"]),
        "checkpoint": str(summary["checkpoint"]),
        "evaluation_root": str(root.resolve()),
        "fid": fid,
        "inception_score_mean": is_mean,
        "inception_score_std": is_std,
        "validation_loss": finite(validation["val/loss"], "validation_loss"),
        "pure_text_macro": finite(
            summary["pure_text"]["macro_average_primary"], "pure_text_macro"
        ),
        "imagenet1k_zeroshot": {
            "top_1_accuracy": metric(
                native["imagenet1k_zeroshot_classification"],
                "top_1_accuracy",
                "imagenet1k.top_1_accuracy",
            ),
            "top_5_accuracy": metric(
                native["imagenet1k_zeroshot_classification"],
                "top_5_accuracy",
                "imagenet1k.top_5_accuracy",
            ),
        },
        "standard_retrieval": {
            task: standard_retrieval_row(result, task)
            for task, result in native["standard_cross_dataset_retrieval"].items()
        },
        "benchmarks": {
            "mmbench_dev_en": metric(
                mmbench,
                "circular_accuracy_language_prior_debiased",
                "mmbench.circular",
            ),
            "mmbench_dev_en_vanilla": metric(
                mmbench,
                "vanilla_accuracy_language_prior_debiased",
                "mmbench.vanilla",
            ),
            "seed_bench_image": metric(
                benchmark_metrics["seed_bench_image"],
                "accuracy_language_prior_debiased",
                "seed_bench_image",
            ),
            "sugarcrepe": metric(
                sugarcrepe["language_prior_debiased_pairwise"],
                "win_rate",
                "sugarcrepe",
            ),
            "aro_vg_relation": metric(
                benchmark_metrics["aro_vg_relation"][
                    "language_prior_debiased_pairwise"
                ],
                "win_rate",
                "aro_vg_relation",
            ),
            "aro_vg_attribution": metric(
                benchmark_metrics["aro_vg_attribution"][
                    "language_prior_debiased_pairwise"
                ],
                "win_rate",
                "aro_vg_attribution",
            ),
        },
        "sugarcrepe_categories": {
            name: metric(
                values["language_prior_debiased_pairwise"],
                "win_rate",
                f"sugar.{name}",
            )
            for name, values in sugarcrepe["categories"].items()
        },
        "pure_text_primary_metrics": {
            task: finite(value, f"pure_text.{task}")
            for task, value in summary["pure_text"]["primary_metrics"].items()
        },
    }


def atomic_write(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, delete=False
        ) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Unified 0.6B selected-protocol checkpoint trend",
        "",
        "Image training uses ImageNet train. ImageNet zero-shot classification uses "
        "all 50K official validation images; classification and retrieval use fixed "
        "alpha-1 language-prior-debiased mean-token likelihood. Runtime hashing is "
        "disabled.",
        "",
        "## Generation, held-out validation, and text",
        "",
        "| step | FID ↓ | IS ↑ | val loss ↓ | text macro ↑ (%) |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['global_step']} | {row['fid']:.6f} | "
            f"{row['inception_score_mean']:.6f} ± {row['inception_score_std']:.6f} | "
            f"{row['validation_loss']:.6f} | "
            f"{100.0 * row['pure_text_macro']:.2f} |"
        )

    lines.extend(
        [
            "",
            "## ImageNet-1K zero-shot classification (50K validation images)",
            "",
            "| step | Top-1 accuracy ↑ (%) | Top-5 accuracy ↑ (%) |",
            "| ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        values = row["imagenet1k_zeroshot"]
        lines.append(
            f"| {row['global_step']} | {100.0 * values['top_1_accuracy']:.2f} | "
            f"{100.0 * values['top_5_accuracy']:.2f} |"
        )

    for task, label in (
        ("mscoco_karpathy_test_5k", "MSCOCO Karpathy 5K test"),
        ("flickr30k_karpathy_test_1k", "Flickr30K Karpathy test"),
    ):
        lines.extend(
            [
                "",
                f"## {label} retrieval",
                "",
                "| step | I2T R@1 ↑ (%) | I2T R@5 ↑ (%) | I2T R@10 ↑ (%) | T2I R@1 ↑ (%) | T2I R@5 ↑ (%) | T2I R@10 ↑ (%) |",
                "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in rows:
            values = row["standard_retrieval"][task]
            lines.append(
                f"| {row['global_step']} | {100.0 * values['i2t_r1']:.2f} | "
                f"{100.0 * values['i2t_r5']:.2f} | "
                f"{100.0 * values['i2t_r10']:.2f} | "
                f"{100.0 * values['t2i_r1']:.2f} | "
                f"{100.0 * values['t2i_r5']:.2f} | "
                f"{100.0 * values['t2i_r10']:.2f} |"
            )

    lines.extend(
        [
            "",
            "## Compositional benchmarks and internal ablation diagnostics",
            "",
            "SugarCrepe and ARO are paper-facing hard-negative/compositional "
            "benchmarks. MMBench and SEED are internal ablation-trend diagnostics "
            "only; their candidate-likelihood values are not paper-table or "
            "leaderboard-comparable free-form scores.",
            "",
            "| step | MMBench circular ↑ (%) | MMBench vanilla ↑ (%) | SEED ↑ (%) | SugarCrepe ↑ (%) | ARO relation ↑ (%) | ARO attribution ↑ (%) |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        values = row["benchmarks"]
        lines.append(
            f"| {row['global_step']} | {100.0 * values['mmbench_dev_en']:.2f} | "
            f"{100.0 * values['mmbench_dev_en_vanilla']:.2f} | "
            f"{100.0 * values['seed_bench_image']:.2f} | "
            f"{100.0 * values['sugarcrepe']:.2f} | "
            f"{100.0 * values['aro_vg_relation']:.2f} | "
            f"{100.0 * values['aro_vg_attribution']:.2f} |"
        )
    lines.extend(
        [
            "",
            "Removed from the selected protocol: custom ImageNet 1K/5K retrieval, "
            "ImageNet-ReaL, custom ImageNet caption negatives, uncalibrated retrieval "
            "scores, POPE, COCO Caption PPL, Winoground, SVO-Probes, "
            "and What’sUp.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    rows = [trend_row(root) for root in args.evaluation_root]
    rows.sort(key=lambda row: row["global_step"])
    steps = [row["global_step"] for row in rows]
    if len(steps) != len(set(steps)):
        raise ValueError(f"duplicate checkpoint steps: {steps}")
    payload = {
        "schema": "unified_native_checkpoint_trend_v4",
        "complete": True,
        "runtime_hashing_enabled": False,
        "dataset_contract": {
            "training_split": "imagenet_train",
            "evaluation_split": "imagenet_val",
            "train_validation_overlap_allowed": False,
        },
        "steps": steps,
        "rows": rows,
    }
    atomic_write(
        args.output_dir / "native_trend.json",
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )
    atomic_write(args.output_dir / "native_trend.md", markdown(rows))
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
