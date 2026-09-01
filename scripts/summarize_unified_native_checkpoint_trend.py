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


def retrieval_row(summary: dict[str, Any], label: str) -> dict[str, float]:
    values = summary["normalized_loglikelihood"]
    row: dict[str, float] = {}
    for direction, prefix in (("image_to_text", "i2t"), ("text_to_image", "t2i")):
        direction_metrics = values[direction]
        for k in (1, 5, 10):
            name = f"instance_recall_at_{k}"
            row[f"{prefix}_r{k}"] = metric(
                direction_metrics, name, f"{label}.{prefix}.r{k}"
            )
    return row


def standard_retrieval_row(
    summary: dict[str, Any], label: str
) -> dict[str, float]:
    values = summary["normalized_loglikelihood"]
    row: dict[str, float] = {}
    for direction, prefix in (("image_to_text", "i2t"), ("text_to_image", "t2i")):
        direction_metrics = values[direction]
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
    contract = summary.get("dataset_contract", {})
    if contract.get("training_split") != "imagenet_train":
        raise ValueError(f"training split is not ImageNet train: {path}")
    if contract.get("evaluation_split") != "imagenet_val":
        raise ValueError(f"evaluation split is not ImageNet val: {path}")
    if contract.get("train_validation_overlap_allowed") is not False:
        raise ValueError(f"ImageNet train/val overlap is permitted: {path}")

    generation = summary["generation"]["imagenet_val_t2i"]
    if generation.get("official_protocol") is not True:
        raise ValueError(f"generation result is not official: {path}")
    if int(generation.get("samples", 0)) != 50_000:
        raise ValueError(f"generation result is not 50K: {path}")
    if generation.get("is_split_assignment") != "stratified_by_synset":
        raise ValueError(f"generation IS split is not synset-stratified: {path}")

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
    return {
        "global_step": int(summary["global_step"]),
        "checkpoint": str(summary["checkpoint"]),
        "evaluation_root": str(root.resolve()),
        "fid": finite(generation["fid"], "fid"),
        "inception_score_mean": finite(
            generation["inception_score_mean"], "inception_score_mean"
        ),
        "inception_score_std": finite(
            generation["inception_score_std"], "inception_score_std"
        ),
        "validation_loss": finite(validation["val/loss"], "validation_loss"),
        "pure_text_macro": finite(
            summary["pure_text"]["macro_average_primary"], "pure_text_macro"
        ),
        "retrieval_1k": retrieval_row(native["retrieval_1k"], "retrieval_1k"),
        "retrieval_5k": retrieval_row(native["retrieval_5k"], "retrieval_5k"),
        "standard_retrieval": {
            task: standard_retrieval_row(result, task)
            for task, result in native["standard_cross_dataset_retrieval"].items()
        },
        "benchmarks": {
            "mmbench_dev_en": metric(
                mmbench,
                "circular_accuracy_normalized_loglikelihood",
                "mmbench.circular",
            ),
            "mmbench_dev_en_vanilla": metric(
                mmbench,
                "vanilla_accuracy_normalized_loglikelihood",
                "mmbench.vanilla",
            ),
            "seed_bench_image": metric(
                benchmark_metrics["seed_bench_image"],
                "accuracy_normalized_loglikelihood",
                "seed_bench_image",
            ),
            "sugarcrepe": metric(
                sugarcrepe,
                "accuracy_normalized_loglikelihood",
                "sugarcrepe",
            ),
            "aro_vg_relation": metric(
                benchmark_metrics["aro_vg_relation"],
                "accuracy_normalized_loglikelihood",
                "aro_vg_relation",
            ),
            "aro_vg_attribution": metric(
                benchmark_metrics["aro_vg_attribution"],
                "accuracy_normalized_loglikelihood",
                "aro_vg_attribution",
            ),
        },
        "sugarcrepe_categories": {
            name: metric(values, "accuracy_normalized_loglikelihood", f"sugar.{name}")
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
        "Image training uses ImageNet train. The custom ImageNet retrieval protocol "
        "uses only official ImageNet val images. Scores use normalized same-position "
        "likelihood without visual calibration. Runtime hashing is disabled.",
        "",
        "## Generation, held-out validation, and text",
        "",
        "| step | FID ↓ | IS ↑ | val loss ↓ | text macro ↑ |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['global_step']} | {row['fid']:.6f} | "
            f"{row['inception_score_mean']:.6f} ± {row['inception_score_std']:.6f} | "
            f"{row['validation_loss']:.6f} | "
            f"{row['pure_text_macro']:.6f} |"
        )

    for subset in ("retrieval_1k", "retrieval_5k"):
        label = "1K" if subset.endswith("1k") else "5K"
        lines.extend(
            [
                "",
                f"## ImageNet-val {label} exact-instance retrieval",
                "",
                "| step | I2T R@1 ↑ | I2T R@5 ↑ | I2T R@10 ↑ | T2I R@1 ↑ | T2I R@5 ↑ | T2I R@10 ↑ |",
                "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in rows:
            values = row[subset]
            lines.append(
                f"| {row['global_step']} | {values['i2t_r1']:.6f} | "
                f"{values['i2t_r5']:.6f} | {values['i2t_r10']:.6f} | "
                f"{values['t2i_r1']:.6f} | {values['t2i_r5']:.6f} | "
                f"{values['t2i_r10']:.6f} |"
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
                "| step | I2T R@1 ↑ | I2T R@5 ↑ | I2T R@10 ↑ | T2I R@1 ↑ | T2I R@5 ↑ | T2I R@10 ↑ |",
                "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in rows:
            values = row["standard_retrieval"][task]
            lines.append(
                f"| {row['global_step']} | {values['i2t_r1']:.6f} | "
                f"{values['i2t_r5']:.6f} | {values['i2t_r10']:.6f} | "
                f"{values['t2i_r1']:.6f} | {values['t2i_r5']:.6f} | "
                f"{values['t2i_r10']:.6f} |"
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
            "| step | MMBench circular ↑ | MMBench vanilla ↑ | SEED ↑ | SugarCrepe ↑ | ARO relation ↑ | ARO attribution ↑ |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        values = row["benchmarks"]
        lines.append(
            f"| {row['global_step']} | {values['mmbench_dev_en']:.6f} | "
            f"{values['mmbench_dev_en_vanilla']:.6f} | "
            f"{values['seed_bench_image']:.6f} | {values['sugarcrepe']:.6f} | "
            f"{values['aro_vg_relation']:.6f} | "
            f"{values['aro_vg_attribution']:.6f} |"
        )
    lines.extend(
        [
            "",
            "Removed from the selected protocol: ImageNet classification/ReaL, "
            "custom ImageNet caption negatives, "
            "visual calibration, POPE, COCO Caption PPL, Winoground, SVO-Probes, "
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
        "schema": "unified_native_checkpoint_trend_v2",
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
