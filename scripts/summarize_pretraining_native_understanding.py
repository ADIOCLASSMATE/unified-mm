#!/usr/bin/env python3
"""Merge paper-facing understanding metrics and internal trend diagnostics."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from utils.evaluation_model_source import resolve_evaluation_model_source


REQUIRED_NATIVE_TASKS = ("retrieval_1k", "retrieval_5k")
STANDARD_RETRIEVAL_TASKS = (
    "mscoco_karpathy_test_5k",
    "flickr30k_karpathy_test_1k",
)
PAPER_BENCHMARK_TASKS = (
    "sugarcrepe",
    "aro_vg_relation",
    "aro_vg_attribution",
)
INTERNAL_ABLATION_TASKS = ("mmbench_dev_en", "seed_bench_image")
REMOVED_BENCHMARK_TASKS = {
    "pope_coco": "chance-level accuracy and degenerate almost-always-no predictions",
    "coco_caption_ppl": "no chance baseline or matched reference model for interpretation",
    "winoground": "official gated assets were unavailable",
    "svo_probes": "official image assets were unavailable",
    "whatsup_controlled": "official controlled-set assets were unavailable",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--imagenet_retrieval_root", type=Path, required=True)
    parser.add_argument("--coco_retrieval_root", type=Path, required=True)
    parser.add_argument("--flickr30k_retrieval_root", type=Path, required=True)
    parser.add_argument("--benchmark_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def atomic_write_text(path: Path, payload: str) -> None:
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


def nested_primary(metrics: dict[str, Any]) -> float:
    value: Any = metrics
    for component in str(metrics["primary_metric"]).split("."):
        value = value[component]
    return float(value)


def normalize_native_task(task: str, raw_summary: dict[str, Any]) -> dict[str, Any]:
    """Validate the single retained normalized-likelihood score variant."""

    summary = deepcopy(raw_summary)
    if not isinstance(summary.get("normalized_loglikelihood"), dict):
        raise ValueError(f"{task} has no normalized likelihood metrics")
    summary["primary_metric"] = (
        "normalized_loglikelihood.mean_bidirectional_instance_recall_at_1"
    )
    return summary


def compact_benchmark_task(task: str, raw_summary: dict[str, Any]) -> dict[str, Any]:
    """Keep paper-facing metrics compact while preserving detailed source files."""

    summary = deepcopy(raw_summary)
    if task in {"aro_vg_relation", "aro_vg_attribution"}:
        summary["metrics"].pop("categories", None)
    return summary


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    source = resolve_evaluation_model_source(checkpoint)
    step = source.global_step

    native_manifest_path = args.imagenet_retrieval_root / "manifest.json"
    native_summary_path = args.imagenet_retrieval_root / "summary.json"
    benchmark_manifest_path = args.benchmark_root / "manifest.json"
    benchmark_summary_path = args.benchmark_root / "summary.json"
    native_manifest = read_json(native_manifest_path)
    native_summary = read_json(native_summary_path)
    benchmark_manifest = read_json(benchmark_manifest_path)
    benchmark_summary = read_json(benchmark_summary_path)
    standard_roots = {
        "mscoco_karpathy_test_5k": args.coco_retrieval_root,
        "flickr30k_karpathy_test_1k": args.flickr30k_retrieval_root,
    }
    standard_manifests = {
        task: read_json(root / "manifest.json")
        for task, root in standard_roots.items()
    }
    standard_summaries = {
        task: read_json(root / "summary.json")
        for task, root in standard_roots.items()
    }

    for payload in (
        native_manifest,
        native_summary,
        benchmark_manifest,
        benchmark_summary,
        *standard_manifests.values(),
        *standard_summaries.values(),
    ):
        if payload.get("runtime_hashing_enabled", True) is not False:
            raise ValueError("understanding component violates the no-hash contract")
    if native_manifest.get("complete") is not True:
        raise ValueError("ImageNet retrieval evaluation is incomplete")
    if benchmark_manifest.get("complete") is not True:
        raise ValueError("retained external benchmark evaluation is incomplete")
    if Path(native_manifest["checkpoint"]).resolve() != checkpoint:
        raise ValueError("ImageNet retrieval checkpoint mismatch")
    if Path(benchmark_manifest["checkpoint"]).resolve() != checkpoint:
        raise ValueError("external benchmark checkpoint mismatch")
    for name, payload in (
        ("imagenet", native_manifest),
        ("benchmark", benchmark_manifest),
        *standard_manifests.items(),
    ):
        weight_source = payload.get("weight_source")
        if source.is_hf_final_ema and weight_source != source.kind:
            raise ValueError(f"{name} component lacks final-HF source identity")
        if weight_source is not None and weight_source != source.kind:
            raise ValueError(f"{name} component weight source mismatch")
    for task in STANDARD_RETRIEVAL_TASKS:
        manifest = standard_manifests[task]
        summary = standard_summaries[task]
        if manifest.get("complete") is not True:
            raise ValueError(f"standard retrieval evaluation is incomplete: {task}")
        if str(manifest.get("task")) != task or str(summary.get("task")) != task:
            raise ValueError(f"standard retrieval task mismatch: {task}")
        if Path(manifest["checkpoint"]).resolve() != checkpoint:
            raise ValueError(f"standard retrieval checkpoint mismatch: {task}")
        if int(summary["checkpoint_step"]) != step:
            raise ValueError(f"standard retrieval checkpoint step mismatch: {task}")
    if int(native_summary["checkpoint_step"]) != step or int(
        benchmark_summary["checkpoint_step"]
    ) != step:
        raise ValueError("understanding component checkpoint steps disagree")

    source_native_tasks = native_summary["tasks"]
    source_benchmarks = benchmark_summary["tasks"]
    missing = sorted(
        (set(REQUIRED_NATIVE_TASKS) - set(source_native_tasks))
        | (set(PAPER_BENCHMARK_TASKS) - set(source_benchmarks))
    )
    missing_internal = sorted(set(INTERNAL_ABLATION_TASKS) - set(source_benchmarks))
    native_tasks = {
        task: normalize_native_task(task, source_native_tasks[task])
        for task in REQUIRED_NATIVE_TASKS
        if task in source_native_tasks
    }
    standard_retrieval = {
        task: deepcopy(standard_summaries[task])
        for task in STANDARD_RETRIEVAL_TASKS
    }
    paper_benchmarks = {
        task: compact_benchmark_task(task, source_benchmarks[task])
        for task in PAPER_BENCHMARK_TASKS
        if task in source_benchmarks
    }
    internal_diagnostics = {
        task: compact_benchmark_task(task, source_benchmarks[task])
        for task in INTERNAL_ABLATION_TASKS
        if task in source_benchmarks
    }
    primary = {
        task: nested_primary(summary)
        for task, summary in native_tasks.items()
    }
    primary.update(
        {
            task: nested_primary(summary["metrics"])
            for task, summary in paper_benchmarks.items()
        }
    )
    primary.update(
        {
            task: nested_primary(summary)
            for task, summary in standard_retrieval.items()
        }
    )

    report = {
        "schema": "pretraining_native_understanding_summary_v3",
        "complete": not missing,
        "runtime_hashing_enabled": False,
        "checkpoint": str(checkpoint),
        "global_step": step,
        "weight_source": source.kind,
        "model_source": source.report(),
        "adaptation": {
            "downstream_finetuning": False,
            "instruction_tuning": False,
            "linear_probe": False,
        },
        "dataset_contract": {
            "image_training_split": "imagenet_train",
            "image_evaluation_split": "imagenet_val",
            "standard_retrieval_evaluation_splits": list(
                STANDARD_RETRIEVAL_TASKS
            ),
            "train_validation_overlap_allowed": False,
            "external_benchmarks_are_out_of_domain": True,
        },
        "primary_metrics": primary,
        "retrieval_1k": native_tasks.get("retrieval_1k"),
        "retrieval_5k": native_tasks.get("retrieval_5k"),
        "standard_cross_dataset_retrieval": standard_retrieval,
        "paper_compositional_benchmarks": paper_benchmarks,
        "internal_ablation_diagnostics": internal_diagnostics,
        "selection_contract": {
            "imagenet_native_metrics_retained_by_design": [
                "retrieval_1k_i2t_t2i_recall_at_1_5_10",
                "retrieval_5k_i2t_t2i_recall_at_1_5_10",
            ],
            "official_compositional_benchmarks": [
                "sugarcrepe",
                "aro_vg_relation",
                "aro_vg_attribution",
            ],
            "standard_retrieval_benchmarks": list(STANDARD_RETRIEVAL_TASKS),
            "paper_table_excluded_internal_ablation_diagnostics": [
                "mmbench_dev_en",
                "seed_bench_image",
            ],
            "removed_custom_caption_negative_protocols": [
                "imagenet_caption_random_negative",
                "imagenet_caption_same_class_negative",
                "imagenet_caption_near_class_negative",
            ],
            "visual_calibration_enabled": False,
            "imagenet_classification_enabled": False,
            "removed_benchmarks": REMOVED_BENCHMARK_TASKS,
        },
        "coverage": {
            "required_tasks": list(REQUIRED_NATIVE_TASKS)
            + list(STANDARD_RETRIEVAL_TASKS)
            + list(PAPER_BENCHMARK_TASKS),
            "paper_tasks": list(REQUIRED_NATIVE_TASKS)
            + list(STANDARD_RETRIEVAL_TASKS)
            + list(PAPER_BENCHMARK_TASKS),
            "internal_ablation_tasks": list(INTERNAL_ABLATION_TASKS),
            "missing_required_tasks": missing,
            "missing_optional_internal_ablation_tasks": missing_internal,
        },
        "components": {
            "imagenet_retrieval_manifest": str(native_manifest_path.resolve()),
            "imagenet_retrieval_summary": str(native_summary_path.resolve()),
            "standard_retrieval": {
                task: {
                    "manifest": str((root / "manifest.json").resolve()),
                    "summary": str((root / "summary.json").resolve()),
                }
                for task, root in standard_roots.items()
            },
            "benchmark_manifest": str(benchmark_manifest_path.resolve()),
            "benchmark_summary": str(benchmark_summary_path.resolve()),
            "benchmark_task_summaries": {
                task: str((args.benchmark_root / "summaries" / f"{task}.json").resolve())
                for task in PAPER_BENCHMARK_TASKS + INTERNAL_ABLATION_TASKS
            },
        },
        "completed_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    destination = args.output_dir / "pretraining_native_understanding_summary.json"
    atomic_write_text(
        destination,
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
