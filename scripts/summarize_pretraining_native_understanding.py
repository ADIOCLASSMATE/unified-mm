#!/usr/bin/env python3
"""Merge paper-facing understanding metrics and internal trend diagnostics."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

from utils.evaluation_model_source import resolve_evaluation_model_source


IMAGENET_CLASSIFICATION_TASK = "imagenet1k_zeroshot_classification"
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
FORMAL_RETRIEVAL_CARDINALITY = {
    "mscoco_karpathy_test_5k": {
        "images": 5_000,
        "captions": 25_010,
        "caption_count_distribution": {"5": 4_990, "6": 10},
    },
    "flickr30k_karpathy_test_1k": {
        "images": 1_000,
        "captions": 5_000,
        "caption_count_distribution": {"5": 1_000},
    },
}
FORMAL_BENCHMARK_RECORDS = {
    "mmbench_dev_en": 4_329,
    "seed_bench_image": 14_233,
    "sugarcrepe": 7_511,
    "aro_vg_relation": 23_937,
    "aro_vg_attribution": 28_748,
}
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
    parser.add_argument("--imagenet_classification_root", type=Path, required=True)
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


def require_unit_interval(value: Any, label: str) -> float:
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{label} must be finite and in [0, 1], got {number}")
    return number


def require_imagenet_metric_contract(summary: dict[str, Any]) -> None:
    if summary.get("complete_formal_target") is not True:
        raise ValueError("ImageNet classification does not cover the formal target")
    expected = {
        "records": 50_000,
        "classes": 1_000,
        "formal_target_records": 50_000,
        "language_prior_image_count": 50_000,
    }
    for key, value in expected.items():
        if int(summary.get(key, -1)) != value:
            raise ValueError(f"ImageNet classification has invalid {key}")
    if summary.get("primary_metric") != "top_1_accuracy":
        raise ValueError("ImageNet classification has an unexpected primary metric")
    if summary.get("accuracy_unit") != "unit_interval":
        raise ValueError("ImageNet classification has an ambiguous accuracy unit")
    if summary.get("class_text_template") != "a photo of a {class_name}.":
        raise ValueError("ImageNet classification uses an unexpected template")
    top1 = require_unit_interval(summary.get("top_1_accuracy"), "ImageNet Top-1")
    top5 = require_unit_interval(summary.get("top_5_accuracy"), "ImageNet Top-5")
    if top5 < top1:
        raise ValueError("ImageNet Top-5 accuracy cannot be below Top-1")


def require_retrieval_metric_contract(
    summary: dict[str, Any], task: str
) -> None:
    contract = FORMAL_RETRIEVAL_CARDINALITY[task]
    if summary.get("complete_formal_target") is not True:
        raise ValueError(f"standard retrieval target is incomplete: {task}")
    for key in ("images", "captions"):
        if int(summary.get(key, -1)) != int(contract[key]):
            raise ValueError(f"standard retrieval {task} has invalid {key}")
    if summary.get("caption_count_distribution") != contract[
        "caption_count_distribution"
    ]:
        raise ValueError(f"standard retrieval {task} has invalid caption counts")
    if summary.get("primary_metric") != "mean_recall_at_1_5_10":
        raise ValueError(f"standard retrieval {task} has an unexpected primary metric")
    if summary.get("recall_unit") != "unit_interval":
        raise ValueError(f"standard retrieval {task} has an ambiguous recall unit")
    if summary.get("rank_unit") != "one_based_candidate_rank":
        raise ValueError(f"standard retrieval {task} has an ambiguous rank unit")
    if summary.get("coco_five_fold_1k_average") is not False:
        raise ValueError(f"standard retrieval {task} is not the full candidate run")

    images = int(contract["images"])
    captions = int(contract["captions"])
    for direction, queries, candidates in (
        ("image_to_text", images, captions),
        ("text_to_image", captions, images),
    ):
        values = summary.get(direction) or {}
        if int(values.get("queries", -1)) != queries:
            raise ValueError(f"{task} {direction} has invalid query count")
        if int(values.get("candidates", -1)) != candidates:
            raise ValueError(f"{task} {direction} has invalid candidate count")
        recalls = [
            require_unit_interval(
                values.get(f"recall_at_{k}"), f"{task} {direction} R@{k}"
            )
            for k in (1, 5, 10)
        ]
        if recalls != sorted(recalls):
            raise ValueError(f"{task} {direction} recall must be monotone in K")
        expected_mean = sum(recalls) / 3.0
        actual_mean = require_unit_interval(
            values.get("mean_recall_at_1_5_10"),
            f"{task} {direction} mean recall",
        )
        if not math.isclose(actual_mean, expected_mean, rel_tol=0.0, abs_tol=1.0e-7):
            raise ValueError(f"{task} {direction} mean recall is inconsistent")
        for rank_name in ("mean_rank", "median_rank"):
            rank_value = float(values.get(rank_name, math.nan))
            if not math.isfinite(rank_value) or not 1.0 <= rank_value <= candidates:
                raise ValueError(f"{task} {direction} has invalid {rank_name}")

    expected_overall = sum(
        float(summary[direction]["mean_recall_at_1_5_10"])
        for direction in ("image_to_text", "text_to_image")
    ) / 2.0
    overall = require_unit_interval(
        summary.get("mean_recall_at_1_5_10"), f"{task} overall mean recall"
    )
    if not math.isclose(overall, expected_overall, rel_tol=0.0, abs_tol=1.0e-7):
        raise ValueError(f"{task} overall mean recall is inconsistent")


def require_benchmark_metric_contract(
    task: str, summary: dict[str, Any]
) -> None:
    metrics = summary.get("metrics") or {}
    if int(metrics.get("records", -1)) != FORMAL_BENCHMARK_RECORDS[task]:
        raise ValueError(f"retained benchmark {task} has invalid record count")
    expected_primary = (
        "language_prior_debiased_pairwise.win_rate"
        if task in PAPER_BENCHMARK_TASKS
        else (
            "circular_accuracy_language_prior_debiased"
            if task == "mmbench_dev_en"
            else "accuracy_language_prior_debiased"
        )
    )
    if metrics.get("primary_metric") != expected_primary:
        raise ValueError(f"retained benchmark {task} has an unexpected primary metric")
    require_unit_interval(nested_primary(metrics), f"retained benchmark {task}")


def require_calibrated_protocol(payload: dict[str, Any], label: str) -> None:
    scoring = payload.get("scoring") or {}
    if scoring.get("primary_candidate_score") != (
        "language_prior_debiased_mean_token_loglikelihood"
    ):
        raise ValueError(f"{label} does not use the required debiased score")
    if float(scoring.get("language_prior_alpha", -1.0)) != 1.0:
        raise ValueError(f"{label} does not fix language-prior alpha to one")
    if scoring.get("language_prior_estimator") != "candidate_image_logmeanexp":
        raise ValueError(f"{label} uses the wrong language-prior estimator")


def require_null_image_calibrated_protocol(
    payload: dict[str, Any], label: str
) -> None:
    scoring = payload.get("scoring") or payload
    if scoring.get("primary_candidate_score") != (
        "language_prior_debiased_mean_token_loglikelihood"
    ):
        raise ValueError(f"{label} does not use the required debiased score")
    if scoring.get("reported_score_variant") != "language_prior_debiased_only":
        raise ValueError(f"{label} retains a removed score variant")
    if float(scoring.get("language_prior_alpha", -1.0)) != 1.0:
        raise ValueError(f"{label} does not fix language-prior alpha to one")
    if scoring.get("language_prior_estimator") != (
        "content_free_gaussian_image_logmeanexp"
    ):
        raise ValueError(f"{label} uses the wrong language-prior estimator")
    if int(scoring.get("language_prior_null_image_count", -1)) != 3:
        raise ValueError(f"{label} does not use exactly three null images")
    if bool(scoring.get("language_prior_uses_labels", True)):
        raise ValueError(f"{label} language-prior estimation uses labels")


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

    native_manifest_path = args.imagenet_classification_root / "manifest.json"
    native_summary_path = args.imagenet_classification_root / "summary.json"
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
        raise ValueError("ImageNet classification evaluation is incomplete")
    if native_manifest.get("project_formal_protocol") is not True:
        raise ValueError("ImageNet classification is not a formal 50K run")
    if native_manifest.get("schema") != (
        "selfless_imagenet1k_zeroshot_classification_evaluation_v2"
    ):
        raise ValueError("obsolete ImageNet evaluation protocol")
    if native_summary.get("schema") != (
        "selfless_imagenet1k_zeroshot_classification_summary_v2"
    ):
        raise ValueError("obsolete ImageNet summary protocol")
    if native_summary.get("task") != IMAGENET_CLASSIFICATION_TASK:
        raise ValueError("ImageNet classification task mismatch")
    if native_summary.get("project_formal_protocol") is not True:
        raise ValueError("ImageNet classification summary is not a formal 50K run")
    require_calibrated_protocol(native_summary, "ImageNet classification")
    require_imagenet_metric_contract(native_summary)
    if benchmark_manifest.get("complete") is not True:
        raise ValueError("retained external benchmark evaluation is incomplete")
    if benchmark_manifest.get("schema") != (
        "selfless_multimodal_likelihood_evaluation_v5"
    ):
        raise ValueError("obsolete retained benchmark evaluation protocol")
    if benchmark_summary.get("schema") != (
        "selfless_multimodal_likelihood_summary_v5"
    ):
        raise ValueError("obsolete retained benchmark summary protocol")
    require_null_image_calibrated_protocol(
        benchmark_manifest, "retained external benchmark manifest"
    )
    require_null_image_calibrated_protocol(
        benchmark_summary, "retained external benchmark summary"
    )
    if benchmark_manifest.get("project_formal_protocol") is not True:
        raise ValueError("retained benchmark evaluation is not a formal MC64 run")
    if benchmark_summary.get("project_formal_protocol") is not True:
        raise ValueError("retained benchmark summary is not a formal MC64 run")
    if int(benchmark_manifest.get("mc_samples", -1)) != 64:
        raise ValueError("retained benchmark evaluation does not use MC64")
    if int(benchmark_summary.get("scoring", {}).get("mc_samples", -1)) != 64:
        raise ValueError("retained benchmark summary does not use MC64")
    if Path(native_manifest["checkpoint"]).resolve() != checkpoint:
        raise ValueError("ImageNet classification checkpoint mismatch")
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
        if manifest.get("schema") != "selfless_cross_dataset_retrieval_evaluation_v3":
            raise ValueError(f"obsolete standard retrieval protocol: {task}")
        if summary.get("schema") != "selfless_cross_dataset_retrieval_summary_v3":
            raise ValueError(f"obsolete standard retrieval summary: {task}")
        if manifest.get("complete") is not True:
            raise ValueError(f"standard retrieval evaluation is incomplete: {task}")
        if manifest.get("project_formal_protocol") is not True:
            raise ValueError(f"standard retrieval is not a formal run: {task}")
        if summary.get("project_formal_protocol") is not True:
            raise ValueError(f"standard retrieval summary is not formal: {task}")
        if str(manifest.get("task")) != task or str(summary.get("task")) != task:
            raise ValueError(f"standard retrieval task mismatch: {task}")
        if Path(manifest["checkpoint"]).resolve() != checkpoint:
            raise ValueError(f"standard retrieval checkpoint mismatch: {task}")
        if int(summary["checkpoint_step"]) != step:
            raise ValueError(f"standard retrieval checkpoint step mismatch: {task}")
        require_calibrated_protocol(summary, f"standard retrieval {task}")
        require_retrieval_metric_contract(summary, task)
    if int(native_summary["checkpoint_step"]) != step or int(
        benchmark_summary["checkpoint_step"]
    ) != step:
        raise ValueError("understanding component checkpoint steps disagree")

    source_benchmarks = benchmark_summary["tasks"]
    missing = sorted(set(PAPER_BENCHMARK_TASKS) - set(source_benchmarks))
    missing_internal = sorted(set(INTERNAL_ABLATION_TASKS) - set(source_benchmarks))
    expected_benchmark_tasks = set(PAPER_BENCHMARK_TASKS + INTERNAL_ABLATION_TASKS)
    if set(source_benchmarks) != expected_benchmark_tasks:
        raise ValueError("retained benchmark summary has an unexpected task set")
    manifest_records = benchmark_manifest.get("records") or {}
    if {
        task: int(manifest_records.get(task, -1))
        for task in FORMAL_BENCHMARK_RECORDS
    } != FORMAL_BENCHMARK_RECORDS:
        raise ValueError("retained benchmark manifest has invalid record counts")
    if benchmark_summary.get("accuracy_and_rate_unit") != "unit_interval":
        raise ValueError("retained benchmark summary has an ambiguous accuracy unit")
    for task, task_summary in source_benchmarks.items():
        require_benchmark_metric_contract(task, task_summary)
    imagenet_classification = deepcopy(native_summary)
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
        "imagenet1k_zeroshot_top_1_accuracy": float(
            imagenet_classification["top_1_accuracy"]
        ),
        "imagenet1k_zeroshot_top_5_accuracy": float(
            imagenet_classification["top_5_accuracy"]
        ),
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
        "schema": "pretraining_native_understanding_summary_v5",
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
        "imagenet1k_zeroshot_classification": imagenet_classification,
        "standard_cross_dataset_retrieval": standard_retrieval,
        "paper_compositional_benchmarks": paper_benchmarks,
        "internal_ablation_diagnostics": internal_diagnostics,
        "selection_contract": {
            "imagenet_zero_shot_metrics": ["top_1_accuracy", "top_5_accuracy"],
            "imagenet_validation_images": 50_000,
            "image_text_matching_score_variant": (
                "language_prior_debiased_mean_token_loglikelihood_only"
            ),
            "language_prior_alpha": 1.0,
            "dense_retrieval_language_prior_estimator": (
                "candidate_image_logmeanexp"
            ),
            "hard_negative_language_prior_estimator": (
                "content_free_gaussian_image_logmeanexp"
            ),
            "hard_negative_language_prior_null_images": 3,
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
            "removed_custom_imagenet_retrieval_protocols": [
                "retrieval_1k",
                "retrieval_5k",
            ],
            "imagenet_classification_enabled": True,
            "removed_benchmarks": REMOVED_BENCHMARK_TASKS,
        },
        "coverage": {
            "required_tasks": [IMAGENET_CLASSIFICATION_TASK]
            + list(STANDARD_RETRIEVAL_TASKS)
            + list(PAPER_BENCHMARK_TASKS),
            "paper_tasks": [IMAGENET_CLASSIFICATION_TASK]
            + list(STANDARD_RETRIEVAL_TASKS)
            + list(PAPER_BENCHMARK_TASKS),
            "internal_ablation_tasks": list(INTERNAL_ABLATION_TASKS),
            "missing_required_tasks": missing,
            "missing_optional_internal_ablation_tasks": missing_internal,
        },
        "components": {
            "imagenet_classification_manifest": str(native_manifest_path.resolve()),
            "imagenet_classification_summary": str(native_summary_path.resolve()),
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
