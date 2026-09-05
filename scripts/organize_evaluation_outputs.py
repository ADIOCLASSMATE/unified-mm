#!/usr/bin/env python3
"""Build the compact, protocol-selected evaluation archive.

This is intentionally non-destructive: it creates a validated archive using
hard links where possible.  Source cleanup is a separate explicit operation.
No content hashes are calculated or stored.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterable


STEPS = (56_000, 58_000, 60_000)
PAPER_BENCHMARKS = ("sugarcrepe", "aro_vg_relation", "aro_vg_attribution")
INTERNAL_BENCHMARKS = ("mmbench_dev_en", "seed_bench_image")
EXPECTED_BENCHMARK_RECORDS = {
    "sugarcrepe": 7_511,
    "aro_vg_relation": 23_937,
    "aro_vg_attribution": 28_748,
    "mmbench_dev_en": 4_329,
    "seed_bench_image": 14_233,
}
STANDARD_RETRIEVAL = {
    "mscoco_karpathy_test_5k": {"images": 5_000, "captions": 25_010},
    "flickr30k_karpathy_test_1k": {"images": 1_000, "captions": 5_000},
}
STANDARD_RETRIEVAL_STEP = 60_000
TEXT_TASKS = (
    "arc_easy",
    "arc_challenge",
    "hellaswag",
    "piqa",
    "winogrande",
    "boolq",
    "openbookqa",
    "mmlu",
)
TEXT_TASK_CONTRACTS = {
    "arc_easy": (2_376, "accuracy_normalized"),
    "arc_challenge": (1_172, "accuracy_normalized"),
    "hellaswag": (10_042, "accuracy_normalized"),
    "piqa": (1_838, "accuracy_normalized"),
    "winogrande": (1_267, "accuracy"),
    "boolq": (3_270, "accuracy"),
    "openbookqa": (500, "accuracy_normalized"),
    "mmlu": (14_042, "accuracy_macro"),
}
LM_EVAL_REFERENCE_COMMIT = "b954108c9baaaa934b4ad842033b31a97ee30816"
GENERATION_PROTOCOL_NAME = (
    "imagenet_val_fid50k_torch_fidelity_stratified_is"
)
GENERATION_REFERENCE_DISTRIBUTION = "imagenet_val_50000"
GENERATION_COMPARISON_SCOPE = "same_protocol_only"
GENERATION_NON_COMPARABILITY_REASON = (
    "validation_reference_and_pytorch_torch_fidelity_extractor"
)
GENERATION_FID_REDUCER = "symmetric_eigendecomposition"
GENERATION_IS_SPLIT_ASSIGNMENT = "stratified_by_synset"
REMOVED_EVALUATIONS = (
    "imagenet_real_top1_top5",
    "custom_imagenet_caption_negatives",
    "custom_imagenet_retrieval_1k_5k",
    "pope_coco",
    "coco_caption_ppl",
    "imagenet_i2t_clip",
    "winoground",
    "svo_probes",
    "whats_up",
    "rank_shards",
    "fid_resume_state",
    "duplicate_logs",
    "uncalibrated_image_text_likelihood",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_root", type=Path, default=Path("output"))
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path("output/evaluation/unified-a-0p6b"),
    )
    parser.add_argument(
        "--latest_core_root",
        type=Path,
        help="Fresh complete core-evaluation root for step 60000.",
    )
    parser.add_argument(
        "--latest_native_root",
        type=Path,
        help="Fresh native-understanding component root for step 60000.",
    )
    parser.add_argument(
        "--coco_retrieval_root",
        type=Path,
        help="Complete step-60000 MSCOCO Karpathy test result root.",
    )
    parser.add_argument(
        "--flickr30k_retrieval_root",
        type=Path,
        help="Complete step-60000 Flickr30K Karpathy test result root.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def link_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def link_tree(source: Path, destination: Path) -> None:
    if not source.is_dir():
        raise FileNotFoundError(source)
    for path in sorted(source.rglob("*")):
        if path.is_file():
            link_file(path, destination / path.relative_to(source))


def core_root(
    output_root: Path, step: int, latest_core_root: Path | None = None
) -> Path:
    if step == STANDARD_RETRIEVAL_STEP and latest_core_root is not None:
        return latest_core_root
    if step == 58_000:
        return output_root / "unified-a-0p6b-full-eval-step58000-ascend16-r1"
    return (
        output_root
        / f"unified-a-0p6b-comprehensive-eval-step{step}-ascend16-r1"
        / "core"
    )


def benchmark_root(
    output_root: Path, step: int, latest_native_root: Path | None = None
) -> Path:
    if step == STANDARD_RETRIEVAL_STEP and latest_native_root is not None:
        return latest_native_root / "retained-benchmarks"
    return (
        output_root
        / f"unified-a-0p6b-comprehensive-eval-step{step}-ascend16-r1"
        / "multimodal-likelihood"
    )


def imagenet_classification_root(
    output_root: Path, step: int, latest_native_root: Path | None = None
) -> Path:
    if step == STANDARD_RETRIEVAL_STEP and latest_native_root is not None:
        return latest_native_root / "imagenet-classification"
    return (
        output_root
        / f"unified-a-0p6b-native-full-eval-step{step}-ascend16-r1"
        / "pretraining-native-understanding"
        / "imagenet-classification"
    )


def validate_no_hash(payload: dict[str, Any], label: str) -> None:
    if payload.get("runtime_hashing_enabled", True) is not False:
        raise ValueError(f"{label} violates the no-hash contract")


def finite(value: Any, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite selected metric: {label}={number}")
    return number


def finite_unit_interval(value: Any, label: str) -> float:
    number = finite(value, label)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"selected metric is outside [0, 1]: {label}={number}")
    return number


def validate_text_summary_for_archive(summary: dict[str, Any]) -> None:
    if summary.get("accuracy_unit") != "unit_interval":
        raise ValueError("pure-text result has an ambiguous accuracy unit")
    if set(summary.get("tasks", {})) != set(TEXT_TASK_CONTRACTS) or set(
        summary.get("primary_metrics", {})
    ) != set(TEXT_TASK_CONTRACTS):
        raise ValueError("pure-text task coverage is incomplete")
    protocol = summary.get("protocol") or {}
    if protocol.get("protocol_schema") != "selfless_text_benchmark_v3":
        raise ValueError("pure-text task protocol is obsolete")
    if protocol.get("normalization") != "original_choice_characters" or (
        protocol.get("winogrande_scoring") != "shared_suffix_given_prefix_and_option"
    ):
        raise ValueError("pure-text scoring protocol is obsolete")
    if (protocol.get("lm_eval_reference") or {}).get("commit") != (
        LM_EVAL_REFERENCE_COMMIT
    ):
        raise ValueError("pure-text lm-eval reference is not frozen")

    primary_values = []
    for task, (samples, metric_name) in TEXT_TASK_CONTRACTS.items():
        metrics = summary["tasks"][task]
        if metrics.get("schema") != "selfless_text_multiple_choice_metrics_v2":
            raise ValueError(f"pure-text task schema is invalid: {task}")
        if metrics.get("protocol_schema") != "selfless_text_benchmark_v3" or (
            metrics.get("normalization") != "original_choice_characters"
        ):
            raise ValueError(f"pure-text task scoring protocol is obsolete: {task}")
        if metrics.get("complete") is not True or metrics.get(
            "runtime_hashing_enabled", True
        ) is not False:
            raise ValueError(f"pure-text task is incomplete or hashed: {task}")
        if metrics.get("task") != task or int(metrics.get("samples", -1)) != samples:
            raise ValueError(f"pure-text task cardinality is invalid: {task}")
        if int(metrics.get("truncated_context_samples", -1)) != 0:
            raise ValueError(f"pure-text task truncated formal prompts: {task}")
        value = finite_unit_interval(metrics.get(metric_name), f"text.{task}")
        if float(summary["primary_metrics"][task]) != value:
            raise ValueError(f"pure-text primary metric mismatch: {task}")
        primary_values.append(value)

    mmlu_metrics = summary["tasks"]["mmlu"]
    mmlu_categories = mmlu_metrics.get("by_category") or {}
    if len(mmlu_categories) != 57:
        raise ValueError("formal MMLU requires all 57 subjects")
    if (
        sum(int(row.get("samples", -1)) for row in mmlu_categories.values())
        != 14_042
    ):
        raise ValueError("MMLU subject sample counts are inconsistent")
    subject_values = [
        finite_unit_interval(row.get("accuracy"), f"text.mmlu.{subject}")
        for subject, row in mmlu_categories.items()
    ]
    if not math.isclose(
        float(mmlu_metrics["accuracy_macro"]),
        sum(subject_values) / len(subject_values),
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("MMLU primary is not the 57-subject macro accuracy")
    if summary.get("macro_average_role") != "internal_cross_task_summary_only":
        raise ValueError("pure-text cross-task macro has the wrong reporting role")
    if not math.isclose(
        finite_unit_interval(summary.get("macro_average_primary"), "text.macro"),
        sum(primary_values) / len(primary_values),
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("pure-text cross-task macro is inconsistent")


def selected_generation(metrics: dict[str, Any]) -> dict[str, Any]:
    validate_no_hash(metrics, "generation")
    if metrics.get("schema") != "selfless_imagenet_val_t2i_fid_is_v2":
        raise ValueError("generation metrics use an obsolete protocol")
    if metrics.get("project_formal_protocol") is not True:
        raise ValueError("generation metrics are not project-formal")
    if metrics.get("leaderboard_comparable_to_adm_dit") is not False:
        raise ValueError("generation metrics have an invalid comparability label")
    if metrics.get("split") != "val":
        raise ValueError("generation metrics do not use ImageNet val")
    if metrics.get("real_source") != "cached_original_imagenet_val":
        raise ValueError("generation metrics use an unexpected real-image source")

    protocol = metrics.get("metric_protocol") or {}
    expected_protocol = {
        "protocol_name": GENERATION_PROTOCOL_NAME,
        "reference_distribution": GENERATION_REFERENCE_DISTRIBUTION,
        "comparison_scope": GENERATION_COMPARISON_SCOPE,
        "not_adm_dit_reason": GENERATION_NON_COMPARABILITY_REASON,
        "fid_reducer": GENERATION_FID_REDUCER,
        "fid_computed": True,
        "is_split_assignment": GENERATION_IS_SPLIT_ASSIGNMENT,
        "is_std": "population",
        "is_splits": 10,
    }
    mismatches = {
        key: {"expected": expected, "actual": protocol.get(key)}
        for key, expected in expected_protocol.items()
        if protocol.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"generation metric protocol is invalid: {mismatches}")
    split_plan = protocol.get("is_split_plan") or {}
    expected_split_plan = {
        "assignment": GENERATION_IS_SPLIT_ASSIGNMENT,
        "source_dataset_split": "val",
        "samples": 50_000,
        "splits": 10,
        "class_count": 1_000,
        "samples_per_class_min": 50,
        "samples_per_class_max": 50,
        "samples_per_class_per_split_min": 5,
        "samples_per_class_per_split_max": 5,
    }
    mismatches = {
        key: {"expected": expected, "actual": split_plan.get(key)}
        for key, expected in expected_split_plan.items()
        if split_plan.get(key) != expected
    }
    if list(split_plan.get("samples_per_split", [])) != [5_000] * 10:
        mismatches["samples_per_split"] = split_plan.get("samples_per_split")
    if list(split_plan.get("classes_per_split", [])) != [1_000] * 10:
        mismatches["classes_per_split"] = split_plan.get("classes_per_split")
    if mismatches:
        raise ValueError(f"generation IS split plan is invalid: {mismatches}")

    samples = int(metrics.get("samples_evaluated", -1))
    if samples != 50_000 or int(metrics.get("samples_requested", -1)) != samples:
        raise ValueError("generation metrics do not contain exactly 50K samples")
    strategies = metrics.get("strategies")
    if not isinstance(strategies, dict) or len(strategies) != 1:
        raise ValueError("generation metrics must contain exactly one strategy")
    strategy_name, strategy = next(iter(strategies.items()))
    if not strategy_name or not isinstance(strategy, dict):
        raise ValueError("generation strategy is invalid")
    if int(strategy.get("count", -1)) != samples:
        raise ValueError("generation strategy sample count is inconsistent")
    fid = finite(strategy.get("fid"), "generation.fid")
    is_mean = finite(strategy.get("inception_score_mean"), "generation.is_mean")
    is_std = finite(strategy.get("inception_score_std"), "generation.is_std")
    is_values = [
        finite(value, f"generation.is_splits[{index}]")
        for index, value in enumerate(strategy.get("inception_score_splits", []))
    ]
    if fid < 0.0:
        raise ValueError("generation FID must be non-negative")
    if len(is_values) != 10:
        raise ValueError("generation IS split metrics are incomplete")
    if is_mean < 1.0 - 1.0e-9 or any(
        value < 1.0 - 1.0e-9 for value in is_values
    ):
        raise ValueError("generation IS is below its theoretical minimum")
    if is_std < 0.0:
        raise ValueError("generation IS std must be non-negative")
    expected_mean = sum(is_values) / len(is_values)
    expected_std = math.sqrt(
        sum((value - expected_mean) ** 2 for value in is_values)
        / len(is_values)
    )
    if not math.isclose(is_mean, expected_mean, rel_tol=0.0, abs_tol=1.0e-10):
        raise ValueError("generation IS mean is inconsistent with split values")
    if not math.isclose(is_std, expected_std, rel_tol=0.0, abs_tol=1.0e-10):
        raise ValueError("generation IS std is inconsistent with split values")
    return {
        "protocol": GENERATION_PROTOCOL_NAME,
        "reference_distribution": GENERATION_REFERENCE_DISTRIBUTION,
        "comparison_scope": GENERATION_COMPARISON_SCOPE,
        "project_formal_protocol": True,
        "leaderboard_comparable_to_adm_dit": False,
        "strategy": strategy_name,
        "samples": samples,
        "fid": fid,
        "inception_score_mean": is_mean,
        "inception_score_std": is_std,
        "is_split_assignment": GENERATION_IS_SPLIT_ASSIGNMENT,
    }


def selected_imagenet_classification(source: dict[str, Any]) -> dict[str, Any]:
    if source.get("schema") != (
        "selfless_imagenet1k_zeroshot_classification_summary_v2"
    ):
        raise ValueError("ImageNet result uses an obsolete evaluation protocol")
    if source.get("task") != "imagenet1k_zeroshot_classification":
        raise ValueError("ImageNet classification task mismatch")
    if source.get("complete_formal_target") is not True:
        raise ValueError("ImageNet classification is incomplete")
    if source.get("project_formal_protocol") is not True:
        raise ValueError("ImageNet classification is not a formal 50K run")
    if int(source.get("records", -1)) != 50_000:
        raise ValueError("ImageNet classification must use all 50K val images")
    if int(source.get("classes", -1)) != 1_000:
        raise ValueError("ImageNet classification must be 1,000-way")
    if int(source.get("formal_target_records", -1)) != 50_000 or int(
        source.get("language_prior_image_count", -1)
    ) != 50_000:
        raise ValueError("ImageNet classification has an invalid 50K prior target")
    if source.get("primary_metric") != "top_1_accuracy":
        raise ValueError("ImageNet classification has an unexpected primary metric")
    if source.get("accuracy_unit") != "unit_interval":
        raise ValueError("ImageNet classification has an ambiguous accuracy unit")
    if source.get("class_text_template") != "a photo of a {class_name}.":
        raise ValueError("ImageNet classification uses an unexpected class template")
    top1 = finite_unit_interval(source.get("top_1_accuracy"), "ImageNet Top-1")
    top5 = finite_unit_interval(source.get("top_5_accuracy"), "ImageNet Top-5")
    if top5 < top1:
        raise ValueError("ImageNet Top-5 accuracy cannot be below Top-1")
    scoring = source.get("scoring") or {}
    if scoring.get("primary_candidate_score") != (
        "language_prior_debiased_mean_token_loglikelihood"
    ):
        raise ValueError("ImageNet classification is not language-prior debiased")
    if float(scoring.get("language_prior_alpha", -1.0)) != 1.0:
        raise ValueError("ImageNet classification must fix alpha=1")
    if scoring.get("language_prior_estimator") != "candidate_image_logmeanexp":
        raise ValueError("ImageNet classification uses the wrong prior estimator")
    return dict(source)


def benchmark_summary(source: dict[str, Any], task: str) -> dict[str, Any]:
    metrics = source["metrics"]
    actual = int(metrics.get("records", -1))
    expected = EXPECTED_BENCHMARK_RECORDS[task]
    if actual != expected:
        raise ValueError(f"{task} records mismatch: {actual} != {expected}")
    expected_primary = (
        "language_prior_debiased_pairwise.win_rate"
        if task in PAPER_BENCHMARKS
        else (
            "circular_accuracy_language_prior_debiased"
            if task == "mmbench_dev_en"
            else "accuracy_language_prior_debiased"
        )
    )
    if metrics.get("primary_metric") != expected_primary:
        raise ValueError(f"{task} uses an obsolete candidate-score protocol")
    primary_value: Any = metrics
    for component in expected_primary.split("."):
        primary_value = primary_value[component]
    finite_unit_interval(primary_value, f"{task}.{expected_primary}")
    compact = dict(source)
    compact_metrics = dict(metrics)
    if task in {"aro_vg_relation", "aro_vg_attribution"}:
        compact_metrics.pop("categories", None)
    compact["metrics"] = compact_metrics
    return compact


def package_standard_retrieval(
    *,
    root: Path,
    destination: Path,
    task: str,
    contract: dict[str, int],
    checkpoint: Path,
    step: int,
) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    summary_path = root / "summary.json"
    manifest = read_json(manifest_path)
    summary = read_json(summary_path)
    validate_no_hash(manifest, f"{task} manifest")
    validate_no_hash(summary, f"{task} summary")
    if manifest.get("schema") != "selfless_cross_dataset_retrieval_evaluation_v3":
        raise ValueError(f"{task} uses an obsolete retrieval protocol")
    if summary.get("schema") != "selfless_cross_dataset_retrieval_summary_v3":
        raise ValueError(f"{task} uses an obsolete retrieval summary")
    if manifest.get("complete") is not True:
        raise ValueError(f"{task} evaluation manifest is incomplete")
    if manifest.get("project_formal_protocol") is not True:
        raise ValueError(f"{task} retrieval manifest is not formal")
    if summary.get("project_formal_protocol") is not True:
        raise ValueError(f"{task} retrieval summary is not formal")
    if summary.get("complete_formal_target") is not True:
        raise ValueError(f"{task} formal target is incomplete")
    if manifest.get("task") != task or summary.get("task") != task:
        raise ValueError(f"{task} result root contains another task")
    if Path(manifest["checkpoint"]).resolve() != checkpoint.resolve():
        raise ValueError(f"{task} result belongs to another checkpoint")
    if int(summary.get("checkpoint_step", -1)) != step:
        raise ValueError(f"{task} result belongs to another checkpoint step")
    for field in ("images", "captions"):
        if int(summary.get(field, -1)) != int(contract[field]):
            raise ValueError(f"{task} has invalid {field} cardinality")
    scoring = summary.get("scoring") or {}
    if scoring.get("primary_candidate_score") != (
        "language_prior_debiased_mean_token_loglikelihood"
    ):
        raise ValueError(f"{task} is not language-prior debiased")
    if float(scoring.get("language_prior_alpha", -1.0)) != 1.0:
        raise ValueError(f"{task} does not fix alpha=1")
    if scoring.get("language_prior_estimator") != "candidate_image_logmeanexp":
        raise ValueError(f"{task} uses the wrong language-prior estimator")
    if summary.get("recall_unit") != "unit_interval":
        raise ValueError(f"{task} has an ambiguous recall unit")
    if summary.get("rank_unit") != "one_based_candidate_rank":
        raise ValueError(f"{task} has an ambiguous rank unit")
    if summary.get("coco_five_fold_1k_average") is not False:
        raise ValueError(f"{task} is not a full-candidate retrieval run")
    if summary.get("primary_metric") != "mean_recall_at_1_5_10":
        raise ValueError(f"{task} has an unexpected primary metric")
    expected_distribution = (
        {"5": 4_990, "6": 10}
        if task == "mscoco_karpathy_test_5k"
        else {"5": 1_000}
    )
    if summary.get("caption_count_distribution") != expected_distribution:
        raise ValueError(f"{task} has an invalid caption-count distribution")
    for direction, queries, candidates in (
        ("image_to_text", int(contract["images"]), int(contract["captions"])),
        ("text_to_image", int(contract["captions"]), int(contract["images"])),
    ):
        values = summary.get(direction) or {}
        if int(values.get("queries", -1)) != queries or int(
            values.get("candidates", -1)
        ) != candidates:
            raise ValueError(f"{task} {direction} query/candidate counts are invalid")
        recalls = [
            finite_unit_interval(
                values.get(f"recall_at_{k}"), f"{task}.{direction}.R@{k}"
            )
            for k in (1, 5, 10)
        ]
        if recalls != sorted(recalls):
            raise ValueError(f"{task} {direction} recall is not monotone in K")
        direction_mean = finite_unit_interval(
            values.get("mean_recall_at_1_5_10"),
            f"{task}.{direction}.mean_recall",
        )
        if not math.isclose(
            direction_mean,
            sum(recalls) / 3.0,
            rel_tol=0.0,
            abs_tol=1.0e-7,
        ):
            raise ValueError(f"{task} {direction} mean recall is inconsistent")
        for rank_name in ("mean_rank", "median_rank"):
            rank_value = finite(
                values.get(rank_name), f"{task}.{direction}.{rank_name}"
            )
            if not 1.0 <= rank_value <= candidates:
                raise ValueError(f"{task} {direction} has an invalid {rank_name}")
    expected_overall = sum(
        float(summary[direction]["mean_recall_at_1_5_10"])
        for direction in ("image_to_text", "text_to_image")
    ) / 2.0
    overall = finite_unit_interval(
        summary.get("mean_recall_at_1_5_10"), f"{task}.mean_recall"
    )
    if not math.isclose(overall, expected_overall, rel_tol=0.0, abs_tol=1.0e-7):
        raise ValueError(f"{task} overall mean recall is inconsistent")
    link_file(manifest_path, destination / "manifest.json")
    link_file(summary_path, destination / "summary.json")
    return summary


def artifact(path: str) -> str:
    return path


def package_step(
    output_root: Path,
    stage: Path,
    step: int,
    *,
    latest_core_root: Path | None = None,
    latest_native_root: Path | None = None,
    standard_roots: dict[str, Path] | None = None,
) -> dict[str, Any]:
    core = core_root(output_root, step, latest_core_root)
    benchmarks = benchmark_root(output_root, step, latest_native_root)
    imagenet = imagenet_classification_root(output_root, step, latest_native_root)
    destination = stage / "checkpoints" / f"step-{step}"

    generation_metrics = read_json(core / "t2i-fid-is" / "metrics.json")
    generation = selected_generation(generation_metrics)
    if not generation["project_formal_protocol"] or generation["samples"] != 50_000:
        raise ValueError(f"step {step} generation is not a project-formal FID50K run")
    if generation["is_split_assignment"] != "stratified_by_synset":
        raise ValueError(f"step {step} uses the obsolete IS split assignment")
    link_file(
        core / "t2i-fid-is" / "metrics.json",
        destination / "paper-generation" / "imagenet-val-fid50k-is" / "metrics.json",
    )
    generation_progress = read_json(
        core / "t2i-fid-is" / "evaluation_progress.json"
    )
    generation_progress["metrics_path"] = "metrics.json"
    write_json(
        destination
        / "paper-generation"
        / "imagenet-val-fid50k-is"
        / "evaluation_progress.json",
        generation_progress,
    )

    text_summary = read_json(core / "text" / "summary.json")
    validate_no_hash(text_summary, f"step {step} text")
    if text_summary.get("schema") != "selfless_text_benchmark_summary_v4":
        raise ValueError(f"step {step} text suite uses an obsolete protocol")
    if text_summary.get("complete") is not True:
        raise ValueError(f"step {step} text suite is incomplete")
    validate_text_summary_for_archive(text_summary)
    checkpoint_path = Path(text_summary["checkpoint"]).resolve()
    if int(text_summary["checkpoint_step"]) != step:
        raise ValueError(f"step {step} text summary checkpoint mismatch")
    link_file(
        core / "text" / "summary.json",
        destination / "paper-text" / "summary.json",
    )
    text_run = read_json(core / "text" / "evaluation_run.json")
    text_run["summary"] = "summary.json"
    write_json(
        destination / "paper-text" / "evaluation_run.json", text_run
    )
    link_tree(core / "text" / "tasks", destination / "paper-text" / "tasks")

    validation_run = read_json(core / "validation" / "evaluation_run.json")
    validate_no_hash(validation_run, f"step {step} heldout validation")
    if validation_run.get("complete") is not True or validation_run.get("imagenet_split") != "val":
        raise ValueError(f"step {step} heldout validation is incomplete or not val")
    validation_run["metrics"] = f"validation_metrics_step_{step}.json"
    write_json(
        destination / "diagnostics" / "heldout-validation" / "evaluation_run.json",
        validation_run,
    )
    link_file(
        core / "validation" / f"validation_metrics_step_{step}.json",
        destination
        / "diagnostics"
        / "heldout-validation"
        / f"validation_metrics_step_{step}.json",
    )
    link_tree(
        core / "validation" / "validation_flow_images",
        destination / "diagnostics" / "qualitative" / "generated-images",
    )
    link_tree(
        core / "validation" / "validation_i2t_captions",
        destination / "diagnostics" / "qualitative" / "generated-captions",
    )

    imagenet_manifest = read_json(imagenet / "manifest.json")
    validate_no_hash(imagenet_manifest, f"step {step} ImageNet classification")
    if imagenet_manifest.get("complete") is not True:
        raise ValueError(f"step {step} ImageNet classification is incomplete")
    if imagenet_manifest.get("project_formal_protocol") is not True:
        raise ValueError(f"step {step} ImageNet classification is not formal")
    if imagenet_manifest.get("schema") != (
        "selfless_imagenet1k_zeroshot_classification_evaluation_v2"
    ):
        raise ValueError(f"step {step} uses an obsolete ImageNet protocol")
    if Path(imagenet_manifest["checkpoint"]).resolve() != checkpoint_path:
        raise ValueError(f"step {step} ImageNet checkpoint mismatch")
    imagenet_summary = selected_imagenet_classification(
        read_json(imagenet / "summary.json")
    )
    if int(imagenet_summary.get("checkpoint_step", -1)) != step:
        raise ValueError(f"step {step} ImageNet summary checkpoint mismatch")
    imagenet_destination = (
        destination
        / "paper-understanding"
        / "imagenet1k-zeroshot-classification-50k"
    )
    link_file(imagenet / "manifest.json", imagenet_destination / "manifest.json")
    link_file(imagenet / "summary.json", imagenet_destination / "summary.json")

    benchmark_manifest = read_json(benchmarks / "manifest.json")
    benchmark_overview = read_json(benchmarks / "summary.json")
    validate_no_hash(benchmark_manifest, f"step {step} benchmark")
    validate_no_hash(benchmark_overview, f"step {step} benchmark summary")
    if benchmark_manifest.get("complete") is not True:
        raise ValueError(f"step {step} benchmark evaluation is incomplete")
    if benchmark_manifest.get("schema") != (
        "selfless_multimodal_likelihood_evaluation_v5"
    ):
        raise ValueError(f"step {step} uses an obsolete benchmark protocol")
    if benchmark_overview.get("schema") != (
        "selfless_multimodal_likelihood_summary_v5"
    ):
        raise ValueError(f"step {step} uses an obsolete benchmark summary")
    if benchmark_manifest.get("project_formal_protocol") is not True:
        raise ValueError(f"step {step} benchmark is not a formal MC64 run")
    if benchmark_overview.get("project_formal_protocol") is not True:
        raise ValueError(f"step {step} benchmark summary is not formal")
    if int(benchmark_manifest.get("mc_samples", -1)) != 64:
        raise ValueError(f"step {step} benchmark does not use MC64")
    if Path(benchmark_manifest["checkpoint"]).resolve() != checkpoint_path:
        raise ValueError(f"step {step} benchmark checkpoint mismatch")
    if int(benchmark_overview.get("checkpoint_step", -1)) != step:
        raise ValueError(f"step {step} benchmark summary checkpoint mismatch")
    if benchmark_manifest.get("primary_candidate_score") != (
        "language_prior_debiased_mean_token_loglikelihood"
    ):
        raise ValueError(f"step {step} benchmark is not language-prior debiased")
    if float(benchmark_manifest.get("language_prior_alpha", -1.0)) != 1.0:
        raise ValueError(f"step {step} benchmark does not fix alpha=1")
    if benchmark_manifest.get("language_prior_estimator") != (
        "content_free_gaussian_image_logmeanexp"
    ):
        raise ValueError(f"step {step} benchmark uses the wrong prior estimator")
    benchmark_scoring = benchmark_overview.get("scoring") or {}
    if benchmark_scoring.get("primary_candidate_score") != (
        "language_prior_debiased_mean_token_loglikelihood"
    ):
        raise ValueError(f"step {step} benchmark summary is not debiased")
    if benchmark_scoring.get("reported_score_variant") != (
        "language_prior_debiased_only"
    ):
        raise ValueError(f"step {step} benchmark summary retains old scores")
    paper_benchmarks: dict[str, Any] = {}
    internal_benchmarks: dict[str, Any] = {}
    for task in PAPER_BENCHMARKS + INTERNAL_BENCHMARKS:
        summary = benchmark_summary(
            read_json(benchmarks / "summaries" / f"{task}.json"), task
        )
        group = (
            "paper-understanding/compositional"
            if task in PAPER_BENCHMARKS
            else "internal-ablation"
        )
        link_file(
            benchmarks / "summaries" / f"{task}.json",
            destination / group / task / "summary.json",
        )
        link_file(
            benchmarks / "predictions" / f"{task}.jsonl",
            destination / group / task / "predictions.jsonl",
        )
        if task in PAPER_BENCHMARKS:
            paper_benchmarks[task] = summary
        else:
            internal_benchmarks[task] = summary

    standard_results: dict[str, Any] = {}
    standard_pending: list[str] = []
    if step == STANDARD_RETRIEVAL_STEP:
        roots = standard_roots or {}
        for task, contract in STANDARD_RETRIEVAL.items():
            root = roots.get(task)
            task_destination = (
                destination
                / "paper-understanding"
                / "standard-retrieval"
                / task
            )
            if root is not None and (root / "summary.json").is_file():
                standard_results[task] = package_standard_retrieval(
                    root=root,
                    destination=task_destination,
                    task=task,
                    contract=contract,
                    checkpoint=checkpoint_path,
                    step=step,
                )
                continue
            standard_pending.append(task)
            write_json(
                task_destination / "STATUS.json",
                {
                    "schema": "pending_evaluation_v1",
                    "status": "pending",
                    "reason": "latest_checkpoint_formal_evaluation_not_completed_yet",
                    "task": task,
                    **contract,
                    "metrics": [
                        "i2t_r1",
                        "i2t_r5",
                        "i2t_r10",
                        "t2i_r1",
                        "t2i_r5",
                        "t2i_r10",
                    ],
                    "runtime_hashing_enabled": False,
                },
            )

    validation_metrics = read_json(
        core / "validation" / f"validation_metrics_step_{step}.json"
    )["metrics"]
    complete_for_protocol = not standard_pending
    report = {
        "schema": "selected_checkpoint_evaluation_archive_v2",
        "complete_for_current_paper_protocol": complete_for_protocol,
        "incomplete_only_because": standard_pending,
        "runtime_hashing_enabled": False,
        "checkpoint": str(checkpoint_path),
        "global_step": step,
        "paper_generation": generation,
        "paper_understanding": {
            "imagenet1k_zeroshot_classification": imagenet_summary,
            "standard_retrieval": standard_results,
            "standard_retrieval_checkpoint_policy": {
                "policy": "latest_retained_checkpoint_only",
                "checkpoint_step": STANDARD_RETRIEVAL_STEP,
                "evaluated_for_this_checkpoint": step == STANDARD_RETRIEVAL_STEP,
            },
            "compositional": paper_benchmarks,
        },
        "paper_text": {
            "macro_average_primary": finite(
                text_summary["macro_average_primary"], f"step {step} text macro"
            ),
            "primary_metrics": text_summary["primary_metrics"],
        },
        "internal_ablation_only": internal_benchmarks,
        "diagnostics": {
            "heldout_validation": {
                key: finite(value, f"step {step} {key}")
                for key, value in validation_metrics.items()
            },
            "qualitative_images_and_captions": True,
        },
        "removed_from_archive": list(REMOVED_EVALUATIONS),
        "artifacts": {
            "generation": artifact("paper-generation/imagenet-val-fid50k-is"),
            "understanding": artifact("paper-understanding"),
            "text": artifact("paper-text"),
            "internal_ablation": artifact("internal-ablation"),
            "diagnostics": artifact("diagnostics"),
        },
        "packaged_at": utc_now(),
    }
    write_json(destination / "summary.json", report)
    return report


def metric(summary: dict[str, Any], task: str) -> float:
    metrics = summary["metrics"]
    key = str(metrics["primary_metric"])
    value: Any = metrics
    for component in key.split("."):
        value = value[component]
    return finite(value, f"{task}.{key}")


def trend_payload(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    compact = []
    pending_tasks: list[str] = []
    for row in rows:
        understanding = row["paper_understanding"]
        classification = understanding["imagenet1k_zeroshot_classification"]
        pending_tasks.extend(row["incomplete_only_because"])
        standard = understanding["standard_retrieval"]
        compact.append(
            {
                "global_step": row["global_step"],
                "checkpoint": row["checkpoint"],
                "fid": row["paper_generation"]["fid"],
                "inception_score_mean": row["paper_generation"]["inception_score_mean"],
                "inception_score_std": row["paper_generation"]["inception_score_std"],
                "imagenet1k_zeroshot": {
                    "top_1_accuracy": classification["top_1_accuracy"],
                    "top_5_accuracy": classification["top_5_accuracy"],
                },
                "compositional": {
                    task: metric(value, task)
                    for task, value in understanding["compositional"].items()
                },
                "text_macro": row["paper_text"]["macro_average_primary"],
                "internal_ablation": {
                    task: metric(value, task)
                    for task, value in row["internal_ablation_only"].items()
                },
                "standard_retrieval": {
                    task: {
                        "image_to_text": value["image_to_text"],
                        "text_to_image": value["text_to_image"],
                    }
                    for task, value in standard.items()
                },
                "standard_retrieval_status": (
                    "evaluated"
                    if standard
                    else (
                        "pending"
                        if row["global_step"] == STANDARD_RETRIEVAL_STEP
                        else "not_scheduled_latest_checkpoint_only"
                    )
                ),
            }
        )
    pending_tasks = sorted(set(pending_tasks))
    return {
        "schema": "selected_checkpoint_evaluation_trend_v2",
        "complete_for_current_paper_protocol": not pending_tasks,
        "pending_tasks": pending_tasks,
        "standard_retrieval_checkpoint_policy": "latest_retained_checkpoint_only",
        "standard_retrieval_checkpoint_step": STANDARD_RETRIEVAL_STEP,
        "runtime_hashing_enabled": False,
        "rows": compact,
    }


def trend_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Unified A 0.6B retained-checkpoint evaluation trend",
        "",
        "MSCOCO/Flickr30K standard retrieval follows the requested "
        "latest-retained-checkpoint-only policy and is evaluated only at step 60000.",
        "",
        "| step | FID ↓ | IS ↑ | ImageNet Top-1 ↑ (%) | ImageNet Top-5 ↑ (%) | SugarCrepe ↑ (%) | ARO relation ↑ (%) | ARO attribution ↑ (%) | text macro ↑ (%) |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["rows"]:
        classification = row["imagenet1k_zeroshot"]
        lines.append(
            f"| {row['global_step']} | {row['fid']:.6f} | "
            f"{row['inception_score_mean']:.6f} ± {row['inception_score_std']:.6f} | "
            f"{100.0 * classification['top_1_accuracy']:.2f} | "
            f"{100.0 * classification['top_5_accuracy']:.2f} | "
            f"{100.0 * row['compositional']['sugarcrepe']:.2f} | "
            f"{100.0 * row['compositional']['aro_vg_relation']:.2f} | "
            f"{100.0 * row['compositional']['aro_vg_attribution']:.2f} | "
            f"{100.0 * row['text_macro']:.2f} |"
        )
    latest = next(
        row
        for row in payload["rows"]
        if int(row["global_step"]) == STANDARD_RETRIEVAL_STEP
    )
    if latest["standard_retrieval"]:
        lines.extend(
            [
                "",
                "## Step 60000 standard retrieval",
                "",
                "| dataset | I2T R@1 (%) | I2T R@5 (%) | I2T R@10 (%) | T2I R@1 (%) | T2I R@5 (%) | T2I R@10 (%) |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        labels = {
            "mscoco_karpathy_test_5k": "MSCOCO Karpathy 5K test",
            "flickr30k_karpathy_test_1k": "Flickr30K Karpathy 1K test",
        }
        for task, values in latest["standard_retrieval"].items():
            i2t = values["image_to_text"]
            t2i = values["text_to_image"]
            lines.append(
                f"| {labels[task]} | {100.0 * i2t['recall_at_1']:.2f} | "
                f"{100.0 * i2t['recall_at_5']:.2f} | "
                f"{100.0 * i2t['recall_at_10']:.2f} | "
                f"{100.0 * t2i['recall_at_1']:.2f} | "
                f"{100.0 * t2i['recall_at_5']:.2f} | "
                f"{100.0 * t2i['recall_at_10']:.2f} |"
            )
    else:
        lines.extend(["", "Step 60000 standard retrieval is still pending."])
    lines.extend(
        [
            "",
            "MMBench and SEED are stored under `internal-ablation/` and are not "
            "paper-main-table metrics.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    destination = args.destination.resolve()
    latest_core_root = (
        args.latest_core_root.resolve() if args.latest_core_root else None
    )
    latest_native_root = (
        args.latest_native_root.resolve() if args.latest_native_root else None
    )
    standard_roots: dict[str, Path] = {}
    if args.coco_retrieval_root:
        standard_roots["mscoco_karpathy_test_5k"] = (
            args.coco_retrieval_root.resolve()
        )
    if args.flickr30k_retrieval_root:
        standard_roots["flickr30k_karpathy_test_1k"] = (
            args.flickr30k_retrieval_root.resolve()
        )
    if destination.exists():
        raise FileExistsError(f"destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = destination.parent / f".{destination.name}.organizing-{os.getpid()}"
    if stage.exists():
        raise FileExistsError(stage)
    stage.mkdir()
    try:
        rows = []
        for step in STEPS:
            rows.append(
                package_step(
                    output_root,
                    stage,
                    step,
                    latest_core_root=latest_core_root,
                    latest_native_root=latest_native_root,
                    standard_roots=standard_roots,
                )
            )
        trend = trend_payload(rows)
        write_json(stage / "trend" / "trend.json", trend)
        atomic_write(stage / "trend" / "trend.md", trend_markdown(trend))
        files = [path for path in stage.rglob("*") if path.is_file()]
        manifest = {
            "schema": "selected_evaluation_archive_manifest_v2",
            "complete": True,
            "runtime_hashing_enabled": False,
            "model": "unified-a-0p6b",
            "checkpoint_steps": list(STEPS),
            "files": len(files) + 1,
            "bytes": 0,
            "paper_protocol_complete": trend[
                "complete_for_current_paper_protocol"
            ],
            "pending_tasks": trend["pending_tasks"],
            "standard_retrieval_checkpoint_policy": (
                "latest_retained_checkpoint_only"
            ),
            "standard_retrieval_checkpoint_step": STANDARD_RETRIEVAL_STEP,
            "checkpoint_inputs_location": str(
                (output_root / "evaluation-checkpoints").resolve()
            ),
            "checkpoint_inputs_are_not_evaluation_outputs": True,
            "created_at": utc_now(),
        }
        manifest_path = stage / "manifest.json"
        write_json(manifest_path, manifest)
        for _ in range(3):
            total_bytes = sum(
                path.stat().st_size
                for path in stage.rglob("*")
                if path.is_file()
            )
            if manifest["bytes"] == total_bytes:
                break
            manifest["bytes"] = total_bytes
            write_json(manifest_path, manifest)
        os.replace(stage, destination)
        stage = None
        print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    finally:
        if stage is not None and stage.exists():
            shutil.rmtree(stage)


if __name__ == "__main__":
    main()
