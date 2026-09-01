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
REMOVED_EVALUATIONS = (
    "imagenet_classification_top1_top5",
    "imagenet_real_top1_top5",
    "visual_calibration",
    "custom_imagenet_caption_negatives",
    "pope_coco",
    "coco_caption_ppl",
    "imagenet_i2t_clip",
    "winoground",
    "svo_probes",
    "whats_up",
    "rank_shards",
    "fid_resume_state",
    "duplicate_logs",
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
    parser.add_argument(
        "--historical_archive_root",
        type=Path,
        help=(
            "Existing compact archive used only for retained step-56000/58000 "
            "results. This avoids requiring already-cleaned raw rank outputs."
        ),
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


def imagenet_retrieval_root(
    output_root: Path, step: int, latest_native_root: Path | None = None
) -> Path:
    if step == STANDARD_RETRIEVAL_STEP and latest_native_root is not None:
        return latest_native_root / "imagenet-retrieval"
    return (
        output_root
        / f"unified-a-0p6b-native-full-eval-step{step}-ascend16-r1"
        / "pretraining-native-understanding"
        / "imagenet-classification-retrieval"
    )


def validate_no_hash(payload: dict[str, Any], label: str) -> None:
    if payload.get("runtime_hashing_enabled", True) is not False:
        raise ValueError(f"{label} violates the no-hash contract")


def finite(value: Any, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite selected metric: {label}={number}")
    return number


def selected_generation(metrics: dict[str, Any]) -> dict[str, Any]:
    validate_no_hash(metrics, "generation")
    strategy = metrics["strategies"]["spatial_halton"]
    return {
        "protocol": "imagenet_val_t2i_fid50k_is_stratified",
        "official_protocol": bool(metrics["official_protocol"]),
        "samples": int(metrics["samples_evaluated"]),
        "fid": finite(strategy["fid"], "generation.fid"),
        "inception_score_mean": finite(
            strategy["inception_score_mean"], "generation.is_mean"
        ),
        "inception_score_std": finite(
            strategy["inception_score_std"], "generation.is_std"
        ),
        "is_split_assignment": metrics["metric_protocol"]["is_split_assignment"],
    }


def selected_imagenet_retrieval(source: dict[str, Any], task: str) -> dict[str, Any]:
    values = source.get("normalized_loglikelihood")
    if not isinstance(values, dict):
        values = source.get("raw_normalized_loglikelihood")
    if not isinstance(values, dict):
        raise ValueError(f"{task} has no uncalibrated normalized likelihood")
    expected = 1_000 if task == "retrieval_1k" else 5_000
    if int(source.get("records", -1)) != expected:
        raise ValueError(f"{task} has an invalid record count")
    return {
        "schema": "selfless_imagenet_retrieval_task_summary_v2",
        "task": task,
        "records": expected,
        "complete_formal_target": source.get("complete_formal_target") is True,
        "selection": source["selection"],
        "images_per_class": int(source["images_per_class"]),
        "prompt": source["prompt"],
        "primary_metric": (
            "normalized_loglikelihood.mean_bidirectional_instance_recall_at_1"
        ),
        "normalized_loglikelihood": values,
        "visual_calibration_enabled": False,
        "runtime_hashing_enabled": False,
    }


def benchmark_summary(source: dict[str, Any], task: str) -> dict[str, Any]:
    metrics = source["metrics"]
    actual = int(metrics.get("records", -1))
    expected = EXPECTED_BENCHMARK_RECORDS[task]
    if actual != expected:
        raise ValueError(f"{task} records mismatch: {actual} != {expected}")
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
    if manifest.get("complete") is not True:
        raise ValueError(f"{task} evaluation manifest is incomplete")
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
    link_file(manifest_path, destination / "manifest.json")
    link_file(summary_path, destination / "summary.json")
    return summary


def artifact(path: str) -> str:
    return path


def package_historical_step(
    archive_root: Path,
    stage: Path,
    step: int,
) -> dict[str, Any]:
    """Reuse a validated compact historical step without reviving stale tasks."""

    if step == STANDARD_RETRIEVAL_STEP:
        raise ValueError("the latest checkpoint must be rebuilt from fresh results")
    source = archive_root / "checkpoints" / f"step-{step}"
    destination = stage / "checkpoints" / f"step-{step}"
    report = read_json(source / "summary.json")
    validate_no_hash(report, f"historical step {step}")
    if int(report.get("global_step", -1)) != step:
        raise ValueError(f"historical step {step} summary mismatch")

    text_summary = read_json(source / "paper-text" / "summary.json")
    validate_no_hash(text_summary, f"historical step {step} text")
    if text_summary.get("complete") is not True:
        raise ValueError(f"historical step {step} text suite is incomplete")
    if set(text_summary.get("primary_metrics", {})) != set(TEXT_TASKS):
        raise ValueError(f"historical step {step} text task set is obsolete")

    selected_directories = (
        "paper-generation",
        "paper-text",
        "internal-ablation",
        "diagnostics",
        "paper-understanding/imagenet-val-custom-retrieval",
        "paper-understanding/compositional",
    )
    for relative in selected_directories:
        link_tree(source / relative, destination / relative)

    # Standard retrieval follows the latest-checkpoint-only policy. Older
    # pending markers were created before that decision and must not survive.
    understanding = dict(report["paper_understanding"])
    understanding["standard_retrieval"] = {}
    understanding["standard_retrieval_checkpoint_policy"] = {
        "policy": "latest_retained_checkpoint_only",
        "checkpoint_step": STANDARD_RETRIEVAL_STEP,
        "evaluated_for_this_checkpoint": False,
    }
    report["paper_understanding"] = understanding
    report["complete_for_current_paper_protocol"] = True
    report["incomplete_only_because"] = []
    report["packaged_at"] = utc_now()
    removed = list(report.get("removed_from_archive", []))
    for item in (*REMOVED_EVALUATIONS, "obsolete_standard_retrieval_pending_markers"):
        if item not in removed:
            removed.append(item)
    report["removed_from_archive"] = removed
    write_json(destination / "summary.json", report)
    return report


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
    imagenet = imagenet_retrieval_root(output_root, step, latest_native_root)
    destination = stage / "checkpoints" / f"step-{step}"

    generation_metrics = read_json(core / "t2i-fid-is" / "metrics.json")
    generation = selected_generation(generation_metrics)
    if not generation["official_protocol"] or generation["samples"] != 50_000:
        raise ValueError(f"step {step} generation is not an official FID50K run")
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
    if text_summary.get("complete") is not True:
        raise ValueError(f"step {step} text suite is incomplete")
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
    validate_no_hash(imagenet_manifest, f"step {step} ImageNet retrieval")
    if imagenet_manifest.get("complete") is not True:
        raise ValueError(f"step {step} ImageNet retrieval is incomplete")
    imagenet_summaries: dict[str, Any] = {}
    for task in ("retrieval_1k", "retrieval_5k"):
        summary = selected_imagenet_retrieval(
            read_json(imagenet / task / "summary.json"), task
        )
        imagenet_summaries[task] = summary
        write_json(
            destination
            / "paper-understanding"
            / "imagenet-val-custom-retrieval"
            / task
            / "summary.json",
            summary,
        )

    benchmark_manifest = read_json(benchmarks / "manifest.json")
    validate_no_hash(benchmark_manifest, f"step {step} benchmark")
    if benchmark_manifest.get("complete") is not True:
        raise ValueError(f"step {step} benchmark evaluation is incomplete")
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
        "schema": "selected_checkpoint_evaluation_archive_v1",
        "complete_for_current_paper_protocol": complete_for_protocol,
        "incomplete_only_because": standard_pending,
        "runtime_hashing_enabled": False,
        "checkpoint": str(checkpoint_path),
        "global_step": step,
        "paper_generation": generation,
        "paper_understanding": {
            "imagenet_val_custom_retrieval": imagenet_summaries,
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
        custom = understanding["imagenet_val_custom_retrieval"]
        pending_tasks.extend(row["incomplete_only_because"])
        standard = understanding["standard_retrieval"]
        compact.append(
            {
                "global_step": row["global_step"],
                "checkpoint": row["checkpoint"],
                "fid": row["paper_generation"]["fid"],
                "inception_score_mean": row["paper_generation"]["inception_score_mean"],
                "inception_score_std": row["paper_generation"]["inception_score_std"],
                "imagenet_retrieval": {
                    task: value["normalized_loglikelihood"]
                    for task, value in custom.items()
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
                    task: value["normalized_loglikelihood"]
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
        "schema": "selected_checkpoint_evaluation_trend_v1",
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
        "| step | FID ↓ | IS ↑ | ImageNet 5K I2T R@1 ↑ | ImageNet 5K T2I R@1 ↑ | SugarCrepe ↑ | ARO relation ↑ | ARO attribution ↑ | text macro ↑ |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["rows"]:
        retrieval = row["imagenet_retrieval"]["retrieval_5k"]
        lines.append(
            f"| {row['global_step']} | {row['fid']:.6f} | "
            f"{row['inception_score_mean']:.6f} ± {row['inception_score_std']:.6f} | "
            f"{retrieval['image_to_text']['instance_recall_at_1']:.6f} | "
            f"{retrieval['text_to_image']['instance_recall_at_1']:.6f} | "
            f"{row['compositional']['sugarcrepe']:.6f} | "
            f"{row['compositional']['aro_vg_relation']:.6f} | "
            f"{row['compositional']['aro_vg_attribution']:.6f} | "
            f"{row['text_macro']:.6f} |"
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
                "| dataset | I2T R@1 | I2T R@5 | I2T R@10 | T2I R@1 | T2I R@5 | T2I R@10 |",
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
                f"| {labels[task]} | {i2t['recall_at_1']:.6f} | "
                f"{i2t['recall_at_5']:.6f} | {i2t['recall_at_10']:.6f} | "
                f"{t2i['recall_at_1']:.6f} | {t2i['recall_at_5']:.6f} | "
                f"{t2i['recall_at_10']:.6f} |"
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
    historical_archive_root = (
        args.historical_archive_root.resolve()
        if args.historical_archive_root
        else None
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
            if step != STANDARD_RETRIEVAL_STEP and historical_archive_root:
                rows.append(
                    package_historical_step(
                        historical_archive_root,
                        stage,
                        step,
                    )
                )
                continue
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
            "schema": "selected_evaluation_archive_manifest_v1",
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
