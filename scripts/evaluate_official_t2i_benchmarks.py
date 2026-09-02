#!/usr/bin/env python3
"""Validate layouts and invoke the official GenEval, DPG-Bench, and MJHQ evaluators."""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.official_t2i_benchmarks import (
    CLEANFID_COMMIT,
    CLEANFID_VERSION,
    DPGBENCH_COMMIT,
    DPGBENCH_IMAGES_PER_PROMPT,
    GENEVAL_COMMIT,
    GENEVAL_IMAGES_PER_PROMPT,
    GENEVAL_TAG_COUNTS,
    MJHQ_CATEGORY_COUNTS,
    MJHQ_COMMIT,
    BenchmarkPrompt,
    index_images,
    load_dpgbench_prompts,
    load_geneval_prompts,
    load_mjhq_prompts,
    require_exact_ids,
)

METRICS_SCHEMA = "official_t2i_benchmark_metrics_v1"
SUMMARY_SCHEMA = "official_t2i_benchmark_summary_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    geneval = subparsers.add_parser("geneval")
    geneval.add_argument("--repository", type=Path, required=True)
    geneval.add_argument("--image_dir", type=Path, required=True)
    geneval.add_argument("--detector_dir", type=Path, required=True)
    geneval.add_argument("--output_dir", type=Path, required=True)
    geneval.add_argument("--python", default=sys.executable)

    dpg = subparsers.add_parser("dpgbench")
    dpg.add_argument("--repository", type=Path, required=True)
    dpg.add_argument("--image_dir", type=Path, required=True)
    dpg.add_argument("--output_dir", type=Path, required=True)
    dpg.add_argument("--python", default=sys.executable)
    dpg.add_argument("--processes", type=int, default=1)
    dpg.add_argument("--port", type=int, default=29500)

    mjhq = subparsers.add_parser("mjhq")
    mjhq.add_argument("--repository", type=Path, required=True)
    mjhq.add_argument("--reference_dir", type=Path, required=True)
    mjhq.add_argument("--image_dir", type=Path, required=True)
    mjhq.add_argument("--output_dir", type=Path, required=True)
    mjhq.add_argument("--python", default=sys.executable)
    mjhq.add_argument("--device", default="cuda")
    mjhq.add_argument("--batch_size", type=int, default=32)
    mjhq.add_argument("--num_workers", type=int, default=12)

    summary = subparsers.add_parser("summary")
    summary.add_argument("--geneval", type=Path, required=True)
    summary.add_argument("--dpgbench", type=Path, required=True)
    summary.add_argument("--mjhq", type=Path, required=True)
    summary.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _completed_at() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _require_python(value: str) -> str:
    path = Path(value)
    if path.parent != Path(".") and not path.is_file():
        raise FileNotFoundError(path)
    return value


def _resolution_counts(paths: list[Path]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for path in paths:
        with Image.open(path) as image:
            counts[f"{image.width}x{image.height}"] += 1
    return dict(sorted(counts.items()))


def _load_generation_manifest(
    image_dir: Path,
    *,
    benchmark: str,
    prompt_count: int,
    images_per_prompt: int,
    source_commit: str,
) -> tuple[dict[str, Any], Path]:
    path = image_dir.parent / "generation_manifest.json"
    manifest = _read_json(path)
    expected = {
        "schema": "official_t2i_benchmark_generation_v1",
        "complete": True,
        "benchmark": benchmark,
        "prompt_count": prompt_count,
        "images_per_prompt": images_per_prompt,
        "generated_image_count": prompt_count * images_per_prompt,
    }
    mismatches = {
        key: {"expected": value, "actual": manifest.get(key)}
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise ValueError(f"invalid {benchmark} generation manifest: {mismatches}")
    if (manifest.get("official_source") or {}).get("commit") != source_commit:
        raise ValueError(f"{benchmark} manifest uses the wrong official prompts")
    if not isinstance(manifest.get("model_source"), dict) or not isinstance(
        manifest.get("generation"), dict
    ):
        raise TypeError(f"{benchmark} manifest lacks model/generation provenance")
    resolution = manifest.get("native_resolution")
    if (
        not isinstance(resolution, list)
        or len(resolution) != 2
        or any(int(value) <= 0 for value in resolution)
    ):
        raise ValueError(f"{benchmark} manifest has an invalid native resolution")
    return manifest, path.resolve()


def validate_geneval_layout(
    image_dir: Path, prompts: list[BenchmarkPrompt]
) -> None:
    expected_prompt_ids = {prompt.prompt_id for prompt in prompts}
    actual_prompt_ids = {
        path.name for path in image_dir.iterdir() if path.is_dir() and path.name.isdigit()
    }
    require_exact_ids(actual_prompt_ids, expected_prompt_ids, "GenEval prompt folders")
    expected_samples = {
        f"{index:05d}" for index in range(GENEVAL_IMAGES_PER_PROMPT)
    }
    for prompt in prompts:
        prompt_dir = image_dir / prompt.prompt_id
        metadata = json.loads(
            (prompt_dir / "metadata.jsonl").read_text(encoding="utf-8")
        )
        if metadata != prompt.metadata:
            raise ValueError(f"GenEval metadata mismatch: {prompt.prompt_id}")
        require_exact_ids(
            index_images(prompt_dir / "samples"),
            expected_samples,
            f"GenEval samples for {prompt.prompt_id}",
        )


def summarize_geneval_rows(
    rows: list[dict[str, Any]], prompts: list[BenchmarkPrompt]
) -> dict[str, Any]:
    expected_pairs = {
        (prompt.prompt_id, f"{sample:05d}")
        for prompt in prompts
        for sample in range(GENEVAL_IMAGES_PER_PROMPT)
    }
    observed_pairs: set[tuple[str, str]] = set()
    tag_correct: dict[str, list[bool]] = defaultdict(list)
    prompt_correct: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        filename = Path(str(row["filename"]))
        pair = (filename.parent.parent.name, filename.stem)
        if pair in observed_pairs:
            raise ValueError(f"duplicate GenEval result: {pair}")
        observed_pairs.add(pair)
        tag = str(row["tag"])
        if tag not in GENEVAL_TAG_COUNTS:
            raise ValueError(f"unknown GenEval task: {tag}")
        correct = bool(row["correct"])
        tag_correct[tag].append(correct)
        prompt_correct[pair[0]].append(correct)
    require_exact_ids(
        (f"{prompt_id}/{sample_id}" for prompt_id, sample_id in observed_pairs),
        (f"{prompt_id}/{sample_id}" for prompt_id, sample_id in expected_pairs),
        "GenEval result rows",
    )
    expected_tag_images = {
        tag: count * GENEVAL_IMAGES_PER_PROMPT
        for tag, count in GENEVAL_TAG_COUNTS.items()
    }
    observed_tag_images = {tag: len(values) for tag, values in tag_correct.items()}
    if observed_tag_images != expected_tag_images:
        raise ValueError(f"GenEval result task counts mismatch: {observed_tag_images}")
    task_scores = {
        tag: sum(values) / len(values) for tag, values in tag_correct.items()
    }
    return {
        "overall": sum(task_scores.values()) / len(GENEVAL_TAG_COUNTS),
        "task_scores": task_scores,
        "image_accuracy": sum(map(bool, (row["correct"] for row in rows)))
        / len(rows),
        "prompt_accuracy_any_of_four": sum(any(values) for values in prompt_correct.values())
        / len(prompts),
        "images": len(rows),
        "prompts": len(prompts),
    }


def run_geneval(args: argparse.Namespace) -> None:
    python = _require_python(args.python)
    repository = args.repository.resolve()
    image_dir = args.image_dir.resolve()
    prompts = load_geneval_prompts(repository)
    validate_geneval_layout(image_dir, prompts)
    manifest, manifest_path = _load_generation_manifest(
        image_dir,
        benchmark="geneval",
        prompt_count=len(prompts),
        images_per_prompt=GENEVAL_IMAGES_PER_PROMPT,
        source_commit=GENEVAL_COMMIT,
    )
    generated_paths = [
        image_dir / prompt.prompt_id / "samples" / f"{sample:05d}.png"
        for prompt in prompts
        for sample in range(GENEVAL_IMAGES_PER_PROMPT)
    ]
    generated_resolutions = _resolution_counts(generated_paths)
    width, height = map(int, manifest["native_resolution"])
    if generated_resolutions != {f"{width}x{height}": len(generated_paths)}:
        raise ValueError("GenEval images disagree with the generation manifest")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = (args.output_dir / "results.jsonl").resolve()
    subprocess.run(
        [
            python,
            str(repository / "evaluation" / "evaluate_images.py"),
            str(image_dir),
            "--outfile",
            str(raw_path),
            "--model-path",
            str(args.detector_dir.resolve()),
        ],
        cwd=repository,
        check=True,
    )
    summary = subprocess.run(
        [
            python,
            str(repository / "evaluation" / "summary_scores.py"),
            str(raw_path),
        ],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    (args.output_dir / "official_summary.txt").write_text(
        summary.stdout, encoding="utf-8"
    )
    rows = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines()]
    metrics = summarize_geneval_rows(rows, prompts)
    match = re.search(r"Overall score \(avg\. over tasks\):\s*([0-9.]+)", summary.stdout)
    if match is None or not math.isclose(
        metrics["overall"], float(match.group(1)), rel_tol=0.0, abs_tol=5.1e-6
    ):
        raise ValueError("GenEval official summary and normalized score disagree")
    result = {
        "schema": METRICS_SCHEMA,
        "complete": True,
        "benchmark": "geneval",
        "official_evaluator": {
            "repository": "djghosh13/geneval",
            "commit": GENEVAL_COMMIT,
        },
        "metric_unit": "unit_interval",
        "primary_metric": "overall",
        "aggregation": "unweighted_mean_over_six_tasks",
        "model_source": manifest["model_source"],
        "generation": manifest["generation"],
        "generated_resolutions": generated_resolutions,
        "generation_manifest": str(manifest_path),
        "metrics": metrics,
        "raw_results": str(raw_path),
        "completed_at": _completed_at(),
    }
    _write_json(args.output_dir / "metrics.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


def validate_dpgbench_layout(
    image_dir: Path, prompts: list[BenchmarkPrompt]
) -> int:
    images = index_images(image_dir)
    require_exact_ids(images, (prompt.prompt_id for prompt in prompts), "DPG-Bench grids")
    resolutions: set[int] = set()
    for path in images.values():
        with Image.open(path) as image:
            if image.width != image.height or image.width % 2:
                raise ValueError(f"invalid DPG-Bench 2x2 grid: {path}")
            resolutions.add(image.width // 2)
    if len(resolutions) != 1:
        raise ValueError(f"DPG-Bench grids have mixed resolutions: {resolutions}")
    return resolutions.pop()


def parse_dpgbench_results(
    path: Path, expected_prompt_ids: set[str]
) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    image_scores: dict[str, float] = {}
    for line in text.splitlines():
        parts = line.split(", ")
        if len(parts) != DPGBENCH_IMAGES_PER_PROMPT + 2:
            continue
        try:
            values = [float(value) for value in parts[1:]]
        except ValueError:
            continue
        prompt_id = Path(parts[0]).stem
        if prompt_id in image_scores:
            raise ValueError(f"duplicate DPG-Bench result: {prompt_id}")
        if any(not 0.0 <= value <= 1.0 for value in values):
            raise ValueError(f"invalid DPG-Bench image score: {prompt_id}")
        image_scores[prompt_id] = values[-1]
    require_exact_ids(image_scores, expected_prompt_ids, "DPG-Bench result rows")

    score_match = re.search(r"DPG-Bench score:\s*([0-9.eE+-]+)", text)
    if score_match is None:
        raise ValueError("official DPG-Bench score is missing")
    official_score = float(score_match.group(1)) / 100.0
    score_from_rows = sum(image_scores.values()) / len(image_scores)
    if not math.isclose(official_score, score_from_rows, rel_tol=0.0, abs_tol=1e-10):
        raise ValueError("DPG-Bench official score and per-image rows disagree")

    section: str | None = None
    category_scores: dict[str, dict[str, float]] = {"l1": {}, "l2": {}}
    for line in text.splitlines():
        if line == "L1 category scores:":
            section = "l1"
            continue
        if line == "L2 category scores:":
            section = "l2"
            continue
        if line.startswith("Image path:"):
            section = None
        if section and line.startswith("\t") and ":" in line:
            name, raw_value = line.strip().rsplit(":", 1)
            value = float(raw_value.strip()) / 100.0
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"invalid DPG-Bench category score: {name}")
            category_scores[section][name] = value
    if not category_scores["l1"] or not category_scores["l2"]:
        raise ValueError("DPG-Bench category breakdown is incomplete")
    return {
        "dpgbench": official_score,
        "l1_category_scores": category_scores["l1"],
        "l2_category_scores": category_scores["l2"],
        "evaluated_grids": len(image_scores),
    }


def run_dpgbench(args: argparse.Namespace) -> None:
    if args.processes <= 0 or not 1 <= args.port <= 65_535:
        raise ValueError("invalid DPG-Bench process/port setting")
    python = _require_python(args.python)
    repository = args.repository.resolve()
    image_dir = args.image_dir.resolve()
    prompts = load_dpgbench_prompts(repository)
    resolution = validate_dpgbench_layout(image_dir, prompts)
    manifest, manifest_path = _load_generation_manifest(
        image_dir,
        benchmark="dpgbench",
        prompt_count=len(prompts),
        images_per_prompt=DPGBENCH_IMAGES_PER_PROMPT,
        source_commit=DPGBENCH_COMMIT,
    )
    if list(map(int, manifest["native_resolution"])) != [resolution, resolution]:
        raise ValueError("DPG-Bench grids disagree with the generation manifest")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = (args.output_dir / "results.txt").resolve()
    command = [
        python,
        "-m",
        "accelerate.commands.launch",
        "--num_machines",
        "1",
        "--num_processes",
        str(args.processes),
        "--mixed_precision",
        "fp16",
        "--main_process_port",
        str(args.port),
    ]
    if args.processes > 1:
        command.append("--multi_gpu")
    command.extend(
        [
            str(repository / "dpg_bench" / "compute_dpg_bench.py"),
            "--image-root-path",
            str(image_dir),
            "--resolution",
            str(resolution),
            "--csv",
            str(repository / "dpg_bench" / "dpg_bench.csv"),
            "--res-path",
            str(raw_path),
            "--pic-num",
            str(DPGBENCH_IMAGES_PER_PROMPT),
            "--vqa-model",
            "mplug",
        ]
    )
    subprocess.run(command, cwd=repository, check=True)
    metrics = parse_dpgbench_results(
        raw_path, {prompt.prompt_id for prompt in prompts}
    )
    result = {
        "schema": METRICS_SCHEMA,
        "complete": True,
        "benchmark": "dpgbench",
        "official_evaluator": {
            "repository": "TencentQQGYLab/ELLA",
            "commit": DPGBENCH_COMMIT,
            "vqa_model": "damo/mplug_visual-question-answering_coco_large_en",
        },
        "metric_unit": "unit_interval",
        "primary_metric": "dpgbench",
        "images_per_prompt": DPGBENCH_IMAGES_PER_PROMPT,
        "sample_resolution": resolution,
        "model_source": manifest["model_source"],
        "generation": manifest["generation"],
        "generation_manifest": str(manifest_path),
        "metrics": metrics,
        "raw_results": str(raw_path),
        "completed_at": _completed_at(),
    }
    _write_json(args.output_dir / "metrics.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


def _validate_mjhq_images(
    root: Path, prompts: list[BenchmarkPrompt], label: str
) -> tuple[dict[str, Path], dict[str, int]]:
    images = index_images(root)
    expected_ids = {prompt.prompt_id for prompt in prompts}
    require_exact_ids(images, expected_ids, label)
    category_by_id = {prompt.prompt_id: prompt.category for prompt in prompts}
    for prompt_id, path in images.items():
        if path.parent.name != category_by_id[prompt_id]:
            raise ValueError(
                f"{label} category mismatch for {prompt_id}: {path.parent.name}"
            )
    return images, _resolution_counts(list(images.values()))


def run_mjhq(args: argparse.Namespace) -> None:
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("invalid clean-fid batch/worker setting")
    python = _require_python(args.python)
    repository = args.repository.resolve()
    prompts = load_mjhq_prompts(repository)
    reference_dir = args.reference_dir.resolve()
    image_dir = args.image_dir.resolve()
    _, reference_resolutions = _validate_mjhq_images(
        reference_dir, prompts, "MJHQ reference images"
    )
    _, generated_resolutions = _validate_mjhq_images(
        image_dir, prompts, "MJHQ generated images"
    )
    manifest, manifest_path = _load_generation_manifest(
        image_dir,
        benchmark="mjhq",
        prompt_count=len(prompts),
        images_per_prompt=1,
        source_commit=MJHQ_COMMIT,
    )
    width, height = map(int, manifest["native_resolution"])
    if generated_resolutions != {f"{width}x{height}": len(prompts)}:
        raise ValueError("MJHQ images disagree with the generation manifest")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = (args.output_dir / "cleanfid_raw.json").resolve()
    worker = Path(__file__).with_name("compute_mjhq_cleanfid.py").resolve()
    subprocess.run(
        [
            python,
            str(worker),
            "--reference_dir",
            str(reference_dir),
            "--generated_dir",
            str(image_dir),
            "--output",
            str(raw_path),
            "--device",
            args.device,
            "--batch_size",
            str(args.batch_size),
            "--num_workers",
            str(args.num_workers),
        ],
        check=True,
    )
    raw = _read_json(raw_path)
    if raw.get("cleanfid_version") != CLEANFID_VERSION:
        raise ValueError("MJHQ worker used an unpinned clean-fid version")
    category_fid = raw.get("category_fid") or {}
    if set(category_fid) != set(MJHQ_CATEGORY_COUNTS):
        raise ValueError("MJHQ category FID coverage is incomplete")
    values = [float(raw["fid"]), *(float(value) for value in category_fid.values())]
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("MJHQ clean-fid produced an invalid FID")
    leaderboard_comparable = (
        reference_resolutions == {"1024x1024": 30_000}
        and generated_resolutions == {"1024x1024": 30_000}
    )
    result = {
        "schema": METRICS_SCHEMA,
        "complete": True,
        "benchmark": "mjhq",
        "official_dataset": {
            "repository": "playgroundai/MJHQ-30K",
            "commit": MJHQ_COMMIT,
        },
        "official_evaluator": {
            "repository": "GaParmar/clean-fid",
            "commit": CLEANFID_COMMIT,
            "version": CLEANFID_VERSION,
            "mode": "clean",
            "model_name": "inception_v3",
        },
        "metric_unit": "fid_distance_lower_is_better",
        "primary_metric": "fid",
        "model_source": manifest["model_source"],
        "generation": manifest["generation"],
        "generation_manifest": str(manifest_path),
        "leaderboard_comparable_at_1024px": leaderboard_comparable,
        "reference_resolutions": reference_resolutions,
        "generated_resolutions": generated_resolutions,
        "metrics": {
            "fid": float(raw["fid"]),
            "category_fid": {
                key: float(value) for key, value in category_fid.items()
            },
        },
        "raw_results": str(raw_path),
        "completed_at": _completed_at(),
    }
    _write_json(args.output_dir / "metrics.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


def run_summary(args: argparse.Namespace) -> None:
    components = {
        "geneval": _read_json(args.geneval),
        "dpgbench": _read_json(args.dpgbench),
        "mjhq": _read_json(args.mjhq),
    }
    for name, value in components.items():
        if (
            value.get("schema") != METRICS_SCHEMA
            or value.get("benchmark") != name
            or value.get("complete") is not True
        ):
            raise ValueError(f"invalid official T2I component: {name}")
    expected_protocols = {
        "geneval": ("unit_interval", "overall", GENEVAL_COMMIT),
        "dpgbench": ("unit_interval", "dpgbench", DPGBENCH_COMMIT),
        "mjhq": ("fid_distance_lower_is_better", "fid", CLEANFID_COMMIT),
    }
    for name, (unit, primary, commit) in expected_protocols.items():
        component = components[name]
        evaluator = component.get("official_evaluator") or {}
        if (
            component.get("metric_unit") != unit
            or component.get("primary_metric") != primary
            or evaluator.get("commit") != commit
        ):
            raise ValueError(f"obsolete official T2I protocol: {name}")
    if (components["mjhq"].get("official_dataset") or {}).get(
        "commit"
    ) != MJHQ_COMMIT or (components["mjhq"].get("official_evaluator") or {}).get(
        "version"
    ) != CLEANFID_VERSION:
        raise ValueError("obsolete MJHQ dataset or clean-fid version")

    model_sources = [component.get("model_source") for component in components.values()]
    if any(not isinstance(source, dict) for source in model_sources) or any(
        source != model_sources[0] for source in model_sources[1:]
    ):
        raise ValueError("official T2I components use different model checkpoints")
    primary_values = {
        "geneval_overall": float(components["geneval"]["metrics"]["overall"]),
        "dpgbench": float(components["dpgbench"]["metrics"]["dpgbench"]),
        "mjhq_fid": float(components["mjhq"]["metrics"]["fid"]),
    }
    if any(
        not math.isfinite(value) or not 0.0 <= value <= 1.0
        for value in (primary_values["geneval_overall"], primary_values["dpgbench"])
    ) or not math.isfinite(primary_values["mjhq_fid"]) or primary_values[
        "mjhq_fid"
    ] < 0.0:
        raise ValueError("official T2I primary metric is out of range")
    result = {
        "schema": SUMMARY_SCHEMA,
        "complete": True,
        "model_source": model_sources[0],
        "primary_metrics": primary_values,
        "mjhq_leaderboard_comparable_at_1024px": components["mjhq"][
            "leaderboard_comparable_at_1024px"
        ],
        "components": {name: str(path.resolve()) for name, path in {
            "geneval": args.geneval,
            "dpgbench": args.dpgbench,
            "mjhq": args.mjhq,
        }.items()},
        "completed_at": _completed_at(),
    }
    _write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


def main() -> None:
    args = parse_args()
    {
        "geneval": run_geneval,
        "dpgbench": run_dpgbench,
        "mjhq": run_mjhq,
        "summary": run_summary,
    }[args.command](args)


if __name__ == "__main__":
    main()
