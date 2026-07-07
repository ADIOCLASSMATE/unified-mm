#!/usr/bin/env python3
"""Run non-EMA mini FID/IS comparisons for the ImageNet flow ablation."""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_BASE_DIR = REPO_ROOT / "output" / "selfless-flow-stage0-imagenet-full-from-qwen3base"
DEFAULT_ABLATION_DIR = (
    REPO_ROOT
    / "output"
    / "selfless-flow-stage0-imagenet-full-from-qwen3base-ablation-image-embedder-nonorm"
)
DEFAULT_INCEPTION_WEIGHTS = (
    REPO_ROOT
    / "output"
    / "cache"
    / "inception"
    / "weights-inception-2015-12-05-6726825d.pth"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare non-EMA ImageNet flow checkpoints with the existing "
            "single-stream FID/IS evaluator."
        )
    )
    parser.add_argument("--step", type=int, default=80000)
    parser.add_argument("--base_dir", type=Path, default=DEFAULT_BASE_DIR)
    parser.add_argument("--ablation_dir", type=Path, default=DEFAULT_ABLATION_DIR)
    parser.add_argument(
        "--models",
        default="base,nonorm",
        help="Comma-separated subset of models to evaluate: base,nonorm.",
    )
    parser.add_argument("--output_dir", type=Path, default=REPO_ROOT / "output" / "ablation_nonema_fid_is")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--nproc_per_model",
        type=int,
        default=4,
        help="Number of torchrun processes/GPU cards used by each model evaluation job.",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--split", choices=["val", "train"], default="val")
    parser.add_argument("--sampling_steps", default="100")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--cfg", type=float, default=1.0)
    parser.add_argument("--cfg_schedule", choices=["constant", "linear"], default="constant")
    parser.add_argument("--flow_solver", choices=["heun", "euler"], default="heun")
    parser.add_argument("--parallel_rate", type=int, default=4)
    parser.add_argument("--strategies", default="causal_sigma,spatial_halton,spatial_uniform,random,hidden_norm,latent_proj_cosine")
    parser.add_argument("--vae_dtype", choices=["fp32", "fp16"], default="fp32")
    parser.add_argument("--fid_feature", type=int, default=2048)
    parser.add_argument("--is_splits", type=int, default=10)
    parser.add_argument(
        "--inception_weights_path",
        default=str(DEFAULT_INCEPTION_WEIGHTS) if DEFAULT_INCEPTION_WEIGHTS.exists() else "",
        help="Optional local torch-fidelity InceptionV3 weights path for FID/IS.",
    )
    parser.add_argument(
        "--real_sources",
        default="imagenet_original",
        help=(
            "Comma-separated FID real distributions. Use imagenet_original for standard mini FID; "
            "add vae_decoded_target_latents to compare against decoded target latents."
        ),
    )
    parser.add_argument(
        "--imagenet_train_dir",
        default="/inspire/dataset/imagenet/v1/ILSVRC/Data/CLS-LOC/train",
    )
    parser.add_argument("--real_image_size", type=int, default=256)
    parser.add_argument("--save_images", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument(
        "--allow_sigma_strategies",
        action="store_true",
        help="Allow sigma/sigma_replay strategies. Disabled by default because real generation cannot know training sigma.",
    )
    parser.add_argument(
        "--allow_missing",
        action="store_true",
        help="Skip missing model directories instead of failing. Useful before the ablation reaches 80k.",
    )
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--no_progress", action="store_true")
    return parser.parse_args()


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def validate_strategies(strategies: list[str], allow_sigma_strategies: bool) -> None:
    if allow_sigma_strategies:
        return
    forbidden = {"sigma", "sigma_replay"}
    found = sorted({strategy.lower() for strategy in strategies} & forbidden)
    if found:
        raise ValueError(
            "Refusing sigma-based generation strategies for real evaluation: "
            f"{found}. These strategies require training/data sigma. "
            "Pass --allow_sigma_strategies only for debugging."
        )


def model_specs(args: argparse.Namespace) -> dict[str, dict[str, Path]]:
    return {
        "base": {
            "run_dir": args.base_dir,
            "config": args.base_dir / "config.yaml",
            "model": args.base_dir / f"hf_model-{args.step}",
        },
        "nonorm": {
            "run_dir": args.ablation_dir,
            "config": args.ablation_dir / "config.yaml",
            "model": args.ablation_dir / f"hf_model-{args.step}",
        },
    }


def validate_non_ema_model(label: str, spec: dict[str, Path], allow_missing: bool) -> bool:
    config_path = spec["config"]
    model_path = spec["model"]
    if model_path.name.endswith("-ema"):
        raise ValueError(f"{label}: refusing to evaluate EMA path: {model_path}")
    missing = [path for path in (config_path, model_path) if not path.exists()]
    if not missing:
        return True
    message = f"{label}: missing required non-EMA evaluation input(s): " + ", ".join(str(p) for p in missing)
    if allow_missing:
        print(f"[skip] {message}", file=sys.stderr)
        return False
    raise FileNotFoundError(message)


def command_for(
    *,
    args: argparse.Namespace,
    spec: dict[str, Path],
    real_source: str,
    run_output_dir: Path,
) -> list[str]:
    evaluator_command = [
        str(REPO_ROOT / "scripts" / "evaluate_single_stream_fid_is.py"),
        "--config",
        str(spec["config"]),
        "--model_path_override",
        str(spec["model"]),
        "--adapter",
        "none",
        "--model_state",
        "",
        "--ema_state",
        "",
        "--output_dir",
        str(run_output_dir),
        "--device",
        str(args.device),
        "--seed",
        str(args.seed),
        "--batch_size",
        str(args.batch_size),
        "--samples",
        str(args.samples),
        "--split",
        str(args.split),
        "--sampling_steps",
        str(args.sampling_steps),
        "--temperature",
        str(args.temperature),
        "--cfg",
        str(args.cfg),
        "--cfg_schedule",
        str(args.cfg_schedule),
        "--flow_solver",
        str(args.flow_solver),
        "--parallel_rate",
        str(args.parallel_rate),
        "--strategies",
        str(args.strategies),
        "--vae_dtype",
        str(args.vae_dtype),
        "--fid_feature",
        str(args.fid_feature),
        "--is_splits",
        str(args.is_splits),
        "--real_source",
        str(real_source),
        "--imagenet_train_dir",
        str(args.imagenet_train_dir),
        "--real_image_size",
        str(args.real_image_size),
    ]
    if args.save_images:
        evaluator_command.append("--save_images")
    if args.inception_weights_path:
        evaluator_command.extend(["--inception_weights_path", str(args.inception_weights_path)])
    if args.allow_sigma_strategies:
        evaluator_command.append("--allow_sigma_strategies")
    if args.no_progress:
        evaluator_command.append("--no_progress")
    if int(args.nproc_per_model) > 1:
        return [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes",
            "1",
            "--nproc_per_node",
            str(args.nproc_per_model),
            *evaluator_command,
        ]
    return [sys.executable, *evaluator_command]


def run_command(command: list[str], *, dry_run: bool) -> None:
    print("$ " + " ".join(shlex.quote(part) for part in command))
    if dry_run:
        return
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def load_metrics(metrics_path: Path) -> dict[str, Any]:
    with metrics_path.open("r", encoding="utf-8") as f:
        metrics = json.load(f)
    ema_state = metrics.get("ema_state", {})
    if ema_state.get("ema_state") is not None:
        raise RuntimeError(f"Expected non-EMA metrics, got EMA state in {metrics_path}: {ema_state}")
    return metrics


def flatten_summary(results: dict[str, dict[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for real_source, by_model in sorted(results.items()):
        base_strategies = by_model.get("base", {}).get("strategies", {})
        for label, metrics in sorted(by_model.items()):
            for strategy, strategy_metrics in sorted(metrics.get("strategies", {}).items()):
                row = {
                    "real_source": real_source,
                    "model": label,
                    "strategy": strategy,
                    "count": strategy_metrics.get("count"),
                    "fid": strategy_metrics.get("fid"),
                    "inception_score_mean": strategy_metrics.get("inception_score_mean"),
                    "inception_score_std": strategy_metrics.get("inception_score_std"),
                    "latent_mse_to_target": strategy_metrics.get("latent_mse_to_target"),
                    "latent_rms": strategy_metrics.get("latent_rms"),
                    "generation_step_max": strategy_metrics.get("generation_step_max"),
                }
                base = base_strategies.get(strategy)
                if base is not None and label != "base":
                    row["fid_delta_vs_base"] = delta(strategy_metrics.get("fid"), base.get("fid"))
                    row["inception_score_delta_vs_base"] = delta(
                        strategy_metrics.get("inception_score_mean"),
                        base.get("inception_score_mean"),
                    )
                    row["latent_mse_delta_vs_base"] = delta(
                        strategy_metrics.get("latent_mse_to_target"),
                        base.get("latent_mse_to_target"),
                    )
                rows.append(row)
    return rows


def delta(value: Any, baseline: Any) -> float | None:
    if isinstance(value, (int, float)) and isinstance(baseline, (int, float)):
        return float(value) - float(baseline)
    return None


def write_summary(output_dir: Path, results: dict[str, dict[str, dict[str, Any]]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_json = output_dir / "summary.json"
    summary_json.write_text(json.dumps(results, indent=2), encoding="utf-8")

    rows = flatten_summary(results)
    summary_csv = output_dir / "summary.csv"
    fieldnames = [
        "real_source",
        "model",
        "strategy",
        "count",
        "fid",
        "fid_delta_vs_base",
        "inception_score_mean",
        "inception_score_delta_vs_base",
        "inception_score_std",
        "latent_mse_to_target",
        "latent_mse_delta_vs_base",
        "latent_rms",
        "generation_step_max",
    ]
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})

    print(f"Saved summary JSON: {summary_json}")
    print(f"Saved summary CSV: {summary_csv}")


def main() -> None:
    args = parse_args()
    requested_models = split_csv(args.models)
    real_sources = split_csv(args.real_sources)
    if not requested_models:
        raise ValueError("--models must not be empty")
    if not real_sources:
        raise ValueError("--real_sources must not be empty")
    validate_strategies(split_csv(args.strategies), args.allow_sigma_strategies)

    specs = model_specs(args)
    unknown = sorted(set(requested_models) - set(specs))
    if unknown:
        raise ValueError(f"Unknown model labels in --models: {unknown}; expected one of {sorted(specs)}")

    active_models = []
    for label in requested_models:
        if validate_non_ema_model(label, specs[label], args.allow_missing):
            active_models.append(label)
    if not active_models:
        raise RuntimeError("No requested non-EMA model directories are available.")

    eval_root = args.output_dir / f"step{args.step}" / f"samples{args.samples}_cfg{args.cfg:g}_{args.cfg_schedule}"
    results: dict[str, dict[str, dict[str, Any]]] = {}

    for real_source in real_sources:
        results.setdefault(real_source, {})
        for label in active_models:
            spec = specs[label]
            run_output_dir = eval_root / real_source / label
            metrics_path = run_output_dir / "metrics.json"
            if args.skip_existing and metrics_path.exists():
                print(f"[skip existing] {metrics_path}")
            else:
                command = command_for(
                    args=args,
                    spec=spec,
                    real_source=real_source,
                    run_output_dir=run_output_dir,
                )
                run_command(command, dry_run=args.dry_run)
            if metrics_path.exists() and not args.dry_run:
                results[real_source][label] = load_metrics(metrics_path)

    if args.dry_run:
        print("Dry run only; no metrics were loaded.")
        return
    if results:
        write_summary(eval_root, results)
    else:
        raise RuntimeError("No metrics were produced. Check missing models or failed evaluator runs.")


if __name__ == "__main__":
    main()
