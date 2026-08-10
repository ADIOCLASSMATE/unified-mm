#!/usr/bin/env python3
"""Strict acceptance checks for canonical ImageNet-100 Ascend FID/IS."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import torch


def require_file(path: Path, label: str) -> Path:
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"missing or empty {label}: {path}")
    return path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_equal(actual, expected, label: str) -> None:
    if actual != expected:
        raise RuntimeError(f"{label} mismatch: {actual!r} != {expected!r}")


def require_finite(value, label: str, *, positive: bool = False) -> float:
    value = float(value)
    if not math.isfinite(value) or (positive and value <= 0):
        raise RuntimeError(f"{label} must be finite{' and positive' if positive else ''}: {value}")
    return value


def write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--real_stats", required=True)
    parser.add_argument("--inception_weights", required=True)
    parser.add_argument("--output_json", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics_path = require_file(Path(args.metrics), "evaluation metrics")
    checkpoint = Path(args.checkpoint)
    real_stats_path = require_file(Path(args.real_stats), "real-stat cache")
    weights_path = require_file(Path(args.inception_weights), "Inception weights")
    for filename in ("config.json", "model.safetensors", "tokenizer.json"):
        require_file(checkpoint / filename, f"EMA checkpoint {filename}")

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    require_equal(metrics.get("official_protocol"), True, "official protocol")
    require_equal(metrics.get("samples_requested"), 10000, "requested samples")
    require_equal(metrics.get("samples_evaluated"), 10000, "evaluated samples")
    require_equal(metrics.get("real_source"), "cached_original_imagenet", "real source")
    require_equal(metrics.get("target_decode_skipped"), True, "target decode")
    require_equal(metrics.get("sampling_steps"), "100", "sampling steps")
    require_equal(metrics.get("cfg"), 3.5, "CFG")
    require_equal(metrics.get("cfg_schedule"), "constant", "CFG schedule")
    require_equal(metrics.get("flow_solver"), "heun", "flow solver")
    require_equal(metrics.get("parallel_rate"), 1, "parallel rate")

    distributed = metrics.get("distributed", {})
    expected_distributed = {
        "enabled": True,
        "world_size": 16,
        "batch_size_global": 4096,
        "batch_size_per_rank": 256,
        "device_type": "npu",
        "distributed_backend": "hccl",
    }
    for field, expected in expected_distributed.items():
        require_equal(distributed.get(field), expected, f"distributed.{field}")
    require_finite(
        distributed.get("peak_device_allocated_mib"),
        "peak allocated memory",
        positive=True,
    )

    precision = metrics.get("precision_protocol", {})
    expected_precision = {
        "model_dtype": "bf16",
        "model_parameter_dtypes": ["torch.bfloat16"],
        "vae_dtype": "fp32",
        "flow_integrator_dtype": "fp32",
        "metric_accumulation_dtype": "torch.float32",
        "autocast_enabled": False,
    }
    for field, expected in expected_precision.items():
        require_equal(precision.get(field), expected, f"precision.{field}")

    contracts = metrics.get("implementation_contracts", {})
    require_equal(
        contracts.get("canonical_initial_noise_enabled"),
        True,
        "canonical initial noise",
    )
    require_equal(
        contracts.get("paired_sample_count"),
        10000,
        "paired sample count",
    )
    require_equal(
        metrics.get("mechanism_diagnostics", {}).get(
            "generated_latent_finite_rate"
        ),
        1.0,
        "generated latent finite rate",
    )

    strategies = metrics.get("strategies", {})
    require_equal(sorted(strategies), ["spatial_halton"], "strategies")
    strategy = strategies["spatial_halton"]
    require_equal(strategy.get("count"), 10000, "strategy sample count")
    fid = require_finite(strategy.get("fid"), "FID")
    if fid < 0:
        raise RuntimeError(f"FID must be nonnegative: {fid}")
    inception_score = require_finite(
        strategy.get("inception_score_mean"),
        "Inception Score",
        positive=True,
    )
    require_finite(strategy.get("inception_score_std"), "Inception Score std")
    require_finite(
        strategy.get("generation_samples_per_second"),
        "generation throughput",
        positive=True,
    )

    real_stats = torch.load(real_stats_path, map_location="cpu")
    require_equal(real_stats["stats"]["count"], 10000, "real-stat count")
    recorded_weights_hash = real_stats.get("metadata", {}).get(
        "feature", {}
    ).get("weights_sha256")
    weights_hash = sha256(weights_path)
    require_equal(
        recorded_weights_hash,
        weights_hash,
        "real-stat Inception weights hash",
    )
    report = {
        "schema": "ascend_imagenet100_fid_is_acceptance_v1",
        "status": "ok",
        "metrics": {
            "path": str(metrics_path.resolve()),
            "sha256": sha256(metrics_path),
            "fid": fid,
            "inception_score_mean": inception_score,
            "samples": 10000,
        },
        "checkpoint": {
            "path": str(checkpoint.resolve()),
            "model_sha256": sha256(checkpoint / "model.safetensors"),
        },
        "real_stats": {
            "path": str(real_stats_path.resolve()),
            "sha256": sha256(real_stats_path),
        },
        "inception_weights": {
            "path": str(weights_path.resolve()),
            "sha256": weights_hash,
        },
        "distributed": distributed,
        "precision_protocol": precision,
    }
    if args.output_json:
        write_json_atomic(Path(args.output_json), report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
