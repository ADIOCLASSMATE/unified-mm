#!/usr/bin/env python3
"""Build deterministic JSON/CSV summaries from validated Flow CFG metrics.

The tool is deliberately offline: ``--job`` values are labels only and no
platform API is queried.  Each ``--point`` must name one completed
``metrics.json`` produced by the formal single-stream Flow evaluator.  The
tool validates the per-point metric contract, requires all protocol-defining
fields to match, then atomically replaces each requested summary file.

Example::

    python scripts/summarize_flow_cfg_sweep.py \
      --point 1.5=output/run/cfg_1p5/metrics.json \
      --point 2.0=output/run/cfg_2p0/metrics.json \
      --job 1.5=flow-cfg-1p5 \
      --job 2.0=flow-cfg-2p0 \
      --checkpoint-sha256 0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef \
      --output-json output/run/summary.json \
      --output-csv output/run/summary.csv
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA = "selfless_flow_cfg_sweep_summary_v1"
STRATEGY = "spatial_halton"
SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
FORMAL_SAMPLE_COUNT = 10_000
FORMAL_WORLD_SIZE = 8
FORMAL_BATCH_SIZE = 512
FORMAL_SEED = 42
FORMAL_GENERATION_STEP_MAX = 256.0
EXPECTED_SELECTION_SHA256 = (
    "f862b4d0bc1c48e089b97f75f651a91d23224fa26bb367347531f545f8162845"
)
EXPECTED_MANIFEST_SHA256 = (
    "6c6fc84e6ec9cb8b92421659d44abbb24d1cc34558213d9fb9db7e4ec44f1c3a"
)
EXPECTED_SPLIT_MANIFEST_SHA256 = (
    "02e5c67c058f95bcca46c82f3c1fc81086f61dcec62ce25049843f09d930a5ba"
)
EXPECTED_INCEPTION_SHA256 = (
    "6726825d0af5f729cebd5821db510b11b1cfad8faad88a03f1befd49fb9129b2"
)
EXPECTED_CONFIG = "configs/ablation/imagenet_flow_100c_80ep.yaml"
EXPECTED_SPLIT_PROTOCOL = {
    "source": "authoritative_split_manifest",
    "order": "validation_split_index",
    "strategy": "stratified",
    "seed": 42,
    "val_samples_per_class": 100,
}
EXPECTED_PROMPT_PROTOCOL = {
    "type": "class_name",
    "mapping": "first comma-separated LOC_synset_mapping name",
}
EXPECTED_TRANSFORM_PROTOCOL = {
    "input_color": "RGB",
    "resize": {
        "type": "shorter_side",
        "size": 256,
        "interpolation": "bicubic",
        "antialias": True,
    },
    "crop": {"type": "center", "height": 256, "width": 256},
    "tensor_range": [0.0, 1.0],
    "inception_input": "uint8_0_255",
    "quantization": "multiply_255_then_uint8_truncate",
    "inception_resize": {
        "size": 299,
        "interpolation": "bilinear",
        "antialias": True,
    },
}
EXPECTED_FEATURE_PROTOCOL = {
    "backend": "torchmetrics.NoTrainInceptionV3/torch-fidelity",
    "extractor_antialias": True,
    "feature": 2048,
    "feature_name": "2048",
    "logits_name": "logits_unbiased",
    "weights_sha256": EXPECTED_INCEPTION_SHA256,
    "weights_filename": "weights-inception-2015-12-05-6726825d.pth",
}


class SummaryError(ValueError):
    """Raised when an input cannot be summarized safely."""


def _parse_cfg_assignment(spec: str, *, option: str) -> tuple[float, str]:
    raw_cfg, separator, value = spec.partition("=")
    if not separator or not raw_cfg.strip() or not value.strip():
        raise SummaryError(
            f"{option} must use CFG=VALUE syntax, got {spec!r}"
        )
    try:
        cfg = float(raw_cfg)
    except ValueError as exc:
        raise SummaryError(f"invalid CFG in {option} {spec!r}") from exc
    if not math.isfinite(cfg) or cfg <= 0.0:
        raise SummaryError(f"CFG in {option} must be finite and positive: {spec!r}")
    return cfg, value.strip()


def _finite_number(value: Any, *, field: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SummaryError(f"{field} must be a finite number, got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise SummaryError(f"{field} must be finite, got {value!r}")
    if minimum is not None and result < minimum:
        raise SummaryError(f"{field} must be >= {minimum}, got {value!r}")
    return result


def _integer(value: Any, *, field: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SummaryError(f"{field} must be an integer, got {value!r}")
    if minimum is not None and value < minimum:
        raise SummaryError(f"{field} must be >= {minimum}, got {value!r}")
    return value


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SummaryError(f"{field} must be an object")
    return value


def _required(payload: Mapping[str, Any], key: str, *, source: Path) -> Any:
    if key not in payload:
        raise SummaryError(f"{source}: missing required field {key!r}")
    return payload[key]


def _count_pngs(directory: Path) -> int | None:
    if not directory.exists():
        return None
    if not directory.is_dir():
        raise SummaryError(f"expected image directory, found non-directory: {directory}")
    try:
        with os.scandir(directory) as entries:
            return sum(
                1
                for entry in entries
                if entry.name.lower().endswith(".png") and entry.is_file()
            )
    except OSError as exc:
        raise SummaryError(f"failed to count images in {directory}: {exc}") from exc


def _load_point(
    cfg: float,
    metrics_path: Path,
    *,
    job_name: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        resolved_metrics_path = metrics_path.expanduser().resolve(strict=True)
    except OSError as exc:
        raise SummaryError(f"metrics file does not exist: {metrics_path}") from exc
    if not resolved_metrics_path.is_file():
        raise SummaryError(f"metrics path is not a file: {resolved_metrics_path}")
    try:
        payload = json.loads(resolved_metrics_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SummaryError(f"failed to read JSON metrics {resolved_metrics_path}: {exc}") from exc
    payload = _mapping(payload, field=str(resolved_metrics_path))

    metrics_cfg = _finite_number(
        _required(payload, "cfg", source=resolved_metrics_path),
        field=f"{resolved_metrics_path}: cfg",
    )
    if not math.isclose(metrics_cfg, cfg, rel_tol=0.0, abs_tol=1.0e-12):
        raise SummaryError(
            f"{resolved_metrics_path}: --point CFG={cfg} does not match metrics cfg={metrics_cfg}"
        )
    if _required(payload, "official_protocol", source=resolved_metrics_path) is not True:
        raise SummaryError(f"{resolved_metrics_path}: official_protocol must be true")
    if _required(payload, "config", source=resolved_metrics_path) != EXPECTED_CONFIG:
        raise SummaryError(
            f"{resolved_metrics_path}: unexpected formal config path"
        )
    expected_states = {
        "adapter": {"adapter": None},
        "model_state": {"model_state": None},
        "ema_state": {"ema_state": None},
    }
    invalid_states = [
        key for key, expected in expected_states.items() if payload.get(key) != expected
    ]
    if invalid_states:
        raise SummaryError(
            f"{resolved_metrics_path}: unexpected runtime state overlays: "
            + ", ".join(invalid_states)
        )

    precision_protocol = payload.get("precision_protocol")
    legacy_bf16_metrics = precision_protocol is None
    if legacy_bf16_metrics:
        precision_protocol = {}
    elif not isinstance(precision_protocol, Mapping):
        raise SummaryError(
            f"{resolved_metrics_path}: precision_protocol must be an object"
        )
    model_dtype = precision_protocol.get("model_dtype", "bf16")
    if model_dtype not in {"bf16", "fp32"}:
        raise SummaryError(
            f"{resolved_metrics_path}: unexpected model_dtype={model_dtype!r}"
        )
    expected_parameter_dtypes = {
        "bf16": ["torch.bfloat16"],
        "fp32": ["torch.float32"],
    }
    if not legacy_bf16_metrics and precision_protocol.get(
        "model_parameter_dtypes"
    ) != expected_parameter_dtypes[model_dtype]:
        raise SummaryError(
            f"{resolved_metrics_path}: model_parameter_dtypes does not match "
            f"model_dtype={model_dtype!r}"
        )
    expected_precision_policy = {
        "schema": "flow_eval_precision_v1",
        "vae_dtype": "fp32",
        "flow_integrator_dtype": "fp32",
        "autocast_enabled": False,
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": True,
        "float32_matmul_precision": "highest",
    }
    actual_precision_policy = {
        key: precision_protocol.get(key) for key in expected_precision_policy
    }
    if not legacy_bf16_metrics and actual_precision_policy != expected_precision_policy:
        raise SummaryError(
            f"{resolved_metrics_path}: precision_protocol policy mismatch"
        )
    checkpoint_weight_dtypes = precision_protocol.get(
        "checkpoint_weight_dtypes"
    )
    model_path = _required(payload, "model_path", source=resolved_metrics_path)
    expected_checkpoint_dtypes = {
        "output/selfless-flow-ablation-imagenet100-80ep/hf_model-final-ema": [
            "fp32"
        ],
        "output/selfless-flow-ablation-imagenet100-80ep/hf_model-final": [
            "bf16"
        ],
    }.get(model_path)
    if not legacy_bf16_metrics and not (
        checkpoint_weight_dtypes == expected_checkpoint_dtypes
        if expected_checkpoint_dtypes is not None
        else (
            isinstance(checkpoint_weight_dtypes, list)
            and bool(checkpoint_weight_dtypes)
        )
    ):
        raise SummaryError(
            f"{resolved_metrics_path}: checkpoint_weight_dtypes does not match "
            f"model_path={model_path!r}"
        )

    metric_protocol = _mapping(
        _required(payload, "metric_protocol", source=resolved_metrics_path),
        field=f"{resolved_metrics_path}: metric_protocol",
    )
    is_splits = _integer(
        _required(metric_protocol, "is_splits", source=resolved_metrics_path),
        field=f"{resolved_metrics_path}: metric_protocol.is_splits",
        minimum=1,
    )
    if is_splits != 10:
        raise SummaryError(
            f"{resolved_metrics_path}: formal protocol requires 10 IS splits, got {is_splits}"
        )
    expected_metric_protocol = {
        "fid_reducer": "symmetric_eigendecomposition",
        "is_split_assignment": "contiguous_by_global_sample_index",
        "is_std": "population",
        "is_splits": 10,
    }
    if dict(metric_protocol) != expected_metric_protocol:
        raise SummaryError(
            f"{resolved_metrics_path}: metric_protocol does not match the formal contract"
        )

    split = _required(payload, "split", source=resolved_metrics_path)
    if split != "val":
        raise SummaryError(f"{resolved_metrics_path}: split must be 'val', got {split!r}")
    parallel_rate = _integer(
        _required(payload, "parallel_rate", source=resolved_metrics_path),
        field=f"{resolved_metrics_path}: parallel_rate",
        minimum=1,
    )
    if parallel_rate != 1:
        raise SummaryError(
            f"{resolved_metrics_path}: formal protocol requires parallel_rate=1"
        )

    samples_requested = _integer(
        _required(payload, "samples_requested", source=resolved_metrics_path),
        field=f"{resolved_metrics_path}: samples_requested",
        minimum=1,
    )
    samples_evaluated = _integer(
        _required(payload, "samples_evaluated", source=resolved_metrics_path),
        field=f"{resolved_metrics_path}: samples_evaluated",
        minimum=1,
    )
    if samples_requested != FORMAL_SAMPLE_COUNT or samples_evaluated != FORMAL_SAMPLE_COUNT:
        raise SummaryError(
            f"{resolved_metrics_path}: formal protocol requires exactly "
            f"{FORMAL_SAMPLE_COUNT} requested/evaluated samples, got "
            f"{samples_requested}/{samples_evaluated}"
        )

    distributed = _mapping(
        _required(payload, "distributed", source=resolved_metrics_path),
        field=f"{resolved_metrics_path}: distributed",
    )
    if _required(distributed, "enabled", source=resolved_metrics_path) is not True:
        raise SummaryError(f"{resolved_metrics_path}: distributed.enabled must be true")
    world_size = _integer(
        _required(distributed, "world_size", source=resolved_metrics_path),
        field=f"{resolved_metrics_path}: distributed.world_size",
        minimum=1,
    )
    if world_size != FORMAL_WORLD_SIZE:
        raise SummaryError(
            f"{resolved_metrics_path}: formal protocol requires world_size="
            f"{FORMAL_WORLD_SIZE}, got {world_size}"
        )
    rank = _integer(
        _required(distributed, "rank", source=resolved_metrics_path),
        field=f"{resolved_metrics_path}: distributed.rank",
        minimum=0,
    )
    local_rank = _integer(
        _required(distributed, "local_rank", source=resolved_metrics_path),
        field=f"{resolved_metrics_path}: distributed.local_rank",
        minimum=0,
    )
    if rank != 0 or local_rank != 0:
        raise SummaryError(
            f"{resolved_metrics_path}: summary metrics must be written by rank/local_rank 0"
        )
    peak_allocated = _finite_number(
        _required(distributed, "peak_cuda_allocated_mib", source=resolved_metrics_path),
        field=f"{resolved_metrics_path}: distributed.peak_cuda_allocated_mib",
        minimum=0.0,
    )
    peak_reserved = _finite_number(
        _required(distributed, "peak_cuda_reserved_mib", source=resolved_metrics_path),
        field=f"{resolved_metrics_path}: distributed.peak_cuda_reserved_mib",
        minimum=0.0,
    )

    real_stats_metadata = _mapping(
        _required(payload, "real_stats_metadata", source=resolved_metrics_path),
        field=f"{resolved_metrics_path}: real_stats_metadata",
    )
    if _required(payload, "real_source", source=resolved_metrics_path) != (
        "cached_original_imagenet"
    ):
        raise SummaryError(
            f"{resolved_metrics_path}: real_source must be cached_original_imagenet"
        )
    real_sample_count = _integer(
        _required(real_stats_metadata, "num_samples", source=resolved_metrics_path),
        field=f"{resolved_metrics_path}: real_stats_metadata.num_samples",
        minimum=1,
    )
    if real_sample_count != samples_evaluated:
        raise SummaryError(
            f"{resolved_metrics_path}: real/fake sample counts differ: "
            f"{real_sample_count} != {samples_evaluated}"
        )
    if real_stats_metadata.get("selection_sha256") != EXPECTED_SELECTION_SHA256:
        raise SummaryError(
            f"{resolved_metrics_path}: unexpected ImageNet-100 selection_sha256"
        )
    real_contract_checks = {
        "schema": real_stats_metadata.get("schema")
        == "qwen_showo_imagenet100_real_stats_v1",
        "protocol": real_stats_metadata.get("protocol")
        == "imagenet100-balanced-val100-per-class-class-name-v1",
        "real_source": real_stats_metadata.get("real_source")
        == "original_imagenet",
        "manifest_sha256": real_stats_metadata.get("manifest_sha256")
        == EXPECTED_MANIFEST_SHA256,
        "split_manifest_sha256": real_stats_metadata.get("split_manifest_sha256")
        == EXPECTED_SPLIT_MANIFEST_SHA256,
        "num_classes": real_stats_metadata.get("num_classes") == 100,
        "split": real_stats_metadata.get("split") == EXPECTED_SPLIT_PROTOCOL,
        "prompt": real_stats_metadata.get("prompt") == EXPECTED_PROMPT_PROTOCOL,
        "transform": real_stats_metadata.get("transform")
        == EXPECTED_TRANSFORM_PROTOCOL,
    }
    class_counts = real_stats_metadata.get("class_counts")
    real_contract_checks["class_counts"] = (
        isinstance(class_counts, dict)
        and len(class_counts) == 100
        and set(class_counts.values()) == {100}
    )
    failed_real_checks = [
        name for name, passed in real_contract_checks.items() if not passed
    ]
    if failed_real_checks:
        raise SummaryError(
            f"{resolved_metrics_path}: real-stats contract failed: "
            + ", ".join(failed_real_checks)
        )
    feature_metadata = _mapping(
        _required(real_stats_metadata, "feature", source=resolved_metrics_path),
        field=f"{resolved_metrics_path}: real_stats_metadata.feature",
    )
    if feature_metadata.get("weights_sha256") != EXPECTED_INCEPTION_SHA256:
        raise SummaryError(
            f"{resolved_metrics_path}: unexpected Inception weights_sha256"
        )
    critical_feature = {
        key: feature_metadata.get(key) for key in EXPECTED_FEATURE_PROTOCOL
    }
    if critical_feature != EXPECTED_FEATURE_PROTOCOL:
        raise SummaryError(
            f"{resolved_metrics_path}: Inception feature contract mismatch"
        )

    strategies = _mapping(
        _required(payload, "strategies", source=resolved_metrics_path),
        field=f"{resolved_metrics_path}: strategies",
    )
    if set(strategies) != {STRATEGY}:
        raise SummaryError(
            f"{resolved_metrics_path}: expected only strategy {STRATEGY!r}, "
            f"got {sorted(strategies)}"
        )
    strategy = _mapping(
        strategies[STRATEGY],
        field=f"{resolved_metrics_path}: strategies.{STRATEGY}",
    )
    count = _integer(
        _required(strategy, "count", source=resolved_metrics_path),
        field=f"{resolved_metrics_path}: strategies.{STRATEGY}.count",
        minimum=1,
    )
    if count != samples_evaluated:
        raise SummaryError(
            f"{resolved_metrics_path}: strategy count {count} != samples {samples_evaluated}"
        )

    fid = _finite_number(
        _required(strategy, "fid", source=resolved_metrics_path),
        field=f"{resolved_metrics_path}: strategies.{STRATEGY}.fid",
        minimum=0.0,
    )
    is_mean = _finite_number(
        _required(strategy, "inception_score_mean", source=resolved_metrics_path),
        field=f"{resolved_metrics_path}: strategies.{STRATEGY}.inception_score_mean",
        minimum=0.0,
    )
    is_std = _finite_number(
        _required(strategy, "inception_score_std", source=resolved_metrics_path),
        field=f"{resolved_metrics_path}: strategies.{STRATEGY}.inception_score_std",
        minimum=0.0,
    )
    is_split_values = _required(strategy, "inception_score_splits", source=resolved_metrics_path)
    if not isinstance(is_split_values, list) or len(is_split_values) != is_splits:
        raise SummaryError(
            f"{resolved_metrics_path}: inception_score_splits must contain {is_splits} values"
        )
    for index, value in enumerate(is_split_values):
        _finite_number(
            value,
            field=f"{resolved_metrics_path}: inception_score_splits[{index}]",
            minimum=0.0,
        )
    latent_mse = _finite_number(
        _required(strategy, "latent_mse_to_target", source=resolved_metrics_path),
        field=f"{resolved_metrics_path}: strategies.{STRATEGY}.latent_mse_to_target",
        minimum=0.0,
    )
    latent_rms = _finite_number(
        _required(strategy, "latent_rms", source=resolved_metrics_path),
        field=f"{resolved_metrics_path}: strategies.{STRATEGY}.latent_rms",
        minimum=0.0,
    )
    generation_step_max = _finite_number(
        _required(strategy, "generation_step_max", source=resolved_metrics_path),
        field=f"{resolved_metrics_path}: strategies.{STRATEGY}.generation_step_max",
        minimum=1.0,
    )
    if generation_step_max != FORMAL_GENERATION_STEP_MAX:
        raise SummaryError(
            f"{resolved_metrics_path}: generation_step_max must be "
            f"{FORMAL_GENERATION_STEP_MAX}, got {generation_step_max}"
        )

    seed = _integer(
        _required(payload, "seed", source=resolved_metrics_path),
        field=f"{resolved_metrics_path}: seed",
    )
    batch_size = _integer(
        _required(payload, "batch_size", source=resolved_metrics_path),
        field=f"{resolved_metrics_path}: batch_size",
        minimum=1,
    )
    formal_generation_checks = {
        "seed": seed == FORMAL_SEED,
        "batch_size": batch_size == FORMAL_BATCH_SIZE,
        "cfg_schedule": payload.get("cfg_schedule") == "constant",
        "sampling_steps": str(payload.get("sampling_steps")) == "100",
        "temperature": payload.get("temperature") == 1.0,
        "flow_solver": payload.get("flow_solver") == "heun",
        "real_image_size": payload.get("real_image_size") == 256,
    }
    failed_generation_checks = [
        name for name, passed in formal_generation_checks.items() if not passed
    ]
    if failed_generation_checks:
        raise SummaryError(
            f"{resolved_metrics_path}: formal generation contract failed: "
            + ", ".join(failed_generation_checks)
        )

    output_path = resolved_metrics_path.parent
    generated_image_count = _count_pngs(output_path / STRATEGY)
    target_image_count = _count_pngs(output_path / "target_decoded")

    row = {
        "cfg": cfg,
        "model_dtype": model_dtype,
        "fid": fid,
        "inception_score_mean": is_mean,
        "inception_score_std": is_std,
        "latent_mse_to_target": latent_mse,
        "latent_rms": latent_rms,
        "generation_step_max": generation_step_max,
        "peak_cuda_allocated_mib": peak_allocated,
        "peak_cuda_reserved_mib": peak_reserved,
        "metrics_path": str(resolved_metrics_path),
        "output_path": str(output_path),
        "image_counts": {
            STRATEGY: generated_image_count,
            "target_decoded": target_image_count,
        },
        "job_name": job_name,
    }

    invariant = {
        "official_protocol": True,
        "metric_protocol": dict(metric_protocol),
        "config": _required(payload, "config", source=resolved_metrics_path),
        "model_path": model_path,
        "model_dtype": model_dtype,
        "checkpoint_weight_dtypes": (
            ["legacy_unrecorded"]
            if legacy_bf16_metrics
            else checkpoint_weight_dtypes
        ),
        "adapter": _required(payload, "adapter", source=resolved_metrics_path),
        "model_state": _required(payload, "model_state", source=resolved_metrics_path),
        "ema_state": _required(payload, "ema_state", source=resolved_metrics_path),
        "split": split,
        "seed": seed,
        "batch_size": batch_size,
        "samples_requested": samples_requested,
        "samples_evaluated": samples_evaluated,
        "distributed": {
            "enabled": True,
            "world_size": world_size,
            "rank": rank,
            "local_rank": local_rank,
        },
        "real_source": _required(payload, "real_source", source=resolved_metrics_path),
        "real_stats_path": _required(payload, "real_stats_path", source=resolved_metrics_path),
        "real_stats_metadata": dict(real_stats_metadata),
        "imagenet_train_dir": _required(payload, "imagenet_train_dir", source=resolved_metrics_path),
        "real_image_size": _required(payload, "real_image_size", source=resolved_metrics_path),
        "cfg_schedule": _required(payload, "cfg_schedule", source=resolved_metrics_path),
        "sampling_steps": _required(payload, "sampling_steps", source=resolved_metrics_path),
        "temperature": _finite_number(
            _required(payload, "temperature", source=resolved_metrics_path),
            field=f"{resolved_metrics_path}: temperature",
            minimum=0.0,
        ),
        "flow_solver": _required(payload, "flow_solver", source=resolved_metrics_path),
        "parallel_rate": parallel_rate,
        "inception_weights_path": _required(
            payload, "inception_weights_path", source=resolved_metrics_path
        ),
        "strategy": STRATEGY,
        "generation_step_max": generation_step_max,
    }
    return row, invariant


def _different_invariants(left: Mapping[str, Any], right: Mapping[str, Any]) -> list[str]:
    keys = sorted(set(left) | set(right))
    return [
        key
        for key in keys
        if key not in left
        or key not in right
        or json.dumps(left[key], sort_keys=True, ensure_ascii=False)
        != json.dumps(right[key], sort_keys=True, ensure_ascii=False)
    ]


def _best_reference(row: Mapping[str, Any], *, metric: str) -> dict[str, Any]:
    reference = {
        "cfg": row["cfg"],
        metric: row[metric],
        "metrics_path": row["metrics_path"],
        "output_path": row["output_path"],
        "job_name": row["job_name"],
    }
    if metric == "inception_score_mean":
        reference["inception_score_std"] = row["inception_score_std"]
    return reference


def build_summary(
    point_specs: Sequence[str],
    job_specs: Sequence[str],
    checkpoint_sha256: str | None,
) -> dict[str, Any]:
    if not point_specs:
        raise SummaryError("at least one --point CFG=METRICS_PATH is required")

    jobs: dict[float, str] = {}
    for spec in job_specs:
        cfg, job_name = _parse_cfg_assignment(spec, option="--job")
        if cfg in jobs:
            raise SummaryError(f"duplicate --job CFG: {cfg}")
        jobs[cfg] = job_name

    points: dict[float, Path] = {}
    for spec in point_specs:
        cfg, raw_path = _parse_cfg_assignment(spec, option="--point")
        if cfg in points:
            raise SummaryError(f"duplicate --point CFG: {cfg}")
        points[cfg] = Path(raw_path)
    unknown_job_cfgs = sorted(set(jobs) - set(points))
    if unknown_job_cfgs:
        raise SummaryError(
            "--job provided for CFG values without --point: "
            + ", ".join(str(value) for value in unknown_job_cfgs)
        )

    normalized_sha256 = None
    if checkpoint_sha256 is not None:
        if not SHA256_RE.fullmatch(checkpoint_sha256):
            raise SummaryError("--checkpoint-sha256 must contain exactly 64 hexadecimal characters")
        normalized_sha256 = checkpoint_sha256.lower()

    rows: list[dict[str, Any]] = []
    common_protocol: dict[str, Any] | None = None
    baseline_path: str | None = None
    for cfg in sorted(points):
        row, invariant = _load_point(
            cfg,
            points[cfg],
            job_name=jobs.get(cfg),
        )
        if common_protocol is None:
            common_protocol = invariant
            baseline_path = row["metrics_path"]
        else:
            differences = _different_invariants(common_protocol, invariant)
            if differences:
                raise SummaryError(
                    f"{row['metrics_path']}: protocol invariants differ from "
                    f"{baseline_path}: {', '.join(differences)}"
                )
        rows.append(row)

    assert common_protocol is not None
    best_fid = min(rows, key=lambda row: (row["fid"], row["cfg"]))
    best_is = min(
        rows,
        key=lambda row: (-row["inception_score_mean"], row["cfg"]),
    )
    return {
        "schema": SCHEMA,
        "checkpoint_sha256": normalized_sha256,
        "point_count": len(rows),
        "cfg_values": [row["cfg"] for row in rows],
        "protocol": common_protocol,
        "points": rows,
        "best_by_fid": _best_reference(best_fid, metric="fid"),
        "best_by_is": _best_reference(best_is, metric="inception_score_mean"),
    }


CSV_FIELDS = (
    "cfg",
    "model_dtype",
    "fid",
    "inception_score_mean",
    "inception_score_std",
    "latent_mse_to_target",
    "latent_rms",
    "generation_step_max",
    "peak_cuda_allocated_mib",
    "peak_cuda_reserved_mib",
    "metrics_path",
    "output_path",
    "generated_image_count",
    "target_image_count",
    "job_name",
)


def render_csv(summary: Mapping[str, Any]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    for point in summary["points"]:
        row = dict(point)
        image_counts = row.pop("image_counts")
        row["generated_image_count"] = image_counts[STRATEGY]
        row["target_image_count"] = image_counts["target_decoded"]
        writer.writerow({field: row.get(field) for field in CSV_FIELDS})
    return buffer.getvalue()


def atomic_write_text(path: Path, content: str) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize validated formal Flow CFG metrics without querying the "
            "job platform. Protocol-defining fields must match across all points."
        )
    )
    parser.add_argument(
        "--point",
        action="append",
        default=[],
        metavar="CFG=METRICS_PATH",
        help="Add one CFG point and its completed metrics.json (repeatable).",
    )
    parser.add_argument(
        "--job",
        action="append",
        default=[],
        metavar="CFG=JOB_NAME",
        help="Attach a job-name label to a CFG point; no remote lookup is performed.",
    )
    parser.add_argument(
        "--checkpoint-sha256",
        default=None,
        metavar="HEX",
        help="Optional 64-hex SHA256 of the evaluated checkpoint weights.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        required=True,
        help="Destination summary JSON; replaced atomically.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        required=True,
        help="Destination flat CSV; replaced atomically.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    json_path = args.output_json.expanduser().resolve()
    csv_path = args.output_csv.expanduser().resolve()
    if json_path == csv_path:
        raise SummaryError("--output-json and --output-csv must be different paths")
    summary = build_summary(
        point_specs=args.point,
        job_specs=args.job,
        checkpoint_sha256=args.checkpoint_sha256,
    )
    input_metrics_paths = {
        Path(point["metrics_path"]).resolve() for point in summary["points"]
    }
    colliding_outputs = sorted(
        str(path)
        for path in (json_path, csv_path)
        if path in input_metrics_paths
    )
    if colliding_outputs:
        raise SummaryError(
            "summary output must not overwrite an input metrics file: "
            + ", ".join(colliding_outputs)
        )
    json_content = json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    csv_content = render_csv(summary)
    atomic_write_text(json_path, json_content)
    atomic_write_text(csv_path, csv_content)
    print(
        json.dumps(
            {
                "output_json": str(json_path),
                "output_csv": str(csv_path),
                "point_count": summary["point_count"],
                "best_by_fid": summary["best_by_fid"],
                "best_by_is": summary["best_by_is"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, SummaryError) as exc:
        raise SystemExit(f"error: {exc}")
