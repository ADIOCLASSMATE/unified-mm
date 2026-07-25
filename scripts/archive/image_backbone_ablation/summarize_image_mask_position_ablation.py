#!/usr/bin/env python3
# Historical Q-factor summarizer retained for evidence audit only.
"""Strictly validate and summarize the 4x3 image-mask-position Q study."""

from __future__ import annotations

import argparse
import csv
import json
import math
import tempfile
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Mapping, Sequence

from scripts.archive.image_backbone_ablation.image_embedder_ablation_matrix import (
    FLOW_HEAD_INVARIANTS,
    TRAINING_PROTOCOL_INVARIANTS,
    TRAINING_PROTOCOL_SCHEMA,
    training_protocol_fingerprint,
)
from scripts.archive.image_backbone_ablation.image_embedder_confirmation_protocol import (
    AUGMENTATION_CONTRACT,
    EVALUATOR_RNG_CONTRACT_SHA256,
    INITIALIZATION_CONTRACT,
    TRAIN_ORDER_CONTRACT,
    canonical_sha256,
)
from scripts.archive.image_backbone_ablation.image_mask_position_ablation_protocol import (
    Q_FACTOR_CONFIG_CONTRACT_SCHEMA,
    Q_FACTOR_DECISION_RULE,
    Q_FACTOR_IDS,
    Q_FACTOR_PROVENANCE_SCHEMA,
    Q_FACTOR_RUNTIME_CONTEXT_SCHEMA,
    Q_FACTOR_SEEDS,
    Q_FACTOR_VARIANTS,
    normalize_q_factor_id,
    q_factor_run_slug,
)


SCHEMA = "selfless_flow_image_mask_position_q_factor_summary_v1"
PAIRING_GATE_SCHEMA = "selfless_flow_image_mask_position_q_factor_pairing_gate_v1"
STRATEGY = "spatial_halton"
T95_DF2 = 4.302652729911275
FID_SIMPLICITY_MARGIN = float(Q_FACTOR_DECISION_RULE["close_fid_absolute_margin"])
IS_SIMPLICITY_MARGIN = float(
    Q_FACTOR_DECISION_RULE["close_inception_score_absolute_margin"]
)
CANONICAL_NOISE_MANIFEST_SCHEMA = "canonical_image_flow_noise_manifest_v1"
ORDERED_EVAL_SAMPLE_MANIFEST_SCHEMA = "ordered_image_embedder_eval_samples_v1"


class SummaryError(ValueError):
    """A formal Q-factor result violated the preregistered contract."""


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise SummaryError(f"{label} must be an object")
    return dict(value)


def _sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise SummaryError(f"{label} must be a lowercase SHA256 digest")
    return value


def _finite(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SummaryError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        raise SummaryError(f"{label} must be {'positive and ' if positive else ''}finite")
    return result


def _integer(value: Any, label: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SummaryError(f"{label} must be an integer")
    if positive and value <= 0:
        raise SummaryError(f"{label} must be positive")
    return value


def _expect(payload: Mapping[str, Any], key: str, expected: Any, source: Path) -> None:
    if payload.get(key) != expected:
        raise SummaryError(
            f"{source}: expected {key}={expected!r}, got {payload.get(key)!r}"
        )


def _parse_run(spec: str) -> tuple[str, int, Path]:
    label, separator, raw_path = spec.partition("=")
    raw_id, seed_separator, raw_seed = label.partition("@")
    if not separator or not seed_separator or not raw_path.strip():
        raise SummaryError(f"--run must use ID@TRAINING_SEED=PATH syntax, got {spec!r}")
    try:
        seed = int(raw_seed)
    except ValueError as exc:
        raise SummaryError(f"invalid training seed in --run {spec!r}") from exc
    return normalize_q_factor_id(raw_id), seed, Path(raw_path).expanduser()


def _nested(payload: Mapping[str, Any], *path: str) -> Any:
    value: Any = payload
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            raise SummaryError(f"config contract is missing {'.'.join(path)}")
        value = value[key]
    return value


def _drop_nested(payload: dict[str, Any], *path: str) -> None:
    target: Any = payload
    for key in path[:-1]:
        if not isinstance(target, dict) or key not in target:
            return
        target = target[key]
    if isinstance(target, dict):
        target.pop(path[-1], None)


def _config_pairing_projection(contract: Mapping[str, Any]) -> dict[str, Any]:
    payload = json.loads(json.dumps(contract, sort_keys=True))
    for path in (
        ("resolved_config", "experiment", "ablation_id"),
        ("resolved_config", "experiment", "parent_ablation_id"),
        ("resolved_config", "experiment", "project"),
        ("resolved_config", "experiment", "name"),
        ("resolved_config", "training", "seed"),
        ("resolved_config", "training", "dataloader_shuffle_seed"),
        ("resolved_config", "evaluation", "checkpoint"),
        ("resolved_config", "model", "image_observed_position_mode"),
        ("resolved_config", "model", "image_mask_position_mode"),
    ):
        _drop_nested(payload, *path)
    return payload


def _validate_config_contract(
    contract: Mapping[str, Any],
    *,
    variant_id: str,
    seed: int,
    digest: str,
    source: Path,
) -> str:
    contract = _mapping(contract, f"{source}: config contract")
    _expect(contract, "schema", Q_FACTOR_CONFIG_CONTRACT_SCHEMA, source)
    if canonical_sha256(contract) != digest:
        raise SummaryError(f"{source}: config contract digest mismatch")
    variant = Q_FACTOR_VARIANTS[variant_id]
    slug = q_factor_run_slug(variant_id, seed)
    expected_paths = {
        ("experiment", "ablation_phase"): "mask_position_q_factor",
        ("experiment", "ablation_id"): variant_id,
        ("experiment", "parent_ablation_id"): variant.parent_ablation_id,
        ("experiment", "project"): slug,
        ("training", "seed"): seed,
        ("training", "dataloader_shuffle_seed"): seed,
        ("evaluation", "seed"): 42,
        ("evaluation", "checkpoint"): f"output/{slug}/hf_model-final-ema",
        ("model", "image_query_stage_mode"): "none",
        ("model", "image_observed_position_mode"): variant.observed_position_mode,
        ("model", "image_mask_position_mode"): variant.mask_position_mode,
        ("model", "image_rope_mode"): "row_col_2d",
        ("model", "image_space_to_depth_factor"): 1,
    }
    resolved = _mapping(contract.get("resolved_config"), f"{source}: resolved config")
    for path, expected in expected_paths.items():
        actual = _nested(resolved, *path)
        if actual != expected:
            raise SummaryError(
                f"{source}: config contract {'.'.join(path)}={actual!r}; "
                f"expected {expected!r}"
            )
    return canonical_sha256(_config_pairing_projection(contract))


def _runtime_context(
    value: Mapping[str, Any], source: Path
) -> tuple[dict[str, Any], str]:
    runtime = _mapping(value, f"{source}: runtime context")
    stored = _sha256(
        runtime.pop("runtime_context_sha256", None),
        f"{source}: runtime context digest",
    )
    if canonical_sha256(runtime) != stored:
        raise SummaryError(f"{source}: runtime context digest mismatch")
    runtime["runtime_context_sha256"] = stored
    expected = {
        "schema": Q_FACTOR_RUNTIME_CONTEXT_SCHEMA,
        "world_size": 8,
        "distributed_type": "DEEPSPEED",
        "gradient_accumulation_steps": 1,
        "mixed_precision": "bf16",
        "cuda_available": True,
        "deepspeed_zero_stage": 2,
    }
    for key, expected_value in expected.items():
        if runtime.get(key) != expected_value:
            raise SummaryError(
                f"{source}: Q-factor runtime requires {key}={expected_value!r}; "
                f"got {runtime.get(key)!r}"
            )
    gpu_name = runtime.get("cuda_device_name")
    if not isinstance(gpu_name, str) or "H100" not in gpu_name.upper():
        raise SummaryError(f"{source}: Q-factor runtime requires an H100 GPU")
    for key in ("torch_version", "cuda_version"):
        if not isinstance(runtime.get(key), str) or runtime[key] in {"", "None"}:
            raise SummaryError(f"{source}: runtime {key} must be recorded")
    software_signature = canonical_sha256(
        {
            key: runtime[key]
            for key in (
                "world_size",
                "distributed_type",
                "gradient_accumulation_steps",
                "mixed_precision",
                "torch_version",
                "cuda_version",
                "cuda_device_name",
                "deepspeed_zero_stage",
            )
        }
    )
    return runtime, software_signature


def _validate_architecture(payload: Mapping[str, Any], variant_id: str, source: Path):
    architecture = _mapping(payload.get("architecture"), f"{source}: architecture")
    variant = Q_FACTOR_VARIANTS[variant_id]
    expected = {
        "ablation_id": variant_id,
        "parent_ablation_id": variant.parent_ablation_id,
        "q_factor_id": variant_id,
        "mask_query_position_factor": variant_id.rsplit("-", 1)[-1],
        "image_query_stage_mode": "none",
        "image_observed_position_mode": variant.observed_position_mode,
        "image_mask_position_mode": variant.mask_position_mode,
        "image_rope_mode": "row_col_2d",
        "image_space_to_depth_factor": 1,
        "image_canonical_grid_side": 16,
        "image_canonical_latent_dim": 16,
        "image_grid_side": 16,
        "image_tokens_per_img": 256,
        "image_latent_dim": 16,
        "padded_sequence_length": 320,
    }
    for key, value in expected.items():
        _expect(architecture, key, value, source)
    flow_head = _mapping(architecture.get("flow_head"), f"{source}: flow head")
    expected_flow_head = {
        "arch": FLOW_HEAD_INVARIANTS["image_flow_head_arch"],
        "depth": FLOW_HEAD_INVARIANTS["image_flow_depth"],
        "width": FLOW_HEAD_INVARIANTS["image_flow_width"],
        "mlp_ratio": FLOW_HEAD_INVARIANTS["image_flow_mlp_ratio"],
        "latent_mixer_heads": FLOW_HEAD_INVARIANTS[
            "image_flow_latent_mixer_heads"
        ],
        "latent_mixer_dropout": FLOW_HEAD_INVARIANTS[
            "image_flow_latent_mixer_dropout"
        ],
        "zero_init_gate": FLOW_HEAD_INVARIANTS[
            "image_flow_latent_mixer_zero_init_gate"
        ],
    }
    if flow_head != expected_flow_head:
        raise SummaryError(f"{source}: frozen flow-head contract drifted")


def _validate_q_provenance(
    payload: Mapping[str, Any],
    *,
    variant_id: str,
    seed: int,
    slug: str,
    source: Path,
) -> dict[str, Any]:
    training = _mapping(payload.get("training_protocol"), f"{source}: training protocol")
    _expect(training, "schema", TRAINING_PROTOCOL_SCHEMA, source)
    _expect(training, "training_seed", seed, source)
    _expect(training, "final_global_step", 35_920, source)
    invariants = _mapping(training.get("invariants"), f"{source}: training invariants")
    if invariants != TRAINING_PROTOCOL_INVARIANTS:
        raise SummaryError(f"{source}: fixed training protocol drifted")
    expected_fingerprint = training_protocol_fingerprint(invariants)
    _expect(training, "invariants_sha256", expected_fingerprint, source)

    artifacts = _mapping(training.get("artifacts"), f"{source}: artifacts")
    expected_paths = {
        "checkpoint_metadata_path": f"output/{slug}/checkpoint-35920/metadata.json",
        "ema_state_path": f"output/{slug}/checkpoint-35920/ema_state.pt",
        "hf_model_weights_path": f"output/{slug}/hf_model-final-ema/model.safetensors",
        "q_factor_provenance_path": f"output/{slug}/q_factor_training_provenance.json",
        "q_factor_hf_provenance_path": (
            f"output/{slug}/hf_model-final-ema/q_factor_training_provenance.json"
        ),
    }
    for key, expected in expected_paths.items():
        _expect(artifacts, key, expected, source)
    for key in ("ema_state_size_bytes", "hf_model_weights_size_bytes"):
        _integer(artifacts.get(key), f"{source}: artifacts.{key}", positive=True)
    checkpoint_hashes = {
        key: _sha256(artifacts.get(key), f"{source}: artifacts.{key}")
        for key in (
            "checkpoint_metadata_sha256",
            "ema_state_sha256",
            "hf_model_weights_sha256",
        )
    }

    q_factor = _mapping(training.get("q_factor"), f"{source}: q_factor metadata")
    declaration_sha = _sha256(q_factor.get("declaration_sha256"), "declaration digest")
    study_sha = _sha256(q_factor.get("study_manifest_sha256"), "study digest")
    parent_sha = _sha256(q_factor.get("parent_summary_sha256"), "parent digest")
    config_sha = _sha256(q_factor.get("config_contract_sha256"), "config digest")
    source_sha = _sha256(q_factor.get("source_manifest_sha256"), "source digest")
    _expect(q_factor, "evaluator_rng_contract_sha256", EVALUATOR_RNG_CONTRACT_SHA256, source)
    _expect(q_factor, "dataloader_shuffle_seed", seed, source)
    provenance_path = f"output/{slug}/q_factor_training_provenance.json"
    _expect(q_factor, "provenance_path", provenance_path, source)
    provenance_sha = _sha256(q_factor.get("provenance_sha256"), "provenance digest")

    provenance = _mapping(q_factor.get("provenance"), f"{source}: compact provenance")
    variant = Q_FACTOR_VARIANTS[variant_id]
    expected_provenance = {
        "schema": Q_FACTOR_PROVENANCE_SCHEMA,
        "q_factor_id": variant_id,
        "parent_ablation_id": variant.parent_ablation_id,
        "training_seed": seed,
        "architecture": asdict(variant),
        "provenance_sha256": provenance_sha,
        "q_factor_declaration_sha256": declaration_sha,
        "study_manifest_sha256": study_sha,
        "parent_summary_sha256": parent_sha,
        "config_contract_sha256": config_sha,
        "runtime_source_manifest_sha256": source_sha,
    }
    for key, expected in expected_provenance.items():
        _expect(provenance, key, expected, source)
    config_projection_sha = _validate_config_contract(
        _mapping(provenance.get("config_contract"), f"{source}: config contract"),
        variant_id=variant_id,
        seed=seed,
        digest=config_sha,
        source=source,
    )

    initial = _mapping(provenance.get("initial_state"), f"{source}: initial state")
    _expect(initial, "contract", INITIALIZATION_CONTRACT, source)
    image_modules = _mapping(initial.get("image_modules"), f"{source}: image modules")
    image_parameter_count = _integer(
        image_modules.get("parameter_count"), f"{source}: image parameter count", positive=True
    )
    image_parameter_schema_sha = _sha256(
        image_modules.get("parameter_schema_sha256"), "image parameter schema digest"
    )
    image_state_sha = _sha256(image_modules.get("state_sha256"), "image state digest")
    special_rows_sha = _sha256(
        initial.get("special_token_rows_sha256"), "special-token rows digest"
    )
    special_tokens = initial.get("special_token_names_and_ids")
    if (
        not isinstance(special_tokens, list)
        or not special_tokens
        or any(
            not isinstance(item, list)
            or len(item) != 2
            or not isinstance(item[0], str)
            or not isinstance(item[1], int)
            for item in special_tokens
        )
    ):
        raise SummaryError(f"{source}: special-token identity list is invalid")

    train_data = _mapping(provenance.get("train_data"), f"{source}: train data")
    _expect(train_data, "contract", TRAIN_ORDER_CONTRACT, source)
    _expect(train_data, "augmentation_contract", AUGMENTATION_CONTRACT, source)
    for key in ("dataloader_shuffle_seed", "augmentation_seed"):
        _expect(train_data, key, seed, source)
    for key, expected in {
        "batch_size_per_rank": 32,
        "total_batch_size": 256,
        "drop_last": True,
    }.items():
        _expect(train_data, key, expected, source)
    _integer(train_data.get("dataset_length"), f"{source}: dataset length", positive=True)
    _integer(train_data.get("num_workers"), f"{source}: num workers", positive=True)
    if not isinstance(train_data.get("persistent_workers"), bool):
        raise SummaryError(f"{source}: persistent_workers must be boolean")
    input_files = _mapping(train_data.get("input_files"), f"{source}: input files")
    if set(input_files) != {"cache", "manifest", "split_manifest", "synset_mapping"}:
        raise SummaryError(f"{source}: dataset input-file set drifted")
    for label, raw_entry in input_files.items():
        entry = _mapping(raw_entry, f"{source}: input file {label}")
        if not isinstance(entry.get("path"), str) or not entry["path"]:
            raise SummaryError(f"{source}: input file {label} path is invalid")
        _integer(entry.get("size_bytes"), f"{source}: input file size", positive=True)
        _sha256(entry.get("sha256"), f"{source}: input file digest")

    generator_sha = _sha256(
        train_data.get("initial_generator_state_sha256"), "generator state digest"
    )
    order_sha = _sha256(
        train_data.get("epoch0_ordered_sample_identity_sha256"), "train order digest"
    )
    augmentation_sha = _sha256(
        train_data.get("epoch0_augmentation_decisions_sha256"),
        "augmentation digest",
    )
    base_seed = _integer(train_data.get("dataloader_base_seed"), "dataloader base seed")
    base_model_sha = _sha256(
        provenance.get("base_model_manifest_sha256"), "base model digest"
    )
    runtime, runtime_software_sha = _runtime_context(
        _mapping(provenance.get("runtime_context"), f"{source}: runtime context"),
        source,
    )

    return {
        "training_protocol_sha256": expected_fingerprint,
        "declaration_sha256": declaration_sha,
        "study_manifest_sha256": study_sha,
        "parent_summary_sha256": parent_sha,
        "config_contract_sha256": config_sha,
        "config_pairing_projection_sha256": config_projection_sha,
        "source_manifest_sha256": source_sha,
        "provenance_sha256": provenance_sha,
        "checkpoint_hashes": checkpoint_hashes,
        "image_parameter_count": image_parameter_count,
        "image_parameter_schema_sha256": image_parameter_schema_sha,
        "image_state_sha256": image_state_sha,
        "special_token_rows_sha256": special_rows_sha,
        "special_token_names_and_ids": special_tokens,
        "initial_generator_state_sha256": generator_sha,
        "dataloader_base_seed": base_seed,
        "dataset_length": int(train_data["dataset_length"]),
        "epoch0_ordered_sample_identity_sha256": order_sha,
        "epoch0_augmentation_decisions_sha256": augmentation_sha,
        "input_files_sha256": canonical_sha256(input_files),
        "base_model_manifest_sha256": base_model_sha,
        "runtime_context": runtime,
        "runtime_software_sha256": runtime_software_sha,
    }


def _load_run(
    variant_id: str, seed: int, path: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    try:
        resolved = path.resolve(strict=True)
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SummaryError(f"failed to read metrics {path}: {exc}") from exc
    payload = _mapping(payload, str(resolved))
    slug = q_factor_run_slug(variant_id, seed)
    for key, expected in {
        "official_protocol": True,
        "config": f"output/{slug}/config.yaml",
        "model_path": f"output/{slug}/hf_model-final-ema",
        "split": "val",
        "seed": 42,
        "batch_size": 512,
        "samples_requested": 10_000,
        "samples_evaluated": 10_000,
        "cfg": 3.5,
        "cfg_schedule": "constant",
        "sampling_steps": "100",
        "temperature": 1.0,
        "flow_solver": "heun",
        "parallel_rate": 1,
        "adapter": {"adapter": None},
        "model_state": {"model_state": None},
        "ema_state": {"ema_state": None},
    }.items():
        _expect(payload, key, expected, resolved)

    distributed = _mapping(payload.get("distributed"), f"{resolved}: distributed")
    _expect(distributed, "enabled", True, resolved)
    _expect(distributed, "world_size", 8, resolved)
    peak_allocated = _finite(
        distributed.get("peak_cuda_allocated_mib"), "peak allocated memory", positive=True
    )
    peak_reserved = _finite(
        distributed.get("peak_cuda_reserved_mib"), "peak reserved memory", positive=True
    )

    precision = _mapping(payload.get("precision_protocol"), f"{resolved}: precision")
    for key, expected in {
        "schema": "flow_eval_precision_v1",
        "model_dtype": "bf16",
        "model_parameter_dtypes": ["torch.bfloat16"],
        "vae_dtype": "fp32",
        "flow_integrator_dtype": "fp32",
        "autocast_enabled": False,
    }.items():
        _expect(precision, key, expected, resolved)
    if not isinstance(precision.get("checkpoint_weight_dtypes"), list) or not precision[
        "checkpoint_weight_dtypes"
    ]:
        raise SummaryError(f"{resolved}: checkpoint weight dtypes are missing")

    metric_protocol = _mapping(payload.get("metric_protocol"), f"{resolved}: metric protocol")
    for key, expected in {
        "fid_reducer": "symmetric_eigendecomposition",
        "is_split_assignment": "contiguous_by_global_sample_index",
        "is_std": "population",
        "is_splits": 10,
    }.items():
        _expect(metric_protocol, key, expected, resolved)

    _validate_architecture(payload, variant_id, resolved)
    evidence = _validate_q_provenance(
        payload,
        variant_id=variant_id,
        seed=seed,
        slug=slug,
        source=resolved,
    )

    contracts = _mapping(payload.get("implementation_contracts"), f"{resolved}: contracts")
    evaluator_contract = _mapping(
        contracts.get("evaluator_rng_contract"), f"{resolved}: evaluator RNG contract"
    )
    evaluator_sha = _sha256(
        contracts.get("evaluator_rng_contract_sha256"), "evaluator RNG digest"
    )
    if canonical_sha256(evaluator_contract) != evaluator_sha:
        raise SummaryError(f"{resolved}: evaluator RNG contract digest mismatch")
    if evaluator_sha != EVALUATOR_RNG_CONTRACT_SHA256:
        raise SummaryError(f"{resolved}: evaluator RNG contract drifted")
    _expect(contracts, "canonical_initial_noise_enabled", True, resolved)
    _expect(
        contracts,
        "canonical_noise_manifest_schema",
        CANONICAL_NOISE_MANIFEST_SCHEMA,
        resolved,
    )
    noise_sha = _sha256(
        contracts.get("canonical_noise_manifest_sha256"), "noise manifest digest"
    )
    _expect(
        contracts,
        "ordered_eval_sample_manifest_schema",
        ORDERED_EVAL_SAMPLE_MANIFEST_SCHEMA,
        resolved,
    )
    sample_sha = _sha256(
        contracts.get("ordered_eval_sample_manifest_sha256"), "sample manifest digest"
    )
    _expect(contracts, "paired_sample_count", 10_000, resolved)

    parameters = _mapping(payload.get("parameters"), f"{resolved}: parameters")
    parameter_counts = {
        key: _integer(parameters.get(key), f"{resolved}: parameters.{key}", positive=True)
        for key in ("total", "trainable", "image_embedder", "flow_head")
    }
    strategies = _mapping(payload.get("strategies"), f"{resolved}: strategies")
    if set(strategies) != {STRATEGY}:
        raise SummaryError(f"{resolved}: expected exactly strategy {STRATEGY!r}")
    metrics = _mapping(strategies[STRATEGY], f"{resolved}: strategy metrics")
    _expect(metrics, "count", 10_000, resolved)
    fid = _finite(metrics.get("fid"), "FID", positive=True)
    is_mean = _finite(metrics.get("inception_score_mean"), "IS", positive=True)
    is_std = _finite(metrics.get("inception_score_std"), "IS std")
    wall = _finite(metrics.get("generation_wall_seconds"), "wall time", positive=True)
    throughput = _finite(
        metrics.get("generation_samples_per_second"), "throughput", positive=True
    )
    if not math.isclose(throughput, 10_000 / wall, rel_tol=1e-6):
        raise SummaryError(f"{resolved}: throughput is inconsistent with wall time")
    for key in ("latent_mse_to_target", "latent_rms", "generation_step_max"):
        _finite(metrics.get(key), f"{resolved}: {key}", positive=True)
    splits = metrics.get("inception_score_splits")
    if not isinstance(splits, list) or len(splits) != 10:
        raise SummaryError(f"{resolved}: expected exactly 10 IS split scores")
    for index, value in enumerate(splits):
        _finite(value, f"{resolved}: IS split {index}", positive=True)

    evidence.update(
        {
            "canonical_noise_manifest_sha256": noise_sha,
            "ordered_eval_sample_manifest_sha256": sample_sha,
            "evaluator_rng_contract_sha256": evaluator_sha,
            "parameters_sha256": canonical_sha256(parameter_counts),
        }
    )
    row = {
        "id": variant_id,
        "training_seed": seed,
        "parent_ablation_id": Q_FACTOR_VARIANTS[variant_id].parent_ablation_id,
        "mask_position_mode": Q_FACTOR_VARIANTS[variant_id].mask_position_mode,
        "observed_position_mode": Q_FACTOR_VARIANTS[variant_id].observed_position_mode,
        "fid": fid,
        "is_mean": is_mean,
        "is_std": is_std,
        "sampling_wall_seconds": wall,
        "sampling_samples_per_second": throughput,
        "peak_cuda_allocated_mib": peak_allocated,
        "peak_cuda_reserved_mib": peak_reserved,
        **parameter_counts,
        "checkpoint_metadata_sha256": evidence["checkpoint_hashes"][
            "checkpoint_metadata_sha256"
        ],
        "ema_state_sha256": evidence["checkpoint_hashes"]["ema_state_sha256"],
        "hf_model_weights_sha256": evidence["checkpoint_hashes"][
            "hf_model_weights_sha256"
        ],
        "metrics_path": str(resolved),
    }
    signature = {
        "metric_protocol": metric_protocol,
        "precision_protocol": precision,
        "real_source": payload.get("real_source"),
        "real_stats_path": payload.get("real_stats_path"),
        "real_stats_metadata": payload.get("real_stats_metadata"),
        "inception_weights_path": payload.get("inception_weights_path"),
        "training_protocol_sha256": evidence["training_protocol_sha256"],
    }
    return row, signature, evidence


def _require_same(evidence: Sequence[Mapping[str, Any]], fields: Sequence[str], scope: str):
    for field in fields:
        values = {str(item[field]) for item in evidence}
        if len(values) != 1:
            raise SummaryError(
                f"Q-factor pairing mismatch for {field} within {scope}: "
                + str([item[field] for item in evidence])
            )


def _validate_pairing(
    rows: Sequence[Mapping[str, Any]], evidence_by_run: Mapping[tuple[str, int], Mapping[str, Any]]
) -> dict[str, Any]:
    evidence = [evidence_by_run[(str(row["id"]), int(row["training_seed"]))] for row in rows]
    global_fields = (
        "study_manifest_sha256",
        "parent_summary_sha256",
        "source_manifest_sha256",
        "input_files_sha256",
        "base_model_manifest_sha256",
        "runtime_software_sha256",
        "evaluator_rng_contract_sha256",
        "canonical_noise_manifest_sha256",
        "ordered_eval_sample_manifest_sha256",
        "parameters_sha256",
        "config_pairing_projection_sha256",
    )
    _require_same(evidence, global_fields, "all 12 runs")

    by_seed: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_seed[int(row["training_seed"])].append(
            evidence_by_run[(str(row["id"]), int(row["training_seed"]))]
        )
    paired_fields = (
        "image_parameter_count",
        "image_parameter_schema_sha256",
        "image_state_sha256",
        "special_token_rows_sha256",
        "special_token_names_and_ids",
        "initial_generator_state_sha256",
        "dataloader_base_seed",
        "dataset_length",
        "epoch0_ordered_sample_identity_sha256",
        "epoch0_augmentation_decisions_sha256",
        "input_files_sha256",
    )
    for seed, seed_evidence in sorted(by_seed.items()):
        _require_same(seed_evidence, paired_fields, f"training seed {seed}")

    if len({items[0]["image_state_sha256"] for items in by_seed.values()}) != 3:
        raise SummaryError("Q-factor training seeds do not have independent initial states")
    if (
        len(
            {
                items[0]["epoch0_ordered_sample_identity_sha256"]
                for items in by_seed.values()
            }
        )
        != 3
    ):
        raise SummaryError("Q-factor training seeds do not have independent train orders")
    return {
        "schema": PAIRING_GATE_SCHEMA,
        "validated_runs": len(rows),
        "validated_training_seeds": sorted(by_seed),
        **{field: evidence[0][field] for field in global_fields},
    }


def _statistic(values: Sequence[float]) -> dict[str, Any]:
    values = [float(value) for value in values]
    if len(values) != 3:
        raise SummaryError(f"paired statistic requires exactly 3 values, got {len(values)}")
    average = mean(values)
    sd = stdev(values)
    sem = sd / math.sqrt(len(values))
    radius = T95_DF2 * sem
    return {
        "n": len(values),
        "values": values,
        "mean": average,
        "sample_std": sd,
        "standard_error": sem,
        "t_critical_95_df2": T95_DF2,
        "ci95": [average - radius, average + radius],
    }


def _paired_effect(
    lookup: Mapping[tuple[str, int], Mapping[str, Any]], candidate: str, reference: str
) -> dict[str, Any]:
    seeds = sorted(Q_FACTOR_SEEDS)
    fid = [
        float(lookup[(candidate, seed)]["fid"]) - float(lookup[(reference, seed)]["fid"])
        for seed in seeds
    ]
    score = [
        float(lookup[(candidate, seed)]["is_mean"])
        - float(lookup[(reference, seed)]["is_mean"])
        for seed in seeds
    ]
    throughput = [
        float(lookup[(candidate, seed)]["sampling_samples_per_second"])
        - float(lookup[(reference, seed)]["sampling_samples_per_second"])
        for seed in seeds
    ]
    fid_stats = _statistic(fid)
    fid_stats["candidate_wins"] = sum(value < 0 for value in fid)
    fid_stats["exact_ties"] = sum(value == 0 for value in fid)
    return {
        "candidate": candidate,
        "reference": reference,
        "seeds": seeds,
        "fid_candidate_minus_reference": fid_stats,
        "is_candidate_minus_reference": _statistic(score),
        "throughput_candidate_minus_reference": _statistic(throughput),
    }


def _interaction(lookup: Mapping[tuple[str, int], Mapping[str, Any]]) -> dict[str, Any]:
    seeds = sorted(Q_FACTOR_SEEDS)

    def values(field: str):
        return [
            (
                float(lookup[("E2-Q0", seed)][field])
                - float(lookup[("E2-Q1", seed)][field])
            )
            - (
                float(lookup[("E2b-Q0", seed)][field])
                - float(lookup[("E2b-Q1", seed)][field])
            )
            for seed in seeds
        ]

    return {
        "formula": "(E2-Q0 - E2-Q1) - (E2b-Q0 - E2b-Q1)",
        "seeds": seeds,
        "fid": _statistic(values("fid")),
        "is": _statistic(values("is_mean")),
        "throughput": _statistic(values("sampling_samples_per_second")),
    }


def _aggregates(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["id"])].append(row)
    output = []
    for variant_id in Q_FACTOR_IDS:
        group = grouped[variant_id]
        fids = [float(item["fid"]) for item in group]
        scores = [float(item["is_mean"]) for item in group]
        rates = [float(item["sampling_samples_per_second"]) for item in group]
        output.append(
            {
                "id": variant_id,
                "parent_ablation_id": Q_FACTOR_VARIANTS[variant_id].parent_ablation_id,
                "observed_position_mode": Q_FACTOR_VARIANTS[
                    variant_id
                ].observed_position_mode,
                "mask_position_mode": Q_FACTOR_VARIANTS[variant_id].mask_position_mode,
                "seeds": sorted(int(item["training_seed"]) for item in group),
                "fid_mean": mean(fids),
                "fid_sample_std": stdev(fids),
                "is_mean": mean(scores),
                "is_sample_std": stdev(scores),
                "sampling_samples_per_second_mean": mean(rates),
                "sampling_samples_per_second_sample_std": stdev(rates),
            }
        )
    return output


def _pareto(aggregates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    frontier = []
    for candidate in aggregates:
        dominated = any(
            other is not candidate
            and float(other["fid_mean"]) <= float(candidate["fid_mean"])
            and float(other["is_mean"]) >= float(candidate["is_mean"])
            and (
                float(other["fid_mean"]) < float(candidate["fid_mean"])
                or float(other["is_mean"]) > float(candidate["is_mean"])
            )
            for other in aggregates
        )
        if not dominated:
            frontier.append(dict(candidate))
    return sorted(frontier, key=lambda item: (item["fid_mean"], -item["is_mean"]))


def _preference(aggregates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_id = {str(item["id"]): item for item in aggregates}
    best = min(aggregates, key=lambda item: (item["fid_mean"], -item["is_mean"]))
    simple = by_id["E2-Q0"]
    fid_delta = float(simple["fid_mean"]) - float(best["fid_mean"])
    is_delta = float(simple["is_mean"]) - float(best["is_mean"])
    close = abs(fid_delta) <= FID_SIMPLICITY_MARGIN and abs(is_delta) <= IS_SIMPLICITY_MARGIN
    selected = "E2-Q0" if close else str(best["id"])
    return {
        "decision_rule_schema": Q_FACTOR_DECISION_RULE["schema"],
        "rule": (
            "Prefer E2-Q0 when its absolute mean-FID gap to the best-FID variant is "
            "<=0.5 and its absolute mean-IS gap is <=1.0; otherwise choose best mean FID."
        ),
        "fid_margin": FID_SIMPLICITY_MARGIN,
        "is_margin": IS_SIMPLICITY_MARGIN,
        "best_fid_id": str(best["id"]),
        "e2_q0_minus_best_fid": fid_delta,
        "e2_q0_minus_best_is": is_delta,
        "within_simplicity_margins": close,
        "selected_id": selected,
        "selected_reason": "simplicity_preference" if close else "best_mean_fid",
    }


def build_summary(run_specs: Sequence[str]) -> dict[str, Any]:
    expected_pairs = {
        (variant_id, seed) for variant_id in Q_FACTOR_IDS for seed in Q_FACTOR_SEEDS
    }
    parsed = []
    seen = set()
    for spec in run_specs:
        variant_id, seed, path = _parse_run(spec)
        key = (variant_id, seed)
        if key in seen:
            raise SummaryError(f"duplicate Q-factor run assignment for {variant_id}@{seed}")
        seen.add(key)
        parsed.append((variant_id, seed, path))
    if seen != expected_pairs:
        raise SummaryError(
            "Q-factor summary requires the exact 4x3 matrix; "
            f"missing={sorted(expected_pairs - seen)}, unexpected={sorted(seen - expected_pairs)}"
        )

    rows = []
    evidence_by_run = {}
    protocol_signature = None
    for variant_id, seed, path in parsed:
        row, signature, evidence = _load_run(variant_id, seed, path)
        if protocol_signature is None:
            protocol_signature = signature
        elif signature != protocol_signature:
            raise SummaryError(f"{path}: formal evaluation protocol differs across Q runs")
        rows.append(row)
        evidence_by_run[(variant_id, seed)] = evidence
    order = {variant_id: index for index, variant_id in enumerate(Q_FACTOR_IDS)}
    rows.sort(key=lambda item: (order[str(item["id"])], int(item["training_seed"])))
    pairing_gate = _validate_pairing(rows, evidence_by_run)
    aggregates = _aggregates(rows)
    ranking_fid = sorted(aggregates, key=lambda item: (item["fid_mean"], -item["is_mean"]))
    ranking_is = sorted(aggregates, key=lambda item: (-item["is_mean"], item["fid_mean"]))
    lookup = {(str(item["id"]), int(item["training_seed"])): item for item in rows}
    effects = {
        "mask_Q0_minus_Q1_at_E2b": _paired_effect(lookup, "E2b-Q0", "E2b-Q1"),
        "mask_Q0_minus_Q1_at_E2": _paired_effect(lookup, "E2-Q0", "E2-Q1"),
        "observed_none_minus_additive_at_Q1": _paired_effect(
            lookup, "E2-Q1", "E2b-Q1"
        ),
        "observed_none_minus_additive_at_Q0": _paired_effect(
            lookup, "E2-Q0", "E2b-Q0"
        ),
    }
    return {
        "schema": SCHEMA,
        "expected": "exact_4x3_q_factor",
        "protocol": protocol_signature,
        "runs": rows,
        "aggregates": aggregates,
        "ranking_by_fid": [item["id"] for item in ranking_fid],
        "ranking_by_is": [item["id"] for item in ranking_is],
        "best_by_fid": ranking_fid[0],
        "best_by_is": ranking_is[0],
        "fid_is_pareto_frontier": _pareto(aggregates),
        "pairing_gate": pairing_gate,
        "paired_effects": effects,
        "interaction": _interaction(lookup),
        "selection": _preference(aggregates),
    }


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, delete=False, encoding="utf-8"
    ) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    temporary.replace(path)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, delete=False, encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    temporary.replace(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", default=[])
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv")
    return parser


def main() -> None:
    args = _parser().parse_args()
    summary = build_summary(args.run)
    _atomic_write(Path(args.output_json), json.dumps(summary, indent=2) + "\n")
    if args.output_csv:
        _write_csv(Path(args.output_csv), summary["runs"])
    print(json.dumps(summary["selection"], indent=2))


if __name__ == "__main__":
    main()
