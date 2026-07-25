#!/usr/bin/env python3
# Historical summarizer retained for evidence audit only.
"""Validate and summarize formal image-embedder ablation evaluations.

Each ``--run`` uses ``ID@TRAINING_SEED=metrics.json`` syntax.  The evaluator
seed remains fixed at 42; the seed in the assignment identifies the training
run whose EMA checkpoint was evaluated.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.archive.image_backbone_ablation.image_embedder_ablation_matrix import (  # noqa: E402
    FLOW_HEAD_INVARIANTS,
    STAGE_BUFFER_IMPLEMENTATION_CONTRACT,
    TRAINING_PROTOCOL_INVARIANTS,
    TRAINING_PROTOCOL_SCHEMA,
    VARIANTS,
    normalize_variant_id,
    run_slug,
    training_protocol_fingerprint,
)
from scripts.archive.image_backbone_ablation.image_embedder_confirmation_protocol import (  # noqa: E402
    AUGMENTATION_CONTRACT,
    CONFIRMATION_MANIFEST_SCHEMA,
    CONFIRMATION_PROVENANCE_SCHEMA,
    CONFIRMATION_SEEDS,
    EVALUATOR_RNG_CONTRACT_SCHEMA,
    EVALUATOR_RNG_CONTRACT_SHA256,
    INITIALIZATION_CONTRACT,
    TRAIN_ORDER_CONTRACT,
    canonical_sha256,
)


SCHEMA = "selfless_flow_image_embedder_ablation_summary_v3"
FIRST_STAGE_IDS = ("E0", "E1", "E2a", "E2b", "E2", "E3")
FULL_MATRIX_IDS = ("E0", "E1", "E2a", "E2b", "E2", "E3", "E4", "E5", "E6", "E7")
EXPANDED_MATRIX_IDS = tuple(VARIANTS)
FOUR_FACTOR_SETTINGS = {
    "E0": (0, 0, 0, 0),
    "E1": (1, 0, 0, 0),
    "E2a": (0, 1, 0, 0),
    "E2b": (0, 0, 1, 0),
    "E2": (0, 1, 1, 0),
    "E3": (0, 0, 0, 1),
    "E4a": (1, 1, 0, 0),
    "E4b": (1, 0, 1, 0),
    "E4": (1, 1, 1, 0),
    "E5": (1, 0, 0, 1),
    "E6a": (0, 1, 0, 1),
    "E6b": (0, 0, 1, 1),
    "E6": (0, 1, 1, 1),
    "E7a": (1, 1, 0, 1),
    "E7b": (1, 0, 1, 1),
    "E7": (1, 1, 1, 1),
}
STRATEGY = "spatial_halton"
NEAR_BEST_FID_MARGIN = 1.0
SPEED_ADVANTAGE_RATIO = 1.5
CANONICAL_NOISE_MANIFEST_SCHEMA = "canonical_image_flow_noise_manifest_v1"
ORDERED_EVAL_SAMPLE_MANIFEST_SCHEMA = "ordered_image_embedder_eval_samples_v1"
CONFIRMATION_SCOPE_AMENDMENT_SCHEMA = (
    "selfless_flow_image_embedder_confirmation_scope_amendment_v1"
)


class SummaryError(ValueError):
    """Raised when a result does not satisfy the formal ablation contract."""


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SummaryError(f"{field} must be an object")
    return value


def _finite(value: Any, field: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SummaryError(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        raise SummaryError(f"{field} must be {'positive and ' if positive else ''}finite")
    return result


def _integer(value: Any, field: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SummaryError(f"{field} must be an integer")
    if positive and value <= 0:
        raise SummaryError(f"{field} must be positive")
    return value


def _sha256(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SummaryError(f"{field} must be a lowercase SHA256 digest")
    return value


def _parse_run(spec: str) -> tuple[str, int, Path]:
    label, separator, raw_path = spec.partition("=")
    raw_id, seed_separator, raw_seed = label.partition("@")
    if not separator or not seed_separator or not raw_path.strip():
        raise SummaryError(
            f"--run must use ID@TRAINING_SEED=PATH syntax, got {spec!r}"
        )
    variant_id = normalize_variant_id(raw_id)
    try:
        seed = int(raw_seed)
    except ValueError as exc:
        raise SummaryError(f"invalid training seed in --run {spec!r}") from exc
    return variant_id, seed, Path(raw_path).expanduser()


def _expect(payload: Mapping[str, Any], key: str, expected: Any, source: Path) -> None:
    actual = payload.get(key)
    if actual != expected:
        raise SummaryError(
            f"{source}: expected {key}={expected!r}, got {actual!r}"
        )


def _protocol_signature(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "metric_protocol": payload.get("metric_protocol"),
        "precision_protocol": payload.get("precision_protocol"),
        "real_source": payload.get("real_source"),
        "real_stats_path": payload.get("real_stats_path"),
        "real_stats_metadata": payload.get("real_stats_metadata"),
        "inception_weights_path": payload.get("inception_weights_path"),
        "split": payload.get("split"),
        "seed": payload.get("seed"),
        "batch_size": payload.get("batch_size"),
        "cfg": payload.get("cfg"),
        "cfg_schedule": payload.get("cfg_schedule"),
        "sampling_steps": payload.get("sampling_steps"),
        "temperature": payload.get("temperature"),
        "flow_solver": payload.get("flow_solver"),
        "parallel_rate": payload.get("parallel_rate"),
        "strategy_keys": sorted(
            _mapping(payload.get("strategies"), "strategies").keys()
        ),
        "training_protocol_invariants_sha256": _mapping(
            payload.get("training_protocol"), "training_protocol"
        ).get("invariants_sha256"),
    }


def _load_run(
    variant_id: str,
    training_seed: int,
    path: Path,
) -> tuple[dict[str, Any], dict[str, Any], Mapping[str, Any]]:
    try:
        resolved = path.resolve(strict=True)
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SummaryError(f"failed to read metrics {path}: {exc}") from exc
    payload = _mapping(payload, str(resolved))

    fixed_fields = {
        "official_protocol": True,
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
    }
    for key, expected in fixed_fields.items():
        _expect(payload, key, expected, resolved)
    for key, expected in {
        "adapter": {"adapter": None},
        "model_state": {"model_state": None},
        "ema_state": {"ema_state": None},
    }.items():
        _expect(payload, key, expected, resolved)

    distributed = _mapping(payload.get("distributed"), f"{resolved}: distributed")
    _expect(distributed, "enabled", True, resolved)
    _expect(distributed, "world_size", 8, resolved)
    peak_allocated = _finite(
        distributed.get("peak_cuda_allocated_mib"),
        f"{resolved}: distributed.peak_cuda_allocated_mib",
        positive=True,
    )
    peak_reserved = _finite(
        distributed.get("peak_cuda_reserved_mib"),
        f"{resolved}: distributed.peak_cuda_reserved_mib",
        positive=True,
    )

    precision = _mapping(payload.get("precision_protocol"), f"{resolved}: precision_protocol")
    _expect(precision, "schema", "flow_eval_precision_v1", resolved)
    _expect(precision, "model_dtype", "bf16", resolved)
    _expect(precision, "model_parameter_dtypes", ["torch.bfloat16"], resolved)
    _expect(precision, "vae_dtype", "fp32", resolved)
    _expect(precision, "flow_integrator_dtype", "fp32", resolved)

    slug = run_slug(variant_id, training_seed)
    _expect(payload, "config", f"output/{slug}/config.yaml", resolved)
    _expect(payload, "model_path", f"output/{slug}/hf_model-final-ema", resolved)

    training_protocol = _mapping(
        payload.get("training_protocol"),
        f"{resolved}: training_protocol",
    )
    _expect(training_protocol, "schema", TRAINING_PROTOCOL_SCHEMA, resolved)
    _expect(training_protocol, "training_seed", training_seed, resolved)
    _expect(training_protocol, "final_global_step", 35_920, resolved)
    invariants = _mapping(
        training_protocol.get("invariants"),
        f"{resolved}: training_protocol.invariants",
    )
    if dict(invariants) != TRAINING_PROTOCOL_INVARIANTS:
        raise SummaryError(f"{resolved}: fixed training protocol drifted")
    expected_fingerprint = training_protocol_fingerprint(dict(invariants))
    _expect(
        training_protocol,
        "invariants_sha256",
        expected_fingerprint,
        resolved,
    )
    artifacts = _mapping(
        training_protocol.get("artifacts"),
        f"{resolved}: training_protocol.artifacts",
    )
    expected_artifact_paths = {
        "checkpoint_metadata_path": f"output/{slug}/checkpoint-35920/metadata.json",
        "ema_state_path": f"output/{slug}/checkpoint-35920/ema_state.pt",
        "hf_model_weights_path": f"output/{slug}/hf_model-final-ema/model.safetensors",
    }
    for key, expected in expected_artifact_paths.items():
        _expect(artifacts, key, expected, resolved)
    for key in ("ema_state_size_bytes", "hf_model_weights_size_bytes"):
        _integer(artifacts.get(key), f"{resolved}: training_protocol.artifacts.{key}", positive=True)

    architecture = _mapping(payload.get("architecture"), f"{resolved}: architecture")
    variant = VARIANTS[variant_id]
    expected_architecture = {
        "ablation_id": variant_id,
        "image_query_stage_mode": variant.query_stage_mode,
        "image_observed_position_mode": variant.observed_position_mode,
        "image_rope_mode": variant.rope_mode,
        "image_space_to_depth_factor": variant.space_to_depth_factor,
        "image_canonical_grid_side": 16,
        "image_canonical_latent_dim": 16,
        "image_grid_side": 16 // variant.space_to_depth_factor,
        "image_tokens_per_img": 256 // (variant.space_to_depth_factor**2),
        "image_latent_dim": 16 * (variant.space_to_depth_factor**2),
        "padded_sequence_length": 320 if variant.space_to_depth_factor == 1 else 128,
    }
    for key, expected in expected_architecture.items():
        _expect(architecture, key, expected, resolved)
    if variant.query_stage_mode == "fixed_sincos":
        implementation_contracts = _mapping(
            payload.get("implementation_contracts"),
            f"{resolved}: implementation_contracts",
        )
        _expect(
            implementation_contracts,
            "image_stage_buffer",
            STAGE_BUFFER_IMPLEMENTATION_CONTRACT,
            resolved,
        )

    flow_head = _mapping(architecture.get("flow_head"), f"{resolved}: architecture.flow_head")
    expected_flow_head = {
        "arch": FLOW_HEAD_INVARIANTS["image_flow_head_arch"],
        "depth": FLOW_HEAD_INVARIANTS["image_flow_depth"],
        "width": FLOW_HEAD_INVARIANTS["image_flow_width"],
        "mlp_ratio": FLOW_HEAD_INVARIANTS["image_flow_mlp_ratio"],
        "latent_mixer_heads": FLOW_HEAD_INVARIANTS["image_flow_latent_mixer_heads"],
        "latent_mixer_dropout": FLOW_HEAD_INVARIANTS[
            "image_flow_latent_mixer_dropout"
        ],
        "zero_init_gate": FLOW_HEAD_INVARIANTS["image_flow_latent_mixer_zero_init_gate"],
    }
    if dict(flow_head) != expected_flow_head:
        raise SummaryError(f"{resolved}: fixed contextual flow-head contract drifted")

    parameters = _mapping(payload.get("parameters"), f"{resolved}: parameters")
    parameter_counts = {
        key: _integer(parameters.get(key), f"{resolved}: parameters.{key}", positive=True)
        for key in ("total", "trainable", "image_embedder", "flow_head")
    }

    strategies = _mapping(payload.get("strategies"), f"{resolved}: strategies")
    if set(strategies) != {STRATEGY}:
        raise SummaryError(
            f"{resolved}: expected exactly one strategy {STRATEGY!r}, "
            f"got {sorted(strategies)!r}"
        )
    result = _mapping(strategies.get(STRATEGY), f"{resolved}: strategies.{STRATEGY}")
    _expect(result, "count", 10_000, resolved)
    fid = _finite(result.get("fid"), f"{resolved}: fid", positive=True)
    is_mean = _finite(
        result.get("inception_score_mean"), f"{resolved}: inception_score_mean", positive=True
    )
    is_std = _finite(
        result.get("inception_score_std"), f"{resolved}: inception_score_std"
    )
    wall_seconds = _finite(
        result.get("generation_wall_seconds"),
        f"{resolved}: generation_wall_seconds",
        positive=True,
    )
    samples_per_second = _finite(
        result.get("generation_samples_per_second"),
        f"{resolved}: generation_samples_per_second",
        positive=True,
    )
    latent_mse = _finite(
        result.get("latent_mse_to_target"),
        f"{resolved}: latent_mse_to_target",
        positive=True,
    )
    latent_rms = _finite(
        result.get("latent_rms"),
        f"{resolved}: latent_rms",
        positive=True,
    )
    generation_step_max = _finite(
        result.get("generation_step_max"),
        f"{resolved}: generation_step_max",
        positive=True,
    )
    expected_rate = 10_000 / wall_seconds
    if not math.isclose(samples_per_second, expected_rate, rel_tol=1.0e-6):
        raise SummaryError(f"{resolved}: generation throughput is inconsistent with wall time")

    row = {
        "id": variant_id,
        "training_seed": training_seed,
        "purpose": variant.purpose,
        "fid": fid,
        "is_mean": is_mean,
        "is_std": is_std,
        "sampling_wall_seconds": wall_seconds,
        "sampling_samples_per_second": samples_per_second,
        "latent_mse_to_target": latent_mse,
        "latent_rms": latent_rms,
        "generation_step_max": generation_step_max,
        "peak_cuda_allocated_mib": peak_allocated,
        "peak_cuda_reserved_mib": peak_reserved,
        **parameter_counts,
        "metrics_path": str(resolved),
        "training_protocol_sha256": expected_fingerprint,
    }
    return row, _protocol_signature(payload), payload


def _validate_confirmation_run(
    row: Mapping[str, Any],
    payload: Mapping[str, Any],
    *,
    candidate_manifest_sha256: str,
) -> dict[str, Any]:
    """Validate checkpoint-bound training and evaluator pairing evidence."""

    source = Path(str(row["metrics_path"]))
    variant_id = str(row["id"])
    training_seed = int(row["training_seed"])
    if training_seed not in CONFIRMATION_SEEDS:
        raise SummaryError(
            f"{source}: strict confirmation provenance is only defined for "
            f"training seeds {sorted(CONFIRMATION_SEEDS)}"
        )
    slug = run_slug(variant_id, training_seed)
    factor = int(VARIANTS[variant_id].space_to_depth_factor)

    training_protocol = _mapping(
        payload.get("training_protocol"), f"{source}: training_protocol"
    )
    confirmation = _mapping(
        training_protocol.get("confirmation"),
        f"{source}: training_protocol.confirmation",
    )
    declaration_sha256 = _sha256(
        confirmation.get("declaration_sha256"),
        f"{source}: confirmation.declaration_sha256",
    )
    screen_summary_sha256 = _sha256(
        confirmation.get("screen_summary_sha256"),
        f"{source}: confirmation.screen_summary_sha256",
    )
    _expect(
        confirmation,
        "candidate_manifest_sha256",
        candidate_manifest_sha256,
        source,
    )
    _expect(
        confirmation,
        "evaluator_rng_contract_sha256",
        EVALUATOR_RNG_CONTRACT_SHA256,
        source,
    )
    _expect(confirmation, "dataloader_shuffle_seed", training_seed, source)
    provenance_sha256 = _sha256(
        confirmation.get("provenance_sha256"),
        f"{source}: confirmation.provenance_sha256",
    )
    expected_provenance_path = f"output/{slug}/confirmation_training_provenance.json"
    _expect(confirmation, "provenance_path", expected_provenance_path, source)

    artifacts = _mapping(
        training_protocol.get("artifacts"), f"{source}: training_protocol.artifacts"
    )
    _expect(
        artifacts,
        "confirmation_provenance_path",
        expected_provenance_path,
        source,
    )
    _expect(
        artifacts,
        "confirmation_hf_provenance_path",
        f"output/{slug}/hf_model-final-ema/confirmation_training_provenance.json",
        source,
    )

    provenance = _mapping(
        confirmation.get("provenance"),
        f"{source}: training_protocol.confirmation.provenance",
    )
    _expect(provenance, "schema", CONFIRMATION_PROVENANCE_SCHEMA, source)
    _expect(provenance, "ablation_id", variant_id, source)
    _expect(provenance, "training_seed", training_seed, source)
    _expect(provenance, "provenance_sha256", provenance_sha256, source)
    _expect(
        provenance,
        "confirmation_declaration_sha256",
        declaration_sha256,
        source,
    )
    _expect(provenance, "space_to_depth_factor", factor, source)

    initial_state = _mapping(
        provenance.get("initial_state"), f"{source}: provenance.initial_state"
    )
    _expect(initial_state, "contract", INITIALIZATION_CONTRACT, source)
    image_modules = _mapping(
        initial_state.get("image_modules"),
        f"{source}: provenance.initial_state.image_modules",
    )
    image_parameter_count = _integer(
        image_modules.get("parameter_count"),
        f"{source}: image_modules.parameter_count",
        positive=True,
    )
    image_parameter_schema_sha256 = _sha256(
        image_modules.get("parameter_schema_sha256"),
        f"{source}: image_modules.parameter_schema_sha256",
    )
    image_state_sha256 = _sha256(
        image_modules.get("state_sha256"),
        f"{source}: image_modules.state_sha256",
    )
    special_token_rows_sha256 = _sha256(
        initial_state.get("special_token_rows_sha256"),
        f"{source}: initial_state.special_token_rows_sha256",
    )
    special_tokens = initial_state.get("special_token_names_and_ids")
    if (
        not isinstance(special_tokens, list)
        or not special_tokens
        or any(
            not isinstance(item, list)
            or len(item) != 2
            or not isinstance(item[0], str)
            or not isinstance(item[1], int)
            or isinstance(item[1], bool)
            for item in special_tokens
        )
    ):
        raise SummaryError(
            f"{source}: initial_state.special_token_names_and_ids is invalid"
        )

    train_data = _mapping(
        provenance.get("train_data"), f"{source}: provenance.train_data"
    )
    _expect(train_data, "contract", TRAIN_ORDER_CONTRACT, source)
    _expect(train_data, "augmentation_contract", AUGMENTATION_CONTRACT, source)
    _expect(train_data, "dataloader_shuffle_seed", training_seed, source)
    _expect(train_data, "augmentation_seed", training_seed, source)
    initial_generator_state_sha256 = _sha256(
        train_data.get("initial_generator_state_sha256"),
        f"{source}: train_data.initial_generator_state_sha256",
    )
    dataloader_base_seed = _integer(
        train_data.get("dataloader_base_seed"),
        f"{source}: train_data.dataloader_base_seed",
    )
    dataset_length = _integer(
        train_data.get("dataset_length"),
        f"{source}: train_data.dataset_length",
        positive=True,
    )
    epoch0_order_sha256 = _sha256(
        train_data.get("epoch0_ordered_sample_identity_sha256"),
        f"{source}: train_data.epoch0_ordered_sample_identity_sha256",
    )
    epoch0_augmentation_sha256 = _sha256(
        train_data.get("epoch0_augmentation_decisions_sha256"),
        f"{source}: train_data.epoch0_augmentation_decisions_sha256",
    )
    _finite(
        train_data.get("latent_hflip_probability"),
        f"{source}: train_data.latent_hflip_probability",
    )
    for key, expected in {
        "batch_size_per_rank": 32,
        "total_batch_size": 256,
        "drop_last": True,
    }.items():
        _expect(train_data, key, expected, source)
    _integer(train_data.get("num_workers"), f"{source}: train_data.num_workers", positive=True)
    if not isinstance(train_data.get("persistent_workers"), bool):
        raise SummaryError(f"{source}: train_data.persistent_workers must be boolean")

    input_files = _mapping(
        train_data.get("input_files"), f"{source}: train_data.input_files"
    )
    required_input_files = {"cache", "manifest", "split_manifest", "synset_mapping"}
    if set(input_files) != required_input_files:
        raise SummaryError(
            f"{source}: train_data.input_files must contain exactly "
            f"{sorted(required_input_files)}"
        )
    for label, raw_entry in input_files.items():
        entry = _mapping(raw_entry, f"{source}: train_data.input_files.{label}")
        if not isinstance(entry.get("path"), str) or not entry["path"]:
            raise SummaryError(f"{source}: input_files.{label}.path must be nonempty")
        _integer(
            entry.get("size_bytes"),
            f"{source}: input_files.{label}.size_bytes",
            positive=True,
        )
        _sha256(entry.get("sha256"), f"{source}: input_files.{label}.sha256")

    base_model_manifest_sha256 = _sha256(
        provenance.get("base_model_manifest_sha256"),
        f"{source}: provenance.base_model_manifest_sha256",
    )
    runtime_source_manifest_sha256 = _sha256(
        provenance.get("runtime_source_manifest_sha256"),
        f"{source}: provenance.runtime_source_manifest_sha256",
    )

    contracts = _mapping(
        payload.get("implementation_contracts"),
        f"{source}: implementation_contracts",
    )
    evaluator_rng_contract = _mapping(
        contracts.get("evaluator_rng_contract"),
        f"{source}: implementation_contracts.evaluator_rng_contract",
    )
    _expect(
        evaluator_rng_contract,
        "schema",
        EVALUATOR_RNG_CONTRACT_SCHEMA,
        source,
    )
    evaluator_rng_contract_sha256 = _sha256(
        contracts.get("evaluator_rng_contract_sha256"),
        f"{source}: implementation_contracts.evaluator_rng_contract_sha256",
    )
    if canonical_sha256(dict(evaluator_rng_contract)) != evaluator_rng_contract_sha256:
        raise SummaryError(f"{source}: evaluator RNG contract digest mismatch")
    if evaluator_rng_contract_sha256 != EVALUATOR_RNG_CONTRACT_SHA256:
        raise SummaryError(f"{source}: evaluator RNG contract is not the frozen contract")
    _expect(contracts, "canonical_initial_noise_enabled", True, source)
    _expect(
        contracts,
        "canonical_noise_manifest_schema",
        CANONICAL_NOISE_MANIFEST_SCHEMA,
        source,
    )
    canonical_noise_manifest_sha256 = _sha256(
        contracts.get("canonical_noise_manifest_sha256"),
        f"{source}: implementation_contracts.canonical_noise_manifest_sha256",
    )
    _expect(
        contracts,
        "ordered_eval_sample_manifest_schema",
        ORDERED_EVAL_SAMPLE_MANIFEST_SCHEMA,
        source,
    )
    ordered_eval_sample_manifest_sha256 = _sha256(
        contracts.get("ordered_eval_sample_manifest_sha256"),
        f"{source}: implementation_contracts.ordered_eval_sample_manifest_sha256",
    )
    _expect(contracts, "paired_sample_count", 10_000, source)

    return {
        "source": str(source),
        "id": variant_id,
        "training_seed": training_seed,
        "space_to_depth_factor": factor,
        "declaration_sha256": declaration_sha256,
        "screen_summary_sha256": screen_summary_sha256,
        "candidate_manifest_sha256": candidate_manifest_sha256,
        "provenance_sha256": provenance_sha256,
        "image_parameter_count": image_parameter_count,
        "image_parameter_schema_sha256": image_parameter_schema_sha256,
        "image_state_sha256": image_state_sha256,
        "special_token_rows_sha256": special_token_rows_sha256,
        "initial_generator_state_sha256": initial_generator_state_sha256,
        "dataloader_base_seed": dataloader_base_seed,
        "dataset_length": dataset_length,
        "epoch0_ordered_sample_identity_sha256": epoch0_order_sha256,
        "epoch0_augmentation_decisions_sha256": epoch0_augmentation_sha256,
        "input_files_sha256": canonical_sha256(dict(input_files)),
        "base_model_manifest_sha256": base_model_manifest_sha256,
        "runtime_source_manifest_sha256": runtime_source_manifest_sha256,
        "evaluator_rng_contract_sha256": evaluator_rng_contract_sha256,
        "canonical_noise_manifest_sha256": canonical_noise_manifest_sha256,
        "ordered_eval_sample_manifest_sha256": ordered_eval_sample_manifest_sha256,
    }


def _require_paired_fields(
    evidence: Sequence[Mapping[str, Any]],
    fields: Sequence[str],
    *,
    scope: str,
) -> None:
    for field in fields:
        values = {str(item[field]) for item in evidence}
        if len(values) != 1:
            details = {
                f"{item['id']}@{item['training_seed']}": item[field]
                for item in evidence
            }
            raise SummaryError(
                f"confirmation pairing mismatch for {field} within {scope}: {details}"
            )


def _validate_confirmation_pairing(
    rows: Sequence[Mapping[str, Any]],
    payloads: Mapping[tuple[str, int], Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    manifest_sha256 = canonical_sha256(dict(manifest))
    evidence = [
        _validate_confirmation_run(
            row,
            payloads[(str(row["id"]), int(row["training_seed"]))],
            candidate_manifest_sha256=manifest_sha256,
        )
        for row in rows
    ]

    by_seed: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for item in evidence:
        by_seed[int(item["training_seed"])].append(item)
    seed_paired_fields = (
        "screen_summary_sha256",
        "candidate_manifest_sha256",
        "special_token_rows_sha256",
        "initial_generator_state_sha256",
        "dataloader_base_seed",
        "dataset_length",
        "epoch0_ordered_sample_identity_sha256",
        "epoch0_augmentation_decisions_sha256",
        "input_files_sha256",
        "base_model_manifest_sha256",
        "runtime_source_manifest_sha256",
    )
    for seed, seed_evidence in sorted(by_seed.items()):
        _require_paired_fields(seed_evidence, seed_paired_fields, scope=f"seed {seed}")

        by_factor: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
        for item in seed_evidence:
            by_factor[int(item["space_to_depth_factor"])].append(item)
        for factor, factor_evidence in sorted(by_factor.items()):
            _require_paired_fields(
                factor_evidence,
                (
                    "image_parameter_count",
                    "image_parameter_schema_sha256",
                    "image_state_sha256",
                ),
                scope=f"seed {seed}, S2D factor {factor}",
            )

    _require_paired_fields(
        evidence,
        (
            "screen_summary_sha256",
            "candidate_manifest_sha256",
            "dataset_length",
            "input_files_sha256",
            "base_model_manifest_sha256",
            "runtime_source_manifest_sha256",
            "evaluator_rng_contract_sha256",
            "canonical_noise_manifest_sha256",
            "ordered_eval_sample_manifest_sha256",
        ),
        scope="all confirmation runs",
    )
    return {
        "schema": "selfless_flow_image_embedder_confirmation_pairing_gate_v1",
        "candidate_manifest_sha256": manifest_sha256,
        "screen_summary_sha256": evidence[0]["screen_summary_sha256"],
        "evaluator_rng_contract_sha256": evidence[0][
            "evaluator_rng_contract_sha256"
        ],
        "canonical_noise_manifest_sha256": evidence[0][
            "canonical_noise_manifest_sha256"
        ],
        "ordered_eval_sample_manifest_sha256": evidence[0][
            "ordered_eval_sample_manifest_sha256"
        ],
        "validated_runs": len(evidence),
    }


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_id: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_id[str(row["id"])].append(row)
    aggregates = []
    for variant_id in VARIANTS:
        group = by_id.get(variant_id, [])
        if not group:
            continue
        fids = [float(row["fid"]) for row in group]
        scores = [float(row["is_mean"]) for row in group]
        rates = [float(row["sampling_samples_per_second"]) for row in group]
        aggregates.append(
            {
                "id": variant_id,
                "seeds": sorted(int(row["training_seed"]) for row in group),
                "runs": len(group),
                "fid_mean": mean(fids),
                "fid_sample_std": stdev(fids) if len(fids) > 1 else None,
                "is_mean": mean(scores),
                "is_sample_std": stdev(scores) if len(scores) > 1 else None,
                "sampling_samples_per_second_mean": mean(rates),
            }
        )
    return aggregates


def _effect(lookup: Mapping[str, Mapping[str, Any]], lhs: str, rhs: str) -> dict[str, float] | None:
    if lhs not in lookup or rhs not in lookup:
        return None
    return {
        "fid_delta": float(lookup[lhs]["fid_mean"]) - float(lookup[rhs]["fid_mean"]),
        "is_delta": float(lookup[lhs]["is_mean"]) - float(lookup[rhs]["is_mean"]),
    }


def _factorial_contrast(
    values: Mapping[tuple[int, ...], float],
    selected_factors: Sequence[int],
) -> float:
    """Average a binary finite difference over all unselected factors."""

    factor_count = len(next(iter(values)))
    selected = tuple(selected_factors)
    remaining = tuple(index for index in range(factor_count) if index not in selected)
    total = 0.0
    for remaining_mask in range(1 << len(remaining)):
        fixed = [0] * factor_count
        for bit_index, factor_index in enumerate(remaining):
            fixed[factor_index] = (remaining_mask >> bit_index) & 1
        contrast = 0.0
        for selected_mask in range(1 << len(selected)):
            setting = list(fixed)
            selected_sum = 0
            for bit_index, factor_index in enumerate(selected):
                bit = (selected_mask >> bit_index) & 1
                setting[factor_index] = bit
                selected_sum += bit
            sign = -1.0 if (len(selected) - selected_sum) % 2 else 1.0
            contrast += sign * values[tuple(setting)]
        total += contrast
    return total / float(1 << len(remaining))


def _expanded_factorial_effects(
    lookup: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any] | None:
    if not all(variant_id in lookup for variant_id in EXPANDED_MATRIX_IDS):
        return None
    factor_names = ("S", "Ra", "Rb", "D")
    result: dict[str, Any] = {}
    for metric in ("fid_mean", "is_mean"):
        values = {
            FOUR_FACTOR_SETTINGS[variant_id]: float(lookup[variant_id][metric])
            for variant_id in EXPANDED_MATRIX_IDS
        }
        main_effects = {
            factor_names[index]: _factorial_contrast(values, (index,))
            for index in range(4)
        }
        two_way = {
            f"{factor_names[first]}x{factor_names[second]}": _factorial_contrast(
                values, (first, second)
            )
            for first in range(4)
            for second in range(first + 1, 4)
        }
        three_way = {
            "x".join(factor_names[index] for index in indices): _factorial_contrast(
                values, indices
            )
            for indices in ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))
        }
        result[metric] = {
            "average_main_effects": main_effects,
            "average_two_way_interactions": two_way,
            "average_three_way_interactions": three_way,
            "four_way_interaction_SxRaxRbxD": _factorial_contrast(
                values, (0, 1, 2, 3)
            ),
        }
    return result


def _effects(aggregates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    lookup = {str(row["id"]): row for row in aggregates}
    pairs = {
        "stage_E1_minus_E0": ("E1", "E0"),
        "remove_observed_position_E2a_minus_E0": ("E2a", "E0"),
        "row_col_rope_E2b_minus_E0": ("E2b", "E0"),
        "full_R_E2_minus_E0": ("E2", "E0"),
        "space_to_depth_E3_minus_E0": ("E3", "E0"),
        "stage_given_R_E4_minus_E2": ("E4", "E2"),
        "stage_given_D_E5_minus_E3": ("E5", "E3"),
        "R_given_D_E6_minus_E3": ("E6", "E3"),
        "stage_given_RD_E7_minus_E6": ("E7", "E6"),
        "R_given_SD_E7_minus_E5": ("E7", "E5"),
        "D_given_SR_E7_minus_E4": ("E7", "E4"),
        "stage_given_Ra_E4a_minus_E2a": ("E4a", "E2a"),
        "stage_given_Rb_E4b_minus_E2b": ("E4b", "E2b"),
        "Ra_given_D_E6a_minus_E3": ("E6a", "E3"),
        "Rb_given_D_E6b_minus_E3": ("E6b", "E3"),
        "stage_given_RaD_E7a_minus_E6a": ("E7a", "E6a"),
        "stage_given_RbD_E7b_minus_E6b": ("E7b", "E6b"),
    }
    result = {
        name: value
        for name, (lhs, rhs) in pairs.items()
        if (value := _effect(lookup, lhs, rhs)) is not None
    }
    if all(key in lookup for key in ("E0", "E2a", "E2b", "E2")):
        result["R_subfactor_interaction"] = {
            metric: (
                float(lookup["E2"][metric])
                - float(lookup["E2a"][metric])
                - float(lookup["E2b"][metric])
                + float(lookup["E0"][metric])
            )
            for metric in ("fid_mean", "is_mean")
        }
    core = ("E0", "E1", "E2", "E3", "E4", "E5", "E6", "E7")
    if all(key in lookup for key in core):
        def value(variant_id: str, metric: str) -> float:
            return float(lookup[variant_id][metric])

        factorial = {}
        for metric in ("fid_mean", "is_mean"):
            s_simple = [
                value("E1", metric) - value("E0", metric),
                value("E4", metric) - value("E2", metric),
                value("E5", metric) - value("E3", metric),
                value("E7", metric) - value("E6", metric),
            ]
            r_simple = [
                value("E2", metric) - value("E0", metric),
                value("E4", metric) - value("E1", metric),
                value("E6", metric) - value("E3", metric),
                value("E7", metric) - value("E5", metric),
            ]
            d_simple = [
                value("E3", metric) - value("E0", metric),
                value("E5", metric) - value("E1", metric),
                value("E6", metric) - value("E2", metric),
                value("E7", metric) - value("E4", metric),
            ]
            sr_at_d0 = s_simple[1] - s_simple[0]
            sr_at_d1 = s_simple[3] - s_simple[2]
            sd_at_r0 = s_simple[2] - s_simple[0]
            sd_at_r1 = s_simple[3] - s_simple[1]
            rd_at_s0 = r_simple[2] - r_simple[0]
            rd_at_s1 = r_simple[3] - r_simple[1]
            factorial[metric] = {
                "average_main_effects": {
                    "S": mean(s_simple),
                    "R": mean(r_simple),
                    "D": mean(d_simple),
                },
                "average_two_way_interactions": {
                    "SxR": mean((sr_at_d0, sr_at_d1)),
                    "SxD": mean((sd_at_r0, sd_at_r1)),
                    "RxD": mean((rd_at_s0, rd_at_s1)),
                },
                "three_way_interaction_SxRxD": sr_at_d1 - sr_at_d0,
            }
        result["factorial_2x2x2"] = factorial
    expanded = _expanded_factorial_effects(lookup)
    if expanded is not None:
        result["factorial_2x2x2x2"] = expanded
    return result


def _pareto(aggregates: Sequence[Mapping[str, Any]]) -> list[str]:
    frontier = []
    for row in aggregates:
        dominated = any(
            other is not row
            and float(other["fid_mean"]) <= float(row["fid_mean"])
            and float(other["is_mean"]) >= float(row["is_mean"])
            and (
                float(other["fid_mean"]) < float(row["fid_mean"])
                or float(other["is_mean"]) > float(row["is_mean"])
            )
            for other in aggregates
        )
        if not dominated:
            frontier.append(str(row["id"]))
    return frontier


def _fid_throughput_pareto(aggregates: Sequence[Mapping[str, Any]]) -> list[str]:
    frontier = []
    for row in aggregates:
        dominated = any(
            other is not row
            and float(other["fid_mean"]) <= float(row["fid_mean"])
            and float(other["sampling_samples_per_second_mean"])
            >= float(row["sampling_samples_per_second_mean"])
            and (
                float(other["fid_mean"]) < float(row["fid_mean"])
                or float(other["sampling_samples_per_second_mean"])
                > float(row["sampling_samples_per_second_mean"])
            )
            for other in aggregates
        )
        if not dominated:
            frontier.append(str(row["id"]))
    return frontier


def _confirmation_candidate_manifest(
    aggregates: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    lookup = {str(row["id"]): row for row in aggregates}
    if set(lookup) != set(EXPANDED_MATRIX_IDS):
        return None
    if any(row.get("seeds") != [42] for row in aggregates):
        return None

    best_fid = min(float(row["fid_mean"]) for row in aggregates)
    near_best = {
        str(row["id"])
        for row in aggregates
        if float(row["fid_mean"]) <= best_fid + NEAR_BEST_FID_MARGIN
    }
    fid_is_pareto = set(_pareto(aggregates))
    fid_throughput_pareto = set(_fid_throughput_pareto(aggregates))
    baseline_rate = float(lookup["E0"]["sampling_samples_per_second_mean"])
    speed_candidates = {
        variant_id
        for variant_id in fid_throughput_pareto
        if float(lookup[variant_id]["sampling_samples_per_second_mean"])
        >= SPEED_ADVANTAGE_RATIO * baseline_rate
    }
    selected = {"E0"} | near_best | fid_is_pareto | speed_candidates
    ordered = [variant_id for variant_id in EXPANDED_MATRIX_IDS if variant_id in selected]
    return {
        "schema": CONFIRMATION_MANIFEST_SCHEMA,
        "screen_summary_schema": SCHEMA,
        "screen_training_seed": 42,
        "confirmation_training_seeds": sorted(CONFIRMATION_SEEDS),
        "near_best_fid_margin": NEAR_BEST_FID_MARGIN,
        "speed_advantage_ratio_vs_e0": SPEED_ADVANTAGE_RATIO,
        "near_best_fid_ids": [
            variant_id for variant_id in EXPANDED_MATRIX_IDS if variant_id in near_best
        ],
        "fid_is_pareto_ids": [
            variant_id for variant_id in EXPANDED_MATRIX_IDS if variant_id in fid_is_pareto
        ],
        "speed_pareto_ids_meeting_threshold": [
            variant_id for variant_id in EXPANDED_MATRIX_IDS if variant_id in speed_candidates
        ],
        "candidate_ids": ordered,
    }


def _t95_critical(sample_count: int) -> float:
    """Two-sided 95% Student-t critical value without adding a SciPy dependency."""

    if sample_count < 2:
        raise SummaryError("a confidence interval requires at least two paired seeds")
    values = {
        1: 12.706,
        2: 4.303,
        3: 3.182,
        4: 2.776,
        5: 2.571,
        6: 2.447,
        7: 2.365,
        8: 2.306,
        9: 2.262,
        10: 2.228,
        11: 2.201,
        12: 2.179,
        13: 2.160,
        14: 2.145,
        15: 2.131,
        16: 2.120,
        17: 2.110,
        18: 2.101,
        19: 2.093,
        20: 2.086,
        21: 2.080,
        22: 2.074,
        23: 2.069,
        24: 2.064,
        25: 2.060,
        26: 2.056,
        27: 2.052,
        28: 2.048,
        29: 2.045,
        30: 2.042,
    }
    degrees_of_freedom = sample_count - 1
    return values.get(degrees_of_freedom, 1.96)


def _paired_statistic(values: Sequence[float]) -> dict[str, Any]:
    count = len(values)
    result = {
        "values": list(values),
        "mean": mean(values),
        "sample_std": stdev(values) if count > 1 else None,
        "ci95": None,
    }
    if count > 1:
        half_width = _t95_critical(count) * float(result["sample_std"]) / math.sqrt(count)
        result["ci95"] = [result["mean"] - half_width, result["mean"] + half_width]
    return result


def _paired_vs_e0(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    lookup = {
        (str(row["id"]), int(row["training_seed"])): row
        for row in rows
    }
    seed_sets: dict[str, set[int]] = defaultdict(set)
    for row in rows:
        seed_sets[str(row["id"])].add(int(row["training_seed"]))
    baseline_seeds = seed_sets.get("E0", set())
    comparisons = {}
    for variant_id, candidate_seeds in seed_sets.items():
        if variant_id == "E0":
            continue
        paired_seeds = sorted(baseline_seeds & candidate_seeds)
        if not paired_seeds:
            continue
        fid_deltas = [
            float(lookup[(variant_id, seed)]["fid"])
            - float(lookup[("E0", seed)]["fid"])
            for seed in paired_seeds
        ]
        is_deltas = [
            float(lookup[(variant_id, seed)]["is_mean"])
            - float(lookup[("E0", seed)]["is_mean"])
            for seed in paired_seeds
        ]
        comparisons[variant_id] = {
            "seeds": paired_seeds,
            "fid_candidate_minus_e0": {
                **_paired_statistic(fid_deltas),
                "candidate_wins": sum(delta < 0.0 for delta in fid_deltas),
                "exact_ties": sum(delta == 0.0 for delta in fid_deltas),
            },
            "is_candidate_minus_e0": _paired_statistic(is_deltas),
        }
    return comparisons


def _validate_seed_contract(rows: Sequence[Mapping[str, Any]], expected: str) -> None:
    seed_sets: dict[str, set[int]] = defaultdict(set)
    for row in rows:
        seed_sets[str(row["id"])].add(int(row["training_seed"]))

    if expected in {"first-stage", "full", "expanded"}:
        required_ids = {
            "first-stage": FIRST_STAGE_IDS,
            "full": FULL_MATRIX_IDS,
            "expanded": EXPANDED_MATRIX_IDS,
        }[expected]
        drift = {
            variant_id: sorted(seed_sets[variant_id])
            for variant_id in required_ids
            if seed_sets[variant_id] != {42}
        }
        if drift:
            raise SummaryError(
                f"expected={expected} is a seed-42 architecture screen; "
                f"got seed sets {drift}"
            )
        return

    if expected != "confirmation":
        return
    if "E0" not in seed_sets or len(seed_sets) < 2:
        raise SummaryError("confirmation requires E0 and at least one candidate")
    common = seed_sets["E0"]
    mismatched = {
        variant_id: sorted(seeds)
        for variant_id, seeds in seed_sets.items()
        if seeds != common
    }
    if mismatched:
        raise SummaryError(
            "confirmation requires identical paired training-seed sets; "
            f"E0={sorted(common)}, mismatched={mismatched}"
        )
    if common != CONFIRMATION_SEEDS:
        raise SummaryError(
            "confirmation requires the preregistered paired training seeds "
            f"{sorted(CONFIRMATION_SEEDS)}, got {sorted(common)}"
        )


def _confirmation_scope(
    manifest: Mapping[str, Any] | None,
    scope_amendment: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], set[str], dict[str, Any]]:
    if manifest is None:
        raise SummaryError(
            "confirmation requires the candidate manifest from the expanded seed-42 screen"
        )
    manifest = dict(_mapping(manifest, "confirmation candidate manifest"))
    if manifest.get("schema") != CONFIRMATION_MANIFEST_SCHEMA:
        raise SummaryError(
            "confirmation candidate manifest schema mismatch: "
            f"expected {CONFIRMATION_MANIFEST_SCHEMA!r}, got {manifest.get('schema')!r}"
        )
    if manifest.get("screen_summary_schema") != SCHEMA:
        raise SummaryError("confirmation candidate manifest has the wrong screen summary schema")
    if manifest.get("screen_training_seed") != 42:
        raise SummaryError("confirmation candidate manifest must come from training seed 42")
    if manifest.get("confirmation_training_seeds") != sorted(CONFIRMATION_SEEDS):
        raise SummaryError("confirmation candidate manifest has the wrong confirmation seed set")
    raw_ids = manifest.get("candidate_ids")
    if not isinstance(raw_ids, list) or not raw_ids:
        raise SummaryError("confirmation candidate manifest candidate_ids must be a non-empty list")
    candidate_ids = [normalize_variant_id(value) for value in raw_ids]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise SummaryError("confirmation candidate manifest contains duplicate candidate IDs")
    candidate_set = set(candidate_ids)
    if "E0" not in candidate_set or len(candidate_set) < 2:
        raise SummaryError("confirmation candidate manifest requires E0 and at least one candidate")
    for key, expected in {
        "near_best_fid_margin": NEAR_BEST_FID_MARGIN,
        "speed_advantage_ratio_vs_e0": SPEED_ADVANTAGE_RATIO,
    }.items():
        if manifest.get(key) != expected:
            raise SummaryError(f"confirmation candidate manifest {key} drifted")
    selected = {"E0"}
    for field in (
        "near_best_fid_ids",
        "fid_is_pareto_ids",
        "speed_pareto_ids_meeting_threshold",
    ):
        values = manifest.get(field)
        if (
            not isinstance(values, list)
            or len(values) != len(set(values))
            or any(value not in EXPANDED_MATRIX_IDS for value in values)
        ):
            raise SummaryError(f"confirmation candidate manifest {field} is invalid")
        selected.update(values)
    expected_ids = [variant_id for variant_id in EXPANDED_MATRIX_IDS if variant_id in selected]
    if candidate_ids != expected_ids:
        raise SummaryError(
            "confirmation candidate manifest candidate_ids do not equal the "
            "preregistered selector union"
        )
    manifest["candidate_ids"] = candidate_ids
    provenance_manifest = dict(manifest)
    if scope_amendment is None:
        return manifest, candidate_set, provenance_manifest

    amendment = dict(
        _mapping(scope_amendment, "confirmation candidate scope amendment")
    )
    if amendment.get("schema") != CONFIRMATION_SCOPE_AMENDMENT_SCHEMA:
        raise SummaryError(
            "confirmation scope amendment schema mismatch: "
            f"expected {CONFIRMATION_SCOPE_AMENDMENT_SCHEMA!r}, "
            f"got {amendment.get('schema')!r}"
        )
    parent_manifest_sha256 = _sha256(
        amendment.get("parent_candidate_manifest_sha256"),
        "confirmation scope amendment parent_candidate_manifest_sha256",
    )
    manifest_sha256 = canonical_sha256(provenance_manifest)
    if parent_manifest_sha256 != manifest_sha256:
        raise SummaryError(
            "confirmation scope amendment does not reference the frozen candidate manifest"
        )
    _sha256(
        amendment.get("parent_screen_summary_sha256"),
        "confirmation scope amendment parent_screen_summary_sha256",
    )
    for field in ("created_at_utc", "reason", "parent_screen_summary_path"):
        if not isinstance(amendment.get(field), str) or not amendment[field].strip():
            raise SummaryError(f"confirmation scope amendment {field} must be nonempty")
    if amendment.get("confirmation_training_seeds") != sorted(CONFIRMATION_SEEDS):
        raise SummaryError("confirmation scope amendment has the wrong confirmation seed set")
    if amendment.get("confirmation_metrics_observed_before_amendment") is not False:
        raise SummaryError(
            "confirmation scope amendment must attest that no confirmation metrics were observed"
        )
    if amendment.get("required_space_to_depth_factor") != 1:
        raise SummaryError(
            "confirmation scope amendment must require space-to-depth factor 1"
        )

    raw_original_ids = amendment.get("original_candidate_ids")
    if not isinstance(raw_original_ids, list):
        raise SummaryError(
            "confirmation scope amendment original_candidate_ids must be a list"
        )
    original_ids = [normalize_variant_id(value) for value in raw_original_ids]
    if original_ids != candidate_ids:
        raise SummaryError(
            "confirmation scope amendment original_candidate_ids do not match the "
            "frozen candidate manifest"
        )

    raw_effective_ids = amendment.get("confirmation_candidate_ids")
    if not isinstance(raw_effective_ids, list) or not raw_effective_ids:
        raise SummaryError(
            "confirmation scope amendment confirmation_candidate_ids must be a non-empty list"
        )
    effective_ids = [normalize_variant_id(value) for value in raw_effective_ids]
    if len(effective_ids) != len(set(effective_ids)):
        raise SummaryError("confirmation scope amendment contains duplicate candidate IDs")
    expected_effective_ids = [
        variant_id
        for variant_id in candidate_ids
        if VARIANTS[variant_id].space_to_depth_factor == 1
    ]
    if effective_ids != expected_effective_ids:
        raise SummaryError(
            "confirmation scope amendment must retain exactly the factor-1 candidates "
            "from the frozen candidate manifest"
        )
    if "E0" not in effective_ids or len(effective_ids) < 2:
        raise SummaryError(
            "confirmation scope amendment requires E0 and at least one factor-1 candidate"
        )

    raw_removed_ids = amendment.get("removed_candidate_ids")
    if not isinstance(raw_removed_ids, list):
        raise SummaryError(
            "confirmation scope amendment removed_candidate_ids must be a list"
        )
    removed_ids = [normalize_variant_id(value) for value in raw_removed_ids]
    expected_removed_ids = [
        variant_id for variant_id in candidate_ids if variant_id not in effective_ids
    ]
    if removed_ids != expected_removed_ids:
        raise SummaryError(
            "confirmation scope amendment removed_candidate_ids do not equal the "
            "factor-2 candidates removed from the frozen manifest"
        )

    effective_scope = {
        "schema": CONFIRMATION_SCOPE_AMENDMENT_SCHEMA,
        "source_candidate_manifest_sha256": manifest_sha256,
        "source_candidate_ids": candidate_ids,
        "candidate_ids": effective_ids,
        "removed_candidate_ids": removed_ids,
        "required_space_to_depth_factor": 1,
        "scope_amendment_sha256": canonical_sha256(amendment),
        "scope_amendment": amendment,
    }
    return effective_scope, set(effective_ids), provenance_manifest


def build_summary(
    run_specs: Sequence[str],
    expected: str = "any",
    *,
    confirmation_manifest: Mapping[str, Any] | None = None,
    confirmation_scope_amendment: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not run_specs:
        raise SummaryError("at least one --run is required")
    rows = []
    payloads: dict[tuple[str, int], Mapping[str, Any]] = {}
    seen = set()
    protocol = None
    for spec in run_specs:
        variant_id, training_seed, path = _parse_run(spec)
        key = (variant_id, training_seed)
        if key in seen:
            raise SummaryError(f"duplicate run assignment for {variant_id}@{training_seed}")
        seen.add(key)
        row, signature, payload = _load_run(variant_id, training_seed, path)
        if protocol is None:
            protocol = signature
        elif signature != protocol:
            raise SummaryError(f"{path}: formal evaluation protocol differs across runs")
        rows.append(row)
        payloads[key] = payload

    required = {
        "any": set(),
        "first-stage": set(FIRST_STAGE_IDS),
        "full": set(FULL_MATRIX_IDS),
        "expanded": set(EXPANDED_MATRIX_IDS),
        "confirmation": {"E0"},
    }[expected]
    present = {str(row["id"]) for row in rows}
    missing = sorted(required - present)
    if missing:
        raise SummaryError(f"expected={expected} is missing variants: {', '.join(missing)}")
    if expected != "confirmation" and confirmation_scope_amendment is not None:
        raise SummaryError(
            "a confirmation scope amendment is only valid with expected=confirmation"
        )
    confirmation_scope = None
    confirmation_provenance_manifest = None
    if expected == "confirmation":
        (
            confirmation_scope,
            candidate_set,
            confirmation_provenance_manifest,
        ) = _confirmation_scope(
            confirmation_manifest,
            confirmation_scope_amendment,
        )
        if present != candidate_set:
            raise SummaryError(
                "confirmation runs must exactly match the screen-derived candidate manifest; "
                f"missing={sorted(candidate_set - present)}, "
                f"unexpected={sorted(present - candidate_set)}"
            )
    _validate_seed_contract(rows, expected)

    confirmation_pairing_gate = None
    if expected == "confirmation":
        confirmation_pairing_gate = _validate_confirmation_pairing(
            rows,
            payloads,
            confirmation_provenance_manifest,
        )
        if confirmation_scope_amendment is not None:
            expected_screen_sha256 = confirmation_scope["scope_amendment"][
                "parent_screen_summary_sha256"
            ]
            if (
                confirmation_pairing_gate["screen_summary_sha256"]
                != expected_screen_sha256
            ):
                raise SummaryError(
                    "confirmation scope amendment parent screen summary does not match "
                    "the checkpoint-bound confirmation provenance"
                )
            confirmation_scope["validated_parent_screen_summary_sha256"] = (
                expected_screen_sha256
            )

    order = {variant_id: index for index, variant_id in enumerate(VARIANTS)}
    rows.sort(key=lambda row: (order[str(row["id"])], int(row["training_seed"])))
    aggregates = _aggregate(rows)
    by_fid = sorted(aggregates, key=lambda row: (row["fid_mean"], -row["is_mean"], row["id"]))
    by_is = sorted(aggregates, key=lambda row: (-row["is_mean"], row["fid_mean"], row["id"]))
    fid_is_pareto = _pareto(aggregates)
    fid_throughput_pareto = _fid_throughput_pareto(aggregates)
    generated_confirmation_manifest = _confirmation_candidate_manifest(aggregates)
    return {
        "schema": SCHEMA,
        "expected": expected,
        "protocol": protocol,
        "runs": rows,
        "aggregates": aggregates,
        "ranking_by_fid": [row["id"] for row in by_fid],
        "ranking_by_is": [row["id"] for row in by_is],
        "best_by_fid": by_fid[0],
        "best_by_is": by_is[0],
        "fid_is_pareto_frontier": fid_is_pareto,
        "fid_throughput_pareto_frontier": fid_throughput_pareto,
        "confirmation_candidate_manifest": generated_confirmation_manifest,
        "confirmation_scope_manifest": confirmation_scope,
        "confirmation_pairing_gate": confirmation_pairing_gate,
        "effects": _effects(aggregates),
        "paired_vs_e0": _paired_vs_e0(rows),
    }


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as handle:
        handle.write(content)
        temporary = Path(handle.name)
    temporary.replace(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", default=[])
    parser.add_argument(
        "--expected",
        choices=("any", "first-stage", "full", "expanded", "confirmation"),
        default="any",
    )
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv")
    parser.add_argument(
        "--confirmation-screen-json",
        help="Expanded seed-42 summary JSON containing confirmation_candidate_manifest.",
    )
    parser.add_argument(
        "--confirmation-scope-amendment-json",
        help=(
            "Pre-metrics scope amendment that narrows the frozen screen manifest while "
            "preserving its checkpoint-bound provenance."
        ),
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    confirmation_manifest = None
    if args.confirmation_screen_json:
        try:
            screen_summary = json.loads(
                Path(args.confirmation_screen_json).read_text(encoding="utf-8")
            )
            screen_summary = _mapping(
                screen_summary,
                "confirmation screen summary",
            )
            if screen_summary.get("schema") != SCHEMA or screen_summary.get("expected") != "expanded":
                raise SummaryError(
                    "--confirmation-screen-json must be an expanded screen summary "
                    f"with schema {SCHEMA}"
                )
            confirmation_manifest = screen_summary.get("confirmation_candidate_manifest")
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise SummaryError(
                f"failed to read confirmation screen summary {args.confirmation_screen_json}: {exc}"
            ) from exc
    confirmation_scope_amendment = None
    if args.confirmation_scope_amendment_json:
        try:
            confirmation_scope_amendment = _mapping(
                json.loads(
                    Path(args.confirmation_scope_amendment_json).read_text(
                        encoding="utf-8"
                    )
                ),
                "confirmation scope amendment",
            )
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise SummaryError(
                "failed to read confirmation scope amendment "
                f"{args.confirmation_scope_amendment_json}: {exc}"
            ) from exc
    summary = build_summary(
        args.run,
        args.expected,
        confirmation_manifest=confirmation_manifest,
        confirmation_scope_amendment=confirmation_scope_amendment,
    )
    _atomic_write(Path(args.output_json), json.dumps(summary, indent=2) + "\n")
    if args.output_csv:
        rows = summary["runs"]
        fieldnames = list(rows[0])
        from io import StringIO

        buffer = StringIO()
        writer = csv.DictWriter(buffer, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        _atomic_write(Path(args.output_csv), buffer.getvalue())


if __name__ == "__main__":
    main()
