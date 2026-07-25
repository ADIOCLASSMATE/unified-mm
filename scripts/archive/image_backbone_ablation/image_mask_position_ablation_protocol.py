#!/usr/bin/env python3
# Completed Q-factor provenance implementation retained for evidence audit only.
"""Strict provenance for the independent image-mask-position Q-factor study.

This protocol is intentionally separate from the historical image-embedder
confirmation protocol.  The old candidate manifest does not authorize the new
Q0 condition, so Q-factor runs carry their own study manifest, declaration,
configuration contract, source manifest, and training provenance.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
from omegaconf import DictConfig, OmegaConf

from scripts.archive.image_backbone_ablation.image_embedder_confirmation_protocol import (
    AUGMENTATION_CONTRACT,
    EVALUATOR_RNG_CONTRACT_SCHEMA,
    EVALUATOR_RNG_CONTRACT_SHA256,
    INITIALIZATION_CONTRACT,
    TRAIN_ORDER_CONTRACT,
    base_model_evidence,
    canonical_sha256,
    file_sha256,
    initial_state_evidence,
    train_data_evidence,
    write_training_provenance,
)


REPO_ROOT = Path(__file__).resolve().parents[3]

Q_FACTOR_PHASE = "mask_position_q_factor"
Q_FACTOR_STUDY_SCHEMA = "selfless_flow_image_mask_position_q_factor_study_v1"
Q_FACTOR_DECLARATION_SCHEMA = (
    "selfless_flow_image_mask_position_q_factor_declaration_v1"
)
Q_FACTOR_PROVENANCE_SCHEMA = (
    "selfless_flow_image_mask_position_q_factor_training_provenance_v1"
)
Q_FACTOR_SOURCE_SCHEMA = "selfless_flow_image_mask_position_q_factor_source_v1"
Q_FACTOR_CONFIG_CONTRACT_SCHEMA = (
    "selfless_flow_image_mask_position_q_factor_config_contract_v1"
)
Q_FACTOR_RUNTIME_CONTEXT_SCHEMA = (
    "selfless_flow_image_mask_position_q_factor_runtime_context_v1"
)
Q_FACTOR_DECISION_RULE = {
    "schema": "selfless_flow_image_mask_position_q_factor_decision_rule_v1",
    "primary_metric": "fid_mean_lower_is_better",
    "guardrail_metric": "inception_score_mean_higher_is_better",
    "preferred_simple_variant": "E2-Q0",
    "close_fid_absolute_margin": 0.5,
    "close_inception_score_absolute_margin": 1.0,
    "preference": (
        "when both margins hold versus the nominal best, prefer no stage, "
        "no observed additive position, and no mask-query additive position"
    ),
}

PARENT_SUMMARY_SCHEMA = "selfless_flow_image_embedder_ablation_summary_v3"
PARENT_PAIRING_GATE_SCHEMA = (
    "selfless_flow_image_embedder_confirmation_pairing_gate_v1"
)
PARENT_CONFIRMATION_IDS = ("E0", "E1", "E2b", "E2", "E4b", "E4")
Q_FACTOR_SEEDS = frozenset((43, 44, 45))


@dataclass(frozen=True)
class QFactorVariant:
    parent_ablation_id: str
    observed_position_mode: str
    mask_position_mode: str
    query_stage_mode: str = "none"
    rope_mode: str = "row_col_2d"
    space_to_depth_factor: int = 1

    @property
    def parent(self) -> str:
        return self.parent_ablation_id

    @property
    def observed(self) -> str:
        return self.observed_position_mode

    @property
    def mask(self) -> str:
        return self.mask_position_mode


Q_FACTOR_VARIANTS: dict[str, QFactorVariant] = {
    "E2b-Q1": QFactorVariant("E2b", "additive_2d", "additive_2d"),
    "E2b-Q0": QFactorVariant("E2b", "additive_2d", "none"),
    "E2-Q1": QFactorVariant("E2", "none", "additive_2d"),
    "E2-Q0": QFactorVariant("E2", "none", "none"),
}
Q_FACTOR_IDS = tuple(Q_FACTOR_VARIANTS)

# This list is deliberately independent of the historical confirmation source
# list.  In particular it includes both the Q-factor config builder and the
# evaluator implementation that consumes the resulting checkpoint.
Q_FACTOR_RUNTIME_SOURCE_FILES = (
    "configs/ablation/imagenet_flow_image_embedder_100c_80ep.yaml",
    "accelerate_configs/8_gpus_deepspeed_zero2.yaml",
    "script/offline_env.sh",
    "script/ablation/pretraining_imagenet_flow_100c_80ep.sh",
    "script/ablation/evaluate_imagenet_flow_100c.sh",
    "script/ablation/train_image_mask_position_ablation.sh",
    "script/ablation/evaluate_image_mask_position_ablation.sh",
    "scripts/image_embedder_ablation_matrix.py",
    "scripts/image_mask_position_ablation_matrix.py",
    "scripts/image_mask_position_ablation_protocol.py",
    "scripts/summarize_image_mask_position_ablation.py",
    "scripts/evaluate_single_stream_fid_is.py",
    "scripts/evaluate_qwen_showo_fid_is.py",
    "scripts/generate_flow_validation_images.py",
    "pretrain/train_selfless_flow.py",
    "utils/dataset_utils.py",
    "utils/dataset_imagenet_flow_cache.py",
    "utils/utils.py",
    "models/modeling_model/modeling_selfless_flow.py",
    "models/modeling_model/image_flow_loss.py",
    "models/modeling_model/image_position_utils.py",
    "models/modeling_model/image_latent_layout.py",
)

_DYNAMIC_CONFIG_PATHS = (
    # ``get_config`` accepts the launcher's ``config=path.yaml`` selector via
    # OmegaConf CLI merging.  The selector identifies the resolved config but
    # is not itself part of the experiment configuration.
    ("config",),
    ("experiment", "output_dir"),
    ("experiment", "q_factor_protocol"),
    ("experiment", "q_factor_provenance_path"),
    ("experiment", "q_factor_provenance_sha256"),
    ("model", "mask_token_id"),
    ("model", "boi_token_id"),
    ("model", "eoi_token_id"),
    ("model", "image_mask_token_id"),
    ("model", "image_offset"),
)


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return dict(value)


def _require_sha256(value: Any, label: str) -> str:
    value = str(value)
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{label} must be a lowercase SHA256 digest, got {value!r}")
    return value


def _plain(value: Any) -> Any:
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
    # A canonical JSON round trip both deep-copies the value and rejects
    # objects that cannot be represented in a declaration.
    return json.loads(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    )


def normalize_q_factor_id(variant_id: str) -> str:
    normalized = str(variant_id).strip().lower()
    for candidate in Q_FACTOR_IDS:
        if candidate.lower() == normalized:
            return candidate
    raise ValueError(
        f"Unknown Q-factor ID {variant_id!r}; expected one of {', '.join(Q_FACTOR_IDS)}"
    )


def q_factor_run_slug(variant_id: str, seed: int) -> str:
    variant_id = normalize_q_factor_id(variant_id)
    return f"selfless-flow-image-embedder-qf-{variant_id.lower()}-seed{int(seed)}"


def is_q_factor_config(config: DictConfig | Mapping[str, Any]) -> bool:
    experiment = config.get("experiment", None)
    if experiment is None:
        return False
    return bool(
        str(experiment.get("ablation_phase", "")).strip() == Q_FACTOR_PHASE
        or experiment.get("q_factor_protocol", None) is not None
    )


def _remove_nested(payload: dict[str, Any], path: tuple[str, ...]) -> None:
    target: Any = payload
    for key in path[:-1]:
        if not isinstance(target, dict) or key not in target:
            return
        target = target[key]
    if isinstance(target, dict):
        target.pop(path[-1], None)


def q_factor_config_contract(config: DictConfig | Mapping[str, Any]) -> dict[str, Any]:
    """Return the stable resolved-config projection bound by a declaration.

    The launcher-only config selector, runtime-populated token IDs, expanded
    output directory, and provenance fields are excluded.  Architecture, data,
    optimization, seed, run identity, and evaluation settings remain covered.
    """

    payload = _plain(config)
    if not isinstance(payload, dict):
        raise ValueError("Q-factor config must be an object")
    for path in _DYNAMIC_CONFIG_PATHS:
        _remove_nested(payload, path)
    return {
        "schema": Q_FACTOR_CONFIG_CONTRACT_SCHEMA,
        "resolved_config": payload,
    }


def runtime_source_evidence(
    repo_root: str | Path = REPO_ROOT,
    *,
    source_files: Iterable[str] = Q_FACTOR_RUNTIME_SOURCE_FILES,
) -> dict[str, Any]:
    root = Path(repo_root)
    required_files = tuple(str(value) for value in source_files)
    if len(required_files) != len(set(required_files)):
        raise ValueError("Q-factor runtime source list contains duplicates")
    entries = []
    for relative in required_files:
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"Q-factor runtime source path must be repo-relative: {relative!r}")
        path = root / relative_path
        if not path.is_file():
            raise ValueError(f"Q-factor runtime source is missing: {path}")
        entries.append(
            {
                "path": relative,
                "size_bytes": int(path.stat().st_size),
                "sha256": file_sha256(path),
            }
        )
    return {
        "schema": Q_FACTOR_SOURCE_SCHEMA,
        "required_files": list(required_files),
        "files": entries,
        "manifest_sha256": canonical_sha256(entries),
    }


def q_factor_runtime_source_evidence(
    repo_root: str | Path = REPO_ROOT,
    *,
    source_files: Iterable[str] = Q_FACTOR_RUNTIME_SOURCE_FILES,
) -> dict[str, Any]:
    """Public, study-specific name for the Q-factor source manifest builder."""

    return runtime_source_evidence(repo_root, source_files=source_files)


def validate_runtime_source_evidence(
    evidence: Mapping[str, Any],
    *,
    repo_root: str | Path = REPO_ROOT,
    source_files: Iterable[str] = Q_FACTOR_RUNTIME_SOURCE_FILES,
    validate_current: bool = True,
) -> dict[str, Any]:
    evidence = _require_mapping(evidence, "Q-factor runtime source evidence")
    expected_files = tuple(str(value) for value in source_files)
    if evidence.get("schema") != Q_FACTOR_SOURCE_SCHEMA:
        raise ValueError("Q-factor runtime source schema mismatch")
    if evidence.get("required_files") != list(expected_files):
        raise ValueError("Q-factor runtime source required-file list drifted")
    entries = evidence.get("files")
    if (
        not isinstance(entries, list)
        or any(not isinstance(item, Mapping) for item in entries)
        or [item.get("path") for item in entries] != list(expected_files)
    ):
        raise ValueError("Q-factor runtime source file entries drifted")
    for entry in entries:
        entry = _require_mapping(entry, "Q-factor runtime source entry")
        if not isinstance(entry.get("size_bytes"), int) or entry["size_bytes"] < 0:
            raise ValueError("Q-factor runtime source size must be a nonnegative integer")
        _require_sha256(entry.get("sha256"), "Q-factor runtime source file digest")
    manifest_sha256 = _require_sha256(
        evidence.get("manifest_sha256"), "Q-factor runtime source manifest digest"
    )
    if canonical_sha256(entries) != manifest_sha256:
        raise ValueError("Q-factor runtime source manifest digest mismatch")
    if validate_current:
        current = runtime_source_evidence(repo_root, source_files=expected_files)
        if current != evidence:
            raise ValueError("Q-factor runtime source changed after preregistration")
    return evidence


def _validate_parent_summary_payload(summary: Mapping[str, Any]) -> dict[str, Any]:
    summary = _require_mapping(summary, "Q-factor parent summary")
    if summary.get("schema") != PARENT_SUMMARY_SCHEMA:
        raise ValueError("Q-factor parent summary schema mismatch")
    if summary.get("expected") != "confirmation":
        raise ValueError("Q-factor parent must be a strict confirmation summary")

    expected_pairs = {
        (variant_id, seed)
        for variant_id in PARENT_CONFIRMATION_IDS
        for seed in Q_FACTOR_SEEDS
    }
    runs = summary.get("runs")
    if not isinstance(runs, list):
        raise ValueError("Q-factor parent summary is missing runs")
    run_pairs = {
        (item.get("id"), item.get("training_seed"))
        for item in runs
        if isinstance(item, Mapping)
    }
    if len(runs) != len(expected_pairs) or run_pairs != expected_pairs:
        raise ValueError("Q-factor parent summary does not contain the exact 6x3 confirmation")

    aggregates = summary.get("aggregates")
    if not isinstance(aggregates, list):
        raise ValueError("Q-factor parent summary is missing aggregates")
    aggregate_ids = [item.get("id") for item in aggregates if isinstance(item, Mapping)]
    if len(aggregate_ids) != len(PARENT_CONFIRMATION_IDS) or set(aggregate_ids) != set(
        PARENT_CONFIRMATION_IDS
    ):
        raise ValueError("Q-factor parent summary aggregate IDs drifted")

    scope = _require_mapping(
        summary.get("confirmation_scope_manifest"),
        "Q-factor parent confirmation scope",
    )
    if scope.get("candidate_ids") != list(PARENT_CONFIRMATION_IDS):
        raise ValueError("Q-factor parent confirmation scope is not the frozen factor-1 set")
    if scope.get("required_space_to_depth_factor") != 1:
        raise ValueError("Q-factor parent confirmation scope must require factor 1")

    pairing_gate = _require_mapping(
        summary.get("confirmation_pairing_gate"), "Q-factor parent pairing gate"
    )
    if pairing_gate.get("schema") != PARENT_PAIRING_GATE_SCHEMA:
        raise ValueError("Q-factor parent pairing-gate schema mismatch")
    if pairing_gate.get("validated_runs") != len(expected_pairs):
        raise ValueError("Q-factor parent pairing gate did not validate all 18 runs")
    _require_sha256(
        pairing_gate.get("evaluator_rng_contract_sha256"),
        "Q-factor parent evaluator RNG contract digest",
    )
    return {
        "schema": PARENT_SUMMARY_SCHEMA,
        "expected": "confirmation",
        "candidate_ids": list(PARENT_CONFIRMATION_IDS),
        "training_seeds": sorted(Q_FACTOR_SEEDS),
        "validated_runs": len(expected_pairs),
        "pairing_gate_sha256": canonical_sha256(pairing_gate),
    }


def load_parent_summary_evidence(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    try:
        raw = path.read_bytes()
        summary = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"failed to read Q-factor parent summary {path}: {exc}") from exc
    validated = _validate_parent_summary_payload(summary)
    return {
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        **validated,
    }


def _resolve_bound_path(path: str, repo_root: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else Path(repo_root) / value


def build_q_factor_study_manifest(
    parent_summary_path: str | Path,
    *,
    repo_root: str | Path = REPO_ROOT,
    source_files: Iterable[str] = Q_FACTOR_RUNTIME_SOURCE_FILES,
) -> dict[str, Any]:
    source_files = tuple(source_files)
    parent_path = _resolve_bound_path(str(parent_summary_path), repo_root)
    parent = load_parent_summary_evidence(parent_path)
    parent["path"] = str(parent_summary_path)
    source = runtime_source_evidence(repo_root, source_files=source_files)
    manifest = {
        "schema": Q_FACTOR_STUDY_SCHEMA,
        "phase": Q_FACTOR_PHASE,
        "q_factor_ids": list(Q_FACTOR_IDS),
        "training_seeds": sorted(Q_FACTOR_SEEDS),
        "evaluation_seed": 42,
        "matrix": {
            variant_id: asdict(variant)
            for variant_id, variant in Q_FACTOR_VARIANTS.items()
        },
        "parent_summary": parent,
        "runtime_source": source,
        "evaluator_rng_contract_schema": EVALUATOR_RNG_CONTRACT_SCHEMA,
        "evaluator_rng_contract_sha256": EVALUATOR_RNG_CONTRACT_SHA256,
        "initialization_contract": INITIALIZATION_CONTRACT,
        "train_order_contract": TRAIN_ORDER_CONTRACT,
        "augmentation_contract": AUGMENTATION_CONTRACT,
        "decision_rule": Q_FACTOR_DECISION_RULE,
    }
    manifest["study_manifest_sha256"] = canonical_sha256(manifest)
    return manifest


def validate_q_factor_study_manifest(
    manifest: Mapping[str, Any],
    *,
    repo_root: str | Path = REPO_ROOT,
    source_files: Iterable[str] = Q_FACTOR_RUNTIME_SOURCE_FILES,
    validate_parent_summary: bool = True,
    validate_runtime_source: bool = True,
) -> dict[str, Any]:
    manifest = _require_mapping(manifest, "Q-factor study manifest")
    stored_digest = manifest.pop("study_manifest_sha256", None)
    _require_sha256(stored_digest, "Q-factor study manifest digest")
    if canonical_sha256(manifest) != stored_digest:
        raise ValueError("Q-factor study manifest digest mismatch")
    manifest["study_manifest_sha256"] = stored_digest

    expected = {
        "schema": Q_FACTOR_STUDY_SCHEMA,
        "phase": Q_FACTOR_PHASE,
        "q_factor_ids": list(Q_FACTOR_IDS),
        "training_seeds": sorted(Q_FACTOR_SEEDS),
        "evaluation_seed": 42,
        "matrix": {
            variant_id: asdict(variant)
            for variant_id, variant in Q_FACTOR_VARIANTS.items()
        },
        "evaluator_rng_contract_schema": EVALUATOR_RNG_CONTRACT_SCHEMA,
        "evaluator_rng_contract_sha256": EVALUATOR_RNG_CONTRACT_SHA256,
        "initialization_contract": INITIALIZATION_CONTRACT,
        "train_order_contract": TRAIN_ORDER_CONTRACT,
        "augmentation_contract": AUGMENTATION_CONTRACT,
        "decision_rule": Q_FACTOR_DECISION_RULE,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"Q-factor study manifest {key} drifted")

    parent = _require_mapping(manifest.get("parent_summary"), "Q-factor parent binding")
    _require_sha256(parent.get("sha256"), "Q-factor parent summary digest")
    _require_sha256(parent.get("pairing_gate_sha256"), "Q-factor parent pairing digest")
    if validate_parent_summary:
        current = load_parent_summary_evidence(
            _resolve_bound_path(str(parent.get("path", "")), repo_root)
        )
        current["path"] = str(parent.get("path", ""))
        if current != parent:
            raise ValueError("Q-factor parent summary changed after preregistration")

    source = validate_runtime_source_evidence(
        _require_mapping(manifest.get("runtime_source"), "Q-factor runtime source"),
        repo_root=repo_root,
        source_files=source_files,
        validate_current=validate_runtime_source,
    )
    if source["manifest_sha256"] != manifest["runtime_source"]["manifest_sha256"]:
        raise ValueError("Q-factor runtime source binding mismatch")
    return manifest


def build_q_factor_declaration(
    *,
    variant_id: str,
    seed: int,
    config_contract: Mapping[str, Any],
    study_manifest: Mapping[str, Any] | None = None,
    parent_summary_path: str | Path | None = None,
    repo_root: str | Path = REPO_ROOT,
    source_files: Iterable[str] = Q_FACTOR_RUNTIME_SOURCE_FILES,
) -> dict[str, Any]:
    variant_id = normalize_q_factor_id(variant_id)
    seed = int(seed)
    if seed not in Q_FACTOR_SEEDS:
        raise ValueError(
            f"Q-factor training seed must be one of {sorted(Q_FACTOR_SEEDS)}, got {seed}"
        )
    source_files = tuple(source_files)
    if study_manifest is None:
        if parent_summary_path is None:
            raise ValueError("parent_summary_path is required without a study_manifest")
        study_manifest = build_q_factor_study_manifest(
            parent_summary_path,
            repo_root=repo_root,
            source_files=source_files,
        )
    study = validate_q_factor_study_manifest(
        study_manifest,
        repo_root=repo_root,
        source_files=source_files,
    )
    contract = _plain(config_contract)
    if not isinstance(contract, dict) or contract.get("schema") != Q_FACTOR_CONFIG_CONTRACT_SCHEMA:
        raise ValueError("Q-factor config contract schema mismatch")
    variant = Q_FACTOR_VARIANTS[variant_id]
    declaration = {
        "schema": Q_FACTOR_DECLARATION_SCHEMA,
        "phase": Q_FACTOR_PHASE,
        "q_factor_id": variant_id,
        "parent_ablation_id": variant.parent_ablation_id,
        "training_seed": seed,
        "dataloader_shuffle_seed": seed,
        "evaluation_seed": 42,
        "run_slug": q_factor_run_slug(variant_id, seed),
        "architecture": asdict(variant),
        "config_contract": contract,
        "config_contract_sha256": canonical_sha256(contract),
        "study_manifest": study,
        "study_manifest_sha256": study["study_manifest_sha256"],
        "parent_summary_sha256": study["parent_summary"]["sha256"],
        "runtime_source_manifest_sha256": study["runtime_source"][
            "manifest_sha256"
        ],
        "source_manifest_sha256": study["runtime_source"]["manifest_sha256"],
        "evaluator_rng_contract_schema": EVALUATOR_RNG_CONTRACT_SCHEMA,
        "evaluator_rng_contract_sha256": EVALUATOR_RNG_CONTRACT_SHA256,
        "initialization_contract": INITIALIZATION_CONTRACT,
        "train_order_contract": TRAIN_ORDER_CONTRACT,
        "augmentation_contract": AUGMENTATION_CONTRACT,
    }
    declaration["declaration_sha256"] = canonical_sha256(declaration)
    return declaration


def validate_q_factor_declaration(
    declaration: Mapping[str, Any],
    *,
    variant_id: str,
    seed: int,
    config_contract: Mapping[str, Any] | None = None,
    repo_root: str | Path = REPO_ROOT,
    source_files: Iterable[str] = Q_FACTOR_RUNTIME_SOURCE_FILES,
    validate_parent_summary: bool = True,
    validate_runtime_source: bool = True,
) -> dict[str, Any]:
    variant_id = normalize_q_factor_id(variant_id)
    seed = int(seed)
    declaration = _require_mapping(declaration, "Q-factor declaration")
    stored_digest = declaration.pop("declaration_sha256", None)
    _require_sha256(stored_digest, "Q-factor declaration digest")
    if canonical_sha256(declaration) != stored_digest:
        raise ValueError("Q-factor declaration digest mismatch")
    declaration["declaration_sha256"] = stored_digest

    variant = Q_FACTOR_VARIANTS[variant_id]
    expected = {
        "schema": Q_FACTOR_DECLARATION_SCHEMA,
        "phase": Q_FACTOR_PHASE,
        "q_factor_id": variant_id,
        "parent_ablation_id": variant.parent_ablation_id,
        "training_seed": seed,
        "dataloader_shuffle_seed": seed,
        "evaluation_seed": 42,
        "run_slug": q_factor_run_slug(variant_id, seed),
        "architecture": asdict(variant),
        "evaluator_rng_contract_schema": EVALUATOR_RNG_CONTRACT_SCHEMA,
        "evaluator_rng_contract_sha256": EVALUATOR_RNG_CONTRACT_SHA256,
        "initialization_contract": INITIALIZATION_CONTRACT,
        "train_order_contract": TRAIN_ORDER_CONTRACT,
        "augmentation_contract": AUGMENTATION_CONTRACT,
    }
    for key, value in expected.items():
        if declaration.get(key) != value:
            raise ValueError(f"Q-factor declaration {key} drifted")

    contract = _require_mapping(
        declaration.get("config_contract"), "Q-factor declaration config contract"
    )
    if contract.get("schema") != Q_FACTOR_CONFIG_CONTRACT_SCHEMA:
        raise ValueError("Q-factor config contract schema mismatch")
    contract_digest = _require_sha256(
        declaration.get("config_contract_sha256"), "Q-factor config contract digest"
    )
    if canonical_sha256(contract) != contract_digest:
        raise ValueError("Q-factor config contract digest mismatch")
    if config_contract is not None and _plain(config_contract) != contract:
        raise ValueError("Q-factor resolved config differs from its declaration")

    study = validate_q_factor_study_manifest(
        _require_mapping(declaration.get("study_manifest"), "Q-factor study manifest"),
        repo_root=repo_root,
        source_files=source_files,
        validate_parent_summary=validate_parent_summary,
        validate_runtime_source=validate_runtime_source,
    )
    for key, expected_value in {
        "study_manifest_sha256": study["study_manifest_sha256"],
        "parent_summary_sha256": study["parent_summary"]["sha256"],
        "runtime_source_manifest_sha256": study["runtime_source"][
            "manifest_sha256"
        ],
        "source_manifest_sha256": study["runtime_source"]["manifest_sha256"],
    }.items():
        if declaration.get(key) != expected_value:
            raise ValueError(f"Q-factor declaration {key} binding mismatch")
    return declaration


def runtime_context_evidence(config: DictConfig | Mapping[str, Any], accelerator=None):
    world_size = int(
        getattr(accelerator, "num_processes", os.environ.get("WORLD_SIZE", 1))
    )
    gradient_accumulation_steps = getattr(
        accelerator, "gradient_accumulation_steps", None
    )
    distributed_type = getattr(accelerator, "distributed_type", None)
    distributed_type = (
        str(distributed_type).split(".")[-1] if distributed_type is not None else None
    )
    zero_stage = None
    state = getattr(accelerator, "state", None)
    plugin = getattr(state, "deepspeed_plugin", None)
    if plugin is not None:
        zero_stage = plugin.deepspeed_config.get("zero_optimization", {}).get(
            "stage",
            plugin.deepspeed_config.get("zero_stage"),
        )
    training = config.get("training", {})
    payload = {
        "schema": Q_FACTOR_RUNTIME_CONTEXT_SCHEMA,
        "world_size": world_size,
        "distributed_type": distributed_type,
        "gradient_accumulation_steps": (
            int(gradient_accumulation_steps)
            if gradient_accumulation_steps is not None
            else None
        ),
        "mixed_precision": str(training.get("mixed_precision", "")),
        "torch_version": str(torch.__version__),
        "cuda_version": str(torch.version.cuda),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_name": (
            torch.cuda.get_device_name(torch.cuda.current_device())
            if torch.cuda.is_available()
            else None
        ),
        "deepspeed_zero_stage": int(zero_stage) if zero_stage is not None else None,
    }
    payload["runtime_context_sha256"] = canonical_sha256(payload)
    return payload


def _validate_runtime_context(value: Mapping[str, Any]) -> dict[str, Any]:
    value = _require_mapping(value, "Q-factor runtime context")
    stored = value.pop("runtime_context_sha256", None)
    _require_sha256(stored, "Q-factor runtime context digest")
    if canonical_sha256(value) != stored:
        raise ValueError("Q-factor runtime context digest mismatch")
    value["runtime_context_sha256"] = stored
    if value.get("schema") != Q_FACTOR_RUNTIME_CONTEXT_SCHEMA:
        raise ValueError("Q-factor runtime context schema mismatch")
    return value


def validate_q_factor_formal_runtime_context(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    value = _validate_runtime_context(value)
    expected = {
        "world_size": 8,
        "distributed_type": "DEEPSPEED",
        "gradient_accumulation_steps": 1,
        "mixed_precision": "bf16",
        "deepspeed_zero_stage": 2,
        "cuda_available": True,
    }
    mismatches = [
        f"{key}={value.get(key)!r} (expected {expected_value!r})"
        for key, expected_value in expected.items()
        if value.get(key) != expected_value
    ]
    if "H100" not in str(value.get("cuda_device_name", "")):
        mismatches.append(
            f"cuda_device_name={value.get('cuda_device_name')!r} (expected H100)"
        )
    if mismatches:
        raise ValueError(
            "Q-factor formal runtime drifted: " + "; ".join(mismatches)
        )
    return value


def build_q_factor_training_provenance(
    *,
    config: DictConfig,
    model,
    train_loader,
    special_token_ids: Mapping[str, int],
    accelerator=None,
    repo_root: str | Path = REPO_ROOT,
    source_files: Iterable[str] = Q_FACTOR_RUNTIME_SOURCE_FILES,
) -> dict[str, Any]:
    if not is_q_factor_config(config):
        raise ValueError("Q-factor training provenance requires a Q-factor config")
    variant_id = normalize_q_factor_id(str(config.experiment.ablation_id))
    seed = int(config.training.seed)
    contract = q_factor_config_contract(config)
    declaration = validate_q_factor_declaration(
        OmegaConf.to_container(config.experiment.q_factor_protocol, resolve=True),
        variant_id=variant_id,
        seed=seed,
        config_contract=contract,
        repo_root=repo_root,
        source_files=source_files,
    )
    source = runtime_source_evidence(repo_root, source_files=source_files)
    runtime = runtime_context_evidence(config, accelerator)
    if bool(config.experiment.get("q_factor_formal", False)):
        validate_q_factor_formal_runtime_context(runtime)
    variant = Q_FACTOR_VARIANTS[variant_id]
    provenance = {
        "schema": Q_FACTOR_PROVENANCE_SCHEMA,
        "phase": Q_FACTOR_PHASE,
        "q_factor_id": variant_id,
        "parent_ablation_id": variant.parent_ablation_id,
        "training_seed": seed,
        "architecture": asdict(variant),
        "q_factor_declaration": declaration,
        "q_factor_declaration_sha256": declaration["declaration_sha256"],
        "study_manifest_sha256": declaration["study_manifest_sha256"],
        "parent_summary_sha256": declaration["parent_summary_sha256"],
        "config_contract_sha256": declaration["config_contract_sha256"],
        "runtime_source_manifest_sha256": source["manifest_sha256"],
        "source_manifest_sha256": source["manifest_sha256"],
        "initial_state": initial_state_evidence(model, special_token_ids),
        "train_data": train_data_evidence(train_loader, config),
        "base_model": base_model_evidence(config),
        "runtime_source": source,
        "runtime_context": runtime,
    }
    provenance["provenance_sha256"] = canonical_sha256(provenance)
    return provenance


def _validate_q_factor_training_provenance(
    provenance: Mapping[str, Any],
    *,
    expected_sha256: str,
    variant_id: str,
    seed: int,
    config: DictConfig | Mapping[str, Any] | None,
    repo_root: str | Path,
    source_files: Iterable[str],
    validate_parent_summary: bool,
    validate_runtime_source: bool,
) -> dict[str, Any]:
    variant_id = normalize_q_factor_id(variant_id)
    seed = int(seed)
    payload = _require_mapping(provenance, "Q-factor training provenance")
    stored_digest = payload.pop("provenance_sha256", None)
    _require_sha256(stored_digest, "Q-factor training provenance digest")
    if stored_digest != expected_sha256 or canonical_sha256(payload) != stored_digest:
        raise ValueError("Q-factor training provenance digest mismatch")
    payload["provenance_sha256"] = stored_digest
    variant = Q_FACTOR_VARIANTS[variant_id]
    expected = {
        "schema": Q_FACTOR_PROVENANCE_SCHEMA,
        "phase": Q_FACTOR_PHASE,
        "q_factor_id": variant_id,
        "parent_ablation_id": variant.parent_ablation_id,
        "training_seed": seed,
        "architecture": asdict(variant),
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"Q-factor training provenance {key} drifted")

    contract = q_factor_config_contract(config) if config is not None else None
    declaration = validate_q_factor_declaration(
        _require_mapping(
            payload.get("q_factor_declaration"),
            "Q-factor provenance declaration",
        ),
        variant_id=variant_id,
        seed=seed,
        config_contract=contract,
        repo_root=repo_root,
        source_files=source_files,
        validate_parent_summary=validate_parent_summary,
        validate_runtime_source=validate_runtime_source,
    )
    bindings = {
        "q_factor_declaration_sha256": declaration["declaration_sha256"],
        "study_manifest_sha256": declaration["study_manifest_sha256"],
        "parent_summary_sha256": declaration["parent_summary_sha256"],
        "config_contract_sha256": declaration["config_contract_sha256"],
        "runtime_source_manifest_sha256": declaration[
            "runtime_source_manifest_sha256"
        ],
        "source_manifest_sha256": declaration["source_manifest_sha256"],
    }
    for key, value in bindings.items():
        if payload.get(key) != value:
            raise ValueError(f"Q-factor training provenance {key} binding mismatch")

    initial_state = _require_mapping(payload.get("initial_state"), "initial state")
    if initial_state.get("contract") != INITIALIZATION_CONTRACT:
        raise ValueError("Q-factor initialization contract drifted")
    train_data = _require_mapping(payload.get("train_data"), "train data")
    if train_data.get("contract") != TRAIN_ORDER_CONTRACT:
        raise ValueError("Q-factor train-order contract drifted")
    if train_data.get("augmentation_contract") != AUGMENTATION_CONTRACT:
        raise ValueError("Q-factor augmentation contract drifted")
    if train_data.get("dataloader_shuffle_seed") != seed:
        raise ValueError("Q-factor dataloader shuffle seed drifted")
    if train_data.get("augmentation_seed") != seed:
        raise ValueError("Q-factor augmentation seed drifted")

    base_model = _require_mapping(payload.get("base_model"), "base model evidence")
    _require_sha256(base_model.get("manifest_sha256"), "base model manifest digest")
    source = validate_runtime_source_evidence(
        _require_mapping(payload.get("runtime_source"), "runtime source evidence"),
        repo_root=repo_root,
        source_files=source_files,
        validate_current=validate_runtime_source,
    )
    if source["manifest_sha256"] != declaration["runtime_source_manifest_sha256"]:
        raise ValueError("Q-factor provenance runtime source differs from declaration")
    runtime_context = _require_mapping(
        payload.get("runtime_context"), "runtime context"
    )
    if config is not None and bool(
        config.get("experiment", {}).get("q_factor_formal", False)
    ):
        validate_q_factor_formal_runtime_context(runtime_context)
    else:
        _validate_runtime_context(runtime_context)
    return payload


def write_q_factor_training_provenance(
    path: str | Path, provenance: Mapping[str, Any]
) -> str:
    """Atomically write a digest-bound Q-factor provenance document."""

    # The shared writer is schema-agnostic and already implements atomic replace.
    return write_training_provenance(path, provenance)


def load_and_validate_q_factor_training_provenance(
    path: str | Path,
    *,
    expected_sha256: str,
    variant_id: str,
    seed: int,
    config: DictConfig | Mapping[str, Any] | None = None,
    repo_root: str | Path = REPO_ROOT,
    source_files: Iterable[str] = Q_FACTOR_RUNTIME_SOURCE_FILES,
    validate_parent_summary: bool = True,
    validate_runtime_source: bool = True,
) -> dict[str, Any]:
    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"missing or invalid Q-factor training provenance: {path}") from exc
    return _validate_q_factor_training_provenance(
        payload,
        expected_sha256=expected_sha256,
        variant_id=variant_id,
        seed=seed,
        config=config,
        repo_root=repo_root,
        source_files=tuple(source_files),
        validate_parent_summary=validate_parent_summary,
        validate_runtime_source=validate_runtime_source,
    )
