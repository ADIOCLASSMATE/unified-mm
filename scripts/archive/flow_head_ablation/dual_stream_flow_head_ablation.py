#!/usr/bin/env python3
"""Closed DF1/DF2 dynamic dual-stream flow-head screen and provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import torch
from omegaconf import DictConfig, OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from models.modeling_model.image_flow_position import (  # noqa: E402
    FLOW_HEAD_POSITION_SPECS,
    resolve_flow_head_position_config,
)
from scripts.archive.image_backbone_ablation.image_embedder_confirmation_protocol import (  # noqa: E402
    base_model_evidence,
    initial_state_evidence,
    train_data_evidence,
)

DEFAULT_BASE_CONFIG = (
    REPO_ROOT / "configs/ablation/imagenet_flow_head_position_100c_80ep.yaml"
)
STUDY_NAME = "dual_stream_flow_head"
SCREEN_SEED = 42
TRAIN_VARIANTS = ("DF1", "DF2")
ALL_VARIANTS = ("DF0", *TRAIN_VARIANTS)
POSITION_VARIANTS = ("FH0", "FH1", "FH4")
TRAIN_CELLS = tuple(
    f"{architecture}-{position}"
    for architecture in TRAIN_VARIANTS
    for position in POSITION_VARIANTS
)
PROVENANCE_FILENAME = "dual_stream_flow_head_training_provenance.json"
PROVENANCE_SCHEMA = "selfless_dual_stream_flow_head_training_provenance_v2"
SOURCE_MANIFEST_SCHEMA = "selfless_dual_stream_flow_head_source_manifest_v2"
SUMMARY_SCHEMA = "selfless_dual_stream_flow_head_ablation_summary_v2"

FLOW_HEAD_INVARIANTS = {
    "image_flow_head_arch": "contextual",
    "image_flow_depth": 8,
    "image_flow_width": 1280,
    "image_flow_mlp_ratio": 1.0,
    "image_flow_latent_mixer_heads": 8,
    "image_flow_latent_mixer_dropout": 0.0,
    "image_flow_latent_mixer_zero_init_gate": True,
    "image_flow_rope_rotate_value": False,
}
EXPECTED_FLOW_HEAD_PARAMETERS = 164_072_976
EXPECTED_DF0_IMAGE_MODULE_TENSOR_COUNT = 177
EXPECTED_DF0_IMAGE_MODULE_SCHEMA_SHA256 = (
    "fef16a6b81e8f472901564a504326cc7eee457d3772dbd44afc5042c3bae9ea9"
)
EXPECTED_DF0_INITIAL_STATE_SHA256 = (
    "2d1a03416bef06958d3f3036623cb1042ee4048741b58a836ad5b61d9b1e89b6"
)

RUNTIME_SOURCE_FILES = (
    "configs/ablation/imagenet_flow_head_position_100c_80ep.yaml",
    "docs/SELFLESS_FLOW_DUAL_STREAM_FLOW_HEAD_ABLATION_PROPOSAL.md",
    "script/ablation/train_eval_dual_stream_flow_head_ablation.sh",
    "scripts/dual_stream_flow_head_ablation.py",
    "scripts/smoke_dual_stream_flow_head_ablation.py",
    "scripts/summarize_dual_stream_flow_head_ablation.py",
    "scripts/evaluate_single_stream_fid_is.py",
    "pretrain/train_selfless_flow.py",
    "utils/dataset_imagenet_flow_cache.py",
    "utils/utils.py",
    "models/modeling_model/image_backbone.py",
    "models/modeling_model/image_flow_loss.py",
    "models/modeling_model/image_flow_position.py",
    "models/modeling_model/image_position_utils.py",
    "models/modeling_model/modeling_selfless_flow.py",
)

INITIALIZATION_CONTRACT = {
    "schema": "dual_stream_flow_head_module_keyed_initialization_v1",
    "policy": (
        "training seed plus stable module name; DF1 and DF2 have identical "
        "learned parameter names, shapes, dtypes, and initial bytes"
    ),
}
TRAIN_ORDER_CONTRACT = {
    "schema": "dual_stream_flow_head_paired_train_order_v1",
    "policy": (
        "dedicated CPU dataloader generator equals training seed; dataset split "
        "remains seed 42; stateless latent hflip uses the same seed"
    ),
}
CACHE_SCHEMA = "selfless_flow_head_content_cache_v1"
STRICT_MASK_CONTRACT = {
    "schema": "selfless_dual_stream_strict_mask_v1",
    "relation": "sigma_k < sigma_q",
    "content_reads": "strictly_earlier_content_only",
    "query_reads": "strictly_earlier_content_only",
    "query_writes_cache": False,
}
SHARED_NOISE_CONTRACT = {
    "schema": "selfless_dual_stream_shared_noise_v1",
    "source": "Qwen3ForCausalLM._shared_noisy_image_latents",
    "construction": (
        "clean_z + randn_like(clean_z) * sampled_input_noise_strength"
    ),
    "backbone_argument": "image_latents_for_model",
    "flow_content_argument": "context_image_latents",
    "tensor_identity": (
        "one sampled context_image_latents tensor is passed to the Qwen backbone "
        "and sliced for the flow-head content stream; no second content noise is sampled"
    ),
    "flow_target": "clean_image_latents",
}


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


STRICT_MASK_SHA256 = canonical_sha256(STRICT_MASK_CONTRACT)
SHARED_NOISE_SHA256 = canonical_sha256(SHARED_NOISE_CONTRACT)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    value = str(value)
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{label} must be a lowercase SHA256 digest, got {value!r}")
    return value


def normalize_variant_id(value: str, *, allow_baseline: bool = False) -> str:
    normalized = str(value).strip().upper().replace("_", "")
    allowed = ALL_VARIANTS if allow_baseline else TRAIN_VARIANTS
    if normalized not in allowed:
        raise ValueError(f"Unknown DF variant {value!r}; expected {allowed}.")
    return normalized


def normalize_position_id(value: str) -> str:
    normalized = str(value).strip().upper().replace("_", "")
    if normalized not in POSITION_VARIANTS:
        raise ValueError(
            f"Unknown position variant {value!r}; expected {POSITION_VARIANTS}."
        )
    return normalized


def cell_id(
    variant_id: str,
    position_id: str,
    *,
    allow_baseline: bool = False,
) -> str:
    variant = normalize_variant_id(
        variant_id, allow_baseline=allow_baseline
    )
    position = normalize_position_id(position_id)
    return f"{variant}-{position}"


def parse_cell_id(
    value: str,
    *,
    default_position: str | None = None,
    allow_baseline: bool = False,
) -> tuple[str, str]:
    normalized = str(value).strip().upper().replace("_", "-")
    if "-" in normalized:
        architecture, position = normalized.split("-", 1)
    elif default_position is not None:
        architecture, position = normalized, default_position
    else:
        raise ValueError(
            f"Cell ID {value!r} must include architecture and position, "
            "for example DF1-FH0."
        )
    return (
        normalize_variant_id(
            architecture, allow_baseline=allow_baseline
        ),
        normalize_position_id(position),
    )


def run_slug(
    variant_id: str,
    position_id: str = "FH0",
    seed: int = SCREEN_SEED,
) -> str:
    variant_id = normalize_variant_id(variant_id)
    position_id = normalize_position_id(position_id)
    return (
        f"selfless-flow-dual-{variant_id.lower()}-"
        f"{position_id.lower()}-s{int(seed)}"
    )


def runtime_source_evidence(repo_root: str | Path = REPO_ROOT) -> dict[str, Any]:
    root = Path(repo_root)
    entries = []
    for relative in RUNTIME_SOURCE_FILES:
        path = root / relative
        if not path.is_file():
            raise ValueError(f"Dual-stream runtime source is missing: {path}")
        entries.append(
            {
                "path": relative,
                "size_bytes": int(path.stat().st_size),
                "sha256": file_sha256(path),
            }
        )
    return {
        "schema": SOURCE_MANIFEST_SCHEMA,
        "files": entries,
        "manifest_sha256": canonical_sha256(entries),
    }


def _config_payload(config: DictConfig) -> dict[str, Any]:
    payload = OmegaConf.to_container(config, resolve=True)
    if not isinstance(payload, dict):
        raise ValueError("resolved config must be a mapping")
    payload.pop("config", None)
    experiment = payload.get("experiment")
    if isinstance(experiment, dict):
        for key in (
            "config_fingerprint",
            "dual_stream_flow_head_provenance_path",
            "dual_stream_flow_head_provenance_sha256",
        ):
            experiment.pop(key, None)
        project = str(experiment.get("project", ""))
        output_dir = Path(str(experiment.get("output_dir", "output")))
        if output_dir.name == project:
            experiment["output_dir"] = str(output_dir.parent)
    model = payload.get("model")
    if isinstance(model, dict):
        for key in (
            "mask_token_id",
            "boi_token_id",
            "eoi_token_id",
            "image_mask_token_id",
            "image_offset",
        ):
            model.pop(key, None)
    return payload


def config_fingerprint(config: DictConfig) -> str:
    return canonical_sha256(_config_payload(config))


def architecture_contract(config: DictConfig) -> dict[str, Any]:
    position, axis_dims = resolve_flow_head_position_config(config)
    if position is None or position.variant not in POSITION_VARIANTS:
        raise ValueError(
            f"DF1/DF2 position contract must be one of {POSITION_VARIANTS}."
        )
    if axis_dims is None:
        raise ValueError("DF1/DF2 require an explicit flow-head position axis split.")
    variant = normalize_variant_id(config.model.image_flow_head_variant)
    position_payload = position.as_contract(axis_dims)
    return {
        "schema": "selfless_dual_stream_flow_head_architecture_v2",
        "cell_id": cell_id(variant, position.variant),
        "variant": variant,
        "position_variant": position.variant,
        "image_backbone_variant": str(config.model.image_backbone_variant),
        "content_state": "dynamic_per_layer",
        "content_block": (
            "shared_attention_mlp" if variant == "DF1" else "shared_attention"
        ),
        "query_kv_source": "dynamic_content_hidden",
        "strict_sigma_causal": True,
        "cache_schema": CACHE_SCHEMA,
        "strict_mask_contract": dict(STRICT_MASK_CONTRACT),
        "strict_mask_sha256": STRICT_MASK_SHA256,
        "shared_noise_contract": dict(SHARED_NOISE_CONTRACT),
        "shared_noise_sha256": SHARED_NOISE_SHA256,
        "query_writes_cache": False,
        "cfg_cache_branches": 2,
        "flow_head_position": position_payload,
        "flow_head_position_sha256": canonical_sha256(position_payload),
        "flow_head_invariants": dict(FLOW_HEAD_INVARIANTS),
        "image_tokens_per_img": int(config.model.image_tokens_per_img),
        "image_latent_dim": int(config.model.image_latent_dim),
    }


def is_dual_stream_flow_head_config(config: DictConfig) -> bool:
    experiment = config.get("experiment")
    return bool(
        experiment is not None
        and str(experiment.get("ablation_study", "")).strip().lower()
        == STUDY_NAME
    )


def validate_ablation_config(
    config: DictConfig,
    expected_id: str | None = None,
    expected_position: str | None = None,
) -> None:
    if not is_dual_stream_flow_head_config(config):
        raise ValueError("experiment.ablation_study must be dual_stream_flow_head")
    config_variant = normalize_variant_id(config.model.image_flow_head_variant)
    config_position = normalize_position_id(config.model.image_flow_position_variant)
    if expected_id is None:
        variant, position = parse_cell_id(
            config.experiment.ablation_id,
            allow_baseline=False,
        )
    elif expected_position is not None:
        variant = normalize_variant_id(expected_id)
        position = normalize_position_id(expected_position)
    else:
        variant, position = parse_cell_id(
            expected_id,
            default_position=config_position,
            allow_baseline=False,
        )
    if config_variant != variant:
        raise ValueError("DF ablation ID does not match model.image_flow_head_variant")
    if config_position != position:
        raise ValueError(
            "DF ablation ID does not match model.image_flow_position_variant"
        )
    expected_cell = cell_id(variant, position)
    explicit_cell_fields = {
        "experiment.ablation_id": expected_cell,
        "experiment.architecture_id": variant,
        "experiment.position_id": position,
        "experiment.flow_head_position_variant": position,
    }
    for key, expected in explicit_cell_fields.items():
        actual = OmegaConf.select(config, key)
        if actual != expected:
            raise ValueError(
                f"DF cell requires {key}={expected!r}, got {actual!r}"
            )
    if str(config.model.image_backbone_variant) != "E2-Q1":
        raise ValueError("The DF screen is fixed to image_backbone_variant=E2-Q1")
    for key, expected in FLOW_HEAD_INVARIANTS.items():
        actual = config.model.get(key)
        if actual != expected:
            raise ValueError(
                f"All DF variants require model.{key}={expected!r}, got {actual!r}"
            )
    position_spec = FLOW_HEAD_POSITION_SPECS[position]
    position_values = {
        "model.image_flow_query_position_mode": (
            position_spec.query_position_mode
        ),
        "model.image_flow_context_position_mode": (
            position_spec.context_position_mode
        ),
        "model.image_flow_rope_mode": position_spec.rope_mode,
    }
    for key, expected in position_values.items():
        actual = OmegaConf.select(config, key)
        if actual != expected:
            raise ValueError(
                f"{expected_cell} requires {key}={expected!r}, got {actual!r}"
            )
    position_payload = position_spec.as_contract((80, 80))
    fixed_values = {
        "dataset.params.split_seed": 42,
        "evaluation.seed": 42,
        "evaluation.samples": 10_000,
        "evaluation.cfg": 3.5,
        "evaluation.parallel_rate": 1,
        "training.seed": 42,
        "training.dataloader_shuffle_seed": 42,
        "training.total_batch_size": 256,
        "training.max_train_steps": 35_920,
        "experiment.validation_flow_probe_times": [0.1, 0.5, 0.9],
        "experiment.dual_stream_cache_schema": CACHE_SCHEMA,
        "experiment.dual_stream_strict_mask_sha256": STRICT_MASK_SHA256,
        "experiment.dual_stream_shared_noise_sha256": SHARED_NOISE_SHA256,
        "experiment.dual_stream_position_contract_sha256": (
            canonical_sha256(position_payload)
        ),
    }
    for key, expected in fixed_values.items():
        actual = OmegaConf.select(config, key)
        if actual != expected:
            raise ValueError(
                f"DF protocol requires {key}={expected!r}, got {actual!r}"
            )
    if str(config.experiment.ablation_phase) != "screen":
        raise ValueError("The proposal defines only a matched seed-42 DF screen")
    expected_slug = run_slug(variant, position)
    if str(config.experiment.project) != expected_slug:
        raise ValueError(
            f"DF run slug mismatch: expected={expected_slug!r}, "
            f"actual={config.experiment.project!r}"
        )
    declared_source = _require_sha256(
        config.experiment.runtime_source_manifest_sha256,
        "DF source manifest",
    )
    current_source = runtime_source_evidence()["manifest_sha256"]
    if declared_source != current_source:
        raise ValueError(
            "DF runtime source changed after config generation: "
            f"declared={declared_source}, current={current_source}"
        )
    declared_fingerprint = _require_sha256(
        config.experiment.config_fingerprint, "DF config fingerprint"
    )
    actual_fingerprint = config_fingerprint(config)
    if declared_fingerprint != actual_fingerprint:
        raise ValueError(
            "DF resolved config fingerprint mismatch: "
            f"declared={declared_fingerprint}, actual={actual_fingerprint}"
        )
    architecture_contract(config)


def build_ablation_config(
    variant_id: str,
    position_id: str = "FH0",
    *,
    base_config: str | Path = DEFAULT_BASE_CONFIG,
) -> DictConfig:
    variant = normalize_variant_id(variant_id)
    position = normalize_position_id(position_id)
    position_spec = FLOW_HEAD_POSITION_SPECS[position]
    ablation_cell = cell_id(variant, position)
    config = OmegaConf.load(Path(base_config))
    slug = run_slug(variant, position)
    config.experiment.project = slug
    config.experiment.name = (
        f"imagenet100-dual-stream-flow-head-{variant.lower()}-"
        f"{position.lower()}-seed42-"
        "qwen3base-8gpu-b256-80ep"
    )
    config.experiment.ablation_study = STUDY_NAME
    config.experiment.ablation_id = ablation_cell
    config.experiment.architecture_id = variant
    config.experiment.position_id = position
    config.experiment.flow_head_position_variant = position
    config.experiment.ablation_phase = "screen"
    config.experiment.validation_flow_probe_times = [0.1, 0.5, 0.9]
    config.experiment.dual_stream_cache_schema = CACHE_SCHEMA
    config.experiment.dual_stream_strict_mask_sha256 = STRICT_MASK_SHA256
    config.experiment.dual_stream_shared_noise_sha256 = SHARED_NOISE_SHA256
    position_payload = position_spec.as_contract((80, 80))
    config.experiment.dual_stream_position_contract_sha256 = canonical_sha256(
        position_payload
    )
    config.training.seed = SCREEN_SEED
    config.training.dataloader_shuffle_seed = SCREEN_SEED
    config.model.image_flow_head_variant = variant
    config.model.image_flow_position_variant = position
    config.model.image_flow_query_position_mode = (
        position_spec.query_position_mode
    )
    config.model.image_flow_context_position_mode = (
        position_spec.context_position_mode
    )
    config.model.image_flow_rope_mode = position_spec.rope_mode
    config.model.image_flow_rope_axis_dims = [80, 80]
    config.model.image_flow_rope_rotate_value = False
    config.evaluation.checkpoint = f"output/{slug}/hf_model-final-ema"
    resolve_flow_head_position_config(config)
    config.experiment.runtime_source_manifest_sha256 = runtime_source_evidence()[
        "manifest_sha256"
    ]
    config.experiment.pop("config_fingerprint", None)
    config.experiment.config_fingerprint = config_fingerprint(config)
    validate_ablation_config(config, ablation_cell)
    return config


def output_run_dir(config: DictConfig) -> Path:
    base = Path(str(config.experiment.output_dir))
    project = str(config.experiment.project)
    return base if base.name == project else base / project


def provenance_path(config: DictConfig) -> Path:
    return output_run_dir(config) / PROVENANCE_FILENAME


def module_parameter_evidence(module) -> dict[str, Any]:
    schema = []
    state_digest = hashlib.sha256()
    parameter_count = 0
    for name, parameter in module.named_parameters():
        tensor = parameter.detach().cpu().contiguous()
        schema.append(
            {
                "name": name,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
            }
        )
        parameter_count += tensor.numel()
        state_digest.update(name.encode("utf-8"))
        state_digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return {
        "parameter_count": int(parameter_count),
        "parameter_schema": schema,
        "parameter_schema_sha256": canonical_sha256(schema),
        "state_sha256": state_digest.hexdigest(),
    }


def build_training_provenance(
    *,
    config: DictConfig,
    model,
    train_loader,
    special_token_ids: Mapping[str, int],
) -> dict[str, Any]:
    validate_ablation_config(config)
    source = runtime_source_evidence()
    initial = initial_state_evidence(model, special_token_ids)
    initial["contract"] = INITIALIZATION_CONTRACT
    initial["flow_head"] = module_parameter_evidence(model.image_flow_head)
    image_modules = initial["image_modules"]
    expected_df0_initial = {
        "parameter_count": EXPECTED_DF0_IMAGE_MODULE_TENSOR_COUNT,
        "parameter_schema_sha256": EXPECTED_DF0_IMAGE_MODULE_SCHEMA_SHA256,
        "state_sha256": EXPECTED_DF0_INITIAL_STATE_SHA256,
    }
    for key, expected in expected_df0_initial.items():
        if image_modules.get(key) != expected:
            raise ValueError(
                "DF initial image-module state no longer matches DF0: "
                f"{key} expected={expected!r}, actual={image_modules.get(key)!r}"
            )
    if (
        int(initial["flow_head"]["parameter_count"])
        != EXPECTED_FLOW_HEAD_PARAMETERS
    ):
        raise ValueError(
            "DF flow-head learned parameter count drifted: "
            f"expected={EXPECTED_FLOW_HEAD_PARAMETERS}, "
            f"actual={initial['flow_head']['parameter_count']}"
        )
    train_data = train_data_evidence(train_loader, config)
    train_data["contract"] = TRAIN_ORDER_CONTRACT
    payload = {
        "schema": PROVENANCE_SCHEMA,
        "ablation_id": str(config.experiment.ablation_id),
        "architecture_id": str(config.experiment.architecture_id),
        "position_id": str(config.experiment.position_id),
        "phase": str(config.experiment.ablation_phase),
        "training_seed": int(config.training.seed),
        "evaluation_seed": int(config.evaluation.seed),
        "config_fingerprint": str(config.experiment.config_fingerprint),
        "architecture": architecture_contract(config),
        "initial_state": initial,
        "train_data": train_data,
        "base_model": base_model_evidence(config),
        "runtime_source": source,
    }
    payload["provenance_sha256"] = canonical_sha256(payload)
    return payload


def write_training_provenance(
    path: str | Path, provenance: Mapping[str, Any]
) -> str:
    path = Path(path)
    payload = dict(provenance)
    stored = payload.pop("provenance_sha256", None)
    _require_sha256(stored, "DF training provenance digest")
    if canonical_sha256(payload) != stored:
        raise ValueError("refusing to write invalid DF training provenance")
    payload["provenance_sha256"] = stored
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)
    return str(stored)


def load_and_validate_training_provenance(
    path: str | Path,
    *,
    config: DictConfig | None = None,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"missing or invalid DF training provenance: {path}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise ValueError("DF training provenance must be an object")
    payload = dict(payload)
    stored = payload.pop("provenance_sha256", None)
    _require_sha256(stored, "DF training provenance digest")
    if expected_sha256 is not None and stored != expected_sha256:
        raise ValueError("DF training provenance binding digest mismatch")
    if canonical_sha256(payload) != stored:
        raise ValueError("DF training provenance content digest mismatch")
    payload["provenance_sha256"] = stored
    if payload.get("schema") != PROVENANCE_SCHEMA:
        raise ValueError("DF training provenance schema mismatch")
    if config is not None:
        validate_ablation_config(config)
        expected = {
            "ablation_id": str(config.experiment.ablation_id),
            "architecture_id": str(config.experiment.architecture_id),
            "position_id": str(config.experiment.position_id),
            "phase": str(config.experiment.ablation_phase),
            "training_seed": int(config.training.seed),
            "evaluation_seed": int(config.evaluation.seed),
            "config_fingerprint": str(config.experiment.config_fingerprint),
            "architecture": architecture_contract(config),
        }
        for key, value in expected.items():
            if payload.get(key) != value:
                raise ValueError(f"DF training provenance {key} drifted")
        flow_head = payload["initial_state"]["flow_head"]
        if int(flow_head["parameter_count"]) != EXPECTED_FLOW_HEAD_PARAMETERS:
            raise ValueError("DF flow-head parameter count provenance drifted")
    return payload


def compact_provenance(payload: Mapping[str, Any]) -> dict[str, Any]:
    initial = payload["initial_state"]["image_modules"]
    flow_head = payload["initial_state"]["flow_head"]
    return {
        "schema": payload["schema"],
        "ablation_id": payload["ablation_id"],
        "architecture_id": payload["architecture_id"],
        "position_id": payload["position_id"],
        "training_seed": int(payload["training_seed"]),
        "evaluation_seed": int(payload["evaluation_seed"]),
        "config_fingerprint": payload["config_fingerprint"],
        "architecture": payload["architecture"],
        "provenance_sha256": payload["provenance_sha256"],
        "runtime_source_manifest_sha256": payload["runtime_source"][
            "manifest_sha256"
        ],
        "initial_parameter_count": int(initial["parameter_count"]),
        "initial_parameter_schema_sha256": initial["parameter_schema_sha256"],
        "initial_parameter_state_sha256": initial["state_sha256"],
        "flow_head_parameter_count": int(flow_head["parameter_count"]),
        "flow_head_parameter_schema_sha256": flow_head[
            "parameter_schema_sha256"
        ],
        "flow_head_initial_state_sha256": flow_head["state_sha256"],
        "train_order_sha256": payload["train_data"][
            "epoch0_ordered_sample_identity_sha256"
        ],
        "augmentation_sha256": payload["train_data"][
            "epoch0_augmentation_decisions_sha256"
        ],
    }


def training_artifacts(
    config: DictConfig, *, model_path: str
) -> dict[str, Any]:
    validate_ablation_config(config)
    final_step = int(config.training.max_train_steps)
    run_dir = output_run_dir(config)
    metadata_path = run_dir / f"checkpoint-{final_step}" / "metadata.json"
    validation_path = run_dir / f"validation_metrics_step_{final_step}.json"
    runtime_metrics_path = run_dir / "training_runtime_metrics.json"
    provenance = provenance_path(config)
    hf_weights = Path(model_path) / "model.safetensors"
    required = (
        metadata_path,
        validation_path,
        runtime_metrics_path,
        provenance,
        hf_weights,
    )
    missing = [str(path) for path in required if not path.is_file() or path.stat().st_size <= 0]
    if missing:
        raise ValueError(f"Missing final DF training artifacts: {missing}")
    payload = load_and_validate_training_provenance(provenance, config=config)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected_binding = {
        "ablation_id": str(config.experiment.ablation_id),
        "phase": str(config.experiment.ablation_phase),
        "config_fingerprint": str(config.experiment.config_fingerprint),
        "runtime_source_manifest_sha256": str(
            config.experiment.runtime_source_manifest_sha256
        ),
        "provenance_path": str(provenance),
        "provenance_sha256": str(payload["provenance_sha256"]),
    }
    if metadata.get(STUDY_NAME) != expected_binding:
        raise ValueError("Final DF checkpoint provenance binding mismatch")
    hf_provenance = Path(model_path) / provenance.name
    load_and_validate_training_provenance(
        hf_provenance,
        config=config,
        expected_sha256=str(payload["provenance_sha256"]),
    )
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    runtime_metrics = json.loads(
        runtime_metrics_path.read_text(encoding="utf-8")
    )
    if (
        runtime_metrics.get("ablation_id")
        != str(config.experiment.ablation_id)
        or int(runtime_metrics.get("global_step", -1)) != final_step
        or int(runtime_metrics.get("world_size", -1)) != 8
    ):
        raise ValueError("Final DF training runtime metrics drifted")
    return {
        "schema": "selfless_dual_stream_flow_head_training_artifacts_v1",
        "final_global_step": final_step,
        "provenance": compact_provenance(payload),
        "validation_metrics_path": str(validation_path),
        "final_validation": validation,
        "training_runtime_metrics_path": str(runtime_metrics_path),
        "training_runtime": runtime_metrics,
        "artifacts": {
            "checkpoint_metadata_sha256": file_sha256(metadata_path),
            "hf_model_sha256": file_sha256(hf_weights),
            "provenance_sha256": str(payload["provenance_sha256"]),
            "training_runtime_metrics_sha256": file_sha256(
                runtime_metrics_path
            ),
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--id", dest="variant_id")
    parser.add_argument("--position", dest="position_id")
    parser.add_argument("--base-config", default=str(DEFAULT_BASE_CONFIG))
    parser.add_argument("--output")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.list:
        print(
            json.dumps(
                {
                    "train_variants": TRAIN_VARIANTS,
                    "position_variants": POSITION_VARIANTS,
                    "train_cells": TRAIN_CELLS,
                },
                indent=2,
            )
        )
        return
    if not args.variant_id or not args.position_id or not args.output:
        raise SystemExit(
            "--id, --position, and --output are required unless --list is used"
        )
    config = build_ablation_config(
        args.variant_id,
        args.position_id,
        base_config=args.base_config,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, output)
    print(output.resolve())


if __name__ == "__main__":
    main()
