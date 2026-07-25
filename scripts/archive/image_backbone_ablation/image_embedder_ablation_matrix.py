#!/usr/bin/env python3
# Historical executable retained for evidence audit, not future experiment launch.
"""Prepare and validate resolved configs for the image-embedder ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from scripts.archive.image_backbone_ablation.image_latent_layout_legacy import (  # noqa: E402
    resolve_image_layout_config,
)
from scripts.archive.image_backbone_ablation.image_embedder_confirmation_protocol import (  # noqa: E402
    CONFIRMATION_SEEDS,
    build_confirmation_declaration,
    file_sha256,
    is_confirmation_config,
    validate_confirmation_declaration,
)


DEFAULT_BASE_CONFIG = REPO_ROOT / "configs/ablation/imagenet_flow_image_embedder_100c_80ep.yaml"
FLOW_HEAD_INVARIANTS = {
    "image_flow_head_arch": "contextual",
    "image_flow_depth": 8,
    "image_flow_width": 1280,
    "image_flow_mlp_ratio": 1.0,
    "image_flow_latent_mixer_heads": 8,
    "image_flow_latent_mixer_dropout": 0.0,
    "image_flow_latent_mixer_zero_init_gate": True,
}
TRAINING_PROTOCOL_SCHEMA = "selfless_flow_image_embedder_training_v1"
STAGE_BUFFER_IMPLEMENTATION_CONTRACT = "fixed_sincos_nonpersistent_rebuild_v1"
TRAINING_PROTOCOL_INVARIANTS = {
    "model.model_path": "public/models/Qwen/Qwen3-0.6B-Base",
    "model.attention_pattern": "ar",
    "model.loss_attention_pattern": "likelihood",
    "model.use_flex_attention": True,
    "model.lambda_text": 0.0,
    "model.lambda_image": 1.0,
    "model.continuous_image_latents": True,
    "model.image_token_embedder_init_mode": "balanced",
    "model.image_token_embedder_latent_rms": 1.0,
    "model.image_generation_head_type": "flow",
    "model.image_flow_num_sampling_steps": "100",
    "model.image_flow_batch_mul": 4,
    "model.image_flow_time_scale": 1000.0,
    "model.image_flow_time_sampling": "logit_normal",
    "model.image_flow_logit_mean": 0.0,
    "model.image_flow_logit_std": 1.0,
    "model.image_flow_time_eps": 1.0e-5,
    "model.image_flow_time_uniform_mix": 0.1,
    "model.image_flow_solver": "heun",
    "model.image_input_noise_strength": 0.01,
    "model.image_input_noise_strength_std": 0.0,
    "model.image_input_noise_strength_min": None,
    "model.image_input_noise_strength_max": None,
    "model.image_uncond_prob": 0.1,
    "model.reinitialize_image_modules": True,
    "model.pretrained_image_flow_adapter": "none",
    "dataset.class_name": "ImageNetFlowCacheDataset",
    "dataset.params.cache_path": (
        "public/datasets/imagenet_ablation_100c_balanced/"
        "vae_latents_mar_kl16/flow_latents_100c_1250pc_fp16.pt"
    ),
    "dataset.params.manifest_jsonl": (
        "public/datasets/imagenet_ablation_100c_balanced/manifest.jsonl"
    ),
    "dataset.params.split_manifest_jsonl": (
        "public/datasets/imagenet_ablation_100c_balanced/"
        "split_seed42_val100.jsonl"
    ),
    "dataset.params.synset_mapping_path": "public/datasets/imagenet/LOC_synset_mapping.txt",
    "dataset.params.num_classes": 100,
    "dataset.params.conditioning_mode": "class_image",
    "dataset.params.label_text": False,
    "dataset.params.val_samples_per_class": 100,
    "dataset.params.split_strategy": "stratified",
    "dataset.params.split_seed": 42,
    "dataset.params.latent_hflip_prob": 0.5,
    "optimizer.name": "adamw",
    "optimizer.params.learning_rate": 1.0e-4,
    "optimizer.params.backbone_learning_rate": 2.0e-5,
    "optimizer.params.projector_learning_rate": 1.0e-4,
    "optimizer.params.flow_learning_rate": 1.0e-4,
    "optimizer.params.special_token_learning_rate": 2.0e-5,
    "optimizer.params.scale_lr": False,
    "optimizer.params.beta1": 0.9,
    "optimizer.params.beta2": 0.95,
    "optimizer.params.weight_decay": 0.01,
    "optimizer.params.epsilon": 1.0e-8,
    "lr_scheduler.scheduler": "wsd",
    "lr_scheduler.params.learning_rate": 1.0e-4,
    "lr_scheduler.params.warmup_steps": 2000,
    "lr_scheduler.params.decay_steps": 8980,
    "lr_scheduler.params.min_lr_scale": 0.1,
    "training.batch_size": 32,
    "training.total_batch_size": 256,
    "training.mixed_precision": "bf16",
    "training.enable_tf32": True,
    "training.max_train_steps": 35920,
    "training.max_grad_norm": 1.0,
    "training.ar_ratio": -1.0,
    "training.mc_samples": 16,
    "training.use_gradient_checkpointing": False,
    "training.step_scheduler_with_optimizer": False,
    "training.from_scratch": False,
    "training.freeze_backbone_for_image_flow_warmup": False,
    "training.use_ema": True,
    "training.ema_decay": 0.999,
    "training.ema_update_after_step": 0,
    "training.ema_validate": False,
    "training.ema_save_adapter": True,
    "training.ema_save_hf_model": True,
    "training.save_image_flow_adapter": True,
}


def training_protocol_fingerprint(invariants: dict[str, object]) -> str:
    encoded = json.dumps(
        invariants,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def training_protocol_metadata(
    config: DictConfig,
    *,
    final_global_step: int | None,
) -> dict[str, object]:
    invariants = {
        key: OmegaConf.select(config, key)
        for key in TRAINING_PROTOCOL_INVARIANTS
    }
    metadata = {
        "schema": TRAINING_PROTOCOL_SCHEMA,
        "training_seed": int(config.training.seed),
        "invariants": invariants,
        "invariants_sha256": training_protocol_fingerprint(invariants),
        "final_global_step": final_global_step,
    }
    if is_confirmation_config(config):
        declaration = validate_confirmation_declaration(
            OmegaConf.to_container(config.experiment.confirmation_protocol, resolve=True),
            variant_id=str(config.experiment.ablation_id),
            seed=int(config.training.seed),
        )
        metadata["confirmation"] = {
            "declaration_sha256": declaration["declaration_sha256"],
            "screen_summary_sha256": declaration["screen_summary_sha256"],
            "candidate_manifest_sha256": declaration["candidate_manifest_sha256"],
            "evaluator_rng_contract_sha256": declaration[
                "evaluator_rng_contract_sha256"
            ],
            "dataloader_shuffle_seed": int(config.training.dataloader_shuffle_seed),
            "provenance_path": str(
                config.experiment.get("confirmation_provenance_path", "")
            ),
            "provenance_sha256": str(
                config.experiment.get("confirmation_provenance_sha256", "")
            ),
        }
    return metadata


@dataclass(frozen=True)
class AblationVariant:
    query_stage_mode: str
    observed_position_mode: str
    rope_mode: str
    space_to_depth_factor: int
    purpose: str


VARIANTS: dict[str, AblationVariant] = {
    "E0": AblationVariant("none", "additive_2d", "sequence_1d", 1, "baseline"),
    "E1": AblationVariant("fixed_sincos", "additive_2d", "sequence_1d", 1, "stage only"),
    "E2a": AblationVariant("none", "none", "sequence_1d", 1, "remove observed additive position only"),
    "E2b": AblationVariant("none", "additive_2d", "row_col_2d", 1, "row/column RoPE only"),
    "E2": AblationVariant("none", "none", "row_col_2d", 1, "full R factor"),
    "E3": AblationVariant("none", "additive_2d", "sequence_1d", 2, "lossless S2D only"),
    "E4a": AblationVariant("fixed_sincos", "none", "sequence_1d", 1, "stage x remove observed additive position"),
    "E4b": AblationVariant("fixed_sincos", "additive_2d", "row_col_2d", 1, "stage x row/column RoPE"),
    "E4": AblationVariant("fixed_sincos", "none", "row_col_2d", 1, "stage x R"),
    "E5": AblationVariant("fixed_sincos", "additive_2d", "sequence_1d", 2, "stage x S2D"),
    "E6a": AblationVariant("none", "none", "sequence_1d", 2, "S2D x remove observed additive position"),
    "E6b": AblationVariant("none", "additive_2d", "row_col_2d", 2, "S2D x row/column RoPE"),
    "E6": AblationVariant("none", "none", "row_col_2d", 2, "R x S2D"),
    "E7a": AblationVariant("fixed_sincos", "none", "sequence_1d", 2, "stage x S2D x remove observed additive position"),
    "E7b": AblationVariant("fixed_sincos", "additive_2d", "row_col_2d", 2, "stage x S2D x row/column RoPE"),
    "E7": AblationVariant("fixed_sincos", "none", "row_col_2d", 2, "full proposal"),
}


def normalize_variant_id(variant_id: str) -> str:
    normalized = str(variant_id).strip().lower()
    for candidate in VARIANTS:
        if candidate.lower() == normalized:
            return candidate
    raise ValueError(
        f"Unknown ablation ID {variant_id!r}; expected one of {', '.join(VARIANTS)}"
    )


def run_slug(variant_id: str, seed: int) -> str:
    return f"selfless-flow-image-embedder-{normalize_variant_id(variant_id).lower()}-seed{int(seed)}"


def validate_ablation_config(config: DictConfig, expected_id: str | None = None) -> None:
    variant_id = normalize_variant_id(expected_id or config.experiment.ablation_id)
    variant = VARIANTS[variant_id]
    actual_variant = (
        str(config.model.image_query_stage_mode),
        str(config.model.image_observed_position_mode),
        str(config.model.image_rope_mode),
        int(config.model.image_space_to_depth_factor),
    )
    expected_variant = (
        variant.query_stage_mode,
        variant.observed_position_mode,
        variant.rope_mode,
        variant.space_to_depth_factor,
    )
    if actual_variant != expected_variant:
        raise ValueError(
            f"{variant_id} architecture mismatch: expected {expected_variant}, got {actual_variant}"
        )
    for key, expected in FLOW_HEAD_INVARIANTS.items():
        actual = config.model.get(key)
        if actual != expected:
            raise ValueError(
                f"All variants must freeze model.{key}={expected!r}; got {actual!r}"
            )
    if str(config.model.image_generation_head_type) != "flow":
        raise ValueError("All variants must use image_generation_head_type=flow")
    for key, expected in TRAINING_PROTOCOL_INVARIANTS.items():
        actual = OmegaConf.select(config, key)
        if actual != expected:
            raise ValueError(
                f"All variants must freeze {key}={expected!r}; got {actual!r}"
            )

    layout = resolve_image_layout_config(config)
    expected_length = 320 if layout["factor"] == 1 else 128
    length_fields = (
        int(config.dataset.params.max_seq_length),
        int(config.dataset.params.pad_to_length),
        int(config.dataset.preprocessing.max_seq_length),
    )
    if length_fields != (expected_length,) * 3:
        raise ValueError(
            f"factor={layout['factor']} requires sequence lengths {(expected_length,) * 3}; "
            f"got {length_fields}"
        )
    if int(config.dataset.params.split_seed) != 42:
        raise ValueError("Dataset split_seed must remain 42 for every training seed")
    if int(config.evaluation.seed) != 42:
        raise ValueError("Evaluation sampling seed must remain 42 for architecture ranking")

    phase = str(config.experiment.get("ablation_phase", "screen"))
    if phase not in {"screen", "confirmation"}:
        raise ValueError(f"Unknown experiment.ablation_phase={phase!r}")
    if phase == "confirmation":
        seed = int(config.training.seed)
        if seed not in CONFIRMATION_SEEDS:
            raise ValueError(
                "Confirmation training seed must be one of "
                f"{sorted(CONFIRMATION_SEEDS)}, got {seed}"
            )
        if int(config.training.get("dataloader_shuffle_seed", -1)) != seed:
            raise ValueError(
                "Confirmation requires training.dataloader_shuffle_seed == training.seed"
            )
        declaration = validate_confirmation_declaration(
            OmegaConf.to_container(config.experiment.confirmation_protocol, resolve=True),
            variant_id=variant_id,
            seed=seed,
        )
        screen_path = Path(str(declaration["screen_summary_path"]))
        if not screen_path.is_file():
            raise ValueError(f"Confirmation screen summary is missing: {screen_path}")
        if file_sha256(screen_path) != declaration["screen_summary_sha256"]:
            raise ValueError("Confirmation screen summary content changed after preregistration")


def build_ablation_config(
    variant_id: str,
    seed: int,
    *,
    base_config: str | Path = DEFAULT_BASE_CONFIG,
    confirmation_screen_json: str | Path | None = None,
) -> DictConfig:
    variant_id = normalize_variant_id(variant_id)
    seed = int(seed)
    variant = VARIANTS[variant_id]
    config = OmegaConf.load(Path(base_config))
    slug = run_slug(variant_id, seed)

    config.experiment.project = slug
    config.experiment.name = (
        f"imagenet100-image-embedder-{variant_id.lower()}-seed{seed}-"
        "qwen3base-8gpu-b256-80ep"
    )
    config.experiment.ablation_id = variant_id
    config.experiment.ablation_phase = "screen"
    config.training.seed = seed
    config.model.image_query_stage_mode = variant.query_stage_mode
    config.model.image_observed_position_mode = variant.observed_position_mode
    config.model.image_rope_mode = variant.rope_mode
    config.model.image_space_to_depth_factor = variant.space_to_depth_factor

    sequence_length = 320 if variant.space_to_depth_factor == 1 else 128
    config.dataset.params.max_seq_length = sequence_length
    config.dataset.params.pad_to_length = sequence_length
    config.dataset.preprocessing.max_seq_length = sequence_length
    config.evaluation.checkpoint = f"output/{slug}/hf_model-final-ema"

    if confirmation_screen_json is not None:
        declaration = build_confirmation_declaration(
            variant_id=variant_id,
            seed=seed,
            screen_path=confirmation_screen_json,
        )
        config.experiment.ablation_phase = "confirmation"
        config.training.dataloader_shuffle_seed = seed
        config.experiment.confirmation_protocol = declaration

    resolve_image_layout_config(config)
    validate_ablation_config(config, variant_id)
    return config


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="Print the experiment matrix as JSON.")
    parser.add_argument("--id", dest="variant_id", help="Variant ID, e.g. E2a, E6b, or E7.")
    parser.add_argument("--seed", type=int, default=42, help="Training seed (default: 42).")
    parser.add_argument("--base-config", default=str(DEFAULT_BASE_CONFIG))
    parser.add_argument(
        "--confirmation-screen-json",
        help=(
            "Expanded seed-42 summary used to preregister a seed-43/44/45 "
            "confirmation run."
        ),
    )
    parser.add_argument("--output", help="Resolved YAML output path.")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.list:
        print(json.dumps({key: asdict(value) for key, value in VARIANTS.items()}, indent=2))
        return
    if not args.variant_id or not args.output:
        raise SystemExit("--id and --output are required unless --list is used")
    config = build_ablation_config(
        args.variant_id,
        args.seed,
        base_config=args.base_config,
        confirmation_screen_json=args.confirmation_screen_json,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, output)
    print(output.resolve())


if __name__ == "__main__":
    main()
