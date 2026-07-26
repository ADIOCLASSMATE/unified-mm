import inspect

import pytest
import torch
from omegaconf import OmegaConf
from transformers import Qwen3Config

from models.modeling_model.image_backbone import (
    DEFAULT_IMAGE_BACKBONE_VARIANT,
    IMAGE_BACKBONE_SPECS,
    SUPPORTED_IMAGE_BACKBONE_VARIANTS,
    resolve_image_backbone_config,
    resolve_model_image_backbone,
)
from models.modeling_model.image_position_utils import (
    build_2d_sincos_position_embedding,
    build_row_col_position_ids,
)
from models.modeling_model.modeling_selfless_flow import (
    ImageTokenEmbedder,
    Qwen3Model,
)
from utils.dataset_imagenet_flow_cache import ImageNetFlowCacheDataset


def _tiny_qwen_config(**overrides):
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    config.mask_token_id = 7
    config.image_mask_token_id = 8
    config.image_tokens_per_img = 4
    config.image_latent_dim = 4
    config.image_input_noise_strength = 0.0
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def test_supported_backbones_are_closed_and_default_to_e2_q0():
    assert DEFAULT_IMAGE_BACKBONE_VARIANT == "E2-Q0"
    assert SUPPORTED_IMAGE_BACKBONE_VARIANTS == ("E2-Q1", "E2-Q0", "E2b-Q0")
    assert {
        key: (spec.observed_position_mode, spec.mask_position_mode)
        for key, spec in IMAGE_BACKBONE_SPECS.items()
    } == {
        "E2-Q1": ("none", "additive_2d"),
        "E2-Q0": ("none", "none"),
        "E2b-Q0": ("additive_2d", "none"),
    }


@pytest.mark.parametrize("variant", SUPPORTED_IMAGE_BACKBONE_VARIANTS)
def test_new_training_config_emits_only_backbone_variant(variant):
    config = OmegaConf.create(
        {
            "model": {
                "image_backbone_variant": variant,
                "image_tokens_per_img": 256,
                "image_latent_dim": 16,
            },
            "dataset": {
                "params": {
                    "image_tokens_per_img": 256,
                    "image_latent_dim": 16,
                }
            },
        }
    )
    spec = resolve_image_backbone_config(config)
    assert spec.variant == variant
    assert set(config.model) == {
        "image_backbone_variant",
        "image_tokens_per_img",
        "image_latent_dim",
    }
    assert "image_space_to_depth_factor" not in config.dataset.params


@pytest.mark.parametrize(
    ("legacy", "expected"),
    [
        (
            {
                "image_query_stage_mode": "none",
                "image_observed_position_mode": "none",
                "image_rope_mode": "row_col_2d",
                "image_space_to_depth_factor": 1,
            },
            "E2-Q1",
        ),
        (
            {
                "image_query_stage_mode": "none",
                "image_observed_position_mode": "none",
                "image_mask_position_mode": "none",
                "image_rope_mode": "row_col_2d",
                "image_space_to_depth_factor": 1,
            },
            "E2-Q0",
        ),
        (
            {
                "image_query_stage_mode": "none",
                "image_observed_position_mode": "additive_2d",
                "image_mask_position_mode": "none",
                "image_rope_mode": "row_col_2d",
                "image_space_to_depth_factor": 1,
            },
            "E2b-Q0",
        ),
    ],
)
def test_exact_legacy_checkpoints_migrate_to_retained_variant(legacy, expected):
    config = _tiny_qwen_config(**legacy)
    spec = resolve_model_image_backbone(config)
    assert spec.variant == expected
    assert config.image_backbone_variant == expected
    for key in legacy:
        assert not hasattr(config, key)


@pytest.mark.parametrize(
    "retired",
    [
        {"image_query_stage_mode": "fixed_sincos"},
        {
            "image_query_stage_mode": "none",
            "image_observed_position_mode": "none",
            "image_rope_mode": "sequence_1d",
        },
        {
            "image_query_stage_mode": "none",
            "image_observed_position_mode": "none",
            "image_rope_mode": "row_col_2d",
            "image_space_to_depth_factor": 2,
        },
        {
            "image_query_stage_mode": "none",
            "image_observed_position_mode": "additive_2d",
            "image_mask_position_mode": "additive_2d",
            "image_rope_mode": "row_col_2d",
        },
    ],
)
def test_retired_architectures_are_rejected(retired):
    with pytest.raises(ValueError, match="retired image-backbone architecture"):
        resolve_model_image_backbone(_tiny_qwen_config(**retired))


def test_old_knobs_cannot_be_combined_with_new_enum():
    config = _tiny_qwen_config(
        image_backbone_variant="E2-Q1",
        image_query_stage_mode="none",
    )
    with pytest.raises(ValueError, match="cannot be combined"):
        resolve_model_image_backbone(config)


@pytest.mark.parametrize("variant", SUPPORTED_IMAGE_BACKBONE_VARIANTS)
def test_model_and_embedder_persist_only_variant(variant):
    config = _tiny_qwen_config(image_backbone_variant=variant)
    model = Qwen3Model(config).eval()
    assert model.image_backbone_variant == variant
    assert model.image_token_embedder.backbone_variant == variant
    assert model.config.image_backbone_variant == variant
    for key in (
        "image_query_stage_mode",
        "image_observed_position_mode",
        "image_mask_position_mode",
        "image_rope_mode",
        "image_space_to_depth_factor",
    ):
        assert not hasattr(model.config, key)


def test_three_embedder_variants_apply_only_the_retained_position_choices():
    mask = torch.zeros(8)
    positions = torch.arange(4)
    latents = torch.zeros(4, 4)
    embedders = {
        variant: ImageTokenEmbedder(
            latent_dim=4,
            hidden_size=8,
            image_tokens_per_img=4,
            backbone_variant=variant,
        ).eval()
        for variant in SUPPORTED_IMAGE_BACKBONE_VARIANTS
    }
    for embedder in embedders.values():
        with torch.no_grad():
            embedder.z_proj.weight.zero_()
            embedder.z_proj.bias.zero_()
            if embedder.image_pos_gain is not None:
                embedder.image_pos_gain.fill_(1.0)

    e2_q1 = embedders["E2-Q1"]
    e2_q0 = embedders["E2-Q0"]
    e2b_q0 = embedders["E2b-Q0"]
    assert e2_q1.image_pos_gain is not None
    assert e2_q0.image_pos_gain is None
    assert e2b_q0.image_pos_gain is not None
    assert torch.count_nonzero(e2_q1.embed_latents(latents, positions)) == 0
    assert torch.count_nonzero(e2_q0.embed_latents(latents, positions)) == 0
    assert torch.count_nonzero(e2b_q0.embed_latents(latents, positions)) > 0
    assert torch.count_nonzero(e2_q1.embed_mask(positions, mask)) > 0
    assert torch.count_nonzero(e2_q0.embed_mask(positions, mask)) == 0
    assert torch.count_nonzero(e2b_q0.embed_mask(positions, mask)) == 0


def test_stage_and_s2d_are_absent_from_active_signatures():
    assert "coordinate_stride" not in inspect.signature(
        build_2d_sincos_position_embedding
    ).parameters
    assert "image_coordinate_stride" not in inspect.signature(
        build_row_col_position_ids
    ).parameters
    assert "image_space_to_depth_factor" not in inspect.signature(
        ImageNetFlowCacheDataset
    ).parameters
    assert "stages" not in inspect.signature(ImageTokenEmbedder.embed_mask).parameters
