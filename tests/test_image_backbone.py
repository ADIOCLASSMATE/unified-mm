import inspect

import pytest
import torch
from omegaconf import OmegaConf
from transformers import Qwen3Config

from models.modeling_model.image_backbone import (
    pure_2d_position_contract,
    validate_image_data_layout,
    validate_model_image_layout,
)
from models.modeling_model.image_position_utils import (
    build_local_row_col_rope,
    build_row_col_position_ids,
)
from models.modeling_model.modeling_selfless_flow import (
    ImageTokenEmbedder,
    Qwen3Model,
)
from utils.dataset_imagenet_flow_cache import ImageNetFlowCacheDataset


def _tiny_qwen_config():
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
    return config


def test_position_contract_is_fixed_pure_2d():
    assert pure_2d_position_contract() == {
        "schema": "selfless_pure_2d_position_v1",
        "backbone": {
            "image_qk_rotary": "row_col_2d",
            "text_qk_rotary": "qwen_sequence_1d",
            "additive_image_position": False,
        },
        "flow_head": {
            "architecture": "dynamic_dual_stream",
            "image_qk_rotary": "row_col_2d",
            "rotate_value": False,
            "additive_image_position": False,
        },
    }


def test_model_and_dataset_layouts_must_match():
    assert validate_model_image_layout(
        {"image_tokens_per_img": 256, "image_latent_dim": 16}
    ) == (256, 16)
    with pytest.raises(ValueError, match="square"):
        validate_model_image_layout(
            {"image_tokens_per_img": 255, "image_latent_dim": 16}
        )

    config = OmegaConf.create(
        {
            "model": {
                "image_tokens_per_img": 256,
                "image_latent_dim": 16,
            },
            "dataset": {
                "params": {
                    "image_tokens_per_img": 256,
                    "image_latent_dim": 8,
                }
            },
        }
    )
    with pytest.raises(ValueError, match="must match"):
        validate_image_data_layout(config)


def test_row_col_ids_preserve_text_and_layout_image_grid():
    token_types = torch.tensor([[0, 2, 1, 1, 1, 1, 2, 0]])
    ids = build_row_col_position_ids(token_types, image_tokens_per_img=4)
    assert ids[:, 0].tolist() == [
        [0, 1, 2, 2, 3, 3, 4, 5],
        [0, 1, 2, 3, 2, 3, 4, 5],
    ]


def test_local_rope_uses_row_and_column_coordinates():
    positions = torch.tensor([[0, 1, 2, 3]])
    cos, sin = build_local_row_col_rope(
        positions,
        image_tokens_per_img=4,
        head_dim=4,
        axis_dims=(2, 2),
    )
    assert cos.shape == sin.shape == (1, 4, 4)
    assert torch.equal(cos[:, 0], torch.ones_like(cos[:, 0]))
    assert torch.equal(sin[:, 0], torch.zeros_like(sin[:, 0]))
    assert not torch.equal(cos[:, 1], cos[:, 2])


def test_embedder_has_projection_only_and_no_position_parameters():
    embedder = ImageTokenEmbedder(
        latent_dim=4,
        hidden_size=8,
        image_tokens_per_img=4,
    )
    assert set(embedder.state_dict()) == {"z_proj.weight", "z_proj.bias"}
    assert embedder.last_init_stats["additive_image_position"] is False
    assert embedder(torch.zeros(5, 4)).shape == (5, 8)
    assert "positions" not in inspect.signature(embedder.forward).parameters


def test_qwen_model_exposes_no_architecture_ablation_state():
    model = Qwen3Model(_tiny_qwen_config())
    state_names = set(model.state_dict())
    assert not any("image_pos" in name for name in state_names)
    assert not any("flow_pos" in name for name in state_names)
    for retired in (
        "image_backbone_variant",
        "image_query_stage_mode",
        "image_observed_position_mode",
        "image_mask_position_mode",
        "image_rope_mode",
    ):
        assert not hasattr(model, retired)


def test_dataset_signature_has_no_retired_layout_or_mode_switches():
    parameters = inspect.signature(ImageNetFlowCacheDataset).parameters
    assert "image_space_to_depth_factor" not in parameters
    assert "condition_payload" not in parameters
    assert "caption_sequence_modes" not in parameters
    assert "caption_prefix" not in parameters
    assert "label_text" not in parameters
    assert "latent_hflip_prob" not in parameters
