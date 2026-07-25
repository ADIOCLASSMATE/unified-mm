"""Archived tests for the retired per-field mask-position API; not part of CI."""
import os
import tempfile

os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

import pytest
import torch
from torch import nn
from transformers import Qwen3Config

from models.modeling_model.image_position_utils import build_row_col_position_ids
from models.modeling_model.modeling_selfless_flow import (
    ImageTokenEmbedder,
    Qwen3ForCausalLM,
    Qwen3Model,
)


def _tiny_config(**overrides):
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    config.mask_token_id = 7
    config.image_mask_token_id = 8
    config.image_latent_dim = 4
    config.image_tokens_per_img = 4
    config.image_input_noise_strength = 0.0
    config.image_flow_width = 16
    config.image_flow_depth = 1
    config.image_flow_latent_mixer_heads = 2
    config.image_flow_num_sampling_steps = "2"
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def _assert_state_dicts_equal(first, second):
    first_state = first.state_dict()
    second_state = second.state_dict()
    assert first_state.keys() == second_state.keys()
    for key in first_state:
        assert torch.equal(first_state[key], second_state[key]), key


def test_legacy_and_explicit_q1_are_bitwise_identical_for_mask_x0_and_xt():
    legacy_config = _tiny_config()
    explicit_config = _tiny_config(
        image_mask_position_mode="additive_2d",
        image_observed_position_mode="additive_2d",
        image_query_stage_mode="none",
    )
    torch.manual_seed(1234)
    legacy = Qwen3Model(legacy_config).eval()
    torch.manual_seed(1234)
    explicit = Qwen3Model(explicit_config).eval()

    assert legacy.image_token_embedder.mask_position_mode == "additive_2d"
    assert explicit.image_token_embedder.mask_position_mode == "additive_2d"
    _assert_state_dicts_equal(legacy, explicit)

    local_positions = torch.tensor([3, 0, 2, 1])
    mask_embedding = torch.randn(16)
    assert torch.equal(
        legacy.image_token_embedder.embed_mask(local_positions, mask_embedding),
        explicit.image_token_embedder.embed_mask(local_positions, mask_embedding),
    )

    input_ids = torch.tensor([[1, 8, 8, 8, 8, 2]])
    token_types = torch.tensor([[0, 1, 1, 1, 1, 2]], dtype=torch.uint8)
    image_latents = torch.randn(1, 6, 4)
    image_latent_mask = torch.tensor(
        [[False, True, False, True, False, False]],
        dtype=torch.bool,
    )
    legacy_x0 = legacy._build_x0_inputs_embeds(
        input_ids,
        token_types,
        image_latents,
        image_latent_mask,
    )
    explicit_x0 = explicit._build_x0_inputs_embeds(
        input_ids,
        token_types,
        image_latents,
        image_latent_mask,
    )
    assert torch.equal(legacy_x0, explicit_x0)
    assert torch.equal(
        legacy._build_xt_inputs_embeds(input_ids, token_types),
        explicit._build_xt_inputs_embeds(input_ids, token_types),
    )


def test_q0_without_stage_is_exact_mask_and_skips_all_spatial_ops(monkeypatch):
    embedder = ImageTokenEmbedder(
        latent_dim=4,
        hidden_size=8,
        image_tokens_per_img=4,
        mask_position_mode="none",
        query_stage_mode="none",
    )

    def unexpected_spatial_op(*_args, **_kwargs):
        pytest.fail("Q0 mask embedding must not look up or scale spatial positions")

    monkeypatch.setattr(embedder, "_lookup_pos_embed", unexpected_spatial_op)
    monkeypatch.setattr(embedder, "_scale_image_pos", unexpected_spatial_op)

    mask_embedding = torch.randn(8)
    positions = torch.tensor([0, 3, 1, 2])
    expected = mask_embedding.to(dtype=embedder.weight_dtype).expand(4, 8)
    assert torch.equal(embedder.embed_mask(positions, mask_embedding), expected)
    assert torch.equal(embedder.embed_mask(positions.flip(0), mask_embedding), expected)

    with pytest.raises(ValueError, match="image local positions"):
        embedder.embed_mask(torch.tensor([0, 4]), mask_embedding)


def test_q0_with_stage_is_exact_mask_plus_stage_and_never_uses_spatial_table(monkeypatch):
    embedder = ImageTokenEmbedder(
        latent_dim=4,
        hidden_size=8,
        image_tokens_per_img=4,
        mask_position_mode="none",
        query_stage_mode="fixed_sincos",
    )
    assert embedder.refresh_nonpersistent_buffers()

    original_lookup = embedder._lookup_pos_embed
    lookup_tables = []

    def tracked_lookup(table, local_positions, dtype):
        lookup_tables.append(table)
        assert table is not embedder.image_pos_embed
        return original_lookup(table, local_positions, dtype)

    def unexpected_scale(*_args, **_kwargs):
        pytest.fail("Q0 mask embedding must not scale spatial positions")

    monkeypatch.setattr(embedder, "_lookup_pos_embed", tracked_lookup)
    monkeypatch.setattr(embedder, "_scale_image_pos", unexpected_scale)

    mask_embedding = torch.randn(8)
    positions = torch.tensor([0, 3, 1, 2])
    stages = torch.tensor([0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0])
    stage_indices = torch.arange(4)
    expected_stage = embedder.image_stage_embed.index_select(0, stage_indices)
    expected = mask_embedding.to(dtype=embedder.weight_dtype).expand(4, 8) + (
        expected_stage * embedder.image_stage_scale
    ).to(dtype=embedder.weight_dtype)

    first = embedder.embed_mask(positions, mask_embedding, stages=stages)
    second = embedder.embed_mask(positions.flip(0), mask_embedding, stages=stages)
    assert torch.equal(first, expected)
    assert torch.equal(second, expected)
    assert lookup_tables
    assert all(table is embedder.image_stage_embed for table in lookup_tables)


def test_mask_mode_changes_neither_state_nor_observed_latent_embeddings():
    torch.manual_seed(2026)
    q1 = ImageTokenEmbedder(
        latent_dim=4,
        hidden_size=8,
        image_tokens_per_img=4,
        observed_position_mode="additive_2d",
        mask_position_mode="additive_2d",
    )
    torch.manual_seed(2026)
    q0 = ImageTokenEmbedder(
        latent_dim=4,
        hidden_size=8,
        image_tokens_per_img=4,
        observed_position_mode="additive_2d",
        mask_position_mode="none",
    )

    _assert_state_dicts_equal(q1, q0)
    latents = torch.randn(4, 4)
    positions = torch.tensor([3, 0, 2, 1])
    assert torch.equal(
        q1.embed_latents(latents, positions),
        q0.embed_latents(latents, positions),
    )


class _CaptureLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.position_ids = None

    def forward(self, X0_hidden_states, XT_hidden_states, attention_mask, **kwargs):
        self.position_ids = kwargs["position_ids"].detach().clone()
        return X0_hidden_states, XT_hidden_states


def test_row_col_position_ids_are_identical_between_q0_and_q1():
    common = {
        "image_observed_position_mode": "none",
        "image_query_stage_mode": "none",
        "image_rope_mode": "row_col_2d",
        "image_canonical_grid_side": 2,
    }
    torch.manual_seed(99)
    q1 = Qwen3Model(
        _tiny_config(image_mask_position_mode="additive_2d", **common)
    ).eval()
    torch.manual_seed(99)
    q0 = Qwen3Model(_tiny_config(image_mask_position_mode="none", **common)).eval()
    q1_capture = _CaptureLayer()
    q0_capture = _CaptureLayer()
    q1.layers = nn.ModuleList([q1_capture])
    q0.layers = nn.ModuleList([q0_capture])
    q1.norm = nn.Identity()
    q0.norm = nn.Identity()

    input_ids = torch.tensor([[1, 8, 8, 8, 8, 2]])
    token_types = torch.tensor([[0, 1, 1, 1, 1, 2]], dtype=torch.uint8)
    image_latents = torch.zeros(1, 6, 4)
    image_latent_mask = torch.zeros_like(token_types, dtype=torch.bool)
    for model in (q1, q0):
        model(
            X0_input_ids=input_ids,
            attention_mask=object(),
            token_types=token_types,
            image_latents=image_latents,
            image_latent_mask=image_latent_mask,
            calculate_likelihood=False,
        )

    expected = build_row_col_position_ids(
        token_types,
        image_tokens_per_img=4,
        canonical_spatial_extent=2,
    )
    assert torch.equal(q1_capture.position_ids, expected)
    assert torch.equal(q0_capture.position_ids, expected)


@pytest.mark.parametrize(
    "mask_position_mode",
    [None, "none"],
    ids=["legacy_defaults_to_q1", "explicit_q0"],
)
def test_hf_round_trip_preserves_mask_position_semantics(mask_position_mode):
    config = _tiny_config()
    if mask_position_mode is not None:
        config.image_mask_position_mode = mask_position_mode
    else:
        assert not hasattr(config, "image_mask_position_mode")
    model = Qwen3ForCausalLM(config).eval()
    positions = torch.tensor([3, 0, 2, 1])
    mask_embedding = torch.randn(16)
    expected = model.image_token_embedder.embed_mask(positions, mask_embedding)

    with tempfile.TemporaryDirectory() as tmpdir:
        model.save_pretrained(tmpdir, safe_serialization=True)
        loaded = Qwen3ForCausalLM.from_pretrained(
            tmpdir,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        ).eval()

    expected_mode = "additive_2d" if mask_position_mode is None else "none"
    assert loaded.image_token_embedder.mask_position_mode == expected_mode
    if mask_position_mode is None:
        assert not hasattr(loaded.config, "image_mask_position_mode")
    else:
        assert loaded.config.image_mask_position_mode == "none"
    assert torch.equal(
        loaded.image_token_embedder.embed_mask(positions, mask_embedding),
        expected,
    )


def test_e2_q0_leaves_image_pos_gain_truly_unused():
    embedder = ImageTokenEmbedder(
        latent_dim=4,
        hidden_size=8,
        image_tokens_per_img=4,
        observed_position_mode="none",
        mask_position_mode="none",
        query_stage_mode="none",
    )
    assert isinstance(embedder.image_pos_gain, nn.Parameter)
    assert embedder.image_pos_gain.requires_grad

    positions = torch.arange(4)
    latents = torch.randn(4, 4)
    mask_embedding = torch.randn(8, requires_grad=True)
    loss = embedder.embed_latents(latents, positions).square().sum()
    loss = loss + embedder.embed_mask(positions, mask_embedding).square().sum()
    loss.backward()

    assert embedder.z_proj.weight.grad is not None
    assert mask_embedding.grad is not None
    assert embedder.image_pos_gain.grad is None
