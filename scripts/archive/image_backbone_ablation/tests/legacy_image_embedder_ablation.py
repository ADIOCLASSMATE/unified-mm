"""Archived tests for retired image-backbone interfaces; not part of CI."""
import os
import tempfile
from pathlib import Path

os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

import torch
import pytest
from torch import nn
from omegaconf import OmegaConf
from transformers import Qwen3Config

from models.modeling_model.image_latent_layout import (
    depth_to_space_2d,
    resolve_image_layout_config,
    restore_canonical_latents_chw,
    space_to_depth_2d,
)
from models.modeling_model.image_position_utils import (
    build_row_col_position_ids,
    build_stage_sincos_embedding,
    compute_image_reveal_stages,
)
from models.modeling_model.modeling_selfless_flow import (
    ImageTokenEmbedder,
    Qwen3ForCausalLM,
    Qwen3Model,
    Qwen3RotaryEmbedding,
    apply_rotary_pos_emb,
)
from utils.dataset_imagenet_flow_cache import (
    ImageNetFlowCacheDataset,
    build_training_data_generator,
)


class _Tokenizer:
    eos_token_id = 14

    def encode(self, text, add_special_tokens=False):
        return [10 + index for index, _ in enumerate(text.split())]


def test_space_to_depth_round_trip_and_channel_order():
    base = torch.arange(2 * 2 * 3).reshape(2, 2, 3)
    packed = space_to_depth_2d(base, 2)
    expected = torch.tensor(
        [[[0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11]]]
    )
    assert torch.equal(packed, expected)
    assert torch.equal(depth_to_space_2d(packed, 2), base)

    for dtype in (torch.float32, torch.bfloat16, torch.float16):
        values = torch.arange(2 * 16 * 16 * 16, dtype=torch.float32).reshape(2, 16, 16, 16).to(dtype)
        assert torch.equal(depth_to_space_2d(space_to_depth_2d(values, 2), 2), values)


def test_restore_canonical_latents_chw_is_exact():
    canonical = torch.arange(2 * 16 * 16 * 16).reshape(2, 16, 16, 16)
    packed_hwc = space_to_depth_2d(canonical.permute(0, 2, 3, 1), 2)
    packed_chw = packed_hwc.permute(0, 3, 1, 2)
    assert torch.equal(restore_canonical_latents_chw(packed_chw, 2), canonical)


def test_layout_factor_is_authoritative_across_model_and_dataset():
    config = OmegaConf.create(
        {
            "model": {
                "image_space_to_depth_factor": 2,
                "image_tokens_per_img": 64,
                "image_latent_dim": 64,
            },
            "dataset": {"params": {}},
        }
    )
    layout = resolve_image_layout_config(config)
    assert layout["image_grid_side"] == 8
    assert config.model.image_tokens_per_img == 64
    assert config.model.image_latent_dim == 64
    assert config.dataset.params.image_tokens_per_img == 64
    assert config.dataset.params.image_latent_dim == 64
    assert config.dataset.params.image_space_to_depth_factor == 2


def test_dataset_flips_canonical_latents_before_space_to_depth():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        canonical = torch.arange(2 * 2 * 2).reshape(1, 4, 2).to(torch.float16)
        torch.save({"latents": canonical, "img_ids": torch.tensor([0])}, root / "latents.pt")
        dataset = ImageNetFlowCacheDataset(
            cache_path=str(root / "latents.pt"),
            tokenizer=_Tokenizer(),
            boi_token_id=11,
            eoi_token_id=12,
            mask_token_id=7,
            eos_token_id=14,
            image_tokens_per_img=1,
            image_latent_dim=8,
            image_space_to_depth_factor=2,
            latent_hflip_prob=1.0,
            conditioning_mode="image_only",
            seed=0,
            label_text=False,
        )
        actual = dataset[0]["image_latents"]
        flipped = canonical[0].view(2, 2, 2).flip(1)
        expected = space_to_depth_2d(flipped, 2).reshape(1, 8)
        assert torch.equal(actual, expected)


def test_explicit_dataloader_shuffle_seed_is_independent_of_global_rng_consumption():
    disabled = OmegaConf.create({"training": {}})
    assert build_training_data_generator(disabled) is None

    config = OmegaConf.create({"training": {"dataloader_shuffle_seed": 43}})
    first = build_training_data_generator(config)
    torch.manual_seed(0)
    torch.rand(10_000)
    second = build_training_data_generator(config)

    assert torch.equal(
        torch.randperm(1_000, generator=first),
        torch.randperm(1_000, generator=second),
    )

    invalid = OmegaConf.create({"training": {"dataloader_shuffle_seed": -1}})
    with pytest.raises(ValueError, match="dataloader_shuffle_seed"):
        build_training_data_generator(invalid)


def test_reveal_stage_is_normalized_within_each_image_span():
    token_types = torch.tensor([[1, 1, 0, 1, 1, 1]], dtype=torch.uint8)
    sigma = torch.tensor([[10, 5, 0, 7, 7, 2]])
    stages = compute_image_reveal_stages(token_types, sigma)
    expected = torch.tensor([[1.0, 0.0, 0.0, 0.5, 0.5, 0.0]])
    assert torch.equal(stages, expected)


def test_reveal_stage_is_independent_of_prompt_offset_and_ties_parallel_queries():
    short_types = torch.tensor([[0, 1, 1, 1, 1, 2]], dtype=torch.uint8)
    long_types = torch.tensor([[0, 0, 0, 1, 1, 1, 1, 2]], dtype=torch.uint8)
    short_sigma = torch.tensor([[0, 13, 11, 13, 12, 20]])
    long_sigma = torch.tensor([[0, 1, 2, 103, 101, 103, 102, 120]])
    short = compute_image_reveal_stages(short_types, short_sigma)[0, 1:5]
    long = compute_image_reveal_stages(long_types, long_sigma)[0, 3:7]
    assert torch.equal(short, long)
    assert torch.allclose(short, torch.tensor([2 / 3, 0.0, 2 / 3, 1 / 3]))


def test_stage_zero_is_exactly_neutral_and_observed_latents_ignore_stage():
    embedder = ImageTokenEmbedder(
        latent_dim=4,
        hidden_size=8,
        image_tokens_per_img=4,
        query_stage_mode="fixed_sincos",
    )
    assert torch.equal(build_stage_sincos_embedding(torch.zeros(4), 8), torch.zeros(4, 8))
    positions = torch.arange(4)
    mask = torch.randn(8)
    stage_zero = embedder.embed_mask(positions, mask, stages=torch.zeros(4))
    baseline = mask.expand(4, 8) + embedder._scale_image_pos(embedder.image_pos_embed)
    assert torch.allclose(stage_zero, baseline)

    valid_stages = torch.linspace(0.0, 1.0, 4)
    table_values = embedder.embed_mask(positions, mask, stages=valid_stages)
    expected_stage_values = baseline + (
        build_stage_sincos_embedding(valid_stages, 8)
        * embedder.image_stage_scale
    ).to(dtype=baseline.dtype)
    assert torch.allclose(table_values, expected_stage_values, atol=1.0e-6, rtol=0.0)

    latents = torch.randn(4, 4)
    observed = embedder.embed_latents(latents, positions)
    assert torch.equal(observed, embedder.embed_latents(latents, positions))


def test_nonpersistent_stage_buffer_is_rebuilt_after_checkpoint_materialization():
    embedder = ImageTokenEmbedder(
        latent_dim=4,
        hidden_size=8,
        image_tokens_per_img=4,
        query_stage_mode="fixed_sincos",
    )
    original_z_proj = embedder.z_proj.weight.detach().clone()
    original_position = embedder.image_pos_embed.detach().clone()
    original_stage_scale = embedder.image_stage_scale.detach().clone()

    embedder.image_stage_embed = torch.full_like(embedder.image_stage_embed, float("nan"))
    embedder._image_stage_buffer_ready = False
    output = embedder.embed_mask(
        torch.arange(4),
        torch.zeros(8),
        stages=torch.linspace(0.0, 1.0, 4),
    )

    expected_stage = build_stage_sincos_embedding(torch.linspace(0.0, 1.0, 4), 8)
    assert torch.isfinite(output).all()
    assert torch.equal(embedder.image_stage_embed, expected_stage)
    assert torch.equal(embedder.z_proj.weight, original_z_proj)
    assert torch.equal(embedder.image_pos_embed, original_position)
    assert torch.equal(embedder.image_stage_scale, original_stage_scale)


def test_to_empty_invalidates_and_rebuilds_nonpersistent_stage_buffer():
    embedder = ImageTokenEmbedder(
        latent_dim=4,
        hidden_size=8,
        image_tokens_per_img=4,
        query_stage_mode="fixed_sincos",
    )
    assert embedder.refresh_nonpersistent_buffers()
    assert embedder._image_stage_buffer_ready

    embedder.to_empty(device=torch.device("cpu"))
    assert not embedder._image_stage_buffer_ready
    embedder._ensure_image_stage_buffer(torch.device("cpu"))

    expected_stage = build_stage_sincos_embedding(torch.linspace(0.0, 1.0, 4), 8)
    assert embedder._image_stage_buffer_ready
    assert torch.equal(embedder.image_stage_embed, expected_stage)


def test_e2a_removes_only_observed_additive_position():
    embedder = ImageTokenEmbedder(
        latent_dim=4,
        hidden_size=8,
        image_tokens_per_img=4,
        observed_position_mode="none",
    )
    with torch.no_grad():
        embedder.z_proj.weight.zero_()
        embedder.z_proj.bias.zero_()
    positions = torch.arange(4)
    observed = embedder.embed_latents(torch.zeros(4, 4), positions)
    query = embedder.embed_mask(positions, torch.zeros(8))
    assert torch.equal(observed, torch.zeros_like(observed))
    assert not torch.allclose(query[0], query[1])


def test_mask_position_none_is_exact_mask_and_skips_spatial_path(monkeypatch):
    embedder = ImageTokenEmbedder(
        latent_dim=4,
        hidden_size=8,
        image_tokens_per_img=4,
        mask_position_mode="none",
    )
    positions = torch.tensor([3, 0, 2, 1])
    mask = torch.randn(8)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Q0 must not look up or scale the spatial position table")

    monkeypatch.setattr(embedder, "_lookup_pos_embed", forbidden)
    monkeypatch.setattr(embedder, "_scale_image_pos", forbidden)
    output = embedder.embed_mask(positions, mask)

    assert torch.equal(output, mask.expand(4, 8))
    with pytest.raises(ValueError, match="must be in"):
        embedder.embed_mask(torch.tensor([0, 4]), mask)


def test_mask_position_none_keeps_optional_stage_embedding_exactly():
    embedder = ImageTokenEmbedder(
        latent_dim=4,
        hidden_size=8,
        image_tokens_per_img=4,
        mask_position_mode="none",
        query_stage_mode="fixed_sincos",
    )
    mask = torch.randn(8)
    positions = torch.tensor([3, 0, 2, 1])
    stages = torch.linspace(0.0, 1.0, 4)
    output = embedder.embed_mask(positions, mask, stages=stages)
    expected = mask.expand(4, 8) + (
        build_stage_sincos_embedding(stages, 8) * embedder.image_stage_scale
    ).to(dtype=mask.dtype)

    assert torch.allclose(output, expected, atol=1.0e-6, rtol=0.0)
    assert torch.equal(output[0], mask)


def test_mask_position_mode_does_not_change_state_or_observed_latents():
    torch.manual_seed(1234)
    additive = ImageTokenEmbedder(
        latent_dim=4,
        hidden_size=8,
        image_tokens_per_img=4,
        observed_position_mode="none",
        mask_position_mode="additive_2d",
    )
    torch.manual_seed(1234)
    none = ImageTokenEmbedder(
        latent_dim=4,
        hidden_size=8,
        image_tokens_per_img=4,
        observed_position_mode="none",
        mask_position_mode="none",
    )

    assert additive.state_dict().keys() == none.state_dict().keys()
    for key, value in additive.state_dict().items():
        assert torch.equal(value, none.state_dict()[key]), key

    latents = torch.randn(4, 4)
    positions = torch.tensor([3, 0, 2, 1])
    assert torch.equal(
        additive.embed_latents(latents, positions),
        none.embed_latents(latents, positions),
    )


def test_row_col_positions_preserve_pure_text_and_physical_image_stride():
    pure_text_types = torch.zeros(1, 5, dtype=torch.uint8)
    pure = build_row_col_position_ids(pure_text_types, 4, canonical_spatial_extent=2)
    expected = torch.arange(5).view(1, 5)
    assert torch.equal(pure[0], expected)
    assert torch.equal(pure[1], expected)

    token_types = torch.tensor([[0, 2, 1, 1, 1, 1, 2, 0]], dtype=torch.uint8)
    positions = build_row_col_position_ids(
        token_types,
        4,
        image_coordinate_stride=2,
        canonical_spatial_extent=4,
    )
    assert positions[0, 0].tolist() == [0, 1, 2, 2, 4, 4, 6, 7]
    assert positions[1, 0].tolist() == [0, 1, 2, 4, 2, 4, 6, 7]


def test_row_col_rotary_is_exactly_1d_when_axes_are_equal():
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
    )
    rotary = Qwen3RotaryEmbedding(config)
    hidden = torch.randn(2, 6, config.hidden_size)
    one_d = torch.arange(6).unsqueeze(0).expand(2, -1)
    two_d = torch.stack([one_d, one_d], dim=0)
    cos_1d, sin_1d = rotary(hidden, one_d)
    cos_2d, sin_2d = rotary(hidden, two_d)
    assert torch.equal(cos_1d, cos_2d)
    assert torch.equal(sin_1d, sin_2d)


def test_row_col_rotary_axes_are_interleaved_and_anchor_translation_is_relative():
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
    )
    rotary = Qwen3RotaryEmbedding(config)
    hidden = torch.randn(1, 2, config.hidden_size)
    positions = torch.tensor([[[5, 8]], [[7, 9]]])
    cos, sin = rotary(hidden, positions)

    col_changed = positions.clone()
    col_changed[1] += 3
    col_cos, col_sin = rotary(hidden, col_changed)
    row_dimensions = torch.tensor([0, 2, 4, 6])
    col_dimensions = torch.tensor([1, 3, 5, 7])
    assert torch.equal(cos.index_select(-1, row_dimensions), col_cos.index_select(-1, row_dimensions))
    assert torch.equal(sin.index_select(-1, row_dimensions), col_sin.index_select(-1, row_dimensions))
    assert not torch.equal(cos.index_select(-1, col_dimensions), col_cos.index_select(-1, col_dimensions))

    q = torch.randn(1, 1, 2, 8)
    k = torch.randn(1, 1, 2, 8)
    q_rot, _, k_rot = apply_rotary_pos_emb(q, None, k, cos, sin)
    shifted_cos, shifted_sin = rotary(hidden, positions + 17)
    shifted_q, _, shifted_k = apply_rotary_pos_emb(
        q,
        None,
        k,
        shifted_cos,
        shifted_sin,
    )
    score = (q_rot[:, :, 0] * k_rot[:, :, 1]).sum(dim=-1)
    shifted_score = (shifted_q[:, :, 0] * shifted_k[:, :, 1]).sum(dim=-1)
    assert torch.allclose(score, shifted_score, atol=1.0e-5, rtol=1.0e-5)


class _CaptureLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, X0_hidden_states, XT_hidden_states, attention_mask, **kwargs):
        self.calls.append(
            {
                "x0": X0_hidden_states.detach().clone(),
                "xt": None if XT_hidden_states is None else XT_hidden_states.detach().clone(),
                "position_ids": kwargs["position_ids"].detach().clone(),
            }
        )
        return X0_hidden_states, XT_hidden_states


def _tiny_model_config(**overrides):
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
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def test_e0_explicit_flags_are_numerically_identical_to_legacy_defaults():
    legacy_config = _tiny_model_config()
    explicit_config = _tiny_model_config(
        image_query_stage_mode="none",
        image_observed_position_mode="additive_2d",
        image_mask_position_mode="additive_2d",
        image_rope_mode="sequence_1d",
        image_space_to_depth_factor=1,
    )
    torch.manual_seed(1234)
    legacy = Qwen3Model(legacy_config)
    torch.manual_seed(1234)
    explicit = Qwen3Model(explicit_config)

    legacy_state = legacy.state_dict()
    explicit_state = explicit.state_dict()
    assert legacy_state.keys() == explicit_state.keys()
    for key in legacy_state:
        assert torch.equal(legacy_state[key], explicit_state[key]), key

    input_ids = torch.tensor([[1, 8, 8, 8, 8, 2]])
    token_types = torch.tensor([[0, 1, 1, 1, 1, 2]], dtype=torch.uint8)
    latents = torch.randn(1, 6, 4)
    legacy_embed = legacy._build_x0_inputs_embeds(input_ids, token_types, latents, None)
    explicit_embed = explicit._build_x0_inputs_embeds(input_ids, token_types, latents, None)
    assert torch.equal(legacy_embed, explicit_embed)


def test_stage_and_row_col_modes_reach_integrated_model_paths():
    config = _tiny_model_config(
        image_query_stage_mode="fixed_sincos",
        image_observed_position_mode="none",
        image_rope_mode="row_col_2d",
        image_canonical_grid_side=2,
    )
    model = Qwen3Model(config)
    capture = _CaptureLayer()
    model.layers = nn.ModuleList([capture])
    model.norm = nn.Identity()
    model.train()

    input_ids = torch.tensor([[1, 8, 8, 8, 8, 2]])
    token_types = torch.tensor([[0, 1, 1, 1, 1, 2]], dtype=torch.uint8)
    sigma = torch.tensor([[0, 4, 1, 3, 2, 5]])
    latents = torch.randn(1, 6, 4)
    model(
        X0_input_ids=input_ids,
        attention_mask=object(),
        token_types=token_types,
        image_latents=latents,
        flow_sigma=sigma,
        calculate_likelihood=True,
    )
    call = capture.calls[-1]
    expected_positions = build_row_col_position_ids(
        token_types,
        4,
        canonical_spatial_extent=2,
    )
    assert torch.equal(call["position_ids"], expected_positions)


    local = model.image_local_positions(token_types)
    image_mask = token_types == 1
    mask_embedding = model._image_mask_embedding(
        input_ids.device,
        model.image_token_embedder.weight_dtype,
    )
    stages = compute_image_reveal_stages(token_types, sigma)
    expected_xt = model.embed_tokens(torch.full_like(input_ids, config.mask_token_id))
    expected_xt[image_mask] = model.image_token_embedder.embed_mask(
        local[image_mask],
        mask_embedding,
        stages=stages[image_mask],
    )
    assert torch.equal(call["xt"], expected_xt)

    expected_observed = model.image_token_embedder.z_proj(latents[image_mask])
    assert torch.equal(call["x0"][image_mask], expected_observed)


def test_fixed_stage_buffer_rebuilds_after_low_memory_hf_round_trip():
    config = _tiny_model_config(
        image_query_stage_mode="fixed_sincos",
        image_mask_position_mode="none",
        image_canonical_grid_side=2,
    )
    model = Qwen3ForCausalLM(config)
    with torch.no_grad():
        model.image_token_embedder.z_proj.weight.fill_(0.125)
        model.image_token_embedder.image_pos_gain.fill_(0.321)
        model.image_token_embedder.image_stage_scale.fill_(0.25)

    expected_z_proj = model.image_token_embedder.z_proj.weight.detach().clone()
    expected_position = model.image_token_embedder.image_pos_embed.detach().clone()
    expected_pos_gain = model.image_token_embedder.image_pos_gain.detach().clone()
    expected_stage_scale = model.image_token_embedder.image_stage_scale.detach().clone()

    with tempfile.TemporaryDirectory() as tmpdir:
        model.save_pretrained(tmpdir, safe_serialization=True)
        loaded = Qwen3ForCausalLM.from_pretrained(
            tmpdir,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

    output = loaded.image_token_embedder.embed_mask(
        torch.arange(4),
        torch.zeros(16),
        stages=torch.linspace(0.0, 1.0, 4),
    )
    expected_stage = build_stage_sincos_embedding(torch.linspace(0.0, 1.0, 4), 16)

    assert torch.isfinite(output).all()
    assert torch.equal(loaded.image_token_embedder.image_stage_embed, expected_stage)
    assert torch.equal(loaded.image_token_embedder.z_proj.weight, expected_z_proj)
    assert torch.equal(loaded.image_token_embedder.image_pos_embed, expected_position)
    assert torch.equal(loaded.image_token_embedder.image_pos_gain, expected_pos_gain)
    assert torch.equal(loaded.image_token_embedder.image_stage_scale, expected_stage_scale)
    assert loaded.config.image_mask_position_mode == "none"
    assert loaded.image_token_embedder.mask_position_mode == "none"


def test_factor_two_contextual_flow_head_uses_block_width_and_physical_positions():
    config = _tiny_model_config(
        image_latent_dim=16,
        image_tokens_per_img=4,
        image_space_to_depth_factor=2,
        image_canonical_grid_side=4,
        image_canonical_latent_dim=4,
        image_flow_width=16,
        image_flow_depth=1,
        image_flow_latent_mixer_heads=2,
        image_flow_head_arch="contextual",
        image_flow_num_sampling_steps="2",
    )
    model = Qwen3ForCausalLM(config)
    head = model.image_flow_head
    assert head.head_arch == "contextual"
    assert head.uses_latent_mixer
    assert head.net.input_proj.in_features == 16
    assert head.net.final_layer.linear.out_features == 16
    expected_pos = model.image_token_embedder.image_pos_embed
    assert torch.equal(expected_pos, head.net.image_pos_embed)

    target = torch.randn(2, 4, 16, requires_grad=True)
    condition = torch.randn(2, 4, config.hidden_size)
    sigma = torch.tensor([[0, 2, 1, 3], [3, 1, 2, 0]])
    loss = head(target, condition, sigma=sigma, context_latents=target.detach())
    loss.backward()
    assert target.grad is not None
    assert torch.isfinite(target.grad).all()
    for bucket in ("00_25", "25_50", "50_75", "75_100"):
        assert f"flow/stage_{bucket}_v_mse" in head.last_forward_stats


def test_space_to_depth_gradient_is_a_pure_permutation():
    canonical = torch.randn(2, 16, 16, 16, requires_grad=True)
    packed = space_to_depth_2d(canonical, 2)
    packed.sum().backward()
    assert torch.equal(canonical.grad, torch.ones_like(canonical))
