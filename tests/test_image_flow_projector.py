import math
import os
import tempfile

os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

import torch
from omegaconf import OmegaConf
from transformers import Qwen3Config

from models.modeling_model.image_flow_loss import TokenFlowMLPHead
from models.modeling_model.modeling_selfless_flow import ImageTokenEmbedder, Qwen3ForCausalLM
from pretrain.train_selfless_flow import _reinitialize_image_modules


def tiny_qwen3_config():
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
    )
    config.mask_token_id = 7
    config.image_mask_token_id = 8
    config.image_latent_dim = 4
    config.image_tokens_per_img = 4
    config.image_flow_width = 8
    config.image_flow_depth = 1
    config.image_flow_num_sampling_steps = "2"
    return config


def test_image_projectors_use_normal_init_with_nextstep_mlp_ratio():
    config = tiny_qwen3_config()
    model = Qwen3ForCausalLM(config)

    assert model.image_token_embedder.init_mode == "balanced"
    assert not hasattr(model.image_token_embedder, "z_proj_ln")
    assert not any("z_proj_ln" in name for name in model.state_dict())
    assert isinstance(model.image_token_embedder.image_pos_gain, torch.nn.Parameter)
    assert isinstance(model.image_flow_condition_proj, torch.nn.Linear)
    assert not torch.allclose(model.image_token_embedder.z_proj.weight, torch.zeros_like(model.image_token_embedder.z_proj.weight))
    assert not torch.allclose(model.image_flow_condition_proj.weight, torch.eye(config.hidden_size))
    assert model.image_token_embedder.z_proj.weight.detach().float().std() > 0
    assert model.image_flow_condition_proj.weight.detach().float().std() > 0
    assert torch.allclose(model.image_token_embedder.z_proj.bias, torch.zeros_like(model.image_token_embedder.z_proj.bias))
    assert torch.allclose(model.image_flow_condition_proj.bias, torch.zeros(config.hidden_size))
    assert model.image_flow_head.net.blocks[0].mlp[0].out_features == config.image_flow_width
    assert torch.allclose(
        model.image_flow_head.net.final_layer.linear.weight,
        torch.zeros_like(model.image_flow_head.net.final_layer.linear.weight),
    )
    assert torch.allclose(
        model.image_flow_head.net.final_layer.linear.bias,
        torch.zeros_like(model.image_flow_head.net.final_layer.linear.bias),
    )
    assert torch.allclose(
        model.image_flow_head.net.final_layer.adaLN_modulation[-1].weight,
        torch.zeros_like(model.image_flow_head.net.final_layer.adaLN_modulation[-1].weight),
    )
    assert torch.allclose(
        model.image_flow_head.net.blocks[0].adaLN_modulation[-1].weight,
        torch.zeros_like(model.image_flow_head.net.blocks[0].adaLN_modulation[-1].weight),
    )


def test_default_image_token_embedder_uses_balanced_init_without_post_norm():
    embedder = ImageTokenEmbedder(latent_dim=4, hidden_size=8, image_tokens_per_img=4)

    assert embedder.init_mode == "balanced"
    assert not hasattr(embedder, "z_proj_ln")
    assert isinstance(embedder.image_pos_gain, torch.nn.Parameter)


def test_image_flow_mlp_ratio_can_widen_resblocks():
    config = tiny_qwen3_config()
    config.image_flow_mlp_ratio = 2.0
    model = Qwen3ForCausalLM(config)

    assert model.image_flow_head.net.blocks[0].mlp[0].out_features == 16

    z = torch.randn(3, config.hidden_size)
    out = model._prepare_image_flow_condition(z)
    assert out.shape == z.shape
    assert torch.isfinite(out).all()


def test_token_mlp_flow_head_contains_no_attention_and_is_strictly_pointwise():
    config = tiny_qwen3_config()
    config.image_flow_head_arch = "token_mlp"
    model = Qwen3ForCausalLM(config)
    head = model.image_flow_head

    assert head.head_arch == "token_mlp"
    assert not head.uses_latent_mixer
    assert isinstance(head.net, TokenFlowMLPHead)
    parameter_names = set(dict(head.named_parameters()))
    assert not any(
        marker in name
        for name in parameter_names
        for marker in ("cross", "attn", "query", "key")
    )
    assert not any(isinstance(module, torch.nn.MultiheadAttention) for module in head.modules())

    with torch.no_grad():
        torch.manual_seed(0)
        head.net.final_layer.linear.weight.normal_()
        head.net.final_layer.linear.bias.normal_()

    x_t = torch.randn(1, 3, config.image_latent_dim)
    t = torch.full((1, 3), 0.5)
    condition = torch.randn(1, 3, config.hidden_size)
    first = head.velocity(
        x_t,
        t,
        condition,
        context_latents=torch.randn(1, 3, config.image_latent_dim),
        context_mask=torch.ones(1, 3, 3, dtype=torch.bool),
        query_positions=torch.tensor([[0, 1, 2]]),
        context_positions=torch.tensor([[0, 1, 2]]),
    )

    changed = x_t.clone()
    changed[:, 1:] = torch.randn_like(changed[:, 1:]) * 100.0
    changed_condition = condition.clone()
    changed_condition[:, 1:] = torch.randn_like(changed_condition[:, 1:]) * 100.0
    second = head.velocity(
        changed,
        t,
        changed_condition,
        context_latents=torch.randn(1, 3, config.image_latent_dim) * 100.0,
        context_mask=torch.zeros(1, 3, 3, dtype=torch.bool),
        query_positions=torch.tensor([[2, 1, 0]]),
        context_positions=torch.tensor([[2, 1, 0]]),
    )

    assert torch.allclose(first[:, 0], second[:, 0])
    assert head.prepare_latent_mixer_cache(torch.randn(1, 3, config.image_latent_dim)) is None


def test_token_mlp_flow_head_arch_round_trips_with_pretrained_save():
    config = tiny_qwen3_config()
    config.image_flow_head_arch = "token_mlp"
    model = Qwen3ForCausalLM(config)

    with tempfile.TemporaryDirectory() as tmpdir:
        model.save_pretrained(tmpdir, safe_serialization=True)
        loaded = Qwen3ForCausalLM.from_pretrained(tmpdir, trust_remote_code=True)

    assert loaded.config.image_flow_head_arch == "token_mlp"
    assert loaded.image_flow_head.head_arch == "token_mlp"
    assert isinstance(loaded.image_flow_head.net, TokenFlowMLPHead)


def test_image_input_noise_is_train_only():
    config = tiny_qwen3_config()
    config.image_input_noise_strength = 1.0
    model = Qwen3ForCausalLM(config).model
    input_ids = torch.tensor([[1, 2, 3, 4]])
    token_types = torch.ones_like(input_ids, dtype=torch.uint8)
    image_latents = torch.zeros(1, 4, config.image_latent_dim)

    model.train()
    torch.manual_seed(0)
    train_a = model._build_x0_inputs_embeds(input_ids, token_types, image_latents, None)
    torch.manual_seed(1)
    train_b = model._build_x0_inputs_embeds(input_ids, token_types, image_latents, None)
    assert not torch.allclose(train_a, train_b)

    model.eval()
    torch.manual_seed(0)
    eval_a = model._build_x0_inputs_embeds(input_ids, token_types, image_latents, None)
    torch.manual_seed(1)
    eval_b = model._build_x0_inputs_embeds(input_ids, token_types, image_latents, None)
    assert torch.allclose(eval_a, eval_b)


def test_reinitialize_image_modules_resets_random_image_weights():
    config = tiny_qwen3_config()
    model = Qwen3ForCausalLM(config)

    with torch.no_grad():
        model.image_token_embedder.z_proj.weight.fill_(0.0)
        model.image_flow_condition_proj.weight.copy_(torch.eye(config.hidden_size))
        model.image_flow_head.net.final_layer.linear.weight.fill_(1.0)

    train_config = OmegaConf.create({"model": {"reinitialize_image_modules": True}})
    assert _reinitialize_image_modules(model, train_config)

    assert model.image_token_embedder.z_proj.weight.detach().float().std() > 0
    assert not torch.allclose(model.image_flow_condition_proj.weight, torch.eye(config.hidden_size))
    assert model.image_flow_condition_proj.weight.detach().float().std() > 0
    assert torch.allclose(
        model.image_flow_head.net.final_layer.linear.weight,
        torch.zeros_like(model.image_flow_head.net.final_layer.linear.weight),
    )


def test_image_token_embedder_uses_fixed_2d_sincos_positions():
    embedder = ImageTokenEmbedder(latent_dim=4, hidden_size=8, image_tokens_per_img=4)
    state_keys = set(embedder.state_dict().keys())

    assert "image_pos_embed.weight" not in state_keys
    assert "flow_pos_embed" not in state_keys
    assert embedder.image_pos_embed.shape == (4, 8)

    positions = torch.arange(4)
    zeros = torch.zeros(4, 4)
    with_image_pos = embedder(zeros, positions)

    assert with_image_pos.shape == (4, 8)
    assert not torch.allclose(embedder.image_pos_embed[0], embedder.image_pos_embed[1])


def test_image_token_embedder_rebuilds_fixed_positions_on_reset():
    embedder = ImageTokenEmbedder(latent_dim=4, hidden_size=8, image_tokens_per_img=4)
    expected = embedder.image_pos_embed.clone()

    embedder.image_pos_embed.fill_(torch.finfo(torch.float32).max)
    embedder._reset_position_buffers()
    actual = embedder.image_pos_embed

    assert torch.isfinite(actual).all()
    assert actual.abs().max() <= 1.0
    assert torch.allclose(actual, expected)


def test_image_token_embedder_uses_balanced_pos_gain_init():
    config = tiny_qwen3_config()
    config.image_token_embedder_init_mode = "balanced"
    config.image_token_embedder_latent_rms = 1.0
    model = Qwen3ForCausalLM(config)

    generator = torch.Generator().manual_seed(0)
    latents = torch.randn(1024, config.image_latent_dim, generator=generator)
    positions = torch.arange(config.image_tokens_per_img).repeat(1024 // config.image_tokens_per_img)
    image_embeds = model.image_token_embedder(latents, positions).detach().float()
    image_rms = image_embeds.pow(2).mean(dim=-1).sqrt().mean()
    target_rms = float(config.initializer_range)
    stats = model.image_token_embedder.last_init_stats

    assert stats is not None
    assert stats["init_mode"] == "balanced"
    assert math.isclose(stats["component_rms"], target_rms / math.sqrt(2.0), rel_tol=1e-6)
    assert isinstance(model.image_token_embedder.image_pos_gain, torch.nn.Parameter)
    assert math.isclose(model.image_token_embedder.image_pos_gain.item(), stats["image_pos_gain"], rel_tol=1e-6)
    assert image_rms < target_rms * 3.0
    assert image_rms > target_rms * 0.3
    assert image_rms < 0.2


def test_balanced_pos_gain_scales_image_pos():
    embedder = ImageTokenEmbedder(
        latent_dim=4,
        hidden_size=8,
        image_tokens_per_img=4,
        init_mode="balanced",
        latent_rms=1.0,
    )
    positions = torch.arange(4)
    zeros_latent = torch.zeros(4, 4)

    with torch.no_grad():
        embedder.z_proj.weight.zero_()
        embedder.z_proj.bias.zero_()

    image_pos = embedder(zeros_latent, positions)

    assert torch.allclose(
        image_pos,
        embedder.image_pos_embed * embedder.image_pos_gain.to(embedder.image_pos_embed.dtype),
    )


def test_image_flow_condition_does_not_add_position():
    config = tiny_qwen3_config()
    model = Qwen3ForCausalLM(config)

    z = torch.randn(4, config.hidden_size)
    expected = model.image_flow_condition_proj(
        z.to(
            device=model.image_flow_condition_proj.weight.device,
            dtype=model.image_flow_condition_proj.weight.dtype,
        )
    )
    no_pos = model._prepare_image_flow_condition(z)

    assert torch.allclose(no_pos, expected)


def test_legacy_flow_pos_embed_key_is_unexpected():
    config = tiny_qwen3_config()
    model = Qwen3ForCausalLM(config)
    state = model.state_dict()
    state["model.image_token_embedder.flow_pos_embed"] = torch.zeros(
        config.image_tokens_per_img,
        config.hidden_size,
    )

    clone = Qwen3ForCausalLM(config)
    missing, unexpected = clone.load_state_dict(state, strict=False)

    assert missing == []
    assert unexpected == ["model.image_token_embedder.flow_pos_embed"]


def test_balanced_image_pos_gain_round_trips_with_pretrained_save():
    config = tiny_qwen3_config()
    config.image_token_embedder_init_mode = "balanced"
    config.image_token_embedder_latent_rms = 1.0
    model = Qwen3ForCausalLM(config)
    with torch.no_grad():
        model.image_token_embedder.image_pos_gain.fill_(0.123)

    with tempfile.TemporaryDirectory() as tmpdir:
        model.save_pretrained(tmpdir, safe_serialization=True)
        loaded = Qwen3ForCausalLM.from_pretrained(tmpdir, trust_remote_code=True)

    assert loaded.config.image_token_embedder_init_mode == "balanced"
    assert math.isclose(float(loaded.config.image_token_embedder_latent_rms), 1.0)
    assert isinstance(loaded.image_token_embedder.image_pos_gain, torch.nn.Parameter)
    assert math.isclose(loaded.image_token_embedder.image_pos_gain.item(), 0.123, rel_tol=1e-6)
