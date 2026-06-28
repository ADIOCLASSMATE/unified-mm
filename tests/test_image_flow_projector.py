import os

os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

import torch
from omegaconf import OmegaConf
from transformers import Qwen3Config

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
    config.image_flow_condition_norm = "none"
    return config


def test_image_projectors_use_normal_init_with_nextstep_mlp_ratio():
    config = tiny_qwen3_config()
    model = Qwen3ForCausalLM(config)

    assert isinstance(model.image_flow_condition_proj, torch.nn.Linear)
    assert not torch.allclose(model.image_token_embedder.z_proj.weight, torch.zeros_like(model.image_token_embedder.z_proj.weight))
    assert not torch.allclose(model.image_flow_condition_proj.weight, torch.eye(config.hidden_size))
    assert model.image_token_embedder.z_proj.weight.detach().float().std() > 0
    assert model.image_flow_condition_proj.weight.detach().float().std() > 0
    assert torch.allclose(model.image_token_embedder.z_proj.bias, torch.zeros_like(model.image_token_embedder.z_proj.bias))
    assert torch.allclose(model.image_flow_condition_proj.bias, torch.zeros(config.hidden_size))
    assert model.image_flow_head.net.res_blocks[0].intermediate_size == config.image_flow_width
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
        model.image_flow_head.net.res_blocks[0].adaLN_modulation[-1].weight,
        torch.zeros_like(model.image_flow_head.net.res_blocks[0].adaLN_modulation[-1].weight),
    )


def test_image_flow_mlp_ratio_can_widen_resblocks():
    config = tiny_qwen3_config()
    config.image_flow_mlp_ratio = 2.0
    model = Qwen3ForCausalLM(config)

    assert model.image_flow_head.net.res_blocks[0].intermediate_size == 16

    z = torch.randn(3, config.hidden_size)
    out = model._prepare_image_flow_condition(z)
    assert out.shape == z.shape
    assert torch.isfinite(out).all()


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
    assert "diffusion_pos_embed.weight" not in state_keys
    assert embedder.image_pos_embed.shape == (4, 8)
    assert embedder.diffusion_pos_embed.shape == (4, 8)

    positions = torch.arange(4)
    zeros = torch.zeros(4, 8)
    with_diffusion_pos = embedder.add_diffusion_pos(zeros, positions)

    assert torch.allclose(with_diffusion_pos, embedder.diffusion_pos_embed)
    assert not torch.allclose(embedder.image_pos_embed[0], embedder.image_pos_embed[1])
