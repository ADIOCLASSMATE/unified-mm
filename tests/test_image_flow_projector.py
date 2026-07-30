import os
import tempfile

os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

import torch
from transformers import Qwen3Config

from models.modeling_model.image_flow_loss import (
    ContextualFlowTransformerHead,
)
from models.modeling_model.modeling_selfless_flow import (
    ImageTokenEmbedder,
    Qwen3ForCausalLM,
)


def _tiny_config():
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
    config.image_flow_width = 32
    config.image_flow_depth = 1
    config.image_flow_num_sampling_steps = "2"
    config.backbone_attention_output_gate = "none"
    return config


def test_model_contains_only_selected_image_modules():
    config = _tiny_config()
    model = Qwen3ForCausalLM(config)

    assert isinstance(model.image_token_embedder, ImageTokenEmbedder)
    assert isinstance(model.image_flow_head.net, ContextualFlowTransformerHead)
    assert isinstance(model.image_flow_condition_proj, torch.nn.Linear)
    assert set(model.image_token_embedder.state_dict()) == {
        "z_proj.weight",
        "z_proj.bias",
    }
    assert model.image_flow_head.net.position_contract()[
        "additive_image_position"
    ] is False
    assert model.image_flow_head.net.position_contract()["rope_mode"] == (
        "row_col_2d"
    )


def test_selected_initialization_and_final_zero_projection():
    model = Qwen3ForCausalLM(_tiny_config())
    embedder = model.image_token_embedder

    assert embedder.last_init_stats["init_mode"] == "pure_2d"
    assert embedder.z_proj.weight.detach().float().std() > 0
    assert torch.count_nonzero(embedder.z_proj.bias) == 0
    assert torch.count_nonzero(
        model.image_flow_head.net.final_layer.linear.weight
    ) == 0
    assert torch.count_nonzero(
        model.image_flow_head.net.final_layer.linear.bias
    ) == 0


def test_flow_head_uses_selected_capacity_contract():
    config = _tiny_config()
    model = Qwen3ForCausalLM(config)
    assert model.image_flow_head.net.blocks[0].mlp[0].out_features == 32
    assert model.image_flow_head.net.num_heads == 8
    assert model.image_flow_head.net.dropout == 0.0

    condition = torch.randn(3, config.hidden_size)
    output = model._prepare_image_flow_condition(condition)
    assert output.shape == condition.shape
    assert torch.isfinite(output).all()


def test_image_input_noise_is_train_only():
    config = _tiny_config()
    config.image_input_noise_strength = 1.0
    model = Qwen3ForCausalLM(config).model
    input_ids = torch.tensor([[1, 2, 3, 4]])
    token_types = torch.ones_like(input_ids, dtype=torch.uint8)
    image_latents = torch.zeros(1, 4, config.image_latent_dim)

    model.train()
    torch.manual_seed(0)
    train_a = model._build_x0_inputs_embeds(
        input_ids, token_types, image_latents, None
    )
    torch.manual_seed(1)
    train_b = model._build_x0_inputs_embeds(
        input_ids, token_types, image_latents, None
    )
    assert not torch.allclose(train_a, train_b)

    model.eval()
    torch.manual_seed(0)
    eval_a = model._build_x0_inputs_embeds(
        input_ids, token_types, image_latents, None
    )
    torch.manual_seed(1)
    eval_b = model._build_x0_inputs_embeds(
        input_ids, token_types, image_latents, None
    )
    assert torch.allclose(eval_a, eval_b)


def test_reset_image_modules_restores_selected_initialization():
    model = Qwen3ForCausalLM(_tiny_config())
    with torch.no_grad():
        model.image_token_embedder.z_proj.weight.zero_()
        model.image_flow_condition_proj.weight.zero_()
        model.image_flow_head.net.final_layer.linear.weight.fill_(1.0)

    model.reset_image_modules()
    assert model.image_token_embedder.z_proj.weight.float().std() > 0
    assert model.image_flow_condition_proj.weight.float().std() > 0
    assert torch.count_nonzero(
        model.image_flow_head.net.final_layer.linear.weight
    ) == 0


def test_selected_architecture_round_trips_without_ablation_fields():
    model = Qwen3ForCausalLM(_tiny_config())
    with tempfile.TemporaryDirectory() as directory:
        model.save_pretrained(directory, safe_serialization=True)
        loaded = Qwen3ForCausalLM.from_pretrained(
            directory,
            trust_remote_code=True,
        )

    assert isinstance(loaded.image_flow_head.net, ContextualFlowTransformerHead)
    for retired in (
        "image_flow_head_arch",
        "image_flow_head_variant",
        "image_flow_position_variant",
        "image_token_embedder_init_mode",
    ):
        assert not hasattr(loaded.config, retired)
