from utils.selfless_flow_optimizer import weight_decay_for_parameter
from omegaconf import OmegaConf
from transformers import Qwen3Config

from models.modeling_model.modeling_selfless_flow import Qwen3ForCausalLM
from pretrain.train_selfless_flow import _apply_trainable_scope


def _tiny_model():
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=0,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=9,
        tie_word_embeddings=True,
    )
    config.mask_token_id = 7
    config.image_mask_token_id = 8
    config.boi_token_id = 11
    config.eoi_token_id = 12
    config.image_latent_dim = 4
    config.image_tokens_per_img = 4
    config.image_flow_width = 32
    config.image_flow_depth = 1
    config.image_flow_num_sampling_steps = "2"
    config.image_flow_batch_mul = 1
    config.image_flow_time_scale = 1000.0
    config.image_flow_time_sampling = "uniform"
    config.image_flow_time_eps = 1e-4
    config.image_flow_time_uniform_mix = 0.0
    config.image_flow_solver = "heun"
    config.image_uncond_prob = 0.0
    config.use_flex_attention = False
    return Qwen3ForCausalLM(config)


def test_flow_head_matrix_weights_receive_configured_weight_decay():
    assert weight_decay_for_parameter(
        "image_flow_head.blocks.0.attn.q_proj.weight",
        global_weight_decay=0.01,
        flow_weight_decay=0.03,
    ) == 0.03
    assert weight_decay_for_parameter(
        "image_flow_head.final_layer.linear.weight",
        global_weight_decay=0.01,
        flow_weight_decay=0.03,
    ) == 0.03


def test_flow_bias_norm_and_input_projectors_remain_decay_free():
    for name in (
        "image_flow_head.blocks.0.attn.q_proj.bias",
        "image_flow_head.blocks.0.norm.weight",
        "model.image_token_embedder.z_proj.weight",
        "image_flow_condition_proj.weight",
    ):
        assert weight_decay_for_parameter(
            name,
            global_weight_decay=0.01,
            flow_weight_decay=0.03,
        ) == 0.0


def test_backbone_matrix_uses_global_weight_decay():
    assert weight_decay_for_parameter(
        "model.layers.0.self_attn.q_proj.weight",
        global_weight_decay=0.01,
        flow_weight_decay=0.03,
    ) == 0.01


def test_image_flow_head_scope_freezes_qwen_and_keeps_condition_projection():
    model = _tiny_model()
    config = OmegaConf.create(
        {"training": {"trainable_scope": "image_flow_head"}}
    )
    summary = _apply_trainable_scope(model, config)

    assert summary["scope"] == "image_flow_head"
    assert summary["trainable_numel"] > 0
    assert summary["frozen_numel"] > 0
    assert model.lm_head.weight is model.model.embed_tokens.weight
    assert not model.model.embed_tokens.weight.requires_grad
    for name, parameter in model.named_parameters():
        expected = name.startswith(
            ("image_flow_head.", "image_flow_condition_proj.")
        )
        assert parameter.requires_grad is expected, name


def test_full_scope_restores_all_parameters_to_trainable():
    model = _tiny_model()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    config = OmegaConf.create({"training": {"trainable_scope": "full"}})
    summary = _apply_trainable_scope(model, config)

    assert summary["scope"] == "full"
    assert summary["frozen_numel"] == 0
    assert all(parameter.requires_grad for parameter in model.parameters())
