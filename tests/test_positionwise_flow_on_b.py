import inspect
import os

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import pytest
import torch

from models.modeling_model.image_flow_loss import FlowLoss
from models.modeling_model.image_flow_loss_positionwise import PositionwiseFlowLoss
from models.modeling_model.modeling_selfless_flow import (
    Qwen3ForCausalLM as BaselineQwen3ForCausalLM,
)
from models.modeling_model.modeling_selfless_flow_positionwise_on_b import (
    POSITIONWISE_ON_B_FLOW_HEAD_ATTENTION_CONTRACT,
    PositionwiseFlowOnBQwen3ForCausalLM,
    SelflessFlowPositionwiseOnBConfig,
    contextual_flow_head_parameter_count,
)
from models.modeling_model.modeling_selfless_flow_positionwise_on_b_generation import (
    PositionwiseFlowOnBGenerationMixin,
)
from utils.utils import get_selfless_mask


def _tiny_config():
    config = SelflessFlowPositionwiseOnBConfig(
        vocab_size=40,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=32,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=9,
        tie_word_embeddings=True,
    )
    values = {
        "architecture_variant": "positionwise_flow_head_on_b",
        "training_objective": "selfless_dual_stream",
        "dual_stream_attention_contract": "xlnet_content_diagonal",
        "flow_head_attention_contract": (
            POSITIONWISE_ON_B_FLOW_HEAD_ATTENTION_CONTRACT
        ),
        "mask_token_id": 7,
        "image_mask_token_id": 8,
        "boi_token_id": 11,
        "eoi_token_id": 12,
        "image_latent_dim": 4,
        "image_tokens_per_img": 4,
        "image_flow_width": 32,
        "image_flow_depth": 1,
        "image_flow_num_sampling_steps": "2",
        "image_flow_batch_mul": 4,
        "image_flow_time_sampling": "uniform",
        "image_flow_time_uniform_mix": 0.0,
        "image_flow_solver": "euler",
        "image_input_noise_strength": 0.0,
        "lambda_text": 0.05,
        "lambda_image": 1.0,
        "use_flex_attention": True,
        "use_cache": False,
    }
    for key, value in values.items():
        setattr(config, key, value)
    return config


def _image_batch():
    input_ids = torch.tensor([[3, 11, 8, 8, 8, 8, 12, 9]])
    token_types = torch.tensor(
        [[0, 2, 1, 1, 1, 1, 2, 0]], dtype=torch.uint8
    )
    sigma = torch.tensor([[0, 1, 3, 4, 5, 6, 2, 7]])
    image_latents = torch.zeros(1, 8, 4)
    image_latents[:, 2:6] = torch.randn(1, 4, 4)
    return input_ids, token_types, sigma, image_latents


def test_f_isolated_model_and_generation_files_do_not_modify_baseline():
    model_source = inspect.getsourcefile(PositionwiseFlowOnBQwen3ForCausalLM)
    generation_source = inspect.getsourcefile(PositionwiseFlowOnBGenerationMixin)
    baseline_source = inspect.getsourcefile(BaselineQwen3ForCausalLM)
    assert model_source and model_source.endswith(
        "modeling_selfless_flow_positionwise_on_b.py"
    )
    assert generation_source and generation_source.endswith(
        "modeling_selfless_flow_positionwise_on_b_generation.py"
    )
    assert baseline_source and baseline_source.endswith("modeling_selfless_flow.py")
    assert "positionwise_flow_head_on_b" not in open(
        baseline_source, encoding="utf-8"
    ).read()


def test_formal_f_head_is_parameter_matched_to_b_with_ascend_aligned_width():
    reference = contextual_flow_head_parameter_count(
        latent_dim=16,
        condition_dim=1024,
        width=1280,
        depth=8,
    )
    # Exact count of PositionwiseFlowLoss at D=16, C=1024, W=1936, depth=8.
    positionwise = (
        (3 + 5 * 8) * 1936**2
        + (1024 + 2 * 16 + 262 + 7 * 8) * 1936
        + 16
    )
    assert 1936 % 16 == 0
    assert reference == 164_072_976
    assert positionwise == 163_828_208
    assert abs(positionwise - reference) / reference < 0.005


def test_f_constructs_only_positionwise_head_and_retains_mul4():
    model = PositionwiseFlowOnBQwen3ForCausalLM(_tiny_config())
    assert model.config.model_type == "selfless_flow_positionwise_on_b"
    assert isinstance(model.image_flow_head, PositionwiseFlowLoss)
    assert not isinstance(model.image_flow_head, FlowLoss)
    assert model.image_flow_batch_mul == 4
    assert model.image_flow_head.net.grad_checkpointing is False


def test_f_rejects_contextual_flow_head_attention_contract():
    config = _tiny_config()
    config.flow_head_attention_contract = "xlnet_content_diagonal"
    with pytest.raises(ValueError, match="not_applicable"):
        PositionwiseFlowOnBQwen3ForCausalLM(config)


def test_legacy_f_without_attention_field_keeps_positionwise_numerics():
    explicit_config = _tiny_config()
    legacy_config = _tiny_config()
    del legacy_config.flow_head_attention_contract

    torch.manual_seed(29)
    explicit = PositionwiseFlowOnBQwen3ForCausalLM(explicit_config).eval()
    torch.manual_seed(29)
    legacy = PositionwiseFlowOnBQwen3ForCausalLM(legacy_config).eval()

    assert legacy.state_dict().keys() == explicit.state_dict().keys()
    for name, value in explicit.state_dict().items():
        torch.testing.assert_close(
            value,
            legacy.state_dict()[name],
            rtol=0,
            atol=0,
        )


def test_f_complete_multimodal_forward_backward():
    torch.manual_seed(7)
    model = PositionwiseFlowOnBQwen3ForCausalLM(_tiny_config()).train()
    input_ids, token_types, sigma, image_latents = _image_batch()
    labels = input_ids.clone()
    labels[:, 2:6] = -100
    output = model(
        X0_input_ids=input_ids,
        labels=labels,
        attention_mask=get_selfless_mask(sigma, 8, "cpu"),
        token_types=token_types,
        image_latents=image_latents,
        image_latent_mask=token_types.eq(1),
        image_span_table=torch.tensor([[0, 0, 2, 6, 4]]),
        image_loss_mask=token_types.eq(1),
        flow_sigma=sigma.float(),
        compute_text_loss=True,
        compute_image_loss=True,
        return_logits=False,
    )
    assert torch.isfinite(output.loss)
    assert output.per_modality_count["image_tokens"].item() == 16
    output.loss.backward()
    assert model.image_flow_head.net.input_proj.weight.grad is not None
    assert model.model.layers[0].self_attn.q_proj.weight.grad is not None


def test_f_generation_uses_b_state_machine_but_reports_isolated_head():
    model = PositionwiseFlowOnBQwen3ForCausalLM(_tiny_config()).eval()
    input_ids, token_types, sigma, _ = _image_batch()
    generated, trace = model.generate_image(
        input_ids=input_ids,
        token_types=token_types,
        sigma=sigma,
        spans=[(0, 2, 6)],
        initial_noise_bank=torch.randn(1, 4, 4),
        flow_cfg=1.0,
        flow_solver="euler",
        flow_num_steps=1,
        order_strategy="spatial_halton",
        use_cache=True,
        return_trace=True,
        _debug_max_generation_steps=2,
    )
    assert generated.shape == (1, 4, 2, 2)
    assert trace["architecture_variant"] == "positionwise_flow_head_on_b"
    assert trace["flow_head_architecture"] == "positionwise_adaln_mlp"
    assert trace["flow_head_attention_contract"] == "not_applicable"
    assert trace["flow_head_content_stream"] is False
    assert trace["flow_head_consumes_prior_latents"] is False
    assert trace["flow_content_cache_peak_bytes_per_sample"] == 0


def test_f_checkpoint_roundtrip_preserves_distinct_model_type(tmp_path):
    model = PositionwiseFlowOnBQwen3ForCausalLM(_tiny_config()).eval()
    model.save_pretrained(tmp_path, safe_serialization=True)
    loaded = PositionwiseFlowOnBQwen3ForCausalLM.from_pretrained(tmp_path).eval()
    assert loaded.config.model_type == "selfless_flow_positionwise_on_b"
    assert loaded.config.architecture_variant == "positionwise_flow_head_on_b"
    assert loaded.state_dict().keys() == model.state_dict().keys()
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, loaded.state_dict()[name], rtol=0, atol=0)
