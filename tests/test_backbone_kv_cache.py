import inspect

import pytest
import torch
from torch.nn.attention.flex_attention import flex_attention
from transformers import Qwen3Config

from models.modeling_model import modeling_selfless_flow as selfless_flow
from models.modeling_model.modeling_selfless_flow import Qwen3ForCausalLM


def _eager_flex_attention(
    query,
    key,
    value,
    attention_mask,
    scaling,
    enable_gqa,
):
    return flex_attention(
        query=query,
        key=key,
        value=value,
        block_mask=attention_mask,
        scale=scaling,
        enable_gqa=enable_gqa,
    )


def _tiny_model(
    attention_contract: str = "selfless_strict",
) -> Qwen3ForCausalLM:
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=9,
    )
    config.mask_token_id = 7
    config.image_mask_token_id = 8
    config.boi_token_id = 11
    config.eoi_token_id = 12
    config.image_latent_dim = 4
    config.image_tokens_per_img = 4
    config.image_flow_width = 32
    config.image_flow_depth = 1
    config.image_flow_num_sampling_steps = "10"
    config.image_flow_batch_mul = 1
    config.image_flow_time_scale = 1000.0
    config.image_flow_time_sampling = "uniform"
    config.image_flow_time_eps = 1.0e-4
    config.image_flow_time_uniform_mix = 0.0
    config.image_flow_solver = "euler"
    config.image_uncond_prob = 0.0
    config.use_flex_attention = False
    config.training_objective = "selfless_dual_stream"
    config.dual_stream_attention_contract = attention_contract
    config.flow_head_attention_contract = attention_contract
    return Qwen3ForCausalLM(config).eval()


def test_production_generation_defaults_to_cache_and_current_cfg():
    image_signature = inspect.signature(Qwen3ForCausalLM.generate_image)
    text_signature = inspect.signature(Qwen3ForCausalLM.generate_text)

    assert image_signature.parameters["use_cache"].default is True
    assert image_signature.parameters["flow_cfg"].default == 3.5
    assert image_signature.parameters["flow_cfg_schedule"].default == "constant"
    assert text_signature.parameters["use_cache"].default is True


@torch.no_grad()
@pytest.mark.parametrize(
    "attention_contract",
    ["selfless_strict", "xlnet_content_diagonal"],
)
def test_ab_text_cache_matches_single_stream_full_reference(
    monkeypatch,
    attention_contract,
):
    monkeypatch.setattr(
        selfless_flow,
        "compiled_flex_attention",
        _eager_flex_attention,
    )
    torch.manual_seed(41)
    model = _tiny_model(attention_contract)
    input_ids = torch.tensor([[3, 4, 0], [5, 6, 7]])
    token_types = torch.tensor(
        [[0, 0, 3], [0, 0, 0]],
        dtype=torch.uint8,
    )

    cached, cached_trace = model.generate_text(
        input_ids,
        token_types=token_types,
        max_new_tokens=4,
        temperature=0.0,
        eos_token_id=-1,
        use_cache=True,
        return_trace=True,
    )
    full, full_trace = model.generate_text(
        input_ids,
        token_types=token_types,
        max_new_tokens=4,
        temperature=0.0,
        eos_token_id=-1,
        use_cache=False,
        return_trace=True,
    )

    torch.testing.assert_close(cached, full, rtol=0, atol=0)
    assert cached_trace["attention_contract"] == attention_contract
    assert cached_trace["backbone_kv_cache_enabled"] is True
    assert full_trace["backbone_kv_cache_enabled"] is False


@torch.no_grad()
@pytest.mark.parametrize(
    "attention_contract",
    ["selfless_strict", "xlnet_content_diagonal"],
)
def test_backbone_static_kv_cache_matches_full_recompute_with_cfg(
    monkeypatch,
    attention_contract,
):
    monkeypatch.setattr(
        selfless_flow,
        "compiled_flex_attention",
        _eager_flex_attention,
    )
    torch.manual_seed(11)
    model = _tiny_model(attention_contract)
    input_ids = torch.tensor(
        [
            [3, 11, 8, 8, 8, 8, 12, 9],
            [4, 5, 11, 8, 8, 8, 8, 12],
        ]
    )
    token_types = torch.tensor(
        [
            [0, 2, 1, 1, 1, 1, 2, 0],
            [0, 0, 2, 1, 1, 1, 1, 2],
        ],
        dtype=torch.uint8,
    )
    sigma = torch.tensor(
        [
            [0.0, 1.0, 4.0, 5.0, 6.0, 7.0, 2.0, 3.0],
            [0.0, 1.0, 2.0, 5.0, 6.0, 7.0, 8.0, 3.0],
        ]
    )
    initial_noise = torch.arange(32, dtype=torch.float32).reshape(2, 4, 4) / 17.0
    kwargs = {
        "input_ids": input_ids,
        "token_types": token_types,
        "sigma": sigma,
        "spans": [(0, 2, 6), (1, 3, 7)],
        "initial_noise_bank": initial_noise,
        "flow_temperature": 0.7,
        "flow_cfg": 2.5,
        "flow_cfg_schedule": "constant",
        "flow_solver": "euler",
        "flow_num_steps": 1,
        "parallel_rate": 1,
        "order_strategy": "spatial_halton",
        "return_trace": True,
        "_debug_max_generation_steps": 4,
    }

    full, full_trace = model.generate(
        "t2i",
        **kwargs,
        use_cache=False,
    )
    backbone_calls = 0
    original_forward = model.model.forward

    def counted_forward(*args, **forward_kwargs):
        nonlocal backbone_calls
        backbone_calls += 1
        return original_forward(*args, **forward_kwargs)

    monkeypatch.setattr(model.model, "forward", counted_forward)
    cached, cached_trace = model.generate(
        "t2i",
        **kwargs,
        use_cache=True,
    )

    torch.testing.assert_close(cached, full, rtol=0.0, atol=0.0)
    # Full recompute and cached decoding use different GEMM query shapes, so
    # their hidden states are numerically equivalent rather than bitwise. The
    # same-shape single/dual-stream contract is tested bitwise separately.
    torch.testing.assert_close(
        cached_trace["debug_conditional_backbone_hidden"],
        full_trace["debug_conditional_backbone_hidden"],
        rtol=0.0,
        atol=5.0e-7,
    )
    torch.testing.assert_close(
        cached_trace["debug_unconditional_backbone_hidden"],
        full_trace["debug_unconditional_backbone_hidden"],
        rtol=0.0,
        atol=5.0e-7,
    )
    assert full_trace["single_stream_attention_contract"] == attention_contract
    assert cached_trace["single_stream_attention_contract"] == attention_contract
    assert full_trace["single_stream_content_self_diagonal"] is (
        attention_contract == "xlnet_content_diagonal"
    )
    assert full_trace["backbone_kv_cache_enabled"] is False
    assert cached_trace["backbone_kv_cache_enabled"] is True
    assert cached_trace["backbone_cfg_batched"] is True
    # One context prefill plus one paired CFG call per generated image token.
    assert backbone_calls == 1 + model.config.image_tokens_per_img
    assert cached_trace["backbone_kv_cache_context_tokens"] == 4
    # The final generated token has no future query and is intentionally left
    # pending instead of paying for a useless terminal cache write.
    assert cached_trace["backbone_kv_cache_tokens_committed"] == 3
    assert cached_trace["backbone_kv_cache_peak_bytes"] > 0
    assert cached_trace["flow_content_cache_peak_bytes_per_sample"] > 0
    assert len(
        cached_trace["flow_cfg_content_cache_divergence_by_layer"]
    ) == 1
    assert all(
        torch.isfinite(torch.tensor(value))
        for value in cached_trace[
            "flow_cfg_content_cache_divergence_by_layer"
        ]
    )


@torch.no_grad()
@pytest.mark.parametrize(
    "attention_contract",
    ["selfless_strict", "xlnet_content_diagonal"],
)
def test_generation_pending_content_fusion_matches_sequential_reference(
    monkeypatch,
    attention_contract,
):
    monkeypatch.setattr(
        selfless_flow,
        "compiled_flex_attention",
        _eager_flex_attention,
    )
    torch.manual_seed(73)
    model = _tiny_model(attention_contract)
    torch.manual_seed(79)
    with torch.no_grad():
        for block in model.image_flow_head.net.blocks:
            block.adaLN_modulation[-1].weight.normal_(0.0, 0.1)
            block.adaLN_modulation[-1].bias.normal_(0.0, 0.1)
        final_layer = model.image_flow_head.net.final_layer
        final_layer.adaLN_modulation[-1].weight.normal_(0.0, 0.1)
        final_layer.adaLN_modulation[-1].bias.normal_(0.0, 0.1)
        final_layer.linear.weight.normal_(0.0, 0.1)
        final_layer.linear.bias.normal_(0.0, 0.1)

    net = model.image_flow_head.net
    fused_forward = net.forward_with_pending_content

    def sequential_reference(
        x,
        t,
        c,
        *,
        latent_mixer_cache,
        context_latents,
        context_conditions,
        context_positions,
        condition_embedding=None,
        time_embedding=None,
        query_positions=None,
        query_rope=None,
    ):
        updated_cache = net._append_content_cache_sequential(
            latent_mixer_cache,
            context_latents=context_latents,
            context_conditions=context_conditions,
            context_positions=context_positions,
        )
        latent_mixer_cache.clear()
        latent_mixer_cache.update(updated_cache)
        return net(
            x,
            t,
            c,
            condition_embedding=condition_embedding,
            time_embedding=time_embedding,
            query_positions=query_positions,
            query_rope=query_rope,
            latent_mixer_cache=latent_mixer_cache,
        )

    kwargs = {
        "input_ids": torch.tensor([[3, 11, 8, 8, 8, 8, 12, 9]]),
        "token_types": torch.tensor(
            [[0, 2, 1, 1, 1, 1, 2, 0]],
            dtype=torch.uint8,
        ),
        "sigma": torch.tensor(
            [[0.0, 1.0, 4.0, 5.0, 6.0, 7.0, 2.0, 3.0]]
        ),
        "spans": [(0, 2, 6)],
        "initial_noise_bank": (
            torch.arange(16, dtype=torch.float32).view(1, 4, 4) / 17.0
        ),
        "flow_temperature": 0.7,
        "flow_cfg": 2.5,
        "flow_cfg_schedule": "constant",
        "flow_solver": "heun",
        "flow_num_steps": 3,
        "parallel_rate": 1,
        "order_strategy": "spatial_halton",
        "use_cache": True,
        "return_trace": True,
        "_debug_max_generation_steps": 4,
    }
    monkeypatch.setattr(
        net,
        "forward_with_pending_content",
        sequential_reference,
    )
    sequential, sequential_trace = model.generate("t2i", **kwargs)
    monkeypatch.setattr(net, "forward_with_pending_content", fused_forward)
    fused, fused_trace = model.generate("t2i", **kwargs)

    tolerance = 0.0 if attention_contract == "selfless_strict" else 1e-6
    torch.testing.assert_close(
        fused,
        sequential,
        rtol=tolerance,
        atol=tolerance,
    )
    assert sequential_trace["flow_content_cache_tokens_committed"] == 3
    assert fused_trace["flow_content_cache_tokens_committed"] == 3


@torch.no_grad()
def test_cache_first_generation_rejects_dynamic_candidate_scoring(monkeypatch):
    monkeypatch.setattr(
        selfless_flow,
        "compiled_flex_attention",
        _eager_flex_attention,
    )
    torch.manual_seed(23)
    model = _tiny_model()
    with pytest.raises(ValueError, match="order_strategy must be one of"):
        model.generate(
            "t2i",
            input_ids=torch.tensor([[3, 11, 8, 8, 8, 8, 12, 9]]),
            token_types=torch.tensor(
                [[0, 2, 1, 1, 1, 1, 2, 0]], dtype=torch.uint8
            ),
            sigma=torch.tensor(
                [[0.0, 1.0, 4.0, 5.0, 6.0, 7.0, 2.0, 3.0]]
            ),
            spans=[(0, 2, 6)],
            initial_noise_bank=torch.zeros(1, 4, 4),
            flow_cfg=1.0,
            flow_solver="euler",
            flow_num_steps=1,
            parallel_rate=1,
            order_strategy="hidden_norm",
            use_cache=True,
            return_trace=True,
        )


@torch.no_grad()
def test_backbone_cache_and_full_path_isolate_packed_segments(monkeypatch):
    monkeypatch.setattr(
        selfless_flow,
        "compiled_flex_attention",
        _eager_flex_attention,
    )
    torch.manual_seed(31)
    model = _tiny_model("xlnet_content_diagonal")
    common = {
        "token_types": torch.tensor(
            [[0, 2, 1, 1, 1, 1, 2, 0, 0, 0, 0]],
            dtype=torch.uint8,
        ),
        "sigma": torch.tensor(
            [[0.0, 1.0, 4.0, 5.0, 6.0, 7.0, 2.0, 3.0, 0.0, 1.0, 2.0]]
        ),
        "segment_ids": torch.tensor(
            [[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]],
            dtype=torch.long,
        ),
        "spans": [(0, 2, 6)],
        "initial_noise_bank": torch.arange(16, dtype=torch.float32).view(1, 4, 4)
        / 13.0,
        "flow_cfg": 1.0,
        "flow_solver": "euler",
        "flow_num_steps": 1,
        "parallel_rate": 1,
        "order_strategy": "spatial_halton",
        "return_trace": True,
    }

    outputs = []
    for distractor in ([20, 21, 22], [29, 30, 31]):
        input_ids = torch.tensor(
            [[3, 11, 8, 8, 8, 8, 12, 9, *distractor]]
        )
        full, full_trace = model.generate(
            "t2i",
            input_ids=input_ids,
            **common,
            use_cache=False,
        )
        cached, cached_trace = model.generate(
            "t2i",
            input_ids=input_ids,
            **common,
            use_cache=True,
        )
        torch.testing.assert_close(cached, full, rtol=0.0, atol=0.0)
        assert full_trace["segment_isolation_enabled"] is True
        assert cached_trace["segment_isolation_enabled"] is True
        outputs.append(cached)

    torch.testing.assert_close(outputs[0], outputs[1], rtol=0.0, atol=0.0)
