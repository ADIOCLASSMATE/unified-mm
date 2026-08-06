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


def _tiny_model() -> Qwen3ForCausalLM:
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
    config.image_flow_num_sampling_steps = "1"
    config.image_flow_batch_mul = 1
    config.image_flow_time_scale = 1000.0
    config.image_flow_time_sampling = "uniform"
    config.image_flow_time_eps = 1.0e-4
    config.image_flow_time_uniform_mix = 0.0
    config.image_flow_solver = "euler"
    config.image_uncond_prob = 0.0
    config.use_flex_attention = False
    return Qwen3ForCausalLM(config).eval()


@torch.no_grad()
def test_backbone_static_kv_cache_matches_full_recompute_with_cfg(monkeypatch):
    monkeypatch.setattr(
        selfless_flow,
        "dynamic_flex_attention",
        _eager_flex_attention,
    )
    torch.manual_seed(11)
    model = _tiny_model()
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
    }

    full, full_trace = model.sample_image_latents_single_stream(
        **kwargs,
        use_backbone_cache=False,
    )
    cached, cached_trace = model.sample_image_latents_single_stream(
        **kwargs,
        use_backbone_cache=True,
    )

    torch.testing.assert_close(cached, full, rtol=0.0, atol=0.0)
    assert full_trace["backbone_kv_cache_enabled"] is False
    assert cached_trace["backbone_kv_cache_enabled"] is True
    assert cached_trace["backbone_kv_cache_context_tokens"] == 4
    # The final generated token has no future query and is intentionally left
    # pending instead of paying for a useless terminal cache write.
    assert cached_trace["backbone_kv_cache_tokens_committed"] == 3
    assert cached_trace["backbone_kv_cache_peak_bytes"] > 0


@torch.no_grad()
def test_backbone_cache_falls_back_for_hidden_candidate_scoring(monkeypatch):
    monkeypatch.setattr(
        selfless_flow,
        "dynamic_flex_attention",
        _eager_flex_attention,
    )
    torch.manual_seed(23)
    model = _tiny_model()
    _, trace = model.sample_image_latents_single_stream(
        input_ids=torch.tensor([[3, 11, 8, 8, 8, 8, 12, 9]]),
        token_types=torch.tensor(
            [[0, 2, 1, 1, 1, 1, 2, 0]], dtype=torch.uint8
        ),
        sigma=torch.tensor([[0.0, 1.0, 4.0, 5.0, 6.0, 7.0, 2.0, 3.0]]),
        spans=[(0, 2, 6)],
        initial_noise_bank=torch.zeros(1, 4, 4),
        flow_cfg=1.0,
        flow_solver="euler",
        flow_num_steps=1,
        parallel_rate=1,
        order_strategy="hidden_norm",
        use_backbone_cache=True,
        return_trace=True,
    )

    assert trace["backbone_kv_cache_enabled"] is False
    assert "full-sequence candidate scoring" in trace[
        "backbone_kv_cache_fallback_reason"
    ]
