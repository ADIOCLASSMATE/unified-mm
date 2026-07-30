from __future__ import annotations

import pytest
import torch
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

import models.modeling_model.modeling_selfless_flow as modeling
from models.modeling_model.modeling_selfless_flow import (
    Qwen3Attention,
    Qwen3DecoderLayer,
)


def _config(gate: str = "none") -> Qwen3Config:
    config = Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=8,
        max_position_embeddings=128,
        attention_dropout=0.0,
        attention_bias=False,
    )
    config.backbone_attention_output_gate = gate
    return config


def _fake_flex_attention(
    query,
    key,
    value,
    attention_mask,
    scaling,
    enable_gqa,
):
    del attention_mask, scaling, enable_gqa
    return query + key + value


def _attention_inputs():
    generator = torch.Generator().manual_seed(101)
    x0 = torch.randn(2, 6, 32, generator=generator)
    xt = torch.randn(2, 6, 32, generator=generator)
    cos = torch.ones(2, 6, 8)
    sin = torch.zeros(2, 6, 8)
    token_types = torch.tensor(
        [
            [0, 2, 1, 1, 1, 1],
            [0, 2, 1, 1, 1, 3],
        ]
    )
    sigma = torch.tensor(
        [
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            [0.0, 1.0, 5.0, 4.0, 3.0, 6.0],
        ]
    )
    return x0, xt, (cos, sin), token_types, sigma


def test_default_mode_has_no_gate_parameters():
    attention = Qwen3Attention(_config("none"), layer_idx=0)
    assert attention.attn_output_gate_proj is None
    assert not any(
        "attn_output_gate_proj" in name
        for name, _ in attention.named_parameters()
    )


def test_gate_construction_preserves_all_preexisting_parameter_initialization():
    torch.manual_seed(43)
    baseline = Qwen3DecoderLayer(_config("none"), layer_idx=0)
    torch.manual_seed(43)
    gated = Qwen3DecoderLayer(
        _config("per_head_identity_sigmoid"),
        layer_idx=0,
    )

    baseline_state = baseline.state_dict()
    gated_state = gated.state_dict()
    for name, value in baseline_state.items():
        torch.testing.assert_close(
            gated_state[name],
            value,
            rtol=0,
            atol=0,
        )
    gate_weight = gated_state["self_attn.attn_output_gate_proj.weight"]
    assert torch.count_nonzero(gate_weight).item() == 0


def test_identity_gate_preserves_forward_and_old_parameter_gradients(
    monkeypatch,
):
    monkeypatch.setattr(
        modeling,
        "compiled_flex_attention",
        _fake_flex_attention,
    )
    torch.manual_seed(7)
    baseline = Qwen3Attention(_config("none"), layer_idx=0)
    gated = Qwen3Attention(
        _config("per_head_identity_sigmoid"),
        layer_idx=0,
    )
    missing, unexpected = gated.load_state_dict(
        baseline.state_dict(),
        strict=False,
    )
    assert missing == ["attn_output_gate_proj.weight"]
    assert unexpected == []
    gated.reset_attention_output_gate()

    x0, xt, position_embeddings, token_types, sigma = _attention_inputs()
    x0_baseline = x0.clone().requires_grad_(True)
    xt_baseline = xt.clone().requires_grad_(True)
    x0_gated = x0.clone().requires_grad_(True)
    xt_gated = xt.clone().requires_grad_(True)
    mask_sentinel = object()

    baseline_output = baseline(
        X0_hidden_states=x0_baseline,
        XT_hidden_states=xt_baseline,
        position_embeddings=position_embeddings,
        attention_mask=mask_sentinel,
        token_types=token_types,
        flow_sigma=sigma,
    )
    gated_output = gated(
        X0_hidden_states=x0_gated,
        XT_hidden_states=xt_gated,
        position_embeddings=position_embeddings,
        attention_mask=mask_sentinel,
        token_types=token_types,
        flow_sigma=sigma,
        record_backbone_gate_stats=True,
        backbone_gate_stats_level="detailed",
    )

    for expected, actual in zip(baseline_output[:2], gated_output[:2]):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    baseline_loss = sum(value.square().mean() for value in baseline_output[:2])
    gated_loss = sum(value.square().mean() for value in gated_output[:2])
    baseline_loss.backward()
    gated_loss.backward()

    baseline_parameters = dict(baseline.named_parameters())
    gated_parameters = dict(gated.named_parameters())
    for name, parameter in baseline_parameters.items():
        assert parameter.grad is not None
        assert gated_parameters[name].grad is not None
        torch.testing.assert_close(
            gated_parameters[name].grad,
            parameter.grad,
            rtol=0,
            atol=0,
        )

    gate_gradient = gated.attn_output_gate_proj.weight.grad
    assert gate_gradient is not None
    assert torch.isfinite(gate_gradient).all()
    assert torch.count_nonzero(gate_gradient).item() > 0
    assert gated.last_gate_stats["x0/mean"].item() == 1.0
    assert gated.last_gate_stats["xt/mean"].item() == 1.0
    assert gated.last_gate_stats["x0/mean_abs_delta"].item() == 0.0
    assert "x0/image_order_q1/mean" in gated.last_gate_stats
    assert "xt/image_order_q4/mean" in gated.last_gate_stats


def test_gate_parameter_count_matches_ci10_proposal():
    config = _config("per_head_identity_sigmoid")
    attention = Qwen3Attention(config, layer_idx=0)
    per_layer = sum(
        parameter.numel()
        for name, parameter in attention.named_parameters()
        if "attn_output_gate_proj" in name
    )
    assert per_layer == config.hidden_size * config.num_attention_heads
    assert 28 * 1024 * 16 == 458_752


def test_gate_contract_rejects_unknown_mode():
    with pytest.raises(
        ValueError,
        match="backbone_attention_output_gate",
    ):
        Qwen3Attention(_config("per_channel"), layer_idx=0)


def test_gate_stats_reject_misaligned_sigma(monkeypatch):
    monkeypatch.setattr(
        modeling,
        "compiled_flex_attention",
        _fake_flex_attention,
    )
    attention = Qwen3Attention(
        _config("per_head_identity_sigmoid"),
        layer_idx=0,
    )
    x0, xt, position_embeddings, token_types, _ = _attention_inputs()
    with pytest.raises(ValueError, match="flow_sigma must align"):
        attention(
            X0_hidden_states=x0,
            XT_hidden_states=xt,
            position_embeddings=position_embeddings,
            attention_mask=object(),
            token_types=token_types,
            flow_sigma=torch.zeros(2, 5),
            record_backbone_gate_stats=True,
            backbone_gate_stats_level="detailed",
        )
