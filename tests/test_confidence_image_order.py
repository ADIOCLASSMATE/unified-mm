import types

import pytest
import torch

from test_static_x0_content_contract import (
    _config, _randomize_flow_output, _eager_flex_attention, selfless_flow,
    Qwen3ForCausalLM, X0_CONTENT_FLOW_CONDITION_CONTRACT,
)
from utils.image_order_strategies import confidence_scores


def _matrix_model(variant):
    cls = Qwen3ForCausalLM
    attention = "selfless_strict" if variant in {"a_x0", "a_legacy"} else "xlnet_content_diagonal"
    kwargs = {} if variant in {"a_legacy", "b_flowdiag", "b_no_flowdiag"} else {
        "condition_contract": X0_CONTENT_FLOW_CONDITION_CONTRACT}
    config = _config(attention, **kwargs)
    if variant == "b_no_flowdiag":
        config.flow_head_attention_contract = "selfless_strict"
    elif variant == "c_on_b":
        from models.modeling_model.modeling_single_stream_text_ar import SingleStreamTextARQwen3ForCausalLM
        cls = SingleStreamTextARQwen3ForCausalLM
        config.architecture_variant = "single_stream_text_ar"
    elif variant == "d_on_b":
        from test_dynamic_xt_contract import tiny_config, DynamicXtQwen3ForCausalLM
        config, cls = tiny_config(), DynamicXtQwen3ForCausalLM
    elif variant == "f_on_b":
        from test_positionwise_flow_on_b import _tiny_config, PositionwiseFlowOnBQwen3ForCausalLM
        config, cls = _tiny_config(), PositionwiseFlowOnBQwen3ForCausalLM
    config.image_tokens_per_img = 36
    config.max_position_embeddings = 64
    model = cls(config).eval()
    if variant == "f_on_b":
        for block in [*model.image_flow_head.net.res_blocks, model.image_flow_head.net.final_layer]:
            torch.nn.init.normal_(block.adaLN_modulation[-1].weight, std=.1)
            torch.nn.init.normal_(block.adaLN_modulation[-1].bias, std=.1)
        torch.nn.init.normal_(model.image_flow_head.net.final_layer.linear.weight, std=.1)
    else:
        _randomize_flow_output(model)
    return model


def test_proxy_scores_have_expected_direction_and_scale_invariance():
    vc = torch.tensor([[[2., 2.], [1., 1.]]])
    vu = torch.tensor([[[2., 2.], [0., 0.]]])
    score = confidence_scores("confidence_cfg", vc, vu, cfg=2)
    assert score[0, 0] == 0 and score[0, 1] > 0
    torch.testing.assert_close(score, confidence_scores("confidence_cfg", vc * 10, vu * 10, cfg=2))
    stable = confidence_scores("confidence_stability", vc, vu, cfg=2, next_guided_velocity=vu + 2 * (vc - vu))
    assert torch.count_nonzero(stable) == 0


@torch.no_grad()
@pytest.mark.parametrize("variant,strategy", [
    ("b_x0", s) for s in ["confidence_cfg", "confidence_cfg_reverse", "confidence_stability", "confidence_halton"]
] + [(v, s) for v in ["a_x0", "a_legacy", "b_flowdiag", "b_no_flowdiag", "c_on_b", "d_on_b", "f_on_b"]
     for s in ["confidence_stability", "confidence_halton"]])
def test_confidence_probes_match_fresh_decode_in_the_recorded_order(monkeypatch, variant, strategy):
    # 36 positions cross two refresh boundaries; different prompts exercise
    # row-specific masks and permutations. The independent decode never probes.
    monkeypatch.setattr(selfless_flow, "compiled_flex_attention", _eager_flex_attention)
    from models.modeling_model import modeling_selfless_flow_dynamic_xt as dynamic
    monkeypatch.setattr(dynamic, "compiled_flex_attention", _eager_flex_attention)
    torch.manual_seed(401)
    model = _matrix_model(variant)
    tokens = torch.tensor([[3, 11] + [8] * 36, [4, 11] + [8] * 36])
    types_ = torch.tensor([[0, 2] + [1] * 36] * 2, dtype=torch.uint8)
    sigma = torch.arange(38).float().repeat(2, 1)
    payload = dict(input_ids=tokens, token_types=types_, sigma=sigma,
                   spans=[(0, 2, 38), (1, 2, 38)], initial_noise_bank=torch.randn(2, 36, 4),
                   flow_cfg=2., flow_solver="heun", flow_num_steps=2, return_trace=True)
    rng = torch.get_rng_state().clone()
    actual, trace = model.generate_image(**payload, order_strategy=strategy)
    assert torch.equal(torch.get_rng_state(), rng), "probes must not consume the decoding RNG"
    ranks = trace["generation_order"].flatten(1)
    torch.testing.assert_close(ranks.sort(1).values, torch.arange(1, 37).repeat(2, 1))
    assert torch.isfinite(trace["order_confidence_proxy"]).all()
    if variant == "d_on_b":
        # Two true Dynamic-XT velocity probes per block, plus the usual
        # two Heun evaluations per solver step for every final reveal.
        assert trace["dynamic_xt_conditional_velocity_evaluations"] == 36 * 2 * 2 + 3 * 2
        assert trace["dynamic_xt_unconditional_velocity_evaluations"] == 36 * 2 * 2 + 3 * 2
        assert trace["dynamic_xt_flow_content_condition_commits"] == 35
    if variant == "f_on_b":
        assert trace["flow_content_cache_peak_bytes_per_sample"] == 0
    recorded = ranks.argsort(1)
    if strategy == "confidence_halton":
        expected = model._halton_image_order(36, 6, torch.device("cpu"))
        torch.testing.assert_close(recorded, expected.repeat(2, 1))
    original = model._image_generation_orders
    model._image_generation_orders = types.MethodType(lambda self, **kw: ("spatial_halton", recorded.clone(), False), model)
    reference, _ = model.generate_image(**payload)
    torch.testing.assert_close(actual, reference, rtol=2e-5, atol=2e-5)
    model._image_generation_orders = original
    # No training-sigma reveal-order information may enter the new policies.
    changed = sigma.clone()
    changed[:, 2:] = changed[:, 2:].flip(1)
    second, second_trace = model.generate_image(**{**payload, "sigma": changed}, order_strategy=strategy)
    torch.testing.assert_close(actual, second, rtol=0, atol=0)
    torch.testing.assert_close(ranks, second_trace["generation_order"].flatten(1))
