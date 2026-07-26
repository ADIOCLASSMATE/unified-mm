import hashlib
import types
from unittest.mock import patch

import pytest
import torch

from models.modeling_model.image_flow_loss import FlowLoss
from models.modeling_model.image_flow_position import FLOW_HEAD_POSITION_SPECS
from models.modeling_model.modeling_selfless_flow import Qwen3ForCausalLM
from pretrain.train_selfless_flow import (
    _validation_flat_query_mixer_context,
    _validation_sequence_mixer_context,
)

BASELINE_CELLS = (("DF1", "FH0"), ("DF1", "FH4"))


def _flow(variant: str, position: str = "FH0") -> FlowLoss:
    spec = FLOW_HEAD_POSITION_SPECS[position]
    torch.manual_seed(314159)
    flow = FlowLoss(
        target_channels=4,
        z_channels=8,
        depth=3,
        width=16,
        num_sampling_steps=2,
        mlp_ratio=1.0,
        image_tokens_per_img=4,
        latent_mixer_heads=4,
        position_variant=position,
        query_position_mode=spec.query_position_mode,
        context_position_mode=spec.context_position_mode,
        rope_mode=spec.rope_mode,
        rope_axis_dims=(2, 2),
        flow_head_variant=variant,
    )
    with torch.no_grad():
        for block in flow.net.blocks:
            block.adaLN_modulation[-1].weight.normal_(0.0, 0.1)
            block.adaLN_modulation[-1].bias.normal_(0.0, 0.1)
        flow.net.final_layer.adaLN_modulation[-1].weight.normal_(0.0, 0.1)
        flow.net.final_layer.adaLN_modulation[-1].bias.normal_(0.0, 0.1)
        flow.net.final_layer.linear.weight.normal_(0.0, 0.1)
        flow.net.final_layer.linear.bias.normal_(0.0, 0.1)
    return flow


def _inputs():
    torch.manual_seed(2718)
    content = torch.randn(1, 4, 4)
    query = torch.randn(1, 4, 4)
    condition = torch.randn(1, 4, 8)
    time = torch.tensor([[0.2, 0.7, 0.4, 0.9]])
    positions = torch.tensor([[2, 0, 3, 1]])
    sigma = torch.arange(4, dtype=torch.float32).view(1, 4)
    strict_mask = sigma.unsqueeze(1) < sigma.unsqueeze(2)
    return content, query, condition, time, positions, sigma, strict_mask


def _parameter_schema(module):
    return tuple(
        (name, tuple(parameter.shape), str(parameter.dtype))
        for name, parameter in module.named_parameters()
    )


def _parameter_hash(module):
    digest = hashlib.sha256()
    for name, parameter in module.named_parameters():
        digest.update(name.encode("utf-8"))
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def test_df1_fh0_is_the_default_and_removed_architectures_are_rejected():
    torch.manual_seed(19)
    default = FlowLoss(
        target_channels=4,
        z_channels=8,
        depth=2,
        width=16,
        num_sampling_steps=2,
        image_tokens_per_img=4,
        latent_mixer_heads=4,
    )
    torch.manual_seed(19)
    explicit = FlowLoss(
        target_channels=4,
        z_channels=8,
        depth=2,
        width=16,
        num_sampling_steps=2,
        image_tokens_per_img=4,
        latent_mixer_heads=4,
        flow_head_variant="DF1",
    )
    assert default.state_dict().keys() == explicit.state_dict().keys()
    assert all(
        torch.equal(default.state_dict()[name], explicit.state_dict()[name])
        for name in default.state_dict()
    )
    assert default.net.cache_contract()["flow_head_variant"] == "DF1"
    for removed in ("DF0", "DF2"):
        with pytest.raises(ValueError, match="only supports DF1"):
            FlowLoss(
                target_channels=4,
                z_channels=8,
                depth=2,
                width=16,
                num_sampling_steps=2,
                image_tokens_per_img=4,
                latent_mixer_heads=4,
                flow_head_variant=removed,
            )


def test_df1_baselines_share_parameter_schema_count_and_initial_state():
    flows = {}
    for position in ("FH0", "FH4"):
        torch.manual_seed(42)
        flows[position] = _flow("DF1", position)
    assert len({_parameter_schema(flow) for flow in flows.values()}) == 1
    assert len(
        {sum(parameter.numel() for parameter in flow.parameters()) for flow in flows.values()}
    ) == 1
    assert len({_parameter_hash(flow) for flow in flows.values()}) == 1
    assert not any(
        "content" in name.lower()
        for name, _ in flows["FH0"].named_parameters()
    )


@pytest.mark.parametrize("variant,position", BASELINE_CELLS)
def test_baseline_position_contracts_are_exact_and_parameter_matched(
    variant, position
):
    flow = _flow(variant, position)
    contract = flow.net.position_contract()
    expected = {
        "FH0": (1, 1, 0),
        "FH4": (0, 0, 1),
    }[position]
    assert (contract["A_q"], contract["A_c"], contract["R_f"]) == expected
    reference = _flow("DF1", "FH0")
    assert _parameter_schema(flow) == _parameter_schema(reference)


def test_fh4_rope_cache_rotates_k_once_and_never_rotates_v():
    flow = _flow("DF1", "FH4").eval()
    content, _, condition, _, positions, _, _ = _inputs()
    block = flow.net.blocks[0]
    hidden = flow.net._initial_content_hidden(content[:, :1], positions[:, :1])
    normalized = block.cross_kv_norm(hidden)
    unrotated_k = block._split_heads(block.cross_k(normalized))
    expected_v = block._split_heads(block.cross_v(normalized))
    cache = flow.append_latent_mixer_cache(
        flow.empty_latent_mixer_cache(),
        context_latents=content[:, 0],
        context_conditions=condition[:, 0],
        context_positions=positions[:, 0],
    )
    layer = cache["layers"][0]
    assert layer["k_rotation_count"] == 1
    assert not torch.equal(layer["k"], unrotated_k)
    torch.testing.assert_close(layer["v"], expected_v)


def test_shared_content_noise_is_sampled_once_and_reused():
    sampled = torch.randn(1, 4, 4)

    class _Backbone:
        def __init__(self):
            self.calls = 0

        def _maybe_add_image_input_noise(self, clean):
            self.calls += 1
            assert clean.shape == sampled.shape
            return sampled

    dummy = types.SimpleNamespace(training=True, model=_Backbone())
    clean = torch.zeros_like(sampled)
    token_types = torch.ones(1, 4, dtype=torch.long)
    shared = Qwen3ForCausalLM._shared_noisy_image_latents(
        dummy, clean, token_types
    )
    assert shared is sampled
    assert dummy.model.calls == 1
    # Production forward binds this object to the backbone argument and then
    # slices the same object for the flow-head content stream.
    image_latents_for_model = shared
    context_image_latents_for_loss = shared.to(shared.device)
    assert image_latents_for_model is context_image_latents_for_loss


def test_periodic_validation_supplies_df1_content_conditions():
    torch.manual_seed(7)
    target = torch.randn(4, 4)
    conditions = torch.randn(4, 8)
    sigma = torch.tensor([3.0, 0.0, 2.0, 1.0])
    positions = torch.arange(4)

    flat = _validation_flat_query_mixer_context(
        target,
        sigma,
        positions,
        conditions,
    )
    assert flat["context_conditions"].shape == (4, 4, 8)
    for query_idx in range(4):
        torch.testing.assert_close(
            flat["context_conditions"][query_idx],
            conditions,
        )

    for position in ("FH0", "FH4"):
        sampled = _flow("DF1", position).eval().sample(
            conditions,
            num_steps=1,
            **flat,
        )
        assert sampled.shape == target.shape
        assert torch.isfinite(sampled).all()


def test_periodic_validation_probe_context_preserves_token_conditions():
    torch.manual_seed(8)
    target = torch.randn(4, 4)
    conditions = torch.randn(4, 8)
    sigma = torch.tensor([3.0, 0.0, 2.0, 1.0])
    positions = torch.arange(4)
    sequence = _validation_sequence_mixer_context(
        target,
        sigma,
        positions,
        conditions,
    )
    assert sequence["context_conditions"].shape == (1, 4, 8)
    torch.testing.assert_close(
        sequence["context_conditions"][0],
        conditions,
    )
    velocity = _flow("DF1", "FH4").eval().velocity(
        torch.randn(1, 4, 4),
        torch.full((1, 4), 0.5),
        conditions.unsqueeze(0),
        **sequence,
    )
    assert velocity.shape == (1, 4, 4)
    assert torch.isfinite(velocity).all()


@pytest.mark.parametrize("variant,position", BASELINE_CELLS)
def test_query_only_loss_routes_gradient_through_content_stream(
    variant, position
):
    flow = _flow(variant, position)
    content, query, condition, time, positions, _, strict_mask = _inputs()
    content = content.requires_grad_(True)
    velocity = flow.velocity(
        query,
        time,
        condition,
        context_latents=content,
        context_mask=strict_mask,
        query_positions=positions,
        context_positions=positions,
    )
    query_only_loss = velocity[:, -1].square().mean()
    query_only_loss.backward()
    assert content.grad is not None
    assert torch.count_nonzero(content.grad[:, :-1]) > 0
    assert torch.count_nonzero(content.grad[:, -1]) == 0


@pytest.mark.parametrize("position", ["FH0", "FH4"])
def test_df1_endpoint_equality_holds_after_every_shared_block(position):
    flow = _flow("DF1", position)
    content, _, condition, _, positions, _, strict_mask = _inputs()
    captured = [[] for _ in flow.net.blocks]
    handles = []
    for layer_idx, block in enumerate(flow.net.blocks):
        handles.append(
            block.register_forward_hook(
                lambda _module, _args, output, idx=layer_idx: captured[idx].append(
                    output.detach().clone()
                )
            )
        )
    try:
        flow.velocity(
            content,
            torch.ones(1, 4),
            condition,
            context_latents=content,
            context_mask=strict_mask,
            query_positions=positions,
            context_positions=positions,
        )
    finally:
        for handle in handles:
            handle.remove()
    for layer_outputs in captured:
        assert len(layer_outputs) == 2
        torch.testing.assert_close(
            layer_outputs[0], layer_outputs[1], atol=1e-6, rtol=1e-6
        )


@pytest.mark.parametrize("variant,position", BASELINE_CELLS)
def test_dynamic_stream_is_strictly_causal_and_has_zero_forbidden_gradients(
    variant, position
):
    flow = _flow(variant, position)
    content, query, condition, time, positions, sigma, strict_mask = _inputs()
    content = content.requires_grad_(True)
    output = flow.velocity(
        query,
        time,
        condition,
        context_latents=content,
        context_mask=strict_mask,
        query_positions=positions,
        context_positions=positions,
    )
    query_idx = 2
    future = sigma[0] >= sigma[0, query_idx]
    changed = content.detach().clone()
    changed[0, future] += 100.0
    changed_output = flow.velocity(
        query,
        time,
        condition,
        context_latents=changed,
        context_mask=strict_mask,
        query_positions=positions,
        context_positions=positions,
    )
    torch.testing.assert_close(
        output[:, query_idx],
        changed_output[:, query_idx],
        atol=1e-6,
        rtol=1e-6,
    )
    output[:, query_idx].sum().backward()
    assert content.grad is not None
    assert torch.count_nonzero(content.grad[0, future]) == 0
    assert torch.count_nonzero(content.grad[0, ~future]) > 0


@pytest.mark.parametrize("variant,position", BASELINE_CELLS)
def test_query_times_are_independent_across_positions(variant, position):
    flow = _flow(variant, position)
    content, query, condition, time, positions, _, strict_mask = _inputs()
    baseline = flow.velocity(
        query,
        time,
        condition,
        context_latents=content,
        context_mask=strict_mask,
        query_positions=positions,
        context_positions=positions,
    )
    changed_time = time.clone()
    changed_time[:, 1] = 0.05
    changed = flow.velocity(
        query,
        changed_time,
        condition,
        context_latents=content,
        context_mask=strict_mask,
        query_positions=positions,
        context_positions=positions,
    )
    torch.testing.assert_close(
        baseline[:, [0, 2, 3]],
        changed[:, [0, 2, 3]],
        atol=1e-6,
        rtol=1e-6,
    )


@pytest.mark.parametrize("variant,position", BASELINE_CELLS)
def test_incremental_cache_matches_full_sequence_query(variant, position):
    flow = _flow(variant, position)
    content, query, condition, time, positions, _, strict_mask = _inputs()
    full = flow.velocity(
        query,
        time,
        condition,
        context_latents=content,
        context_mask=strict_mask,
        query_positions=positions,
        context_positions=positions,
    )
    cache = flow.empty_latent_mixer_cache()
    for token_idx in range(3):
        cache = flow.append_latent_mixer_cache(
            cache,
            context_latents=content[:, token_idx],
            context_conditions=condition[:, token_idx],
            context_positions=positions[:, token_idx],
        )
    incremental = flow.velocity(
        query[:, -1],
        time[:, -1],
        condition[:, -1],
        query_positions=positions[:, -1],
        latent_mixer_cache=cache,
    )
    torch.testing.assert_close(
        full[:, -1], incremental, atol=1e-6, rtol=1e-6
    )


@pytest.mark.parametrize("variant,position", BASELINE_CELLS)
def test_empty_and_single_context_caches_are_finite(variant, position):
    flow = _flow(variant, position)
    content, query, condition, time, positions, _, _ = _inputs()
    cache = flow.empty_latent_mixer_cache()
    empty = flow.velocity(
        query[:, 0],
        time[:, 0],
        condition[:, 0],
        query_positions=positions[:, 0],
        latent_mixer_cache=cache,
    )
    cache = flow.append_latent_mixer_cache(
        cache,
        context_latents=content[:, 0],
        context_conditions=condition[:, 0],
        context_positions=positions[:, 0],
    )
    single = flow.velocity(
        query[:, 1],
        time[:, 1],
        condition[:, 1],
        query_positions=positions[:, 1],
        latent_mixer_cache=cache,
    )
    assert torch.isfinite(empty).all()
    assert torch.isfinite(single).all()
    (empty.sum() + single.sum()).backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in flow.parameters()
    )


def test_cfg_uses_distinct_dynamic_content_caches():
    flow = _flow("DF1")
    content, query, condition, time, positions, _, _ = _inputs()
    cond_cache = flow.append_latent_mixer_cache(
        flow.empty_latent_mixer_cache(),
        context_latents=content[:, 0],
        context_conditions=condition[:, 0],
        context_positions=positions[:, 0],
    )
    uncond_cache = flow.append_latent_mixer_cache(
        flow.empty_latent_mixer_cache(),
        context_latents=content[:, 0],
        context_conditions=torch.zeros_like(condition[:, 0]),
        context_positions=positions[:, 0],
    )
    paired_cache = flow.stack_latent_mixer_caches([cond_cache, uncond_cache])
    paired = flow.velocity(
        torch.cat([query[:, 1], query[:, 1]], dim=0),
        torch.cat([time[:, 1], time[:, 1]], dim=0),
        torch.cat([condition[:, 1], torch.zeros_like(condition[:, 1])], dim=0),
        query_positions=torch.cat([positions[:, 1], positions[:, 1]], dim=0),
        latent_mixer_cache=paired_cache,
    )
    separate_cond = flow.velocity(
        query[:, 1],
        time[:, 1],
        condition[:, 1],
        query_positions=positions[:, 1],
        latent_mixer_cache=cond_cache,
    )
    separate_uncond = flow.velocity(
        query[:, 1],
        time[:, 1],
        torch.zeros_like(condition[:, 1]),
        query_positions=positions[:, 1],
        latent_mixer_cache=uncond_cache,
    )
    torch.testing.assert_close(paired[:1], separate_cond, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(paired[1:], separate_uncond, atol=1e-6, rtol=1e-6)
    assert any(
        not torch.equal(cond_layer["k"], uncond_layer["k"])
        for cond_layer, uncond_layer in zip(
            cond_cache["layers"], uncond_cache["layers"]
        )
    )


def test_single_stream_generation_commits_df1_cfg_caches_incrementally():
    flow = _flow("DF1")

    class _Backbone:
        def __call__(self, X0_input_ids, attention_mask, **_kwargs):
            batch, seq_len = X0_input_ids.shape
            hidden = torch.arange(
                batch * seq_len * 8, dtype=torch.float32
            ).view(batch, seq_len, 8)
            hidden = hidden + attention_mask.float().mean(dim=-1, keepdim=True)
            return types.SimpleNamespace(last_hidden_state=hidden)

    dummy = types.SimpleNamespace(
        config=types.SimpleNamespace(image_tokens_per_img=4, boi_token_id=11),
        model=_Backbone(),
        image_flow_head=flow,
        _prepare_image_flow_condition=lambda hidden: hidden,
    )
    input_ids = torch.tensor([[10, 11, 7, 7, 7, 7, 12, 13]])
    token_types = torch.tensor([[0, 2, 1, 1, 1, 1, 2, 2]])
    sigma = torch.tensor([[0, 1, 4, 5, 3, 6, 2, 7]])
    spans = [(0, 2, 6)]

    def _mask(sigma, seq_len, device, **kwargs):
        value = sigma.detach().clone()
        if kwargs.get("image_uncond_rows") is not None:
            value = value + 100
        return value

    with patch("utils.utils.get_selfless_mask", side_effect=_mask):
        generated, trace = Qwen3ForCausalLM.sample_image_latents_single_stream(
            dummy,
            input_ids=input_ids,
            token_types=token_types,
            sigma=sigma,
            spans=spans,
            image_latent_dim=4,
            initial_noise_bank=torch.zeros(1, 4, 4),
            flow_cfg=2.0,
            flow_num_steps=2,
            parallel_rate=1,
            order_strategy="spatial_halton",
            return_trace=True,
        )
    assert generated.shape == (1, 4, 2, 2)
    assert torch.isfinite(generated).all()
    assert trace["flow_head_variant"] == "DF1"
    assert trace["flow_content_cache_peak_bytes_per_sample"] > 0
    assert len(trace["flow_cfg_content_cache_divergence_by_layer"]) == 3
