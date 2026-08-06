import inspect
import types

import torch

from models.modeling_model.image_flow_loss import FlowLoss


def _flow(*, grad_checkpointing: bool = False) -> FlowLoss:
    torch.manual_seed(314159)
    flow = FlowLoss(
        target_channels=4,
        z_channels=8,
        depth=3,
        width=32,
        num_sampling_steps=2,
        grad_checkpointing=grad_checkpointing,
        image_tokens_per_img=4,
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


def test_flow_loss_exposes_no_architecture_ablation_arguments():
    parameters = inspect.signature(FlowLoss).parameters
    for retired in (
        "head_arch",
        "flow_head_variant",
        "position_variant",
        "query_position_mode",
        "context_position_mode",
        "rope_mode",
        "rope_axis_dims",
    ):
        assert retired not in parameters


def test_position_and_cache_contracts_are_fixed():
    flow = _flow()
    assert flow.net.position_contract() == {
        "schema": "selfless_flow_head_pure_2d_v1",
        "architecture": "dynamic_dual_stream",
        "additive_image_position": False,
        "rope_mode": "row_col_2d",
        "rope_axis_dims": [2, 2],
        "rotate_value": False,
    }
    assert flow.net.cache_contract() == {
        "schema": "selfless_flow_head_content_cache_v1",
        "content_update": "shared_attention_mlp",
        "strict_context": True,
        "query_writes_cache": False,
        "position_contract": flow.net.position_contract(),
    }


def test_dynamic_content_stream_is_strictly_causal():
    flow = _flow()
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
        context_conditions=condition,
    )

    query_index = 2
    future = sigma[0] >= sigma[0, query_index]
    changed_content = content.detach().clone()
    changed_content[0, future] += 100.0
    changed_output = flow.velocity(
        query,
        time,
        condition,
        context_latents=changed_content,
        context_mask=strict_mask,
        query_positions=positions,
        context_positions=positions,
    )
    torch.testing.assert_close(
        output[:, query_index],
        changed_output[:, query_index],
        atol=1e-6,
        rtol=1e-6,
    )

    output[:, query_index].sum().backward()
    assert content.grad is not None
    assert torch.count_nonzero(content.grad[0, future]) == 0
    assert torch.count_nonzero(content.grad[0, ~future]) > 0


def test_incremental_cache_matches_full_sequence_last_query():
    flow = _flow()
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
    for token_index in range(3):
        cache = flow.append_latent_mixer_cache(
            cache,
            context_latents=content[:, token_index],
            context_conditions=condition[:, token_index],
            context_positions=positions[:, token_index],
        )
    incremental = flow.velocity(
        query[:, -1],
        time[:, -1],
        condition[:, -1],
        query_positions=positions[:, -1],
        latent_mixer_cache=cache,
    )
    torch.testing.assert_close(
        full[:, -1],
        incremental,
        atol=1e-6,
        rtol=1e-6,
    )


def test_empty_and_single_content_cache_are_finite():
    flow = _flow()
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


def test_stacked_cfg_cache_matches_separate_branches():
    flow = _flow()
    content, query, condition, time, positions, _, _ = _inputs()
    conditional = flow.append_latent_mixer_cache(
        flow.empty_latent_mixer_cache(),
        context_latents=content[:, 0],
        context_conditions=condition[:, 0],
        context_positions=positions[:, 0],
    )
    unconditional = flow.append_latent_mixer_cache(
        flow.empty_latent_mixer_cache(),
        context_latents=content[:, 0],
        context_conditions=torch.zeros_like(condition[:, 0]),
        context_positions=positions[:, 0],
    )
    stacked = flow.stack_latent_mixer_caches(
        [conditional, unconditional]
    )
    paired = flow.velocity(
        torch.cat([query[:, 1], query[:, 1]], dim=0),
        torch.cat([time[:, 1], time[:, 1]], dim=0),
        torch.cat(
            [condition[:, 1], torch.zeros_like(condition[:, 1])],
            dim=0,
        ),
        query_positions=torch.cat(
            [positions[:, 1], positions[:, 1]],
            dim=0,
        ),
        latent_mixer_cache=stacked,
    )
    separate_conditional = flow.velocity(
        query[:, 1],
        time[:, 1],
        condition[:, 1],
        query_positions=positions[:, 1],
        latent_mixer_cache=conditional,
    )
    separate_unconditional = flow.velocity(
        query[:, 1],
        time[:, 1],
        torch.zeros_like(condition[:, 1]),
        query_positions=positions[:, 1],
        latent_mixer_cache=unconditional,
    )
    torch.testing.assert_close(
        paired[:1],
        separate_conditional,
        atol=1e-6,
        rtol=1e-6,
    )
    torch.testing.assert_close(
        paired[1:],
        separate_unconditional,
        atol=1e-6,
        rtol=1e-6,
    )


def test_forward_and_sampling_are_finite():
    flow = _flow()
    content, _, condition, _, positions, sigma, strict_mask = _inputs()
    loss = flow(
        content,
        condition,
        context_latents=content,
        sigma=sigma,
        image_positions=positions,
    )
    sample = flow.sample(
        condition,
        num_steps=1,
        cfg=1.0,
        context_latents=content,
        context_mask=strict_mask,
        query_positions=positions,
        context_positions=positions,
        context_conditions=condition,
    )
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert sample.shape == content.shape
    assert torch.isfinite(sample).all()


def test_training_uses_float32_flow_objective_and_bfloat16_network(monkeypatch):
    flow = _flow().to(dtype=torch.bfloat16)
    content, _, condition, _, positions, sigma, _ = _inputs()
    observed = {}
    original_velocity = flow.velocity

    def capture_velocity(self, x_t, t, z, **context_kwargs):
        del self
        observed["objective_x_t_dtype"] = x_t.dtype
        return original_velocity(x_t, t, z, **context_kwargs)

    def fake_net(self, x_t, t, c, **context_kwargs):
        del self, t, context_kwargs
        observed["network_x_t_dtype"] = x_t.dtype
        observed["network_condition_dtype"] = c.dtype
        return torch.zeros_like(x_t)

    monkeypatch.setattr(
        flow,
        "velocity",
        types.MethodType(capture_velocity, flow),
    )
    monkeypatch.setattr(flow.net, "forward", types.MethodType(fake_net, flow.net))
    loss = flow(
        content,
        condition,
        context_latents=content,
        sigma=sigma,
        image_positions=positions,
    )

    assert observed == {
        "objective_x_t_dtype": torch.float32,
        "network_x_t_dtype": torch.bfloat16,
        "network_condition_dtype": torch.bfloat16,
    }
    assert loss.dtype == torch.float32


def test_disabling_flow_stats_preserves_output_and_skips_stat_tensors():
    flow = _flow()
    content, query, condition, time, positions, _, strict_mask = _inputs()
    without_stats = flow.velocity(
        query,
        time,
        condition,
        context_latents=content,
        context_mask=strict_mask,
        query_positions=positions,
        context_positions=positions,
        context_conditions=condition,
        record_stats=False,
    )
    assert flow.net.last_gate_abs_per_token is None
    assert all(block.last_update_rms_per_token is None for block in flow.net.blocks)

    with_stats = flow.velocity(
        query,
        time,
        condition,
        context_latents=content,
        context_mask=strict_mask,
        query_positions=positions,
        context_positions=positions,
        context_conditions=condition,
        record_stats=True,
    )
    torch.testing.assert_close(with_stats, without_stats, rtol=0.0, atol=0.0)
    assert flow.net.last_gate_abs_per_token is not None
    assert all(block.last_update_rms_per_token is not None for block in flow.net.blocks)


def test_dual_stream_checkpointing_matches_forward_and_gradients_exactly():
    eager = _flow(grad_checkpointing=False)
    checkpointed = _flow(grad_checkpointing=True)
    checkpointed.load_state_dict(eager.state_dict(), strict=True)
    content, query, condition, time, positions, _, strict_mask = _inputs()
    eager_content = content.clone().requires_grad_(True)
    eager_query = query.clone().requires_grad_(True)
    checkpointed_content = content.clone().requires_grad_(True)
    checkpointed_query = query.clone().requires_grad_(True)

    kwargs = {
        "context_mask": strict_mask,
        "query_positions": positions,
        "context_positions": positions,
        "context_conditions": condition,
    }
    eager_output = eager.velocity(
        eager_query,
        time,
        condition,
        context_latents=eager_content,
        **kwargs,
    )
    checkpointed_output = checkpointed.velocity(
        checkpointed_query,
        time,
        condition,
        context_latents=checkpointed_content,
        **kwargs,
    )
    torch.testing.assert_close(
        checkpointed_output,
        eager_output,
        rtol=0.0,
        atol=0.0,
    )

    probe = torch.linspace(0.1, 1.0, eager_output.numel()).reshape_as(
        eager_output
    )
    (eager_output * probe).sum().backward()
    (checkpointed_output * probe).sum().backward()
    torch.testing.assert_close(
        checkpointed_content.grad,
        eager_content.grad,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        checkpointed_query.grad,
        eager_query.grad,
        rtol=0.0,
        atol=0.0,
    )
    for (eager_name, eager_parameter), (
        checkpointed_name,
        checkpointed_parameter,
    ) in zip(eager.named_parameters(), checkpointed.named_parameters()):
        assert checkpointed_name == eager_name
        if eager_parameter.grad is None:
            assert checkpointed_parameter.grad is None
        else:
            torch.testing.assert_close(
                checkpointed_parameter.grad,
                eager_parameter.grad,
                rtol=0.0,
                atol=0.0,
            )
