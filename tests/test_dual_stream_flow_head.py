import inspect

import torch

from models.modeling_model.image_flow_loss import FlowLoss


def _flow() -> FlowLoss:
    torch.manual_seed(314159)
    flow = FlowLoss(
        target_channels=4,
        z_channels=8,
        depth=3,
        width=32,
        num_sampling_steps=2,
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
