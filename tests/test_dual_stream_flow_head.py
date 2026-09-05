import inspect
import types

import pytest
import torch

from models.modeling_model.image_flow_loss import FlowLoss


def _flow(
    *,
    grad_checkpointing: bool = False,
    attention_contract: str = "xlnet_content_diagonal",
) -> FlowLoss:
    torch.manual_seed(314159)
    flow = FlowLoss(
        target_channels=4,
        z_channels=8,
        depth=3,
        width=32,
        num_sampling_steps=10,
        grad_checkpointing=grad_checkpointing,
        image_tokens_per_img=4,
        flow_head_attention_contract=attention_contract,
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


def _pending_step(
    flow,
    cache,
    *,
    content,
    content_condition,
    content_position,
    query,
    query_time,
    query_condition,
    query_position,
):
    return flow.net.forward_with_pending_content(
        query,
        flow._scale_time(query_time),
        query_condition,
        latent_mixer_cache=cache,
        context_latents=content,
        context_conditions=content_condition,
        context_positions=content_position,
        query_positions=query_position,
    )


def _legacy_strict_velocity(
    flow,
    query,
    time,
    condition,
    content,
    strict_mask,
    positions,
):
    """Reference the pre-split flow-head forward used by old A/B weights."""

    net = flow.net
    model_dtype = net.input_proj.weight.dtype
    model_device = net.input_proj.weight.device
    query = query.to(device=model_device, dtype=model_dtype)
    condition = condition.to(device=model_device, dtype=model_dtype)
    content = content.to(device=model_device, dtype=model_dtype)
    time = time.to(device=model_device)
    strict_mask = strict_mask.to(device=model_device, dtype=torch.bool)
    positions = positions.to(device=model_device)

    batch_shape = query.shape[:-1]
    query, query_positions, squeeze = net._ensure_sequence(query, positions)
    query_rope = net._build_rope(query_positions, model_dtype)
    query = net.input_proj(query)
    time_embedding = net._shape_time(flow._scale_time(time), batch_shape)
    condition_embedding = net.cond_embed(condition)
    modulation = time_embedding + condition_embedding
    if modulation.dim() == 2:
        modulation = modulation.unsqueeze(1)

    batch_size, sequence_length, _ = content.shape
    context_positions = net._positions(
        positions,
        batch_size,
        sequence_length,
        model_device,
    )
    context_rope = net._build_rope(context_positions, model_dtype)
    content_hidden = net._initial_content_hidden(content, context_positions)
    content_modulation = net._content_condition(condition)
    prepared_strict_mask = net.blocks[0].prepare_context_mask(
        strict_mask,
        batch_size,
        sequence_length,
        sequence_length,
        model_device,
    )
    for block in net.blocks:
        layer_cache = block.prepare_cross_cache(
            content_hidden,
            context_positions=context_positions,
            context_rope=context_rope,
        )
        content_hidden = block(
            content_hidden,
            content_modulation,
            layer_cache=layer_cache,
            context_mask=prepared_strict_mask,
            query_positions=context_positions,
            query_rope=context_rope,
            include_mlp=True,
        )
        query = block(
            query,
            modulation,
            layer_cache=layer_cache,
            context_mask=prepared_strict_mask,
            query_positions=query_positions,
            query_rope=query_rope,
        )
    output = net.final_layer(query, modulation)
    return output.squeeze(1) if squeeze else output


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
    flow = _flow()
    assert not hasattr(flow, "append_latent_mixer_cache")
    assert not hasattr(flow.net, "append_latent_mixer_cache")


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
        "schema": "selfless_flow_head_content_cache_v2",
        "content_update": "shared_attention_mlp",
        "query_context": "strict_sigma_causal",
        "flow_head_attention_contract": "xlnet_content_diagonal",
        "content_self_diagonal": True,
        "query_writes_cache": False,
        "position_contract": flow.net.position_contract(),
    }


def test_dynamic_query_stream_stays_strict_with_content_diagonal():
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


def test_direct_forward_routes_distinct_content_and_query_masks(monkeypatch):
    flow = _flow()
    content, query, condition, time, positions, sigma, strict_mask = _inputs()
    observed_masks = []
    block = flow.net.blocks[0]
    original_forward = block.forward

    def capture_forward(*args, **kwargs):
        mask = kwargs["context_mask"]
        observed_masks.append(mask[0].detach().clone() if isinstance(mask, tuple) else mask.detach().clone())
        return original_forward(*args, **kwargs)

    monkeypatch.setattr(block, "forward", capture_forward)
    flow.velocity(
        query,
        time,
        condition,
        context_latents=content,
        context_mask=strict_mask,
        query_positions=positions,
        context_positions=positions,
        context_conditions=condition,
    )

    assert len(observed_masks) == 2
    torch.testing.assert_close(
        observed_masks[0],
        sigma.unsqueeze(1) <= sigma.unsqueeze(2),
    )
    torch.testing.assert_close(observed_masks[1], strict_mask)


def test_training_context_uses_sigma_leq_for_tied_content_ranks():
    flow = _flow()
    content, _, condition, _, positions, _, _ = _inputs()
    sigma = torch.tensor([[0.0, 1.0, 1.0, 2.0]])

    context = flow._training_context(content, sigma, positions)

    torch.testing.assert_close(
        context["context_mask"],
        sigma.unsqueeze(1) < sigma.unsqueeze(2),
    )
    torch.testing.assert_close(
        context["content_attention_mask"],
        sigma.unsqueeze(1) <= sigma.unsqueeze(2),
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_legacy_strict_forward_and_gradients_are_numerically_identical(dtype):
    legacy = _flow(attention_contract="selfless_strict").to(dtype=dtype)
    current = _flow(attention_contract="selfless_strict").to(dtype=dtype)
    current.load_state_dict(legacy.state_dict(), strict=True)
    content, query, condition, time, positions, _, strict_mask = _inputs()
    legacy_content = content.clone().requires_grad_(True)
    legacy_query = query.clone().requires_grad_(True)
    current_content = content.clone().requires_grad_(True)
    current_query = query.clone().requires_grad_(True)

    legacy_output = _legacy_strict_velocity(
        legacy,
        legacy_query,
        time,
        condition,
        legacy_content,
        strict_mask,
        positions,
    )
    current_output = current.velocity(
        current_query,
        time,
        condition,
        context_latents=current_content,
        context_mask=strict_mask,
        content_attention_mask=strict_mask,
        query_positions=positions,
        context_positions=positions,
        context_conditions=condition,
    )
    torch.testing.assert_close(current_output, legacy_output, rtol=0.0, atol=0.0)

    probe = torch.linspace(0.1, 1.0, legacy_output.numel()).reshape_as(
        legacy_output
    )
    (legacy_output * probe).sum().backward()
    (current_output * probe).sum().backward()
    torch.testing.assert_close(
        current_content.grad,
        legacy_content.grad,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        current_query.grad,
        legacy_query.grad,
        rtol=0.0,
        atol=0.0,
    )
    for (legacy_name, legacy_parameter), (
        current_name,
        current_parameter,
    ) in zip(legacy.named_parameters(), current.named_parameters()):
        assert current_name == legacy_name
        if legacy_parameter.grad is None:
            assert current_parameter.grad is None
        else:
            torch.testing.assert_close(
                current_parameter.grad,
                legacy_parameter.grad,
                rtol=0.0,
                atol=0.0,
            )


@pytest.mark.parametrize(
    "attention_contract",
    ["selfless_strict", "xlnet_content_diagonal"],
)
def test_incremental_cache_matches_full_sequence_last_query(attention_contract):
    flow = _flow(attention_contract=attention_contract)
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
        _pending_step(
            flow,
            cache,
            content=content[:, token_index],
            content_condition=condition[:, token_index],
            content_position=positions[:, token_index],
            query=query[:, token_index + 1],
            query_time=time[:, token_index + 1],
            query_condition=condition[:, token_index + 1],
            query_position=positions[:, token_index + 1],
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


@pytest.mark.parametrize(
    "attention_contract",
    ["selfless_strict", "xlnet_content_diagonal"],
)
@pytest.mark.parametrize("capacity", [None, 4])
def test_pending_content_fusion_matches_sequential_reference(
    attention_contract,
    capacity,
):
    flow = _flow(attention_contract=attention_contract).eval()
    content, query, condition, time, positions, _, _ = _inputs()
    sequential_cache = flow.empty_latent_mixer_cache(capacity=capacity)
    fused_cache = flow.empty_latent_mixer_cache(capacity=capacity)

    for token_index in range(3):
        sequential_cache = flow.net._append_content_cache_sequential(
            sequential_cache,
            context_latents=content[:, token_index],
            context_conditions=condition[:, token_index],
            context_positions=positions[:, token_index],
        )
        sequential = flow.velocity(
            query[:, token_index + 1],
            time[:, token_index + 1],
            condition[:, token_index + 1],
            query_positions=positions[:, token_index + 1],
            latent_mixer_cache=sequential_cache,
        )
        fused = _pending_step(
            flow,
            fused_cache,
            content=content[:, token_index],
            content_condition=condition[:, token_index],
            content_position=positions[:, token_index],
            query=query[:, token_index + 1],
            query_time=time[:, token_index + 1],
            query_condition=condition[:, token_index + 1],
            query_position=positions[:, token_index + 1],
        )

        tolerance = 0.0 if attention_contract == "selfless_strict" else 1e-6
        torch.testing.assert_close(
            fused,
            sequential,
            rtol=tolerance,
            atol=tolerance,
        )
        for fused_layer, sequential_layer in zip(
            fused_cache["layers"],
            sequential_cache["layers"],
        ):
            active_length = token_index + 1
            for name in ("k", "v"):
                torch.testing.assert_close(
                    fused_layer[name][:, :, :active_length],
                    sequential_layer[name][:, :, :active_length],
                    rtol=tolerance,
                    atol=tolerance,
                )


def test_corrected_pending_step_uses_one_two_row_block_call(monkeypatch):
    flow = _flow(attention_contract="xlnet_content_diagonal").eval()
    content, query, condition, time, positions, _, _ = _inputs()
    query_lengths = []

    for block in flow.net.blocks:
        original_forward = block.forward

        def counted_forward(*args, _original=original_forward, **kwargs):
            query_lengths.append(args[0].shape[1])
            return _original(*args, **kwargs)

        monkeypatch.setattr(block, "forward", counted_forward)

    _pending_step(
        flow,
        flow.empty_latent_mixer_cache(capacity=4),
        content=content[:, 0],
        content_condition=condition[:, 0],
        content_position=positions[:, 0],
        query=query[:, 1],
        query_time=time[:, 1],
        query_condition=condition[:, 1],
        query_position=positions[:, 1],
    )

    assert query_lengths == [2] * len(flow.net.blocks)


def test_cfg_sampling_duplicates_unpaired_pending_content():
    flow = _flow(attention_contract="xlnet_content_diagonal").eval()
    content, query, condition, _, positions, _, _ = _inputs()
    paired_condition = torch.cat(
        [condition[:, 1], torch.zeros_like(condition[:, 1])],
        dim=0,
    )

    sample = flow.sample(
        paired_condition,
        cfg=2.5,
        solver="euler",
        num_steps=1,
        query_positions=positions[:, 1],
        latent_mixer_cache=flow.empty_latent_mixer_cache(),
        pending_context_latents=content[:, 0],
        pending_context_conditions=condition[:, 0],
        pending_context_positions=positions[:, 0],
        initial_noise=query[:, 1],
    )

    assert sample.shape == query[:, 1].shape
    assert torch.isfinite(sample).all()


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
    single = _pending_step(
        flow,
        cache,
        content=content[:, 0],
        content_condition=condition[:, 0],
        content_position=positions[:, 0],
        query=query[:, 1],
        query_time=time[:, 1],
        query_condition=condition[:, 1],
        query_position=positions[:, 1],
    )
    assert torch.isfinite(empty).all()
    assert torch.isfinite(single).all()


def test_fixed_capacity_content_cache_matches_growing_cache():
    flow = _flow()
    content, query, condition, time, positions, _, _ = _inputs()
    growing = flow.empty_latent_mixer_cache()
    fixed = flow.empty_latent_mixer_cache(capacity=3)
    storage_pointers = [
        (layer["k_storage"].data_ptr(), layer["v_storage"].data_ptr())
        for layer in fixed["layers"]
    ]

    for token_index in range(3):
        kwargs = {
            "content": content[:, token_index],
            "content_condition": condition[:, token_index],
            "content_position": positions[:, token_index],
            "query": query[:, token_index + 1],
            "query_time": time[:, token_index + 1],
            "query_condition": condition[:, token_index + 1],
            "query_position": positions[:, token_index + 1],
        }
        _pending_step(flow, growing, **kwargs)
        _pending_step(flow, fixed, **kwargs)

        assert fixed["active_length"] == token_index + 1
        assert fixed["capacity"] == 3
        assert storage_pointers == [
            (
                layer["k_storage"].data_ptr(),
                layer["v_storage"].data_ptr(),
            )
            for layer in fixed["layers"]
        ]
        for growing_layer, fixed_layer in zip(
            growing["layers"], fixed["layers"]
        ):
            active_length = token_index + 1
            torch.testing.assert_close(
                fixed_layer["k"][:, :, :active_length],
                growing_layer["k"],
            )
            torch.testing.assert_close(
                fixed_layer["v"][:, :, :active_length],
                growing_layer["v"],
            )

    growing_velocity = flow.velocity(
        query[:, -1],
        time[:, -1],
        condition[:, -1],
        query_positions=positions[:, -1],
        latent_mixer_cache=growing,
    )
    fixed_velocity = flow.velocity(
        query[:, -1],
        time[:, -1],
        condition[:, -1],
        query_positions=positions[:, -1],
        latent_mixer_cache=fixed,
    )
    torch.testing.assert_close(fixed_velocity, growing_velocity)
    stacked_fixed = flow.stack_latent_mixer_caches([fixed])
    assert stacked_fixed["layers"][0]["k"].shape[2] == 3
    stacked_velocity = flow.velocity(
        query[:, -1],
        time[:, -1],
        condition[:, -1],
        query_positions=positions[:, -1],
        latent_mixer_cache=stacked_fixed,
    )
    torch.testing.assert_close(stacked_velocity, growing_velocity)


def test_stacked_cfg_cache_matches_separate_branches():
    flow = _flow()
    content, query, condition, time, positions, _, _ = _inputs()
    conditional = flow.empty_latent_mixer_cache()
    _pending_step(
        flow,
        conditional,
        content=content[:, 0],
        content_condition=condition[:, 0],
        content_position=positions[:, 0],
        query=query[:, 1],
        query_time=time[:, 1],
        query_condition=condition[:, 1],
        query_position=positions[:, 1],
    )
    unconditional = flow.empty_latent_mixer_cache()
    _pending_step(
        flow,
        unconditional,
        content=content[:, 0],
        content_condition=torch.zeros_like(condition[:, 0]),
        content_position=positions[:, 0],
        query=query[:, 1],
        query_time=time[:, 1],
        query_condition=torch.zeros_like(condition[:, 1]),
        query_position=positions[:, 1],
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


def test_cached_sampling_prepares_context_mask_once(monkeypatch):
    flow = _flow()
    content, _, condition, _, positions, _, strict_mask = _inputs()
    cache = flow.prepare_latent_mixer_cache(
        context_latents=content,
        context_mask=strict_mask,
        context_positions=positions,
        context_conditions=condition,
    )
    prepare_calls = 0
    rope_calls = 0
    condition_embedding_calls = 0
    original_prepare = flow.net.blocks[0].prepare_context_mask
    original_rope = flow.net._build_rope
    original_condition_embedding = flow.net.cond_embed.forward

    def counted_prepare(*args, **kwargs):
        nonlocal prepare_calls
        prepare_calls += 1
        return original_prepare(*args, **kwargs)

    def counted_rope(*args, **kwargs):
        nonlocal rope_calls
        rope_calls += 1
        return original_rope(*args, **kwargs)

    def counted_condition_embedding(*args, **kwargs):
        nonlocal condition_embedding_calls
        condition_embedding_calls += 1
        return original_condition_embedding(*args, **kwargs)

    monkeypatch.setattr(
        flow.net.blocks[0],
        "prepare_context_mask",
        counted_prepare,
    )
    monkeypatch.setattr(flow.net, "_build_rope", counted_rope)
    monkeypatch.setattr(
        flow.net.cond_embed,
        "forward",
        counted_condition_embedding,
    )
    flow.sample(
        condition,
        num_steps=3,
        solver="euler",
        cfg=1.0,
        query_positions=positions,
        latent_mixer_cache=cache,
        initial_noise=torch.zeros_like(content),
    )

    assert prepare_calls == 1
    assert rope_calls == 1
    assert condition_embedding_calls == 1


def test_sampling_reuses_timestep_embeddings(monkeypatch):
    flow = _flow()
    _, _, condition, _, positions, _, _ = _inputs()
    time_embedding_calls = 0
    original_forward = flow.net.time_embed.forward

    def counted_forward(*args, **kwargs):
        nonlocal time_embedding_calls
        time_embedding_calls += 1
        return original_forward(*args, **kwargs)

    monkeypatch.setattr(flow.net.time_embed, "forward", counted_forward)
    sample_kwargs = {
        "num_steps": 3,
        "solver": "heun",
        "cfg": 1.0,
        "query_positions": positions,
        "initial_noise": torch.zeros(1, 4, 4),
    }
    flow.sample(condition, **sample_kwargs)
    flow.sample(condition, **sample_kwargs)

    assert time_embedding_calls == 4


@pytest.mark.parametrize("resume_training", [True, False])
@torch.no_grad()
def test_sampling_refreshes_time_embeddings_after_unversioned_updates(
    resume_training,
):
    flow = _flow()
    model = torch.nn.ModuleList([flow]).eval()
    _, _, condition, _, positions, _, _ = _inputs()
    sample_kwargs = {
        "num_steps": 3,
        "solver": "heun",
        "cfg": 1.0,
        "query_positions": positions,
        "initial_noise": torch.zeros(1, 4, 4),
    }
    previous = flow.sample(condition, **sample_kwargs)
    versions = tuple(
        parameter._version for parameter in flow.net.time_embed.parameters()
    )
    if resume_training:
        assert model.train() is model
    flow.net.time_embed.mlp[2].bias.data.add_(1.0)
    assert versions == tuple(
        parameter._version for parameter in flow.net.time_embed.parameters()
    )
    assert model.eval() is model

    actual = flow.sample(condition, **sample_kwargs)
    reloaded = _flow().eval()
    reloaded.load_state_dict(flow.state_dict(), strict=True)
    expected = reloaded.sample(condition, **sample_kwargs)

    assert not torch.allclose(previous, expected)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_training_loss_does_not_use_cached_inference_time_embeddings():
    flow = _flow().train()
    content, _, condition, _, positions, sigma, _ = _inputs()
    times, _ = flow._inference_time_grid(3, content.device)
    with torch.no_grad():
        flow._inference_time_embeddings(times, tuple(condition.shape[:-1]))
        flow.net.time_embed.mlp[2].bias.data.add_(1.0)
    forward_kwargs = {
        "context_latents": content,
        "sigma": sigma,
        "image_positions": positions,
    }

    torch.manual_seed(424242)
    with_stale_cache = flow(content, condition, **forward_kwargs)
    flow._inference_time_embedding_cache = None
    torch.manual_seed(424242)
    without_cache = flow(content, condition, **forward_kwargs)

    torch.testing.assert_close(with_stale_cache, without_cache, rtol=0.0, atol=0.0)


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
