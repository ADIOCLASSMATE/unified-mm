import os
import copy

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import pytest
import torch
from torch.nn.attention.flex_attention import flex_attention
from transformers import Qwen3Config

from models.modeling_model import modeling_selfless_flow as selfless_flow
from models.modeling_model.image_flow_loss import FlowLoss
from models.modeling_model.modeling_selfless_flow import (
    LEGACY_FLOW_CONDITION_CONTRACT,
    X0_CONTENT_FLOW_CONDITION_CONTRACT,
    Qwen3ForCausalLM,
    X0ContentFlowLoss,
)
from utils.utils import get_selfless_mask


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


def _config(attention_contract, *, condition_contract=...):
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
    config.image_flow_batch_mul = 4
    config.image_flow_time_scale = 1000.0
    config.image_flow_time_sampling = "uniform"
    config.image_flow_time_eps = 1.0e-4
    config.image_flow_time_uniform_mix = 0.0
    config.image_flow_solver = "euler"
    config.image_input_noise_strength = 0.0
    config.image_uncond_prob = 0.0
    config.lambda_text = 0.05
    config.lambda_image = 1.0
    config.use_flex_attention = False
    config.training_objective = "selfless_dual_stream"
    config.dual_stream_attention_contract = attention_contract
    config.flow_head_attention_contract = attention_contract
    if condition_contract is not ...:
        config.flow_condition_contract = condition_contract
    return config


def _batch(attention_contract):
    input_ids = torch.tensor([[3, 11, 8, 8, 8, 8, 12, 9]])
    token_types = torch.tensor(
        [[0, 2, 1, 1, 1, 1, 2, 0]],
        dtype=torch.uint8,
    )
    sigma = torch.tensor([[0.0, 1.0, 4.0, 5.0, 6.0, 7.0, 2.0, 3.0]])
    strict_mask = get_selfless_mask(sigma, 8, "cpu")
    content_mask = get_selfless_mask(
        sigma,
        8,
        "cpu",
        include_diagonal=attention_contract == "xlnet_content_diagonal",
    )
    image_latents = torch.zeros(1, 8, 4)
    torch.manual_seed(20260903)
    image_latents[:, 2:6] = torch.randn(1, 4, 4)
    return {
        "X0_input_ids": input_ids,
        "labels": input_ids.clone(),
        "attention_mask": strict_mask,
        "content_attention_mask": content_mask,
        "token_types": token_types,
        "image_latents": image_latents,
        "image_local_positions": torch.tensor(
            [[-1, -1, 0, 1, 2, 3, -1, -1]]
        ),
        "image_span_table": torch.tensor([[0, 0, 2, 6, 0]]),
        "image_loss_mask": token_types.eq(1),
        "flow_sigma": sigma,
        "compute_text_loss": False,
        "compute_image_loss": True,
        "record_flow_stats": True,
        "return_logits": False,
    }


def _randomize_flow_output(model):
    torch.manual_seed(314159)
    with torch.no_grad():
        for block in model.image_flow_head.net.blocks:
            block.adaLN_modulation[-1].weight.normal_(0.0, 0.1)
            block.adaLN_modulation[-1].bias.normal_(0.0, 0.1)
        final = model.image_flow_head.net.final_layer
        final.adaLN_modulation[-1].weight.normal_(0.0, 0.1)
        final.adaLN_modulation[-1].bias.normal_(0.0, 0.1)
        final.linear.weight.normal_(0.0, 0.1)
        final.linear.bias.normal_(0.0, 0.1)


@pytest.mark.parametrize("checkpointing", [False, True])
@pytest.mark.parametrize("attention_contract", ["selfless_strict", "xlnet_content_diagonal"])
def test_shared_content_rf_batch_preserves_loss_and_all_gradients(
    monkeypatch, checkpointing, attention_contract,
):
    torch.manual_seed(67)
    config = _config(attention_contract, condition_contract=X0_CONTENT_FLOW_CONDITION_CONTRACT)
    config.image_flow_depth = 3
    reference = Qwen3ForCausalLM(config).train()
    _randomize_flow_output(reference)
    shared = copy.deepcopy(reference)
    shared.config.image_flow_share_content = True
    reference.image_flow_head.net.grad_checkpointing = checkpointing
    shared.image_flow_head.net.grad_checkpointing = checkpointing
    batch = _batch(attention_contract)
    # Two distinct images catch repeat-major ordering mistakes that B=1 hides.
    for key, value in list(batch.items()):
        if torch.is_tensor(value):
            batch[key] = value.repeat(2, *([1] * (value.dim() - 1)))
    batch["image_span_table"][1, 0] = 1
    batch["attention_mask"] = get_selfless_mask(batch["flow_sigma"], 8, "cpu")
    batch["content_attention_mask"] = get_selfless_mask(
        batch["flow_sigma"], 8, "cpu", include_diagonal=attention_contract == "xlnet_content_diagonal",
    )
    batch["image_latents"][1, 2:6] += 0.7
    batch["image_local_positions"][1, 2:6] = torch.tensor([2, 0, 3, 1])
    batch["record_flow_stats"] = False
    content_shapes, query_shapes = [], []
    for block in shared.image_flow_head.net.blocks:
        original_cache = block.prepare_cross_cache

        def cache(hidden, *args, original_cache=original_cache, **kwargs):
            content_shapes.append(tuple(hidden.shape[:2]))
            return original_cache(hidden, *args, **kwargs)

        monkeypatch.setattr(block, "prepare_cross_cache", cache)
        block.register_forward_pre_hook(lambda _module, args: query_shapes.append(tuple(args[0].shape[:2])))

    torch.manual_seed(811)
    expected = reference(**batch)
    torch.manual_seed(811)
    actual = shared(**batch)
    torch.testing.assert_close(actual.loss, expected.loss, rtol=2e-6, atol=2e-6)
    assert int(actual.per_modality_count["image_tokens"]) == 32
    assert content_shapes == [(2, 4)] * 3
    assert query_shapes == [(2, 4), (2, 16)] * 3
    assert shared.image_flow_head.net.last_training_batch_layout["checkpointed_blocks"] == (3 if checkpointing else 0)
    expected.loss.backward()
    actual.loss.backward()
    for (name, ref_parameter), (shared_name, parameter) in zip(
        reference.named_parameters(), shared.named_parameters(), strict=True,
    ):
        assert name == shared_name
        if ref_parameter.grad is None:
            assert parameter.grad is None, name
        else:
            torch.testing.assert_close(parameter.grad, ref_parameter.grad, rtol=2e-4, atol=2e-6, msg=name)


def test_shared_content_checkpointing_skips_non_image_minibatches(monkeypatch):
    config = _config("xlnet_content_diagonal", condition_contract=X0_CONTENT_FLOW_CONDITION_CONTRACT)
    config.image_flow_share_content = True
    config.image_flow_grad_checkpointing = True
    model = Qwen3ForCausalLM(config).train()

    def unexpected_flow(*args, **kwargs):
        raise AssertionError("I2T/text minibatches must not execute the flow head")

    monkeypatch.setattr(model.image_flow_head.net, "forward", unexpected_flow)
    batch = _batch("xlnet_content_diagonal")
    batch.update(compute_image_loss=False, compute_text_loss=True, record_flow_stats=False)
    result = model(**batch)
    assert torch.isfinite(result.loss)
    result.loss.backward()


@pytest.mark.parametrize(
    "attention_contract",
    ["selfless_strict", "xlnet_content_diagonal"],
)
def test_missing_contract_keeps_exact_legacy_a_b_model_path(attention_contract):
    torch.manual_seed(101)
    missing = Qwen3ForCausalLM(_config(attention_contract)).train()
    torch.manual_seed(101)
    explicit = Qwen3ForCausalLM(
        _config(
            attention_contract,
            condition_contract=LEGACY_FLOW_CONDITION_CONTRACT,
        )
    ).train()

    assert type(missing.image_flow_head) is FlowLoss
    assert type(explicit.image_flow_head) is FlowLoss
    assert missing.config.flow_condition_contract == (
        LEGACY_FLOW_CONDITION_CONTRACT
    )
    assert missing.state_dict().keys() == explicit.state_dict().keys()
    for name, value in missing.state_dict().items():
        torch.testing.assert_close(
            value,
            explicit.state_dict()[name],
            rtol=0,
            atol=0,
        )

    batch = _batch(attention_contract)
    torch.manual_seed(909)
    missing_output = missing(**batch)
    torch.manual_seed(909)
    explicit_output = explicit(**batch)
    torch.testing.assert_close(
        missing_output.loss,
        explicit_output.loss,
        rtol=0,
        atol=0,
    )
    missing_output.loss.backward()
    explicit_output.loss.backward()
    for (missing_name, missing_parameter), (
        explicit_name,
        explicit_parameter,
    ) in zip(missing.named_parameters(), explicit.named_parameters(), strict=True):
        assert missing_name == explicit_name
        if missing_parameter.grad is None:
            assert explicit_parameter.grad is None
        else:
            torch.testing.assert_close(
                missing_parameter.grad,
                explicit_parameter.grad,
                rtol=0,
                atol=0,
            )


@pytest.mark.parametrize(
    "attention_contract",
    ["selfless_strict", "xlnet_content_diagonal"],
)
def test_new_a_b_initialization_and_state_keys_match_legacy(attention_contract):
    torch.manual_seed(123)
    legacy = Qwen3ForCausalLM(_config(attention_contract))
    torch.manual_seed(123)
    current = Qwen3ForCausalLM(
        _config(
            attention_contract,
            condition_contract=X0_CONTENT_FLOW_CONDITION_CONTRACT,
        )
    )
    assert type(legacy.image_flow_head) is FlowLoss
    assert type(current.image_flow_head) is X0ContentFlowLoss
    assert legacy.state_dict().keys() == current.state_dict().keys()
    for name, value in legacy.state_dict().items():
        torch.testing.assert_close(
            value,
            current.state_dict()[name],
            rtol=0,
            atol=0,
        )


@pytest.mark.parametrize(
    "attention_contract",
    ["selfless_strict", "xlnet_content_diagonal"],
)
def test_training_routes_xt_query_and_x0_content_conditions(
    monkeypatch,
    attention_contract,
):
    torch.manual_seed(211)
    model = Qwen3ForCausalLM(
        _config(
            attention_contract,
            condition_contract=X0_CONTENT_FLOW_CONDITION_CONTRACT,
        )
    ).train()
    backbone_capture = {}
    flow_capture = {}
    original_backbone = model.model.forward
    original_flow = model.image_flow_head.forward

    def capture_backbone(*args, **kwargs):
        output = original_backbone(*args, **kwargs)
        backbone_capture["xt"] = output.last_hidden_state
        backbone_capture["x0"] = output["x0_last_hidden_state"]
        return output

    def capture_flow(*args, **kwargs):
        flow_capture["query"] = kwargs["z"]
        flow_capture["content"] = kwargs["context_conditions"]
        return original_flow(*args, **kwargs)

    monkeypatch.setattr(model.model, "forward", capture_backbone)
    monkeypatch.setattr(model.image_flow_head, "forward", capture_flow)
    output = model(**_batch(attention_contract))
    assert torch.isfinite(output.loss)
    assert int(output.per_modality_count["image_tokens"]) == 16

    image_indices = torch.arange(2, 6).view(1, 4)
    gather = image_indices.unsqueeze(-1).expand(-1, -1, 32)
    expected_query = model._prepare_image_flow_condition(
        torch.gather(backbone_capture["xt"], 1, gather)
    ).repeat(4, 1, 1)
    expected_content = model._prepare_image_flow_condition(
        torch.gather(backbone_capture["x0"], 1, gather)
    ).repeat(4, 1, 1)
    torch.testing.assert_close(flow_capture["query"], expected_query)
    torch.testing.assert_close(flow_capture["content"], expected_content)
    assert not torch.equal(flow_capture["query"], flow_capture["content"])
    output.loss.backward()
    for parameter in model.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()


@pytest.mark.parametrize(
    "attention_contract",
    ["selfless_strict", "xlnet_content_diagonal"],
)
def test_current_x0_token_cannot_leak_into_current_velocity(
    attention_contract,
):
    torch.manual_seed(319)
    model = Qwen3ForCausalLM(
        _config(
            attention_contract,
            condition_contract=X0_CONTENT_FLOW_CONDITION_CONTRACT,
        )
    ).eval()
    _randomize_flow_output(model)
    batch = _batch(attention_contract)
    input_ids = batch["X0_input_ids"]
    token_types = batch["token_types"]
    full_sigma = batch["flow_sigma"]
    image_sigma = full_sigma[:, 2:6]
    strict_flow = image_sigma.unsqueeze(1) < image_sigma.unsqueeze(2)
    content_flow = (
        image_sigma.unsqueeze(1) <= image_sigma.unsqueeze(2)
        if attention_contract == "xlnet_content_diagonal"
        else strict_flow
    )
    x_t = torch.randn(1, 4, 4)
    t = torch.rand(1, 4)
    positions = torch.arange(4).view(1, 4)

    def velocity(image_latents):
        backbone = model.model(
            X0_input_ids=input_ids,
            attention_mask=batch["attention_mask"],
            content_attention_mask=batch["content_attention_mask"],
            token_types=token_types,
            image_latents=image_latents,
            calculate_likelihood=True,
            return_x0_hidden_state=True,
        )
        query_condition = model._prepare_image_flow_condition(
            backbone.last_hidden_state[:, 2:6]
        )
        content_condition = model._prepare_image_flow_condition(
            backbone["x0_last_hidden_state"][:, 2:6]
        )
        prediction = model.image_flow_head.velocity(
            x_t,
            t,
            query_condition,
            context_latents=image_latents[:, 2:6],
            context_mask=strict_flow,
            content_attention_mask=content_flow,
            query_positions=positions,
            context_positions=positions,
            context_conditions=content_condition,
        )
        return prediction, query_condition, content_condition

    clean = batch["image_latents"].clone()
    changed = clean.clone()
    current_index = 1
    changed[:, 2 + current_index] += 25.0
    base_velocity, base_query, base_content = velocity(clean)
    changed_velocity, changed_query, changed_content = velocity(changed)

    torch.testing.assert_close(
        changed_query[:, current_index],
        base_query[:, current_index],
        rtol=0,
        atol=0,
    )
    assert not torch.allclose(
        changed_content[:, current_index],
        base_content[:, current_index],
    )
    torch.testing.assert_close(
        changed_velocity[:, current_index],
        base_velocity[:, current_index],
        rtol=0,
        atol=1.0e-6,
    )


@pytest.mark.parametrize(
    "attention_contract",
    ["selfless_strict", "xlnet_content_diagonal"],
)
@torch.no_grad()
def test_cached_generation_commits_fused_x0_and_matches_full_recompute(
    monkeypatch,
    attention_contract,
):
    monkeypatch.setattr(
        selfless_flow,
        "compiled_flex_attention",
        _eager_flex_attention,
    )
    torch.manual_seed(401)
    model = Qwen3ForCausalLM(
        _config(
            attention_contract,
            condition_contract=X0_CONTENT_FLOW_CONDITION_CONTRACT,
        )
    ).eval()
    _randomize_flow_output(model)
    input_ids = torch.tensor([[3, 11, 8, 8, 8, 8, 12, 9]])
    token_types = torch.tensor(
        [[0, 2, 1, 1, 1, 1, 2, 0]],
        dtype=torch.uint8,
    )
    sigma = torch.tensor([[0.0, 1.0, 4.0, 5.0, 6.0, 7.0, 2.0, 3.0]])
    common = {
        "input_ids": input_ids,
        "token_types": token_types,
        "sigma": sigma,
        "spans": [(0, 2, 6)],
        "initial_noise_bank": (
            torch.arange(16, dtype=torch.float32).view(1, 4, 4) / 19.0
        ),
        "flow_temperature": 0.7,
        "flow_cfg": 2.0,
        "flow_cfg_schedule": "constant",
        "flow_solver": "heun",
        "flow_num_steps": 2,
        "parallel_rate": 1,
        "order_strategy": "sigma",
        "return_trace": True,
        "_debug_max_generation_steps": 4,
    }
    full, full_trace = model.generate("t2i", **common, use_cache=False)

    fused_x0_conditions = []
    committed_conditions = []
    original_backbone = model.model.forward
    original_pending = model.image_flow_head.net.forward_with_pending_content

    def capture_backbone(*args, **kwargs):
        output = original_backbone(*args, **kwargs)
        if kwargs.get("cache_write_prefix") == 1:
            conditional_hidden, unconditional_hidden = (
                output.last_hidden_state[:, 0].chunk(2, dim=0)
            )
            fused_x0_conditions.append(
                torch.cat(
                    [
                        model._prepare_image_flow_condition(
                            conditional_hidden
                        ),
                        model._prepare_image_flow_condition(
                            unconditional_hidden
                        ),
                    ],
                    dim=0,
                ).detach().clone()
            )
        return output

    def capture_pending(*args, **kwargs):
        committed_conditions.append(
            kwargs["context_conditions"].detach().clone()
        )
        return original_pending(*args, **kwargs)

    monkeypatch.setattr(model.model, "forward", capture_backbone)
    monkeypatch.setattr(
        model.image_flow_head.net,
        "forward_with_pending_content",
        capture_pending,
    )
    cached, cached_trace = model.generate("t2i", **common, use_cache=True)

    torch.testing.assert_close(cached, full, rtol=1.0e-5, atol=1.0e-5)
    assert len(fused_x0_conditions) == 3
    # Pending content is committed once before the token's ODE evaluations;
    # the subsequent Heun evaluations reuse the updated flow-content cache.
    assert len(committed_conditions) == 3
    for actual, expected in zip(
        committed_conditions,
        fused_x0_conditions,
        strict=True,
    ):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    retired_query_condition = torch.cat(
        [
            model._prepare_image_flow_condition(
                cached_trace["debug_conditional_backbone_hidden"][0]
            ),
            model._prepare_image_flow_condition(
                cached_trace["debug_unconditional_backbone_hidden"][0]
            ),
        ],
        dim=0,
    )
    assert not torch.allclose(committed_conditions[0], retired_query_condition)
    assert cached_trace["flow_condition_contract"] == (
        X0_CONTENT_FLOW_CONDITION_CONTRACT
    )
    assert cached_trace["flow_content_condition"] == "backbone_x0_hidden"
    assert cached_trace["flow_content_condition_commit"] == (
        "fused_previous_x0_with_current_query"
    )
    assert cached_trace["flow_content_cache_tokens_committed"] == 3
    assert full_trace["flow_content_cache_tokens_committed"] == 3
