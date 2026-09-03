import inspect
import os

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import pytest
import torch
from transformers import (
    AutoConfig,
    Qwen3Config,
)
from transformers import (
    Qwen3ForCausalLM as TransformersQwen3ForCausalLM,
)

from models.modeling_model.image_flow_loss import FlowLoss
from models.modeling_model.modeling_selfless_flow import (
    Qwen3ForCausalLM as StaticQwen3ForCausalLM,
)
from models.modeling_model.modeling_selfless_flow import (
    Qwen3Model,
)
from models.modeling_model.modeling_selfless_flow_dynamic_xt import (
    DYNAMIC_XT_ATTENTION_CONTRACT,
    DYNAMIC_XT_FLOW_BATCH_MUL,
    DYNAMIC_XT_FLOW_HEAD_ATTENTION_CONTRACT,
    DynamicXtQwen3ForCausalLM,
    DynamicXtQwen3Model,
    SelflessFlowDynamicXtConfig,
)
from models.modeling_model.modeling_selfless_flow_dynamic_xt_generation import (
    DynamicXtGenerationMixin,
)
from models.modeling_model.rectified_flow_state import (
    sample_rectified_flow_training_state,
)
from utils.utils import get_selfless_mask


def tiny_config(config_class=SelflessFlowDynamicXtConfig):
    config = config_class(
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
    config.image_flow_depth = 2
    config.image_flow_num_sampling_steps = "10"
    config.image_flow_batch_mul = DYNAMIC_XT_FLOW_BATCH_MUL
    config.image_flow_time_scale = 1000.0
    config.image_flow_time_sampling = "uniform"
    config.image_flow_time_eps = 1.0e-4
    config.image_flow_time_uniform_mix = 0.0
    config.image_flow_solver = "heun"
    config.image_input_noise_strength = 0.0
    config.image_uncond_prob = 0.1
    config.lambda_text = 0.05
    config.lambda_image = 1.0
    config.architecture_variant = "dynamic_xt"
    config.training_objective = "selfless_dual_stream"
    config.dual_stream_attention_contract = DYNAMIC_XT_ATTENTION_CONTRACT
    config.flow_head_attention_contract = (
        DYNAMIC_XT_FLOW_HEAD_ATTENTION_CONTRACT
    )
    config.use_cache = False
    return config


def test_shared_rf_helper_preserves_per_token_fp32_math_and_rng_order():
    target = torch.arange(24, dtype=torch.bfloat16).view(2, 3, 4)

    def sampled_times(count, device):
        assert count == 6
        return torch.rand(count, device=device)

    torch.manual_seed(123)
    state = sample_rectified_flow_training_state(
        target,
        sample_times=sampled_times,
        device=torch.device("cpu"),
    )
    torch.manual_seed(123)
    expected_t = sampled_times(6, torch.device("cpu")).view(2, 3)
    expected_noise = torch.randn(target.shape, dtype=torch.float32)
    expected_x_t = (
        (1.0 - expected_t.unsqueeze(-1)) * expected_noise
        + expected_t.unsqueeze(-1) * target.float()
    )
    torch.testing.assert_close(state.noise, expected_noise)
    torch.testing.assert_close(state.t, expected_t)
    torch.testing.assert_close(state.x_t, expected_x_t)
    torch.testing.assert_close(state.v_target, target.float() - expected_noise)


def test_flow_head_accepts_presampled_state_without_resampling():
    flow = FlowLoss(
        target_channels=4,
        z_channels=32,
        depth=1,
        width=32,
        num_sampling_steps="10",
        time_sampling="uniform",
        uniform_mix=0.0,
        image_tokens_per_img=4,
    )
    target = torch.randn(1, 4, 4)
    state = flow.sample_training_state(target)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("flow time was sampled twice")

    flow._sample_times = forbidden
    loss = flow(
        target=target,
        z=torch.randn(1, 4, 32),
        sigma=torch.arange(4).view(1, 4).float(),
        image_positions=torch.arange(4).view(1, 4),
        context_latents=target,
        training_state=state,
    )
    assert torch.isfinite(loss)


def test_dynamic_xt_replaces_only_image_query_embeddings():
    model = DynamicXtQwen3Model(tiny_config())
    input_ids = torch.tensor([[3, 11, 8, 8, 8, 8, 12, 9]])
    token_types = torch.tensor([[0, 2, 1, 1, 1, 1, 2, 0]])
    x_t = torch.randn(1, 8, 4)
    times = torch.rand(1, 8)
    query_mask = token_types.eq(1)
    static_xt = model._build_xt_inputs_embeds(input_ids, token_types, True)
    dynamic_xt = model._build_dynamic_xt_inputs_embeds(
        input_ids=input_ids,
        token_types=token_types,
        image_spans_present=True,
        xt_flow_latents=x_t,
        xt_flow_times=times,
        xt_flow_query_mask=query_mask,
    )
    torch.testing.assert_close(dynamic_xt[~query_mask], static_xt[~query_mask])
    assert not torch.equal(dynamic_xt[query_mask], static_xt[query_mask])


def test_dynamic_model_has_distinct_type_and_only_time_embedder_capacity():
    model = DynamicXtQwen3ForCausalLM(tiny_config())
    expected = 256 * 32 + 32 + 32 * 32 + 32
    assert model.config.model_type == "selfless_flow_dynamic_xt"
    assert model.model_type == "selfless_flow_dynamic_xt"
    assert model.dynamic_xt_parameter_count() == expected
    assert not hasattr(Qwen3Model(tiny_config(Qwen3Config)), "backbone_flow_time_embedder")


def test_dynamic_training_contract_preserves_four_states_with_shared_content():
    source = inspect.getsource(DynamicXtQwen3ForCausalLM.forward)
    assert "checkpoint(" not in source
    assert "repeated_targets = targets.repeat(repeats, 1, 1)" in source
    assert source.count("hidden_states = self.model(") == 1
    assert 'output["per_modality_loss"]' in source
    assert 'output["per_modality_count"]' in source

    invalid = tiny_config()
    invalid.image_flow_batch_mul = 1
    with pytest.raises(ValueError, match="image_flow_batch_mul=4"):
        DynamicXtQwen3ForCausalLM(invalid)

    invalid = tiny_config()
    invalid.flow_head_attention_contract = "selfless_strict"
    with pytest.raises(ValueError, match="flow_head_attention_contract"):
        DynamicXtQwen3ForCausalLM(invalid)


def test_dynamic_xt_is_isolated_from_baseline_model_file():
    baseline_source = inspect.getsourcefile(StaticQwen3ForCausalLM)
    assert baseline_source is not None
    assert "dynamic_xt" not in open(
        baseline_source,
        encoding="utf-8",
    ).read().lower()


def test_dynamic_xt_generation_is_a_separate_implementation_file():
    model_source = inspect.getsourcefile(DynamicXtQwen3ForCausalLM)
    generation_source = inspect.getsourcefile(DynamicXtGenerationMixin)
    assert model_source is not None
    assert generation_source is not None
    assert model_source != generation_source
    model_text = open(model_source, encoding="utf-8").read()
    generation_text = open(generation_source, encoding="utf-8").read()
    assert "def generate_image(" not in model_text
    assert "def _make_backbone_flow_condition_evaluator(" not in model_text
    assert "def generate_image(" in generation_text
    assert "def _make_backbone_flow_condition_evaluator(" in generation_text


def test_dynamic_only_initialization_preserves_static_parameters_and_rng():
    seed = 456
    torch.manual_seed(seed)
    static = StaticQwen3ForCausalLM(tiny_config(Qwen3Config))
    rng_after_static = torch.random.get_rng_state()

    torch.manual_seed(seed)
    dynamic = DynamicXtQwen3ForCausalLM(tiny_config())
    rng_after_dynamic = torch.random.get_rng_state()

    dynamic_parameters = dict(dynamic.named_parameters())
    for name, static_parameter in static.named_parameters():
        torch.testing.assert_close(
            dynamic_parameters[name],
            static_parameter,
            rtol=0,
            atol=0,
        )
    for module in dynamic.model.backbone_flow_time_embedder.mlp:
        if isinstance(module, torch.nn.Linear):
            assert torch.isfinite(module.weight).all()
            assert module.weight.std() > 0
            torch.testing.assert_close(
                module.bias,
                torch.zeros_like(module.bias),
                rtol=0,
                atol=0,
            )
    assert torch.equal(rng_after_dynamic, rng_after_static)


def test_dynamic_presampling_keeps_static_x0_then_rf_rng_order():
    source = inspect.getsource(DynamicXtQwen3ForCausalLM.forward)
    input_noise_offset = source.index("self._shared_noisy_image_latents")
    rf_state_offset = source.index("self.image_flow_head.sample_training_state")
    backbone_offset = source.index("hidden_states = self.model")
    assert input_noise_offset < rf_state_offset < backbone_offset

    static_config = tiny_config(Qwen3Config)
    dynamic_config = tiny_config()
    static_config.image_input_noise_strength = 0.01
    dynamic_config.image_input_noise_strength = 0.01
    static = StaticQwen3ForCausalLM(static_config).train()
    dynamic = DynamicXtQwen3ForCausalLM(dynamic_config).train()
    image_latents = torch.randn(1, 8, 4)
    token_types = torch.tensor([[0, 2, 1, 1, 1, 1, 2, 0]])
    targets = image_latents[:, 2:6]

    torch.manual_seed(789)
    static_context = static._shared_noisy_image_latents(image_latents, token_types)
    repeated_targets = targets.repeat(DYNAMIC_XT_FLOW_BATCH_MUL, 1, 1)
    static_state = static.image_flow_head.sample_training_state(repeated_targets)
    torch.manual_seed(789)
    dynamic_context = dynamic._shared_noisy_image_latents(image_latents, token_types)
    dynamic_state = dynamic.image_flow_head.sample_training_state(
        repeated_targets
    )

    torch.testing.assert_close(dynamic_context, static_context, rtol=0, atol=0)
    for field in ("t", "noise", "x_t", "v_target"):
        torch.testing.assert_close(
            getattr(dynamic_state, field),
            getattr(static_state, field),
            rtol=0,
            atol=0,
        )


def test_dynamic_training_runs_one_content_batch_and_four_query_batches():
    config = tiny_config()
    config.dynamic_xt_t2i_gradient_checkpointing = True
    model = DynamicXtQwen3ForCausalLM(config).train()
    assert model.model.gradient_checkpointing is False
    assert all(not layer.gradient_checkpointing for layer in model.model.layers)
    input_ids = torch.tensor([[3, 11, 8, 8, 8, 8, 12, 9]])
    token_types = torch.tensor(
        [[0, 2, 1, 1, 1, 1, 2, 0]],
        dtype=torch.uint8,
    )
    sigma = torch.tensor([[0.0, 1.0, 4.0, 5.0, 6.0, 7.0, 2.0, 3.0]])
    image_latents = torch.randn(1, 8, 4)
    strict_mask = get_selfless_mask(sigma, 8, "cpu")
    content_mask = get_selfless_mask(
        sigma,
        8,
        "cpu",
        include_diagonal=True,
    )

    output = model(
        X0_input_ids=input_ids,
        labels=input_ids,
        attention_mask=strict_mask,
        content_attention_mask=content_mask,
        token_types=token_types,
        image_latents=image_latents,
        image_local_positions=torch.tensor(
            [[-1, -1, 0, 1, 2, 3, -1, -1]]
        ),
        image_span_table=torch.tensor([[0, 0, 2, 6, 0]]),
        flow_sigma=sigma,
        compute_text_loss=False,
        compute_image_loss=True,
    )

    assert torch.isfinite(output.loss)
    assert output.dynamic_xt_query_batch_mul == 4
    assert output.dynamic_xt_content_batch_size == 1
    assert output.dynamic_xt_query_batch_size == 4
    assert output.dynamic_xt_checkpointed_layers == config.num_hidden_layers
    assert int(output.per_modality_count["image_tokens"]) == 16
    torch.testing.assert_close(output.loss, output.per_modality_loss["image_loss"])
    output.loss.backward()
    time_gradient = model.model.backbone_flow_time_embedder.mlp[0].weight.grad
    assert time_gradient is not None
    assert torch.isfinite(time_gradient).all()
    assert all(not layer.gradient_checkpointing for layer in model.model.layers)


def test_four_query_streams_match_four_independent_dynamic_forwards():
    model = DynamicXtQwen3Model(tiny_config()).eval()
    input_ids = torch.tensor([[3, 11, 8, 8, 8, 8, 12, 9]])
    token_types = torch.tensor(
        [[0, 2, 1, 1, 1, 1, 2, 0]],
        dtype=torch.uint8,
    )
    sigma = torch.tensor([[0.0, 1.0, 4.0, 5.0, 6.0, 7.0, 2.0, 3.0]])
    image_latents = torch.randn(1, 8, 4)
    x_t = torch.randn(4, 8, 4)
    times = torch.rand(4, 8)
    query_mask = token_types.eq(1).repeat(4, 1)
    strict_mask = get_selfless_mask(sigma, 8, "cpu")
    content_mask = get_selfless_mask(
        sigma,
        8,
        "cpu",
        include_diagonal=True,
    )

    batched = model(
        X0_input_ids=input_ids,
        attention_mask=strict_mask,
        content_attention_mask=content_mask,
        token_types=token_types,
        image_latents=image_latents,
        calculate_likelihood=True,
        xt_flow_latents=x_t,
        xt_flow_times=times,
        xt_flow_query_mask=query_mask,
    ).last_hidden_state
    independent = torch.cat(
        [
            model(
                X0_input_ids=input_ids,
                attention_mask=strict_mask,
                content_attention_mask=content_mask,
                token_types=token_types,
                image_latents=image_latents,
                calculate_likelihood=True,
                xt_flow_latents=x_t[index : index + 1],
                xt_flow_times=times[index : index + 1],
                xt_flow_query_mask=query_mask[index : index + 1],
            ).last_hidden_state
            for index in range(4)
        ],
        dim=0,
    )

    torch.testing.assert_close(batched, independent, rtol=1.0e-5, atol=1.0e-6)


def test_non_t2i_microbatch_keeps_dynamic_parameters_in_backward_graph():
    config = tiny_config()
    config.dynamic_xt_t2i_gradient_checkpointing = True
    model = DynamicXtQwen3ForCausalLM(config).train()
    input_ids = torch.tensor([[3, 4, 5, 9]])
    token_types = torch.zeros_like(input_ids, dtype=torch.uint8)
    sigma = torch.arange(4, dtype=torch.float32).unsqueeze(0)
    strict_mask = get_selfless_mask(sigma, 4, "cpu")
    content_mask = get_selfless_mask(
        sigma,
        4,
        "cpu",
        include_diagonal=True,
    )

    output = model(
        X0_input_ids=input_ids,
        labels=input_ids,
        attention_mask=strict_mask,
        content_attention_mask=content_mask,
        token_types=token_types,
        flow_sigma=sigma,
        compute_text_loss=True,
        compute_image_loss=False,
    )
    output.loss.backward()

    assert model.model.last_dynamic_xt_checkpointed_layers == 0
    assert model.model.gradient_checkpointing is False
    assert all(not layer.gradient_checkpointing for layer in model.model.layers)
    for parameter in model.model.backbone_flow_time_embedder.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_dynamic_checkpoint_roundtrip_keeps_distinct_model_identity(tmp_path):
    model = DynamicXtQwen3ForCausalLM(tiny_config())
    expected = {
        name: value.detach().clone()
        for name, value in model.model.backbone_flow_time_embedder.state_dict().items()
    }
    model.save_pretrained(tmp_path)

    loaded_config = AutoConfig.from_pretrained(tmp_path)
    assert isinstance(loaded_config, SelflessFlowDynamicXtConfig)
    assert loaded_config.model_type == "selfless_flow_dynamic_xt"
    loaded = DynamicXtQwen3ForCausalLM.from_pretrained(tmp_path)
    for name, value in loaded.model.backbone_flow_time_embedder.state_dict().items():
        torch.testing.assert_close(value, expected[name], rtol=0, atol=0)


def test_dynamic_pretrained_init_preserves_common_missing_parameters(tmp_path):
    source_config = tiny_config(Qwen3Config)
    TransformersQwen3ForCausalLM(source_config).save_pretrained(tmp_path)

    seed = 987
    torch.manual_seed(seed)
    static = StaticQwen3ForCausalLM.from_pretrained(tmp_path)
    torch.manual_seed(seed)
    dynamic_payload = source_config.to_dict()
    dynamic_payload.pop("model_type", None)
    dynamic = DynamicXtQwen3ForCausalLM.from_pretrained(
        tmp_path,
        config=SelflessFlowDynamicXtConfig(**dynamic_payload),
    )

    dynamic_parameters = dict(dynamic.named_parameters())
    for name, static_parameter in static.named_parameters():
        torch.testing.assert_close(
            dynamic_parameters[name],
            static_parameter,
            rtol=0,
            atol=0,
        )
    for module in dynamic.model.backbone_flow_time_embedder.mlp:
        if isinstance(module, torch.nn.Linear):
            assert torch.isfinite(module.weight).all()
            assert module.weight.std() > 0
            torch.testing.assert_close(
                module.bias,
                torch.zeros_like(module.bias),
                rtol=0,
                atol=0,
            )


def test_dynamic_evaluation_adapter_restores_backbone_time_embedder(tmp_path):
    from utils.image_generation_io import load_adapter

    source = DynamicXtQwen3ForCausalLM(tiny_config())
    expected = {
        name: value.detach().clone()
        for name, value in source.model.backbone_flow_time_embedder.state_dict().items()
    }
    path = tmp_path / "dynamic_xt_adapter.pt"
    torch.save({"backbone_flow_time_embedder": expected}, path)

    target = DynamicXtQwen3ForCausalLM(tiny_config())
    with torch.no_grad():
        for parameter in target.model.backbone_flow_time_embedder.parameters():
            parameter.zero_()
    report = load_adapter(target, str(path))
    assert report["backbone_flow_time_embedder"] == len(expected)
    for name, value in target.model.backbone_flow_time_embedder.state_dict().items():
        torch.testing.assert_close(value, expected[name], rtol=0, atol=0)


def test_inference_contract_uses_read_only_x0_cache_and_dynamic_heun_hook():
    source = inspect.getsource(
        DynamicXtQwen3ForCausalLM._make_backbone_flow_condition_evaluator
    )
    assert "cache_read_only=True" in source
    assert "xt_flow_latents=aligned_x_t" in source
    assert "torch.cat([conditional, unconditional]" in source


@pytest.mark.parametrize("use_cache", [False, True])
def test_dynamic_generation_recomputes_condition_in_dedicated_path(use_cache):
    model = DynamicXtQwen3ForCausalLM(tiny_config()).eval()
    input_ids = torch.tensor([[3, 11, 8, 8, 8, 8, 12, 9]])
    token_types = torch.tensor(
        [[0, 2, 1, 1, 1, 1, 2, 0]],
        dtype=torch.uint8,
    )
    sigma = torch.tensor([[0.0, 1.0, 4.0, 5.0, 6.0, 7.0, 2.0, 3.0]])

    generated, trace = model.generate(
        "t2i",
        input_ids=input_ids,
        token_types=token_types,
        sigma=sigma,
        spans=[(0, 2, 6)],
        image_latent_dim=4,
        initial_noise_bank=torch.zeros(1, 4, 4),
        flow_temperature=1.0,
        flow_cfg=1.0,
        flow_solver="heun",
        flow_num_steps=1,
        parallel_rate=1,
        order_strategy="sigma",
        use_cache=use_cache,
        return_trace=True,
        _debug_max_generation_steps=1,
    )

    assert tuple(generated.shape) == (1, 4, 2, 2)
    assert torch.isfinite(generated).all()
    assert trace["backbone_condition_mode"] == "dynamic_xt"
    assert trace["dynamic_xt_conditional_velocity_evaluations"] == 2
    assert trace["dynamic_xt_unconditional_velocity_evaluations"] == 0
    assert trace["dynamic_xt_query_cache_policy"] == "read_only_x0_kv"


def test_heun_calls_dynamic_condition_for_predictor_and_corrector():
    flow = FlowLoss(
        target_channels=4,
        z_channels=32,
        depth=1,
        width=32,
        num_sampling_steps="10",
        time_sampling="uniform",
        uniform_mix=0.0,
        image_tokens_per_img=4,
    )
    calls = []

    def evaluator(x_t, t):
        calls.append((x_t.detach().clone(), t.detach().clone()))
        return torch.zeros(x_t.shape[0], 32)

    flow.sample(
        torch.zeros(1, 32),
        solver="heun",
        num_steps=1,
        initial_noise=torch.zeros(1, 4),
        condition_evaluator=evaluator,
    )
    assert len(calls) == 2
    assert calls[0][1].item() == 0.0
    assert calls[1][1].item() == 1.0
