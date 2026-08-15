import inspect

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
    DynamicXtQwen3ForCausalLM,
    DynamicXtQwen3Model,
    SelflessFlowDynamicXtConfig,
)
from models.modeling_model.rectified_flow_state import (
    sample_rectified_flow_training_state,
)


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
    config.image_flow_num_sampling_steps = "2"
    config.image_flow_batch_mul = 1
    config.image_flow_time_scale = 1000.0
    config.image_flow_time_sampling = "uniform"
    config.image_flow_time_eps = 1.0e-4
    config.image_flow_time_uniform_mix = 0.0
    config.image_flow_solver = "heun"
    config.image_input_noise_strength = 0.0
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
        num_sampling_steps="2",
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
    backbone_offset = source.index("def run_backbone")
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
    static_state = static.image_flow_head.sample_training_state(targets)
    torch.manual_seed(789)
    dynamic_context = dynamic._shared_noisy_image_latents(image_latents, token_types)
    dynamic_state = dynamic.image_flow_head.sample_training_state(targets)

    torch.testing.assert_close(dynamic_context, static_context, rtol=0, atol=0)
    for field in ("t", "noise", "x_t", "v_target"):
        torch.testing.assert_close(
            getattr(dynamic_state, field),
            getattr(static_state, field),
            rtol=0,
            atol=0,
        )


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
    from scripts.generate_flow_validation_images import load_adapter

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


def test_heun_calls_dynamic_condition_for_predictor_and_corrector():
    flow = FlowLoss(
        target_channels=4,
        z_channels=32,
        depth=1,
        width=32,
        num_sampling_steps="1",
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
