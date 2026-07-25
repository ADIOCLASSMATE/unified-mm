import inspect

import pytest
import torch
from omegaconf import OmegaConf
from transformers import Qwen3Config

from models.modeling_model.image_flow_loss import (
    ContextualFlowTransformerHead,
    FlowLoss,
)
from models.modeling_model.image_flow_position import (
    FLOW_HEAD_POSITION_SPECS,
    SUPPORTED_FLOW_HEAD_POSITION_VARIANTS,
    resolve_flow_head_position_config,
    resolve_model_flow_head_position,
    validate_flow_rope_axis_dims,
)
from models.modeling_model.image_position_utils import (
    apply_local_row_col_rope,
    build_local_row_col_rope,
)
from models.modeling_model.modeling_selfless_flow import Qwen3ForCausalLM


def _tiny_config(**overrides):
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    config.mask_token_id = 7
    config.image_mask_token_id = 8
    config.image_tokens_per_img = 4
    config.image_latent_dim = 4
    config.image_backbone_variant = "E2-Q1"
    config.image_input_noise_strength = 0.0
    config.image_flow_head_arch = "contextual"
    config.image_flow_width = 8
    config.image_flow_depth = 1
    config.image_flow_latent_mixer_heads = 2
    config.image_flow_rope_axis_dims = [2, 2]
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def _head(variant):
    spec = FLOW_HEAD_POSITION_SPECS[variant]
    torch.manual_seed(11)
    return ContextualFlowTransformerHead(
        in_channels=4,
        model_channels=8,
        out_channels=4,
        z_channels=8,
        num_res_blocks=1,
        latent_mixer_heads=2,
        image_tokens_per_img=16,
        query_position_mode=spec.query_position_mode,
        context_position_mode=spec.context_position_mode,
        rope_mode=spec.rope_mode,
        rope_axis_dims=(2, 2),
        position_variant=variant,
    )


def _nontrivial_head(variant):
    head = _head(variant)
    with torch.no_grad():
        block = head.blocks[0]
        width = head.model_channels
        block.adaLN_modulation[-1].bias[2 * width : 3 * width].fill_(0.7)
        block.adaLN_modulation[-1].bias[5 * width : 6 * width].fill_(0.4)
        head.final_layer.linear.weight.normal_(std=0.1)
        head.final_layer.linear.bias.normal_(std=0.1)
    return head


def _inputs():
    generator = torch.Generator().manual_seed(17)
    return {
        "x": torch.randn(2, 3, 4, generator=generator),
        "t": torch.rand(2, 3, generator=generator),
        "c": torch.randn(2, 3, 8, generator=generator),
        "context": torch.randn(2, 4, 4, generator=generator),
        "mask": torch.tensor(
            [
                [[True, True, False, False]] * 3,
                [[True, True, True, True]] * 3,
            ]
        ),
        "query_positions": torch.tensor([[0, 1, 5], [6, 10, 15]]),
        "context_positions": torch.tensor([[0, 1, 4, 5], [2, 7, 8, 13]]),
    }


def test_preregistered_flow_head_matrix_is_exact():
    assert SUPPORTED_FLOW_HEAD_POSITION_VARIANTS == (
        "FH0",
        "FH1",
        "FH2",
        "FH3",
        "FH4",
    )
    assert {
        key: (
            int(spec.query_additive),
            int(spec.context_additive),
            int(spec.uses_rope),
        )
        for key, spec in FLOW_HEAD_POSITION_SPECS.items()
    } == {
        "FH0": (1, 1, 0),
        "FH1": (1, 1, 1),
        "FH2": (1, 0, 0),
        "FH3": (1, 0, 1),
        "FH4": (0, 0, 1),
    }


@pytest.mark.parametrize("variant", SUPPORTED_FLOW_HEAD_POSITION_VARIANTS)
def test_resolver_persists_exact_modes_and_validates_ablation_id(variant):
    spec = FLOW_HEAD_POSITION_SPECS[variant]
    config = OmegaConf.create(
        {
            "experiment": {"ablation_id": variant},
            "model": {
                "image_backbone_variant": "E2-Q1",
                "image_flow_head_arch": "contextual",
                "image_flow_width": 1280,
                "image_flow_latent_mixer_heads": 8,
                "image_flow_position_variant": variant,
                "image_flow_query_position_mode": spec.query_position_mode,
                "image_flow_context_position_mode": spec.context_position_mode,
                "image_flow_rope_mode": spec.rope_mode,
                "image_flow_rope_axis_dims": [80, 80],
                "image_flow_rope_rotate_value": False,
            },
        }
    )
    resolved, dims = resolve_flow_head_position_config(config)
    assert resolved == spec
    assert dims == (80, 80)
    assert config.experiment.flow_head_position_variant == variant


def test_resolver_rejects_unregistered_modes_value_rotation_and_wrong_backbone():
    with pytest.raises(ValueError, match="preregistered"):
        resolve_model_flow_head_position(
            _tiny_config(
                image_flow_query_position_mode="none",
                image_flow_context_position_mode="additive_2d",
                image_flow_rope_mode="none",
            )
        )
    with pytest.raises(ValueError, match="never rotates V"):
        resolve_model_flow_head_position(
            _tiny_config(image_flow_rope_rotate_value=True)
        )
    with pytest.raises(ValueError, match="requires image_backbone_variant=E2-Q1"):
        resolve_flow_head_position_config(
            OmegaConf.create(
                {
                    "experiment": {"ablation_id": "FH0"},
                    "model": {
                        "image_backbone_variant": "E2-Q0",
                        "image_flow_head_arch": "contextual",
                        "image_flow_width": 1280,
                        "image_flow_latent_mixer_heads": 8,
                    },
                }
            )
        )


def test_head_dimension_160_is_split_into_even_80_80_axes():
    assert validate_flow_rope_axis_dims([80, 80], head_dim=160) == (80, 80)
    with pytest.raises(ValueError, match="sum"):
        validate_flow_rope_axis_dims([80, 78], head_dim=160)
    with pytest.raises(ValueError, match="positive even"):
        validate_flow_rope_axis_dims([79, 81], head_dim=160)


def test_same_position_joint_rotation_preserves_dot_product():
    generator = torch.Generator().manual_seed(23)
    q = torch.randn(2, 3, 4, 8, generator=generator)
    k = torch.randn(2, 3, 4, 8, generator=generator)
    positions = torch.tensor([[0, 1, 4, 5], [6, 10, 14, 15]])
    q_rot = apply_local_row_col_rope(
        q,
        positions,
        image_tokens_per_img=16,
        axis_dims=(4, 4),
    )
    k_rot = apply_local_row_col_rope(
        k,
        positions,
        image_tokens_per_img=16,
        axis_dims=(4, 4),
    )
    expected = (q * k).sum(dim=-1)
    actual = (q_rot * k_rot).sum(dim=-1)
    assert torch.allclose(actual, expected, atol=2e-5, rtol=2e-5)


def test_row_and_column_frequency_pairs_are_isolated_and_interleaved():
    row_cos, row_sin = build_local_row_col_rope(
        torch.tensor([[0, 4]]),
        image_tokens_per_img=16,
        head_dim=8,
        axis_dims=(4, 4),
    )
    col_cos, col_sin = build_local_row_col_rope(
        torch.tensor([[0, 1]]),
        image_tokens_per_img=16,
        head_dim=8,
        axis_dims=(4, 4),
    )
    row_dims = torch.tensor([0, 2, 4, 6])
    col_dims = torch.tensor([1, 3, 5, 7])
    assert torch.equal(row_cos[:, 0, col_dims], row_cos[:, 1, col_dims])
    assert torch.equal(row_sin[:, 0, col_dims], row_sin[:, 1, col_dims])
    assert torch.equal(col_cos[:, 0, row_dims], col_cos[:, 1, row_dims])
    assert torch.equal(col_sin[:, 0, row_dims], col_sin[:, 1, row_dims])
    assert not torch.equal(row_cos[:, 0, row_dims], row_cos[:, 1, row_dims])
    assert not torch.equal(col_cos[:, 0, col_dims], col_cos[:, 1, col_dims])


def test_relative_attention_score_is_translation_invariant():
    generator = torch.Generator().manual_seed(29)
    q = torch.randn(1, 2, 1, 8, generator=generator)
    k = torch.randn(1, 2, 1, 8, generator=generator)
    q0 = apply_local_row_col_rope(
        q,
        torch.tensor([[0]]),
        image_tokens_per_img=16,
        axis_dims=(4, 4),
    )
    k0 = apply_local_row_col_rope(
        k,
        torch.tensor([[1]]),
        image_tokens_per_img=16,
        axis_dims=(4, 4),
    )
    q_shift = apply_local_row_col_rope(
        q,
        torch.tensor([[4]]),
        image_tokens_per_img=16,
        axis_dims=(4, 4),
    )
    k_shift = apply_local_row_col_rope(
        k,
        torch.tensor([[5]]),
        image_tokens_per_img=16,
        axis_dims=(4, 4),
    )
    assert torch.allclose(
        (q0 * k0).sum(dim=-1),
        (q_shift * k_shift).sum(dim=-1),
        atol=2e-5,
        rtol=2e-5,
    )


@pytest.mark.parametrize("variant", SUPPORTED_FLOW_HEAD_POSITION_VARIANTS)
def test_all_variants_have_identical_learned_parameter_schema(variant):
    reference = _head("FH0")
    candidate = _head(variant)
    expected = {name: tuple(value.shape) for name, value in reference.named_parameters()}
    actual = {name: tuple(value.shape) for name, value in candidate.named_parameters()}
    assert actual == expected
    assert sum(value.numel() for value in candidate.parameters()) == sum(
        value.numel() for value in reference.parameters()
    )


@pytest.mark.parametrize("variant", SUPPORTED_FLOW_HEAD_POSITION_VARIANTS)
def test_cached_and_direct_context_paths_match(variant):
    head = _nontrivial_head(variant).eval()
    inputs = _inputs()
    direct = head(
        inputs["x"],
        inputs["t"],
        inputs["c"],
        context_latents=inputs["context"],
        context_mask=inputs["mask"],
        query_positions=inputs["query_positions"],
        context_positions=inputs["context_positions"],
    )
    cache = head.prepare_latent_mixer_cache(
        inputs["context"],
        inputs["mask"],
        inputs["context_positions"],
    )
    cached = head(
        inputs["x"],
        inputs["t"],
        inputs["c"],
        query_positions=inputs["query_positions"],
        latent_mixer_cache=cache,
    )
    assert torch.allclose(direct, cached, atol=2e-5, rtol=2e-5)


def test_k_is_rotated_once_v_is_unrotated_and_cache_contract_is_checked():
    head = _head("FH3").eval()
    inputs = _inputs()
    block = head.blocks[0]
    context_hidden = head.input_proj(inputs["context"])
    normalized = block.cross_kv_norm(context_hidden)
    expected_v = block._split_heads(block.cross_v(normalized))
    cache = head.prepare_latent_mixer_cache(
        inputs["context"],
        inputs["mask"],
        inputs["context_positions"],
    )
    layer = cache["layers"][0]
    assert layer["k_rotation_count"] == 1
    assert torch.equal(layer["v"], expected_v)
    bad_cache = dict(cache)
    bad_cache["position_contract"] = dict(cache["position_contract"])
    bad_cache["position_contract"]["variant"] = "FH0"
    with pytest.raises(ValueError, match="position contract mismatch"):
        head(
            inputs["x"],
            inputs["t"],
            inputs["c"],
            query_positions=inputs["query_positions"],
            latent_mixer_cache=bad_cache,
        )


def test_empty_single_and_partially_masked_contexts_are_finite():
    head = _nontrivial_head("FH4").eval()
    inputs = _inputs()
    no_context = head(
        inputs["x"],
        inputs["t"],
        inputs["c"],
        query_positions=inputs["query_positions"],
    )
    assert torch.isfinite(no_context).all()
    for context_len in (1, 4):
        context = inputs["context"][:, :context_len]
        positions = inputs["context_positions"][:, :context_len]
        mask = inputs["mask"][:, :, :context_len].clone()
        mask[0] = False
        output = head(
            inputs["x"],
            inputs["t"],
            inputs["c"],
            context_latents=context,
            context_mask=mask,
            query_positions=inputs["query_positions"],
            context_positions=positions,
        )
        assert torch.isfinite(output).all()


def test_training_context_stays_strictly_earlier_without_future_leakage():
    flow = FlowLoss(
        target_channels=4,
        z_channels=8,
        depth=1,
        width=8,
        num_sampling_steps=2,
        image_tokens_per_img=4,
        latent_mixer_heads=2,
        rope_axis_dims=(2, 2),
        position_variant="FH3",
        query_position_mode="additive_2d",
        context_position_mode="none",
        rope_mode="row_col_2d",
    )
    sigma = torch.tensor([[2.0, 0.0, 3.0, 1.0]])
    target = torch.randn(1, 4, 4)
    positions = torch.arange(4).unsqueeze(0)
    context = flow._training_context(target, sigma, positions)
    expected = sigma.unsqueeze(1) < sigma.unsqueeze(2)
    assert torch.equal(context["context_mask"], expected)
    assert not torch.diagonal(context["context_mask"], dim1=1, dim2=2).any()


def test_context_count_bucket_losses_and_gates_are_recorded():
    flow = FlowLoss(
        target_channels=4,
        z_channels=8,
        depth=1,
        width=8,
        num_sampling_steps=2,
        image_tokens_per_img=100,
        latent_mixer_heads=2,
        rope_axis_dims=(2, 2),
        position_variant="FH0",
        query_position_mode="additive_2d",
        context_position_mode="additive_2d",
        rope_mode="none",
    )
    flow.net.set_attention_diagnostics(True)
    target = torch.randn(1, 66, 4)
    condition = torch.randn(1, 66, 8)
    sigma = torch.arange(66, dtype=torch.float32).unsqueeze(0)
    positions = torch.arange(66).unsqueeze(0)
    loss = flow(
        target,
        condition,
        sigma=sigma,
        image_positions=positions,
    )
    assert torch.isfinite(loss)
    for tag in ("0", "1", "2_4", "5_16", "17_64", "65_plus"):
        assert torch.isfinite(flow.last_forward_stats[f"flow/context_{tag}_v_mse"])
        assert torch.isfinite(flow.last_forward_stats[f"flow/context_{tag}_gate_abs"])
        assert torch.isfinite(
            flow.last_forward_stats[f"flow/context_{tag}_attention_entropy"]
        )
        assert torch.isfinite(
            flow.last_forward_stats[f"flow/context_{tag}_attention_distance"]
        )


def test_guidance_velocity_delta_is_bucketed_by_visible_context_count():
    flow = FlowLoss(
        target_channels=4,
        z_channels=8,
        depth=1,
        width=8,
        num_sampling_steps=2,
        image_tokens_per_img=100,
        latent_mixer_heads=2,
        rope_axis_dims=(2, 2),
    )
    counts = (0, 1, 3, 10, 40, 65)
    context_mask = torch.zeros(6, 1, 65, dtype=torch.bool)
    for row, count in enumerate(counts):
        context_mask[row, :, :count] = True
    flow._record_guidance_delta(
        torch.ones(6, 4),
        {"latent_mixer_cache": {"context_mask": context_mask}},
    )
    diagnostics = flow.guidance_diagnostics()
    assert set(diagnostics) == {"0", "1", "2_4", "5_16", "17_64", "65_plus"}
    for values in diagnostics.values():
        assert values["sum"].item() == pytest.approx(1.0)
        assert values["count"].item() == pytest.approx(1.0)


def test_model_config_checkpoint_contract_and_parameter_schema_match_all_variants():
    schemas = {}
    for variant in SUPPORTED_FLOW_HEAD_POSITION_VARIANTS:
        config = _tiny_config(image_flow_position_variant=variant)
        model = Qwen3ForCausalLM(config).eval()
        spec = FLOW_HEAD_POSITION_SPECS[variant]
        assert model.config.image_flow_position_variant == variant
        assert model.image_flow_head.net.position_contract() == spec.as_contract(
            (2, 2)
        )
        schemas[variant] = {
            name: tuple(parameter.shape)
            for name, parameter in model.image_flow_head.named_parameters()
        }
    assert all(schema == schemas["FH0"] for schema in schemas.values())


def test_public_signatures_never_expose_rotate_value_on_attention_call():
    assert "rotate_value" not in inspect.signature(
        ContextualFlowTransformerHead.forward
    ).parameters
