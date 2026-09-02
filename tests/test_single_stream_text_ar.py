import pytest
import torch
import torch.nn.functional as F
from transformers import AutoConfig, Qwen3Config

from models.modeling_model.modeling_selfless_flow import Qwen3ForCausalLM
from models.modeling_model.modeling_single_stream_text_ar import (
    SingleStreamTextARConfig,
    SingleStreamTextARQwen3ForCausalLM,
    _materialize_allowed_mask,
    _physical_causal_mask,
)
from utils.utils import get_selfless_mask


def _tiny_config(config_class=SingleStreamTextARConfig):
    config = config_class(
        vocab_size=40,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=32,
        tie_word_embeddings=True,
    )
    values = {
        "architecture_variant": (
            "single_stream_text_ar"
            if config_class is SingleStreamTextARConfig
            else "selfless_contextual"
        ),
        "training_objective": "selfless_dual_stream",
        "dual_stream_attention_contract": "xlnet_content_diagonal",
        "mask_token_id": 7,
        "image_mask_token_id": 8,
        "boi_token_id": 11,
        "eoi_token_id": 12,
        "image_latent_dim": 4,
        "image_tokens_per_img": 4,
        "image_flow_width": 32,
        "image_flow_depth": 1,
        "image_flow_num_sampling_steps": "10",
        "image_flow_batch_mul": 1,
        "image_flow_time_sampling": "uniform",
        "image_input_noise_strength": 0.0,
        "lambda_text": 0.05,
        "lambda_image": 1.0,
        "use_flex_attention": True,
    }
    for key, value in values.items():
        setattr(config, key, value)
    return config


def _paired_models():
    torch.manual_seed(17)
    baseline = Qwen3ForCausalLM(_tiny_config(Qwen3Config))
    torch.manual_seed(17)
    text_ar = SingleStreamTextARQwen3ForCausalLM(_tiny_config())
    return baseline, text_ar


def test_text_ar_has_identical_parameters_and_initial_rng_contract():
    torch.manual_seed(29)
    baseline = Qwen3ForCausalLM(_tiny_config(Qwen3Config))
    baseline_rng = torch.random.get_rng_state()
    torch.manual_seed(29)
    text_ar = SingleStreamTextARQwen3ForCausalLM(_tiny_config())
    text_ar_rng = torch.random.get_rng_state()

    baseline_state = baseline.state_dict()
    text_ar_state = text_ar.state_dict()
    assert baseline_state.keys() == text_ar_state.keys()
    for name, value in baseline_state.items():
        torch.testing.assert_close(value, text_ar_state[name], rtol=0, atol=0)
    torch.testing.assert_close(baseline_rng, text_ar_rng, rtol=0, atol=0)


def test_text_ar_returns_previous_single_stream_hidden_at_target_positions():
    baseline, text_ar = _paired_models()
    baseline.eval()
    text_ar.eval()
    input_ids = torch.tensor([[3, 4, 5, 6]])
    token_types = torch.zeros_like(input_ids, dtype=torch.uint8)
    sigma = torch.arange(4).unsqueeze(0)
    strict_mask = get_selfless_mask(
        sigma,
        4,
        "cpu",
        segment_ids=torch.zeros_like(sigma),
    )
    causal_mask = _physical_causal_mask(
        strict_mask,
        input_ids=input_ids,
        token_types=token_types,
        segment_ids=None,
        cache_position=None,
    )
    raw_content = baseline.model(
        X0_input_ids=input_ids,
        attention_mask=causal_mask,
        token_types=token_types,
        calculate_likelihood=False,
    ).last_hidden_state
    aligned = text_ar.model(
        X0_input_ids=input_ids,
        attention_mask=strict_mask,
        token_types=token_types,
        calculate_likelihood=True,
    ).last_hidden_state

    torch.testing.assert_close(
        aligned[:, 1:], raw_content[:, :-1], rtol=0, atol=0
    )
    torch.testing.assert_close(
        aligned[:, 0], torch.zeros_like(aligned[:, 0]), rtol=0, atol=0
    )


def test_text_ar_loss_is_target_aligned_next_token_ce_and_skips_xt():
    _, model = _paired_models()
    model.train()

    def forbidden_xt(*_args, **_kwargs):
        raise AssertionError("text-only c microbatch constructed the XT stream")

    model.model._build_xt_inputs_embeds = forbidden_xt
    input_ids = torch.tensor([[3, 4, 5, 6, 9, 10]])
    token_types = torch.zeros_like(input_ids, dtype=torch.uint8)
    segment_ids = torch.tensor([[0, 0, 0, 1, 1, 1]])
    sigma = torch.tensor([[0, 1, 2, 0, 1, 2]])
    labels = input_ids.clone()
    labels[:, 0] = -100
    labels[:, 3] = -100
    attention_mask = get_selfless_mask(
        sigma,
        input_ids.shape[1],
        "cpu",
        segment_ids=segment_ids,
    )
    output = model(
        X0_input_ids=input_ids,
        labels=labels,
        attention_mask=attention_mask,
        token_types=token_types,
        image_span_table=torch.empty(0, 5, dtype=torch.long),
        image_loss_mask=torch.zeros_like(input_ids, dtype=torch.bool),
        _text_segment_ids=segment_ids,
        compute_text_loss=True,
        compute_image_loss=False,
        return_logits=False,
    )
    valid = labels.ne(-100)
    manual = F.cross_entropy(
        model.lm_head(output.last_hidden_state[valid]),
        labels[valid],
    )
    assert output.per_modality_count["text_tokens"].item() == 4
    torch.testing.assert_close(
        output.per_modality_loss["text_loss"], manual, rtol=0, atol=1.0e-6
    )


def test_i2t_first_caption_target_is_conditioned_on_physical_image_prefix():
    _, model = _paired_models()
    model.eval()
    input_ids = torch.tensor([[3, 11, 8, 8, 8, 8, 12, 5, 6]])
    token_types = torch.tensor(
        [[0, 2, 1, 1, 1, 1, 2, 0, 2]], dtype=torch.uint8
    )
    # Baseline sigma deliberately puts EOI before every image token.
    sigma = torch.tensor([[0, 1, 3, 4, 5, 6, 2, 7, 8]])
    attention_mask = get_selfless_mask(sigma, input_ids.shape[1], "cpu")
    labels = input_ids.clone()
    labels[:, :7] = -100
    image_latents = torch.zeros(1, input_ids.shape[1], 4)
    image_latents[:, 2:6] = torch.randn(1, 4, 4)
    common = {
        "X0_input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "token_types": token_types,
        "image_latent_mask": token_types.eq(1),
        "image_span_table": torch.tensor([[0, 0, 2, 6, 0]]),
        "image_loss_mask": torch.zeros_like(input_ids, dtype=torch.bool),
        "compute_text_loss": True,
        "compute_image_loss": False,
        "return_logits": False,
    }
    first = model(image_latents=image_latents, **common)
    changed = image_latents.clone()
    changed[:, 2:6] += 10.0
    second = model(image_latents=changed, **common)

    first_caption_position = 7
    assert not torch.equal(
        first.last_hidden_state[:, first_caption_position],
        second.last_hidden_state[:, first_caption_position],
    )


def test_t2i_hidden_loss_and_gradients_are_exactly_baseline_a():
    baseline, text_ar = _paired_models()
    baseline.train()
    text_ar.train()
    input_ids = torch.tensor([[3, 11, 8, 8, 8, 8, 12]])
    token_types = torch.tensor([[0, 2, 1, 1, 1, 1, 2]], dtype=torch.uint8)
    sigma = torch.tensor([[0, 1, 3, 4, 5, 6, 2]])
    attention_mask = get_selfless_mask(sigma, input_ids.shape[1], "cpu")
    image_latents = torch.zeros(1, input_ids.shape[1], 4)
    image_latents[:, 2:6] = torch.randn(1, 4, 4)
    common = {
        "X0_input_ids": input_ids,
        "labels": input_ids.clone(),
        "attention_mask": attention_mask,
        "token_types": token_types,
        "image_latents": image_latents,
        "image_span_table": torch.tensor([[0, 0, 2, 6, 0]]),
        "image_loss_mask": token_types.eq(1),
        "compute_text_loss": False,
        "compute_image_loss": True,
        "return_logits": False,
        "record_flow_stats": False,
        "flow_sigma": sigma,
    }
    torch.manual_seed(123)
    baseline_output = baseline(**common)
    torch.manual_seed(123)
    text_ar_output = text_ar(**common)
    torch.testing.assert_close(
        baseline_output.last_hidden_state,
        text_ar_output.last_hidden_state,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        baseline_output.per_modality_loss["image_loss"],
        text_ar_output.per_modality_loss["image_loss"],
        rtol=0,
        atol=0,
    )

    baseline_output.loss.backward()
    text_ar_output.loss.backward()
    text_ar_parameters = dict(text_ar.named_parameters())
    for name, parameter in baseline.named_parameters():
        other = text_ar_parameters[name]
        if parameter.grad is None or other.grad is None:
            assert parameter.grad is None and other.grad is None
        else:
            torch.testing.assert_close(parameter.grad, other.grad, rtol=0, atol=0)


def test_physical_causal_text_mask_ignores_image_conditioning_dropout_edges():
    input_ids = torch.tensor([[3, 11, 8, 8, 12, 5]])
    token_types = torch.tensor([[0, 2, 1, 1, 2, 0]], dtype=torch.uint8)
    sigma = torch.tensor([[0, 1, 3, 4, 2, 5]])
    dropped = get_selfless_mask(
        sigma,
        input_ids.shape[1],
        "cpu",
        input_ids=input_ids,
        token_types=token_types,
        boi_token_id=11,
        image_uncond_rows=torch.tensor([True]),
    )

    causal = _physical_causal_mask(
        dropped,
        input_ids=input_ids,
        token_types=token_types,
        segment_ids=None,
        cache_position=None,
    )
    allowed = _materialize_allowed_mask(
        causal,
        batch_size=1,
        query_length=input_ids.shape[1],
        device=input_ids.device,
    )
    expected = torch.ones(6, 6, dtype=torch.bool).tril().unsqueeze(0)
    torch.testing.assert_close(allowed, expected, rtol=0, atol=0)


def test_image_conditioning_without_labels_is_exactly_baseline_a():
    baseline, text_ar = _paired_models()
    baseline.eval()
    text_ar.eval()
    input_ids = torch.tensor([[3, 11, 8, 8, 8, 8, 12]])
    token_types = torch.tensor([[0, 2, 1, 1, 1, 1, 2]], dtype=torch.uint8)
    sigma = torch.tensor([[0, 1, 3, 4, 5, 6, 2]])
    common = {
        "X0_input_ids": input_ids,
        "attention_mask": get_selfless_mask(
            sigma, input_ids.shape[1], "cpu"
        ),
        "token_types": token_types,
        "image_latents": torch.randn(1, input_ids.shape[1], 4),
        "image_span_table": torch.tensor([[0, 0, 2, 6, 0]]),
        "flow_sigma": sigma,
        "calculate_likelihood": True,
        "return_logits": False,
    }

    baseline_output = baseline(**common)
    text_ar_output = text_ar(**common)

    torch.testing.assert_close(
        baseline_output.last_hidden_state,
        text_ar_output.last_hidden_state,
        rtol=0,
        atol=0,
    )


def test_joint_validation_merges_source_specific_text_and_image_passes():
    _, model = _paired_models()
    model.eval()
    input_ids = torch.tensor([[3, 11, 8, 8, 8, 8, 12, 5, 6]])
    token_types = torch.tensor(
        [[0, 2, 1, 1, 1, 1, 2, 0, 2]], dtype=torch.uint8
    )
    sigma = torch.tensor([[0, 1, 3, 4, 5, 6, 2, 7, 8]])
    labels = input_ids.clone()
    labels[:, :7] = -100
    image_latents = torch.zeros(1, input_ids.shape[1], 4)
    image_latents[:, 2:6] = torch.randn(1, 4, 4)
    common = {
        "X0_input_ids": input_ids,
        "labels": labels,
        "attention_mask": get_selfless_mask(
            sigma, input_ids.shape[1], "cpu"
        ),
        "token_types": token_types,
        "image_latents": image_latents,
        "image_span_table": torch.tensor([[0, 0, 2, 6, 0]]),
        "image_loss_mask": token_types.eq(1),
        "return_logits": False,
        "record_flow_stats": True,
        "flow_sigma": sigma,
    }

    torch.manual_seed(101)
    joint = model(
        **common,
        compute_text_loss=True,
        compute_image_loss=True,
    )
    torch.manual_seed(101)
    image_only = model(
        **common,
        compute_text_loss=False,
        compute_image_loss=True,
    )
    text_only = model(
        **common,
        compute_text_loss=True,
        compute_image_loss=False,
    )

    torch.testing.assert_close(
        joint.per_modality_loss["text_loss"],
        text_only.per_modality_loss["text_loss"],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        joint.per_modality_loss["image_loss"],
        image_only.per_modality_loss["image_loss"],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        joint.last_hidden_state,
        image_only.last_hidden_state,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        joint.loss,
        model.lambda_text * joint.per_modality_loss["text_loss"]
        + model.lambda_image * joint.per_modality_loss["image_loss"],
        rtol=0,
        atol=0,
    )
    assert joint.per_modality_count["text_tokens"].item() == 2
    assert joint.per_modality_count["image_tokens"].item() == 4
    assert joint.flow_debug_stats.keys() == image_only.flow_debug_stats.keys()
    for key, value in joint.flow_debug_stats.items():
        torch.testing.assert_close(
            value, image_only.flow_debug_stats[key], rtol=0, atol=0
        )


@torch.no_grad()
def test_c_caption_cache_matches_full_reference_and_never_builds_xt():
    _, model = _paired_models()
    model.eval()

    def forbidden_xt(*_args, **_kwargs):
        raise AssertionError("C generation must remain single-stream")

    model.model._build_xt_inputs_embeds = forbidden_xt
    input_ids = torch.tensor([[3, 11, 8, 8, 8, 8, 12]])
    token_types = torch.tensor(
        [[0, 2, 1, 1, 1, 1, 2]],
        dtype=torch.uint8,
    )
    sigma = torch.tensor([[0.0, 1.0, 3.0, 4.0, 5.0, 6.0, 2.0]])
    image_latents = torch.zeros(1, 7, 4)
    image_latents[:, 2:6] = torch.randn(1, 4, 4)
    kwargs = {
        "max_new_tokens": 4,
        "token_types": token_types,
        "sigma": sigma,
        "image_latents": image_latents,
        "temperature": 0.0,
        "eos_token_id": -1,
        "return_trace": True,
    }

    cached, cached_trace = model.generate_text(
        input_ids,
        use_cache=True,
        **kwargs,
    )
    full, full_trace = model.generate_text(
        input_ids,
        use_cache=False,
        **kwargs,
    )

    torch.testing.assert_close(cached, full, rtol=0, atol=0)
    assert cached_trace == {
        "generation_mode": "single_stream_text_ar",
        "attention_contract": "physical_causal",
        "backbone_kv_cache_enabled": True,
        "generated_tokens": 4,
    }
    assert full_trace["backbone_kv_cache_enabled"] is False


def test_c_image_generation_inherits_configured_baseline_b_attention():
    _, model = _paired_models()
    assert model._generation_attention_contract() == "xlnet_content_diagonal"

    # Old checkpoints remain readable even though new formal C runs use B.
    model.config.dual_stream_attention_contract = "selfless_strict"
    assert model._generation_attention_contract() == "selfless_strict"


@torch.no_grad()
def test_c_on_b_image_cache_matches_its_full_reference():
    _, model = _paired_models()
    model.eval()
    common = {
        "input_ids": torch.tensor([[3, 11, 8, 8, 8, 8, 12]]),
        "token_types": torch.tensor(
            [[0, 2, 1, 1, 1, 1, 2]],
            dtype=torch.uint8,
        ),
        "sigma": torch.tensor([[0.0, 1.0, 3.0, 4.0, 5.0, 6.0, 2.0]]),
        "spans": [(0, 2, 6)],
        "initial_noise_bank": (
            torch.arange(16, dtype=torch.float32).view(1, 4, 4) / 11.0
        ),
        "flow_cfg": 1.0,
        "flow_solver": "euler",
        "flow_num_steps": 1,
        "order_strategy": "spatial_halton",
        "return_trace": True,
    }

    cached, cached_trace = model.generate_image(
        **common,
        use_cache=True,
    )
    full, full_trace = model.generate_image(
        **common,
        use_cache=False,
    )

    torch.testing.assert_close(cached, full, rtol=0, atol=0)
    assert cached_trace["attention_contract"] == "xlnet_content_diagonal"
    assert cached_trace["backbone_kv_cache_enabled"] is True
    assert full_trace["backbone_kv_cache_enabled"] is False


def test_text_ar_checkpoint_roundtrip_keeps_distinct_model_identity(tmp_path):
    model = SingleStreamTextARQwen3ForCausalLM(_tiny_config()).eval()
    model.save_pretrained(tmp_path)

    config = AutoConfig.from_pretrained(tmp_path)
    assert isinstance(config, SingleStreamTextARConfig)
    assert config.model_type == "selfless_flow_single_stream_text_ar"
    assert config.architecture_variant == "single_stream_text_ar"
    loaded = SingleStreamTextARQwen3ForCausalLM.from_pretrained(tmp_path)
    expected = model.state_dict()
    for name, value in loaded.state_dict().items():
        torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
