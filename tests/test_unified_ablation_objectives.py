import math

import pytest
import torch
import torch.nn.functional as F
from transformers import Qwen3Config

from models.modeling_model.modeling_selfless_flow import Qwen3ForCausalLM
from utils.utils import (
    get_selfless_mask,
    get_showo_mae_mask,
    sample_showo_mae_image_mask,
)


def _allowed(mask, q_idx: int, kv_idx: int) -> bool:
    value = mask.mask_mod(
        torch.tensor(0),
        torch.tensor(0),
        torch.tensor(q_idx),
        torch.tensor(kv_idx),
    )
    return bool(value.item() if torch.is_tensor(value) else value)


def _tiny_model(
    objective: str, *, num_hidden_layers: int = 1
) -> Qwen3ForCausalLM:
    config = Qwen3Config(
        vocab_size=40,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=32,
        tie_word_embeddings=True,
    )
    values = {
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
        "training_objective": objective,
        "dual_stream_attention_contract": "selfless_strict",
    }
    for key, value in values.items():
        setattr(config, key, value)
    return Qwen3ForCausalLM(config)


def test_xlnet_b_content_mask_has_diagonal_but_query_mask_does_not():
    sigma = torch.tensor([[0, 1, 2, 3]])
    segment_ids = torch.zeros_like(sigma)
    strict = get_selfless_mask(
        sigma,
        4,
        "cpu",
        segment_ids=segment_ids,
    )
    content = get_selfless_mask(
        sigma,
        4,
        "cpu",
        segment_ids=segment_ids,
        include_diagonal=True,
    )
    for index in range(4):
        assert not _allowed(strict, index, index)
        assert _allowed(content, index, index)
    assert _allowed(strict, 3, 2)
    assert not _allowed(content, 2, 3)


def test_xlnet_b_hybrid_generation_mask_only_adds_content_self_edges():
    sigma = torch.tensor([[0, 1, 1, 2]])
    content_queries = torch.tensor([[True, True, False, True]])
    hybrid = get_selfless_mask(
        sigma,
        4,
        "cpu",
        diagonal_query_mask=content_queries,
    )

    assert _allowed(hybrid, 0, 0)
    assert _allowed(hybrid, 1, 1)
    assert not _allowed(hybrid, 2, 2)
    assert _allowed(hybrid, 3, 3)
    # Equal generation sigma is not a content edge unless it is the same
    # physical token.  Unrevealed mask positions must not see one another.
    assert not _allowed(hybrid, 1, 2)
    assert not _allowed(hybrid, 2, 1)


@pytest.mark.parametrize(
    "attention_contract",
    ["selfless_strict", "xlnet_content_diagonal"],
)
def test_ab_hybrid_single_stream_matches_dual_stream_query(
    attention_contract,
):
    torch.manual_seed(17)
    model = _tiny_model(
        "selfless_dual_stream", num_hidden_layers=2
    ).eval()
    model.config.dual_stream_attention_contract = attention_contract
    input_ids = torch.tensor([[3, 11, 8, 8, 8, 8, 12, 0]])
    token_types = torch.tensor(
        [[0, 2, 1, 1, 1, 1, 2, 3]], dtype=torch.uint8
    )
    sigma = torch.tensor([[0.0, 1.0, 3.0, 4.0, 7.0, 7.0, 2.0, 8.0]])
    image_latents = torch.randn(1, 8, 4)
    visible_content = torch.tensor(
        [[False, False, True, True, False, False, False, False]]
    )
    content_queries = token_types.ne(3) & (
        token_types.ne(1) | visible_content
    )
    query_mask = get_selfless_mask(sigma, 8, "cpu")
    content_mask = get_selfless_mask(
        sigma,
        8,
        "cpu",
        include_diagonal=attention_contract == "xlnet_content_diagonal",
    )
    hybrid_mask = get_selfless_mask(
        sigma,
        8,
        "cpu",
        diagonal_query_mask=(
            content_queries
            if attention_contract == "xlnet_content_diagonal"
            else None
        ),
    )

    dual_layers = []
    handles = [
        layer.register_forward_hook(
            lambda _module, _inputs, output: dual_layers.append(
                (output[0].detach().clone(), output[1].detach().clone())
            )
        )
        for layer in model.model.layers
    ]
    dual_query = model.model(
        X0_input_ids=input_ids,
        attention_mask=query_mask,
        content_attention_mask=content_mask,
        token_types=token_types,
        image_latents=image_latents,
        calculate_likelihood=True,
    ).last_hidden_state
    for handle in handles:
        handle.remove()

    hybrid_layers = []
    handles = [
        layer.register_forward_hook(
            lambda _module, _inputs, output: hybrid_layers.append(
                output[0].detach().clone()
            )
        )
        for layer in model.model.layers
    ]
    hybrid = model.model(
        X0_input_ids=input_ids,
        attention_mask=hybrid_mask,
        token_types=token_types,
        image_latents=image_latents,
        image_latent_mask=visible_content,
        calculate_likelihood=False,
    ).last_hidden_state
    for handle in handles:
        handle.remove()

    valid_rows = token_types.ne(3)
    for (dual_content, dual_query_layer), hybrid_layer in zip(
        dual_layers, hybrid_layers, strict=True
    ):
        mixed_reference = torch.where(
            content_queries.unsqueeze(-1),
            dual_content,
            dual_query_layer,
        )
        assert torch.equal(
            hybrid_layer[valid_rows], mixed_reference[valid_rows]
        )

    # The final hidden at every valid row is bitwise identical: visible rows
    # select dual-stream X0 and masked image rows select dual-stream XT.
    dual_content_final = model.model.norm(dual_layers[-1][0])
    mixed_final = torch.where(
        content_queries.unsqueeze(-1),
        dual_content_final,
        dual_query,
    )
    assert torch.equal(hybrid[valid_rows], mixed_final[valid_rows])


def test_xlnet_b_content_diagonal_reaches_later_query_layers():
    torch.manual_seed(9)
    model = _tiny_model(
        "selfless_dual_stream", num_hidden_layers=2
    ).train()
    input_ids = torch.tensor([[3, 4, 5, 6]])
    token_types = torch.zeros_like(input_ids, dtype=torch.uint8)
    sigma = torch.arange(4).unsqueeze(0)
    query_mask = get_selfless_mask(
        sigma,
        4,
        "cpu",
        segment_ids=torch.zeros_like(sigma),
    )
    content_mask = get_selfless_mask(
        sigma,
        4,
        "cpu",
        segment_ids=torch.zeros_like(sigma),
        include_diagonal=True,
    )
    common = dict(
        X0_input_ids=input_ids,
        attention_mask=query_mask,
        token_types=token_types,
        image_span_table=torch.empty(0, 5, dtype=torch.long),
        calculate_likelihood=True,
    )
    selfless_hidden = model.model(**common).last_hidden_state
    xlnet_hidden = model.model(
        **common, content_attention_mask=content_mask
    ).last_hidden_state
    assert torch.isfinite(xlnet_hidden).all()
    assert not torch.allclose(selfless_hidden, xlnet_hidden)


def test_showo_omni_mask_is_causal_for_text_and_full_for_image():
    input_ids = torch.tensor([[11, 8, 8, 12, 5]])
    token_types = torch.tensor([[2, 1, 1, 2, 0]], dtype=torch.uint8)
    mask = get_showo_mae_mask(
        input_ids=input_ids,
        token_types=token_types,
        device="cpu",
        boi_token_id=11,
    )
    assert _allowed(mask, 1, 0)
    assert _allowed(mask, 1, 1)
    assert _allowed(mask, 1, 2)
    assert not _allowed(mask, 1, 3)
    assert _allowed(mask, 4, 4)
    assert _allowed(mask, 4, 2)
    assert not _allowed(mask, 3, 4)

    uncond = get_showo_mae_mask(
        input_ids=input_ids,
        token_types=token_types,
        device="cpu",
        boi_token_id=11,
        image_uncond_rows=torch.tensor([True]),
    )
    assert not _allowed(uncond, 1, 0)
    assert _allowed(uncond, 1, 2)


def test_showo_mask_sampling_matches_official_cosine_random_permutation():
    span_table = torch.tensor([[0, 0, 1, 4, 0], [1, 0, 1, 4, 1]])
    eligible = torch.zeros(2, 6, dtype=torch.bool)
    eligible[0, 1:5] = True
    generator = torch.Generator().manual_seed(123)
    sampled, ratios = sample_showo_mae_image_mask(
        image_span_table=span_table,
        full_image_loss_mask=eligible,
        image_tokens_per_img=4,
        generator=generator,
    )

    reference = torch.Generator().manual_seed(123)
    timesteps = torch.rand(2, generator=reference)
    expected_ratios = torch.cos(timesteps * (math.pi * 0.5))
    counts = (4 * expected_ratios).round().clamp(min=1, max=4)
    order = torch.rand(2, 4, generator=reference).argsort(dim=-1)
    expected_local = (order < counts.unsqueeze(-1)) & eligible[:, 1:5]
    assert torch.equal(ratios, expected_ratios)
    assert torch.equal(sampled[:, 1:5], expected_local)
    assert sampled[0].sum() == counts[0]
    assert sampled[1].sum() == 0


def test_showo_single_stream_text_uses_next_token_targets():
    model = _tiny_model("showo_mae_flow").train()
    input_ids = torch.tensor([[3, 4, 5, 6]])
    token_types = torch.zeros_like(input_ids, dtype=torch.uint8)
    labels = input_ids.clone()
    labels[:, 0] = -100
    mask = get_showo_mae_mask(
        input_ids=input_ids,
        token_types=token_types,
        device="cpu",
        boi_token_id=11,
        segment_ids=torch.zeros_like(input_ids),
    )
    output = model(
        X0_input_ids=input_ids,
        labels=labels,
        attention_mask=mask,
        token_types=token_types,
        image_span_table=torch.empty(0, 5, dtype=torch.long),
        image_loss_mask=torch.zeros_like(input_ids, dtype=torch.bool),
        compute_text_loss=True,
        compute_image_loss=False,
        return_logits=False,
    )
    manual = F.cross_entropy(
        model.lm_head(output.last_hidden_state[:, :-1]).reshape(-1, 40),
        labels[:, 1:].reshape(-1),
    )
    assert output.per_modality_count["text_tokens"].item() == 3
    assert torch.allclose(
        output.per_modality_loss["text_loss"], manual, atol=1.0e-6
    )


def test_showo_masked_clean_latent_cannot_enter_backbone_content():
    model = _tiny_model("showo_mae_flow").train()
    input_ids = torch.tensor([[11, 8, 8, 8, 8, 12]])
    token_types = torch.tensor([[2, 1, 1, 1, 1, 2]], dtype=torch.uint8)
    visible = torch.tensor([[False, True, False, True, False, False]])
    latents = torch.randn(1, 6, 4)
    changed = latents.clone()
    changed[:, 2] += 1000.0
    changed[:, 4] -= 1000.0
    mask = get_showo_mae_mask(
        input_ids=input_ids,
        token_types=token_types,
        device="cpu",
        boi_token_id=11,
    )
    kwargs = dict(
        X0_input_ids=input_ids,
        attention_mask=mask,
        token_types=token_types,
        image_latent_mask=visible,
        image_span_table=torch.tensor([[0, 0, 1, 4, 0]]),
        calculate_likelihood=True,
    )
    first = model.model(image_latents=latents, **kwargs).last_hidden_state
    second = model.model(image_latents=changed, **kwargs).last_hidden_state
    assert torch.equal(first, second)


def test_showo_flow_context_excludes_masked_clean_latents():
    model = _tiny_model("showo_mae_flow").eval()
    net = model.image_flow_head.net
    x = torch.randn(1, 4, 4)
    t = torch.rand(1, 4)
    condition = torch.randn(1, 4, 16)
    context = torch.randn(1, 4, 4)
    changed = context.clone()
    changed[:, 1] += 1000.0
    changed[:, 3] -= 1000.0
    visible = torch.tensor([[True, False, True, False]])
    context_mask = visible.unsqueeze(1).expand(-1, 4, -1)
    positions = torch.arange(4).unsqueeze(0)
    kwargs = dict(
        x=x,
        t=t,
        c=condition,
        context_mask=context_mask,
        query_positions=positions,
        context_positions=positions,
        context_conditions=condition,
    )
    first = net(context_latents=context, **kwargs)
    second = net(context_latents=changed, **kwargs)
    assert torch.equal(first, second)


def test_showo_mae_flow_end_to_end_trains_only_sampled_image_positions():
    model = _tiny_model("showo_mae_flow").train()
    input_ids = torch.tensor([[3, 11, 8, 8, 8, 8, 12, 9]])
    token_types = torch.tensor(
        [[0, 2, 1, 1, 1, 1, 2, 0]], dtype=torch.uint8
    )
    visible = torch.tensor(
        [[False, False, True, False, True, False, False, False]]
    )
    masked = token_types.eq(1) & ~visible
    latents = torch.zeros(1, 8, 4)
    latents[:, 2:6] = torch.randn(1, 4, 4)
    attention_mask = get_showo_mae_mask(
        input_ids=input_ids,
        token_types=token_types,
        device="cpu",
        boi_token_id=11,
    )
    output = model(
        X0_input_ids=input_ids,
        labels=input_ids.clone(),
        attention_mask=attention_mask,
        token_types=token_types,
        image_latents=latents,
        image_latent_mask=visible,
        image_local_positions=torch.tensor(
            [[-1, -1, 0, 1, 2, 3, -1, -1]]
        ),
        image_span_table=torch.tensor([[0, 0, 2, 6, 0]]),
        image_loss_mask=masked,
        compute_text_loss=False,
        compute_image_loss=True,
        record_flow_stats=True,
        return_logits=False,
    )
    assert torch.isfinite(output.loss)
    assert output.per_modality_count["text_tokens"].item() == 0
    assert output.per_modality_count["image_tokens"].item() == 2
    assert torch.isfinite(output.per_modality_loss["image_loss"])
    output.loss.backward()
    gradient = model.image_flow_head.net.final_layer.linear.weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
