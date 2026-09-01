import os

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import torch
from transformers import Qwen3Config

from models.modeling_model.modeling_selfless_flow import Qwen3ForCausalLM
from scripts.evaluate_selfless_text_benchmarks import (
    ChoiceRequest,
    DEFAULT_TASKS,
    MultipleChoiceExample,
    encode_choice,
    preprocess_hellaswag,
    score_choice_requests,
)
from utils.utils import get_selfless_mask


class _CharacterTokenizer:
    eos_token_id = 0

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [ord(character) for character in text]


def _tiny_model(attention_contract="selfless_strict"):
    config = Qwen3Config(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
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
        "lambda_text": 0.05,
        "lambda_image": 1.0,
        "use_flex_attention": True,
        "dual_stream_attention_contract": attention_contract,
    }
    for key, value in values.items():
        setattr(config, key, value)
    return Qwen3ForCausalLM(config).eval()


def test_choice_encoding_marks_only_continuation_and_left_truncates_context():
    example = MultipleChoiceExample(
        item_index=0,
        item_id="example",
        task="unit",
        context="abcdef",
        choices=("gh",),
        label=0,
    )
    request = encode_choice(_CharacterTokenizer(), example, 0, max_length=5)
    assert request.input_ids == tuple(ord(character) for character in "ef gh")
    assert request.target_start == 2
    assert request.truncated_context_tokens == 4
    assert request.boundary_adjusted is False


def test_choice_scorer_uses_same_position_query_stream_logits():
    torch.manual_seed(3)
    model = _tiny_model()
    request = ChoiceRequest(
        item_index=0,
        choice_index=0,
        input_ids=(3, 4, 5),
        target_start=2,
        boundary_adjusted=False,
        truncated_context_tokens=0,
    )
    actual = score_choice_requests(
        model,
        [request],
        batch_size=1,
        lm_head_chunk_tokens=8,
        device=torch.device("cpu"),
    )[0]

    input_ids = torch.tensor([[3, 4, 5]], dtype=torch.long)
    sigma = torch.arange(3, dtype=torch.float32).unsqueeze(0)
    attention_mask = get_selfless_mask(
        sigma=sigma,
        seq_len=3,
        device=torch.device("cpu"),
    )
    logits = model(
        X0_input_ids=input_ids,
        attention_mask=attention_mask,
        calculate_likelihood=True,
    ).logits
    expected = torch.log_softmax(logits[0, 2].float(), dim=-1)[5].item()
    assert actual.token_count == 1
    assert abs(actual.loglikelihood - expected) < 1.0e-6


def test_choice_scorer_uses_b_diagonal_content_and_strict_query_masks():
    torch.manual_seed(13)
    model = _tiny_model("xlnet_content_diagonal")
    request = ChoiceRequest(
        item_index=0,
        choice_index=0,
        input_ids=(3, 4, 5),
        target_start=2,
        boundary_adjusted=False,
        truncated_context_tokens=0,
    )
    actual = score_choice_requests(
        model,
        [request],
        batch_size=1,
        lm_head_chunk_tokens=8,
        device=torch.device("cpu"),
    )[0]

    input_ids = torch.tensor([[3, 4, 5]], dtype=torch.long)
    sigma = torch.arange(3, dtype=torch.float32).unsqueeze(0)
    query_mask = get_selfless_mask(sigma, 3, "cpu")
    content_mask = get_selfless_mask(
        sigma,
        3,
        "cpu",
        include_diagonal=True,
    )
    expected_logits = model(
        X0_input_ids=input_ids,
        attention_mask=query_mask,
        content_attention_mask=content_mask,
        calculate_likelihood=True,
    ).logits
    expected = torch.log_softmax(
        expected_logits[0, 2].float(), dim=-1
    )[5].item()
    assert abs(actual.loglikelihood - expected) < 1.0e-6

    wrong_logits = model(
        X0_input_ids=input_ids,
        attention_mask=query_mask,
        calculate_likelihood=True,
    ).logits
    wrong = torch.log_softmax(wrong_logits[0, 2].float(), dim=-1)[5].item()
    assert actual.loglikelihood != wrong


def test_hellaswag_preprocessing_matches_public_task_contract():
    assert preprocess_hellaswag(" x [title] [artifact]  y ") == "x. y"
