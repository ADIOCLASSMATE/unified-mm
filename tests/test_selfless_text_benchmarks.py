import os
from copy import deepcopy

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import torch
import pytest
from transformers import Qwen3Config

from models.modeling_model.modeling_selfless_flow import Qwen3ForCausalLM
from scripts.evaluate_selfless_text_benchmarks import (
    ChoiceRequest,
    MultipleChoiceExample,
    encode_choice,
    primary_metric,
    preprocess_hellaswag,
    score_choice_requests,
)
from scripts.summarize_unified_full_evaluation import (
    LM_EVAL_REFERENCE_COMMIT,
    TEXT_TASK_PROTOCOLS,
    validate_generation_summary,
    validate_text_summary,
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


def test_mmlu_primary_is_subject_macro_not_example_micro_accuracy():
    metrics = {
        "accuracy": 0.75,
        "accuracy_macro": 0.625,
        "accuracy_normalized": 0.80,
    }
    assert primary_metric("mmlu", metrics) == 0.625


def _formal_text_summary():
    tasks = {}
    primary = {}
    for task, (samples, metric_name) in TEXT_TASK_PROTOCOLS.items():
        metrics = {
            "schema": "selfless_text_multiple_choice_metrics_v1",
            "complete": True,
            "runtime_hashing_enabled": False,
            "task": task,
            "samples": samples,
            "accuracy": 0.5,
            "accuracy_normalized": 0.5,
            "truncated_context_samples": 0,
        }
        if task == "mmlu":
            metrics["accuracy_macro"] = 0.5
            metrics["by_category"] = {
                f"subject_{index:02d}": {
                    "samples": samples - 56 if index == 0 else 1,
                    "accuracy": 0.5,
                }
                for index in range(57)
            }
        tasks[task] = metrics
        primary[task] = metrics[metric_name]
    return {
        "schema": "selfless_text_benchmark_summary_v3",
        "accuracy_unit": "unit_interval",
        "tasks": tasks,
        "primary_metrics": primary,
        "macro_average_primary": 0.5,
        "macro_average_role": "internal_cross_task_summary_only",
        "protocol": {
            "protocol_schema": "selfless_text_benchmark_v2",
            "lm_eval_reference": {"commit": LM_EVAL_REFERENCE_COMMIT},
        },
    }


def test_formal_text_summary_requires_57_subject_mmlu_macro():
    summary = _formal_text_summary()
    validate_text_summary(summary, formal=True)

    invalid = deepcopy(summary)
    invalid["tasks"]["mmlu"]["by_category"].pop("subject_56")
    with pytest.raises(ValueError, match="57 subjects"):
        validate_text_summary(invalid, formal=True)


def _formal_generation_summary():
    return {
        "project_formal_protocol": True,
        "leaderboard_comparable_to_adm_dit": False,
        "protocol_name": "imagenet_val_fid50k_torch_fidelity_stratified_is",
        "reference_distribution": "imagenet_val_50000",
        "comparison_scope": "same_protocol_only",
        "not_adm_dit_reason": (
            "validation_reference_and_pytorch_torch_fidelity_extractor"
        ),
        "fid_reducer": "symmetric_eigendecomposition",
        "strategy": "spatial_halton",
        "samples": 50_000,
        "is_split_assignment": "stratified_by_synset",
        "is_split_plan": {"splits": 10},
        "fid": 5.0,
        "inception_score_mean": 2.0,
        "inception_score_std": 0.0,
        "inception_score_splits": [2.0] * 10,
    }


def test_full_summary_validates_project_generation_protocol():
    validate_generation_summary(_formal_generation_summary(), formal=True)

    invalid = _formal_generation_summary()
    invalid["leaderboard_comparable_to_adm_dit"] = True
    with pytest.raises(ValueError, match="comparability"):
        validate_generation_summary(invalid, formal=True)
