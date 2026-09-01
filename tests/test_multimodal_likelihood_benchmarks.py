import math
import os
from argparse import Namespace
from pathlib import Path

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import torch
from transformers import Qwen3Config

from models.modeling_model.modeling_selfless_flow import Qwen3ForCausalLM
from scripts.evaluate_multimodal_likelihood_benchmarks import (
    LikelihoodExample,
    PosteriorCache,
    arithmetic_seed,
    caption_perplexity_metrics,
    encode_candidate,
    mmbench_circular_metrics,
    paired_image_ranking_metrics,
    pairwise_ranking_metrics,
    score_candidate_requests,
    summarize_task,
    whatsup_metrics,
    winoground_metrics,
)
from scripts.evaluate_imagenet_pretraining_native import (
    retrieval_metrics,
    score_text_candidates,
    score_text_candidates_cached_prefix,
)


class CharacterTokenizer:
    eos_token_id = 2

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [10 + ord(character) % 89 for character in str(text) if character != " "]


def example(**overrides):
    values = {
        "item_index": 0,
        "item_id": "item-0",
        "task": "toy",
        "kind": "multiple_choice",
        "image_id": 1234,
        "prompt": "Question answer",
        "candidates": ("red", "blue"),
        "label": 0,
        "category": "color",
        "metadata": {},
    }
    values.update(overrides)
    return LikelihoodExample(**values)


def test_candidate_sequence_scores_same_position_suffix_after_complete_image():
    request = encode_candidate(
        CharacterTokenizer(),
        example(),
        0,
        image_tokens=4,
        boi_token_id=101,
        eoi_token_id=102,
        image_mask_token_id=103,
        max_length=64,
        image_sigma_order="sequential",
        seed=42,
    )
    assert request.input_ids[request.image_start - 1] == 101
    assert request.input_ids[request.image_start : request.image_start + 4] == (
        103,
        103,
        103,
        103,
    )
    assert request.input_ids[request.image_start + 4] == 102
    assert request.target_start == request.image_start + 5
    assert request.token_types[request.image_start : request.image_start + 4] == (
        1,
        1,
        1,
        1,
    )
    image_sigma = request.sigma[request.image_start : request.image_start + 4]
    target_sigma = request.sigma[request.target_start :]
    assert list(image_sigma) == sorted(image_sigma)
    assert min(target_sigma) > max(image_sigma)


def test_candidate_prompt_truncation_keeps_candidate_and_image_contract():
    request = encode_candidate(
        CharacterTokenizer(),
        example(prompt="x" * 100, candidates=("ok", "no")),
        0,
        image_tokens=4,
        boi_token_id=101,
        eoi_token_id=102,
        image_mask_token_id=103,
        max_length=16,
        image_sigma_order="random",
        seed=42,
    )
    assert len(request.input_ids) == 16
    assert request.truncated_prompt_tokens > 0
    assert request.target_start < len(request.input_ids)


def test_arithmetic_seeds_and_random_order_are_deterministic_without_hashes():
    assert arithmetic_seed(42, 99, 11) == arithmetic_seed(42, 99, 11)
    first = encode_candidate(
        CharacterTokenizer(),
        example(image_id=99),
        0,
        image_tokens=8,
        boi_token_id=101,
        eoi_token_id=102,
        image_mask_token_id=103,
        max_length=64,
        image_sigma_order="random",
        seed=42,
    )
    second = encode_candidate(
        CharacterTokenizer(),
        example(image_id=99),
        0,
        image_tokens=8,
        boi_token_id=101,
        eoi_token_id=102,
        image_mask_token_id=103,
        max_length=64,
        image_sigma_order="random",
        seed=42,
    )
    assert first.sigma == second.sigma


def score(ll, tokens):
    normalized = ll / tokens
    return {
        "loglikelihood": float(ll),
        "normalized_loglikelihood": float(normalized),
        "perplexity": math.exp(-normalized),
        "token_count": int(tokens),
        "greedy": False,
        "truncated_prompt_tokens": 0,
    }


def classification_row(item_index, label, scores, category="all"):
    raw = max(range(len(scores)), key=lambda index: scores[index]["loglikelihood"])
    normalized = max(
        range(len(scores)),
        key=lambda index: scores[index]["normalized_loglikelihood"],
    )
    return {
        "item_index": item_index,
        "item_id": str(item_index),
        "task": "toy",
        "kind": "multiple_choice",
        "image_id": item_index,
        "label": label,
        "category": category,
        "metadata": {},
        "candidate_scores": scores,
        "prediction_raw": raw,
        "prediction_normalized": normalized,
        "correct_raw": raw == label,
        "correct_normalized": normalized == label,
    }


def test_primary_multiple_choice_metric_uses_length_normalized_likelihood():
    row = classification_row(0, 0, [score(-4.0, 4), score(-3.0, 2)])
    summary = summarize_task([row])
    metrics = summary["metrics"]
    assert metrics["accuracy_raw_loglikelihood"] == 0.0
    assert metrics["accuracy_normalized_loglikelihood"] == 1.0
    assert metrics["primary_metric"] == "accuracy_normalized_loglikelihood"


def test_pairwise_ranking_reports_strict_tie_instead_of_index_tie_break_win():
    row = classification_row(0, 0, [score(-2.0, 2), score(-2.0, 2)])
    row["kind"] = "pairwise_caption_ranking"
    metrics = pairwise_ranking_metrics([row])
    assert metrics["accuracy_normalized_loglikelihood"] == 1.0
    assert metrics["normalized_pairwise"] == {
        "win_rate": 0.0,
        "tie_rate": 1.0,
        "loss_rate": 0.0,
        "mean_margin": 0.0,
        "median_margin": 0.0,
    }
    assert metrics["primary_metric"] == "normalized_pairwise.win_rate"


def test_mmbench_strict_circular_metric_requires_every_rotation():
    rows = []
    for item_index, original_index, is_original, label in (
        (0, 10, True, 0),
        (1, 10, False, 0),
        (2, 20, True, 0),
        (3, 20, False, 1),
    ):
        row = classification_row(
            item_index,
            label,
            [score(-1.0, 1), score(-2.0, 1)],
        )
        row["kind"] = "mmbench_circular_multiple_choice"
        row["metadata"] = {
            "original_index": original_index,
            "is_original": is_original,
        }
        rows.append(row)
    metrics = mmbench_circular_metrics(rows)
    assert metrics["original_questions"] == 2
    assert metrics["vanilla_accuracy_normalized_loglikelihood"] == 1.0
    assert metrics["circular_accuracy_normalized_loglikelihood"] == 0.5
    assert metrics["primary_metric"] == (
        "circular_accuracy_normalized_loglikelihood"
    )


def test_caption_perplexity_is_token_weighted():
    rows = []
    for index, value in enumerate((score(-2.0, 1), score(-6.0, 3))):
        rows.append(
            {
                "kind": "caption_perplexity",
                "candidate_scores": [value],
            }
        )
    metrics = caption_perplexity_metrics(rows)
    assert metrics["tokens"] == 4
    assert metrics["token_nll"] == 2.0
    assert metrics["token_perplexity"] == math.exp(2.0)


def test_winoground_reports_text_image_and_group_scores():
    def row(group, slot, left, right):
        return {
            "kind": "winoground_pair",
            "metadata": {"group_id": group, "image_slot": slot},
            "candidate_scores": [score(left, 1), score(right, 1)],
        }

    metrics = winoground_metrics(
        [
            row("g0", 0, -1.0, -2.0),
            row("g0", 1, -3.0, -1.0),
        ]
    )
    assert metrics["groups"] == 1
    assert metrics["normalized"] == {
        "text_score": 1.0,
        "image_score": 1.0,
        "group_score": 1.0,
    }


def test_svo_pairing_reports_shared_caption_image_win_and_categories():
    def row(pair, role, negative_type, value):
        return {
            "kind": "svo_image_pair",
            "category": negative_type,
            "metadata": {
                "pair_id": pair,
                "image_role": role,
                "negative_type": negative_type,
            },
            "candidate_scores": [score(value, 2)],
        }

    metrics = paired_image_ranking_metrics(
        [
            row("s0", "positive", "subject", -2.0),
            row("s0", "negative", "subject", -4.0),
            row("s1", "positive", "verb", -6.0),
            row("s1", "negative", "verb", -4.0),
        ]
    )
    assert metrics["pairs"] == 2
    assert metrics["normalized"]["win_rate"] == 0.5
    assert metrics["normalized"]["categories"]["subject"]["win_rate"] == 1.0
    assert metrics["normalized"]["categories"]["verb"]["win_rate"] == 0.0


def test_whatsup_reports_official_individual_pair_and_set_accuracy():
    relations = ("left", "right", "on", "under")
    rows = []
    for index, relation in enumerate(relations):
        rows.append(
            {
                "kind": "whatsup_controlled_spatial",
                "label": 0,
                "prediction_raw": 0,
                "prediction_normalized": 1 if relation == "under" else 0,
                "metadata": {
                    "subset": "A",
                    "set_id": "cup::0",
                    "relation": relation,
                },
            }
        )
    metrics = whatsup_metrics(rows)
    assert metrics["sets"] == 1
    assert metrics["raw"]["individual_accuracy"] == 1.0
    assert metrics["raw"]["pair_accuracy"] == 1.0
    assert metrics["raw"]["set_accuracy"] == 1.0
    assert metrics["normalized"]["individual_accuracy"] == 0.75
    assert metrics["normalized"]["pair_accuracy"] == 0.5
    assert metrics["normalized"]["set_accuracy"] == 0.0


def test_retrieval_metrics_report_instance_and_same_class_relevance():
    matrix = torch.tensor(
        [
            [3.0, 2.0, 0.0],
            [2.5, 2.0, 0.0],
            [0.0, 0.5, 3.0],
        ]
    )
    metrics = retrieval_metrics(
        matrix,
        torch.tensor([0, 0, 1]),
    )
    i2t = metrics["normalized_loglikelihood"]["image_to_text"]
    assert math.isclose(i2t["instance_recall_at_1"], 2 / 3, rel_tol=1.0e-6)
    assert i2t["class_relevance_recall_at_1"] == 1.0


def test_posterior_cache_uses_deterministic_arithmetic_sampling(tmp_path):
    shard_dir = tmp_path / "cache"
    shard_dir.mkdir()
    torch.save(
        {
            "posterior_stats": torch.cat(
                [
                    torch.ones(1, 4, 2, dtype=torch.float16),
                    torch.full((1, 4, 2), 0.5, dtype=torch.float16),
                ],
                dim=-1,
            ),
            "img_ids": torch.tensor([77], dtype=torch.long),
            "metadata": {
                "format": "imagenet_kl16_scaled_posterior_v1",
                "stats_layout": "scaled_mean_then_scaled_std",
                "runtime_hashing_enabled": False,
            },
        },
        shard_dir / "shard-00000-of-00001.pt",
    )
    cache = PosteriorCache(
        shard_dir,
        expected_image_tokens=4,
        expected_latent_dim=2,
        seed=42,
    )
    assert torch.equal(cache.sample(77), cache.sample(77))
    assert tuple(cache.sample(77).shape) == (4, 2)


def test_tiny_multimodal_forward_scores_candidate_at_same_position(tmp_path):
    config = Qwen3Config(
        vocab_size=40,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
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
        "training_objective": "selfless_dual_stream",
        "dual_stream_attention_contract": "selfless_strict",
    }
    for key, value in values.items():
        setattr(config, key, value)
    model = Qwen3ForCausalLM(config).eval()

    shard_dir = tmp_path / "tiny-cache"
    shard_dir.mkdir()
    torch.save(
        {
            "posterior_stats": torch.cat(
                [torch.zeros(1, 4, 4), torch.ones(1, 4, 4) * 0.1], dim=-1
            ).half(),
            "img_ids": torch.tensor([77]),
            "metadata": {
                "format": "imagenet_kl16_scaled_posterior_v1",
                "stats_layout": "scaled_mean_then_scaled_std",
                "runtime_hashing_enabled": False,
            },
        },
        shard_dir / "shard-00000-of-00001.pt",
    )
    cache = PosteriorCache(
        shard_dir,
        expected_image_tokens=4,
        expected_latent_dim=4,
        seed=42,
    )

    class TinyTokenizer:
        eos_token_id = 2

        def encode(self, text, add_special_tokens=False):
            assert add_special_tokens is False
            return [3 + ord(character) % 20 for character in text if character != " "]

    item = example(image_id=77)
    requests = [
        encode_candidate(
            TinyTokenizer(),
            item,
            candidate_index,
            image_tokens=4,
            boi_token_id=11,
            eoi_token_id=12,
            image_mask_token_id=8,
            max_length=64,
            image_sigma_order="random",
            seed=42,
        )
        for candidate_index in range(2)
    ]
    backend_args = Namespace(
        request_chunk_size=8,
        batch_size_per_rank=2,
        lm_head_chunk_tokens=8,
        max_length=64,
        seed=42,
        scoring_backend="cached_prefix",
    )
    backend_kwargs = {
        "model": model,
        "tokenizer": TinyTokenizer(),
        "cache": cache,
        "image_id": 77,
        "item_id": "cached-prefix-toy",
        "prompt": item.prompt,
        "candidates": item.candidates,
        "args": backend_args,
        "device": torch.device("cpu"),
        "image_sigma_order": "random",
    }
    for attention_contract in (
        "selfless_strict",
        "xlnet_content_diagonal",
    ):
        scores = score_candidate_requests(
            model,
            requests,
            cache,
            batch_size=2,
            lm_head_chunk_tokens=8,
            attention_contract=attention_contract,
            device=torch.device("cpu"),
        )
        assert [value.token_count for value in scores] == [3, 4]
        assert all(math.isfinite(value.loglikelihood) for value in scores)

        contract_kwargs = {
            **backend_kwargs,
            "attention_contract": attention_contract,
        }
        reference = score_text_candidates(**contract_kwargs)
        cached = score_text_candidates_cached_prefix(**contract_kwargs)
        assert torch.equal(reference[0], cached[0])
        assert torch.equal(reference[1], cached[1])


def test_new_runtime_paths_do_not_import_digest_implementations():
    repository = Path(__file__).resolve().parents[1]
    for relative in (
        "scripts/prepare_multimodal_likelihood_assets.py",
        "scripts/evaluate_multimodal_likelihood_benchmarks.py",
    ):
        source = (repository / relative).read_text(encoding="utf-8")
        assert "hashlib" not in source
        assert "sha256" not in source.lower()


def test_ascend_cache_validation_uses_torch_npu_compatible_mmap_path():
    repository = Path(__file__).resolve().parents[1]
    source = (
        repository
        / "script/selfless/prepare_multimodal_likelihood_cache_ascend16.sh"
    ).read_text(encoding="utf-8")
    assert 'torch.load(str(path), map_location="cpu", mmap=True' in source
    assert "--no_hash" in source
