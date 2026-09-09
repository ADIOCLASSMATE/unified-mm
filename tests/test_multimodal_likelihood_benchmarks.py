import math
import os
from argparse import Namespace
from pathlib import Path

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import pytest
import torch
from transformers import Qwen3Config

from models.modeling_model.modeling_selfless_flow import Qwen3ForCausalLM
from utils.evaluation.multimodal_likelihood import (
    CandidateScore,
    DEBIASED_SCORE,
    LikelihoodExample,
    PosteriorCache,
    arithmetic_seed,
    build_prediction_rows,
    encode_candidate,
    encode_candidate_mc,
    mmbench_circular_metrics,
    paired_image_ranking_metrics,
    pairwise_ranking_metrics,
    score_candidate_requests,
    summarize_task,
    validate_language_prior_contract,
    whatsup_metrics,
    winoground_metrics,
)
from utils.evaluation.native_understanding import (
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


def test_formal_null_image_contract_is_fully_pinned(tmp_path):
    images = []
    for index, seed in enumerate((17_071, 29_129, 43_231)):
        path = tmp_path / f"gaussian-{index:02d}.png"
        path.touch()
        images.append(
            {
                "image_id": 9_000_000_000 + index,
                "path": str(path),
                "seed": seed,
            }
        )
    prior = {
        "estimator": "content_free_gaussian_image_logmeanexp",
        "count": 3,
        "images": images,
        "pixel_space": "vae_preprocess_normalized_minus1_to_plus1",
        "normalized_gaussian_mean": 0.0,
        "normalized_gaussian_std": 0.25,
        "clamp": [-1.0, 1.0],
        "storage": "lossless_rgb_png",
        "uses_benchmark_labels": False,
    }
    validate_language_prior_contract(prior)

    prior["images"][0]["seed"] += 1
    with pytest.raises(ValueError, match="seeds"):
        validate_language_prior_contract(prior)


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


def test_mc_expands_image_orders_and_shares_them_across_candidates():
    kwargs = {
        "image_tokens": 8,
        "boi_token_id": 101,
        "eoi_token_id": 102,
        "image_mask_token_id": 103,
        "max_length": 64,
        "image_sigma_order": "random",
        "seed": 42,
        "mc_samples": 64,
    }
    candidates = [
        encode_candidate_mc(CharacterTokenizer(), example(), index, **kwargs)
        for index in range(2)
    ]
    mc_one = encode_candidate(
        CharacterTokenizer(),
        example(),
        0,
        **{key: value for key, value in kwargs.items() if key != "mc_samples"},
    )
    assert candidates[0][0] == mc_one
    assert all(len(requests) == 64 for requests in candidates)
    assert [request.mc_sample_index for request in candidates[0]] == list(range(64))
    assert len({request.sigma for request in candidates[0]}) > 1
    for left, right in zip(*candidates):
        left_order = left.sigma[left.image_start : left.image_start + 8]
        right_order = right.sigma[right.image_start : right.image_start + 8]
        assert left_order == right_order
    assert candidates[0][0].input_ids is candidates[0][1].input_ids


def test_mc_prediction_uses_mean_loglikelihood_and_records_dispersion():
    requests = [
        request
        for candidate_index in range(2)
        for request in encode_candidate_mc(
            CharacterTokenizer(),
            example(),
            candidate_index,
            image_tokens=4,
            boi_token_id=101,
            eoi_token_id=102,
            image_mask_token_id=103,
            max_length=64,
            image_sigma_order="random",
            seed=42,
            mc_samples=2,
        )
    ]
    scores = [
        CandidateScore(-3.0, -1.0, 3, True),
        CandidateScore(-6.0, -2.0, 3, False),
        CandidateScore(-4.0, -1.0, 4, True),
        CandidateScore(-4.0, -1.0, 4, True),
    ]
    null_image_ids = (9_000_000_000, 9_000_000_001, 9_000_000_002)
    prior_requests = [
        request
        for candidate_index in range(2)
        for null_image_id in null_image_ids
        for request in encode_candidate_mc(
            CharacterTokenizer(),
            example(image_id=null_image_id),
            candidate_index,
            image_tokens=4,
            boi_token_id=101,
            eoi_token_id=102,
            image_mask_token_id=103,
            max_length=64,
            image_sigma_order="random",
            seed=42,
            mc_samples=2,
        )
    ]
    prior_scores = []
    for candidate_index in range(2):
        token_count = 3 if candidate_index == 0 else 4
        prior_value = -2.0 if candidate_index == 0 else -0.5
        for _ in null_image_ids:
            prior_scores.extend(
                [
                    CandidateScore(
                        prior_value * token_count,
                        prior_value,
                        token_count,
                        False,
                    ),
                    CandidateScore(
                        prior_value * token_count,
                        prior_value,
                        token_count,
                        False,
                    ),
                ]
            )
    row = build_prediction_rows(
        [example()],
        requests,
        scores,
        prior_requests=prior_requests,
        prior_scores=prior_scores,
        null_image_ids=null_image_ids,
        mc_samples=2,
    )[0]
    first, second = row["candidate_scores"]
    assert first[DEBIASED_SCORE] == 0.5
    assert second[DEBIASED_SCORE] == -0.5
    assert first["estimated_language_prior_log_score"] == -2.0
    assert first["conditional_mc_mean_token_loglikelihood_std"] == 0.5
    assert "normalized_loglikelihood" not in first
    assert "loglikelihood" not in first
    assert second["mc_samples"] == 2
    assert row["prediction_language_prior_debiased"] == 0


def score(value, tokens=1):
    return {
        DEBIASED_SCORE: float(value),
        "token_count": int(tokens),
        "truncated_prompt_tokens": 0,
        "mc_samples": 1,
        "language_prior_null_images": 3,
    }


def classification_row(item_index, label, scores, category="all"):
    prediction = max(range(len(scores)), key=lambda index: scores[index][DEBIASED_SCORE])
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
        "prediction_language_prior_debiased": prediction,
        "correct_language_prior_debiased": prediction == label,
    }


def test_primary_multiple_choice_metric_uses_only_debiased_likelihood():
    row = classification_row(0, 0, [score(0.4, 4), score(0.3, 2)])
    summary = summarize_task([row])
    metrics = summary["metrics"]
    assert metrics["accuracy_language_prior_debiased"] == 1.0
    assert metrics["primary_metric"] == "accuracy_language_prior_debiased"


def test_pairwise_ranking_reports_strict_tie_instead_of_index_tie_break_win():
    row = classification_row(0, 0, [score(0.0, 2), score(0.0, 2)])
    row["kind"] = "pairwise_caption_ranking"
    metrics = pairwise_ranking_metrics([row])
    assert metrics["accuracy_language_prior_debiased"] == 1.0
    assert metrics["language_prior_debiased_pairwise"] == {
        "win_rate": 0.0,
        "tie_rate": 1.0,
        "loss_rate": 0.0,
        "mean_margin": 0.0,
        "median_margin": 0.0,
    }
    assert metrics["primary_metric"] == "language_prior_debiased_pairwise.win_rate"


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
            [score(1.0), score(0.0)],
        )
        row["kind"] = "mmbench_circular_multiple_choice"
        row["metadata"] = {
            "original_index": original_index,
            "is_original": is_original,
        }
        rows.append(row)
    metrics = mmbench_circular_metrics(rows)
    assert metrics["original_questions"] == 2
    assert metrics["vanilla_accuracy_language_prior_debiased"] == 1.0
    assert metrics["circular_accuracy_language_prior_debiased"] == 0.5
    assert metrics["primary_metric"] == (
        "circular_accuracy_language_prior_debiased"
    )


def test_winoground_reports_text_image_and_group_scores():
    def row(group, slot, left, right):
        return {
            "kind": "winoground_pair",
            "metadata": {"group_id": group, "image_slot": slot},
            "candidate_scores": [score(left, 1), score(right, 1)],
        }

    metrics = winoground_metrics(
        [
            row("g0", 0, 2.0, 1.0),
            row("g0", 1, 0.0, 2.0),
        ]
    )
    assert metrics["groups"] == 1
    assert metrics["language_prior_debiased"] == {
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
            row("s0", "positive", "subject", 2.0),
            row("s0", "negative", "subject", 1.0),
            row("s1", "positive", "verb", 0.0),
            row("s1", "negative", "verb", 1.0),
        ]
    )
    assert metrics["pairs"] == 2
    assert metrics["language_prior_debiased"]["win_rate"] == 0.5
    assert metrics["language_prior_debiased"]["categories"]["subject"]["win_rate"] == 1.0
    assert metrics["language_prior_debiased"]["categories"]["verb"]["win_rate"] == 0.0


def test_whatsup_reports_official_individual_pair_and_set_accuracy():
    relations = ("left", "right", "on", "under")
    rows = []
    for index, relation in enumerate(relations):
        rows.append(
            {
                "kind": "whatsup_controlled_spatial",
                "label": 0,
                "prediction_language_prior_debiased": (
                    1 if relation == "under" else 0
                ),
                "metadata": {
                    "subset": "A",
                    "set_id": "cup::0",
                    "relation": relation,
                },
            }
        )
    metrics = whatsup_metrics(rows)
    assert metrics["sets"] == 1
    assert metrics["language_prior_debiased"]["individual_accuracy"] == 0.75
    assert metrics["language_prior_debiased"]["pair_accuracy"] == 0.5
    assert metrics["language_prior_debiased"]["set_accuracy"] == 0.0


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
