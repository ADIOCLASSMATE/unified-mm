import json
from pathlib import Path

import pytest

from scripts.organize_evaluation_outputs import (
    metric as archive_metric,
    selected_generation,
)
from scripts.summarize_unified_checkpoint_trend import read_summary, trend_row


def _summary(step: int) -> dict:
    return {
        "schema": "unified_full_checkpoint_evaluation_summary_v4",
        "complete": True,
        "profile": "formal",
        "runtime_hashing_enabled": False,
        "global_step": step,
        "checkpoint": f"/checkpoint-{step}",
        "dataset_contract": {
            "training_split": "imagenet_train",
            "evaluation_split": "imagenet_val",
        },
        "generation": {
            "imagenet_val_t2i": {
                "project_formal_protocol": True,
                "leaderboard_comparable_to_adm_dit": False,
                "protocol_name": (
                    "imagenet_val_fid50k_torch_fidelity_stratified_is"
                ),
                "reference_distribution": "imagenet_val_50000",
                "comparison_scope": "same_protocol_only",
                "strategy": "spatial_halton",
                "samples": 50_000,
                "is_split_assignment": "stratified_by_synset",
                "fid": 5.0,
                "inception_score_mean": 200.0,
                "inception_score_std": 3.0,
            }
        },
        "understanding": {
            "heldout_validation": {
                "val/loss": 1.0,
                "val/loss_i2t": 2.0,
                "val/loss_t2i": 0.5,
                "val/ppl_text": 10.0,
            },
        },
        "pure_text": {
            "macro_average_primary": 0.46,
            "primary_metrics": {"task": 0.46},
        },
    }


def test_trend_row_extracts_comparable_metrics(tmp_path: Path):
    root = tmp_path / "eval"
    root.mkdir()
    payload = _summary(42_000)
    (root / "full_evaluation_summary.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )

    row = trend_row(root, read_summary(root))

    assert row["global_step"] == 42_000
    assert row["fid"] == 5.0
    assert row["inception_score_mean"] == 200.0
    assert row["pure_text_primary_metrics"] == {"task": 0.46}


def test_trend_rejects_non_val_evaluation(tmp_path: Path):
    root = tmp_path / "eval"
    root.mkdir()
    payload = _summary(42_000)
    payload["dataset_contract"]["evaluation_split"] = "imagenet_train"
    (root / "full_evaluation_summary.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="not ImageNet val"):
        read_summary(root)


def test_archive_metric_resolves_nested_primary_metric():
    summary = {
        "metrics": {
            "primary_metric": "language_prior_debiased_pairwise.win_rate",
            "language_prior_debiased_pairwise": {"win_rate": 0.625},
        }
    }

    assert archive_metric(summary, "sugarcrepe") == 0.625


def _generation_metrics() -> dict:
    split_plan = {
        "assignment": "stratified_by_synset",
        "source_dataset_split": "val",
        "samples": 50_000,
        "splits": 10,
        "class_count": 1_000,
        "samples_per_class_min": 50,
        "samples_per_class_max": 50,
        "samples_per_class_per_split_min": 5,
        "samples_per_class_per_split_max": 5,
        "samples_per_split": [5_000] * 10,
        "classes_per_split": [1_000] * 10,
    }
    return {
        "schema": "selfless_imagenet_val_t2i_fid_is_v2",
        "runtime_hashing_enabled": False,
        "project_formal_protocol": True,
        "leaderboard_comparable_to_adm_dit": False,
        "split": "val",
        "real_source": "cached_original_imagenet_val",
        "samples_requested": 50_000,
        "samples_evaluated": 50_000,
        "metric_protocol": {
            "protocol_name": (
                "imagenet_val_fid50k_torch_fidelity_stratified_is"
            ),
            "reference_distribution": "imagenet_val_50000",
            "comparison_scope": "same_protocol_only",
            "not_adm_dit_reason": (
                "validation_reference_and_pytorch_torch_fidelity_extractor"
            ),
            "fid_reducer": "symmetric_eigendecomposition",
            "fid_computed": True,
            "is_split_assignment": "stratified_by_synset",
            "is_split_plan": split_plan,
            "is_std": "population",
            "is_splits": 10,
        },
        "strategies": {
            "spatial_halton": {
                "count": 50_000,
                "fid": 5.0,
                "inception_score_mean": 2.0,
                "inception_score_std": 0.0,
                "inception_score_splits": [2.0] * 10,
            }
        },
    }


def test_archive_generation_accepts_only_explicit_same_protocol_result():
    selected = selected_generation(_generation_metrics())

    assert selected["protocol"] == (
        "imagenet_val_fid50k_torch_fidelity_stratified_is"
    )
    assert selected["leaderboard_comparable_to_adm_dit"] is False


def test_archive_generation_rejects_adm_dit_comparability_claim():
    payload = _generation_metrics()
    payload["leaderboard_comparable_to_adm_dit"] = True

    with pytest.raises(ValueError, match="comparability"):
        selected_generation(payload)


def test_archive_generation_rejects_inconsistent_is_summary():
    payload = _generation_metrics()
    payload["strategies"]["spatial_halton"]["inception_score_std"] = 1.0

    with pytest.raises(ValueError, match="std is inconsistent"):
        selected_generation(payload)
