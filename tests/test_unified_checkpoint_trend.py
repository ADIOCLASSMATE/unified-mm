import json
from pathlib import Path

import pytest

from scripts.organize_evaluation_outputs import metric as archive_metric
from scripts.summarize_unified_checkpoint_trend import read_summary, trend_row


def _summary(step: int) -> dict:
    return {
        "complete": True,
        "profile": "formal",
        "global_step": step,
        "checkpoint": f"/checkpoint-{step}",
        "dataset_contract": {
            "training_split": "imagenet_train",
            "evaluation_split": "imagenet_val",
        },
        "generation": {
            "imagenet_val_t2i": {
                "official_protocol": True,
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
            "primary_metric": "normalized_pairwise.win_rate",
            "normalized_pairwise": {"win_rate": 0.625},
        }
    }

    assert archive_metric(summary, "sugarcrepe") == 0.625
