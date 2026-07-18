import json
from pathlib import Path

import pytest

from scripts.summarize_showo_cfg_sweep import SummaryError, build_summary
from scripts.validate_showo_cfg_metrics import (
    EXPECTED_CHECKPOINT,
    EXPECTED_CONFIG,
    EXPECTED_INCEPTION_SHA256,
    EXPECTED_MAGVIT,
    EXPECTED_MANIFEST_SHA256,
    EXPECTED_PROMPT,
    EXPECTED_PROTOCOL,
    EXPECTED_REAL_STATS,
    EXPECTED_SELECTION_SHA256,
    EXPECTED_SPLIT,
    EXPECTED_SPLIT_MANIFEST_SHA256,
    EXPECTED_TRANSFORM,
    REPO_ROOT,
    validate_metrics,
)


def formal_metrics(common_cfg: float, *, fid: float = 30.0, is_mean: float = 50.0):
    guidance = common_cfg - 1.0
    return {
        "protocol": EXPECTED_PROTOCOL,
        "official_protocol": True,
        "config": str((REPO_ROOT / EXPECTED_CONFIG).resolve()),
        "checkpoint": str((REPO_ROOT / EXPECTED_CHECKPOINT).resolve()),
        "samples": 10_000,
        "seed": 42,
        "sampling": {
            "method": "maskgit",
            "timesteps": 12,
            "guidance_scale": guidance,
            "common_cfg_scale": common_cfg,
            "guidance_formula": "(1+s)*conditional-s*unconditional",
            "temperature": 1.0,
            "temperature_schedule": (
                "official_showo_cumulative_one_minus_ratio"
            ),
            "mask_schedule": "cosine",
        },
        "tokenizer": {
            "type": "official_showo_magvitv2",
            "path": str(EXPECTED_MAGVIT),
            "image_vocab_size": 8192,
            "tokens_per_image": 256,
            "decode_dtype": "float32",
        },
        "real_stats": {
            "path": str(EXPECTED_REAL_STATS),
            "metadata": {
                "schema": "qwen_showo_imagenet100_real_stats_v1",
                "protocol": EXPECTED_PROTOCOL,
                "real_source": "original_imagenet",
                "manifest_sha256": EXPECTED_MANIFEST_SHA256,
                "split_manifest_sha256": EXPECTED_SPLIT_MANIFEST_SHA256,
                "selection_sha256": EXPECTED_SELECTION_SHA256,
                "num_samples": 10_000,
                "num_classes": 100,
                "class_counts": {
                    f"n{index:08d}": 100 for index in range(100)
                },
                "split": EXPECTED_SPLIT,
                "prompt": EXPECTED_PROMPT,
                "transform": EXPECTED_TRANSFORM,
                "feature": {
                    "backend": (
                        "torchmetrics.NoTrainInceptionV3/torch-fidelity"
                    ),
                    "feature": 2048,
                    "feature_name": "2048",
                    "logits_name": "logits_unbiased",
                    "weights_sha256": EXPECTED_INCEPTION_SHA256,
                },
            },
        },
        "metrics": {
            "fid": fid,
            "fid_feature": 2048,
            "inception_score_mean": is_mean,
            "inception_score_std": 1.0,
            "inception_score_splits": [is_mean] * 10,
        },
        "distributed": {"world_size": 8, "local_batch_size": 8},
        "saved_images": False,
    }


def write_metrics(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def validate_synthetic(path: Path, guidance: float):
    return validate_metrics(
        path,
        expected_guidance_scale=guidance,
        require_images=False,
        expected_checkpoint_sha256=None,
        expected_samples_sha256=None,
    )


def test_validator_accepts_both_cfg_conventions(tmp_path):
    path = write_metrics(tmp_path / "metrics.json", formal_metrics(2.5))

    _, errors, _ = validate_synthetic(path, guidance=1.5)

    assert errors == []


def test_validator_rejects_nonofficial_temperature_schedule(tmp_path):
    payload = formal_metrics(2.5)
    payload["sampling"]["temperature_schedule"] = "linear_from_base"
    path = write_metrics(tmp_path / "metrics.json", payload)

    _, errors, _ = validate_synthetic(path, guidance=1.5)

    assert "temperature_schedule" in errors


def test_validator_rejects_cfg_mapping_drift(tmp_path):
    payload = formal_metrics(2.5)
    payload["sampling"]["common_cfg_scale"] = 1.5
    path = write_metrics(tmp_path / "metrics.json", payload)

    _, errors, _ = validate_synthetic(path, guidance=1.5)

    assert "common_cfg_scale" in errors


def test_summary_selects_fid_and_is_winners(tmp_path):
    low = write_metrics(
        tmp_path / "low" / "metrics.json",
        formal_metrics(2.0, fid=25.0, is_mean=45.0),
    )
    high = write_metrics(
        tmp_path / "high" / "metrics.json",
        formal_metrics(3.0, fid=27.0, is_mean=55.0),
    )

    summary = build_summary(
        [f"2.0={low}", f"3.0={high}"],
        ["2.0=job-low", "3.0=job-high"],
        require_images=False,
        expected_checkpoint_sha256=None,
        expected_samples_sha256=None,
    )

    assert summary["best_by_fid"]["common_cfg_scale"] == 2.0
    assert summary["best_by_is"]["common_cfg_scale"] == 3.0
    assert summary["points"][0]["showo_guidance_scale"] == 1.0


def test_summary_rejects_cfg_below_conditional_only(tmp_path):
    path = write_metrics(tmp_path / "metrics.json", formal_metrics(1.0))

    with pytest.raises(SummaryError, match=">=1.0"):
        build_summary(
            [f"0.5={path}"],
            [],
            require_images=False,
            expected_checkpoint_sha256=None,
            expected_samples_sha256=None,
        )
