import json

import pytest

from scripts.ensure_flow_cfg_sweep_contract import (
    FIXED_ARTIFACTS,
    REQUIRED_CHECKPOINT_SIDECARS,
    checkpoint_sidecar_sha256,
)
from scripts.summarize_flow_cfg_sweep import SummaryError, build_summary
from scripts.validate_flow_cfg_metrics import (
    EXPECTED_CONFIG,
    EXPECTED_FEATURE_PROTOCOL,
    EXPECTED_INCEPTION_WEIGHTS_PATH,
    EXPECTED_MANIFEST_SHA256,
    EXPECTED_MODEL_PATH,
    EXPECTED_PROMPT_PROTOCOL,
    EXPECTED_REAL_STATS_PATH,
    EXPECTED_SELECTION_SHA256,
    EXPECTED_SPLIT_MANIFEST_SHA256,
    EXPECTED_SPLIT_PROTOCOL,
    EXPECTED_TRANSFORM_PROTOCOL,
    validate_metrics,
)


NON_EMA_MODEL_PATH = (
    "output/selfless-flow-ablation-imagenet100-80ep/hf_model-final"
)


def _formal_metrics(
    cfg,
    *,
    model_path=EXPECTED_MODEL_PATH,
    model_dtype="bf16",
    fid=20.0,
    is_mean=50.0,
):
    parameter_dtype = {
        "bf16": "torch.bfloat16",
        "fp32": "torch.float32",
    }[model_dtype]
    checkpoint_weight_dtype = (
        "bf16" if model_path == NON_EMA_MODEL_PATH else "fp32"
    )
    return {
        "official_protocol": True,
        "config": EXPECTED_CONFIG,
        "model_path": model_path,
        "precision_protocol": {
            "schema": "flow_eval_precision_v1",
            "model_dtype": model_dtype,
            "model_parameter_dtypes": [parameter_dtype],
            "checkpoint_weight_dtypes": [checkpoint_weight_dtype],
            "vae_dtype": "fp32",
            "flow_integrator_dtype": "fp32",
            "autocast_enabled": False,
            "cuda_matmul_allow_tf32": False,
            "cudnn_allow_tf32": True,
            "float32_matmul_precision": "highest",
        },
        "adapter": {"adapter": None},
        "model_state": {"model_state": None},
        "ema_state": {"ema_state": None},
        "split": "val",
        "seed": 42,
        "batch_size": 512,
        "samples_requested": 10_000,
        "samples_evaluated": 10_000,
        "distributed": {
            "enabled": True,
            "world_size": 8,
            "rank": 0,
            "local_rank": 0,
            "peak_cuda_allocated_mib": 12_000.0,
            "peak_cuda_reserved_mib": 20_000.0,
        },
        "cfg": cfg,
        "cfg_schedule": "constant",
        "sampling_steps": "100",
        "temperature": 1.0,
        "flow_solver": "heun",
        "parallel_rate": 1,
        "metric_protocol": {
            "fid_reducer": "symmetric_eigendecomposition",
            "is_split_assignment": "contiguous_by_global_sample_index",
            "is_std": "population",
            "is_splits": 10,
        },
        "real_source": "cached_original_imagenet",
        "real_stats_path": str(EXPECTED_REAL_STATS_PATH),
        "real_stats_metadata": {
            "schema": "qwen_showo_imagenet100_real_stats_v1",
            "protocol": "imagenet100-balanced-val100-per-class-class-name-v1",
            "real_source": "original_imagenet",
            "manifest_sha256": EXPECTED_MANIFEST_SHA256,
            "split_manifest_sha256": EXPECTED_SPLIT_MANIFEST_SHA256,
            "num_samples": 10_000,
            "num_classes": 100,
            "class_counts": {f"n{index:08d}": 100 for index in range(100)},
            "selection_sha256": EXPECTED_SELECTION_SHA256,
            "split": EXPECTED_SPLIT_PROTOCOL,
            "prompt": EXPECTED_PROMPT_PROTOCOL,
            "transform": EXPECTED_TRANSFORM_PROTOCOL,
            "feature": EXPECTED_FEATURE_PROTOCOL,
        },
        "imagenet_train_dir": "/inspire/dataset/imagenet/v1/train",
        "real_image_size": 256,
        "inception_weights_path": str(EXPECTED_INCEPTION_WEIGHTS_PATH),
        "strategies": {
            "spatial_halton": {
                "count": 10_000,
                "fid": fid,
                "inception_score_mean": is_mean,
                "inception_score_std": 1.0,
                "inception_score_splits": [is_mean] * 10,
                "latent_mse_to_target": 2.0,
                "latent_rms": 1.1,
                "generation_step_max": 256.0,
            }
        },
    }


def _write_metrics(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_validator_accepts_explicit_non_ema_model_path(tmp_path):
    path = _write_metrics(
        tmp_path / "metrics.json",
        _formal_metrics(3.5, model_path=NON_EMA_MODEL_PATH),
    )

    _, default_errors, _ = validate_metrics(path, 3.5, False)
    assert "model_path" in default_errors

    _, explicit_errors, _ = validate_metrics(
        path,
        3.5,
        False,
        expected_model_path=NON_EMA_MODEL_PATH,
    )
    assert explicit_errors == []


def test_validator_and_summarizer_keep_model_dtype_separate(tmp_path):
    bf16_path = _write_metrics(
        tmp_path / "bf16.json",
        _formal_metrics(3.5, model_dtype="bf16"),
    )
    fp32_path = _write_metrics(
        tmp_path / "fp32.json",
        _formal_metrics(4.0, model_dtype="fp32"),
    )

    _, wrong_errors, _ = validate_metrics(fp32_path, 4.0, False)
    assert {"model_dtype", "model_parameter_dtypes"} <= set(wrong_errors)

    _, explicit_errors, _ = validate_metrics(
        fp32_path,
        4.0,
        False,
        expected_model_dtype="fp32",
    )
    assert explicit_errors == []

    with pytest.raises(SummaryError, match="model_dtype"):
        build_summary(
            [f"3.5={bf16_path}", f"4.0={fp32_path}"],
            [],
            None,
        )


def test_legacy_metrics_without_dtype_are_treated_as_bf16(tmp_path):
    payload = _formal_metrics(3.5)
    payload.pop("precision_protocol")
    path = _write_metrics(tmp_path / "legacy.json", payload)

    _, errors, _ = validate_metrics(path, 3.5, False)
    assert errors == []
    summary = build_summary([f"3.5={path}"], [], None)
    assert summary["protocol"]["model_dtype"] == "bf16"


def test_summarizer_selects_fid_and_is_winners(tmp_path):
    cfg35 = _write_metrics(
        tmp_path / "cfg35.json",
        _formal_metrics(3.5, fid=19.0, is_mean=50.0),
    )
    cfg40 = _write_metrics(
        tmp_path / "cfg40.json",
        _formal_metrics(4.0, fid=20.0, is_mean=55.0),
    )

    summary = build_summary(
        [f"3.5={cfg35}", f"4.0={cfg40}"],
        ["3.5=job-a", "4.0=job-b"],
        "a" * 64,
    )

    assert summary["best_by_fid"]["cfg"] == 3.5
    assert summary["best_by_is"]["cfg"] == 4.0
    assert summary["checkpoint_sha256"] == "a" * 64


def test_summarizer_rejects_nonformal_generation_contract(tmp_path):
    payload = _formal_metrics(3.5)
    payload["sampling_steps"] = "99"
    path = _write_metrics(tmp_path / "metrics.json", payload)

    with pytest.raises(SummaryError, match="sampling_steps"):
        build_summary([f"3.5={path}"], [], None)


def test_validator_rejects_metric_and_state_corruption(tmp_path):
    payload = _formal_metrics(3.5)
    payload["metric_protocol"]["fid_reducer"] = "legacy_sqrtm"
    payload["adapter"] = {"adapter": "unexpected.pt"}
    payload["strategies"]["spatial_halton"]["fid"] = -1.0
    payload["strategies"]["spatial_halton"]["inception_score_mean"] = True
    path = _write_metrics(tmp_path / "metrics.json", payload)

    _, errors, _ = validate_metrics(path, 3.5, False)

    assert {"metric_protocol", "adapter", "fid", "is_mean"} <= set(errors)


def test_summarizer_rejects_real_stats_contract_drift(tmp_path):
    payload = _formal_metrics(3.5)
    payload["real_stats_metadata"]["manifest_sha256"] = "0" * 64
    path = _write_metrics(tmp_path / "metrics.json", payload)

    with pytest.raises(SummaryError, match="manifest_sha256"):
        build_summary([f"3.5={path}"], [], None)


def test_sweep_contract_binds_checkpoint_sidecars_and_validation_inputs(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    for filename in REQUIRED_CHECKPOINT_SIDECARS:
        (checkpoint / filename).write_text(f"{filename}\n", encoding="utf-8")
    (checkpoint / "generation_config.json").write_text("{}\n", encoding="utf-8")
    (checkpoint / "model.safetensors").write_bytes(b"weights-bound-separately")

    before = checkpoint_sidecar_sha256(checkpoint)
    assert set(before) == {
        *REQUIRED_CHECKPOINT_SIDECARS,
        "generation_config.json",
    }
    assert "model.safetensors" not in before

    (checkpoint / "tokenizer.json").write_text("changed\n", encoding="utf-8")
    after = checkpoint_sidecar_sha256(checkpoint)
    assert after["tokenizer.json"] != before["tokenizer.json"]
    assert {
        "flow_latent_cache",
        "manifest",
        "split_manifest",
        "synset_mapping",
    } <= set(FIXED_ARTIFACTS)
