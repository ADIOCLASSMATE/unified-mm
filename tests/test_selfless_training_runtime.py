from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from utils.selfless_training_runtime import (
    LEGACY_RESUME_SCHEMA,
    RESUME_SCHEMA,
    RESUME_SIGNATURE_VERSION,
    TrainingWindow,
    build_legacy_resume_signature,
    build_resume_signature,
    validate_resume_metadata,
    validate_wsd_contract,
)


def _config(*, max_train_steps: int = 100):
    return OmegaConf.create(
        {
            "model": {"name": "tiny", "width": 16},
            "dataset": {"params": {"path": "/data", "row_length": 2048}},
            "optimizer": {
                "name": "adamw",
                "params": {"learning_rate": 1.0e-4},
            },
            "lr_scheduler": {
                "scheduler": "wsd",
                "params": {
                    "warmup_steps": 10,
                    "decay_steps": 20,
                    "min_lr_scale": 0.1,
                },
            },
            "training": {
                "batch_size": 2,
                "total_batch_size": 16,
                "mixed_precision": "bf16",
                "gradient_accumulation_dtype": "fp32",
                "seed": 7,
                "max_train_steps": max_train_steps,
                "max_grad_norm": 1.0,
                "trainable_scope": "full",
            },
            "experiment": {"output_dir": "/tmp/ignored"},
        }
    )


def _signature(config):
    return build_resume_signature(
        config,
        world_size=4,
        gradient_accumulation_steps=2,
    )


def test_resume_signature_covers_future_training_controls():
    baseline = _config()
    changed_steps = _config(max_train_steps=101)
    changed_grad_clip = _config()
    changed_grad_clip.training.max_grad_norm = 0.5

    assert _signature(baseline) != _signature(changed_steps)
    assert _signature(baseline) != _signature(changed_grad_clip)

    changed_output = _config()
    changed_output.experiment.output_dir = "/another/output"
    assert _signature(baseline) == _signature(changed_output)


def test_v3_resume_metadata_requires_exact_signature(tmp_path: Path):
    config = _config()
    metadata = {
        "schema": RESUME_SCHEMA,
        "config_signature_version": RESUME_SIGNATURE_VERSION,
        "config_signature": _signature(config),
    }
    validate_resume_metadata(
        metadata,
        checkpoint_dir=tmp_path / "checkpoint-5",
        config=config,
        world_size=4,
        gradient_accumulation_steps=2,
        current_signature=_signature(config),
    )

    metadata["config_signature"] = "wrong"
    with pytest.raises(RuntimeError, match="inexact continuation"):
        validate_resume_metadata(
            metadata,
            checkpoint_dir=tmp_path / "checkpoint-5",
            config=config,
            world_size=4,
            gradient_accumulation_steps=2,
            current_signature=_signature(config),
        )


def test_v2_resume_uses_immutable_config_to_cover_omitted_fields(tmp_path: Path):
    saved_config = _config(max_train_steps=100)
    OmegaConf.save(saved_config, tmp_path / "config.yaml")
    checkpoint_dir = tmp_path / "checkpoint-5"
    checkpoint_dir.mkdir()
    metadata = {
        "schema": LEGACY_RESUME_SCHEMA,
        "config_signature": build_legacy_resume_signature(
            saved_config,
            world_size=4,
            gradient_accumulation_steps=2,
        ),
    }

    validate_resume_metadata(
        metadata,
        checkpoint_dir=checkpoint_dir,
        config=saved_config,
        world_size=4,
        gradient_accumulation_steps=2,
        current_signature=_signature(saved_config),
    )

    changed = _config(max_train_steps=101)
    # max_train_steps was omitted from the legacy signature, so only the new
    # immutable-config comparison catches this inexact continuation.
    with pytest.raises(RuntimeError, match="immutable config differs"):
        validate_resume_metadata(
            metadata,
            checkpoint_dir=checkpoint_dir,
            config=changed,
            world_size=4,
            gradient_accumulation_steps=2,
            current_signature=_signature(changed),
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("max_train_steps", 0, "must be positive"),
        ("warmup_steps", -1, "must be non-negative"),
        ("decay_steps", 95, "must not exceed"),
        ("min_lr_scale", 1.1, "must be in"),
    ],
)
def test_wsd_contract_rejects_invalid_schedules(field, value, message):
    config = _config()
    if field == "max_train_steps":
        config.training[field] = value
    else:
        config.lr_scheduler.params[field] = value
    with pytest.raises(ValueError, match=message):
        validate_wsd_contract(config)


def test_training_window_accounts_without_per_batch_device_work():
    window = TrainingWindow(started_at=10.0)
    window.record_batch(
        rows=3,
        sequence_length=2048,
        logical_images=7,
        pack_stats=(5000, 1792, 1144, 2048),
        data_wait_seconds=0.125,
    )
    window.record_optimizer_step()
    window.exclude_elapsed(2.5)

    values = window.as_tensor(torch.device("cpu"))
    assert values[:9].tolist() == [
        1.0,
        1.0,
        7.0,
        3.0,
        6144.0,
        5000.0,
        1792.0,
        1144.0,
        0.125,
    ]
    assert window.started_at == 12.5
