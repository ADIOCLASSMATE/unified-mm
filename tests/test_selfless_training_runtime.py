import pytest
import torch
from omegaconf import OmegaConf

from utils.selfless_training_runtime import (
    RESUME_CONTRACT_VERSION,
    RESUME_SCHEMA,
    TrainingWindow,
    build_resume_contract,
    build_sampler_resume_state,
    gradient_norm_log_payload,
    training_stop_step,
    validate_sampler_resume_state,
    validate_resume_contract,
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


def _contract(config):
    return build_resume_contract(
        config,
        world_size=4,
        gradient_accumulation_steps=2,
    )


def test_resume_contract_covers_future_training_controls():
    baseline = _config()
    changed_steps = _config(max_train_steps=101)
    changed_grad_clip = _config()
    changed_grad_clip.training.max_grad_norm = 0.5

    assert _contract(baseline) != _contract(changed_steps)
    assert _contract(baseline) != _contract(changed_grad_clip)

    changed_output = _config()
    changed_output.experiment.output_dir = "/another/output"
    assert _contract(baseline) == _contract(changed_output)

    changed_output_policy = _config()
    changed_output_policy.experiment.save_ema_eval_every = 10
    changed_output_policy.experiment.save_model_with_ema_eval = True
    changed_output_policy.experiment.save_image_flow_adapter = False
    changed_output_policy.experiment.save_final_image_flow_adapter = True
    assert _contract(baseline) == _contract(changed_output_policy)


def test_stage_stop_boundary_does_not_change_resume_contract():
    baseline = _config()
    staged = _config()
    staged.training.stop_after_steps = 20
    assert _contract(staged) == _contract(baseline)
    assert training_stop_step(staged) == 20

    staged.training.stop_after_steps = 101
    with pytest.raises(ValueError, match="stop_after_steps"):
        training_stop_step(staged)


def test_grad_norm_event_is_independent_of_loss_log_cadence():
    # 1202 is intentionally not divisible by the production loss log cadence
    # of 50.  The event must still be emitted at its own exact step.
    payload = gradient_norm_log_payload(
        global_step=1202,
        every=1202,
        pre_clip_norm=torch.tensor(1.5),
        max_norm=1.0,
    )
    assert payload == {
        "train/global_grad_norm_pre_clip": 1.5,
        "train/grad_clip_max_norm": 1.0,
        "train/grad_clip_applied": 1.0,
    }
    assert gradient_norm_log_payload(
        global_step=1200,
        every=1202,
        pre_clip_norm=1.5,
        max_norm=1.0,
    ) is None


def test_sampler_resume_state_round_trip_is_strict():
    state = build_sampler_resume_state(
        epoch=2,
        batches_consumed_in_epoch=17,
        shuffle_seed=42,
        prepared_dataloader_length=4808,
    )
    validate_sampler_resume_state(
        state,
        epoch=2,
        batches_consumed_in_epoch=17,
        shuffle_seed=42,
        prepared_dataloader_length=4808,
    )
    with pytest.raises(RuntimeError, match="inexact continuation"):
        validate_sampler_resume_state(
            state,
            epoch=2,
            batches_consumed_in_epoch=18,
            shuffle_seed=42,
            prepared_dataloader_length=4808,
        )


def test_v3_resume_metadata_requires_exact_readable_contract():
    config = _config()
    metadata = {
        "schema": RESUME_SCHEMA,
        "config_contract_version": RESUME_CONTRACT_VERSION,
        "config_contract": _contract(config),
    }
    validate_resume_contract(
        metadata,
        current_contract=_contract(config),
    )

    metadata["config_contract"] = _contract(_config(max_train_steps=101))
    with pytest.raises(RuntimeError, match="inexact continuation"):
        validate_resume_contract(
            metadata,
            current_contract=_contract(config),
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
