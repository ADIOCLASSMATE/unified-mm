import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from scripts.select_unified_lr_sweep import select


MANIFEST = Path(
    "configs/protocols/unified_baseline_lr_sweep_1b_ascend64.yaml"
)


def test_frozen_sweep_grid_and_no_hash_contract():
    sweep = OmegaConf.load(MANIFEST)
    base = OmegaConf.load(sweep.base_config)

    assert sweep.schema == "unified_lr_sweep_v1"
    assert sweep.world_size == 64
    assert sweep.nodes * sweep.npu_per_node == sweep.world_size
    assert sweep.stop_after_steps == 955
    assert sweep.nominal_text_targets_at_stop == (
        sweep.stop_after_steps * sweep.nominal_text_targets_per_step
    )
    assert sweep.selection.runtime_hashing_enabled is False
    assert sweep.selection.require_all_arms is True
    assert base.training.runtime_hashing_enabled is False
    assert base.training.max_train_steps == 95415
    assert base.lr_scheduler.params.warmup_steps == 596
    assert base.lr_scheduler.params.decay_steps == 23854

    arms = list(sweep.arms)
    assert len(arms) == 9
    assert {
        (float(arm.backbone_lr), float(arm.flow_lr)) for arm in arms
    } == {
        (backbone_lr, flow_lr)
        for backbone_lr in (2.0e-4, 2.4e-4, 3.0e-4)
        for flow_lr in (3.0e-5, 4.0e-5, 5.0e-5)
    }
    assert {int(arm.wave) for arm in arms} == {1, 2, 3}
    assert all(sum(int(arm.wave) == wave for arm in arms) == 3 for wave in (1, 2, 3))

    launcher = Path(
        "script/selfless/pretraining_unified_baseline_lr_sweep_arm_ascend64.sh"
    ).read_text(encoding="utf-8")
    assert "STOP_AFTER_STEPS=\"955\"" in launcher
    assert "SAVE_EVERY=\"955\"" in launcher
    assert "VAL_EVERY=\"955\"" in launcher
    assert "SAVE_EMA_EVAL_EVERY=\"0\"" in launcher
    assert "RESUME_FROM=\"none\"" in launcher
    assert 'WANDB_MODE="disabled"' in launcher
    assert "sha256" not in launcher.lower()
    assert "hashlib" not in launcher.lower()


def _write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _fake_arm_artifacts(
    root: Path,
    arm,
    *,
    text_loss: float,
    image_loss: float,
    world_size: int,
):
    run_root = root / arm["id"]
    checkpoint = run_root / "checkpoint-955"
    _write_json(
        checkpoint / "checkpoint_complete.json",
        {
            "schema": "selfless_caption_checkpoint_complete_v1",
            "global_step": 955,
        },
    )
    _write_json(
        checkpoint / "metadata.json",
        {
            "schema": "selfless_caption_training_checkpoint_v3",
            "global_step": 955,
            "world_size": world_size,
            "gradient_accumulation_steps": 4,
            "config_contract_version": 1,
            "mixed_data_state_schema": "unified_mixed_data_state_v1",
            "config_contract": {
                "experiment": {
                    "project": f"unified-sweep/{arm['id']}",
                    "name": f"unified-sweep-{arm['id']}",
                    "output_dir": str(run_root),
                },
                "model": {
                    "training_objective": "selfless_dual_stream",
                    "dual_stream_attention_contract": "selfless_strict",
                },
                "training": {"runtime_hashing_enabled": False},
                "optimizer": {
                    "params": {
                        "learning_rate": arm["backbone_lr"],
                        "backbone_learning_rate": arm["backbone_lr"],
                        "special_token_learning_rate": arm["backbone_lr"],
                        "flow_learning_rate": arm["flow_lr"],
                        "projector_learning_rate": arm["flow_lr"],
                    }
                },
                "lr_scheduler": {
                    "params": {
                        "learning_rate": arm["backbone_lr"],
                        "warmup_steps": 596,
                        "decay_steps": 23854,
                        "min_lr_scale": 0.1,
                    }
                },
            },
        },
    )
    for rank in range(world_size):
        (checkpoint / f"data_state_rank_{rank:05d}.pt").touch()
        (checkpoint / f"random_states_{rank}.pkl").touch()
        (checkpoint / f"ema_shard_rank_{rank:05d}.safetensors").touch()
    _write_json(
        run_root / "validation_metrics_step_955.json",
        {
            "schema": "selfless_flow_validation_metrics_v1",
            "global_step": 955,
            "training_seed": 42,
            "validation_seed": 424242,
            "metrics": {
                "val/loss_text": text_loss,
                "val/loss_image_flow": image_loss,
                "val/text_target_tokens": 100.0,
                "val/image_target_tokens": 200.0,
            },
        },
    )
    _write_json(
        run_root / "training_runtime_metrics.json",
        {
            "schema": "selfless_training_runtime_metrics_v1",
            "global_step": 955,
            "world_size": world_size,
            "finite_loss_microbatches_checked": 3820,
            "cumulative_training_wall_seconds": 100.0,
        },
    )


def test_selector_requires_complete_readable_artifacts_and_balances_modalities(tmp_path):
    output_root = tmp_path / "runs"
    arms = [
        {
            "id": f"arm-{index}",
            "job_name": f"job-{index}",
            "backbone_lr": 2.0e-4 + index * 1.0e-6,
            "flow_lr": 3.0e-5 + index * 1.0e-7,
        }
        for index in range(9)
    ]
    losses = [(0.9, 3.0), (1.0, 1.0)] + [
        (float(index), float(index)) for index in range(2, 9)
    ]
    for arm, (text_loss, image_loss) in zip(arms, losses):
        _fake_arm_artifacts(
            output_root,
            arm,
            text_loss=text_loss,
            image_loss=image_loss,
            world_size=2,
        )

    manifest = {
        "schema": "unified_lr_sweep_v1",
        "output_root": str(output_root),
        "world_size": 2,
        "seed": 42,
        "validation_seed": 424242,
        "stop_after_steps": 955,
        "selection": {
            "required_arms": 9,
            "runtime_hashing_enabled": False,
        },
        "formal_continuation": {"max_train_steps": 95415},
        "arms": arms,
    }
    manifest_path = tmp_path / "sweep.yaml"
    OmegaConf.save(OmegaConf.create(manifest), manifest_path)

    report = select(manifest_path)
    assert report["all_arms_complete"] is True
    assert report["runtime_hashing_enabled"] is False
    assert report["winner"]["arm_id"] == "arm-1"
    assert report["formal_continuation"]["restart_from_step_zero"] is False
    assert report["formal_continuation"]["resume_from_checkpoint"].endswith(
        "arm-1/checkpoint-955"
    )

    (output_root / "arm-8" / "checkpoint-955" / "random_states_1.pkl").unlink()
    with pytest.raises(ValueError, match="incomplete RNG state"):
        select(manifest_path)


def test_selector_rejects_non_lr_contract_drift(tmp_path):
    output_root = tmp_path / "runs"
    arms = [
        {
            "id": f"arm-{index}",
            "job_name": f"job-{index}",
            "backbone_lr": 2.0e-4,
            "flow_lr": 3.0e-5,
        }
        for index in range(9)
    ]
    for arm in arms:
        _fake_arm_artifacts(
            output_root,
            arm,
            text_loss=1.0,
            image_loss=1.0,
            world_size=2,
        )
    metadata_path = (
        output_root / "arm-8" / "checkpoint-955" / "metadata.json"
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["config_contract"]["model"][
        "dual_stream_attention_contract"
    ] = "xlnet_content_diagonal"
    _write_json(metadata_path, metadata)

    manifest = {
        "schema": "unified_lr_sweep_v1",
        "output_root": str(output_root),
        "world_size": 2,
        "seed": 42,
        "validation_seed": 424242,
        "stop_after_steps": 955,
        "selection": {
            "required_arms": 9,
            "runtime_hashing_enabled": False,
        },
        "formal_continuation": {"max_train_steps": 95415},
        "arms": arms,
    }
    manifest_path = tmp_path / "sweep.yaml"
    OmegaConf.save(OmegaConf.create(manifest), manifest_path)
    with pytest.raises(ValueError, match="not ablation a"):
        select(manifest_path)


def test_selector_source_does_not_import_hash_implementation():
    source = Path("scripts/select_unified_lr_sweep.py").read_text(encoding="utf-8")
    assert "import hashlib" not in source
    assert "sha256" not in source.lower()
