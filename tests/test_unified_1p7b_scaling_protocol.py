import math
from pathlib import Path

from omegaconf import OmegaConf


CONFIG_PATH = Path(
    "configs/selfless/unified_baseline_1p7b_100b_ascend_64npu.yaml"
)
MANIFEST_PATH = Path(
    "configs/protocols/unified_baseline_1p7b_lr_sweep_1b_ascend64.yaml"
)
BASE_LAUNCHER = Path(
    "script/selfless/"
    "pretraining_unified_baseline_1p7b_ascend_64npu_100b.sh"
)
ARM_LAUNCHER = Path(
    "script/selfless/"
    "pretraining_unified_baseline_1p7b_lr_sweep_arm_ascend64.sh"
)
SUITE_LAUNCHER = Path(
    "script/selfless/"
    "pretraining_unified_baseline_1p7b_lr_sweep_ascend64.sh"
)
SMOKE_LAUNCHER = Path(
    "script/selfless/smoke_unified_baseline_1p7b_ascend16.sh"
)


def test_1p7b_preserves_global_data_and_token_budget_on_64_npus():
    config = OmegaConf.load(CONFIG_PATH)
    schedule = list(config.dataset.params.schedule)
    sources = config.dataset.params.sources

    assert config.model.model_path.endswith("Qwen--Qwen3-1.7B-Base")
    assert config.model.dual_stream_attention_contract == (
        "xlnet_content_diagonal"
    )
    assert str(config.experiment.project).startswith("unified-b-")
    assert sources.climbmix.tokenizer_path == config.model.model_path
    assert schedule == ["climbmix", "t2i", "climbmix", "i2t"]
    assert int(sources.climbmix.sequence_length) == 2048
    assert int(sources.climbmix.micro_batch_size) == 4
    assert int(sources.t2i.micro_batch_size) == 16
    assert int(sources.i2t.micro_batch_size) == 16

    rows = sum(int(sources[name].micro_batch_size) for name in schedule) * 64
    targets = schedule.count("climbmix") * 4 * 64 * (2048 - 1)
    assert rows == int(config.training.total_batch_size) == 2560
    assert targets == int(
        config.training.nominal_text_targets_per_step_64npu
    ) == 1_048_064
    assert math.ceil(100_000_000_000 / targets) == 95_415
    assert int(config.training.max_train_steps) == 95_415
    assert int(config.training.stop_after_steps) == 955
    assert int(config.training.gradient_accumulation_steps) == len(schedule)
    assert config.training.runtime_hashing_enabled is False
    assert config.training.from_scratch is False
    assert config.model.image_flow_grad_checkpointing is True
    assert config.training.use_gradient_checkpointing is False
    assert int(config.experiment.checkpoints_total_limit) == 3
    assert int(config.experiment.checkpoint_milestone_every) == 0
    assert int(config.experiment.save_ema_eval_every) == 12_510
    assert bool(config.experiment.save_model_with_ema_eval)
    assert str(config.experiment.ema_eval_dtype) == "bf16"
    assert float(config.optimizer.params.learning_rate) == 2.4e-4
    assert float(config.optimizer.params.backbone_learning_rate) == 2.4e-4
    assert float(config.optimizer.params.special_token_learning_rate) == 2.4e-4
    assert float(config.optimizer.params.projector_learning_rate) == 6.0e-5
    assert float(config.optimizer.params.flow_learning_rate) == 6.0e-5
    assert float(config.lr_scheduler.params.learning_rate) == 2.4e-4


def test_1p7b_lr_grid_is_width_scaled_and_complete():
    manifest = OmegaConf.load(MANIFEST_PATH)
    config = OmegaConf.load(manifest.base_config)

    assert manifest.schema == "unified_lr_sweep_v1"
    assert str(manifest.sweep_project).startswith("unified-b-")
    assert str(manifest.formal_output_root).startswith("output/unified-b-")
    assert int(manifest.nodes) == 4
    assert int(manifest.npu_per_node) == 16
    assert int(manifest.world_size) == 64
    assert int(manifest.stop_after_steps) == 955
    assert int(manifest.nominal_text_targets_at_stop) == (
        int(manifest.stop_after_steps)
        * int(manifest.nominal_text_targets_per_step)
    )
    assert manifest.selection.require_all_arms is True
    assert manifest.selection.runtime_hashing_enabled is False
    assert float(manifest.task_loss_logging.model_weights.text) == 0.05
    assert float(manifest.task_loss_logging.model_weights.image) == 1.0
    assert manifest.task_loss_logging.contribution_sum_matches == "step_loss"
    assert manifest.global_batch_policy.name == (
        "preserve_0p6b_global_physical_rows_and_text_targets"
    )
    assert manifest.memory_policy.allocator_env == "PYTORCH_NPU_ALLOC_CONF"
    assert manifest.memory_policy.allocator_config == (
        "expandable_segments:True"
    )
    assert manifest.memory_policy.preserve_micro_batches is True
    assert (
        manifest.memory_policy.preserve_gradient_accumulation_steps is True
    )
    expected_center = 3.0e-4 / math.sqrt(2048 / 1024)
    assert math.isclose(
        float(manifest.lr_scaling.width_scaled_backbone_lr),
        expected_center,
    )

    arms = list(manifest.arms)
    assert len(arms) == 9
    assert {
        (float(arm.backbone_lr), float(arm.flow_lr)) for arm in arms
    } == {
        (backbone_lr, flow_lr)
        for backbone_lr in (1.8e-4, 2.1e-4, 2.4e-4)
        for flow_lr in (4.0e-5, 5.0e-5, 6.0e-5)
    }
    assert sorted(int(arm.order) for arm in arms) == list(range(1, 10))
    assert int(config.lr_scheduler.params.warmup_steps) == 596
    assert int(config.lr_scheduler.params.decay_steps) == 23_854


def test_1p7b_launchers_freeze_64_card_no_hash_sweep_only():
    base = BASE_LAUNCHER.read_text(encoding="utf-8")
    arm = ARM_LAUNCHER.read_text(encoding="utf-8")
    suite = SUITE_LAUNCHER.read_text(encoding="utf-8")
    smoke = SMOKE_LAUNCHER.read_text(encoding="utf-8")
    accelerate = Path(
        "accelerate_configs/64_npus_4nodes_deepspeed_zero2.yaml"
    ).read_text(encoding="utf-8")

    assert "EXPECTED_NUM_MACHINES=4" in base
    assert "EXPECTED_WORLD_SIZE=64" in base
    assert (
        'export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"' in base
    )
    assert "num_machines: 4" in accelerate
    assert "num_processes: 64" in accelerate
    assert 'STOP_AFTER_STEPS="955"' in arm
    assert 'SAVE_EVERY="955"' in arm
    assert 'VAL_EVERY="955"' in arm
    assert 'VALIDATION_IMAGE_EVERY="1000000000"' in arm
    assert 'WANDB_MODE="disabled"' in arm
    assert 'BACKBONE_LR="${BACKBONE_LR:-2.4e-4}"' in base
    assert 'FLOW_LR="${FLOW_LR:-6.0e-5}"' in base
    assert 'SAVE_EMA_EVAL_EVERY="${SAVE_EMA_EVAL_EVERY:-12510}"' in base
    assert '--save-ema-eval-every "${SAVE_EMA_EVAL_EVERY}"' in base
    assert 'ABLATION="${ABLATION:-b}"' in base
    assert 'model.dual_stream_attention_contract=xlnet_content_diagonal' in base
    assert 'export ABLATION="b"' in arm
    assert 'export ABLATION="b"' in smoke
    assert 'BACKBONE_LR="${BACKBONE_LR:-2.4e-4}"' in smoke
    assert 'FLOW_LR="${FLOW_LR:-6.0e-5}"' in smoke

    for arm_id in (
        "b18e5-f4e5",
        "b18e5-f5e5",
        "b18e5-f6e5",
        "b21e5-f4e5",
        "b21e5-f5e5",
        "b21e5-f6e5",
        "b24e5-f4e5",
        "b24e5-f5e5",
        "b24e5-f6e5",
    ):
        assert arm_id in arm
        assert arm_id in suite
    assert "select_unified_lr_sweep.py" in suite
    assert "intentionally not auto-started" in suite
    assert "formal_ascend" not in suite

    for source in (base, arm, suite):
        assert "hashlib" not in source.lower()
        assert "sha256" not in source.lower()
