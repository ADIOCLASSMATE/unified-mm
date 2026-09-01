from pathlib import Path

from omegaconf import OmegaConf


MANIFEST = Path("configs/protocols/unified_ablation_100b_ascend64.yaml")
LAUNCHER = Path(
    "script/selfless/pretraining_unified_ablation_100b_ascend64.sh"
)
B_LAUNCHER = Path(
    "script/selfless/pretraining_unified_ablation_b_0p6b_formal_ascend64.sh"
)
RESUME_C_LAUNCHER = Path(
    "script/selfless/pretraining_unified_ablation_c_resume_ascend64.sh"
)
BASE_LAUNCHER = Path(
    "script/selfless/pretraining_unified_baseline_ascend_64npu_100b.sh"
)
SMOKE_LAUNCHER = Path(
    "script/selfless/smoke_unified_baseline_ascend16.sh"
)


def test_formal_protocol_is_comparable_complete_and_no_hash():
    protocol = OmegaConf.load(MANIFEST)
    base = OmegaConf.load(protocol.base_config)

    assert protocol.schema == "unified_ablation_100b_v2"
    assert protocol.world_size == 64
    assert protocol.nodes * protocol.npu_per_node == protocol.world_size
    assert protocol.target_text_tokens == 100_000_000_000
    assert protocol.max_train_steps == 95_415
    assert protocol.save_every == 2_000
    assert protocol.save_ema_eval_every == 0
    assert protocol.validation_every == protocol.save_every
    assert protocol.checkpoints_total_limit == 3
    assert protocol.checkpoint_milestone_every == 0
    assert protocol.save_final_resumable_checkpoint is True
    assert protocol.qualitative_outputs.generated_images is True
    assert protocol.qualitative_outputs.image_to_text_captions is True
    assert protocol.runtime_hashing_enabled is False
    assert list(protocol.task_loss_logging.raw_normalized) == [
        "train/loss_climbmix",
        "train/loss_t2i",
        "train/loss_i2t",
    ]
    assert list(protocol.task_loss_logging.weighted_contributions) == [
        "train/weighted_contribution_climbmix",
        "train/weighted_contribution_t2i",
        "train/weighted_contribution_i2t",
    ]
    assert protocol.task_loss_logging.contribution_sum_matches == "step_loss"
    assert float(protocol.task_loss_logging.model_weights.text) == 0.05
    assert float(protocol.task_loss_logging.model_weights.image) == 1.0
    assert int(
        protocol.task_loss_logging.accumulation.gradient_accumulation_steps
    ) == 4
    assert protocol.task_loss_logging.persistent_jsonl == (
        "training_metrics.jsonl"
    )
    assert base.training.runtime_hashing_enabled is False
    assert protocol.shared.climbmix_tokenization == "online"
    assert protocol.shared.text_sequence_length == 2048
    assert protocol.shared.start == "qwen_pretrained_step_0"
    assert protocol.shared.resume_from_checkpoint == "none"
    assert protocol.shared.historical_lr_sweep.rerun is False
    assert float(
        protocol.shared.historical_lr_sweep.backbone_and_special_lr
    ) == 3.0e-4
    assert float(
        protocol.shared.historical_lr_sweep.flow_and_projector_lr
    ) == 5.0e-5
    assert protocol.shared.imagenet.train.split == "train"
    assert protocol.shared.imagenet.validation.split == "val"
    assert protocol.shared.imagenet.train_validation_overlap_allowed is False

    assert protocol.ablations.a.start == "qwen_pretrained_step_0"
    assert protocol.ablations.b.start == "qwen_pretrained_step_0"
    assert protocol.ablations.a.query_attention == (
        protocol.ablations.b.query_attention
    )
    assert protocol.ablations.a.content_attention == "sigma_kv < sigma_q"
    assert protocol.ablations.b.content_attention == "sigma_kv <= sigma_q"
    assert protocol.comparability.a_vs_b_only_allowed_difference == (
        "model.dual_stream_attention_contract"
    )
    assert protocol.comparability.query_stream_identical is True
    assert protocol.ablations.c.start == "qwen_pretrained_step_0"
    assert protocol.comparability.a_vs_c_only_allowed_difference == (
        "model.architecture_variant"
    )
    assert protocol.ablations.c.architecture_variant == "single_stream_text_ar"
    assert protocol.ablations.c.training_objective == "selfless_dual_stream"
    assert protocol.ablations.c.text_stream == "single_x0"
    assert protocol.ablations.c.text_attention == "physical_position_causal"
    assert protocol.ablations.c.image_stream == (
        "identical_to_a_selfless_dual_stream"
    )
    assert protocol.ablations.c.image_loss_positions == "identical_to_a"
    assert protocol.ablations.g.status == "deferred"


def test_formal_launcher_freezes_current_baseline_contract():
    source = LAUNCHER.read_text(encoding="utf-8")
    assert 'BACKBONE_LR="3.0e-4"' in source
    assert 'FLOW_LR="5.0e-5"' in source
    assert 'RESUME_FROM="none"' in source
    assert 'SAVE_EVERY="2000"' in source
    assert 'CHECKPOINTS_TOTAL_LIMIT="3"' in source
    assert 'CHECKPOINT_MILESTONE_EVERY="0"' in source
    assert 'VALIDATION_IMAGE_EVERY="2000"' in source
    assert 'VALIDATION_I2T_EVERY="2000"' in source
    assert 'SAVE_EMA_EVAL_EVERY="0"' in source
    assert 'SAVE_FINAL_CHECKPOINT="true"' in source
    assert 'PRESERVE_MODEL_CONTRACT="false"' in source
    assert 'WANDB_MODE="disabled"' in source
    assert "SELECTION" not in source
    assert "sweep checkpoint" in source
    assert "hashlib" not in source.lower()
    assert "sha256" not in source.lower()

    base_source = BASE_LAUNCHER.read_text(encoding="utf-8")
    assert 'SAVE_EMA_EVAL_EVERY="${SAVE_EMA_EVAL_EVERY:-12510}"' in base_source
    assert '--save-ema-eval-every "${SAVE_EMA_EVAL_EVERY}"' in base_source
    assert 'DUAL_STREAM_ATTENTION_CONTRACT="selfless_strict"' in base_source
    assert (
        'DUAL_STREAM_ATTENTION_CONTRACT="xlnet_content_diagonal"'
        in base_source
    )
    assert (
        '"model.dual_stream_attention_contract=${DUAL_STREAM_ATTENTION_CONTRACT}"'
        in base_source
    )
    assert 'ARCHITECTURE_VARIANT="single_stream_text_ar"' in base_source
    assert (
        '"model.architecture_variant=${ARCHITECTURE_VARIANT}"' in base_source
    )


def test_dedicated_b_launcher_selects_only_xlnet_content_diagonal_arm():
    source = B_LAUNCHER.read_text(encoding="utf-8")
    assert 'export ABLATION="b"' in source
    assert "BACKBONE_LR" not in source
    assert "FLOW_LR" not in source
    assert "RESUME_FROM" not in source
    assert (
        "exec bash script/selfless/"
        "pretraining_unified_ablation_100b_ascend64.sh"
    ) in source

    smoke_source = SMOKE_LAUNCHER.read_text(encoding="utf-8")
    assert 'ABLATION="${ABLATION:-a}"' in smoke_source
    assert "xlnet_content_diagonal" in smoke_source
    assert 'ARCHITECTURE_VARIANT="single_stream_text_ar"' in smoke_source
    assert 'TRAINING_OBJECTIVE="showo_mae_flow"' not in smoke_source
    assert '--ablation "${ABLATION}"' in smoke_source


def test_dedicated_c_resume_launcher_is_scoped_to_its_completed_checkpoint():
    source = RESUME_C_LAUNCHER.read_text(encoding="utf-8")
    formal_source = LAUNCHER.read_text(encoding="utf-8")

    assert 'export ABLATION="c"' in source
    assert 'export ALLOW_FORMAL_RESUME="true"' in source
    assert 'checkpoint-${RESUME_STEP}' in source
    assert 'resume-step-${RESUME_STEP}-${RESUME_ATTEMPT}' in source
    assert "pretraining_unified_ablation_100b_ascend64.sh" in source
    assert 'RESUME_FROM="none"' in formal_source
    assert 'FORMAL_RESUME_FROM is required for formal resume' in formal_source
    assert 'checkpoint_complete.json' in formal_source
    assert 'metadata.json' in formal_source
