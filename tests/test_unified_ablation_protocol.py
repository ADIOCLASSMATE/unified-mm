from pathlib import Path

from omegaconf import OmegaConf

MANIFEST = Path("configs/protocols/unified_ablation_100b_ascend64.yaml")
LAUNCHER = Path(
    "script/selfless/pretraining_unified_ablation_100b_ascend64.sh"
)
B_LAUNCHER = Path(
    "script/selfless/pretraining_unified_ablation_b_0p6b_formal_ascend64.sh"
)
C_LAUNCHER = Path(
    "script/selfless/"
    "pretraining_unified_ablation_c_on_b_0p6b_formal_ascend64.sh"
)
D_LAUNCHER = Path(
    "script/selfless/"
    "pretraining_unified_ablation_d_on_b_0p6b_formal_ascend64.sh"
)
E_LAUNCHER = Path(
    "script/selfless/"
    "pretraining_unified_ablation_e_on_b_0p6b_formal_ascend64.sh"
)
F_LAUNCHER = Path(
    "script/selfless/"
    "pretraining_unified_ablation_f_on_b_0p6b_formal_ascend64.sh"
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

    assert protocol.schema == "unified_ablation_100b_v5"
    assert protocol.platform_project == (
        "多模态大模型新架构评测探索与scaling-law"
    )
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
    assert protocol.shared.image_flow_batch_mul == 4
    assert protocol.shared.start == "qwen_pretrained_step_0"
    assert protocol.shared.resume_from_checkpoint == "none"
    assert protocol.shared.selected_lr.rerun is False
    assert float(
        protocol.shared.selected_lr.backbone_and_special_lr
    ) == 3.0e-4
    assert float(
        protocol.shared.selected_lr.flow_and_projector_lr
    ) == 5.0e-5
    assert protocol.shared.imagenet.train.split == "train"
    assert protocol.shared.imagenet.validation.split == "val"
    assert protocol.shared.imagenet.train_validation_overlap_allowed is False

    assert set(protocol.ablations) == {"b", "c", "d", "e", "f"}
    assert set(protocol.run_projects) == {"b", "c", "d", "e", "f"}
    assert base.model.dual_stream_attention_contract == (
        "xlnet_content_diagonal"
    )
    assert base.model.flow_head_attention_contract == (
        "xlnet_content_diagonal"
    )
    assert protocol.ablations.b.start == "qwen_pretrained_step_0"
    assert protocol.ablations.b.status == "main_baseline"
    assert protocol.ablations.b.query_attention == "sigma_kv < sigma_q"
    assert protocol.ablations.b.content_attention == "sigma_kv <= sigma_q"
    assert protocol.ablations.b.flow_head_query_attention == (
        "sigma_kv < sigma_q"
    )
    assert protocol.ablations.b.flow_head_content_attention == (
        "sigma_kv <= sigma_q"
    )
    assert protocol.comparability.baseline == "b"
    assert protocol.comparability.b_vs_c_query_stream_identical is True
    assert protocol.comparability.b_vs_d_query_stream_identical is False
    assert protocol.comparability.qwen_pretrained_source_identical is True
    assert protocol.comparability.pretrained_backbone_weights_identical is True
    assert (
        protocol.comparability.architecture_specific_modules_initialized_from_training_seed
        == 42
    )
    assert protocol.comparability.b_vs_d_static_contract_identical is True
    assert protocol.comparability.b_vs_d_image_flow_batch_mul_identical is True
    assert protocol.ablations.c.start == "qwen_pretrained_step_0"
    assert protocol.comparability.b_vs_c_only_allowed_difference == (
        "model.architecture_variant"
    )
    assert protocol.ablations.c.architecture_variant == "single_stream_text_ar"
    assert protocol.ablations.c.training_objective == "selfless_dual_stream"
    assert protocol.ablations.c.text_stream == "single_x0"
    assert protocol.ablations.c.text_attention == "physical_position_causal"
    assert protocol.ablations.c.dual_stream_attention_contract == (
        "xlnet_content_diagonal"
    )
    assert protocol.ablations.c.flow_head_attention_contract == (
        "xlnet_content_diagonal"
    )
    assert protocol.ablations.c.image_stream == (
        "identical_to_b_selfless_dual_stream"
    )
    assert protocol.ablations.c.image_attention == "sigma_kv <= sigma_q"
    assert protocol.ablations.c.image_loss_positions == "identical_to_b"
    assert protocol.ablations.d.start == "qwen_pretrained_step_0"
    assert protocol.comparability.b_vs_d_only_allowed_difference == (
        "model.architecture_variant"
    )
    assert protocol.ablations.d.architecture_variant == "dynamic_xt"
    assert protocol.ablations.d.training_objective == "selfless_dual_stream"
    assert protocol.ablations.d.dual_stream_attention_contract == (
        "xlnet_content_diagonal"
    )
    assert protocol.ablations.d.flow_head_attention_contract == (
        "xlnet_content_diagonal"
    )
    assert protocol.ablations.d.query_attention == "sigma_kv < sigma_q"
    assert protocol.ablations.d.content_attention == "sigma_kv <= sigma_q"
    assert protocol.ablations.d.condition_refresh == (
        "every_ode_velocity_evaluation"
    )
    assert protocol.ablations.d.content_stream_batch == "B"
    assert protocol.ablations.d.dynamic_query_scope == "t2i_only"
    assert protocol.ablations.d.query_stream_batch == "4B_for_t2i_B_otherwise"
    assert protocol.ablations.d.image_flow_batch_mul == 4
    assert protocol.ablations.d.t2i_micro_batch_size_per_rank == 16
    assert protocol.ablations.d.rf_states_per_t2i_microbatch_per_rank == 64
    assert protocol.ablations.d.global_gradient_checkpointing is False
    assert protocol.ablations.d.gradient_checkpointing == (
        "t2i_dynamic_decoder_layers_only"
    )
    assert protocol.ablations.d.startup_loss_trace.until_step == 10
    assert protocol.ablations.d.startup_loss_trace.synchronization == (
        "one_gather_per_optimizer_boundary"
    )
    assert protocol.ablations.d.startup_loss_trace.changes_training_math is False
    assert protocol.ablations.d.startup_bf16_overflow_guard.until_step == 1
    assert (
        protocol.ablations.d.startup_bf16_overflow_guard.disabled_after_step
        == 1
    )
    assert (
        protocol.ablations.d.startup_bf16_overflow_guard.steady_state_overhead
        == "none"
    )
    assert protocol.comparability.systems_only_differences.d == [
        "t2i_dynamic_decoder_activation_checkpointing",
        "first_update_bf16_overflow_guard",
    ]
    assert (
        protocol.comparability.systems_only_differences_change_objective
        is False
    )
    assert protocol.ablations.d.time_embedder_optimizer_role == "backbone"
    assert protocol.ablations.d.content_compute == "once_per_layer"
    assert protocol.ablations.e.status == "ready"
    assert protocol.ablations.e.base == "b"
    assert protocol.ablations.e.architecture_variant == "selfless_contextual"
    assert protocol.ablations.e.image_sigma_order == "sequential"
    assert protocol.ablations.e.generation_order == "sequential"
    assert protocol.ablations.f.status == "ready"
    assert protocol.ablations.f.architecture_variant == (
        "positionwise_flow_head_on_b"
    )
    assert protocol.ablations.f.flow_head_attention_contract == (
        "not_applicable"
    )
    assert protocol.ablations.f.flow_head.content_stream is False
    assert protocol.ablations.f.flow_head.width == 1936
    assert protocol.ablations.f.flow_head.parameters == 163_828_208
    assert protocol.ablations.f.flow_head.reference_b_parameters == 164_072_976
    assert float(protocol.ablations.f.flow_head.relative_difference) < 0.005


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
    assert "baseline-b contract" in source
    assert "hashlib" not in source.lower()
    assert "sha256" not in source.lower()

    base_source = BASE_LAUNCHER.read_text(encoding="utf-8")
    assert 'SAVE_EMA_EVAL_EVERY="${SAVE_EMA_EVAL_EVERY:-12510}"' in base_source
    assert '--save-ema-eval-every "${SAVE_EMA_EVAL_EVERY}"' in base_source
    assert 'ABLATION="${ABLATION:-b}"' in base_source
    assert 'DUAL_STREAM_ATTENTION_CONTRACT="selfless_strict"' not in base_source
    assert 'DUAL_STREAM_ATTENTION_CONTRACT="xlnet_content_diagonal"' in base_source
    assert "unified-c-on-b-0p6b-100b-imagenet-split-s42-r1" in base_source
    assert "unified-d-on-b-0p6b-100b-imagenet-split-s42-r4" in base_source
    assert "unified-e-on-b-0p6b-100b-imagenet-split-s42-r1" in base_source
    assert "unified-f-on-b-0p6b-100b-imagenet-split-s42-r1" in base_source
    assert (
        '"model.dual_stream_attention_contract=${DUAL_STREAM_ATTENTION_CONTRACT}"'
        in base_source
    )
    assert 'ARCHITECTURE_VARIANT="single_stream_text_ar"' in base_source
    assert 'ARCHITECTURE_VARIANT="dynamic_xt"' in base_source
    assert 'ARCHITECTURE_VARIANT="positionwise_flow_head_on_b"' in base_source
    assert 'TRAIN_ENTRY="pretrain/train_selfless_flow_dynamic_xt.py"' in base_source
    assert '"model.image_flow_batch_mul=${IMAGE_FLOW_BATCH_MUL}"' in base_source
    assert (
        '"model.dynamic_xt_t2i_gradient_checkpointing=${DYNAMIC_XT_T2I_GRADIENT_CHECKPOINTING}"'
        in base_source
    )
    assert '"training.use_gradient_checkpointing=false"' in base_source
    assert (
        '"model.architecture_variant=${ARCHITECTURE_VARIANT}"' in base_source
    )
    assert '"model.training_image_sigma_order=${IMAGE_SIGMA_ORDER}"' in base_source
    assert '"dataset.params.image.image_sigma_order=${IMAGE_SIGMA_ORDER}"' in base_source
    assert '"model.image_flow_width=${FLOW_HEAD_WIDTH}"' in base_source
    assert (
        '"model.flow_head_attention_contract=${FLOW_HEAD_ATTENTION_CONTRACT}"'
        in base_source
    )
    assert 'DEFAULT_FLOW_HEAD_ATTENTION_CONTRACT="not_applicable"' in base_source
    assert '"model.image_flow_grad_checkpointing=false"' in base_source


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
    assert 'ABLATION="${ABLATION:-b}"' in smoke_source
    assert 'SAVE_FINAL="${SAVE_FINAL:-false}"' in smoke_source
    assert '"experiment.save_final=${SAVE_FINAL}"' in smoke_source
    assert "xlnet_content_diagonal" in smoke_source
    assert 'ARCHITECTURE_VARIANT="single_stream_text_ar"' in smoke_source
    assert "unified-c-on-b-qwen3-0.6b-smoke-ascend16" in smoke_source
    assert "unified-d-on-b-qwen3-0.6b-smoke-ascend16" in smoke_source
    assert 'TRAIN_ENTRY="pretrain/train_selfless_flow_dynamic_xt.py"' in smoke_source
    assert '"model.image_flow_batch_mul=${IMAGE_FLOW_BATCH_MUL}"' in smoke_source
    assert (
        '"model.dynamic_xt_t2i_gradient_checkpointing=${DYNAMIC_XT_T2I_GRADIENT_CHECKPOINTING}"'
        in smoke_source
    )
    assert '"training.use_gradient_checkpointing=false"' in smoke_source
    assert 'TRAINING_OBJECTIVE="showo_mae_flow"' not in smoke_source
    assert '--ablation "${ABLATION}"' in smoke_source


def test_dedicated_c_on_b_launcher_forces_fresh_b_based_run():
    source = C_LAUNCHER.read_text(encoding="utf-8")
    formal_source = LAUNCHER.read_text(encoding="utf-8")

    assert 'export ABLATION="c"' in source
    assert "unified-c-on-b-0p6b" in source
    assert 'export ALLOW_FORMAL_RESUME="false"' in source
    assert "retired C-on-A" in source
    assert "FORMAL_RESUME_FROM" not in source
    assert "pretraining_unified_ablation_100b_ascend64.sh" in source
    assert 'RESUME_FROM="none"' in formal_source
    assert 'FORMAL_RESUME_FROM is required for formal resume' in formal_source
    assert 'checkpoint_complete.json' in formal_source
    assert 'metadata.json' in formal_source


def test_dedicated_d_on_b_launcher_forces_fresh_mul4_run():
    source = D_LAUNCHER.read_text(encoding="utf-8")
    formal_source = LAUNCHER.read_text(encoding="utf-8")

    assert 'export ABLATION="d"' in source
    assert "unified-d-on-b-0p6b" in source
    assert "s42-r4" in source
    assert 'export ALLOW_FORMAL_RESUME="false"' in source
    assert "retired A-based Dynamic-XT" in source
    assert "FORMAL_RESUME_FROM" not in source
    assert "pretraining_unified_ablation_100b_ascend64.sh" in source
    assert 'export IMAGE_FLOW_BATCH_MUL="4"' in source
    assert 'export DYNAMIC_XT_T2I_GRADIENT_CHECKPOINTING="true"' in source
    assert (
        'export FLOW_HEAD_ATTENTION_CONTRACT="xlnet_content_diagonal"'
        in source
    )
    assert 'export DEEPSPEED_BF16_OVERFLOW_CHECK_UNTIL_STEP="1"' in source
    assert 'export DEBUG_LOSS_TRACE_UNTIL_STEP="10"' in source
    assert "synchronizing individual microbatches" in source
    assert "ABLATION must be b, c, d, e, or f" in formal_source


def test_dedicated_e_f_launchers_are_fresh_b_based_and_isolated():
    e_source = E_LAUNCHER.read_text(encoding="utf-8")
    f_source = F_LAUNCHER.read_text(encoding="utf-8")
    formal_source = LAUNCHER.read_text(encoding="utf-8")

    for arm, source in (("e", e_source), ("f", f_source)):
        assert f'export ABLATION="{arm}"' in source
        assert f"unified-{arm}-on-b-0p6b" in source
        assert 'export ALLOW_FORMAL_RESUME="false"' in source
        assert 'export IMAGE_FLOW_BATCH_MUL="4"' in source
        assert "pretraining_unified_ablation_100b_ascend64.sh" in source
    assert "deterministic serialized left-to-right" in e_source
    assert "dedicated parameter-" in f_source
    assert (
        'export FLOW_HEAD_ATTENTION_CONTRACT="xlnet_content_diagonal"'
        in e_source
    )
    assert 'export FLOW_HEAD_ATTENTION_CONTRACT="not_applicable"' in f_source
    assert 'e) ARM_NAME="e-on-b"' in formal_source
    assert 'f) ARM_NAME="f-on-b"' in formal_source
