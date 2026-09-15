from pathlib import Path

import torch
from omegaconf import OmegaConf

import utils.selfless_training_runtime as runtime
import utils.sharded_ema as sharded_ema
from utils.dataset_imagenet_flow_cache import ImageNetFlowCacheDataset


def test_baseline_configuration_freezes_no_hash_training_contract(monkeypatch):
    config = OmegaConf.load(
        "configs/selfless/unified_baseline_100b_ascend_64npu.yaml"
    )
    assert config.training.runtime_hashing_enabled is False
    assert "caption_manifest_sha256" not in config.dataset.params.image
    assert list(config.dataset.params.schedule) == [
        "climbmix",
        "t2i",
        "climbmix",
        "i2t",
    ]
    assert config.dataset.params.image.image_sigma_order == "random"
    assert config.model.backbone_attention_output_gate == "none"
    assert int(config.experiment.save_every) == 2_000
    assert int(config.experiment.checkpoints_total_limit) == 3
    assert int(config.experiment.checkpoint_milestone_every) == 125_100
    assert bool(config.experiment.save_final_checkpoint)
    assert int(config.experiment.save_ema_eval_every) == 25_020
    assert bool(config.experiment.save_model_with_ema_eval)
    assert str(config.experiment.ema_eval_dtype) == "bf16"
    assert int(config.experiment.val_every) == 10_000
    assert int(config.experiment.downstream_validation.seed) == 424_242
    assert int(config.experiment.downstream_validation.mc_samples) == 16
    assert not {
        "validation_image_every",
        "validation_i2t_every",
        "validation_single_stream_parallel_rate",
    }.intersection(config.experiment)

    contract = runtime.build_resume_contract(
        config,
        world_size=64,
        gradient_accumulation_steps=4,
    )
    assert contract["world_size"] == 64
    assert "sha256" not in str(contract).lower()

    layout = sharded_ema.build_sharded_ema_layout(
        torch.nn.Linear(4, 4),
        world_size=2,
        chunk_numel=4,
    )
    assert layout["layout_validation"] == "readable_field_equality"
    assert "layout_fingerprint" not in layout

    dataset = ImageNetFlowCacheDataset.__new__(ImageNetFlowCacheDataset)
    dataset.seed = 42
    assert dataset._stable_sample_seed(7, 3, "vae_posterior") >= 0

    dataset_source = Path(
        "utils/dataset_imagenet_flow_cache.py"
    ).read_text(encoding="utf-8")
    assert "hashlib" not in dataset_source
    assert "runtime_hashing_enabled" not in dataset_source
    assert "caption_manifest_sha256" not in dataset_source


def test_project_formal_fid50k_launcher_uses_independent_val_and_no_hashing():
    launcher = Path(
        "script/selfless/evaluate_unified_t2i_fid_is_ascend16.sh"
    ).read_text(encoding="utf-8")

    assert 'SAMPLES="${T2I_SAMPLES:-50000}"' in launcher
    assert '--samples "${SAMPLES}"' in launcher
    assert "--require_formal_protocol" in launcher
    assert "--no_runtime_hashing" not in launcher
    assert "--split" not in launcher
    assert "--is_split_assignment" not in launcher
    assert "posterior_stats_imagenet1k_val_fp16.pt" in launcher
    assert "manifest_val.jsonl" in launcher
    assert "indexed/val/manifest.json" in launcher
    assert "sha256sum" not in launcher
    assert "hashlib" not in launcher


def test_baseline_launchers_disable_parent_tokenizer_parallelism():
    for path in (
        Path("script/selfless/smoke_unified_baseline_ascend16.sh"),
        Path(
            "script/selfless/"
            "pretraining_unified_baseline_ascend_64npu_100b.sh"
        ),
    ):
        launcher = path.read_text(encoding="utf-8")
        assert "export TOKENIZERS_PARALLELISM=false" in launcher
        assert "export TOKENIZERS_PARALLELISM=true" not in launcher
        assert 'WANDB_MODE="${WANDB_MODE:-disabled}"' in launcher

    smoke_launcher = Path(
        "script/selfless/smoke_unified_baseline_ascend16.sh"
    ).read_text(encoding="utf-8")
    assert (
        'VALIDATION_SINGLE_STREAM_PARALLEL_RATE="${VALIDATION_SINGLE_STREAM_PARALLEL_RATE:-1}"'
        in smoke_launcher
    )


def test_unified_training_can_skip_tracker_initialization():
    source = Path("pretrain/train_selfless_flow.py").read_text(
        encoding="utf-8"
    )
    assert 'os.environ.setdefault("WANDB_MODE", "disabled")' in source
    assert 'os.environ.get("WANDB_MODE", "disabled")' in source
    assert 'log_with="wandb" if use_wandb_tracker else None' in source
    assert "accelerator.is_main_process and use_wandb_tracker" in source

    dataloader_source = Path("utils/imagenet_flow_dataloaders.py").read_text(
        encoding="utf-8"
    )
    assert "dataset.params.validation is required" in dataloader_source
    assert "_assert_independent_imagenet_splits" in dataloader_source
    assert "validation_overlap_train" not in dataloader_source


def test_pretraining_native_evaluation_never_calculates_content_hashes():
    for path in (
        Path("scripts/evaluate_imagenet_pretraining_native.py"),
        Path("scripts/summarize_pretraining_native_understanding.py"),
        Path("scripts/summarize_unified_native_full_evaluation.py"),
        Path("scripts/summarize_unified_native_checkpoint_trend.py"),
        Path("script/selfless/evaluate_pretraining_native_imagenet_ascend16.sh"),
        Path("script/selfless/evaluate_multimodal_likelihood_ascend16.sh"),
        Path(
            "script/selfless/"
            "evaluate_unified_native_full_checkpoint_ascend16.sh"
        ),
    ):
        source = path.read_text(encoding="utf-8").lower()
        assert "hashlib" not in source
        assert "sha256sum" not in source
        assert "runtime_hashing_enabled" in source

    protocol = OmegaConf.load(
        "configs/protocols/pretraining_native_understanding_evaluation_ascend16.yaml"
    )
    assert protocol.runtime_hashing_enabled is False
    assert protocol.outputs.no_hashes is True
    assert protocol.scoring.retrieval_and_classification_primary_score == (
        "language_prior_debiased_mean_token_loglikelihood"
    )
    assert protocol.outputs.image_text_matching_score_variant == (
        "conditional_aro_debiased_other_tasks"
    )

    evaluator = Path("scripts/evaluate_imagenet_pretraining_native.py").read_text(
        encoding="utf-8"
    )
    launcher = Path(
        "script/selfless/evaluate_pretraining_native_imagenet_ascend16.sh"
    ).read_text(encoding="utf-8")
    assert "visual_calibrated_loglikelihood" not in evaluator
    assert "CALIBRATION_IMAGES" not in launcher
    assert "language_prior_debiased_scores" in evaluator
    assert "LANGUAGE_PRIOR_ALPHA" in evaluator


def test_removed_image_text_protocols_have_no_executable_entrypoint():
    assert not Path("scripts/evaluate_imagenet1k_i2t_clip.py").exists()
    assert not Path(
        "script/selfless/evaluate_t2i_fid50k_official_ascend16.sh"
    ).exists()
    imagenet = Path(
        "scripts/evaluate_imagenet_pretraining_native.py"
    ).read_text(encoding="utf-8")
    assert "retrieval_1k" not in imagenet
    assert "retrieval_5k" not in imagenet


def test_formal_image_text_launchers_pin_complete_protocols():
    imagenet = Path(
        "script/selfless/evaluate_pretraining_native_imagenet_ascend16.sh"
    ).read_text(encoding="utf-8")
    retrieval = Path(
        "script/selfless/evaluate_cross_dataset_retrieval_ascend16.sh"
    ).read_text(encoding="utf-8")
    multimodal = Path(
        "script/selfless/evaluate_multimodal_likelihood_ascend16.sh"
    ).read_text(encoding="utf-8")
    assert "LIMIT=0" in imagenet and "--require_formal_protocol" in imagenet
    assert "LIMIT=0" in retrieval and "--require_formal_protocol" in retrieval
    assert "LIMIT=0" in multimodal and "MC=64" in multimodal
    assert "--require_formal_protocol" in multimodal
    assert "vae_posterior_mar_kl16_v2" in multimodal
