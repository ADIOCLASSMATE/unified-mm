#!/usr/bin/env python3
"""Fail-closed preflight for the 0.6B token-matched single-source runs.

The protocol is intentionally checked twice: all three run configurations are
validated structurally on every invocation, while heavyweight dataset audits
are performed only for the selected active source(s).  This lets a pure-text
job remain independent of ImageNet availability, and vice versa.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path
from typing import Any, Iterable

import torch
from omegaconf import DictConfig, OmegaConf
from transformers import AutoConfig, AutoTokenizer

from scripts.validate_unified_baseline import (
    _audit_imagenet_manifest,
    _load_image_cache,
    _require_file,
    _validate_text_index,
)
from utils.climbmix_online_dataset import ClimbMixOnlineBatchDataset
from utils.imagenet_synthetic_text_index import ImageNetSyntheticTextIndex
from utils.selfless_training_runtime import validate_wsd_contract


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = (
    REPO_ROOT
    / "configs/protocols/unified_single_source_0p6b_100b_ascend16.yaml"
)
SUPPORTED_SOURCES = ("climbmix", "i2t", "t2i")
EXPECTED_TARGET_PHYSICAL_TOKENS = 100_000_595_968
EXPECTED_WORLD_SIZE = 16
EXPECTED_MODEL_PATH = "public/models/Qwen--Qwen3-0.6B-Base"


def _repo_path(value: str | Path) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else REPO_ROOT / path


def _normalized_path(value: str | Path) -> Path:
    return _repo_path(value).resolve()


def _plain(value: Any) -> Any:
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _expect(label: str, actual: Any, expected: Any) -> None:
    actual = _plain(actual)
    expected = _plain(expected)
    if isinstance(expected, float):
        try:
            matches = math.isclose(
                float(actual), expected, rel_tol=0.0, abs_tol=1.0e-15
            )
        except (TypeError, ValueError):
            matches = False
    else:
        matches = actual == expected
    if not matches:
        raise ValueError(f"{label} mismatch: {actual!r} != {expected!r}")


def _noneish(value: Any) -> bool:
    return value is None or str(value).strip().lower() in {"", "none", "null"}


def _source_from_schedule(config: DictConfig) -> tuple[str, tuple[str, ...]]:
    if str(config.dataset.class_name) != "UnifiedMixedDataset":
        raise ValueError("dataset.class_name must be UnifiedMixedDataset")
    schedule = tuple(
        str(source).strip().lower() for source in config.dataset.params.schedule
    )
    if not schedule:
        raise ValueError("dataset.params.schedule must not be empty")
    active = set(schedule)
    if len(active) != 1 or not active.issubset(SUPPORTED_SOURCES):
        raise ValueError(
            "single-source schedule must repeat exactly one supported source; "
            f"got {list(schedule)}"
        )
    return schedule[0], schedule


def _validate_global_protocol(protocol: DictConfig) -> DictConfig:
    _expect("protocol.schema", protocol.schema, "unified_single_source_0p6b_100b_v2")
    _expect("protocol.world_size", int(protocol.world_size), EXPECTED_WORLD_SIZE)
    _expect("protocol.nodes", int(protocol.nodes), 1)
    _expect("protocol.npu_per_node", int(protocol.npu_per_node), 16)
    _expect(
        "protocol node topology",
        int(protocol.nodes) * int(protocol.npu_per_node),
        int(protocol.world_size),
    )
    _expect("protocol.seed", int(protocol.seed), 42)
    _expect(
        "protocol.target_physical_tokens_per_run",
        int(protocol.target_physical_tokens_per_run),
        EXPECTED_TARGET_PHYSICAL_TOKENS,
    )
    _expect("protocol.initialization", protocol.initialization, "qwen3_0p6b_base_step_zero")
    _expect("protocol.architecture_variant", protocol.architecture_variant, "selfless_contextual")
    _expect("protocol.training_objective", protocol.training_objective, "selfless_dual_stream")
    _expect(
        "protocol.dual_stream_attention_contract",
        protocol.dual_stream_attention_contract,
        "xlnet_content_diagonal",
    )
    _expect(
        "protocol.flow_head_attention_contract",
        protocol.flow_head_attention_contract,
        "xlnet_content_diagonal",
    )
    _expect(
        "protocol.flow_condition_contract",
        protocol.flow_condition_contract,
        "backbone_xt_shared_query_content",
    )
    _expect("protocol.runtime_hashing_enabled", bool(protocol.runtime_hashing_enabled), False)
    _expect("protocol.wandb_mode", protocol.wandb_mode, "disabled")

    loss = protocol.loss_scaling_contract
    _expect("loss scaling mode", loss.mode, "full_single_task_objective")
    _expect("loss lambda_text", float(loss.lambda_text), 0.05)
    _expect("loss lambda_image", float(loss.lambda_image), 1.0)
    _expect(
        "combined-gradient dilution flag",
        bool(loss.matches_combined_source_gradient_fraction),
        False,
    )

    combined = protocol.combined_reference
    _expect("combined world size", int(combined.world_size), 64)
    _expect(
        "combined source schedule",
        list(combined.source_schedule),
        ["climbmix", "t2i", "climbmix", "i2t"],
    )
    computed_combined_total = 0
    for source in SUPPORTED_SOURCES:
        reference = combined.source_contributions[source]
        occurrences = list(combined.source_schedule).count(source)
        _expect(
            f"combined {source} occurrence count",
            int(reference.occurrences),
            occurrences,
        )
        width_key = (
            "sequence_length" if source == "climbmix" else "padded_sequence_length"
        )
        computed = (
            int(combined.world_size)
            * occurrences
            * int(reference.per_rank_micro_batch)
            * int(reference[width_key])
        )
        _expect(
            f"combined {source} physical tokens",
            int(reference.physical_tokens_per_optimizer_step),
            computed,
        )
        computed_combined_total += computed
    _expect(
        "combined physical tokens per optimizer step",
        int(combined.physical_tokens_per_optimizer_step),
        computed_combined_total,
    )

    _expect(
        "single-source run set",
        set(protocol.single_source_runs.keys()),
        set(SUPPORTED_SOURCES),
    )
    for source in SUPPORTED_SOURCES:
        run = protocol.single_source_runs[source]
        reference = combined.source_contributions[source]
        _expect(
            f"protocol {source} per-step source contribution",
            int(run.physical_tokens_per_optimizer_step),
            int(reference.physical_tokens_per_optimizer_step),
        )
        _expect(
            f"protocol {source} target",
            int(run.actual_physical_tokens),
            EXPECTED_TARGET_PHYSICAL_TOKENS,
        )
        _expect(
            f"protocol {source} max*per-step",
            int(run.max_train_steps)
            * int(run.physical_tokens_per_optimizer_step),
            EXPECTED_TARGET_PHYSICAL_TOKENS,
        )
        _expect(
            f"protocol {source} overshoot",
            int(run.overshoot_physical_tokens),
            EXPECTED_TARGET_PHYSICAL_TOKENS - 100_000_000_000,
        )

    optimizer = protocol.shared_optimizer
    _expect("shared backbone LR", float(optimizer.backbone_and_special_learning_rate), 3.0e-4)
    _expect("shared flow LR", float(optimizer.flow_and_projector_learning_rate), 5.0e-5)
    _expect("shared beta1", float(optimizer.beta1), 0.9)
    _expect("shared beta2", float(optimizer.beta2), 0.95)
    _expect("shared weight decay", float(optimizer.weight_decay), 0.01)
    _expect("shared flow weight decay", float(optimizer.flow_weight_decay), 0.01)
    _expect("shared max grad norm", float(optimizer.max_grad_norm), 1.0)

    checkpointing = protocol.shared_checkpointing
    _expect("shared save cadence", int(checkpointing.save_every), 2000)
    _expect("shared validation cadence", int(checkpointing.validation_every), 10000)
    _expect("shared checkpoint retention", int(checkpointing.checkpoints_total_limit), 3)
    _expect("shared checkpoint milestones", int(checkpointing.checkpoint_milestone_every), 125_100)
    _expect("shared paired-model evaluation export cadence", int(checkpointing.save_ema_eval_every), 25_020)
    _expect("shared paired-model evaluation export enabled", bool(checkpointing.save_model_with_ema_eval), True)
    _expect(
        "shared final resumable checkpoint",
        bool(checkpointing.save_final_resumable_checkpoint),
        True,
    )

    base_path = _repo_path(protocol.base_config)
    if not base_path.is_file():
        raise FileNotFoundError(f"missing protocol base config: {base_path}")
    return OmegaConf.load(base_path)


def _validate_shared_model_optimizer(
    config: DictConfig,
    base: DictConfig,
    protocol: DictConfig,
    *,
    source: str,
) -> None:
    # The historical single-source controls remain configuration-equivalent to
    # their original B baseline. The combined baseline now opts into split
    # XT/X0 conditioning, so override only that newly versioned field before
    # comparing the otherwise identical model contracts.
    expected_model = OmegaConf.create(
        OmegaConf.to_container(base.model, resolve=True)
    )
    expected_model.flow_condition_contract = (
        protocol.flow_condition_contract
    )
    _expect(f"{source} model contract", config.model, expected_model)
    _expect(f"{source} optimizer contract", config.optimizer, base.optimizer)
    _expect(f"{source} model_path", config.model.model_path, EXPECTED_MODEL_PATH)
    _expect(
        f"{source} architecture",
        config.model.architecture_variant,
        protocol.architecture_variant,
    )
    _expect(
        f"{source} training objective",
        config.model.training_objective,
        protocol.training_objective,
    )
    _expect(
        f"{source} attention contract",
        config.model.dual_stream_attention_contract,
        protocol.dual_stream_attention_contract,
    )
    _expect(
        f"{source} flow-head attention contract",
        config.model.flow_head_attention_contract,
        protocol.flow_head_attention_contract,
    )
    _expect(
        f"{source} flow condition contract",
        config.model.flow_condition_contract,
        protocol.flow_condition_contract,
    )
    _expect(
        f"{source} lambda_text",
        float(config.model.lambda_text),
        float(protocol.loss_scaling_contract.lambda_text),
    )
    _expect(
        f"{source} lambda_image",
        float(config.model.lambda_image),
        float(protocol.loss_scaling_contract.lambda_image),
    )
    _expect(f"{source} backbone output gate", config.model.backbone_attention_output_gate, "none")

    optimizer = config.optimizer.params
    shared = protocol.shared_optimizer
    _expect(f"{source} optimizer name", config.optimizer.name, "adamw")
    _expect(f"{source} learning rate", float(optimizer.learning_rate), float(shared.backbone_and_special_learning_rate))
    _expect(f"{source} backbone LR", float(optimizer.backbone_learning_rate), float(shared.backbone_and_special_learning_rate))
    _expect(f"{source} special-token LR", float(optimizer.special_token_learning_rate), float(shared.backbone_and_special_learning_rate))
    _expect(f"{source} projector LR", float(optimizer.projector_learning_rate), float(shared.flow_and_projector_learning_rate))
    _expect(f"{source} flow LR", float(optimizer.flow_learning_rate), float(shared.flow_and_projector_learning_rate))
    _expect(f"{source} scale_lr", bool(optimizer.scale_lr), False)
    _expect(f"{source} beta1", float(optimizer.beta1), float(shared.beta1))
    _expect(f"{source} beta2", float(optimizer.beta2), float(shared.beta2))
    _expect(f"{source} weight_decay", float(optimizer.weight_decay), float(shared.weight_decay))
    _expect(f"{source} flow_weight_decay", float(optimizer.flow_weight_decay), float(shared.flow_weight_decay))


def _validate_training_runtime_contract(
    config: DictConfig,
    protocol: DictConfig,
    run: DictConfig,
    *,
    source: str,
) -> None:
    training = config.training
    experiment = config.experiment
    checkpointing = protocol.shared_checkpointing

    _expect(f"{source} training seed", int(training.seed), int(protocol.seed))
    _expect(f"{source} shuffle seed", int(training.dataloader_shuffle_seed), int(protocol.seed))
    _expect(f"{source} runtime hashing", bool(training.runtime_hashing_enabled), False)
    _expect(f"{source} target physical tokens", int(training.target_physical_tokens), EXPECTED_TARGET_PHYSICAL_TOKENS)
    _expect(f"{source} max train steps", int(training.max_train_steps), int(run.max_train_steps))
    _expect(f"{source} formal stop", int(training.stop_after_steps), int(training.max_train_steps))
    _expect(
        f"{source} max*per-step",
        int(training.max_train_steps) * int(training.physical_tokens_per_optimizer_step),
        EXPECTED_TARGET_PHYSICAL_TOKENS,
    )
    _expect(f"{source} from_scratch", bool(training.from_scratch), False)
    if not _noneish(experiment.resume_from_checkpoint):
        raise ValueError(f"{source} must start at Qwen step zero, not resume")
    if not _noneish(config.model.pretrained_image_flow_adapter):
        raise ValueError(f"{source} must not preload an image-flow adapter")

    _expect(f"{source} max grad norm", float(training.max_grad_norm), float(protocol.shared_optimizer.max_grad_norm))
    _expect(f"{source} trainable scope", training.trainable_scope, "full")
    _expect(f"{source} EMA enabled", bool(training.use_ema), True)
    _expect(f"{source} EMA decay", float(training.ema_decay), 0.9999)
    _expect(f"{source} EMA start", int(training.ema_update_after_step), 0)
    _expect(f"{source} EMA validation", bool(training.ema_validate), True)
    _expect(f"{source} EMA adapter save", bool(training.ema_save_adapter), True)
    _expect(f"{source} EMA HF save", bool(training.ema_save_hf_model), True)
    _expect(
        f"{source} paired-model evaluation export cadence",
        int(experiment.save_ema_eval_every),
        int(checkpointing.save_ema_eval_every),
    )
    _expect(
        f"{source} paired-model evaluation export enabled",
        bool(experiment.save_model_with_ema_eval),
        True,
    )
    _expect(
        f"{source} complete EMA evaluation export dtype",
        str(experiment.ema_eval_dtype).lower(),
        "bf16",
    )
    _expect(f"{source} mixed precision", training.mixed_precision, "bf16")
    _expect(f"{source} accumulation dtype", training.gradient_accumulation_dtype, "fp32")

    _expect(f"{source} save cadence", int(experiment.save_every), int(checkpointing.save_every))
    _expect(f"{source} validation cadence", int(experiment.val_every), int(checkpointing.validation_every))
    _expect(f"{source} checkpoint retention", int(experiment.checkpoints_total_limit), int(checkpointing.checkpoints_total_limit))
    _expect(f"{source} checkpoint milestones", int(experiment.checkpoint_milestone_every), int(checkpointing.checkpoint_milestone_every))
    _expect(f"{source} save_final", bool(experiment.save_final), True)
    _expect(f"{source} save final checkpoint", bool(experiment.save_final_checkpoint), True)

    _expect(f"{source} scheduler", config.lr_scheduler.scheduler, "wsd")
    _expect(f"{source} WSD warmup", int(config.lr_scheduler.params.warmup_steps), int(run.warmup_steps))
    _expect(f"{source} WSD decay", int(config.lr_scheduler.params.decay_steps), int(run.decay_steps))
    _expect(f"{source} WSD min scale", float(config.lr_scheduler.params.min_lr_scale), 0.1)
    _expect(
        f"{source} scheduler learning rate",
        float(config.lr_scheduler.params.learning_rate),
        float(config.optimizer.params.learning_rate),
    )
    validate_wsd_contract(config)

    _expect(f"{source} experiment name/project", experiment.name, experiment.project)
    expected_output = Path(str(experiment.output_dir)) / str(experiment.project)
    _expect(f"{source} protocol output root", Path(str(run.output_root)), expected_output)


def _validate_batch_token_contract(
    config: DictConfig,
    protocol: DictConfig,
    run: DictConfig,
    *,
    source: str,
) -> dict[str, Any]:
    observed_source, schedule = _source_from_schedule(config)
    _expect("selected active source", observed_source, source)
    sources = config.dataset.params.sources
    _expect(f"{source} configured source set", set(sources.keys()), {source})
    source_config = sources[source]

    accumulation = int(config.training.gradient_accumulation_steps)
    _expect(f"{source} gradient accumulation", accumulation, len(schedule))
    _expect(f"{source} schedule repetition", list(schedule), [source] * accumulation)
    _expect(f"{source} protocol gradient accumulation", accumulation, int(run.gradient_accumulation_steps))

    micro_batch = int(source_config.micro_batch_size)
    _expect(f"{source} per-rank microbatch", micro_batch, int(run.per_rank_micro_batch))
    _expect(f"{source} training.batch_size", int(config.training.batch_size), micro_batch)

    if source == "climbmix":
        width = int(source_config.sequence_length)
        widths = [width] * accumulation
        _expect(f"{source} preprocessing length", int(config.dataset.preprocessing.max_seq_length), width)
    else:
        if "pad_to_length_schedule" not in source_config:
            raise ValueError(f"{source} requires pad_to_length_schedule")
        widths = [int(width) for width in source_config.pad_to_length_schedule]
        _expect(f"{source} scheduled width count", len(widths), accumulation)
        if any(width <= 0 for width in widths):
            raise ValueError(f"{source} pad widths must all be positive: {widths}")
        image = config.dataset.params.image
        max_seq_length = int(image.max_seq_length)
        _expect(f"{source} preprocessing max length", int(config.dataset.preprocessing.max_seq_length), max_seq_length)
        if max(widths) > max_seq_length:
            raise ValueError(
                f"{source} scheduled width exceeds max_seq_length: "
                f"{max(widths)} > {max_seq_length}"
            )
        multiple = int(image.pad_to_multiple_of)
        if any(width % multiple for width in widths):
            raise ValueError(
                f"{source} scheduled widths must be multiples of {multiple}: {widths}"
            )
        _expect(f"{source} source dataloader workers", int(source_config.dataloader_workers), 0)
        _expect(f"{source} training dataloader workers", int(config.training.dataloader_workers), 0)
        packing = image.get("packing", None)
        if packing is not None and bool(packing.get("enabled", False)):
            raise ValueError(f"{source} segment packing must be disabled")
        for key in ("truncate", "truncation", "allow_truncation", "truncate_text"):
            if bool(image.get(key, False)):
                raise ValueError(f"{source} image.{key} must be false")

        audit = protocol.full_training_length_audit[source]
        audited_max = int(audit.max)
        if audited_max > min(widths):
            raise ValueError(
                f"{source} audited full-corpus max={audited_max} exceeds "
                f"minimum scheduled width={min(widths)}"
            )
        expected_audit = (
            {"records": 7_687_002, "max": 446}
            if source == "i2t"
            else {"records": 15_374_004, "max": 372}
        )
        _expect(f"{source} audited record count", int(audit.records), expected_audit["records"])
        _expect(f"{source} audited max", audited_max, expected_audit["max"])
        if source == "i2t":
            _expect("I2T selected variants per image", int(audit.selected_variants_per_image), 6)
            _expect("I2T excludes original captions", bool(image.caption_include_original), False)
            _expect("I2T records fit 448", int(audit.greater_than_448), 0)
        else:
            _expect("T2I selected variants per image", int(audit.selected_variants_per_image), 12)
            _expect("T2I records fit 384", int(audit.greater_than_384), 0)

    _expect(f"{source} protocol pad schedule", widths, [int(width) for width in run.pad_to_length_schedule])
    per_rank_tokens = micro_batch * sum(widths)
    global_tokens = int(protocol.world_size) * per_rank_tokens
    logical_rows = int(protocol.world_size) * micro_batch * accumulation

    _expect(f"{source} protocol per-rank tokens", int(run.physical_tokens_per_rank_per_optimizer_step), per_rank_tokens)
    _expect(f"{source} protocol global tokens", int(run.physical_tokens_per_optimizer_step), global_tokens)
    _expect(f"{source} source global tokens", int(source_config.expected_global_physical_tokens_per_optimizer_step), global_tokens)
    _expect(f"{source} training global tokens", int(config.training.physical_tokens_per_optimizer_step), global_tokens)
    _expect(
        f"{source} combined-source contribution",
        global_tokens,
        int(protocol.combined_reference.source_contributions[source].physical_tokens_per_optimizer_step),
    )
    _expect(f"{source} total physical rows", int(config.training.total_batch_size), logical_rows)
    if source != "climbmix":
        _expect(f"{source} protocol logical samples", int(run.logical_samples_per_optimizer_step), logical_rows)

    return {
        "schedule": list(schedule),
        "pad_to_length_schedule": widths,
        "micro_batch_size_per_rank": micro_batch,
        "logical_rows_per_optimizer_step": logical_rows,
        "physical_tokens_per_rank_per_optimizer_step": per_rank_tokens,
        "physical_tokens_per_optimizer_step": global_tokens,
    }


def _validate_run_contract(
    *,
    config_path: Path,
    config: DictConfig,
    source: str,
    protocol: DictConfig,
    base: DictConfig,
) -> dict[str, Any]:
    run = protocol.single_source_runs[source]
    _expect(
        f"{source} protocol config path",
        _normalized_path(run.config),
        config_path.resolve(),
    )
    _validate_shared_model_optimizer(config, base, protocol, source=source)
    _validate_training_runtime_contract(config, protocol, run, source=source)
    batch = _validate_batch_token_contract(config, protocol, run, source=source)
    ema_eval_every = int(config.experiment.save_ema_eval_every)
    return {
        "source": source,
        "config": str(config_path),
        "job_name": str(run.job_name),
        "output_root": str(run.output_root),
        "max_train_steps": int(config.training.max_train_steps),
        "target_physical_tokens": int(config.training.target_physical_tokens),
        "warmup_steps": int(config.lr_scheduler.params.warmup_steps),
        "decay_steps": int(config.lr_scheduler.params.decay_steps),
        "batch_contract": batch,
        "ema_evaluation_export": {
            "enabled": ema_eval_every > 0,
            "every_optimizer_steps": ema_eval_every,
            "dtype": str(config.experiment.ema_eval_dtype).lower(),
            "artifacts": ["current_model", "ema_model", "pair_manifest"],
        },
    }


def _audit_model_assets(config: DictConfig) -> dict[str, Any]:
    model_path = _repo_path(config.model.model_path)
    _require_file(str(model_path / "config.json"), "Qwen config")
    _require_file(str(model_path / "model.safetensors"), "Qwen weights")
    _require_file(str(model_path / "tokenizer.json"), "Qwen tokenizer")
    source_model_config = AutoConfig.from_pretrained(
        model_path, local_files_only=True
    )
    _expect("Qwen source model type", source_model_config.model_type, "qwen3")
    return {"path": str(model_path), "model_type": source_model_config.model_type}


def _audit_climbmix_assets(config: DictConfig) -> dict[str, Any]:
    source = config.dataset.params.sources.climbmix
    shard_paths = sorted(glob.glob(str(_repo_path(source.shard_glob))))
    if len(shard_paths) != 100:
        raise ValueError(f"expected 100 ClimbMix shards, got {len(shard_paths)}")
    for shard_path in shard_paths:
        _require_file(shard_path, "ClimbMix shard")
    tokenizer_path = _repo_path(source.tokenizer_path)
    _require_file(str(tokenizer_path / "tokenizer.json"), "ClimbMix tokenizer")
    _expect("ClimbMix/Qwen tokenizer path", tokenizer_path.resolve(), _repo_path(config.model.model_path).resolve())
    return {"kind": "climbmix", "shards": len(shard_paths), "tokenizer_path": str(tokenizer_path)}


def _audit_imagenet_assets(config: DictConfig) -> dict[str, Any]:
    image = config.dataset.params.image
    _expect("ImageNet training split", image.expected_split, "train")
    _expect("ImageNet training records", int(image.expected_records), 1_281_167)
    if image.get("validation", None) is None:
        raise ValueError("dataset.params.image.validation is required")
    validation = image.validation
    _expect("ImageNet validation split", validation.expected_split, "val")
    _expect("ImageNet validation records", int(validation.expected_records), 50_000)

    cache_path = _require_file(str(_repo_path(image.cache_path)), "ImageNet train KL16 cache")
    train_manifest_path = _require_file(str(_repo_path(image.manifest_jsonl)), "ImageNet train manifest")
    _require_file(str(_repo_path(image.caption_jsonl)), "ImageNet train captions")
    train_index_path = _require_file(str(_repo_path(image.synthetic_text_index_manifest)), "ImageNet train synthetic-text seek index")
    val_cache_path = _require_file(str(_repo_path(validation.cache_path)), "ImageNet val KL16 cache")
    val_manifest_path = _require_file(str(_repo_path(validation.manifest_jsonl)), "ImageNet val manifest")
    _require_file(str(_repo_path(validation.caption_jsonl)), "ImageNet val captions")
    val_index_path = _require_file(str(_repo_path(validation.synthetic_text_index_manifest)), "ImageNet val synthetic-text seek index")

    train_cache = _load_image_cache(cache_path, expected_records=1_281_167)
    val_cache = _load_image_cache(val_cache_path, expected_records=50_000)
    _validate_text_index(train_index_path, split="train", records=1_281_167)
    val_text_index = _validate_text_index(val_index_path, split="val", records=50_000)
    if val_cache.get("metadata", {}).get("runtime_hashing_enabled") is not False:
        raise ValueError("ImageNet val cache must use the no-hash runtime contract")
    if val_text_index.get("runtime_hashing_enabled") is not False:
        raise ValueError("ImageNet val text index must use the no-hash runtime contract")

    val_manifest = _audit_imagenet_manifest(
        val_manifest_path,
        expected_split="val",
        expected_records=50_000,
        retain_identities=True,
    )
    train_manifest = _audit_imagenet_manifest(
        train_manifest_path,
        expected_split="train",
        expected_records=1_281_167,
        forbidden_identities=val_manifest["identities"],
    )
    return {
        "kind": "imagenet",
        "train": {
            "cache_shape": list(train_cache["posterior_stats"].shape),
            "manifest_records": int(train_manifest["records"]),
            "classes": int(train_manifest["classes"]),
        },
        "validation": {
            "cache_shape": list(val_cache["posterior_stats"].shape),
            "manifest_records": int(val_manifest["records"]),
            "classes": int(val_manifest["classes"]),
            "runtime_hashing_enabled": False,
        },
    }


def _build_probe_tokenizer(config: DictConfig):
    tokenizer = AutoTokenizer.from_pretrained(
        _repo_path(config.model.model_path),
        fix_mistral_regex=True,
        local_files_only=True,
    )
    special_tokens = ["<|mdm_mask|>", "<|boi|>", "<|eoi|>", "<|img_mask|>"]
    missing = [token for token in special_tokens if token not in tokenizer.get_vocab()]
    if missing:
        tokenizer.add_tokens(missing, special_tokens=True)
    if tokenizer.eos_token_id is None:
        raise ValueError("Qwen tokenizer has no eos_token_id")
    return tokenizer


def _tokenizer_probe(
    config: DictConfig,
    *,
    source: str,
    world_size: int,
) -> dict[str, Any]:
    tokenizer = _build_probe_tokenizer(config)
    if source == "climbmix":
        source_config = config.dataset.params.sources.climbmix
        shard_paths = sorted(glob.glob(str(_repo_path(source_config.shard_glob))))
        dataset = ClimbMixOnlineBatchDataset(
            shard_paths=shard_paths,
            tokenizer=tokenizer,
            eos_token_id=int(tokenizer.eos_token_id),
            sequence_length=int(source_config.sequence_length),
            micro_batch_size=int(source_config.micro_batch_size),
            rank=0,
            world_size=int(world_size),
            seed=int(config.training.seed),
            tokenizer_batch_documents=int(source_config.tokenizer_batch_documents),
            max_document_chars=int(source_config.max_document_chars),
            rayon_num_threads=int(source_config.rayon_num_threads),
        )
        batch = next(iter(dataset))
        return {
            "source": source,
            "shape": list(batch["input_ids"].shape),
            "valid_tokens": int(batch["pack_stats"][0]),
            "supervised_tokens": int(batch["supervised_text_tokens"]),
            "eos_token_id": int(tokenizer.eos_token_id),
        }

    image = config.dataset.params.image
    index = ImageNetSyntheticTextIndex(
        _repo_path(image.synthetic_text_index_manifest)
    )
    try:
        if source == "i2t":
            row = index.read_caption(0)
            candidates = [
                str(item.get(image.caption_list_text_key, "")).strip()
                for item in row[image.caption_list_key]
                if bool(image.caption_include_original)
                or str(item.get("source", "")).strip().lower() != "original"
            ]
            serialized_lengths = [
                len(tokenizer.encode(str(image.caption_i2t_prefix), add_special_tokens=False))
                + int(image.image_tokens_per_img)
                + 3
                + len(tokenizer.encode(text, add_special_tokens=False))
                for text in candidates
            ]
        else:
            row = index.read_t2i(0)
            candidates = [
                str(item.get("prompt", "")).strip()
                for item in row.get("model_result", {}).get("prompts", [])
            ]
            serialized_lengths = [
                len(
                    tokenizer.encode(
                        f"{image.caption_t2i_prefix} {text}",
                        add_special_tokens=False,
                    )
                )
                + int(image.image_tokens_per_img)
                + 3
                for text in candidates
            ]
    finally:
        index.close()
    if not candidates or any(not text for text in candidates):
        raise ValueError(f"{source} tokenizer probe found empty candidate text")
    expected_count = 6 if source == "i2t" else 12
    _expect(f"{source} tokenizer probe candidate count", len(candidates), expected_count)
    return {
        "source": source,
        "row_index": 0,
        "candidate_count": len(candidates),
        "serialized_lengths": serialized_lengths,
        "max_serialized_length": max(serialized_lengths),
        "eos_token_id": int(tokenizer.eos_token_id),
    }


def _check_npus(required_count: int, *, world_size: int) -> dict[str, Any] | None:
    required_count = int(required_count)
    if required_count <= 0:
        return None
    _expect("required NPU count/formal world size", required_count, world_size)
    import torch_npu  # noqa: F401

    available = bool(torch.npu.is_available())
    count = int(torch.npu.device_count())
    if not available or count != required_count:
        raise RuntimeError(
            f"expected {required_count} NPUs, got available={available}, count={count}"
        )
    return {"available": available, "count": count}


def validate_preflight(
    *,
    protocol_path: str | Path = DEFAULT_PROTOCOL,
    config_paths: Iterable[str | Path] | None = None,
    source: str | None = None,
    run_project: str | None = None,
    formal_world_size: int = EXPECTED_WORLD_SIZE,
    backbone_lr: float | None = None,
    flow_lr: float | None = None,
    save_ema_eval_every: int | None = None,
    require_npu_count: int = 0,
    tokenizer_probe: bool = False,
    audit_assets: bool = True,
) -> dict[str, Any]:
    """Validate the frozen experiment and return a JSON-serializable report.

    ``audit_assets=False`` exists for focused unit tests only; the CLI never
    exposes a switch that can bypass data/model audits.
    """

    protocol_path = _repo_path(protocol_path)
    protocol = OmegaConf.load(protocol_path)
    base = _validate_global_protocol(protocol)
    _expect(
        "formal world size",
        int(formal_world_size),
        int(protocol.world_size),
    )

    configured: dict[str, tuple[Path, DictConfig]] = {}
    for candidate_source in SUPPORTED_SOURCES:
        path = _repo_path(protocol.single_source_runs[candidate_source].config)
        if not path.is_file():
            raise FileNotFoundError(
                f"missing {candidate_source} single-source config: {path}"
            )
        configured[candidate_source] = (path, OmegaConf.load(path))

    # Structural validation always covers all three configs, so a shared
    # target or architecture drift cannot pass merely by selecting one job.
    run_reports = {
        candidate_source: _validate_run_contract(
            config_path=path,
            config=config,
            source=candidate_source,
            protocol=protocol,
            base=base,
        )
        for candidate_source, (path, config) in configured.items()
    }

    selected_sources: list[str]
    if config_paths:
        selected_paths = {_normalized_path(path) for path in config_paths}
        known_paths = {
            candidate_source: path.resolve()
            for candidate_source, (path, _) in configured.items()
        }
        unknown = selected_paths.difference(known_paths.values())
        if unknown:
            raise ValueError(
                "--config must name a config frozen in the protocol; unknown="
                f"{sorted(str(path) for path in unknown)}"
            )
        selected_sources = [
            candidate_source
            for candidate_source in SUPPORTED_SOURCES
            if known_paths[candidate_source] in selected_paths
        ]
    else:
        selected_sources = list(SUPPORTED_SOURCES)

    if source is not None:
        source = str(source).strip().lower()
        if source not in SUPPORTED_SOURCES:
            raise ValueError(f"unsupported --source={source!r}")
        if config_paths and selected_sources != [source]:
            raise ValueError(
                f"--source={source!r} does not match selected configs "
                f"{selected_sources}"
            )
        selected_sources = [source]

    if run_project is not None and not str(run_project).strip():
        raise ValueError("--run-project must be non-empty when provided")
    if save_ema_eval_every is not None:
        if len(selected_sources) != 1:
            raise ValueError(
                "--save-ema-eval-every requires exactly one selected source"
            )
        selected_source = selected_sources[0]
        _expect(
            f"{selected_source} runtime complete EMA export cadence",
            int(save_ema_eval_every),
            int(
                configured[selected_source][1]
                .experiment.save_ema_eval_every
            ),
        )

    requested_backbone_lr = float(
        protocol.shared_optimizer.backbone_and_special_learning_rate
        if backbone_lr is None
        else backbone_lr
    )
    requested_flow_lr = float(
        protocol.shared_optimizer.flow_and_projector_learning_rate
        if flow_lr is None
        else flow_lr
    )
    _expect(
        "requested backbone LR/protocol",
        requested_backbone_lr,
        float(protocol.shared_optimizer.backbone_and_special_learning_rate),
    )
    _expect(
        "requested flow LR/protocol",
        requested_flow_lr,
        float(protocol.shared_optimizer.flow_and_projector_learning_rate),
    )
    for selected_source in selected_sources:
        optimizer = configured[selected_source][1].optimizer.params
        _expect(
            f"{selected_source} requested backbone LR/config",
            requested_backbone_lr,
            float(optimizer.backbone_learning_rate),
        )
        _expect(
            f"{selected_source} requested flow LR/config",
            requested_flow_lr,
            float(optimizer.flow_learning_rate),
        )

    model_assets = None
    dataset_assets: dict[str, Any] = {}
    probes: dict[str, Any] = {}
    if audit_assets:
        model_assets = _audit_model_assets(configured[selected_sources[0]][1])
        imagenet_audit = None
        for selected_source in selected_sources:
            selected_config = configured[selected_source][1]
            if selected_source == "climbmix":
                dataset_assets[selected_source] = _audit_climbmix_assets(
                    selected_config
                )
            else:
                if imagenet_audit is None:
                    imagenet_audit = _audit_imagenet_assets(selected_config)
                dataset_assets[selected_source] = imagenet_audit
            if tokenizer_probe:
                probes[selected_source] = _tokenizer_probe(
                    selected_config,
                    source=selected_source,
                    world_size=int(protocol.world_size),
                )

    npu = _check_npus(
        require_npu_count, world_size=int(protocol.world_size)
    )
    return {
        "schema": "unified_single_source_preflight_v1",
        "protocol": str(protocol_path),
        "platform_project": str(protocol.platform_project),
        "run_project": run_project,
        "world_size": int(protocol.world_size),
        "nodes": int(protocol.nodes),
        "npu_per_node": int(protocol.npu_per_node),
        "selected_sources": selected_sources,
        "target_physical_tokens_per_run": int(
            protocol.target_physical_tokens_per_run
        ),
        "optimizer_overrides": {
            "backbone_and_special_lr": requested_backbone_lr,
            "flow_and_projector_lr": requested_flow_lr,
        },
        "runtime_hashing_enabled": False,
        "initialization": {
            "contract": str(protocol.initialization),
            "from_scratch": False,
            "resume_from_checkpoint": None,
            "model": model_assets,
        },
        "runs": run_reports,
        "dataset_assets": dataset_assets,
        "tokenizer_probes": probes,
        "npu": npu,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--config", action="append")
    parser.add_argument("--source", choices=SUPPORTED_SOURCES)
    parser.add_argument(
        "--formal-world-size", type=int, default=EXPECTED_WORLD_SIZE
    )
    parser.add_argument("--require-npu-count", type=int, default=0)
    parser.add_argument("--tokenizer-probe", action="store_true")
    parser.add_argument("--run-project")
    parser.add_argument("--backbone-lr", type=float)
    parser.add_argument("--flow-lr", type=float)
    parser.add_argument("--save-ema-eval-every", type=int)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = validate_preflight(
        protocol_path=args.protocol,
        config_paths=args.config,
        source=args.source,
        run_project=args.run_project,
        formal_world_size=args.formal_world_size,
        backbone_lr=args.backbone_lr,
        flow_lr=args.flow_lr,
        save_ema_eval_every=args.save_ema_eval_every,
        require_npu_count=args.require_npu_count,
        tokenizer_probe=args.tokenizer_probe,
        audit_assets=True,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
