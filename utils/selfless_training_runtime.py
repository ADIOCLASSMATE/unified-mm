"""Cold-path training contracts and low-overhead runtime accounting."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import torch
from omegaconf import OmegaConf

RESUME_CONTRACT_VERSION = 1
RESUME_SCHEMA = "selfless_caption_training_checkpoint_v3"
SAMPLER_RESUME_SCHEMA = "selfless_caption_sampler_resume_v1"


def build_resume_contract(
    config,
    *,
    world_size: int,
    gradient_accumulation_steps: int,
) -> dict[str, Any]:
    """Return every configuration field that affects exact continuation."""

    training = OmegaConf.to_container(config.training, resolve=True)
    # ``stop_after_steps`` is an operational stage boundary, not a numerical
    # training control.  Excluding it lets a checkpoint continue from Stage 1
    # to Stage 2 while every optimizer/scheduler/data/EMA control stays strict.
    training.pop("stop_after_steps", None)
    return {
        "contract_version": RESUME_CONTRACT_VERSION,
        "model": OmegaConf.to_container(config.model, resolve=True),
        "dataset": OmegaConf.to_container(config.dataset, resolve=True),
        "optimizer": OmegaConf.to_container(config.optimizer, resolve=True),
        "lr_scheduler": OmegaConf.to_container(
            config.lr_scheduler,
            resolve=True,
        ),
        # Preserve the complete training section so future numerical controls
        # remain strict without requiring an allow-list update.
        "training": training,
        "world_size": int(world_size),
        "gradient_accumulation_steps": int(gradient_accumulation_steps),
    }


def validate_resume_contract(
    metadata: dict[str, Any],
    *,
    current_contract: dict[str, Any],
    allow_s2_infra_migration: bool = False,
) -> list[dict[str, Any]]:
    """Validate an unhashed, human-readable continuation contract."""

    if metadata.get("schema") != RESUME_SCHEMA:
        raise RuntimeError(
            "Readable resume contracts require checkpoint schema "
            f"{RESUME_SCHEMA!r}, got {metadata.get('schema')!r}."
        )
    if int(metadata.get("config_contract_version", -1)) != (
        RESUME_CONTRACT_VERSION
    ):
        raise RuntimeError(
            "Unsupported readable resume contract version: "
            f"{metadata.get('config_contract_version')!r}"
        )
    if metadata.get("config_contract") != current_contract:
        if allow_s2_infra_migration:
            from utils.showo2_unified_protocol import validate_s2_infra_migration
            return validate_s2_infra_migration(metadata.get("config_contract"), current_contract)
        raise RuntimeError(
            "Resume configuration differs from the checkpoint; refusing an "
            "inexact continuation."
        )
    return []


def validate_wsd_contract(config) -> None:
    scheduler_name = str(config.lr_scheduler.get("scheduler", "wsd")).lower()
    if scheduler_name != "wsd":
        raise ValueError(
            f"Selfless-Flow supports only lr_scheduler.scheduler='wsd', got {scheduler_name!r}"
        )
    warmup = int(config.lr_scheduler.params.warmup_steps)
    decay = int(config.lr_scheduler.params.decay_steps)
    total = int(config.training.max_train_steps)
    min_lr_ratio = float(config.lr_scheduler.params.min_lr_scale)
    if total <= 0:
        raise ValueError(f"max_train_steps must be positive, got {total}")
    if warmup < 0 or decay < 0:
        raise ValueError(
            f"WSD warmup/decay steps must be non-negative, got {warmup}/{decay}"
        )
    if warmup + decay > total:
        raise ValueError(
            "WSD warmup_steps + decay_steps must not exceed max_train_steps: "
            f"{warmup} + {decay} > {total}"
        )
    if not 0.0 <= min_lr_ratio <= 1.0:
        raise ValueError(
            f"WSD min_lr_scale must be in [0, 1], got {min_lr_ratio}"
        )


def training_stop_step(config) -> int:
    """Resolve a resumable stage boundary without changing the WSD horizon."""

    total = int(config.training.max_train_steps)
    stop = int(config.training.get("stop_after_steps", total))
    if stop <= 0 or stop > total:
        raise ValueError(
            "training.stop_after_steps must be in (0, max_train_steps], "
            f"got {stop} with max_train_steps={total}"
        )
    return stop


def build_sampler_resume_state(
    *,
    epoch: int,
    batches_consumed_in_epoch: int,
    shuffle_seed: int,
    prepared_dataloader_length: int,
) -> dict[str, Any]:
    """Describe the deterministic sampler cursor used for exact continuation."""

    epoch = int(epoch)
    offset = int(batches_consumed_in_epoch)
    length = int(prepared_dataloader_length)
    if epoch < 0 or offset < 0 or length <= 0 or offset > length:
        raise ValueError(
            "invalid sampler resume cursor: "
            f"epoch={epoch}, offset={offset}, dataloader_length={length}"
        )
    return {
        "schema": SAMPLER_RESUME_SCHEMA,
        "epoch": epoch,
        "batches_consumed_in_epoch": offset,
        "shuffle_seed": int(shuffle_seed),
        "prepared_dataloader_length": length,
        "restore_method": "reseed_epoch_then_skip_prepared_batches",
    }


def validate_sampler_resume_state(
    state: dict[str, Any],
    *,
    epoch: int,
    batches_consumed_in_epoch: int,
    shuffle_seed: int,
    prepared_dataloader_length: int,
) -> None:
    expected = build_sampler_resume_state(
        epoch=epoch,
        batches_consumed_in_epoch=batches_consumed_in_epoch,
        shuffle_seed=shuffle_seed,
        prepared_dataloader_length=prepared_dataloader_length,
    )
    if state != expected:
        raise RuntimeError(
            "Sampler resume state differs from the deterministic data cursor; "
            "refusing an inexact continuation."
        )


def gradient_norm_log_payload(
    *,
    global_step: int,
    every: int,
    pre_clip_norm,
    max_norm: float,
) -> dict[str, float] | None:
    """Build a standalone grad-norm event independent of loss log cadence."""

    if int(every) <= 0:
        raise ValueError(f"gradient norm cadence must be positive, got {every}")
    if int(global_step) % int(every):
        return None
    value = float(torch.as_tensor(pre_clip_norm).detach().float().item())
    limit = float(max_norm)
    if not math.isfinite(value):
        raise FloatingPointError(
            f"non-finite pre-clip gradient norm at global_step={global_step}"
        )
    return {
        "train/global_grad_norm_pre_clip": value,
        "train/grad_clip_max_norm": limit,
        "train/grad_clip_applied": float(value > limit),
    }


@dataclass
class TrainingWindow:
    """CPU-only counters; conversion to one tiny tensor happens at log time."""

    started_at: float = 0.0
    optimizer_steps: int = 0
    micro_batches: int = 0
    logical_images: int = 0
    physical_rows: int = 0
    physical_tokens: int = 0
    valid_tokens: int = 0
    image_tokens: int = 0
    padding_tokens: int = 0
    data_wait_seconds: float = 0.0

    def __post_init__(self) -> None:
        if self.started_at == 0.0:
            self.started_at = time.perf_counter()

    def record_batch(
        self,
        *,
        rows: int,
        sequence_length: int,
        logical_images: int,
        pack_stats,
        data_wait_seconds: float,
    ) -> None:
        self.micro_batches += 1
        self.logical_images += int(logical_images)
        self.physical_rows += int(rows)
        self.physical_tokens += int(rows) * int(sequence_length)
        self.data_wait_seconds += float(data_wait_seconds)
        if pack_stats is not None:
            valid, image, padding, _ = map(int, pack_stats)
            self.valid_tokens += valid
            self.image_tokens += image
            self.padding_tokens += padding

    def record_optimizer_step(self) -> None:
        self.optimizer_steps += 1

    def exclude_elapsed(self, seconds: float) -> None:
        """Exclude checkpoint/validation time from the training-only window."""

        self.started_at += max(0.0, float(seconds))

    def as_tensor(self, device: torch.device) -> torch.Tensor:
        return torch.tensor(
            [
                self.optimizer_steps,
                self.micro_batches,
                self.logical_images,
                self.physical_rows,
                self.physical_tokens,
                self.valid_tokens,
                self.image_tokens,
                self.padding_tokens,
                self.data_wait_seconds,
                time.perf_counter() - self.started_at,
            ],
            device=device,
            # Ascend 910B does not support FP64 and would implicitly cast this
            # HCCL bookkeeping tensor.  Per-log-window counters remain exactly
            # representable at FP32 for the production logging cadence.
            dtype=torch.float32,
        )

    def reset(self) -> None:
        self.started_at = time.perf_counter()
        self.optimizer_steps = 0
        self.micro_batches = 0
        self.logical_images = 0
        self.physical_rows = 0
        self.physical_tokens = 0
        self.valid_tokens = 0
        self.image_tokens = 0
        self.padding_tokens = 0
        self.data_wait_seconds = 0.0
