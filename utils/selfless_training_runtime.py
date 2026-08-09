"""Cold-path training contracts and low-overhead runtime accounting."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

RESUME_SIGNATURE_VERSION = 3
LEGACY_RESUME_SCHEMA = "selfless_caption_training_checkpoint_v2"
RESUME_SCHEMA = "selfless_caption_training_checkpoint_v3"

_LEGACY_TRAINING_KEYS = (
    "batch_size",
    "total_batch_size",
    "mixed_precision",
    "gradient_accumulation_dtype",
    "seed",
    "dataloader_shuffle_seed",
    "use_ema",
    "ema_decay",
    "ema_update_after_step",
    "ema_shard_chunk_numel",
    "trainable_scope",
)


def _stable_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_resume_signature(
    config,
    *,
    world_size: int,
    gradient_accumulation_steps: int,
) -> str:
    """Hash every configuration field that can affect training continuation."""

    payload = {
        "signature_version": RESUME_SIGNATURE_VERSION,
        "model": OmegaConf.to_container(config.model, resolve=True),
        "dataset": OmegaConf.to_container(config.dataset, resolve=True),
        "optimizer": OmegaConf.to_container(config.optimizer, resolve=True),
        "lr_scheduler": OmegaConf.to_container(
            config.lr_scheduler,
            resolve=True,
        ),
        # Hash the complete training section so future numerical controls are
        # strict by default instead of requiring an allow-list update.
        "training": OmegaConf.to_container(config.training, resolve=True),
        "world_size": int(world_size),
        "gradient_accumulation_steps": int(gradient_accumulation_steps),
    }
    return _stable_hash(payload)


def build_legacy_resume_signature(
    config,
    *,
    world_size: int,
    gradient_accumulation_steps: int,
) -> str:
    """Reproduce the v2 signature for strict migration of old checkpoints."""

    training = OmegaConf.to_container(config.training, resolve=True)
    payload = {
        "model": OmegaConf.to_container(config.model, resolve=True),
        "dataset": OmegaConf.to_container(config.dataset, resolve=True),
        "optimizer": OmegaConf.to_container(config.optimizer, resolve=True),
        "lr_scheduler": OmegaConf.to_container(
            config.lr_scheduler,
            resolve=True,
        ),
        "training": {
            key: training.get(key) for key in _LEGACY_TRAINING_KEYS
        },
        "world_size": int(world_size),
        "gradient_accumulation_steps": int(gradient_accumulation_steps),
    }
    return _stable_hash(payload)


def validate_resume_metadata(
    metadata: dict[str, Any],
    *,
    checkpoint_dir: Path,
    config,
    world_size: int,
    gradient_accumulation_steps: int,
    current_signature: str,
) -> None:
    """Validate v3 checkpoints and safely migrate strict v2 checkpoints."""

    schema = metadata.get("schema")
    if schema == RESUME_SCHEMA:
        if int(metadata.get("config_signature_version", -1)) != (
            RESUME_SIGNATURE_VERSION
        ):
            raise RuntimeError(
                "Unsupported resume config signature version: "
                f"{metadata.get('config_signature_version')!r}"
            )
        if metadata.get("config_signature") != current_signature:
            raise RuntimeError(
                "Resume config signature differs from the checkpoint; "
                "refusing an inexact continuation."
            )
        return

    if schema != LEGACY_RESUME_SCHEMA:
        raise RuntimeError(
            f"Unsupported training checkpoint metadata schema: {schema!r}"
        )

    legacy_signature = build_legacy_resume_signature(
        config,
        world_size=world_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )
    if metadata.get("config_signature") != legacy_signature:
        raise RuntimeError(
            "Legacy resume config signature differs from the checkpoint."
        )

    # v2 omitted max_train_steps and other numerical training controls.  The
    # immutable run config is therefore required to prove a strict migration.
    saved_config_path = checkpoint_dir.parent / "config.yaml"
    if not saved_config_path.is_file():
        raise RuntimeError(
            "Cannot strictly resume a v2 checkpoint without its immutable "
            f"run config: {saved_config_path}"
        )
    saved_config = OmegaConf.load(saved_config_path)
    saved_signature = build_resume_signature(
        saved_config,
        world_size=world_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )
    if saved_signature != current_signature:
        raise RuntimeError(
            "Legacy checkpoint immutable config differs from the current "
            "training contract; refusing an inexact continuation."
        )


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
