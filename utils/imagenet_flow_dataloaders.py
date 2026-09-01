"""Independent ImageNet-train/ImageNet-val DataLoader assembly."""

from __future__ import annotations

import copy
from functools import partial
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, RandomSampler, Subset

from utils.imagenet_flow_batching import collate_imagenet_flow_cache
from utils.multimodal_segment_packing import (
    collate_segment_packed,
    is_power_of_two,
)


class ScheduledPadCollator:
    """Cycle through fixed dense-attention widths in the main process.

    The cursor advances only after a batch was collated successfully.  Callers
    checkpoint only at a complete schedule boundary, where ``position == 0``.
    """

    def __init__(
        self,
        pad_to_length_schedule,
        *,
        pad_to_multiple_of: int | None = None,
    ) -> None:
        schedule = tuple(int(width) for width in pad_to_length_schedule)
        if not schedule or any(width <= 0 for width in schedule):
            raise ValueError(
                "pad_to_length_schedule must contain positive widths, got "
                f"{list(schedule)}"
            )
        multiple = (
            None
            if pad_to_multiple_of is None
            else int(pad_to_multiple_of)
        )
        if multiple is not None:
            if multiple <= 0:
                raise ValueError(
                    "pad_to_multiple_of must be positive when configured"
                )
            invalid = [
                width for width in schedule if width % multiple != 0
            ]
            if invalid:
                raise ValueError(
                    "every scheduled pad width must already be divisible by "
                    f"pad_to_multiple_of={multiple}; invalid={invalid}"
                )
        self.schedule = schedule
        self.pad_to_multiple_of = multiple
        self._position = 0

    @property
    def position(self) -> int:
        return int(self._position)

    def reset(self, position: int = 0) -> None:
        position = int(position)
        if not 0 <= position < len(self.schedule):
            raise ValueError(
                f"invalid scheduled-pad position={position} for "
                f"length={len(self.schedule)}"
            )
        self._position = position

    def __call__(self, batch):
        position = self.position
        width = self.schedule[position]
        result = collate_imagenet_flow_cache(
            batch,
            pad_to_length=width,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        result["pad_schedule_position"] = position
        result["scheduled_pad_to_length"] = width
        self._position = (position + 1) % len(self.schedule)
        return result


def training_samples_per_epoch(config, dataset_size: int) -> int | None:
    """Return the exact per-epoch sample budget for fixed global batches."""

    configured = config.training.get("samples_per_epoch", None)
    if configured is None:
        return None
    sample_budget = int(configured)
    global_batch = int(config.training.total_batch_size)
    if sample_budget <= 0:
        raise ValueError(
            f"training.samples_per_epoch must be positive, got {sample_budget}"
        )
    if sample_budget > int(dataset_size):
        raise ValueError(
            "training.samples_per_epoch cannot exceed the training split: "
            f"{sample_budget} > {dataset_size}"
        )
    if sample_budget % global_batch:
        raise ValueError(
            "training.samples_per_epoch must be divisible by the global batch "
            f"size: {sample_budget} % {global_batch} != 0"
        )
    return sample_budget


def _build_cache_dataset(config, params, tokenizer):
    from utils.dataset_imagenet_flow_cache import ImageNetFlowCacheDataset

    return ImageNetFlowCacheDataset(
        cache_path=params.cache_path,
        tokenizer=tokenizer,
        boi_token_id=config.model.boi_token_id,
        eoi_token_id=config.model.eoi_token_id,
        mask_token_id=config.model.mask_token_id,
        eos_token_id=tokenizer.eos_token_id,
        image_tokens_per_img=params.get(
            "image_tokens_per_img", config.model.image_tokens_per_img
        ),
        image_latent_dim=params.get(
            "image_latent_dim", config.model.image_latent_dim
        ),
        manifest_jsonl=params.get("manifest_jsonl", None),
        expected_split=params.get("expected_split", None),
        expected_records=params.get("expected_records", None),
        synset_mapping_path=params.get("synset_mapping_path", None),
        conditioning_mode=params.get("conditioning_mode", None),
        caption_jsonl=params.get("caption_jsonl", None),
        caption_text_key=params.get("caption_text_key", "recaption_short"),
        caption_list_key=params.get("caption_list_key", "captions"),
        caption_list_text_key=params.get("caption_list_text_key", "text"),
        caption_path_key=params.get("caption_path_key", "path"),
        caption_id_key=params.get("caption_id_key", "id"),
        caption_validation_index=params.get("caption_validation_index", 0),
        t2i_prompt_validation_index=params.get(
            "t2i_prompt_validation_index", 0
        ),
        caption_sequence_modes=params.get("caption_sequence_modes", None),
        synthetic_text_index_manifest=params.get(
            "synthetic_text_index_manifest", None
        ),
        caption_t2i_prefix=params.get(
            "caption_t2i_prefix",
            "Generate an image matching this description:",
        ),
        caption_i2t_prefix=params.get(
            "caption_i2t_prefix",
            "Describe this image in one detailed caption:",
        ),
        caption_include_original=params.get("caption_include_original", True),
        cache_caption_tokens=params.get("cache_caption_tokens", False),
        max_seq_length=params.get(
            "max_seq_length", config.dataset.preprocessing.max_seq_length
        ),
        model_context_length=params.get("model_context_length", None),
        max_samples=params.get("max_samples", -1),
        seed=config.training.seed,
        image_sigma_order=params.get("image_sigma_order", "random"),
    )


def _independent_validation_params(params):
    validation = params.get("validation", None)
    if validation is None:
        return None
    merged = copy.deepcopy(params)
    merged.pop("validation", None)
    for key, value in validation.items():
        merged[key] = copy.deepcopy(value)
    return merged


def _assert_independent_imagenet_splits(train_dataset, val_dataset) -> None:
    if train_dataset.dataset_split != "train":
        raise ValueError(
            "independent ImageNet training dataset must declare split='train'"
        )
    if val_dataset.dataset_split != "val":
        raise ValueError(
            "independent ImageNet validation dataset must declare split='val'"
        )
    train_identities = {
        (train_dataset.synsets[img_id], Path(path).name)
        for img_id, path in train_dataset.source_paths_full.items()
    }
    val_identities = {
        (val_dataset.synsets[img_id], Path(path).name)
        for img_id, path in val_dataset.source_paths_full.items()
    }
    overlap = train_identities.intersection(val_identities)
    if overlap:
        raise ValueError(
            "ImageNet train/val image identity overlap detected; first="
            f"{sorted(overlap)[:8]}"
        )


def build_imagenet_flow_cache_dataloaders(config, tokenizer):
    params = config.dataset.params
    packing = params.get("packing", None)
    emit_pack_audit = bool(
        packing is not None and packing.get("audit_manifests", False)
    )
    dataset = _build_cache_dataset(config, params, tokenizer)
    validation_params = _independent_validation_params(params)
    if validation_params is None:
        raise ValueError(
            "dataset.params.validation is required: training must use the "
            "complete ImageNet train cache and validation must use the "
            "independent ImageNet val cache"
        )
    validation_dataset = _build_cache_dataset(config, validation_params, tokenizer)
    _assert_independent_imagenet_splits(dataset, validation_dataset)
    train_indices = list(range(len(dataset)))
    val_indices = list(range(len(validation_dataset)))
    dataset.set_training_indices(train_indices)
    validation_dataset.set_training_indices([])
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(validation_dataset, val_indices)

    pad_to_length = params.get("pad_to_length", None)
    if params.get("pad_to_max_length", False):
        pad_to_length = params.get(
            "max_seq_length",
            config.dataset.preprocessing.max_seq_length,
        )
    unpacked_collate_fn = partial(
        collate_imagenet_flow_cache,
        pad_to_length=pad_to_length,
        pad_to_multiple_of=params.get("pad_to_multiple_of", None),
    )

    pad_to_length_schedule = params.get(
        "pad_to_length_schedule", None
    )
    packing_enabled = bool(
        packing is not None and packing.get("enabled", False)
    )
    if pad_to_length_schedule is not None and packing_enabled:
        raise ValueError(
            "pad_to_length_schedule and segment packing cannot be enabled "
            "together"
        )
    if pad_to_length_schedule is not None:
        train_collate_fn = ScheduledPadCollator(
            pad_to_length_schedule,
            pad_to_multiple_of=params.get("pad_to_multiple_of", None),
        )
    elif packing_enabled:
        algorithm = str(
            packing.get(
                "algorithm", "deterministic_best_fit_decreasing"
            )
        )
        if algorithm != "deterministic_best_fit_decreasing":
            raise ValueError(
                f"unsupported packing algorithm={algorithm!r}"
            )
        overflow_policy = str(
            packing.get(
                "overflow_policy", "dedicated_next_power_of_two"
            )
        )
        if overflow_policy != "dedicated_next_power_of_two":
            raise ValueError(
                f"unsupported overflow_policy={overflow_policy!r}"
            )
        nominal_capacity = int(packing.get("nominal_capacity", 2048))
        if not is_power_of_two(nominal_capacity):
            raise ValueError(
                "packing.nominal_capacity must be a positive power of two, "
                f"got {nominal_capacity}"
            )
        train_collate_fn = partial(
            collate_segment_packed,
            nominal_capacity=nominal_capacity,
            image_uncond_prob=float(
                config.model.get("image_uncond_prob", 0.0)
            ),
            emit_audit_manifest=emit_pack_audit,
        )
    else:
        train_collate_fn = unpacked_collate_fn

    # Validation/generation keeps one logical sample per physical row.
    val_collate_fn = unpacked_collate_fn
    train_generator = build_training_data_generator(config)
    epoch_sample_budget = training_samples_per_epoch(
        config,
        len(train_dataset),
    )
    train_sampler = (
        RandomSampler(
            train_dataset,
            replacement=False,
            num_samples=epoch_sample_budget,
            generator=train_generator,
        )
        if epoch_sample_budget is not None
        else None
    )
    worker_count = int(config.training.dataloader_workers)
    if pad_to_length_schedule is not None and worker_count != 0:
        raise ValueError(
            "pad_to_length_schedule uses a stateful collator and requires "
            "training.dataloader_workers=0"
        )
    worker_kwargs: dict[str, Any] = {}
    if worker_count > 0:
        worker_kwargs["prefetch_factor"] = int(
            config.training.get("dataloader_prefetch_factor", 4)
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=worker_count,
        pin_memory=True,
        drop_last=True,
        collate_fn=train_collate_fn,
        persistent_workers=worker_count > 0,
        generator=train_generator,
        **worker_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=worker_count,
        pin_memory=True,
        drop_last=False,
        collate_fn=val_collate_fn,
        persistent_workers=worker_count > 0,
        **worker_kwargs,
    )
    return train_loader, val_loader


def build_training_data_generator(config) -> torch.Generator | None:
    """Build a shuffle/worker RNG only when explicitly requested."""

    raw_seed = config.training.get("dataloader_shuffle_seed", None)
    if raw_seed is None:
        return None
    seed = int(raw_seed)
    if seed < 0:
        raise ValueError(
            "training.dataloader_shuffle_seed must be non-negative, "
            f"got {seed}"
        )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator
