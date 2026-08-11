"""Dataset splitting and DataLoader assembly for ImageNet flow training."""

from __future__ import annotations

import copy
import json
import random
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch.utils.data import DataLoader, RandomSampler, Subset

from utils.imagenet_flow_batching import collate_imagenet_flow_cache
from utils.multimodal_segment_packing import (
    collate_segment_packed,
    is_power_of_two,
)

if TYPE_CHECKING:
    from utils.dataset_imagenet_flow_cache import ImageNetFlowCacheDataset


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


def _split_key_for_index(
    dataset: ImageNetFlowCacheDataset, idx: int
) -> str:
    img_id = int(dataset.img_ids[int(idx)].item())
    return dataset.synsets.get(img_id, "")


def _build_split_indices(
    dataset: ImageNetFlowCacheDataset,
    val_ratio: float,
    seed: int,
    strategy: str,
    val_samples_per_class: int | None = None,
) -> tuple[list[int], list[int]]:
    n_items = len(dataset)
    if n_items <= 0:
        return [], []
    val_size = max(1, int(n_items * float(val_ratio)))
    val_size = min(val_size, max(1, n_items - 1))
    strategy = str(strategy or "stratified").lower()
    fixed_val_size = (
        int(val_samples_per_class)
        if val_samples_per_class is not None
        else None
    )
    if fixed_val_size is not None and fixed_val_size <= 0:
        raise ValueError(
            "val_samples_per_class must be positive, "
            f"got {val_samples_per_class}"
        )
    if fixed_val_size is not None and strategy not in {
        "stratified",
        "synset",
        "stratified_synset",
    }:
        raise ValueError(
            "val_samples_per_class requires a stratified split strategy, "
            f"got {strategy!r}."
        )

    if strategy in {"contiguous", "tail"}:
        train_size = max(0, n_items - val_size)
        return list(range(train_size)), list(range(train_size, n_items))

    rng = random.Random(int(seed))
    if strategy in {"shuffle", "shuffled", "random"}:
        indices = list(range(n_items))
        rng.shuffle(indices)
        return indices[val_size:], indices[:val_size]

    if strategy not in {"stratified", "synset", "stratified_synset"}:
        raise ValueError(
            f"Unknown ImageNet flow split_strategy={strategy!r}; "
            "expected stratified, shuffled, or contiguous."
        )

    groups: dict[str, list[int]] = {}
    for idx in range(n_items):
        key = _split_key_for_index(dataset, idx)
        groups.setdefault(key, []).append(idx)
    if len(groups) <= 1:
        indices = list(range(n_items))
        rng.shuffle(indices)
        single_group_val_size = (
            fixed_val_size if fixed_val_size is not None else val_size
        )
        single_group_val_size = min(
            single_group_val_size, max(1, n_items - 1)
        )
        return (
            indices[single_group_val_size:],
            indices[:single_group_val_size],
        )

    train_indices: list[int] = []
    val_indices: list[int] = []
    for key in sorted(groups):
        group_indices = list(groups[key])
        rng.shuffle(group_indices)
        if fixed_val_size is not None:
            group_val_size = fixed_val_size
        else:
            group_val_size = max(
                1, int(len(group_indices) * float(val_ratio))
            )
        if len(group_indices) > 1:
            group_val_size = min(
                group_val_size, len(group_indices) - 1
            )
        else:
            group_val_size = 0
        val_indices.extend(group_indices[:group_val_size])
        train_indices.extend(group_indices[group_val_size:])

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return train_indices, val_indices


def _load_explicit_split_indices(
    dataset: ImageNetFlowCacheDataset,
    split_manifest_jsonl: str,
) -> tuple[list[int], list[int]]:
    """Resolve a shared split manifest by image id."""

    path = Path(split_manifest_jsonl)
    if not path.exists():
        raise FileNotFoundError(path)
    index_by_image_id = {
        int(image_id.item()): index
        for index, image_id in enumerate(dataset.img_ids)
    }
    split_rows: dict[str, list[tuple[int, int]]] = {
        "train": [],
        "validation": [],
    }
    seen: set[int] = set()
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            image_id = int(row["img_id"])
            split = str(row["split"]).lower()
            if split == "val":
                split = "validation"
            if split not in split_rows:
                raise ValueError(
                    f"{path}:{line_number} has unsupported split={split!r}"
                )
            if image_id in seen:
                raise ValueError(f"duplicate img_id={image_id} in {path}")
            if image_id not in index_by_image_id:
                raise ValueError(
                    f"{path}:{line_number} img_id={image_id} is absent "
                    "from cache"
                )
            expected_synset = dataset.synsets.get(image_id, "")
            row_synset = str(row.get("synset", expected_synset))
            if row_synset != expected_synset:
                raise ValueError(
                    f"{path}:{line_number} synset mismatch for "
                    f"img_id={image_id}: {row_synset!r} != "
                    f"{expected_synset!r}"
                )
            split_index = int(
                row.get("split_index", len(split_rows[split]))
            )
            split_rows[split].append(
                (split_index, index_by_image_id[image_id])
            )
            seen.add(image_id)
    if seen != set(index_by_image_id):
        missing = sorted(set(index_by_image_id).difference(seen))
        raise ValueError(
            f"{path} does not cover the complete cache; missing "
            f"{len(missing)} image ids, first={missing[:8]}"
        )
    train_indices = [
        index
        for _, index in sorted(
            split_rows["train"], key=lambda item: item[0]
        )
    ]
    val_indices = [
        index
        for _, index in sorted(
            split_rows["validation"], key=lambda item: item[0]
        )
    ]
    if not train_indices or not val_indices:
        raise ValueError(
            f"{path} must contain non-empty train and validation assignments"
        )
    return train_indices, val_indices


def _build_dataset_subsets(
    dataset: ImageNetFlowCacheDataset,
    train_indices: list[int],
    val_indices: list[int],
    *,
    validation_overlap_train: bool,
) -> tuple[Subset, Subset]:
    """Build train/validation views without changing validation RNG semantics."""

    validation_dataset = dataset
    if validation_overlap_train:
        train_indices = list(range(len(dataset)))
        # Both views share mmap-backed posterior tensors and immutable metadata,
        # while keeping separate training masks. Validation therefore remains
        # deterministic even though its rows are also available to training.
        validation_dataset = copy.copy(dataset)
        validation_dataset.set_training_indices([])
    dataset.set_training_indices(train_indices)
    return (
        Subset(dataset, train_indices),
        Subset(validation_dataset, val_indices),
    )


def build_imagenet_flow_cache_dataloaders(config, tokenizer):
    # Imported lazily to keep the dataset module's compatibility re-exports
    # free of a module-import cycle.
    from utils.dataset_imagenet_flow_cache import ImageNetFlowCacheDataset

    params = config.dataset.params
    packing = params.get("packing", None)
    emit_pack_audit = bool(
        packing is not None and packing.get("audit_manifests", False)
    )
    dataset = ImageNetFlowCacheDataset(
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
        synset_mapping_path=params.get("synset_mapping_path", None),
        conditioning_mode=params.get("conditioning_mode", None),
        caption_jsonl=params.get("caption_jsonl", None),
        caption_text_key=params.get(
            "caption_text_key", "recaption_short"
        ),
        caption_list_key=params.get("caption_list_key", "captions"),
        caption_list_text_key=params.get(
            "caption_list_text_key", "text"
        ),
        caption_path_key=params.get("caption_path_key", "path"),
        caption_id_key=params.get("caption_id_key", "id"),
        caption_validation_index=params.get(
            "caption_validation_index", 0
        ),
        cache_caption_tokens=params.get("cache_caption_tokens", False),
        max_seq_length=params.get(
            "max_seq_length", config.dataset.preprocessing.max_seq_length
        ),
        model_context_length=params.get("model_context_length", None),
        caption_manifest_sha256=params.get(
            "caption_manifest_sha256", None
        ),
        max_samples=params.get("max_samples", -1),
        seed=config.training.seed,
        emit_audit_metadata=emit_pack_audit,
    )

    val_ratio = params.get("val_ratio", 0.001)
    val_samples_per_class = params.get("val_samples_per_class", None)
    split_seed = params.get("split_seed", config.training.seed)
    split_strategy = params.get("split_strategy", "stratified")
    split_manifest_jsonl = params.get("split_manifest_jsonl", None)
    if split_manifest_jsonl:
        train_indices, val_indices = _load_explicit_split_indices(
            dataset, str(split_manifest_jsonl)
        )
    else:
        train_indices, val_indices = _build_split_indices(
            dataset=dataset,
            val_ratio=float(val_ratio),
            seed=int(split_seed),
            strategy=str(split_strategy),
            val_samples_per_class=(
                int(val_samples_per_class)
                if val_samples_per_class is not None
                else None
            ),
        )
    train_dataset, val_dataset = _build_dataset_subsets(
        dataset,
        train_indices,
        val_indices,
        validation_overlap_train=bool(
            params.get("validation_overlap_train", False)
        ),
    )

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

    packing_enabled = bool(
        packing is not None and packing.get("enabled", False)
    )
    if packing_enabled:
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
