"""Scheduled loader for ClimbMix + ImageNet T2I/I2T training sources.

The combined baseline remains exactly four microbatches in this order::

    ClimbMix text -> ImageNet T2I -> ClimbMix text -> ImageNet I2T

Single-source controls may instead repeat one source for the complete optimizer
update.  In that mode, inactive datasets and cursors are never constructed.

The ClimbMix worker may prefetch, but only the cursor attached to a batch that
the trainer consumed is committed.  Image streams keep independent epoch and
prepared-batch offsets.  Their rank-local state is saved alongside the regular
Accelerate checkpoint without computing hashes.
"""

from __future__ import annotations

import copy
import glob
import os
from functools import partial
from pathlib import Path
from typing import Any, Iterator

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset, Subset

from utils.climbmix_online_dataset import ClimbMixOnlineBatchDataset
from utils.distributed_io import run_io_phase
from utils.imagenet_flow_dataloaders import (
    ScheduledPadCollator,
    _build_cache_dataset,
    _independent_validation_params,
    build_imagenet_flow_cache_dataloaders,
)
from utils.imagenet_flow_batching import collate_imagenet_flow_cache


MIXED_DATA_STATE_SCHEMA = "unified_mixed_data_state_v1"
BASELINE_SOURCE_SCHEDULE = ("climbmix", "t2i", "climbmix", "i2t")
SUPPORTED_SOURCE_NAMES = frozenset(("climbmix", "t2i", "i2t"))


def _normalize_source_schedule(schedule) -> tuple[str, ...]:
    normalized = tuple(str(name).strip().lower() for name in schedule)
    if not normalized:
        raise ValueError("dataset.params.schedule must not be empty")
    unsupported = sorted(set(normalized).difference(SUPPORTED_SOURCE_NAMES))
    if unsupported:
        raise ValueError(f"unsupported training sources: {unsupported}")
    if normalized == BASELINE_SOURCE_SCHEDULE:
        return normalized
    if len(set(normalized)) == 1:
        return normalized
    raise ValueError(
        "dataset.params.schedule must be the frozen combined baseline or one "
        "source repeated for the complete optimizer update; got "
        f"{list(normalized)}"
    )


def _resolved_copy(config):
    return OmegaConf.create(OmegaConf.to_container(config, resolve=True))


def _source_config(config, source_name: str):
    sources = config.dataset.params.get("sources", None)
    if sources is None or source_name not in sources:
        raise ValueError(f"missing dataset.params.sources.{source_name}")
    return sources[source_name]


def _build_image_source(config, tokenizer, source_name: str):
    source = _source_config(config, source_name)
    image_config = _resolved_copy(config)
    image_config.dataset.class_name = "ImageNetFlowCacheDataset"
    image_config.dataset.params = copy.deepcopy(config.dataset.params.image)
    image_config.dataset.params.caption_sequence_modes = [source_name]
    image_config.dataset.params.pop("pad_to_length_schedule", None)
    pad_to_length_schedule = source.get(
        "pad_to_length_schedule", None
    )
    if pad_to_length_schedule is not None:
        image_config.dataset.params.pad_to_length_schedule = list(
            pad_to_length_schedule
        )
    image_config.dataset.preprocessing = copy.deepcopy(
        config.dataset.preprocessing
    )
    image_config.training.batch_size = int(source.micro_batch_size)
    image_config.training.total_batch_size = int(source.micro_batch_size)
    image_config.training.dataloader_workers = int(
        source.get("dataloader_workers", 0)
    )
    image_config.training.dataloader_prefetch_factor = int(
        source.get("dataloader_prefetch_factor", 4)
    )
    image_config.training.samples_per_epoch = None
    image_config.training.optimizer_steps_per_epoch = None
    image_config.training.num_train_epochs = None
    return build_imagenet_flow_cache_dataloaders(image_config, tokenizer)


def _clone_image_train_loader_for_i2t(config, t2i_loader: DataLoader):
    """Share mmap/metadata with T2I while keeping an independent I2T epoch."""

    if not isinstance(t2i_loader.dataset, Subset):
        raise TypeError("expected the ImageNet training dataset to be a Subset")
    t2i_dataset = t2i_loader.dataset.dataset
    i2t_dataset = copy.copy(t2i_dataset)
    i2t_dataset.caption_sequence_modes = ("i2t",)
    i2t_dataset._epoch_state = torch.zeros(
        (), dtype=torch.int64
    ).share_memory_()
    i2t_dataset.text_cache = {}
    i2t_dataset.sequence_cache = {}
    if t2i_dataset.synthetic_text_index is not None:
        i2t_dataset.synthetic_text_index = (
            t2i_dataset.synthetic_text_index.clone()
        )
    i2t_subset = Subset(i2t_dataset, t2i_loader.dataset.indices)

    source = _source_config(config, "i2t")
    if int(source.micro_batch_size) != int(t2i_loader.batch_size):
        raise ValueError(
            "The shared ImageNet loader requires identical T2I/I2T "
            "microbatch sizes."
        )
    generator = torch.Generator()
    generator.manual_seed(
        int(
            config.training.get(
                "dataloader_shuffle_seed", config.training.seed
            )
        )
    )
    worker_count = int(source.get("dataloader_workers", 0))
    worker_kwargs: dict[str, Any] = {}
    if worker_count > 0:
        worker_kwargs["prefetch_factor"] = int(
            source.get("dataloader_prefetch_factor", 4)
        )
    return DataLoader(
        i2t_subset,
        batch_size=int(source.micro_batch_size),
        shuffle=True,
        num_workers=worker_count,
        pin_memory=True,
        drop_last=True,
        collate_fn=t2i_loader.collate_fn,
        persistent_workers=worker_count > 0,
        generator=generator,
        **worker_kwargs,
    )


class PairedImageTaskValidationDataset(Dataset):
    """Emit T2I then I2T for every held-out ImageNet validation image."""

    task_modes = ("t2i", "i2t")

    def __init__(self, source_subset: Subset) -> None:
        if not isinstance(source_subset, Subset):
            raise TypeError(
                "expected the ImageNet validation dataset to be a Subset"
            )
        self.source_indices = list(source_subset.indices)
        source_dataset = source_subset.dataset
        self.datasets = {}
        for task_mode in self.task_modes:
            task_dataset = copy.copy(source_dataset)
            task_dataset.caption_sequence_modes = (task_mode,)
            task_dataset._epoch_state = torch.zeros(
                (), dtype=torch.int64
            ).share_memory_()
            task_dataset.text_cache = {}
            task_dataset.sequence_cache = {}
            if source_dataset.synthetic_text_index is not None:
                task_dataset.synthetic_text_index = (
                    source_dataset.synthetic_text_index.clone()
                )
            self.datasets[task_mode] = task_dataset

    def __len__(self) -> int:
        return len(self.source_indices) * len(self.task_modes)

    def __getitem__(self, index: int):
        index = int(index)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        image_offset, task_offset = divmod(index, len(self.task_modes))
        task_mode = self.task_modes[task_offset]
        return self.datasets[task_mode][self.source_indices[image_offset]]


def _clone_image_validation_loader_for_joint(
    t2i_loader: DataLoader,
) -> DataLoader:
    """Build an interleaved two-task view of official ImageNet validation."""

    joint_dataset = PairedImageTaskValidationDataset(t2i_loader.dataset)

    worker_count = int(t2i_loader.num_workers)
    worker_kwargs: dict[str, Any] = {}
    if worker_count > 0:
        worker_kwargs["prefetch_factor"] = int(
            t2i_loader.prefetch_factor
        )
    return DataLoader(
        joint_dataset,
        batch_size=int(t2i_loader.batch_size),
        shuffle=False,
        num_workers=worker_count,
        pin_memory=bool(t2i_loader.pin_memory),
        drop_last=False,
        collate_fn=t2i_loader.collate_fn,
        persistent_workers=worker_count > 0,
        **worker_kwargs,
    )


def build_unified_image_validation_dataloader(
    config,
    tokenizer,
    *,
    task_modes=("t2i", "i2t"),
    batch_size: int | None = None,
    num_workers: int | None = None,
) -> DataLoader:
    """Build only the independent ImageNet-val view used by UnifiedMixedDataset.

    Offline evaluation must not construct the 1.28M-row ImageNet-train view or
    the ClimbMix stream.  This helper keeps the exact training-time validation
    serialization while opening only the held-out validation cache and index.
    """

    if str(config.dataset.class_name) != "UnifiedMixedDataset":
        raise ValueError(
            "build_unified_image_validation_dataloader requires "
            "dataset.class_name='UnifiedMixedDataset'"
        )
    image_params = config.dataset.params.get("image", None)
    if image_params is None:
        raise ValueError("missing dataset.params.image")
    validation_params = _independent_validation_params(image_params)
    if validation_params is None:
        raise ValueError(
            "offline unified evaluation requires dataset.params.image.validation"
        )
    if str(validation_params.get("expected_split", "")).lower() != "val":
        raise ValueError(
            "offline unified evaluation requires expected_split='val'"
        )

    normalized_modes = tuple(str(mode).strip().lower() for mode in task_modes)
    if not normalized_modes or any(mode not in {"t2i", "i2t"} for mode in normalized_modes):
        raise ValueError(f"unsupported validation task modes: {normalized_modes}")
    if len(set(normalized_modes)) != len(normalized_modes):
        raise ValueError("validation task modes must not contain duplicates")

    validation_dataset = _build_cache_dataset(config, validation_params, tokenizer)
    validation_dataset.set_training_indices([])
    validation_dataset.caption_sequence_modes = normalized_modes
    source_subset = Subset(
        validation_dataset,
        list(range(len(validation_dataset))),
    )
    if normalized_modes == ("t2i", "i2t"):
        dataset = PairedImageTaskValidationDataset(source_subset)
    elif len(normalized_modes) == 1:
        dataset = source_subset
    else:
        raise ValueError(
            "multi-task validation order must be exactly ('t2i', 'i2t')"
        )

    if batch_size is None:
        batch_size = int(_source_config(config, normalized_modes[0]).micro_batch_size)
    if num_workers is None:
        num_workers = int(
            _source_config(config, normalized_modes[0]).get(
                "dataloader_workers", 0
            )
        )
    batch_size = int(batch_size)
    num_workers = int(num_workers)
    if batch_size <= 0 or num_workers < 0:
        raise ValueError(
            f"invalid validation loader settings: batch_size={batch_size}, "
            f"num_workers={num_workers}"
        )

    pad_to_length = validation_params.get("pad_to_length", None)
    if validation_params.get("pad_to_max_length", False):
        pad_to_length = validation_params.get(
            "max_seq_length", config.dataset.preprocessing.max_seq_length
        )
    collate_fn = partial(
        collate_imagenet_flow_cache,
        pad_to_length=pad_to_length,
        pad_to_multiple_of=validation_params.get("pad_to_multiple_of", None),
    )
    worker_kwargs: dict[str, Any] = {}
    if num_workers > 0:
        source = _source_config(config, normalized_modes[0])
        worker_kwargs["prefetch_factor"] = int(
            source.get("dataloader_prefetch_factor", 4)
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
        persistent_workers=num_workers > 0,
        **worker_kwargs,
    )


def _walk_dataloader_candidates(dataloader):
    seen: set[int] = set()
    pending = [dataloader]
    while pending:
        candidate = pending.pop()
        if candidate is None or id(candidate) in seen:
            continue
        seen.add(id(candidate))
        yield candidate
        for attribute in (
            "dataloader",
            "base_dataloader",
            "dataset",
            "sampler",
            "batch_sampler",
        ):
            nested = getattr(candidate, attribute, None)
            if nested is not None:
                pending.append(nested)


def _set_dataloader_epoch(dataloader, *, epoch: int, seed: int) -> None:
    epoch = int(epoch)
    seed = int(seed)
    for candidate in _walk_dataloader_candidates(dataloader):
        set_epoch = getattr(candidate, "set_epoch", None)
        if callable(set_epoch):
            set_epoch(epoch)
        generator = getattr(candidate, "generator", None)
        if isinstance(generator, torch.Generator):
            generator.manual_seed(seed + epoch)


def _scheduled_pad_collator(dataloader) -> ScheduledPadCollator | None:
    matches = []
    seen: set[int] = set()
    for candidate in _walk_dataloader_candidates(dataloader):
        collate_fn = getattr(candidate, "collate_fn", None)
        if (
            isinstance(collate_fn, ScheduledPadCollator)
            and id(collate_fn) not in seen
        ):
            matches.append(collate_fn)
            seen.add(id(collate_fn))
    if len(matches) > 1:
        raise RuntimeError(
            "prepared DataLoader exposes multiple scheduled-pad collators"
        )
    return matches[0] if matches else None


class ScheduledCombinedLoader:
    """Infinite fixed-schedule loader with rank-local exact data recovery."""

    is_mixed_sources = True
    state_schema = MIXED_DATA_STATE_SCHEMA

    def __init__(
        self,
        *,
        config,
        tokenizer,
        image_loaders: dict[str, DataLoader],
    ) -> None:
        self.config = config
        self.tokenizer = tokenizer
        self.image_loaders = dict(image_loaders)
        self.schedule = _normalize_source_schedule(
            config.dataset.params.schedule
        )
        self.active_sources = tuple(dict.fromkeys(self.schedule))
        active_image_sources = set(self.active_sources).intersection(
            {"t2i", "i2t"}
        )
        if set(self.image_loaders) != active_image_sources:
            raise ValueError(
                "image_loaders must contain exactly the active image sources: "
                f"runtime={sorted(self.image_loaders)}, "
                f"expected={sorted(active_image_sources)}"
            )
        self._pad_collators = {
            source_name: collator
            for source_name, loader in self.image_loaders.items()
            if (collator := _scheduled_pad_collator(loader)) is not None
        }
        configured_pad_sources = {
            source_name
            for source_name in active_image_sources
            if _source_config(config, source_name).get(
                "pad_to_length_schedule", None
            )
            is not None
        }
        if set(self._pad_collators) != configured_pad_sources:
            raise ValueError(
                "scheduled-pad collators must match configured active "
                f"sources: runtime={sorted(self._pad_collators)}, "
                f"configured={sorted(configured_pad_sources)}"
            )
        if self._pad_collators:
            if len(self.active_sources) != 1:
                raise ValueError(
                    "pad_to_length_schedule is supported only for a repeated "
                    "single-source schedule"
                )
            source_name = self.active_sources[0]
            collator = self._pad_collators.get(source_name)
            if collator is None:
                raise ValueError(
                    "active scheduled-pad source has no matching collator"
                )
            source = _source_config(config, source_name)
            configured_schedule = tuple(
                int(width)
                for width in source.get("pad_to_length_schedule", ())
            )
            if configured_schedule != collator.schedule:
                raise ValueError(
                    "source pad_to_length_schedule differs from its collator: "
                    f"configured={list(configured_schedule)}, "
                    f"runtime={list(collator.schedule)}"
                )
            if len(collator.schedule) != len(self.schedule):
                raise ValueError(
                    "pad_to_length_schedule length must equal the repeated "
                    "source schedule and gradient accumulation length: "
                    f"pad_widths={len(collator.schedule)}, "
                    f"source_slots={len(self.schedule)}"
                )
            if int(source.get("dataloader_workers", 0)) != 0:
                raise ValueError(
                    "pad_to_length_schedule requires source "
                    "dataloader_workers=0"
                )
        self.dataset = self
        self.num_workers = sum(
            int(loader.num_workers) for loader in self.image_loaders.values()
        )
        if "climbmix" in self.active_sources:
            self.num_workers += int(
                _source_config(config, "climbmix").dataloader_workers
            )
        self.persistent_workers = self.num_workers > 0
        self.prefetch_factor = None
        self._prepared = False
        self._rank = 0
        self._world_size = 1
        self._micro_step = 0
        self._image_state = {
            source_name: {"epoch": 0, "batches_consumed": 0}
            for source_name in self.active_sources
            if source_name in {"t2i", "i2t"}
        }
        self._image_iterators: dict[str, Iterator] = {}
        self._text_dataset = None
        self._text_loader = None
        self._text_iterator = None
        self._committed_text_state: dict[str, Any] | None = None
        self._reset_pad_schedule_after_resume_skip: set[str] = set()

    def __len__(self) -> int:
        return int(self.config.training.max_train_steps) * len(self.schedule)

    @property
    def schedule_position(self) -> int:
        return self._micro_step % len(self.schedule)

    def runtime_description(self) -> dict[str, Any]:
        description = {
            "schedule": list(self.schedule),
            "active_sources": list(self.active_sources),
            "image_workers_per_source": {
                name: int(loader.num_workers)
                for name, loader in self.image_loaders.items()
            },
            "climbmix_workers": 0,
            "climbmix_prefetch_factor": None,
        }
        if "climbmix" in self.active_sources:
            climbmix = _source_config(self.config, "climbmix")
            description["climbmix_workers"] = int(
                climbmix.dataloader_workers
            )
            description["climbmix_prefetch_factor"] = (
                int(climbmix.dataloader_prefetch_factor)
                if int(climbmix.dataloader_workers) > 0
                else None
            )
        return description

    @property
    def climbmix_validation_exclusion(self):
        return self._text_dataset.exclusion_contract if self._text_dataset is not None else None

    @property
    def climbmix_shard_paths(self):
        return self._text_dataset.shard_paths if self._text_dataset is not None else ()

    def prepare_with_accelerator(self, accelerator):
        if self._prepared:
            raise RuntimeError("ScheduledCombinedLoader was prepared twice")
        self._rank = int(accelerator.process_index)
        self._world_size = int(accelerator.num_processes)
        prepared_image_loaders = {}
        for source_name, loader in self.image_loaders.items():
            prepared_loader = accelerator.prepare_data_loader(loader)
            if source_name in self._pad_collators:
                # Accelerate's DataLoaderShard fetches one batch ahead before
                # every yield.  That is normally harmless, but it advances a
                # stateful width schedule past the optimizer boundary.  Its
                # base DataLoader already owns the rank-sharded batch sampler;
                # iterating that directly preserves sharding without hidden
                # collate calls.  The trainer moves every tensor to the device.
                base_loader = getattr(
                    prepared_loader, "base_dataloader", None
                )
                if base_loader is None:
                    raise RuntimeError(
                        "scheduled padding requires an Accelerate prepared "
                        "DataLoader with an accessible rank-sharded base loader"
                    )
                prepared_loader = base_loader
            prepared_image_loaders[source_name] = prepared_loader
        self.image_loaders = prepared_image_loaders
        for source_name in tuple(self._pad_collators):
            collator = _scheduled_pad_collator(
                self.image_loaders[source_name]
            )
            if collator is None:
                raise RuntimeError(
                    "prepared DataLoader lost its scheduled-pad collator for "
                    f"source={source_name!r}"
                )
            self._pad_collators[source_name] = collator

        if "climbmix" not in self.active_sources:
            self._prepared = True
            return self

        climbmix = _source_config(self.config, "climbmix")
        shard_paths = tuple(
            sorted(glob.glob(str(climbmix.shard_glob)))
        )
        if not shard_paths:
            raise FileNotFoundError(
                f"ClimbMix shard glob matched no files: {climbmix.shard_glob}"
            )
        self._text_dataset = ClimbMixOnlineBatchDataset(
            shard_paths=shard_paths,
            tokenizer_path=str(
                climbmix.get("tokenizer_path", self.config.model.model_path)
            ),
            eos_token_id=int(self.tokenizer.eos_token_id),
            sequence_length=int(climbmix.sequence_length),
            micro_batch_size=int(climbmix.micro_batch_size),
            rank=self._rank,
            world_size=self._world_size,
            seed=int(self.config.training.seed),
            tokenizer_batch_documents=int(
                climbmix.get("tokenizer_batch_documents", 32)
            ),
            max_document_chars=int(
                climbmix.get("max_document_chars", 262_144)
            ),
            rayon_num_threads=int(climbmix.get("rayon_num_threads", 2)),
            validation_exclusion_manifest=climbmix.get("validation_exclusion_manifest"),
        )
        worker_count = int(climbmix.dataloader_workers)
        loader_kwargs: dict[str, Any] = {}
        if worker_count > 0:
            loader_kwargs["prefetch_factor"] = int(
                climbmix.get("dataloader_prefetch_factor", 4)
            )
        self._text_loader = DataLoader(
            self._text_dataset,
            batch_size=None,
            num_workers=worker_count,
            pin_memory=True,
            persistent_workers=worker_count > 0,
            **loader_kwargs,
        )
        self._prepared = True
        return self

    def _new_image_iterator(self, source_name: str) -> Iterator:
        loader = self.image_loaders[source_name]
        state = self._image_state[source_name]
        epoch = int(state["epoch"])
        offset = int(state["batches_consumed"])
        seed = int(
            self.config.training.get(
                "dataloader_shuffle_seed", self.config.training.seed
            )
        )
        _set_dataloader_epoch(loader, epoch=epoch, seed=seed)
        iterator = iter(loader)
        skipped = 0
        while skipped < offset:
            try:
                next(iterator)
            except StopIteration as exc:
                raise RuntimeError(
                    f"invalid {source_name} resume cursor: epoch={epoch}, "
                    f"batches_consumed={offset}, loader ended at {skipped}"
                ) from exc
            skipped += 1
        if source_name in self._reset_pad_schedule_after_resume_skip:
            self._pad_collators[source_name].reset(0)
            self._reset_pad_schedule_after_resume_skip.remove(source_name)
        return iterator

    def _next_image_batch(self, source_name: str) -> dict[str, Any]:
        if source_name not in self._image_iterators:
            self._image_iterators[source_name] = self._new_image_iterator(
                source_name
            )
        iterator = self._image_iterators[source_name]
        try:
            batch = next(iterator)
        except StopIteration:
            state = self._image_state[source_name]
            state["epoch"] = int(state["epoch"]) + 1
            state["batches_consumed"] = 0
            iterator = self._new_image_iterator(source_name)
            self._image_iterators[source_name] = iterator
            batch = next(iterator)
        self._image_state[source_name]["batches_consumed"] = (
            int(self._image_state[source_name]["batches_consumed"]) + 1
        )
        if source_name in self._pad_collators:
            pad_position = int(batch.get("pad_schedule_position", -1))
            if pad_position != self.schedule_position:
                raise RuntimeError(
                    "source and pad schedules lost alignment: "
                    f"source={source_name!r}, "
                    f"source_position={self.schedule_position}, "
                    f"pad_position={pad_position}"
                )
        batch["source_name"] = source_name
        return batch

    def _next_text_batch(self) -> dict[str, Any]:
        if self._text_loader is None:
            raise RuntimeError("ScheduledCombinedLoader is not prepared")
        if self._text_iterator is None:
            self._text_iterator = iter(self._text_loader)
        batch = next(self._text_iterator)
        state = batch.pop("stream_state")
        self._committed_text_state = copy.deepcopy(state)
        return batch

    def __iter__(self):
        if not self._prepared:
            raise RuntimeError(
                "Call prepare_with_accelerator before iterating the mixed loader"
            )
        while True:
            source_name = self.schedule[self.schedule_position]
            if source_name == "climbmix":
                batch = self._next_text_batch()
            else:
                batch = self._next_image_batch(source_name)
            batch["source_schedule_position"] = self.schedule_position
            self._micro_step += 1
            yield batch

    def state_dict(self, *, global_step: int) -> dict[str, Any]:
        expected_micro_step = int(global_step) * len(self.schedule)
        if self._micro_step != expected_micro_step or self.schedule_position != 0:
            raise RuntimeError(
                "Mixed data checkpoint must be committed at an optimizer "
                f"boundary: micro_step={self._micro_step}, "
                f"expected={expected_micro_step}, "
                f"schedule_position={self.schedule_position}"
            )
        if (
            "climbmix" in self.active_sources
            and int(global_step) > 0
            and self._committed_text_state is None
        ):
            raise RuntimeError("no consumed ClimbMix cursor is available")
        pad_schedule_positions = {
            source_name: collator.position
            for source_name, collator in self._pad_collators.items()
        }
        non_boundary_positions = {
            source_name: position
            for source_name, position in pad_schedule_positions.items()
            if position != 0
        }
        if non_boundary_positions:
            raise RuntimeError(
                "scheduled padding must be checkpointed at a complete width "
                f"cycle boundary, got {non_boundary_positions}"
            )
        state = {
            "schema": MIXED_DATA_STATE_SCHEMA,
            "global_step": int(global_step),
            "micro_step": int(self._micro_step),
            "schedule": list(self.schedule),
            "schedule_position": int(self.schedule_position),
            "rank": int(self._rank),
            "world_size": int(self._world_size),
            "image_sources": copy.deepcopy(self._image_state),
        }
        if "climbmix" in self.active_sources:
            state["climbmix"] = copy.deepcopy(
                self._committed_text_state
            )
        if pad_schedule_positions:
            state["pad_schedule_positions"] = pad_schedule_positions
        return state

    def save_state(self, checkpoint_dir: str | Path, accelerator, global_step: int):
        checkpoint_dir = Path(checkpoint_dir)
        def write_state():
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            payload = self.state_dict(global_step=int(global_step))
            path = checkpoint_dir / f"data_state_rank_{self._rank:05d}.pt"
            temp_path = checkpoint_dir / f".{path.name}.tmp-{os.getpid()}"
            torch.save(payload, temp_path)
            os.replace(temp_path, path)

        run_io_phase(accelerator, write_state, description="mixed data state save")

    def load_state(self, checkpoint_dir: str | Path, accelerator, global_step: int):
        if self._text_iterator is not None or self._image_iterators:
            raise RuntimeError("mixed data state must be loaded before iteration")
        path = Path(checkpoint_dir) / f"data_state_rank_{self._rank:05d}.pt"
        if not path.is_file():
            raise FileNotFoundError(path)
        state = torch.load(path, map_location="cpu", weights_only=False)
        expected = {
            "schema": MIXED_DATA_STATE_SCHEMA,
            "global_step": int(global_step),
            "micro_step": int(global_step) * len(self.schedule),
            "schedule": list(self.schedule),
            "schedule_position": 0,
            "rank": self._rank,
            "world_size": self._world_size,
        }
        for key, value in expected.items():
            if state.get(key) != value:
                raise RuntimeError(
                    f"mixed data resume mismatch for {key}: "
                    f"checkpoint={state.get(key)!r}, expected={value!r}"
                )
        image_sources = state.get("image_sources")
        expected_image_sources = set(self._image_state)
        if (
            not isinstance(image_sources, dict)
            or set(image_sources) != expected_image_sources
        ):
            raise RuntimeError("invalid mixed image-source cursor state")
        for source_name, cursor in image_sources.items():
            epoch = int(cursor.get("epoch", -1))
            offset = int(cursor.get("batches_consumed", -1))
            if epoch < 0 or offset < 0:
                raise RuntimeError(
                    f"invalid {source_name} cursor: {cursor!r}"
                )
        if self._pad_collators:
            positions = state.get("pad_schedule_positions")
            expected_positions = {
                source_name: 0 for source_name in self._pad_collators
            }
            if positions != expected_positions:
                raise RuntimeError(
                    "mixed data resume mismatch for scheduled padding: "
                    f"checkpoint={positions!r}, "
                    f"expected={expected_positions!r}"
                )
            for collator in self._pad_collators.values():
                collator.reset(0)
            self._reset_pad_schedule_after_resume_skip = set(
                self._pad_collators
            )
        elif "pad_schedule_positions" in state:
            raise RuntimeError(
                "checkpoint has scheduled padding state but the current "
                "loader does not"
            )
        text_active = "climbmix" in self.active_sources
        climbmix_state = state.get("climbmix")
        if text_active:
            if int(global_step) > 0 and not isinstance(
                climbmix_state, dict
            ):
                raise RuntimeError(
                    "checkpoint has no committed ClimbMix cursor"
                )
        elif "climbmix" in state:
            raise RuntimeError(
                "inactive ClimbMix source must not have checkpoint state"
            )
        self._micro_step = int(state["micro_step"])
        self._image_state = copy.deepcopy(image_sources)
        self._committed_text_state = copy.deepcopy(climbmix_state)
        if text_active:
            if self._text_dataset is None:
                raise RuntimeError(
                    "ScheduledCombinedLoader is not prepared for ClimbMix"
                )
            self._text_dataset.set_resume_state(climbmix_state)
        accelerator.wait_for_everyone()


def build_unified_mixed_dataloaders(config, tokenizer):
    schedule = _normalize_source_schedule(config.dataset.params.schedule)
    active_sources = tuple(dict.fromkeys(schedule))
    image_loaders = {}
    validation_loader = None
    if schedule == BASELINE_SOURCE_SCHEDULE:
        t2i_train, t2i_validation = _build_image_source(
            config, tokenizer, "t2i"
        )
        image_loaders["t2i"] = t2i_train
        image_loaders["i2t"] = _clone_image_train_loader_for_i2t(
            config, t2i_train
        )
        validation_loader = _clone_image_validation_loader_for_joint(
            t2i_validation
        )
    else:
        source_name = active_sources[0]
        if source_name in {"t2i", "i2t"}:
            train_loader, validation_loader = _build_image_source(
                config, tokenizer, source_name
            )
            image_loaders[source_name] = train_loader
    train_loader = ScheduledCombinedLoader(
        config=config,
        tokenizer=tokenizer,
        image_loaders=image_loaders,
    )
    return train_loader, validation_loader


__all__ = [
    "BASELINE_SOURCE_SCHEDULE",
    "MIXED_DATA_STATE_SCHEMA",
    "SUPPORTED_SOURCE_NAMES",
    "PairedImageTaskValidationDataset",
    "ScheduledCombinedLoader",
    "build_unified_image_validation_dataloader",
    "build_unified_mixed_dataloaders",
]
