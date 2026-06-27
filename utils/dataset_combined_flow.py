import bisect
import glob
import os
import random
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional

import torch
from datasets import load_from_disk
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset, Subset

from .dataset_imagenet_flow_cache import build_imagenet_flow_cache_dataloaders


class TextArrowDataset(Dataset):
    def __init__(
        self,
        tokenized_path: str,
        max_seq_length: int = 1024,
        pad_token_id: int = 0,
        sigma_mode: str = "ar",
        seed: int = 42,
        max_samples: int = -1,
        rows_per_shard: Optional[int] = None,
    ):
        self.tokenized_path = Path(tokenized_path)
        self.dataset = None
        self.shard_paths: List[Path] = []
        self.shard_lengths: List[int] = []
        self._loaded_shard_id: Optional[int] = None
        self._loaded_shard = None
        self.rows_per_shard = rows_per_shard
        self._init_storage()
        self.max_seq_length = int(max_seq_length)
        self.pad_token_id = int(pad_token_id)
        self.sigma_mode = str(sigma_mode)
        self.seed = int(seed)
        self.epoch = 0
        self.max_samples = int(max_samples)
        self._shard_order = list(range(len(self.shard_paths)))
        self._ordered_lengths: List[int] = []
        self._ordered_offsets: List[int] = []
        self._refresh_order()

        if self.max_seq_length <= 0:
            raise ValueError("TextArrowDataset max_seq_length must be positive.")
        if self.sigma_mode != "ar":
            raise ValueError(f"Unsupported text sigma_mode={self.sigma_mode!r}; use 'ar'.")

    def _init_storage(self) -> None:
        if (self.tokenized_path / "dataset_info.json").exists():
            self.dataset = load_from_disk(str(self.tokenized_path), keep_in_memory=False)
            return

        self.shard_paths = [Path(p) for p in sorted(glob.glob(str(self.tokenized_path / "shard-*")))]
        if not self.shard_paths:
            raise FileNotFoundError(f"No Arrow dataset or shard-* directories found at {self.tokenized_path}")
        if self.rows_per_shard is None:
            self.rows_per_shard = len(load_from_disk(str(self.shard_paths[0]), keep_in_memory=False))
        self.shard_lengths = [int(self.rows_per_shard)] * len(self.shard_paths)

    def _refresh_order(self) -> None:
        if self.dataset is not None:
            return
        self._shard_order = list(range(len(self.shard_paths)))
        random.Random(self.seed + self.epoch).shuffle(self._shard_order)

        remaining = self._total_shard_rows()
        if self.max_samples > 0:
            remaining = min(remaining, self.max_samples)
        offsets = [0]
        lengths = []
        for shard_id in self._shard_order:
            if remaining <= 0:
                break
            length = min(self.shard_lengths[shard_id], remaining)
            lengths.append(length)
            offsets.append(offsets[-1] + length)
            remaining -= length
        self._ordered_lengths = lengths
        self._ordered_offsets = offsets

    def _total_shard_rows(self) -> int:
        return sum(self.shard_lengths)

    def _locate_shard_row(self, idx: int) -> tuple[int, int]:
        pos = bisect.bisect_right(self._ordered_offsets, idx) - 1
        if pos < 0 or pos >= len(self._ordered_lengths):
            raise IndexError(idx)
        return self._shard_order[pos], idx - self._ordered_offsets[pos]

    def _load_shard(self, shard_id: int):
        if self._loaded_shard_id != shard_id:
            self._loaded_shard = load_from_disk(str(self.shard_paths[shard_id]), keep_in_memory=False)
            self._loaded_shard_id = shard_id
        return self._loaded_shard

    def __len__(self) -> int:
        if self.dataset is not None:
            length = len(self.dataset)
            return min(length, self.max_samples) if self.max_samples > 0 else length
        return self._ordered_offsets[-1]

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self._refresh_order()

    def _slice_ids(self, ids: List[int], idx: int) -> List[int]:
        if len(ids) <= self.max_seq_length:
            return ids
        max_offset = len(ids) - self.max_seq_length
        offset = random.Random(self.seed + self.epoch * 1_000_003 + idx).randint(0, max_offset)
        return ids[offset : offset + self.max_seq_length]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.dataset is not None:
            row = self.dataset[int(idx)]
        else:
            shard_id, row_idx = self._locate_shard_row(int(idx))
            shard = self._load_shard(shard_id)
            row = shard[int(row_idx) % len(shard)]
        ids = self._slice_ids(list(row["input_ids"]), int(idx))
        if not ids:
            ids = [self.pad_token_id]

        input_ids = torch.tensor(ids, dtype=torch.long)
        length = input_ids.numel()
        return {
            "input_ids": input_ids,
            "token_types": torch.zeros(length, dtype=torch.uint8),
            "sigma": torch.arange(length, dtype=torch.long),
            "labels": input_ids.clone(),
        }


def collate_text_arrow(
    batch: List[Dict[str, torch.Tensor]],
    pad_token_id: int = 0,
    pad_to_multiple_of: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    batch_max_len = max(item["input_ids"].shape[0] for item in batch)
    max_len = batch_max_len
    if pad_to_multiple_of and max_len % pad_to_multiple_of:
        max_len = ((max_len + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of

    bsz = len(batch)
    input_ids = torch.full((bsz, max_len), int(pad_token_id), dtype=torch.long)
    token_types = torch.full((bsz, max_len), 3, dtype=torch.uint8)
    sigma = torch.full((bsz, max_len), max_len, dtype=torch.long)
    labels = torch.full((bsz, max_len), -100, dtype=torch.long)

    for i, item in enumerate(batch):
        length = item["input_ids"].shape[0]
        input_ids[i, :length] = item["input_ids"]
        token_types[i, :length] = item["token_types"]
        sigma[i, :length] = item["sigma"]
        labels[i, :length] = item["labels"]

    valid_tokens = (token_types != 3).sum()
    padding_tokens = (token_types == 3).sum()
    return {
        "input_ids": input_ids,
        "token_types": token_types,
        "sigma": sigma,
        "labels": labels,
        "pack_stats": torch.tensor([valid_tokens, 0, padding_tokens, max_len], dtype=torch.long),
    }


class CombinedBatchDataLoader:
    def __init__(
        self,
        image_loader,
        text_loader,
        text_batch_ratio: float,
        seed: int = 42,
        mode: str = "train",
        max_text_batches: Optional[int] = None,
        batch_schedule: str = "random",
        accumulation_steps: Optional[int] = None,
        text_batches_per_accumulation: Optional[int] = None,
    ):
        self.image_loader = image_loader
        self.text_loader = text_loader
        self.text_batch_ratio = float(text_batch_ratio)
        self.seed = int(seed)
        self.mode = str(mode)
        self.epoch = 0
        self.max_text_batches = max_text_batches
        self.dataset = self
        self.batch_schedule = str(batch_schedule or "random").lower()
        if self.batch_schedule not in {"random", "accumulation"}:
            raise ValueError(f"Unsupported batch_schedule={batch_schedule!r}; use 'random' or 'accumulation'.")
        self.accumulation_steps = int(accumulation_steps or 0)
        if text_batches_per_accumulation is None:
            text_batches_per_accumulation = round(self.text_batch_ratio * self.accumulation_steps)
            if self.text_batch_ratio > 0.0 and text_batches_per_accumulation == 0:
                text_batches_per_accumulation = 1
        self.text_batches_per_accumulation = int(text_batches_per_accumulation)
        if self.batch_schedule == "accumulation":
            if self.accumulation_steps <= 0:
                raise ValueError("batch_schedule='accumulation' requires accumulation_steps > 0.")
            self.text_batches_per_accumulation = max(
                0,
                min(self.text_batches_per_accumulation, self.accumulation_steps),
            )

    def __len__(self) -> int:
        if self.mode == "train":
            return len(self.image_loader)
        text_batches = len(self.text_loader)
        if self.max_text_batches is not None:
            text_batches = min(text_batches, int(self.max_text_batches))
        return len(self.image_loader) + text_batches

    def _next_or_restart(self, iterator, loader):
        try:
            return next(iterator), iterator
        except StopIteration:
            iterator = iter(loader)
            return next(iterator), iterator

    def _use_text_batch(self, step: int, rng: random.Random) -> bool:
        if self.batch_schedule == "accumulation":
            if self.text_batches_per_accumulation <= 0:
                return False
            position = step % self.accumulation_steps
            return position >= self.accumulation_steps - self.text_batches_per_accumulation
        return rng.random() < self.text_batch_ratio

    def __iter__(self):
        image_iter = iter(self.image_loader)
        text_iter = iter(self.text_loader)
        rng = random.Random(self.seed + self.epoch * 1_000_003)
        total = len(self)

        for step in range(total):
            use_text = self._use_text_batch(step, rng)
            if self.mode != "train":
                use_text = step >= len(self.image_loader)
            if use_text:
                batch, text_iter = self._next_or_restart(text_iter, self.text_loader)
                batch["batch_source"] = "text"
            else:
                batch, image_iter = self._next_or_restart(image_iter, self.image_loader)
                batch["batch_source"] = "image"
            yield batch

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self._set_epoch_on(self.image_loader, self.epoch)
        self._set_epoch_on(self.text_loader, self.epoch)

    @staticmethod
    def _set_epoch_on(loader, epoch: int) -> None:
        dataset = getattr(loader, "dataset", None)
        while dataset is not None:
            if hasattr(dataset, "set_epoch"):
                dataset.set_epoch(epoch)
                break
            dataset = getattr(dataset, "dataset", None)

        sampler = getattr(loader, "sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        batch_sampler = getattr(loader, "batch_sampler", None)
        sampler = getattr(batch_sampler, "sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)

    def prepare_with_accelerator(self, accelerator):
        image_loader, text_loader = accelerator.prepare(self.image_loader, self.text_loader)
        prepared = CombinedBatchDataLoader(
            image_loader=image_loader,
            text_loader=text_loader,
            text_batch_ratio=self.text_batch_ratio,
            seed=self.seed,
            mode=self.mode,
            max_text_batches=self.max_text_batches,
            batch_schedule=self.batch_schedule,
            accumulation_steps=self.accumulation_steps,
            text_batches_per_accumulation=self.text_batches_per_accumulation,
        )
        prepared.epoch = self.epoch
        return prepared


def _get_text_params(config):
    params = config.dataset.params
    return params.text if "text" in params else params


def build_text_arrow_dataloaders(config, tokenizer):
    params = _get_text_params(config)
    dataset = TextArrowDataset(
        tokenized_path=params.tokenized_path,
        max_seq_length=params.get("max_seq_length", config.dataset.preprocessing.max_seq_length),
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        sigma_mode=params.get("sigma_mode", "ar"),
        seed=config.training.seed,
        max_samples=params.get("max_samples", -1),
        rows_per_shard=params.get("rows_per_shard", None),
    )

    val_ratio = params.get("val_ratio", 0.001)
    val_size = max(1, int(len(dataset) * val_ratio))
    max_val_samples = params.get("max_val_samples", -1)
    if max_val_samples and max_val_samples > 0:
        val_size = min(val_size, int(max_val_samples))
    train_size = max(0, len(dataset) - val_size)

    val_dataset = Subset(dataset, list(range(val_size)))
    train_dataset = Subset(dataset, list(range(val_size, val_size + train_size)))
    collate_fn = partial(
        collate_text_arrow,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        pad_to_multiple_of=params.get("pad_to_multiple_of", 64),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=params.get("batch_size", config.training.batch_size),
        shuffle=False,
        num_workers=params.get("dataloader_workers", config.training.dataloader_workers),
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
        persistent_workers=params.get("dataloader_workers", config.training.dataloader_workers) > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=params.get("batch_size", config.training.batch_size),
        shuffle=False,
        num_workers=params.get("dataloader_workers", config.training.dataloader_workers),
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
        persistent_workers=params.get("dataloader_workers", config.training.dataloader_workers) > 0,
    )
    return train_loader, val_loader


def _build_text_loaders(config, tokenizer):
    return build_text_arrow_dataloaders(config, tokenizer)


def _infer_accumulation_steps(config) -> int:
    world_size = max(1, int(os.environ.get("WORLD_SIZE", "1")))
    total_batch_size = int(config.training.total_batch_size)
    batch_size = int(config.training.batch_size)
    return max(1, (total_batch_size // batch_size) // world_size)


def build_combined_flow_dataloaders(config, tokenizer):
    params = config.dataset.params
    image_config = OmegaConf.create(OmegaConf.to_container(config, resolve=False))
    image_config.dataset.params = params.image
    image_train_loader, image_val_loader = build_imagenet_flow_cache_dataloaders(image_config, tokenizer)
    text_train_loader, text_val_loader = _build_text_loaders(config, tokenizer)

    text_batch_ratio = float(params.get("text_batch_ratio", 0.4))
    batch_schedule = str(params.get("batch_schedule", "random"))
    accumulation_steps = params.get("accumulation_steps", None)
    if accumulation_steps is None or str(accumulation_steps).lower() == "auto":
        accumulation_steps = _infer_accumulation_steps(config)
    text_batches_per_accumulation = params.get("text_batches_per_accumulation", None)
    if text_batches_per_accumulation is not None and str(text_batches_per_accumulation).lower() == "auto":
        text_batches_per_accumulation = None
    train_loader = CombinedBatchDataLoader(
        image_loader=image_train_loader,
        text_loader=text_train_loader,
        text_batch_ratio=text_batch_ratio,
        seed=config.training.seed,
        mode="train",
        batch_schedule=batch_schedule,
        accumulation_steps=accumulation_steps,
        text_batches_per_accumulation=text_batches_per_accumulation,
    )

    val_text_batches = int(params.get("val_text_batches", 20))
    val_loader = CombinedBatchDataLoader(
        image_loader=image_val_loader,
        text_loader=text_val_loader,
        text_batch_ratio=text_batch_ratio,
        seed=config.training.seed,
        mode="val",
        max_text_batches=val_text_batches,
        batch_schedule="random",
    )
    return train_loader, val_loader
