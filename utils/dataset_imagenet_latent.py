"""
Packed ImageNet prompt dataset for continuous VAE-latent flow training.

Rows are read from an OmniCorpus-compatible JSONL document file.
Each image segment becomes:
    <|boi|> [image latent placeholders] <|eoi|>

The discrete placeholder ids are only used to give the text tokenizer a valid
sequence. The model consumes the real image latents from the parallel
`image_latents` tensor at token_type == 1 positions.
"""

import json
import random
from bisect import bisect_left, insort
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm


class ImageNetLatentPackedDataset(Dataset):
    def __init__(
        self,
        docs_jsonl: str,
        latent_dir: str,
        tokenizer,
        boi_token_id: int,
        eoi_token_id: int,
        mask_token_id: int,
        eos_token_id: int,
        max_seq_length: int = 2048,
        image_tokens_per_img: int = 1024,
        image_latent_dim: int = 4,
        latent_key: str = "latent",
        pack_strategy: str = "best_fit_decreasing",
        shuffle: bool = True,
        seed: int = 42,
        max_samples: int = -1,
        log_pack_stats: bool = True,
    ):
        self.docs_jsonl = Path(docs_jsonl)
        self.latent_dir = Path(latent_dir)
        self.tokenizer = tokenizer
        self.boi_id = boi_token_id
        self.eoi_id = eoi_token_id
        self.mask_id = mask_token_id
        self.eos_id = eos_token_id
        self.max_seq_length = max_seq_length
        self.image_tokens_per_img = image_tokens_per_img
        self.image_latent_dim = image_latent_dim
        self.latent_key = latent_key
        self.pack_strategy = pack_strategy
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.log_pack_stats = log_pack_stats

        self.records: List[Dict[str, object]] = []
        self.lengths: List[int] = []
        self._load_records(max_samples=max_samples)
        self._packs: List[List[int]] = []
        self._pack_lengths: List[int] = []
        self._build_packs()

    def _load_records(self, max_samples: int) -> None:
        if not self.docs_jsonl.exists():
            raise FileNotFoundError(self.docs_jsonl)
        if not self.latent_dir.exists():
            raise FileNotFoundError(self.latent_dir)

        skipped = 0
        with self.docs_jsonl.open() as f:
            for line in tqdm(f, desc="Indexing ImageNet latent JSONL", unit="docs"):
                if max_samples > 0 and len(self.records) >= max_samples:
                    break
                if not line.strip():
                    continue
                row = json.loads(line)
                img_ids = [int(x) for x in row["img_ids"]]
                if any(not (self.latent_dir / f"{img_id:012d}.pt").exists() for img_id in img_ids):
                    skipped += 1
                    continue

                ids: List[int] = []
                types: List[int] = []
                image_ids_in_order: List[int] = []
                for seg in row["segments"]:
                    if seg["type"] == "text":
                        text_ids = self.tokenizer.encode(seg["content"], add_special_tokens=False)
                        ids.extend(text_ids)
                        types.extend([0] * len(text_ids))
                    elif seg["type"] == "image":
                        img_id = img_ids[int(seg["img_idx"])]
                        ids.extend([self.boi_id] + [self.mask_id] * self.image_tokens_per_img + [self.eoi_id])
                        types.extend([2] + [1] * self.image_tokens_per_img + [2])
                        image_ids_in_order.append(img_id)

                ids.append(self.eos_id)
                types.append(2)
                if len(ids) > self.max_seq_length:
                    skipped += 1
                    continue
                self.records.append(
                    {
                        "input_ids": ids,
                        "token_types": types,
                        "image_ids": image_ids_in_order,
                    }
                )
                self.lengths.append(len(ids))

        if not self.records:
            raise ValueError(
                f"No usable records found in {self.docs_jsonl}; skipped={skipped}. "
                "Check max_seq_length, image_tokens_per_img, and latent_dir."
            )
        if skipped:
            print(f"Skipped {skipped} ImageNet records while indexing latent dataset.")

    def _pack_greedy(self, indices: List[int]) -> Tuple[List[List[int]], List[int]]:
        packs: List[List[int]] = []
        pack_lengths: List[int] = []
        current: List[int] = []
        current_len = 0
        for idx in indices:
            length = self.lengths[idx]
            if current and current_len + length > self.max_seq_length:
                packs.append(current)
                pack_lengths.append(current_len)
                current = []
                current_len = 0
            current.append(idx)
            current_len += length
        if current:
            packs.append(current)
            pack_lengths.append(current_len)
        return packs, pack_lengths

    def _pack_best_fit(self, indices: List[int]) -> Tuple[List[List[int]], List[int]]:
        packs: List[List[int]] = []
        pack_lengths: List[int] = []
        bins_by_remaining: Dict[int, List[int]] = {}
        remaining_values: List[int] = []

        def add_open_bin(remaining: int, pack_idx: int) -> None:
            if remaining <= 0:
                return
            if remaining not in bins_by_remaining:
                bins_by_remaining[remaining] = []
                insort(remaining_values, remaining)
            bins_by_remaining[remaining].append(pack_idx)

        def pop_best_bin(length: int) -> Optional[Tuple[int, int]]:
            pos = bisect_left(remaining_values, length)
            if pos == len(remaining_values):
                return None
            remaining = remaining_values[pos]
            pack_idx = bins_by_remaining[remaining].pop()
            if not bins_by_remaining[remaining]:
                del bins_by_remaining[remaining]
                remaining_values.pop(pos)
            return pack_idx, remaining

        for idx in indices:
            length = self.lengths[idx]
            best = pop_best_bin(length)
            if best is None:
                pack_idx = len(packs)
                packs.append([idx])
                pack_lengths.append(length)
                add_open_bin(self.max_seq_length - length, pack_idx)
                continue

            pack_idx, remaining = best
            packs[pack_idx].append(idx)
            pack_lengths[pack_idx] += length
            add_open_bin(remaining - length, pack_idx)

        return packs, pack_lengths

    def _log_pack_stats(self) -> None:
        if not self.log_pack_stats or not self._pack_lengths:
            return
        total_tokens = sum(self._pack_lengths)
        capacity = len(self._pack_lengths) * self.max_seq_length
        fill = total_tokens / max(1, capacity)
        min_len = min(self._pack_lengths)
        max_len = max(self._pack_lengths)
        mean_len = total_tokens / len(self._pack_lengths)
        print(
            "ImageNet latent packing: "
            f"strategy={self.pack_strategy}, packs={len(self._pack_lengths)}, "
            f"fill={fill:.3%}, mean_len={mean_len:.1f}, "
            f"min_len={min_len}, max_len={max_len}, "
            f"waste_tokens={capacity - total_tokens}"
        )

    def _build_packs(self) -> None:
        indices = list(range(len(self.lengths)))
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(indices)

        strategy = self.pack_strategy.lower()
        if strategy in {"greedy", "next_fit"}:
            packs, pack_lengths = self._pack_greedy(indices)
        elif strategy in {"best_fit", "best_fit_decreasing", "bfd"}:
            if strategy in {"best_fit_decreasing", "bfd"}:
                indices.sort(key=lambda idx: self.lengths[idx], reverse=True)
            packs, pack_lengths = self._pack_best_fit(indices)
        else:
            raise ValueError(
                f"Unknown pack_strategy={self.pack_strategy!r}; "
                "expected greedy, best_fit, or best_fit_decreasing"
            )
        self._packs = packs
        self._pack_lengths = pack_lengths
        if self.epoch == 0:
            self._log_pack_stats()

    def _latent_to_tokens(self, img_id: int) -> torch.Tensor:
        path = self.latent_dir / f"{img_id:012d}.pt"
        obj = torch.load(path, map_location="cpu")
        latent = obj[self.latent_key]
        if latent.dim() != 3:
            raise ValueError(f"{path}: expected latent [C,H,W], got {tuple(latent.shape)}")
        c, h, w = latent.shape
        tokens = latent.permute(1, 2, 0).reshape(h * w, c).float()
        if tokens.shape != (self.image_tokens_per_img, self.image_latent_dim):
            raise ValueError(
                f"{path}: expected {(self.image_tokens_per_img, self.image_latent_dim)}, "
                f"got {tuple(tokens.shape)}"
            )
        return tokens

    def _assign_sigma_and_labels(
        self, ids: List[int], types: List[int], seed: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sigma = [self.max_seq_length] * len(ids)
        labels = list(ids)
        counter = 0
        img_positions: List[int] = []

        for i, token_type in enumerate(types):
            if token_type == 3:
                labels[i] = -100
                continue
            if token_type == 1:
                labels[i] = -100
                img_positions.append(i)
                continue

            sigma[i] = counter
            if token_type == 2 and ids[i] == self.eoi_id:
                labels[i] = -100
                if img_positions:
                    n = len(img_positions)
                    order = random.Random(seed + counter).sample(range(n), n)
                    for j, pos in enumerate(img_positions):
                        sigma[pos] = counter + 1 + order[j]
                    counter += n
                    img_positions = []
            counter += 1

        if img_positions:
            raise ValueError("orphan image positions without EOI")

        return torch.tensor(sigma, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self._packs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ids: List[int] = []
        types: List[int] = []
        latents: List[torch.Tensor] = []
        for row_idx in self._packs[idx]:
            record = self.records[int(row_idx)]
            ids.extend(record["input_ids"])
            types.extend(record["token_types"])
            for img_id in record["image_ids"]:
                latents.append(self._latent_to_tokens(int(img_id)))

        image_latents = torch.zeros(len(ids), self.image_latent_dim, dtype=torch.float32)
        cursor = 0
        image_idx = 0
        while cursor < len(types):
            if types[cursor] != 1:
                cursor += 1
                continue
            start = cursor
            while cursor < len(types) and types[cursor] == 1:
                cursor += 1
            end = cursor
            image_latents[start:end] = latents[image_idx]
            image_idx += 1

        sigma, labels = self._assign_sigma_and_labels(ids, types, self.seed + idx + self.epoch * 1_000_003)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "token_types": torch.tensor(types, dtype=torch.uint8),
            "sigma": sigma,
            "labels": labels,
            "image_latents": image_latents,
        }

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        # Keep pack identities stable because train/val are split by pack index.
        # DataLoader(shuffle=True) handles batch order; epoch only changes sigma sampling.


def collate_imagenet_latent(
    batch: List[Dict[str, torch.Tensor]],
    pad_to_length: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    batch_max_len = max(item["input_ids"].shape[0] for item in batch)
    max_len = pad_to_length or batch_max_len
    if batch_max_len > max_len:
        raise ValueError(f"Batch max length {batch_max_len} exceeds pad_to_length={max_len}")
    bsz = len(batch)
    latent_dim = batch[0]["image_latents"].shape[-1]
    input_ids = torch.zeros(bsz, max_len, dtype=torch.long)
    token_types = torch.full((bsz, max_len), 3, dtype=torch.uint8)
    sigma = torch.full((bsz, max_len), max_len, dtype=torch.long)
    labels = torch.full((bsz, max_len), -100, dtype=torch.long)
    image_latents = torch.zeros(bsz, max_len, latent_dim, dtype=torch.float32)

    for i, item in enumerate(batch):
        length = item["input_ids"].shape[0]
        input_ids[i, :length] = item["input_ids"]
        token_types[i, :length] = item["token_types"]
        sigma[i, :length] = item["sigma"]
        labels[i, :length] = item["labels"]
        image_latents[i, :length] = item["image_latents"]

    valid_tokens = (token_types != 3).sum()
    image_tokens = (token_types == 1).sum()
    padding_tokens = (token_types == 3).sum()
    return {
        "input_ids": input_ids,
        "token_types": token_types,
        "sigma": sigma,
        "labels": labels,
        "image_latents": image_latents,
        "pack_stats": torch.tensor(
            [valid_tokens, image_tokens, padding_tokens, max_len],
            dtype=torch.long,
        ),
    }


def build_imagenet_latent_dataloaders(config, tokenizer):
    params = config.dataset.params
    dataset = ImageNetLatentPackedDataset(
        docs_jsonl=params.docs_jsonl,
        latent_dir=params.latent_dir,
        tokenizer=tokenizer,
        boi_token_id=config.model.boi_token_id,
        eoi_token_id=config.model.eoi_token_id,
        mask_token_id=config.model.mask_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_seq_length=params.get("max_seq_length", 2048),
        image_tokens_per_img=params.get("image_tokens_per_img", config.model.image_tokens_per_img),
        image_latent_dim=params.get("image_latent_dim", config.model.image_latent_dim),
        latent_key=params.get("latent_key", "latent"),
        pack_strategy=params.get("pack_strategy", "best_fit_decreasing"),
        shuffle=True,
        seed=config.training.seed,
        max_samples=params.get("max_samples", -1),
        log_pack_stats=params.get("log_pack_stats", True),
    )

    val_ratio = params.get("val_ratio", 0.001)
    val_size = max(1, int(len(dataset) * val_ratio))
    train_size = max(0, len(dataset) - val_size)
    train_dataset = Subset(dataset, list(range(train_size)))
    val_dataset = Subset(dataset, list(range(train_size, len(dataset))))

    pad_to_length = params.get("pad_to_length", None)
    if params.get("pad_to_max_length", False):
        pad_to_length = params.get("max_seq_length", 2048)
    collate_fn = partial(collate_imagenet_latent, pad_to_length=pad_to_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.dataloader_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.dataloader_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
    )
    return train_loader, val_loader
