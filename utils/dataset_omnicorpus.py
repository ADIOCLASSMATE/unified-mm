"""
OmniCorpus-only packed multimodal dataset.

Training consumes pre-tokenized Arrow shards with rows:
    {"input_ids": List[int], "token_types": List[int]}

Token types:
    0 = text, 1 = image, 2 = special (BOI/EOI/EOS), 3 = padding

The dataset builds pack indices over documents without materializing all token
lists into Python memory. Image spans are validated and never split.
"""

import glob
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from datasets import concatenate_datasets, load_from_disk
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm


class OmniCorpusPackedDataset(Dataset):
    def __init__(
        self,
        tokenized_path: str,
        boi_token_id: int,
        eoi_token_id: int,
        eos_token_id: int,
        max_seq_length: int = 2048,
        image_tokens_per_img: int = 256,
        image_vocab_size: int = 262144,
        shuffle: bool = True,
        seed: int = 42,
        max_samples: int = -1,
    ):
        self.tokenized_path = tokenized_path
        self.boi_id = boi_token_id
        self.eoi_id = eoi_token_id
        self.eos_id = eos_token_id
        self.max_seq_length = max_seq_length
        self.image_tokens_per_img = image_tokens_per_img
        self.image_vocab_size = image_vocab_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        self.dataset = self._load_arrow(tokenized_path)
        if max_samples > 0 and len(self.dataset) > max_samples:
            self.dataset = self.dataset.select(range(max_samples))

        self.lengths = [0] * len(self.dataset)
        for i, row in enumerate(tqdm(self.dataset, desc="Indexing OmniCorpus Arrow", unit="docs")):
            ids = row["input_ids"]
            types = row["token_types"]
            self._validate_image_spans(ids, types, f"row {i}")
            if len(ids) > max_seq_length:
                raise ValueError(
                    f"row {i} length {len(ids)} exceeds max_seq_length={max_seq_length}; "
                    "fix preprocessing so images are dropped whole and text is trimmed offline"
                )
            self.lengths[i] = len(ids)

        self._packs: List[List[int]] = []
        self._build_packs()

    @staticmethod
    def _load_arrow(path: str):
        root = Path(path)
        if (root / "dataset_info.json").exists():
            return load_from_disk(str(root), keep_in_memory=False)

        shard_paths = sorted(glob.glob(str(root / "shard-*")))
        if not shard_paths:
            raise FileNotFoundError(f"No Arrow dataset or shard-* directories found at {path}")
        datasets = [load_from_disk(p, keep_in_memory=False) for p in shard_paths]
        return concatenate_datasets(datasets)

    def _validate_image_spans(self, ids: Sequence[int], types: Sequence[int], context: str) -> None:
        if len(ids) != len(types):
            raise ValueError(f"{context}: input_ids/token_types length mismatch")
        i = 0
        while i < len(types):
            if types[i] != 1:
                i += 1
                continue
            start = i
            while i < len(types) and types[i] == 1:
                i += 1
            end = i
            if end - start != self.image_tokens_per_img:
                raise ValueError(f"{context}: image span length {end - start}")
            image_ids = ids[start:end]
            if min(image_ids) < 0 or max(image_ids) >= self.image_vocab_size:
                raise ValueError(
                    f"{context}: image token ids must be raw codebook ids in "
                    f"[0, {self.image_vocab_size}); got range [{min(image_ids)}, {max(image_ids)}]. "
                    "Rebuild Arrow shards with the current no-offset image token format."
                )
            if start == 0 or ids[start - 1] != self.boi_id:
                raise ValueError(f"{context}: image span missing BOI")
            if end >= len(ids) or ids[end] != self.eoi_id:
                raise ValueError(f"{context}: image span missing EOI")

    def _build_packs(self) -> None:
        indices = list(range(len(self.lengths)))
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(indices)

        packs: List[List[int]] = []
        current: List[int] = []
        current_len = 0
        for idx in indices:
            length = self.lengths[idx]
            if current and current_len + length > self.max_seq_length:
                packs.append(current)
                current = []
                current_len = 0
            current.append(idx)
            current_len += length
        if current:
            packs.append(current)
        self._packs = packs

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
        for row_idx in self._packs[idx]:
            row = self.dataset[int(row_idx)]
            ids.extend(row["input_ids"])
            types.extend(row["token_types"])
        self._validate_image_spans(ids, types, f"pack {idx}")
        sigma, labels = self._assign_sigma_and_labels(ids, types, self.seed + idx + self.epoch * 1_000_003)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "token_types": torch.tensor(types, dtype=torch.uint8),
            "sigma": sigma,
            "labels": labels,
        }

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        random.Random(self.seed + self.epoch).shuffle(self._packs)


def collate_omnicorpus(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    max_len = max(item["input_ids"].shape[0] for item in batch)
    bsz = len(batch)
    input_ids = torch.zeros(bsz, max_len, dtype=torch.long)
    token_types = torch.full((bsz, max_len), 3, dtype=torch.uint8)
    sigma = torch.full((bsz, max_len), max_len, dtype=torch.long)
    labels = torch.full((bsz, max_len), -100, dtype=torch.long)

    for i, item in enumerate(batch):
        length = item["input_ids"].shape[0]
        input_ids[i, :length] = item["input_ids"]
        token_types[i, :length] = item["token_types"]
        sigma[i, :length] = item["sigma"]
        labels[i, :length] = item["labels"]

    return {
        "input_ids": input_ids,
        "token_types": token_types,
        "sigma": sigma,
        "labels": labels,
    }


def build_omnicorpus_dataloaders(config, tokenizer):
    params = config.dataset.params
    dataset = OmniCorpusPackedDataset(
        tokenized_path=params.tokenized_path,
        boi_token_id=config.model.boi_token_id,
        eoi_token_id=config.model.eoi_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_seq_length=params.get("max_seq_length", 2048),
        image_tokens_per_img=params.get("image_tokens_per_img", 256),
        image_vocab_size=config.model.get("image_vocab_size", 262144),
        shuffle=True,
        seed=config.training.seed,
        max_samples=params.get("max_samples", -1),
    )

    val_ratio = params.get("val_ratio", 0.001)
    val_size = max(1, int(len(dataset) * val_ratio))
    train_size = max(0, len(dataset) - val_size)
    train_indices = list(range(train_size))
    val_indices = list(range(train_size, len(dataset)))

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.dataloader_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_omnicorpus,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.dataloader_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_omnicorpus,
    )
    return train_loader, val_loader
