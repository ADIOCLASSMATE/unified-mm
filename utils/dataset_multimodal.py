"""
Multimodal dataset class for selfless attention training.

Replaces dataset_arrow.py for multimodal (text+image) training.
Reads JSONL files, tokenizes text on-the-fly, loads pre-computed image tokens.

Image tokens stored with offset: img_codebook_idx + TEXT_VOCAB_SIZE
to avoid collision with text token IDs.

Returns: (input_ids, token_types), task_mode per sample
Token types: 0=text, 1=image, 2=special(BOS/BOI/EOI), 3=padding
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset, DataLoader


class MultimodalDataset(Dataset):
    """Map-style dataset for multimodal selfless attention training.

    Reads one or more JSONL files containing Level 1+2+3 synthesized data.
    Tokenizes text on-the-fly using the project tokenizer.
    Loads pre-computed image tokens lazily from .pt files.
    """

    TEXT_VOCAB_SIZE = 151936  # Qwen3-0.6B

    def __init__(
        self,
        jsonl_paths: List[str],
        tokenizer,
        image_token_dir: str,
        boi_token_id: int,
        eoi_token_id: int,
        eos_token_id: int,
        max_seq_length: int = 2048,
        image_tokens_per_img: int = 512,
        shuffle: bool = True,
        seed: int = 42,
    ):
        self.jsonl_paths = jsonl_paths
        self.tokenizer = tokenizer
        self.image_token_dir = Path(image_token_dir)
        self.bos_id = tokenizer.bos_token_id
        self.boi_id = boi_token_id
        self.eoi_id = eoi_token_id
        self.eos_id = eos_token_id
        self.pad_id = getattr(tokenizer, "pad_token_id", None) or 0
        self.max_seq_length = max_seq_length
        self.image_tokens_per_img = image_tokens_per_img  # 512 for product_quant=2

        # Load all samples into memory (Phase 1 scale: ~1M samples, fine in RAM)
        self.samples: List[Dict] = []
        for path in jsonl_paths:
            p = Path(path)
            if not p.exists():
                print(f"Warning: JSONL file not found: {path}")
                continue
            with open(p) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.samples.append(json.loads(line))

        if shuffle:
            random.Random(seed).shuffle(self.samples)

        # Cached image tokens to avoid repeated disk reads
        self._image_cache: Dict[int, torch.Tensor] = {}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        task_mode = sample.get("task_mode", "text_only")

        if task_mode == "text_only":
            return self._build_text_only(sample)
        elif task_mode == "text_to_image":
            return self._build_text_to_image(sample)
        elif task_mode == "image_to_text":
            return self._build_image_to_text(sample)
        elif task_mode == "interleaved":
            return self._build_interleaved(sample)
        else:
            raise ValueError(f"Unknown task_mode: {task_mode}")

    # ─── Builders ───────────────────────────────────────────

    def _build_text_only(self, sample):
        text_ids = self.tokenizer.encode(sample["text"], add_special_tokens=False)
        max_text_len = self.max_seq_length - 2  # BOS + EOS

        if len(text_ids) > max_text_len:
            text_ids = text_ids[:max_text_len]

        ids = [self.bos_id] + text_ids + [self.eos_id]
        types = [2] + [0] * len(text_ids) + [2]

        return (torch.tensor(ids, dtype=torch.long),
                torch.tensor(types, dtype=torch.uint8)), "text_only"

    def _build_text_to_image(self, sample):
        text_ids = self.tokenizer.encode(sample["text"], add_special_tokens=False)
        img_tokens = self._load_image_tokens(sample["img_id"])
        img_tokens = img_tokens + self.TEXT_VOCAB_SIZE  # Offset

        max_text_len = self.max_seq_length - self.image_tokens_per_img - 3
        if len(text_ids) > max_text_len:
            text_ids = text_ids[:max_text_len]

        ids = [self.bos_id] + text_ids + [self.boi_id] + img_tokens.tolist() + [self.eoi_id]
        types = [2] + [0] * len(text_ids) + [2] + [1] * self.image_tokens_per_img + [2]

        return (torch.tensor(ids, dtype=torch.long),
                torch.tensor(types, dtype=torch.uint8)), "text_to_image"

    def _build_image_to_text(self, sample):
        text_ids = self.tokenizer.encode(sample["text"], add_special_tokens=False)
        img_tokens = self._load_image_tokens(sample["img_id"])
        img_tokens = img_tokens + self.TEXT_VOCAB_SIZE

        max_text_len = self.max_seq_length - self.image_tokens_per_img - 3
        if len(text_ids) > max_text_len:
            text_ids = text_ids[:max_text_len]

        ids = [self.bos_id] + [self.boi_id] + img_tokens.tolist() + [self.eoi_id] + text_ids
        types = [2] + [2] + [1] * self.image_tokens_per_img + [2] + [0] * len(text_ids)

        return (torch.tensor(ids, dtype=torch.long),
                torch.tensor(types, dtype=torch.uint8)), "image_to_text"

    def _build_interleaved(self, sample):
        ids = [self.bos_id]
        types = [2]

        for seg in sample.get("segments", []):
            if seg["type"] == "text":
                t = self.tokenizer.encode(seg["content"], add_special_tokens=False)
                ids.extend(t)
                types.extend([0] * len(t))
            elif seg["type"] == "image":
                img_idx = seg.get("img_idx", 0)
                img_id = sample["img_ids"][img_idx]
                img = self._load_image_tokens(img_id)
                img = img + self.TEXT_VOCAB_SIZE
                ids.extend([self.boi_id] + img.tolist() + [self.eoi_id])
                types.extend([2] + [1] * self.image_tokens_per_img + [2])

        # Truncate: preferentially trim text segments
        if len(ids) > self.max_seq_length:
            ids, types = self._smart_truncate(ids, types)

        return (torch.tensor(ids, dtype=torch.long),
                torch.tensor(types, dtype=torch.uint8)), "interleaved"

    # ─── Helpers ────────────────────────────────────────────

    def _load_image_tokens(self, img_id: int) -> torch.Tensor:
        if img_id not in self._image_cache:
            token_path = self.image_token_dir / f"{img_id:012d}.pt"
            if not token_path.exists():
                raise FileNotFoundError(
                    f"Image token file for img_id={img_id} not found at {token_path}. "
                    f"Run scripts/encode_coco_images.py first."
                )
            self._image_cache[img_id] = torch.load(token_path)
        return self._image_cache[img_id].clone()

    def _smart_truncate(self, ids, types):
        """Truncate to max_seq_length, preserving image tokens and special tokens."""
        # Strategy: find long text runs and trim them
        overflow = len(ids) - self.max_seq_length
        if overflow <= 0:
            return ids, types

        # Find all text segments and trim from longest
        text_spans = []
        i = 0
        while i < len(types):
            if types[i] == 0:  # text
                start = i
                while i < len(types) and types[i] == 0:
                    i += 1
                text_spans.append((start, i, i - start))
            else:
                i += 1

        # Sort by length descending, trim longest first
        text_spans.sort(key=lambda x: x[2], reverse=True)

        trimmed = 0
        for start, end, length in text_spans:
            to_trim = min(length - 10, overflow - trimmed)  # keep at least 10 tokens
            if to_trim > 0:
                trim_from = start  # trim from beginning of text span
                ids = ids[:trim_from] + ids[trim_from + to_trim:]
                types = types[:trim_from] + types[trim_from + to_trim:]
                trimmed += to_trim
                if trimmed >= overflow:
                    break

        return ids[:self.max_seq_length], types[:self.max_seq_length]

    def set_epoch(self, epoch: int):
        """Reshuffle samples at start of each epoch."""
        random.Random(42 + epoch).shuffle(self.samples)


def collate_multimodal(batch):
    """Collate function: pad sequences to batch max length.

    Args:
        batch: list of ((input_ids, token_types), task_mode) tuples

    Returns:
        dict with input_ids [B, max_len], token_types [B, max_len],
        task_mode list of strings
    """
    max_len = max(item[0][0].shape[0] for item in batch)
    B = len(batch)

    input_ids = torch.full((B, max_len), 0, dtype=torch.long)  # 0 = pad_token_id
    token_types = torch.full((B, max_len), 3, dtype=torch.uint8)  # 3 = padding
    task_modes = []

    for i, ((ids, types), mode) in enumerate(batch):
        L = ids.shape[0]
        input_ids[i, :L] = ids
        token_types[i, :L] = types
        task_modes.append(mode)

    return {
        "input_ids": input_ids,
        "token_types": token_types,
        "task_mode": task_modes,
    }


def build_multimodal_dataloaders(config, tokenizer):
    """Build train/val dataloaders from config."""
    params = config.dataset.params

    train_dataset = MultimodalDataset(
        jsonl_paths=params.jsonl_paths,
        tokenizer=tokenizer,
        image_token_dir=params.image_token_dir,
        boi_token_id=config.model.boi_token_id,
        eoi_token_id=config.model.eoi_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_seq_length=params.get("max_seq_length", 2048),
        image_tokens_per_img=params.get("image_tokens_per_img", 512),
        shuffle=True,
    )

    val_size = max(1, int(len(train_dataset) * 0.001))
    val_indices = list(range(len(train_dataset) - val_size, len(train_dataset)))
    train_indices = list(range(len(train_dataset) - val_size))

    from torch.utils.data import Subset
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(train_dataset, val_indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.dataloader_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_multimodal,
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.dataloader_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_multimodal,
    )

    return train_loader, val_loader
