"""
Cache-backed full ImageNet latent dataset for unified image-flow training.

The cache is the merged tensor produced by the VAE encoder:
    {"latents": [N, image_tokens, latent_dim], "img_ids": [N], ...}

Each item is represented as:
    optional text prompt, <|boi|>, image mask tokens, <|eoi|>, <eos>

The real image latent tokens live only in the parallel `image_latents` tensor.
The X0 stream receives those latent embeddings; visibility is controlled by the
selfless attention sigma/order, while the XT stream remains mask queries.

Text, BOI, and EOS positions can be trained with cross-entropy labels. EOI is
kept as context for image tokens but is ignored by the CE loss.
"""

import json
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Subset


class ImageNetFlowCacheDataset(Dataset):
    def __init__(
        self,
        cache_path: str,
        tokenizer,
        boi_token_id: int,
        eoi_token_id: int,
        mask_token_id: int,
        eos_token_id: int,
        image_tokens_per_img: int = 256,
        image_latent_dim: int = 16,
        manifest_jsonl: Optional[str] = None,
        prompt_template: str = "",
        synset_mapping_path: Optional[str] = None,
        max_seq_length: Optional[int] = None,
        max_samples: int = -1,
        seed: int = 42,
    ):
        self.cache_path = Path(cache_path)
        if not self.cache_path.exists():
            raise FileNotFoundError(self.cache_path)

        obj = torch.load(self.cache_path, map_location="cpu")
        self.latents = obj["latents"]
        self.img_ids = obj.get("img_ids", torch.arange(self.latents.shape[0]))
        if max_samples is not None and max_samples > 0:
            self.latents = self.latents[:max_samples]
            self.img_ids = self.img_ids[:max_samples]

        if self.latents.dim() != 3:
            raise ValueError(f"Expected latents [N,T,C], got {tuple(self.latents.shape)}")
        if self.latents.shape[1:] != (image_tokens_per_img, image_latent_dim):
            raise ValueError(
                f"Expected latent shape (*,{image_tokens_per_img},{image_latent_dim}), "
                f"got {tuple(self.latents.shape)}"
            )

        self.tokenizer = tokenizer
        self.boi_id = int(boi_token_id)
        self.eoi_id = int(eoi_token_id)
        self.mask_id = int(mask_token_id)
        self.eos_id = int(eos_token_id)
        self.image_tokens_per_img = int(image_tokens_per_img)
        self.image_latent_dim = int(image_latent_dim)
        self.prompt_template = prompt_template or ""
        self.max_seq_length = int(max_seq_length) if max_seq_length else None
        self.seed = int(seed)
        self.epoch = 0
        self.synsets = self._load_synsets(manifest_jsonl)
        self.synset_names = self._load_synset_names(synset_mapping_path)
        self.prompt_cache = self._build_prompt_cache()
        self.sequence_cache = self._build_sequence_cache()
        self.fixed_length = 1 + self.image_tokens_per_img + 2 if not self.prompt_template else None
        if self.fixed_length is not None:
            self.fixed_input_ids = torch.tensor(
                [self.boi_id] + [self.mask_id] * self.image_tokens_per_img + [self.eoi_id, self.eos_id],
                dtype=torch.long,
            )
            self.fixed_token_types = torch.tensor(
                [2] + [1] * self.image_tokens_per_img + [2, 2],
                dtype=torch.uint8,
            )
            self.fixed_labels = torch.tensor(
                [self.boi_id]
                + [-100] * self.image_tokens_per_img
                + [-100, self.eos_id],
                dtype=torch.long,
            )

    def _load_synsets(self, manifest_jsonl: Optional[str]) -> Dict[int, str]:
        if not manifest_jsonl or not self.prompt_template:
            return {}
        path = Path(manifest_jsonl)
        if not path.exists():
            raise FileNotFoundError(path)
        synsets: Dict[int, str] = {}
        with path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                synsets[int(row["img_id"])] = str(row.get("synset", ""))
        return synsets

    def _load_synset_names(self, synset_mapping_path: Optional[str]) -> Dict[str, Tuple[str, str]]:
        if not synset_mapping_path:
            return {}
        path = Path(synset_mapping_path)
        if not path.exists():
            raise FileNotFoundError(path)
        names: Dict[str, Tuple[str, str]] = {}
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                synset, _, raw_names = line.partition(" ")
                class_names = raw_names.strip()
                class_name = class_names.split(",", 1)[0].strip() if class_names else synset
                names[synset] = (class_name, class_names or class_name)
        return names

    def _render_prompt(self, img_id: int, synset: str) -> str:
        class_name, class_names = self.synset_names.get(synset, (synset, synset))
        return self.prompt_template.format(
            img_id=int(img_id),
            synset=synset,
            class_name=class_name,
            class_names=class_names,
        )

    def _build_prompt_cache(self) -> Dict[str, torch.Tensor]:
        if not self.prompt_template:
            return {}

        prompt_cache: Dict[str, torch.Tensor] = {}
        if self.synsets:
            unique_synsets = sorted(set(self.synsets.values()))
            for synset in unique_synsets:
                text = self._render_prompt(0, synset)
                prompt_cache[text] = torch.tensor(
                    self.tokenizer.encode(text, add_special_tokens=False),
                    dtype=torch.long,
                )
        return prompt_cache

    def __len__(self) -> int:
        return int(self.latents.shape[0])

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _prompt_ids(self, img_id: int) -> torch.Tensor:
        if not self.prompt_template:
            return torch.empty(0, dtype=torch.long)
        synset = self.synsets.get(int(img_id), "")
        text = self._render_prompt(int(img_id), synset)
        cached = self.prompt_cache.get(text)
        if cached is None:
            cached = torch.tensor(self.tokenizer.encode(text, add_special_tokens=False), dtype=torch.long)
            self.prompt_cache[text] = cached
        return cached

    def _make_sequence_tensors(self, prompt_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        prompt_len = int(prompt_ids.numel())
        input_ids = torch.cat(
            [
                prompt_ids,
                torch.tensor(
                    [self.boi_id]
                    + [self.mask_id] * self.image_tokens_per_img
                    + [self.eoi_id, self.eos_id],
                    dtype=torch.long,
                ),
            ]
        )
        token_types = torch.cat(
            [
                torch.zeros(prompt_len, dtype=torch.uint8),
                torch.tensor([2] + [1] * self.image_tokens_per_img + [2, 2], dtype=torch.uint8),
            ]
        )
        labels = torch.cat(
            [
                prompt_ids,
                torch.tensor(
                    [self.boi_id]
                    + [-100] * self.image_tokens_per_img
                    + [-100, self.eos_id],
                    dtype=torch.long,
                ),
            ]
        )
        return {
            "input_ids": input_ids,
            "token_types": token_types,
            "labels": labels,
            "prompt_len": torch.tensor(prompt_len, dtype=torch.long),
            "image_start": torch.tensor(prompt_len + 1, dtype=torch.long),
        }

    def _build_sequence_cache(self) -> Dict[str, Dict[str, torch.Tensor]]:
        if not self.prompt_template:
            return {}
        sequence_cache: Dict[str, Dict[str, torch.Tensor]] = {}
        for text, prompt_ids in self.prompt_cache.items():
            sequence_cache[text] = self._make_sequence_tensors(prompt_ids)
        return sequence_cache

    def _sequence_tensors(self, img_id: int) -> Dict[str, torch.Tensor]:
        synset = self.synsets.get(int(img_id), "")
        text = self._render_prompt(int(img_id), synset)
        cached = self.sequence_cache.get(text)
        if cached is None:
            prompt_ids = self._prompt_ids(img_id)
            cached = self._make_sequence_tensors(prompt_ids)
            self.sequence_cache[text] = cached
        return cached

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.fixed_length is not None:
            return {
                "input_ids": self.fixed_input_ids,
                "token_types": self.fixed_token_types,
                "labels": self.fixed_labels,
                "image_latents": self.latents[idx],
                "prompt_len": torch.tensor(0, dtype=torch.long),
                "image_start": torch.tensor(1, dtype=torch.long),
            }

        img_id = int(self.img_ids[idx].item())
        sequence = self._sequence_tensors(img_id)
        length = int(sequence["input_ids"].numel())
        if self.max_seq_length is not None and length > self.max_seq_length:
            raise ValueError(
                f"ImageNetFlowCacheDataset item length {length} exceeds max_seq_length={self.max_seq_length}. "
                "Use a shorter prompt_template or increase dataset.params.max_seq_length."
            )

        return {
            "input_ids": sequence["input_ids"],
            "token_types": sequence["token_types"],
            "labels": sequence["labels"],
            "image_latents": self.latents[idx],
            "prompt_len": sequence["prompt_len"],
            "image_start": sequence["image_start"],
        }


def collate_imagenet_flow_cache(
    batch: List[Dict[str, torch.Tensor]],
    pad_to_length: Optional[int] = None,
    pad_to_multiple_of: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    first = batch[0]
    if first["image_start"].item() == 1 and first["input_ids"].shape[0] == first["image_latents"].shape[0] + 3:
        bsz = len(batch)
        image_tokens = first["image_latents"].shape[0]
        latent_dim = first["image_latents"].shape[-1]
        latent_dtype = first["image_latents"].dtype
        seq_len = image_tokens + 3

        input_ids = first["input_ids"].unsqueeze(0).expand(bsz, seq_len).clone()
        token_types = first["token_types"].unsqueeze(0).expand(bsz, seq_len).clone()
        labels = first["labels"].unsqueeze(0).expand(bsz, seq_len).clone()
        image_latents = torch.zeros(bsz, seq_len, latent_dim, dtype=latent_dtype)
        image_latents[:, 1 : 1 + image_tokens] = torch.stack(
            [item["image_latents"] for item in batch],
            dim=0,
        )

        sigma = torch.empty(bsz, seq_len, dtype=torch.long)
        sigma[:, 0] = 0
        sigma[:, 1 + image_tokens] = 1
        sigma[:, 2 + image_tokens] = image_tokens + 2
        order = torch.rand(bsz, image_tokens).argsort(dim=-1)
        sigma[:, 1 : 1 + image_tokens] = 2 + order

        return {
            "input_ids": input_ids,
            "token_types": token_types,
            "sigma": sigma,
            "labels": labels,
            "image_latents": image_latents,
            "pack_stats": torch.tensor(
                [bsz * seq_len, bsz * image_tokens, 0, seq_len],
                dtype=torch.long,
            ),
        }

    batch_max_len = max(item["input_ids"].shape[0] for item in batch)
    max_len = pad_to_length or batch_max_len
    if pad_to_multiple_of and max_len % pad_to_multiple_of:
        max_len = ((max_len + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of
    if batch_max_len > max_len:
        raise ValueError(f"Batch max length {batch_max_len} exceeds pad_to_length={max_len}")
    bsz = len(batch)
    latent_dim = batch[0]["image_latents"].shape[-1]
    latent_dtype = batch[0]["image_latents"].dtype
    image_tokens = batch[0]["image_latents"].shape[0]

    input_ids = torch.zeros(bsz, max_len, dtype=torch.long)
    token_types = torch.full((bsz, max_len), 3, dtype=torch.uint8)
    sigma = torch.full((bsz, max_len), max_len, dtype=torch.long)
    labels = torch.full((bsz, max_len), -100, dtype=torch.long)
    image_latents = torch.zeros(bsz, max_len, latent_dim, dtype=latent_dtype)
    order = torch.rand(bsz, image_tokens).argsort(dim=-1)

    for i, item in enumerate(batch):
        length = item["input_ids"].shape[0]
        prompt_len = int(item["prompt_len"].item())
        image_start = int(item["image_start"].item())
        eoi_pos = image_start + image_tokens
        eos_pos = eoi_pos + 1

        input_ids[i, :length] = item["input_ids"]
        token_types[i, :length] = item["token_types"]
        labels[i, :length] = item["labels"]
        image_latents[i, image_start:eoi_pos] = item["image_latents"]

        if prompt_len:
            sigma[i, :prompt_len] = torch.arange(prompt_len, dtype=torch.long)
        sigma[i, prompt_len] = prompt_len
        sigma[i, eoi_pos] = prompt_len + 1
        sigma[i, image_start:eoi_pos] = prompt_len + 2 + order[i]
        sigma[i, eos_pos] = prompt_len + image_tokens + 2

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


def build_imagenet_flow_cache_dataloaders(config, tokenizer):
    params = config.dataset.params
    dataset = ImageNetFlowCacheDataset(
        cache_path=params.cache_path,
        tokenizer=tokenizer,
        boi_token_id=config.model.boi_token_id,
        eoi_token_id=config.model.eoi_token_id,
        mask_token_id=config.model.mask_token_id,
        eos_token_id=tokenizer.eos_token_id,
        image_tokens_per_img=params.get("image_tokens_per_img", config.model.image_tokens_per_img),
        image_latent_dim=params.get("image_latent_dim", config.model.image_latent_dim),
        manifest_jsonl=params.get("manifest_jsonl", None),
        prompt_template=params.get("prompt_template", ""),
        synset_mapping_path=params.get("synset_mapping_path", None),
        max_seq_length=params.get("max_seq_length", config.dataset.preprocessing.max_seq_length),
        max_samples=params.get("max_samples", -1),
        seed=config.training.seed,
    )

    val_ratio = params.get("val_ratio", 0.001)
    val_size = max(1, int(len(dataset) * val_ratio))
    train_size = max(0, len(dataset) - val_size)
    train_dataset = Subset(dataset, list(range(train_size)))
    val_dataset = Subset(dataset, list(range(train_size, len(dataset))))

    pad_to_length = params.get("pad_to_length", None)
    if params.get("pad_to_max_length", False):
        pad_to_length = params.get("max_seq_length", config.dataset.preprocessing.max_seq_length)
    pad_to_multiple_of = params.get("pad_to_multiple_of", None)

    collate_fn = partial(
        collate_imagenet_flow_cache,
        pad_to_length=pad_to_length,
        pad_to_multiple_of=pad_to_multiple_of,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.dataloader_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
        persistent_workers=config.training.dataloader_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.dataloader_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
        persistent_workers=config.training.dataloader_workers > 0,
    )
    return train_loader, val_loader
