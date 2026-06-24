"""
Cache-backed full ImageNet latent dataset for dual-stream image-flow warmup.

The cache is the merged tensor produced by the VAE encoder:
    {"latents": [N, image_tokens, latent_dim], "img_ids": [N], ...}

Each item is represented as:
    optional text prompt, <|boi|>, image mask tokens, <|eoi|>, <eos>

The real image latent tokens live only in the parallel `image_latents` tensor.
The X0 stream receives those latent embeddings; visibility is controlled by the
selfless attention sigma/order, while the XT stream remains mask queries.
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional

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
        self.seed = int(seed)
        self.epoch = 0
        self.synsets = self._load_synsets(manifest_jsonl)
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
            self.fixed_labels = torch.full((self.fixed_length,), -100, dtype=torch.long)

    def _load_synsets(self, manifest_jsonl: Optional[str]) -> Dict[int, str]:
        if not manifest_jsonl or "{synset}" not in self.prompt_template:
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

    def __len__(self) -> int:
        return int(self.latents.shape[0])

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _prompt_ids(self, img_id: int) -> List[int]:
        if not self.prompt_template:
            return []
        synset = self.synsets.get(int(img_id), "")
        text = self.prompt_template.format(img_id=int(img_id), synset=synset)
        return self.tokenizer.encode(text, add_special_tokens=False)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.fixed_length is not None:
            return {
                "input_ids": self.fixed_input_ids,
                "token_types": self.fixed_token_types,
                "labels": self.fixed_labels,
                "image_latents": self.latents[idx],
            }

        img_id = int(self.img_ids[idx].item())
        prompt_ids = self._prompt_ids(img_id)
        image_start = len(prompt_ids) + 1

        ids = (
            prompt_ids
            + [self.boi_id]
            + [self.mask_id] * self.image_tokens_per_img
            + [self.eoi_id, self.eos_id]
        )
        types = (
            [0] * len(prompt_ids)
            + [2]
            + [1] * self.image_tokens_per_img
            + [2, 2]
        )

        length = len(ids)
        sigma = torch.empty(length, dtype=torch.long)
        labels = torch.full((length,), -100, dtype=torch.long)

        counter = 0
        for pos in range(len(prompt_ids)):
            sigma[pos] = counter
            counter += 1

        boi_pos = len(prompt_ids)
        sigma[boi_pos] = counter
        counter += 1

        eoi_pos = image_start + self.image_tokens_per_img
        sigma[eoi_pos] = counter
        rng = random.Random(self.seed + self.epoch * 1_000_003 + idx)
        order = rng.sample(range(self.image_tokens_per_img), self.image_tokens_per_img)
        for local_pos, order_value in enumerate(order):
            sigma[image_start + local_pos] = counter + 1 + order_value
        counter += self.image_tokens_per_img + 1

        sigma[eoi_pos + 1] = counter

        image_latents = torch.zeros(length, self.image_latent_dim, dtype=self.latents.dtype)
        image_latents[image_start : image_start + self.image_tokens_per_img] = self.latents[idx]

        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "token_types": torch.tensor(types, dtype=torch.uint8),
            "sigma": sigma,
            "labels": labels,
            "image_latents": image_latents,
        }


def collate_imagenet_flow_cache(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    first = batch[0]
    if "sigma" not in first and first["image_latents"].dim() == 2:
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

    max_len = max(item["input_ids"].shape[0] for item in batch)
    bsz = len(batch)
    latent_dim = batch[0]["image_latents"].shape[-1]
    latent_dtype = batch[0]["image_latents"].dtype

    input_ids = torch.zeros(bsz, max_len, dtype=torch.long)
    token_types = torch.full((bsz, max_len), 3, dtype=torch.uint8)
    sigma = torch.full((bsz, max_len), max_len, dtype=torch.long)
    labels = torch.full((bsz, max_len), -100, dtype=torch.long)
    image_latents = torch.zeros(bsz, max_len, latent_dim, dtype=latent_dtype)

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
        max_samples=params.get("max_samples", -1),
        seed=config.training.seed,
    )

    val_ratio = params.get("val_ratio", 0.001)
    val_size = max(1, int(len(dataset) * val_ratio))
    train_size = max(0, len(dataset) - val_size)
    train_dataset = Subset(dataset, list(range(train_size)))
    val_dataset = Subset(dataset, list(range(train_size, len(dataset))))

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.dataloader_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_imagenet_flow_cache,
        persistent_workers=config.training.dataloader_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.dataloader_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_imagenet_flow_cache,
        persistent_workers=config.training.dataloader_workers > 0,
    )
    return train_loader, val_loader
