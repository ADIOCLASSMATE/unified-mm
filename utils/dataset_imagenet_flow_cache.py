"""
Cache-backed ImageNet latent dataset for unified image-flow training.

The cache is the merged tensor produced by the VAE encoder:
    {"latents": [N, image_tokens, latent_dim], "img_ids": [N], ...}

This loader supports three explicit image-side conditioning modes:
    image_only:   <|boi|> image <|eoi|> <eos>
    class_image:  class_name <|boi|> image <|eoi|> <eos>
    caption_image:
        T2I: task_prefix caption <|boi|> image <|eoi|> <eos>
        I2T: task_prefix <|boi|> image <|eoi|> caption <eos>

Pure text batches are handled separately by TextArrowDataset.
"""

import json
import random
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Subset


DEFAULT_CAPTION_T2I_PREFIXES = [
    "Generate an image matching this caption:",
    "Create an image described by this caption:",
    "Draw a realistic image for this caption:",
    "Produce an image that follows this description:",
    "Render the scene described here:",
    "Create a visual scene from this caption:",
    "Generate a picture based on this description:",
    "Make an image that matches the following caption:",
    "Turn this caption into an image:",
    "Synthesize an image for this visual description:",
]

DEFAULT_CAPTION_I2T_PREFIXES = [
    "Describe this image in one detailed caption.",
    "Write a detailed caption for this image.",
    "Summarize the visual content of this image.",
    "Provide a natural-language caption for this image.",
    "Describe the main subject, setting, and visible details.",
    "Write one caption that explains what is visible in this image.",
    "Describe this picture clearly and naturally.",
    "Create a caption for the image shown here.",
    "State what this image shows in a detailed caption.",
    "Caption this image with the important visual details.",
]


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
        synset_mapping_path: Optional[str] = None,
        conditioning_mode: Optional[str] = None,
        caption_jsonl: Optional[str | Sequence[str]] = None,
        caption_text_key: str = "recaption_short",
        caption_path_key: str = "path",
        caption_id_key: str = "id",
        caption_sequence_modes: Optional[Sequence[str]] = None,
        caption_t2i_prefixes: Optional[Sequence[str]] = None,
        caption_i2t_prefixes: Optional[Sequence[str]] = None,
        caption_missing_policy: str = "error",
        caption_max_tokens: int = 192,
        cache_caption_tokens: bool = False,
        max_seq_length: Optional[int] = None,
        max_samples: int = -1,
        seed: int = 42,
        latent_hflip_prob: float = 0.0,
        label_text: bool = True,
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
        self.max_seq_length = int(max_seq_length) if max_seq_length else None
        self.seed = int(seed)
        self.epoch = 0
        self.label_text = bool(label_text)
        self.latent_hflip_prob = float(latent_hflip_prob)
        if not 0.0 <= self.latent_hflip_prob <= 1.0:
            raise ValueError(f"latent_hflip_prob must be in [0, 1], got {latent_hflip_prob}")
        self.latent_side = int(self.image_tokens_per_img ** 0.5)
        if self.latent_hflip_prob > 0.0 and self.latent_side * self.latent_side != self.image_tokens_per_img:
            raise ValueError(
                "latent_hflip_prob requires image_tokens_per_img to be a square grid, "
                f"got {self.image_tokens_per_img}."
            )
        self.augmentation_train_size: Optional[int] = None
        self.augmentation_index_mask: Optional[torch.Tensor] = None

        self.synsets, self.source_paths = self._load_manifest(manifest_jsonl)
        self.synset_names = self._load_synset_names(synset_mapping_path)
        self.caption_jsonl = caption_jsonl
        self.caption_text_key = str(caption_text_key)
        self.caption_path_key = str(caption_path_key)
        self.caption_id_key = str(caption_id_key)
        self.caption_missing_policy = str(caption_missing_policy or "error").lower()
        self.caption_max_tokens = int(caption_max_tokens or 0)
        self.cache_caption_tokens = bool(cache_caption_tokens)
        self.text_cache: Dict[str, torch.Tensor] = {}
        self.sequence_cache: Dict[str, Dict[str, torch.Tensor]] = {}

        if conditioning_mode is None:
            if caption_jsonl:
                conditioning_mode = "caption_image"
            elif manifest_jsonl and synset_mapping_path:
                conditioning_mode = "class_image"
            else:
                conditioning_mode = "image_only"
        self.conditioning_mode = str(conditioning_mode).lower()
        if self.conditioning_mode not in {"image_only", "class_image", "caption_image"}:
            raise ValueError(
                f"Unknown conditioning_mode={conditioning_mode!r}; "
                "expected image_only, class_image, or caption_image."
            )

        if self.conditioning_mode == "class_image":
            if not self.synsets:
                raise ValueError("conditioning_mode='class_image' requires manifest_jsonl with synset values.")
            if not self.synset_names:
                raise ValueError("conditioning_mode='class_image' requires synset_mapping_path.")

        self.caption_sequence_modes = self._normalize_caption_modes(caption_sequence_modes)
        self.caption_t2i_prefixes = self._normalize_prefixes(
            caption_t2i_prefixes,
            DEFAULT_CAPTION_T2I_PREFIXES,
            "caption_t2i_prefixes",
        )
        self.caption_i2t_prefixes = self._normalize_prefixes(
            caption_i2t_prefixes,
            DEFAULT_CAPTION_I2T_PREFIXES,
            "caption_i2t_prefixes",
        )
        self.captions: Dict[int, str] = {}
        if self.conditioning_mode == "caption_image":
            if not caption_jsonl:
                raise ValueError("conditioning_mode='caption_image' requires caption_jsonl.")
            self.captions = self._load_captions(caption_jsonl)
            if not self.captions:
                raise ValueError(f"No captions matched cache img_ids from {caption_jsonl}.")

        self.fixed_length = None
        if self.conditioning_mode == "image_only":
            self.fixed_length = 1 + self.image_tokens_per_img + 2
            if self.label_text:
                fixed_labels = [self.boi_id] + [-100] * self.image_tokens_per_img + [-100, self.eos_id]
            else:
                fixed_labels = [-100] * self.fixed_length
            self.fixed_input_ids = torch.tensor(
                [self.boi_id] + [self.mask_id] * self.image_tokens_per_img + [self.eoi_id, self.eos_id],
                dtype=torch.long,
            )
            self.fixed_token_types = torch.tensor(
                [2] + [1] * self.image_tokens_per_img + [2, 2],
                dtype=torch.uint8,
            )
            self.fixed_labels = torch.tensor(fixed_labels, dtype=torch.long)

    def _load_manifest(self, manifest_jsonl: Optional[str]) -> Tuple[Dict[int, str], Dict[int, str]]:
        if not manifest_jsonl:
            return {}, {}
        path = Path(manifest_jsonl)
        if not path.exists():
            raise FileNotFoundError(path)
        synsets: Dict[int, str] = {}
        source_paths: Dict[int, str] = {}
        with path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                img_id = int(row["img_id"])
                synsets[img_id] = str(row.get("synset", ""))
                source_path = row.get("source_path")
                if source_path:
                    source_paths[img_id] = self._relative_image_path(str(source_path))
        return synsets, source_paths

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

    def _normalize_caption_modes(self, modes: Optional[Sequence[str]]) -> List[str]:
        if modes is None:
            return ["t2i", "i2t"]
        normalized = [str(mode).lower() for mode in modes if str(mode).strip()]
        if not normalized:
            raise ValueError("caption_sequence_modes must not be empty.")
        invalid = [mode for mode in normalized if mode not in {"t2i", "i2t"}]
        if invalid:
            raise ValueError(f"Unsupported caption_sequence_modes={invalid}; expected t2i and/or i2t.")
        return normalized

    def _normalize_prefixes(
        self,
        prefixes: Optional[Sequence[str]],
        default_prefixes: Sequence[str],
        field_name: str,
    ) -> List[str]:
        source = default_prefixes if prefixes is None else prefixes
        normalized = [str(prefix).strip() for prefix in source if str(prefix).strip()]
        if not normalized:
            raise ValueError(f"{field_name} must not be empty.")
        return normalized

    def _caption_jsonl_paths(self, caption_jsonl: str | Sequence[str]) -> List[Path]:
        if isinstance(caption_jsonl, (str, Path)):
            return [Path(caption_jsonl)]
        return [Path(path) for path in caption_jsonl]

    def _relative_image_path(self, path: str) -> str:
        parts = Path(path).parts
        if len(parts) >= 2:
            return "/".join(parts[-2:])
        return path

    def _load_caption_rows(self, caption_jsonl: str | Sequence[str]) -> Tuple[Dict[str, str], Dict[str, str]]:
        captions_by_path: Dict[str, str] = {}
        captions_by_id: Dict[str, str] = {}
        for path in self._caption_jsonl_paths(caption_jsonl):
            if not path.exists():
                raise FileNotFoundError(path)
            with path.open() as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    caption = str(row.get(self.caption_text_key, "")).strip()
                    if not caption:
                        continue
                    row_path = row.get(self.caption_path_key)
                    if row_path:
                        captions_by_path[self._relative_image_path(str(row_path))] = caption
                    row_id = row.get(self.caption_id_key)
                    if row_id:
                        captions_by_id[str(row_id)] = caption
        return captions_by_path, captions_by_id

    def _load_captions(self, caption_jsonl: str | Sequence[str]) -> Dict[int, str]:
        captions_by_path, captions_by_id = self._load_caption_rows(caption_jsonl)
        captions: Dict[int, str] = {}
        missing: List[str] = []
        for img_id_tensor in self.img_ids:
            img_id = int(img_id_tensor.item())
            rel_path = self.source_paths.get(img_id)
            caption = captions_by_path.get(rel_path or "")
            if caption is None and rel_path:
                caption = captions_by_id.get(Path(rel_path).stem)
            if caption is None:
                if len(missing) < 5:
                    missing.append(rel_path or str(img_id))
                continue
            captions[img_id] = caption

        missing_count = int(self.img_ids.numel()) - len(captions)
        if missing_count and self.caption_missing_policy == "error":
            raise ValueError(
                f"Missing {missing_count} captions for cache img_ids. "
                f"Examples: {missing}. Use matching caption_jsonl or set caption_missing_policy='fallback_class'."
            )
        if missing_count and self.caption_missing_policy == "fallback_class":
            for img_id_tensor in self.img_ids:
                img_id = int(img_id_tensor.item())
                if img_id not in captions:
                    captions[img_id] = self._class_name_for_img_id(img_id)
        elif missing_count:
            raise ValueError(
                f"Unknown caption_missing_policy={self.caption_missing_policy!r}; "
                "expected error or fallback_class."
            )
        return captions

    def __len__(self) -> int:
        return int(self.latents.shape[0])

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def set_augmentation_train_size(self, train_size: int) -> None:
        self.augmentation_train_size = int(train_size)
        self.augmentation_index_mask = None

    def set_augmentation_indices(self, indices: Sequence[int]) -> None:
        mask = torch.zeros(len(self), dtype=torch.bool)
        if len(indices) > 0:
            mask[torch.as_tensor(list(indices), dtype=torch.long)] = True
        self.augmentation_train_size = None
        self.augmentation_index_mask = mask

    def _augment_latents(self, latents: torch.Tensor, idx: int) -> torch.Tensor:
        if self.latent_hflip_prob <= 0.0:
            return latents
        idx = int(idx)
        if self.augmentation_index_mask is not None:
            if idx < 0 or idx >= int(self.augmentation_index_mask.numel()) or not bool(self.augmentation_index_mask[idx]):
                return latents
        elif self.augmentation_train_size is not None and idx >= self.augmentation_train_size:
            return latents
        rng = random.Random(self.seed + self.epoch * 1_000_003 + int(idx) * 9_176 + 13_579)
        if rng.random() >= self.latent_hflip_prob:
            return latents
        return latents.view(self.latent_side, self.latent_side, self.image_latent_dim).flip(1).reshape_as(latents)

    def _text_ids(self, text: str, cache: bool = True) -> torch.Tensor:
        text = text.strip()
        if not text:
            return torch.empty(0, dtype=torch.long)
        if not cache:
            return torch.tensor(self.tokenizer.encode(text, add_special_tokens=False), dtype=torch.long)
        cached = self.text_cache.get(text)
        if cached is None:
            cached = torch.tensor(self.tokenizer.encode(text, add_special_tokens=False), dtype=torch.long)
            self.text_cache[text] = cached
        return cached

    def _fit_text_ids(self, prefix_ids: torch.Tensor, suffix_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.max_seq_length is None:
            return prefix_ids, suffix_ids

        available_text = self.max_seq_length - (self.image_tokens_per_img + 3)
        if available_text < 0:
            raise ValueError(
                f"max_seq_length={self.max_seq_length} is too short for "
                f"{self.image_tokens_per_img} image tokens plus BOI/EOI/EOS."
            )
        prefix_len = int(prefix_ids.numel())
        suffix_len = int(suffix_ids.numel())
        if prefix_len + suffix_len <= available_text:
            return prefix_ids, suffix_ids

        if prefix_len and not suffix_len:
            return prefix_ids[:available_text], suffix_ids
        if suffix_len and not prefix_len:
            return prefix_ids, suffix_ids[:available_text]

        keep_prefix = min(prefix_len, available_text)
        keep_suffix = max(0, available_text - keep_prefix)
        return prefix_ids[:keep_prefix], suffix_ids[:keep_suffix]

    def _make_sequence_tensors(self, prefix_ids: torch.Tensor, suffix_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        prefix_ids, suffix_ids = self._fit_text_ids(prefix_ids, suffix_ids)
        prefix_len = int(prefix_ids.numel())
        suffix_len = int(suffix_ids.numel())
        input_ids = torch.cat(
            [
                prefix_ids,
                torch.tensor(
                    [self.boi_id]
                    + [self.mask_id] * self.image_tokens_per_img
                    + [self.eoi_id],
                    dtype=torch.long,
                ),
                suffix_ids,
                torch.tensor([self.eos_id], dtype=torch.long),
            ]
        )
        token_types = torch.cat(
            [
                torch.zeros(prefix_len, dtype=torch.uint8),
                torch.tensor([2] + [1] * self.image_tokens_per_img + [2], dtype=torch.uint8),
                torch.zeros(suffix_len, dtype=torch.uint8),
                torch.tensor([2], dtype=torch.uint8),
            ]
        )
        if self.label_text:
            labels = torch.cat(
                [
                    prefix_ids,
                    torch.tensor(
                        [self.boi_id]
                        + [-100] * self.image_tokens_per_img
                        + [-100],
                        dtype=torch.long,
                    ),
                    suffix_ids,
                    torch.tensor([self.eos_id], dtype=torch.long),
                ]
            )
        else:
            labels = torch.full_like(input_ids, -100)
        return {
            "input_ids": input_ids,
            "token_types": token_types,
            "labels": labels,
            "prompt_len": torch.tensor(prefix_len, dtype=torch.long),
            "suffix_len": torch.tensor(suffix_len, dtype=torch.long),
            "image_start": torch.tensor(prefix_len + 1, dtype=torch.long),
        }

    def _class_name_for_img_id(self, img_id: int) -> str:
        synset = self.synsets.get(int(img_id), "")
        class_name, _ = self.synset_names.get(synset, (synset, synset))
        class_name = str(class_name).strip()
        if not class_name:
            raise ValueError(f"No class name found for img_id={img_id}.")
        return class_name

    def _class_sequence_tensors(self, img_id: int) -> Dict[str, torch.Tensor]:
        class_name = self._class_name_for_img_id(img_id)
        cached = self.sequence_cache.get(class_name)
        if cached is None:
            cached = self._make_sequence_tensors(self._text_ids(class_name), torch.empty(0, dtype=torch.long))
            self.sequence_cache[class_name] = cached
        return cached

    def _caption_mode_for_sample(self, idx: int) -> str:
        mode_idx = (int(idx) * 1103515245 + self.epoch * 1_000_003 + self.seed) % len(self.caption_sequence_modes)
        return self.caption_sequence_modes[mode_idx]

    def _caption_prefix_for_sample(self, idx: int, mode: str) -> str:
        prefixes = self.caption_i2t_prefixes if mode == "i2t" else self.caption_t2i_prefixes
        prefix_idx = (
            int(idx) * 214013
            + self.epoch * 2_654_435_761
            + self.seed * 17
            + (1 if mode == "i2t" else 0)
        ) % len(prefixes)
        return prefixes[prefix_idx]

    def _caption_ids_for_img_id(self, img_id: int) -> torch.Tensor:
        caption = self.captions.get(int(img_id))
        if caption is None:
            if self.caption_missing_policy == "fallback_class":
                caption = self._class_name_for_img_id(int(img_id))
            else:
                raise KeyError(f"No caption for img_id={img_id}")
        ids = self._text_ids(caption, cache=self.cache_caption_tokens)
        if self.caption_max_tokens > 0:
            ids = ids[: self.caption_max_tokens]
        return ids

    def _caption_sequence_tensors(self, idx: int, img_id: int) -> Dict[str, torch.Tensor]:
        mode = self._caption_mode_for_sample(idx)
        prefix = self._caption_prefix_for_sample(idx, mode)
        caption_ids = self._caption_ids_for_img_id(img_id)
        prefix_ids = self._text_ids(prefix)
        if mode == "i2t":
            return self._make_sequence_tensors(prefix_ids, caption_ids)
        text_ids = torch.cat([prefix_ids, caption_ids])
        return self._make_sequence_tensors(text_ids, torch.empty(0, dtype=torch.long))

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        latents = self._augment_latents(self.latents[idx], int(idx))
        if self.conditioning_mode == "image_only":
            return {
                "input_ids": self.fixed_input_ids,
                "token_types": self.fixed_token_types,
                "labels": self.fixed_labels,
                "image_latents": latents,
                "prompt_len": torch.tensor(0, dtype=torch.long),
                "suffix_len": torch.tensor(0, dtype=torch.long),
                "image_start": torch.tensor(1, dtype=torch.long),
            }

        img_id = int(self.img_ids[idx].item())
        if self.conditioning_mode == "class_image":
            sequence = self._class_sequence_tensors(img_id)
        else:
            sequence = self._caption_sequence_tensors(int(idx), img_id)

        length = int(sequence["input_ids"].numel())
        if self.max_seq_length is not None and length > self.max_seq_length:
            raise ValueError(
                f"ImageNetFlowCacheDataset item length {length} exceeds max_seq_length={self.max_seq_length}. "
                "Increase max_seq_length or reduce caption_max_tokens."
            )

        return {
            "input_ids": sequence["input_ids"],
            "token_types": sequence["token_types"],
            "labels": sequence["labels"],
            "image_latents": latents,
            "prompt_len": sequence["prompt_len"],
            "suffix_len": sequence["suffix_len"],
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
        suffix_len = int(item["suffix_len"].item())
        image_start = int(item["image_start"].item())
        eoi_pos = image_start + image_tokens
        suffix_start = eoi_pos + 1
        eos_pos = length - 1

        input_ids[i, :length] = item["input_ids"]
        token_types[i, :length] = item["token_types"]
        labels[i, :length] = item["labels"]
        image_latents[i, image_start:eoi_pos] = item["image_latents"]

        if prompt_len:
            sigma[i, :prompt_len] = torch.arange(prompt_len, dtype=torch.long)
        sigma[i, prompt_len] = prompt_len
        sigma[i, eoi_pos] = prompt_len + 1
        sigma[i, image_start:eoi_pos] = prompt_len + 2 + order[i]
        suffix_sigma_start = prompt_len + image_tokens + 2
        if suffix_len:
            sigma[i, suffix_start:eos_pos] = suffix_sigma_start + torch.arange(suffix_len, dtype=torch.long)
        sigma[i, eos_pos] = suffix_sigma_start + suffix_len

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


def _split_key_for_index(dataset: ImageNetFlowCacheDataset, idx: int) -> str:
    img_id = int(dataset.img_ids[int(idx)].item())
    return dataset.synsets.get(img_id, "")


def _build_split_indices(
    dataset: ImageNetFlowCacheDataset,
    val_ratio: float,
    seed: int,
    strategy: str,
) -> Tuple[List[int], List[int]]:
    n_items = len(dataset)
    if n_items <= 0:
        return [], []
    val_size = max(1, int(n_items * float(val_ratio)))
    val_size = min(val_size, max(1, n_items - 1))
    strategy = str(strategy or "stratified").lower()

    if strategy in {"contiguous", "tail"}:
        train_size = max(0, n_items - val_size)
        return list(range(train_size)), list(range(train_size, n_items))

    rng = random.Random(int(seed))
    if strategy in {"shuffle", "shuffled", "random"}:
        indices = list(range(n_items))
        rng.shuffle(indices)
        val_indices = indices[:val_size]
        train_indices = indices[val_size:]
        return train_indices, val_indices

    if strategy not in {"stratified", "synset", "stratified_synset"}:
        raise ValueError(
            f"Unknown ImageNet flow split_strategy={strategy!r}; "
            "expected stratified, shuffled, or contiguous."
        )

    groups: Dict[str, List[int]] = {}
    for idx in range(n_items):
        key = _split_key_for_index(dataset, idx)
        groups.setdefault(key, []).append(idx)
    if len(groups) <= 1:
        indices = list(range(n_items))
        rng.shuffle(indices)
        val_indices = indices[:val_size]
        train_indices = indices[val_size:]
        return train_indices, val_indices

    train_indices: List[int] = []
    val_indices: List[int] = []
    for key in sorted(groups):
        group_indices = list(groups[key])
        rng.shuffle(group_indices)
        group_val_size = max(1, int(len(group_indices) * float(val_ratio)))
        if len(group_indices) > 1:
            group_val_size = min(group_val_size, len(group_indices) - 1)
        else:
            group_val_size = 0
        val_indices.extend(group_indices[:group_val_size])
        train_indices.extend(group_indices[group_val_size:])

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return train_indices, val_indices


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
        synset_mapping_path=params.get("synset_mapping_path", None),
        conditioning_mode=params.get("conditioning_mode", None),
        caption_jsonl=params.get("caption_jsonl", None),
        caption_text_key=params.get("caption_text_key", "recaption_short"),
        caption_path_key=params.get("caption_path_key", "path"),
        caption_id_key=params.get("caption_id_key", "id"),
        caption_sequence_modes=params.get("caption_sequence_modes", None),
        caption_t2i_prefixes=params.get("caption_t2i_prefixes", None),
        caption_i2t_prefixes=params.get("caption_i2t_prefixes", None),
        caption_missing_policy=params.get("caption_missing_policy", "error"),
        caption_max_tokens=params.get("caption_max_tokens", 192),
        cache_caption_tokens=params.get("cache_caption_tokens", False),
        max_seq_length=params.get("max_seq_length", config.dataset.preprocessing.max_seq_length),
        max_samples=params.get("max_samples", -1),
        seed=config.training.seed,
        latent_hflip_prob=params.get("latent_hflip_prob", 0.0),
        label_text=params.get("label_text", True),
    )

    val_ratio = params.get("val_ratio", 0.001)
    split_seed = params.get("split_seed", config.training.seed)
    split_strategy = params.get("split_strategy", "stratified")
    train_indices, val_indices = _build_split_indices(
        dataset=dataset,
        val_ratio=float(val_ratio),
        seed=int(split_seed),
        strategy=str(split_strategy),
    )
    dataset.set_augmentation_indices(train_indices)
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

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
