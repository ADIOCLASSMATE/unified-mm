"""
Cache-backed ImageNet latent dataset for unified image-flow training.

The cache is the merged tensor produced by the VAE encoder:
    {"latents": [N, image_tokens, latent_dim], "img_ids": [N], ...}

This loader supports exactly two conditioning modes:
    class:    class_name <|boi|> image <|eoi|> <eos>
    caption:  fixed_T2I_prefix caption <|boi|> image <|eoi|> <eos>
"""

import hashlib
import json
import random
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from models.modeling_model.image_backbone import (
    validate_direct_latent_cache_shape,
)
from utils.multimodal_segment_packing import collate_segment_packed


DEFAULT_CAPTION_PREFIX = "Generate an image matching this description:"


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
        cache_caption_tokens: bool = False,
        max_seq_length: Optional[int] = None,
        model_context_length: Optional[int] = None,
        caption_manifest_sha256: Optional[str] = None,
        max_samples: int = -1,
        seed: int = 42,
        latent_hflip_prob: float = 0.0,
    ):
        self.cache_path = Path(cache_path)
        if not self.cache_path.exists():
            raise FileNotFoundError(self.cache_path)

        # Memory-map the merged latent cache so distributed ranks and
        # persistent workers reuse OS pages instead of copying the full
        # ImageNet tensor into every process.
        obj = torch.load(
            self.cache_path,
            map_location="cpu",
            mmap=True,
            weights_only=True,
        )
        self.latents = obj["latents"]
        self.img_ids = obj.get("img_ids", torch.arange(self.latents.shape[0]))
        if max_samples is not None and max_samples > 0:
            self.latents = self.latents[:max_samples]
            self.img_ids = self.img_ids[:max_samples]

        self.canonical_latent_side = validate_direct_latent_cache_shape(
            self.latents,
            image_tokens_per_img=image_tokens_per_img,
            image_latent_dim=image_latent_dim,
        )
        self.canonical_image_latent_dim = int(image_latent_dim)

        self.tokenizer = tokenizer
        self.boi_id = int(boi_token_id)
        self.eoi_id = int(eoi_token_id)
        self.mask_id = int(mask_token_id)
        self.eos_id = int(eos_token_id)
        self.image_tokens_per_img = int(image_tokens_per_img)
        self.image_latent_dim = int(image_latent_dim)
        self.max_seq_length = int(max_seq_length) if max_seq_length else None
        self.model_context_length = (
            int(model_context_length) if model_context_length else 32768
        )
        self.seed = int(seed)
        # DataLoader workers keep their own Dataset object, so a plain Python
        # integer would become stale when persistent_workers=True.  Tensor
        # storage remains shared after the Dataset is sent to workers, which
        # lets the training process advance the epoch without recreating them.
        self._epoch_state = torch.zeros((), dtype=torch.int64).share_memory_()
        self.latent_hflip_prob = float(latent_hflip_prob)
        if not 0.0 <= self.latent_hflip_prob <= 1.0:
            raise ValueError(f"latent_hflip_prob must be in [0, 1], got {latent_hflip_prob}")
        self.latent_side = self.canonical_latent_side
        if self.latent_hflip_prob > 0.0 and self.canonical_latent_side * self.canonical_latent_side != int(self.latents.shape[1]):
            raise ValueError(
                "latent_hflip_prob requires image_tokens_per_img to be a square grid, "
                f"got canonical cache shape {tuple(self.latents.shape[1:])}."
            )
        self.augmentation_train_size: Optional[int] = None
        self.augmentation_index_mask: Optional[torch.Tensor] = None

        self.synsets, self.source_paths = self._load_manifest(manifest_jsonl)
        self.synset_names = self._load_synset_names(synset_mapping_path)
        self.caption_jsonl = caption_jsonl
        self.caption_text_key = str(caption_text_key)
        self.caption_path_key = str(caption_path_key)
        self.caption_id_key = str(caption_id_key)
        self.caption_manifest_sha256 = (
            str(caption_manifest_sha256).strip().lower()
            if caption_manifest_sha256
            else None
        )
        self.cache_caption_tokens = bool(cache_caption_tokens)
        self.text_cache: Dict[str, torch.Tensor] = {}
        self.sequence_cache: Dict[str, Dict[str, torch.Tensor]] = {}

        if conditioning_mode is None:
            conditioning_mode = "caption" if caption_jsonl else "class"
        self.conditioning_mode = str(conditioning_mode).strip().lower()
        if self.conditioning_mode not in {"class", "caption"}:
            raise ValueError(
                f"Unknown conditioning_mode={conditioning_mode!r}; "
                "expected 'class' or 'caption'."
            )

        if not self.synsets:
            raise ValueError("class/caption conditioning requires manifest_jsonl")
        if self.conditioning_mode == "class" and not self.synset_names:
            raise ValueError("conditioning_mode='class' requires synset_mapping_path")
        self.captions: Dict[int, str] = {}
        if self.conditioning_mode == "caption":
            if not caption_jsonl:
                raise ValueError("conditioning_mode='caption' requires caption_jsonl")
            self._validate_caption_manifest_digest(caption_jsonl)
            self.captions = self._load_captions(caption_jsonl)
            if not self.captions:
                raise ValueError(f"No captions matched cache img_ids from {caption_jsonl}.")

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
                if img_id in synsets:
                    raise ValueError(
                        f"duplicate img_id={img_id} in manifest {path}"
                    )
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

    def _caption_jsonl_paths(self, caption_jsonl: str | Sequence[str]) -> List[Path]:
        if isinstance(caption_jsonl, (str, Path)):
            return [Path(caption_jsonl)]
        return [Path(path) for path in caption_jsonl]

    def _validate_caption_manifest_digest(
        self,
        caption_jsonl: str | Sequence[str],
    ) -> None:
        if not self.caption_manifest_sha256:
            return
        paths = self._caption_jsonl_paths(caption_jsonl)
        if len(paths) != 1:
            raise ValueError(
                "caption_manifest_sha256 requires exactly one frozen caption "
                f"manifest, got {paths}"
            )
        digest = hashlib.sha256()
        with paths[0].open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        actual = digest.hexdigest()
        if actual != self.caption_manifest_sha256:
            raise ValueError(
                "caption manifest digest mismatch: "
                f"expected={self.caption_manifest_sha256}, actual={actual}, "
                f"path={paths[0]}"
            )

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
                        relative_path = self._relative_image_path(str(row_path))
                        if relative_path in captions_by_path:
                            raise ValueError(
                                f"duplicate caption path={relative_path!r} in {path}"
                            )
                        captions_by_path[relative_path] = caption
                    row_id = row.get(self.caption_id_key)
                    if row_id:
                        normalized_id = str(row_id)
                        if normalized_id in captions_by_id:
                            raise ValueError(
                                f"duplicate caption id={normalized_id!r} in {path}"
                            )
                        captions_by_id[normalized_id] = caption
        return captions_by_path, captions_by_id

    def _load_captions(self, caption_jsonl: str | Sequence[str]) -> Dict[int, str]:
        captions_by_path, captions_by_id = self._load_caption_rows(caption_jsonl)
        captions: Dict[int, str] = {}
        missing: List[str] = []
        for img_id_tensor in self.img_ids:
            img_id = int(img_id_tensor.item())
            rel_path = self.source_paths.get(img_id)
            caption_by_path = captions_by_path.get(rel_path or "")
            caption_by_id = (
                captions_by_id.get(Path(rel_path).stem) if rel_path else None
            )
            if (
                caption_by_path is not None
                and caption_by_id is not None
                and caption_by_path != caption_by_id
            ):
                raise ValueError(
                    f"caption path/id conflict for img_id={img_id}, "
                    f"path={rel_path!r}"
                )
            caption = (
                caption_by_path
                if caption_by_path is not None
                else caption_by_id
            )
            if caption is None:
                if len(missing) < 5:
                    missing.append(rel_path or str(img_id))
                continue
            captions[img_id] = caption

        missing_count = int(self.img_ids.numel()) - len(captions)
        if missing_count:
            raise ValueError(
                f"Missing {missing_count} captions for cache img_ids. "
                f"Examples: {missing}. Use a caption manifest matching the cache."
            )
        return captions

    def __len__(self) -> int:
        return int(self.latents.shape[0])

    @property
    def epoch(self) -> int:
        """Current epoch, shared with persistent DataLoader worker copies."""

        return int(self._epoch_state.item())

    @epoch.setter
    def epoch(self, epoch: int) -> None:
        self._epoch_state.fill_(int(epoch))

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

    def _augment_latents(
        self,
        latents: torch.Tensor,
        idx: int,
        current_epoch: Optional[int] = None,
    ) -> torch.Tensor:
        if not self._latent_hflip_applied(idx, current_epoch):
            return latents
        return latents.view(
            self.canonical_latent_side,
            self.canonical_latent_side,
            self.canonical_image_latent_dim,
        ).flip(1).reshape_as(latents)

    def _latent_hflip_applied(
        self,
        idx: int,
        current_epoch: Optional[int] = None,
    ) -> bool:
        if self.latent_hflip_prob <= 0.0:
            return False
        if current_epoch is None:
            current_epoch = self.epoch
        idx = int(idx)
        if self.augmentation_index_mask is not None:
            if idx < 0 or idx >= int(self.augmentation_index_mask.numel()) or not bool(self.augmentation_index_mask[idx]):
                return False
        elif self.augmentation_train_size is not None and idx >= self.augmentation_train_size:
            return False
        rng = random.Random(
            self.seed + current_epoch * 1_000_003 + int(idx) * 9_176 + 13_579
        )
        return rng.random() < self.latent_hflip_prob

    def _stable_sample_seed(
        self,
        idx: int,
        current_epoch: int,
        purpose: str,
    ) -> int:
        payload = (
            f"{self.seed}:{int(current_epoch)}:{int(idx)}:{str(purpose)}"
        ).encode("utf-8")
        return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & (
            (1 << 63) - 1
        )

    def _apply_latent_layout(self, latents: torch.Tensor) -> torch.Tensor:
        return latents.reshape(self.image_tokens_per_img, self.image_latent_dim)

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
        raise ValueError(
            "Serialized text exceeds max_seq_length; caption truncation is not "
            f"supported: text_tokens={prefix_len + suffix_len}, "
            f"available_text={available_text}."
        )

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

    def _caption_sequence_tensors(self, img_id: int) -> Dict[str, torch.Tensor]:
        caption = self.captions.get(int(img_id))
        if caption is None:
            raise KeyError(f"No caption for img_id={img_id}")
        serialized_prompt = f"{DEFAULT_CAPTION_PREFIX} {caption}"
        cached = (
            self.sequence_cache.get(serialized_prompt)
            if self.cache_caption_tokens
            else None
        )
        if cached is not None:
            return cached
        sequence = self._make_sequence_tensors(
            self._text_ids(
                serialized_prompt,
                cache=self.cache_caption_tokens,
            ),
            torch.empty(0, dtype=torch.long),
        )
        if self.cache_caption_tokens:
            self.sequence_cache[serialized_prompt] = sequence
        return sequence

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Capture one shared epoch value so every stochastic choice for this
        # sample uses the same epoch even if the parent advances immediately.
        current_epoch = self.epoch
        hflip_applied = self._latent_hflip_applied(int(idx), current_epoch)
        latents = self._augment_latents(self.latents[idx], int(idx), current_epoch)
        latents = self._apply_latent_layout(latents)
        img_id = int(self.img_ids[idx].item())
        reveal_seed = self._stable_sample_seed(
            int(idx), current_epoch, "image_reveal_order"
        )
        cfg_dropout_seed = self._stable_sample_seed(
            int(idx), current_epoch, "cfg_dropout"
        )
        if self.conditioning_mode == "class":
            sequence = self._class_sequence_tensors(img_id)
        else:
            sequence = self._caption_sequence_tensors(img_id)

        length = int(sequence["input_ids"].numel())
        if self.max_seq_length is not None and length > self.max_seq_length:
            raise ValueError(
                f"ImageNetFlowCacheDataset item length {length} exceeds max_seq_length={self.max_seq_length}. "
                "Increase max_seq_length."
            )
        if (
            self.model_context_length is not None
            and length > self.model_context_length
        ):
            raise ValueError(
                f"Serialized sample length {length} exceeds model context "
                f"window {self.model_context_length} for img_id={img_id}."
            )
        token_ids_sha256 = hashlib.sha256(
            sequence["input_ids"].contiguous().numpy().tobytes()
        ).hexdigest()
        augmentation_sha256 = hashlib.sha256(
            json.dumps(
                {
                    "epoch": int(current_epoch),
                    "hflip": bool(hflip_applied),
                    "img_id": img_id,
                    "sample_index": int(idx),
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

        return {
            "input_ids": sequence["input_ids"],
            "token_types": sequence["token_types"],
            "labels": sequence["labels"],
            "image_latents": latents,
            "prompt_len": sequence["prompt_len"],
            "suffix_len": sequence["suffix_len"],
            "image_start": sequence["image_start"],
            "img_id": torch.tensor(img_id, dtype=torch.long),
            "sample_index": torch.tensor(int(idx), dtype=torch.long),
            "serialized_length": torch.tensor(length, dtype=torch.long),
            "token_ids_sha256": token_ids_sha256,
            "augmentation_sha256": augmentation_sha256,
            "reveal_seed": torch.tensor(reveal_seed, dtype=torch.long),
            "cfg_dropout_seed": torch.tensor(
                cfg_dropout_seed, dtype=torch.long
            ),
        }


def collate_imagenet_flow_cache(
    batch: List[Dict[str, torch.Tensor]],
    pad_to_length: Optional[int] = None,
    pad_to_multiple_of: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
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
    val_samples_per_class: Optional[int] = None,
) -> Tuple[List[int], List[int]]:
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
            f"val_samples_per_class must be positive, got {val_samples_per_class}"
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
        single_group_val_size = fixed_val_size if fixed_val_size is not None else val_size
        single_group_val_size = min(single_group_val_size, max(1, n_items - 1))
        val_indices = indices[:single_group_val_size]
        train_indices = indices[single_group_val_size:]
        return train_indices, val_indices

    train_indices: List[int] = []
    val_indices: List[int] = []
    for key in sorted(groups):
        group_indices = list(groups[key])
        rng.shuffle(group_indices)
        if fixed_val_size is not None:
            group_val_size = fixed_val_size
        else:
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


def _load_explicit_split_indices(
    dataset: ImageNetFlowCacheDataset,
    split_manifest_jsonl: str,
) -> Tuple[List[int], List[int]]:
    """Resolve a shared split manifest by image id.

    This keeps training and evaluation on exactly the declared members even
    when cache row order differs.
    """

    path = Path(split_manifest_jsonl)
    if not path.exists():
        raise FileNotFoundError(path)
    index_by_image_id = {
        int(image_id.item()): index
        for index, image_id in enumerate(dataset.img_ids)
    }
    split_rows: Dict[str, List[Tuple[int, int]]] = {
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
                    f"{path}:{line_number} img_id={image_id} is absent from cache"
                )
            expected_synset = dataset.synsets.get(image_id, "")
            row_synset = str(row.get("synset", expected_synset))
            if row_synset != expected_synset:
                raise ValueError(
                    f"{path}:{line_number} synset mismatch for img_id={image_id}: "
                    f"{row_synset!r} != {expected_synset!r}"
                )
            split_index = int(row.get("split_index", len(split_rows[split])))
            split_rows[split].append((split_index, index_by_image_id[image_id]))
            seen.add(image_id)
    if seen != set(index_by_image_id):
        missing = sorted(set(index_by_image_id).difference(seen))
        raise ValueError(
            f"{path} does not cover the complete cache; missing {len(missing)} "
            f"image ids, first={missing[:8]}"
        )
    train_indices = [
        index for _, index in sorted(split_rows["train"], key=lambda item: item[0])
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
        cache_caption_tokens=params.get("cache_caption_tokens", False),
        max_seq_length=params.get("max_seq_length", config.dataset.preprocessing.max_seq_length),
        model_context_length=params.get("model_context_length", None),
        caption_manifest_sha256=params.get(
            "caption_manifest_sha256", None
        ),
        max_samples=params.get("max_samples", -1),
        seed=config.training.seed,
        latent_hflip_prob=params.get("latent_hflip_prob", 0.0),
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
    dataset.set_augmentation_indices(train_indices)
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    pad_to_length = params.get("pad_to_length", None)
    if params.get("pad_to_max_length", False):
        pad_to_length = params.get("max_seq_length", config.dataset.preprocessing.max_seq_length)
    pad_to_multiple_of = params.get("pad_to_multiple_of", None)

    packing = params.get("packing", None)
    packing_enabled = bool(
        packing is not None and packing.get("enabled", False)
    )
    unpacked_collate_fn = partial(
        collate_imagenet_flow_cache,
        pad_to_length=pad_to_length,
        pad_to_multiple_of=pad_to_multiple_of,
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
                "overflow_policy", "dedicated_round_up_128"
            )
        )
        if overflow_policy != "dedicated_round_up_128":
            raise ValueError(
                f"unsupported overflow_policy={overflow_policy!r}"
            )
        train_collate_fn = partial(
            collate_segment_packed,
            nominal_capacity=int(
                packing.get("nominal_capacity", 2048)
            ),
            overflow_multiple=128,
            image_uncond_prob=float(
                config.model.get("image_uncond_prob", 0.0)
            ),
        )
    else:
        train_collate_fn = unpacked_collate_fn

    # Packing is a training throughput optimization. Validation and generation
    # operate on one logical sample per physical row so that loss, positions,
    # sample ordering, and generated images cannot accidentally mix segments.
    val_collate_fn = unpacked_collate_fn

    train_generator = build_training_data_generator(config)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.dataloader_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=train_collate_fn,
        persistent_workers=config.training.dataloader_workers > 0,
        generator=train_generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.dataloader_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=val_collate_fn,
        persistent_workers=config.training.dataloader_workers > 0,
    )
    return train_loader, val_loader


def build_training_data_generator(config) -> torch.Generator | None:
    """Build an architecture-independent shuffle/worker RNG when explicitly requested."""

    raw_seed = config.training.get("dataloader_shuffle_seed", None)
    if raw_seed is None:
        return None
    seed = int(raw_seed)
    if seed < 0:
        raise ValueError(f"training.dataloader_shuffle_seed must be non-negative, got {seed}")
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator
