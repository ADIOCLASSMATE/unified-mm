"""ImageNet-100 data pipeline for class-conditional Qwen-Show-o training.

The dataset consumes only:

* a packed MAGVITv2 cache: ``{"image_ids": [N], "tokens": [N, T]}``;
* the ImageNet subset JSONL manifest containing ``img_id`` and ``synset``;
* the standard ImageNet synset-to-class-name mapping.

An optional authoritative split JSONL can pin cache-order-derived train/val
membership by ``img_id`` without adding any per-sample training content.

Every item is T2I-only and has the following unified-vocabulary layout::

    <|t2i|> class-name <|boi|> image-token... <|eoi|> <eos>

Text is causal while the complete BOI/image/EOI span uses Show-o's
bidirectional omni-attention. Image targets are supervised at the same
positions as their masked inputs.
"""

from __future__ import annotations

import json
import math
import random
from functools import lru_cache, partial
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Subset


TEXT_TOKEN_TYPE = 0
IMAGE_TOKEN_TYPE = 1
SPECIAL_TOKEN_TYPE = 2
PADDING_TOKEN_TYPE = 3
IGNORE_INDEX = -100


def _config_get(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(key, default)
    getter = getattr(config, "get", None)
    if callable(getter):
        try:
            return getter(key, default)
        except TypeError:
            pass
    return getattr(config, key, default)


def _load_packed_cache(path: Path, mmap: bool) -> Mapping[str, Any]:
    kwargs: Dict[str, Any] = {"map_location": "cpu"}
    if mmap:
        kwargs["mmap"] = True
    try:
        return torch.load(path, weights_only=True, **kwargs)
    except TypeError:
        kwargs.pop("mmap", None)
        return torch.load(path, **kwargs)


def _read_manifest(path: Path) -> Tuple[Dict[int, str], Dict[int, str]]:
    synsets: Dict[int, str] = {}
    source_paths: Dict[int, str] = {}
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if "img_id" not in row:
                raise ValueError(f"{path}:{line_number} is missing img_id")
            img_id = int(row["img_id"])
            if img_id in synsets:
                raise ValueError(f"duplicate img_id={img_id} in {path}")
            synset = str(row.get("synset", "")).strip()
            if not synset:
                raise ValueError(f"{path}:{line_number} is missing synset")
            synsets[img_id] = synset
            if row.get("source_path"):
                source_paths[img_id] = str(row["source_path"])
    if not synsets:
        raise ValueError(f"no ImageNet samples found in {path}")
    return synsets, source_paths


def _read_synset_mapping(path: Path) -> Dict[str, str]:
    names: Dict[str, str] = {}
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            synset, separator, raw_names = line.partition(" ")
            if not separator:
                continue
            class_name = raw_names.split(",", 1)[0].strip()
            names[synset] = class_name or synset
    if not names:
        raise ValueError(f"no synset names found in {path}")
    return names


class QwenShowOImageNetDataset(Dataset):
    """Packed MAGVITv2 ImageNet dataset with one T2I view per image."""

    def __init__(
        self,
        tokens_path: str,
        manifest_jsonl: str,
        synset_mapping_path: str,
        tokenizer: Any,
        t2i_token_id: int,
        boi_token_id: int,
        eoi_token_id: int,
        eos_token_id: int,
        image_offset: int,
        image_vocab_size: int = 8192,
        image_tokens_per_img: int = 256,
        t2i_prefix: str = "",
        class_prompt_template: str = "{class_name}",
        max_text_tokens: Optional[int] = None,
        max_seq_length: Optional[int] = None,
        max_samples: int = -1,
        mmap: bool = True,
    ):
        self.tokens_path = Path(tokens_path)
        self.manifest_path = Path(manifest_jsonl)
        self.synset_mapping_path = Path(synset_mapping_path)
        for path in (self.tokens_path, self.manifest_path, self.synset_mapping_path):
            if not path.exists():
                raise FileNotFoundError(path)

        payload = _load_packed_cache(self.tokens_path, mmap=bool(mmap))
        image_ids = payload.get("image_ids", payload.get("img_ids"))
        image_tokens = payload.get("tokens")
        if image_ids is None or image_tokens is None:
            raise ValueError(
                f"{self.tokens_path} must contain image_ids (or img_ids) and tokens"
            )
        self.cached_image_ids = torch.as_tensor(image_ids).long().contiguous()
        self.image_tokens = torch.as_tensor(image_tokens).contiguous()
        if self.cached_image_ids.ndim != 1:
            raise ValueError(
                f"expected image_ids [N], got {tuple(self.cached_image_ids.shape)}"
            )
        if self.image_tokens.ndim != 2:
            raise ValueError(
                f"expected image tokens [N,T], got {tuple(self.image_tokens.shape)}"
            )
        if self.image_tokens.shape[0] != self.cached_image_ids.numel():
            raise ValueError("image_ids and tokens have different sample counts")
        if self.image_tokens.shape[1] != int(image_tokens_per_img):
            raise ValueError(
                f"expected {image_tokens_per_img} tokens per image, "
                f"got {self.image_tokens.shape[1]}"
            )
        if self.cached_image_ids.numel() != torch.unique(self.cached_image_ids).numel():
            raise ValueError(f"duplicate image IDs found in {self.tokens_path}")
        if self.image_tokens.numel():
            token_min = int(self.image_tokens.min().item())
            token_max = int(self.image_tokens.max().item())
            if token_min < 0 or token_max >= int(image_vocab_size):
                raise ValueError(
                    f"MAGVITv2 codes must be in [0,{int(image_vocab_size)}), "
                    f"got [{token_min},{token_max}]"
                )

        manifest_synsets, source_paths = _read_manifest(self.manifest_path)
        synset_names = _read_synset_mapping(self.synset_mapping_path)
        missing_manifest = [
            int(img_id)
            for img_id in self.cached_image_ids.tolist()
            if int(img_id) not in manifest_synsets
        ]
        if missing_manifest:
            raise ValueError(
                "packed cache contains image IDs absent from the subset manifest: "
                f"{missing_manifest[:10]}"
            )
        used_synsets = {
            manifest_synsets[int(img_id)] for img_id in self.cached_image_ids.tolist()
        }
        missing_names = sorted(used_synsets.difference(synset_names))
        if missing_names:
            raise ValueError(
                f"synset mapping is missing {len(missing_names)} classes: {missing_names[:10]}"
            )

        limit = self.cached_image_ids.numel()
        if max_samples is not None and int(max_samples) > 0:
            limit = min(limit, int(max_samples))
            self.cached_image_ids = self.cached_image_ids[:limit]
            self.image_tokens = self.image_tokens[:limit]

        self.tokenizer = tokenizer
        self.t2i_token_id = int(t2i_token_id)
        self.boi_token_id = int(boi_token_id)
        self.eoi_token_id = int(eoi_token_id)
        self.eos_token_id = int(eos_token_id)
        self.image_offset = int(image_offset)
        self.image_vocab_size = int(image_vocab_size)
        self.image_tokens_per_img = int(image_tokens_per_img)
        self.t2i_prefix = str(t2i_prefix)
        self.class_prompt_template = str(class_prompt_template)
        self.max_text_tokens = (
            int(max_text_tokens) if max_text_tokens is not None else None
        )
        self.max_seq_length = (
            int(max_seq_length) if max_seq_length is not None else None
        )
        if self.image_offset < 0:
            raise ValueError(f"image_offset must be non-negative, got {self.image_offset}")
        image_range = range(
            self.image_offset, self.image_offset + self.image_vocab_size
        )
        colliding_specials = [
            token_id
            for token_id in (
                self.t2i_token_id,
                self.boi_token_id,
                self.eoi_token_id,
                self.eos_token_id,
            )
            if token_id in image_range
        ]
        if colliding_specials:
            raise ValueError(
                "text/special token IDs overlap the unified image vocabulary: "
                f"{colliding_specials}"
            )

        class_by_synset = {
            synset: class_id for class_id, synset in enumerate(sorted(used_synsets))
        }
        self.rows: List[Tuple[int, int, str, str, int]] = []
        self.synsets: Dict[int, str] = {}
        self.source_paths: Dict[int, str] = {}
        for token_row, image_id_tensor in enumerate(self.cached_image_ids.tolist()):
            image_id = int(image_id_tensor)
            synset = manifest_synsets[image_id]
            class_name = synset_names[synset]
            self.rows.append(
                (token_row, image_id, synset, class_name, class_by_synset[synset])
            )
            self.synsets[image_id] = synset
            if image_id in source_paths:
                self.source_paths[image_id] = source_paths[image_id]
        self.img_ids = self.cached_image_ids
        self.class_names = {
            synset: synset_names[synset] for synset in sorted(used_synsets)
        }

        for class_name in self.class_names.values():
            text_length = self._conditional_text_ids(class_name).numel()
            sequence_length = 1 + text_length + 1 + self.image_tokens_per_img + 2
            if self.max_seq_length is not None and sequence_length > self.max_seq_length:
                raise ValueError(
                    f"class prompt {class_name!r} needs sequence length {sequence_length}, "
                    f"exceeding max_seq_length={self.max_seq_length}"
                )

    @staticmethod
    def _join_prompt(prefix: str, text: str) -> str:
        prefix = str(prefix)
        text = str(text).strip()
        if not prefix:
            return text
        if not text:
            return prefix.strip()
        separator = "" if prefix[-1].isspace() else " "
        return f"{prefix}{separator}{text}".strip()

    @staticmethod
    def _format_class_prompt(template: str, class_name: str) -> str:
        try:
            return str(template).format(class_name=class_name).strip()
        except IndexError:
            return str(template).format(class_name).strip()

    @lru_cache(maxsize=2048)
    def _text_ids(self, prompt: str) -> torch.Tensor:
        ids = self.tokenizer.encode(prompt, add_special_tokens=False) if prompt else []
        if self.max_text_tokens is not None:
            ids = ids[: self.max_text_tokens]
        return torch.tensor(ids, dtype=torch.long)

    def _conditional_text_ids(self, class_name: str) -> torch.Tensor:
        class_prompt = self._format_class_prompt(
            self.class_prompt_template, class_name
        )
        return self._text_ids(self._join_prompt(self.t2i_prefix, class_prompt))

    def _unconditional_text_ids(self) -> torch.Tensor:
        return self._text_ids(self.t2i_prefix.strip())

    def _build_sequence(
        self,
        text_ids: torch.Tensor,
        unified_image_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        input_ids = torch.cat(
            [
                torch.tensor([self.t2i_token_id], dtype=torch.long),
                text_ids,
                torch.tensor([self.boi_token_id], dtype=torch.long),
                unified_image_ids,
                torch.tensor(
                    [self.eoi_token_id, self.eos_token_id], dtype=torch.long
                ),
            ]
        )
        image_start = 1 + int(text_ids.numel()) + 1
        token_types = torch.cat(
            [
                torch.tensor([SPECIAL_TOKEN_TYPE], dtype=torch.uint8),
                torch.full(
                    (text_ids.numel(),), TEXT_TOKEN_TYPE, dtype=torch.uint8
                ),
                torch.tensor([SPECIAL_TOKEN_TYPE], dtype=torch.uint8),
                torch.full(
                    (self.image_tokens_per_img,),
                    IMAGE_TOKEN_TYPE,
                    dtype=torch.uint8,
                ),
                torch.tensor(
                    [SPECIAL_TOKEN_TYPE, SPECIAL_TOKEN_TYPE], dtype=torch.uint8
                ),
            ]
        )
        return input_ids, token_types, image_start

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        token_row, image_id, synset, class_name, class_id = self.rows[int(index)]
        local_image_ids = self.image_tokens[token_row].long()
        unified_image_ids = local_image_ids + self.image_offset
        input_ids, token_types, image_start = self._build_sequence(
            self._conditional_text_ids(class_name), unified_image_ids
        )
        (
            unconditional_input_ids,
            unconditional_token_types,
            unconditional_image_start,
        ) = self._build_sequence(
            self._unconditional_text_ids(), unified_image_ids
        )
        return {
            "input_ids": input_ids,
            "token_types": token_types,
            "image_start": torch.tensor(image_start, dtype=torch.long),
            "unconditional_input_ids": unconditional_input_ids,
            "unconditional_token_types": unconditional_token_types,
            "unconditional_image_start": torch.tensor(
                unconditional_image_start, dtype=torch.long
            ),
            "image_token_ids": local_image_ids,
            "sample_id": torch.tensor(image_id, dtype=torch.long),
            "class_id": torch.tensor(class_id, dtype=torch.long),
            "class_name": class_name,
            "synset": synset,
        }


def build_showo_omni_attention_mask(
    token_types: torch.Tensor,
    lengths: torch.Tensor,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return causal-text/full-image additive attention of shape ``[B,1,L,L]``."""

    if token_types.ndim != 2 or lengths.ndim != 1:
        raise ValueError("token_types must be [B,L] and lengths must be [B]")
    batch_size, sequence_length = token_types.shape
    if lengths.numel() != batch_size:
        raise ValueError("lengths has a different batch size from token_types")
    device = token_types.device
    causal = torch.tril(
        torch.ones(
            sequence_length, sequence_length, device=device, dtype=torch.bool
        )
    )
    allowed = causal.unsqueeze(0).expand(batch_size, -1, -1).clone()

    for row in range(batch_size):
        length = int(lengths[row].item())
        if not 0 < length <= sequence_length:
            raise ValueError(f"invalid sequence length {length}")
        image_positions = (
            token_types[row, :length] == IMAGE_TOKEN_TYPE
        ).nonzero(as_tuple=True)[0]
        if image_positions.numel():
            gaps = (
                image_positions[1:] != image_positions[:-1] + 1
            ).nonzero(as_tuple=True)[0] + 1
            boundaries = torch.cat(
                [
                    torch.zeros(1, device=device, dtype=torch.long),
                    gaps,
                    torch.tensor(
                        [image_positions.numel()], device=device, dtype=torch.long
                    ),
                ]
            )
            for start, end in zip(
                boundaries[:-1].tolist(), boundaries[1:].tolist()
            ):
                span = image_positions[start:end]
                boi_position = int(span[0].item()) - 1
                eoi_position = int(span[-1].item()) + 1
                if boi_position < 0 or eoi_position >= length:
                    raise ValueError("image span is missing BOI or EOI")
                allowed[
                    row,
                    boi_position : eoi_position + 1,
                    : eoi_position + 1,
                ] = True

        allowed[row, :, length:] = False
        if length < sequence_length:
            padding_positions = torch.arange(
                length, sequence_length, device=device
            )
            allowed[row, padding_positions, :] = False
            allowed[row, padding_positions, padding_positions] = True

    if dtype == torch.bool:
        return allowed.unsqueeze(1)
    if not torch.empty((), dtype=dtype).is_floating_point():
        raise TypeError(f"attention dtype must be floating point or bool, got {dtype}")
    additive = torch.zeros(
        (batch_size, 1, sequence_length, sequence_length),
        device=device,
        dtype=dtype,
    )
    return additive.masked_fill(~allowed.unsqueeze(1), torch.finfo(dtype).min)


def _drop_class_condition(
    input_ids: torch.Tensor,
    token_types: torch.Tensor,
    image_start: int,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    boi_position = int(image_start) - 1
    if boi_position < 1:
        raise ValueError("expected task token before BOI")
    return (
        torch.cat([input_ids[:1], input_ids[boi_position:]]),
        torch.cat([token_types[:1], token_types[boi_position:]]),
        2,
    )


def collate_qwen_showo_imagenet(
    batch: Sequence[Mapping[str, Any]],
    *,
    pad_token_id: int,
    image_mask_token_id: int,
    cond_dropout_prob: float = 0.0,
    min_masking_rate: float = 0.0,
    fixed_mask_ratio: Optional[float] = None,
    pad_to_length: Optional[int] = None,
    pad_to_multiple_of: Optional[int] = 64,
    attention_dtype: torch.dtype = torch.float32,
    mask_seed: Optional[int] = None,
    generator: Optional[torch.Generator] = None,
) -> Dict[str, Any]:
    """Apply condition dropout, cosine masking, padding and omni-attention."""

    if not batch:
        raise ValueError("cannot collate an empty batch")
    if not 0.0 <= float(cond_dropout_prob) <= 1.0:
        raise ValueError("cond_dropout_prob must be in [0,1]")
    if not 0.0 <= float(min_masking_rate) <= 1.0:
        raise ValueError("min_masking_rate must be in [0,1]")
    if fixed_mask_ratio is not None and not 0.0 <= float(fixed_mask_ratio) <= 1.0:
        raise ValueError("fixed_mask_ratio must be in [0,1]")

    prepared = []
    for item in batch:
        input_ids = item["input_ids"].long()
        token_types = item["token_types"].to(torch.uint8)
        image_start = int(item["image_start"])
        condition_dropped = bool(
            cond_dropout_prob > 0.0
            and torch.rand((), generator=generator).item() < cond_dropout_prob
        )
        if condition_dropped:
            if "unconditional_input_ids" in item:
                input_ids = item["unconditional_input_ids"].long()
                token_types = item["unconditional_token_types"].to(torch.uint8)
                image_start = int(item["unconditional_image_start"])
            else:
                input_ids, token_types, image_start = _drop_class_condition(
                    input_ids, token_types, image_start
                )
        prepared.append(
            (item, input_ids, token_types, image_start, condition_dropped)
        )

    max_length = max(int(values[1].numel()) for values in prepared)
    if pad_to_length is not None:
        if int(pad_to_length) < max_length:
            raise ValueError(
                f"pad_to_length={pad_to_length} is smaller than batch max {max_length}"
            )
        max_length = int(pad_to_length)
    if pad_to_multiple_of:
        multiple = int(pad_to_multiple_of)
        if multiple <= 0:
            raise ValueError("pad_to_multiple_of must be positive")
        max_length = ((max_length + multiple - 1) // multiple) * multiple

    batch_size = len(prepared)
    image_tokens_per_img = int(batch[0]["image_token_ids"].numel())
    input_ids = torch.full(
        (batch_size, max_length), int(pad_token_id), dtype=torch.long
    )
    token_types = torch.full(
        (batch_size, max_length), PADDING_TOKEN_TYPE, dtype=torch.uint8
    )
    labels = torch.full(
        (batch_size, max_length), IGNORE_INDEX, dtype=torch.long
    )
    image_token_mask = torch.zeros(
        (batch_size, max_length), dtype=torch.bool
    )
    masked_image_positions = torch.zeros_like(image_token_mask)
    lengths = torch.empty(batch_size, dtype=torch.long)
    image_starts = torch.empty(batch_size, dtype=torch.long)
    condition_dropped = torch.empty(batch_size, dtype=torch.bool)
    mask_ratios = torch.empty(batch_size, dtype=torch.float32)
    actual_mask_ratios = torch.empty(batch_size, dtype=torch.float32)
    target_image_ids = torch.empty(
        (batch_size, image_tokens_per_img), dtype=torch.long
    )

    for row, (
        item,
        row_input_ids,
        row_token_types,
        image_start,
        row_condition_dropped,
    ) in enumerate(prepared):
        row_image_tokens = item["image_token_ids"].long()
        if row_image_tokens.numel() != image_tokens_per_img:
            raise ValueError("all samples in a batch must have the same image length")
        image_end = image_start + image_tokens_per_img
        length = int(row_input_ids.numel())
        if image_end > length:
            raise ValueError("image span exceeds sequence length")
        expected_targets = row_input_ids[image_start:image_end].clone()

        row_generator = generator
        if mask_seed is not None:
            row_generator = torch.Generator()
            sample_id = int(item["sample_id"])
            row_generator.manual_seed(
                (int(mask_seed) + sample_id * 1_000_003) % (2**63 - 1)
            )
        if fixed_mask_ratio is None:
            timestep = torch.rand((), generator=row_generator)
            requested_ratio = torch.cos(timestep * math.pi * 0.5).clamp_min(
                float(min_masking_rate)
            )
        else:
            requested_ratio = torch.tensor(float(fixed_mask_ratio))
        num_masked = int(
            torch.round(requested_ratio * image_tokens_per_img)
            .clamp(min=1, max=image_tokens_per_img)
            .item()
        )
        order = torch.randperm(image_tokens_per_img, generator=row_generator)
        selected = order[:num_masked] + image_start

        input_ids[row, :length] = row_input_ids
        token_types[row, :length] = row_token_types
        input_ids[row, selected] = int(image_mask_token_id)
        labels[row, selected] = expected_targets[order[:num_masked]]
        image_token_mask[row, image_start:image_end] = True
        masked_image_positions[row, selected] = True
        lengths[row] = length
        image_starts[row] = image_start
        condition_dropped[row] = row_condition_dropped
        mask_ratios[row] = requested_ratio
        actual_mask_ratios[row] = num_masked / image_tokens_per_img
        target_image_ids[row] = row_image_tokens

    attention_mask = build_showo_omni_attention_mask(
        token_types=token_types,
        lengths=lengths,
        dtype=attention_dtype,
    )
    return {
        "input_ids": input_ids,
        "token_types": token_types,
        "labels": labels,
        "attention_mask": attention_mask,
        "image_token_mask": image_token_mask,
        "masked_image_positions": masked_image_positions,
        "lengths": lengths,
        "image_starts": image_starts,
        "mask_ratios": mask_ratios,
        "actual_mask_ratios": actual_mask_ratios,
        "condition_dropped": condition_dropped,
        "target_image_ids": target_image_ids,
        "sample_ids": torch.stack([item["sample_id"] for item in batch]).long(),
        "class_ids": torch.stack([item["class_id"] for item in batch]).long(),
        "class_names": [str(item["class_name"]) for item in batch],
        "synsets": [str(item["synset"]) for item in batch],
    }


def _split_key_for_index(dataset: QwenShowOImageNetDataset, index: int) -> str:
    image_id = int(dataset.img_ids[int(index)].item())
    return dataset.synsets.get(image_id, "")


def _build_split_indices(
    dataset: QwenShowOImageNetDataset,
    val_ratio: float,
    seed: int,
    strategy: str,
    val_samples_per_class: Optional[int] = None,
) -> Tuple[List[int], List[int]]:
    """Mirror ``dataset_imagenet_flow_cache._build_split_indices`` exactly."""

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
            f"Unknown ImageNet VQ split_strategy={strategy!r}; "
            "expected stratified, shuffled, or contiguous."
        )

    groups: Dict[str, List[int]] = {}
    for index in range(n_items):
        key = _split_key_for_index(dataset, index)
        groups.setdefault(key, []).append(index)
    if len(groups) <= 1:
        indices = list(range(n_items))
        rng.shuffle(indices)
        single_group_val_size = (
            fixed_val_size if fixed_val_size is not None else val_size
        )
        single_group_val_size = min(
            single_group_val_size, max(1, n_items - 1)
        )
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
            group_val_size = max(
                1, int(len(group_indices) * float(val_ratio))
            )
        if len(group_indices) > 1:
            group_val_size = min(group_val_size, len(group_indices) - 1)
        else:
            group_val_size = 0
        val_indices.extend(group_indices[:group_val_size])
        train_indices.extend(group_indices[group_val_size:])

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return train_indices, val_indices


def _build_explicit_split_indices(
    dataset: QwenShowOImageNetDataset,
    split_manifest_jsonl: str,
    val_samples_per_class: Optional[int] = None,
) -> Tuple[List[int], List[int]]:
    """Resolve authoritative split membership by image ID.

    A validation-only file is sufficient; its complement becomes training.
    Alternatively, a file may enumerate both train and validation rows, in
    which case it must cover the complete packed cache. Validation row order
    is defined by contiguous ``split_index`` values (``evaluation_index`` is
    accepted as a validation-only alias); otherwise file order is preserved.
    """

    path = Path(split_manifest_jsonl)
    if not path.exists():
        raise FileNotFoundError(path)
    index_by_image_id = {
        int(image_id): index
        for index, image_id in enumerate(dataset.img_ids.tolist())
    }
    seen_image_ids = set()
    train_rows: List[Tuple[Optional[int], int, int]] = []
    val_rows: List[Tuple[Optional[int], int, int]] = []
    has_explicit_train = False
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = [
                key for key in ("img_id", "synset", "split") if key not in row
            ]
            if missing:
                raise ValueError(
                    f"{path}:{line_number} is missing required fields {missing}"
                )
            image_id = int(row["img_id"])
            if image_id in seen_image_ids:
                raise ValueError(f"duplicate img_id={image_id} in {path}")
            if image_id not in index_by_image_id:
                raise ValueError(
                    f"{path}:{line_number} references img_id={image_id} "
                    "which is absent from the packed cache"
                )
            seen_image_ids.add(image_id)
            dataset_index = index_by_image_id[image_id]
            expected_synset = dataset.synsets[image_id]
            actual_synset = str(row["synset"])
            if actual_synset != expected_synset:
                raise ValueError(
                    f"{path}:{line_number} synset mismatch for img_id={image_id}: "
                    f"{actual_synset!r} != {expected_synset!r}"
                )
            split = str(row["split"]).strip().lower()
            if split in {"validation", "val"}:
                declared_index = row.get(
                    "split_index", row.get("evaluation_index")
                )
                val_rows.append(
                    (
                        int(declared_index)
                        if declared_index is not None
                        else None,
                        dataset_index,
                        line_number,
                    )
                )
            elif split == "train":
                has_explicit_train = True
                declared_index = row.get("split_index")
                train_rows.append(
                    (
                        int(declared_index)
                        if declared_index is not None
                        else None,
                        dataset_index,
                        line_number,
                    )
                )
            else:
                raise ValueError(
                    f"{path}:{line_number} has unsupported split={split!r}"
                )
    if not val_rows:
        raise ValueError(f"{path} contains no validation rows")

    def ordered_indices(
        rows: List[Tuple[Optional[int], int, int]],
        split_name: str,
    ) -> List[int]:
        has_indices = [declared is not None for declared, _, _ in rows]
        if any(has_indices) and not all(has_indices):
            raise ValueError(
                f"{path}: either every {split_name} row or no {split_name} "
                "row must contain split_index"
            )
        if all(has_indices):
            rows = sorted(rows, key=lambda values: int(values[0]))
            for expected, (declared, _, line_number) in enumerate(rows):
                if int(declared) != expected:
                    raise ValueError(
                        f"{path}:{line_number} {split_name} split_index="
                        f"{declared}, expected {expected}"
                    )
        return [dataset_index for _, dataset_index, _ in rows]

    val_indices = ordered_indices(val_rows, "validation")
    train_indices = ordered_indices(train_rows, "train")
    if has_explicit_train:
        missing_ids = set(index_by_image_id).difference(seen_image_ids)
        if missing_ids:
            raise ValueError(
                "an explicit train+validation split must cover the complete "
                f"packed cache; missing {len(missing_ids)} image IDs"
            )
    else:
        validation_set = set(val_indices)
        train_indices = [
            index
            for index in range(len(dataset))
            if index not in validation_set
        ]

    if val_samples_per_class is not None:
        expected_count = int(val_samples_per_class)
        val_counts: Dict[str, int] = {}
        for index in val_indices:
            synset = _split_key_for_index(dataset, index)
            val_counts[synset] = val_counts.get(synset, 0) + 1
        dataset_synsets = set(dataset.synsets.values())
        if set(val_counts) != dataset_synsets:
            missing_synsets = sorted(dataset_synsets.difference(val_counts))
            raise ValueError(
                f"{path} validation split omits classes: {missing_synsets[:10]}"
            )
        wrong_counts = {
            synset: count
            for synset, count in val_counts.items()
            if count != expected_count
        }
        if wrong_counts:
            raise ValueError(
                f"{path} must contain exactly {expected_count} validation "
                f"samples per class; mismatches={dict(list(wrong_counts.items())[:10])}"
            )
    return train_indices, val_indices


def build_qwen_showo_imagenet_dataloaders(config: Any, tokenizer: Any):
    """Build the 115k/10k train/validation ImageNet-100 loaders."""

    params = config.dataset.params
    model = config.model
    training = config.training
    dataset = QwenShowOImageNetDataset(
        tokens_path=_config_get(
            params, "tokens_path", _config_get(params, "cache_path")
        ),
        manifest_jsonl=_config_get(params, "manifest_jsonl"),
        synset_mapping_path=_config_get(params, "synset_mapping_path"),
        tokenizer=tokenizer,
        t2i_token_id=_config_get(model, "t2i_token_id"),
        boi_token_id=_config_get(model, "boi_token_id"),
        eoi_token_id=_config_get(model, "eoi_token_id"),
        eos_token_id=tokenizer.eos_token_id,
        image_offset=_config_get(model, "image_offset"),
        image_vocab_size=int(_config_get(model, "image_vocab_size", 8192)),
        image_tokens_per_img=int(
            _config_get(
                params,
                "image_tokens_per_img",
                _config_get(model, "image_tokens_per_img", 256),
            )
        ),
        t2i_prefix=_config_get(params, "t2i_prefix", ""),
        class_prompt_template=_config_get(
            params, "class_prompt_template", "{class_name}"
        ),
        max_text_tokens=_config_get(params, "max_text_tokens"),
        max_seq_length=_config_get(
            params,
            "max_seq_length",
            _config_get(_config_get(config.dataset, "preprocessing"), "max_seq_length"),
        ),
        max_samples=int(_config_get(params, "max_samples", -1)),
        mmap=bool(_config_get(params, "mmap", True)),
    )

    split_seed = int(_config_get(params, "split_seed", 42))
    val_samples_per_class = int(
        _config_get(params, "val_samples_per_class", 100)
    )
    split_manifest_jsonl = _config_get(params, "split_manifest_jsonl")
    if split_manifest_jsonl:
        train_indices, val_indices = _build_explicit_split_indices(
            dataset=dataset,
            split_manifest_jsonl=str(split_manifest_jsonl),
            val_samples_per_class=val_samples_per_class,
        )
    else:
        train_indices, val_indices = _build_split_indices(
            dataset=dataset,
            val_ratio=float(_config_get(params, "val_ratio", 0.08)),
            seed=split_seed,
            strategy=str(_config_get(params, "split_strategy", "stratified")),
            val_samples_per_class=val_samples_per_class,
        )
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    pad_to_length = _config_get(params, "pad_to_length")
    pad_to_multiple_of = _config_get(params, "pad_to_multiple_of", 64)
    cond_dropout_prob = float(
        _config_get(
            params,
            "cond_dropout_prob",
            _config_get(training, "cond_dropout_prob", 0.1),
        )
    )
    min_masking_rate = float(
        _config_get(
            params,
            "min_masking_rate",
            _config_get(training, "min_masking_rate", 0.0),
        )
    )
    evaluation = _config_get(config, "evaluation")
    val_mask_ratio = float(
        _config_get(
            evaluation,
            "image_mask_ratio",
            _config_get(params, "val_mask_ratio", 0.75),
        )
    )
    mixed_precision = str(
        _config_get(training, "mixed_precision", "no")
    ).lower()
    if mixed_precision == "bf16":
        attention_dtype = torch.bfloat16
    elif mixed_precision == "fp16":
        attention_dtype = torch.float16
    else:
        attention_dtype = torch.float32
    common_collate = dict(
        pad_token_id=int(pad_token_id),
        image_mask_token_id=int(_config_get(model, "image_mask_token_id")),
        pad_to_length=pad_to_length,
        pad_to_multiple_of=pad_to_multiple_of,
        attention_dtype=attention_dtype,
    )
    train_collate = partial(
        collate_qwen_showo_imagenet,
        cond_dropout_prob=cond_dropout_prob,
        min_masking_rate=min_masking_rate,
        fixed_mask_ratio=None,
        mask_seed=None,
        **common_collate,
    )
    val_collate = partial(
        collate_qwen_showo_imagenet,
        cond_dropout_prob=0.0,
        min_masking_rate=min_masking_rate,
        fixed_mask_ratio=val_mask_ratio,
        mask_seed=int(_config_get(params, "val_mask_seed", split_seed)),
        **common_collate,
    )

    workers = int(_config_get(training, "dataloader_workers", 0))
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(_config_get(training, "batch_size")),
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=workers > 0,
        collate_fn=train_collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(
            _config_get(
                training,
                "val_batch_size",
                _config_get(training, "batch_size"),
            )
        ),
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=workers > 0,
        collate_fn=val_collate,
    )
    return train_loader, val_loader


def _generation_item_metadata(item: Any) -> Tuple[str, str, Optional[int]]:
    if isinstance(item, str):
        return item, "", None
    if isinstance(item, Mapping):
        return (
            str(item["class_name"]),
            str(item.get("synset", "")),
            int(item["sample_id"]) if item.get("sample_id") is not None else None,
        )
    return (
        str(getattr(item, "class_name")),
        str(getattr(item, "synset", "")),
        (
            int(getattr(item, "sample_id"))
            if getattr(item, "sample_id", None) is not None
            else None
        ),
    )


def build_qwen_showo_generation_batch(
    class_names_or_items: Sequence[Any],
    tokenizer: Any,
    *,
    t2i_token_id: int,
    boi_token_id: int,
    eoi_token_id: int,
    image_mask_token_id: int,
    image_tokens_per_img: int = 256,
    eos_token_id: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    class_prompt_template: str = "{class_name}",
    t2i_prefix: str = "",
    max_text_tokens: Optional[int] = None,
    pad_to_multiple_of: Optional[int] = 64,
    attention_dtype: torch.dtype = torch.float32,
) -> Dict[str, Any]:
    """Build paired conditional/unconditional all-mask inputs for CFG sampling."""

    if not class_names_or_items:
        raise ValueError("class_names_or_items must not be empty")
    eos_token_id = (
        int(tokenizer.eos_token_id)
        if eos_token_id is None
        else int(eos_token_id)
    )
    if pad_token_id is None:
        pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = eos_token_id

    metadata = [_generation_item_metadata(item) for item in class_names_or_items]

    def make_sequence(class_name: str, conditional: bool):
        class_prompt = (
            QwenShowOImageNetDataset._format_class_prompt(
                class_prompt_template, class_name
            )
            if conditional
            else ""
        )
        prompt = QwenShowOImageNetDataset._join_prompt(
            t2i_prefix, class_prompt
        )
        text_ids = (
            tokenizer.encode(prompt, add_special_tokens=False) if prompt else []
        )
        if max_text_tokens is not None:
            text_ids = text_ids[: int(max_text_tokens)]
        text_tensor = torch.tensor(text_ids, dtype=torch.long)
        ids = torch.cat(
            [
                torch.tensor([int(t2i_token_id)], dtype=torch.long),
                text_tensor,
                torch.tensor([int(boi_token_id)], dtype=torch.long),
                torch.full(
                    (int(image_tokens_per_img),),
                    int(image_mask_token_id),
                    dtype=torch.long,
                ),
                torch.tensor(
                    [int(eoi_token_id), eos_token_id], dtype=torch.long
                ),
            ]
        )
        types = torch.cat(
            [
                torch.tensor([SPECIAL_TOKEN_TYPE], dtype=torch.uint8),
                torch.full(
                    (text_tensor.numel(),), TEXT_TOKEN_TYPE, dtype=torch.uint8
                ),
                torch.tensor([SPECIAL_TOKEN_TYPE], dtype=torch.uint8),
                torch.full(
                    (int(image_tokens_per_img),),
                    IMAGE_TOKEN_TYPE,
                    dtype=torch.uint8,
                ),
                torch.tensor(
                    [SPECIAL_TOKEN_TYPE, SPECIAL_TOKEN_TYPE], dtype=torch.uint8
                ),
            ]
        )
        return ids, types, 2 + int(text_tensor.numel())

    conditional = [make_sequence(name, True) for name, _, _ in metadata]
    unconditional = [make_sequence(name, False) for name, _, _ in metadata]
    max_length = max(
        sequence[0].numel() for sequence in conditional + unconditional
    )
    if pad_to_multiple_of:
        multiple = int(pad_to_multiple_of)
        max_length = ((max_length + multiple - 1) // multiple) * multiple

    def pad_sequences(sequences):
        batch_size = len(sequences)
        ids = torch.full(
            (batch_size, max_length), int(pad_token_id), dtype=torch.long
        )
        types = torch.full(
            (batch_size, max_length), PADDING_TOKEN_TYPE, dtype=torch.uint8
        )
        lengths = torch.empty(batch_size, dtype=torch.long)
        image_starts = torch.empty(batch_size, dtype=torch.long)
        image_mask = torch.zeros((batch_size, max_length), dtype=torch.bool)
        for row, (row_ids, row_types, image_start) in enumerate(sequences):
            length = int(row_ids.numel())
            ids[row, :length] = row_ids
            types[row, :length] = row_types
            lengths[row] = length
            image_starts[row] = image_start
            image_mask[
                row, image_start : image_start + int(image_tokens_per_img)
            ] = True
        attention = build_showo_omni_attention_mask(
            token_types=types, lengths=lengths, dtype=attention_dtype
        )
        return ids, types, lengths, image_starts, image_mask, attention

    (
        conditional_ids,
        conditional_types,
        conditional_lengths,
        conditional_starts,
        conditional_image_mask,
        conditional_attention,
    ) = pad_sequences(conditional)
    (
        unconditional_ids,
        unconditional_types,
        unconditional_lengths,
        unconditional_starts,
        unconditional_image_mask,
        unconditional_attention,
    ) = pad_sequences(unconditional)
    return {
        "input_ids": conditional_ids,
        "conditional_input_ids": conditional_ids,
        "token_types": conditional_types,
        "conditional_token_types": conditional_types,
        "attention_mask": conditional_attention,
        "conditional_attention_mask": conditional_attention,
        "image_token_mask": conditional_image_mask,
        "conditional_image_token_mask": conditional_image_mask,
        "lengths": conditional_lengths,
        "image_starts": conditional_starts,
        "uncond_input_ids": unconditional_ids,
        "unconditional_input_ids": unconditional_ids,
        "uncond_token_types": unconditional_types,
        "unconditional_token_types": unconditional_types,
        "uncond_attention_mask": unconditional_attention,
        "unconditional_attention_mask": unconditional_attention,
        "uncond_image_token_mask": unconditional_image_mask,
        "unconditional_image_token_mask": unconditional_image_mask,
        "uncond_lengths": unconditional_lengths,
        "uncond_image_starts": unconditional_starts,
        "class_names": [name for name, _, _ in metadata],
        "synsets": [synset for _, synset, _ in metadata],
        "sample_ids": [sample_id for _, _, sample_id in metadata],
    }
