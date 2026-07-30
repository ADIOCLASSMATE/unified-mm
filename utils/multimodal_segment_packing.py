"""Deterministic segment-aware packing for multimodal Selfless-Flow batches."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

from models.modeling_model.image_position_utils import (
    build_row_col_position_ids,
)


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def round_up(value: int, multiple: int) -> int:
    value = int(value)
    multiple = int(multiple)
    if value <= 0 or multiple <= 0:
        raise ValueError(
            f"value and multiple must be positive, got {value}, {multiple}"
        )
    return ((value + multiple - 1) // multiple) * multiple


@dataclass(frozen=True)
class PackedRow:
    sample_indices: tuple[int, ...]
    padded_length: int
    overflow: bool

    @property
    def used_length(self) -> int:
        raise AttributeError(
            "PackedRow.used_length depends on the source lengths; use "
            "row_used_length(row, lengths)."
        )


def row_used_length(row: PackedRow, lengths: Sequence[int]) -> int:
    return sum(int(lengths[index]) for index in row.sample_indices)


def deterministic_best_fit_decreasing(
    lengths: Sequence[int],
    img_ids: Sequence[int],
    *,
    nominal_capacity: int,
    overflow_multiple: int = 128,
) -> list[PackedRow]:
    """Pack indivisible samples with deterministic best-fit-decreasing.

    Normal samples are sorted by ``(-length, img_id, source_index)``. Each is
    placed into the row with the smallest remaining capacity that still fits;
    ties use row creation order. Oversize samples receive dedicated rows.
    Finally, rows are sorted by their minimum image id for deterministic replay.
    """

    if len(lengths) != len(img_ids):
        raise ValueError(
            f"lengths/img_ids mismatch: {len(lengths)} != {len(img_ids)}"
        )
    capacity = int(nominal_capacity)
    if capacity <= 0:
        raise ValueError(f"nominal_capacity must be positive, got {capacity}")
    normalized_lengths = [int(value) for value in lengths]
    normalized_img_ids = [int(value) for value in img_ids]
    if any(length <= 0 for length in normalized_lengths):
        raise ValueError(f"all segment lengths must be positive: {lengths}")
    if len(set(normalized_img_ids)) != len(normalized_img_ids):
        raise ValueError("img_ids must be unique within one packed microbatch")

    normal = sorted(
        (
            (index, normalized_lengths[index], normalized_img_ids[index])
            for index in range(len(normalized_lengths))
            if normalized_lengths[index] <= capacity
        ),
        key=lambda item: (-item[1], item[2], item[0]),
    )
    overflow = sorted(
        (
            (index, normalized_lengths[index], normalized_img_ids[index])
            for index in range(len(normalized_lengths))
            if normalized_lengths[index] > capacity
        ),
        key=lambda item: (item[2], item[0]),
    )

    mutable_rows: list[list[int]] = []
    remaining: list[int] = []
    for index, length, _ in normal:
        candidates = [
            (space - length, row_index)
            for row_index, space in enumerate(remaining)
            if space >= length
        ]
        if candidates:
            _, row_index = min(candidates)
        else:
            row_index = len(mutable_rows)
            mutable_rows.append([])
            remaining.append(capacity)
        mutable_rows[row_index].append(index)
        remaining[row_index] -= length

    rows = [
        PackedRow(tuple(indices), capacity, False)
        for indices in mutable_rows
    ]
    rows.extend(
        PackedRow(
            (index,),
            round_up(length, overflow_multiple),
            True,
        )
        for index, length, _ in overflow
    )
    rows.sort(
        key=lambda row: (
            min(normalized_img_ids[index] for index in row.sample_indices),
            row.overflow,
            row.sample_indices,
        )
    )
    return rows


def _segment_sigma(
    item: Mapping[str, Any],
    *,
    image_tokens: int,
) -> torch.Tensor:
    length = int(item["input_ids"].shape[0])
    prompt_len = int(item["prompt_len"].item())
    suffix_len = int(item["suffix_len"].item())
    image_start = int(item["image_start"].item())
    eoi_pos = image_start + image_tokens
    suffix_start = eoi_pos + 1
    eos_pos = length - 1
    if eoi_pos >= length or eos_pos < eoi_pos:
        raise ValueError(
            f"invalid image span for segment length={length}: "
            f"start={image_start}, image_tokens={image_tokens}"
        )

    sigma = torch.empty(length, dtype=torch.long)
    if prompt_len:
        sigma[:prompt_len] = torch.arange(prompt_len, dtype=torch.long)
    sigma[prompt_len] = prompt_len
    sigma[eoi_pos] = prompt_len + 1

    reveal_seed = int(item["reveal_seed"].item())
    generator = torch.Generator()
    generator.manual_seed(reveal_seed)
    order = torch.rand(image_tokens, generator=generator).argsort()
    sigma[image_start:eoi_pos] = prompt_len + 2 + order

    suffix_sigma_start = prompt_len + image_tokens + 2
    if suffix_len:
        sigma[suffix_start:eos_pos] = (
            suffix_sigma_start + torch.arange(suffix_len, dtype=torch.long)
        )
    sigma[eos_pos] = suffix_sigma_start + suffix_len
    return sigma


def _cfg_dropout_for_item(
    item: Mapping[str, Any],
    probability: float,
) -> bool:
    if probability <= 0.0:
        return False
    if probability >= 1.0:
        return True
    seed = int(item["cfg_dropout_seed"].item())
    return random.Random(seed).random() < probability


def collate_segment_packed(
    batch: list[dict[str, Any]],
    *,
    nominal_capacity: int = 2048,
    overflow_multiple: int = 128,
    image_uncond_prob: float = 0.0,
) -> dict[str, Any]:
    """Pack complete text-image samples into block-diagonal physical rows."""

    if not batch:
        raise ValueError("cannot pack an empty batch")
    lengths = [int(item["input_ids"].shape[0]) for item in batch]
    img_ids = [int(item["img_id"].item()) for item in batch]
    rows = deterministic_best_fit_decreasing(
        lengths,
        img_ids,
        nominal_capacity=nominal_capacity,
        overflow_multiple=overflow_multiple,
    )
    if not rows:
        raise ValueError("packing produced no rows")

    image_tokens = int(batch[0]["image_latents"].shape[0])
    latent_dim = int(batch[0]["image_latents"].shape[-1])
    latent_dtype = batch[0]["image_latents"].dtype
    for item in batch:
        if tuple(item["image_latents"].shape) != (
            image_tokens,
            latent_dim,
        ):
            raise ValueError("all samples in a packed batch must share latent shape")

    physical_length = max(row.padded_length for row in rows)
    row_count = len(rows)
    input_ids = torch.zeros(row_count, physical_length, dtype=torch.long)
    token_types = torch.full(
        (row_count, physical_length), 3, dtype=torch.uint8
    )
    sigma = torch.zeros(row_count, physical_length, dtype=torch.long)
    labels = torch.full(
        (row_count, physical_length), -100, dtype=torch.long
    )
    image_latents = torch.zeros(
        row_count,
        physical_length,
        latent_dim,
        dtype=latent_dtype,
    )
    segment_ids = torch.full(
        (row_count, physical_length), -1, dtype=torch.long
    )
    position_ids = torch.zeros(
        2, row_count, physical_length, dtype=torch.long
    )
    image_uncond_mask = torch.zeros(
        row_count, physical_length, dtype=torch.bool
    )

    span_rows: list[list[int]] = []
    pack_rows: list[dict[str, Any]] = []
    sample_token_hashes: list[str] = []
    augmentation_hashes: list[str] = []
    for row_index, row in enumerate(rows):
        cursor = 0
        segment_records = []
        for segment_id, sample_index in enumerate(row.sample_indices):
            item = batch[sample_index]
            length = lengths[sample_index]
            end = cursor + length
            if end > row.padded_length:
                raise AssertionError(
                    f"row overflowed its plan: {end} > {row.padded_length}"
                )
            input_ids[row_index, cursor:end] = item["input_ids"]
            token_types[row_index, cursor:end] = item["token_types"]
            labels[row_index, cursor:end] = item["labels"]
            segment_ids[row_index, cursor:end] = segment_id
            sigma[row_index, cursor:end] = _segment_sigma(
                item,
                image_tokens=image_tokens,
            )
            local_position_ids = build_row_col_position_ids(
                item["token_types"].unsqueeze(0),
                image_tokens,
            )
            position_ids[:, row_index, cursor:end] = local_position_ids[:, 0]

            image_start = cursor + int(item["image_start"].item())
            image_end = image_start + image_tokens
            image_latents[row_index, image_start:image_end] = item[
                "image_latents"
            ]
            if _cfg_dropout_for_item(item, float(image_uncond_prob)):
                image_uncond_mask[row_index, image_start:image_end] = True

            img_id = img_ids[sample_index]
            span_rows.append(
                [row_index, segment_id, image_start, image_end, img_id]
            )
            token_hash = str(item["token_ids_sha256"])
            augmentation_hash = str(item["augmentation_sha256"])
            sample_token_hashes.append(token_hash)
            augmentation_hashes.append(augmentation_hash)
            segment_records.append(
                {
                    "segment_id": segment_id,
                    "img_id": img_id,
                    "source_index": int(sample_index),
                    "start": cursor,
                    "end": end,
                    "serialized_length": length,
                    "token_ids_sha256": token_hash,
                    "augmentation_sha256": augmentation_hash,
                }
            )
            cursor = end
        pack_rows.append(
            {
                "row": row_index,
                "padded_length": row.padded_length,
                "used_length": cursor,
                "overflow": row.overflow,
                "segments": segment_records,
            }
        )

    valid_tokens = int((segment_ids >= 0).sum().item())
    image_token_count = int((token_types == 1).sum().item())
    padding_tokens = int(token_types.numel() - valid_tokens)
    manifest_payload = {
        "schema": "selfless_segment_pack_batch_v1",
        "nominal_capacity": int(nominal_capacity),
        "overflow_multiple": int(overflow_multiple),
        "physical_length": physical_length,
        "rows": pack_rows,
    }
    manifest_sha256 = canonical_sha256(manifest_payload)
    return {
        "input_ids": input_ids,
        "token_types": token_types,
        "sigma": sigma,
        "position_ids": position_ids,
        "segment_ids": segment_ids,
        "labels": labels,
        "image_latents": image_latents,
        "image_span_table": torch.tensor(span_rows, dtype=torch.long),
        "image_uncond_mask": image_uncond_mask,
        "valid_token_count": torch.tensor(valid_tokens, dtype=torch.long),
        "padding_token_count": torch.tensor(
            padding_tokens, dtype=torch.long
        ),
        "image_count": torch.tensor(len(batch), dtype=torch.long),
        "pack_capacity": torch.tensor(
            int(nominal_capacity), dtype=torch.long
        ),
        "pack_manifest_sha256": manifest_sha256,
        "pack_manifest": manifest_payload,
        "sample_img_ids": torch.tensor(img_ids, dtype=torch.long),
        "sample_token_sha256": sample_token_hashes,
        "augmentation_sha256": augmentation_hashes,
        "pack_stats": torch.tensor(
            [
                valid_tokens,
                image_token_count,
                padding_tokens,
                physical_length,
            ],
            dtype=torch.long,
        ),
        "pack_details": torch.tensor(
            [
                len(batch),
                row_count,
                int(nominal_capacity),
                sum(int(row.overflow) for row in rows),
            ],
            dtype=torch.long,
        ),
    }
