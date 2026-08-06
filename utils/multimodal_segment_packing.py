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
from utils.imagenet_flow_sequence import build_selfless_sigma, scalar_int


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def is_power_of_two(value: int) -> bool:
    value = int(value)
    return value > 0 and value & (value - 1) == 0


def next_power_of_two(value: int) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"value must be positive, got {value}")
    return 1 << (value - 1).bit_length()


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
    if not is_power_of_two(capacity):
        raise ValueError(
            "nominal_capacity must be a positive power of two, "
            f"got {capacity}"
        )
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
            next_power_of_two(length),
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


def _cfg_dropout_for_item(
    item: Mapping[str, Any],
    probability: float,
) -> bool:
    if probability <= 0.0:
        return False
    if probability >= 1.0:
        return True
    seed = scalar_int(item, "cfg_dropout_seed")
    return random.Random(seed).random() < probability


def collate_segment_packed(
    batch: list[dict[str, Any]],
    *,
    nominal_capacity: int = 2048,
    image_uncond_prob: float = 0.0,
    emit_audit_manifest: bool = True,
) -> dict[str, Any]:
    """Pack complete text-image samples into block-diagonal physical rows."""

    if not batch:
        raise ValueError("cannot pack an empty batch")
    lengths = [int(item["input_ids"].shape[0]) for item in batch]
    img_ids = [scalar_int(item, "img_id") for item in batch]
    rows = deterministic_best_fit_decreasing(
        lengths,
        img_ids,
        nominal_capacity=nominal_capacity,
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
    image_local_positions = torch.full(
        (row_count, physical_length), -1, dtype=torch.long
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
            sigma[row_index, cursor:end] = build_selfless_sigma(
                item,
                image_tokens=image_tokens,
            )
            local_position_ids = build_row_col_position_ids(
                item["token_types"].unsqueeze(0),
                image_tokens,
            )
            position_ids[:, row_index, cursor:end] = local_position_ids[:, 0]

            image_start = cursor + scalar_int(item, "image_start")
            image_end = image_start + image_tokens
            image_latents[row_index, image_start:image_end] = item[
                "image_latents"
            ]
            image_local_positions[row_index, image_start:image_end] = torch.arange(
                image_tokens,
                dtype=torch.long,
            )
            if _cfg_dropout_for_item(item, float(image_uncond_prob)):
                image_uncond_mask[row_index, image_start:image_end] = True

            img_id = img_ids[sample_index]
            span_rows.append(
                [row_index, segment_id, image_start, image_end, img_id]
            )
            if emit_audit_manifest:
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
        if emit_audit_manifest:
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
    result = {
        "input_ids": input_ids,
        "token_types": token_types,
        "sigma": sigma,
        "position_ids": position_ids,
        "image_local_positions": image_local_positions,
        "segment_ids": segment_ids,
        "labels": labels,
        "image_latents": image_latents,
        "image_span_table": torch.tensor(span_rows, dtype=torch.long),
        "image_uncond_mask": image_uncond_mask,
        "valid_token_count": valid_tokens,
        "padding_token_count": padding_tokens,
        "image_count": len(batch),
        "pack_capacity": int(nominal_capacity),
        "sample_img_ids": img_ids,
        "pack_stats": (
            valid_tokens,
            image_token_count,
            padding_tokens,
            physical_length,
        ),
        "pack_details": (
            len(batch),
            row_count,
            int(nominal_capacity),
            sum(int(row.overflow) for row in rows),
        ),
    }
    if emit_audit_manifest:
        manifest_payload = {
            "schema": "selfless_segment_pack_batch_v2",
            "nominal_capacity": int(nominal_capacity),
            "overflow_policy": "dedicated_next_power_of_two",
            "physical_length": physical_length,
            "rows": pack_rows,
        }
        result.update(
            {
                "pack_manifest_sha256": canonical_sha256(manifest_payload),
                "pack_manifest": manifest_payload,
                "sample_token_sha256": sample_token_hashes,
                "augmentation_sha256": augmentation_hashes,
            }
        )
    return result
