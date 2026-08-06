"""Unpacked batch construction for ImageNet flow validation/generation."""

from __future__ import annotations

from typing import Any, Optional

import torch

from models.modeling_model.image_position_utils import (
    build_row_col_position_ids,
)
from utils.imagenet_flow_sequence import build_selfless_sigma, scalar_int


def collate_imagenet_flow_cache(
    batch: list[dict[str, Any]],
    pad_to_length: Optional[int] = None,
    pad_to_multiple_of: Optional[int] = None,
) -> dict[str, Any]:
    """Keep one logical sample per row for validation and generation."""

    if not batch:
        raise ValueError("cannot collate an empty batch")
    batch_max_len = max(item["input_ids"].shape[0] for item in batch)
    max_len = int(pad_to_length or batch_max_len)
    if pad_to_multiple_of and max_len % int(pad_to_multiple_of):
        multiple = int(pad_to_multiple_of)
        max_len = ((max_len + multiple - 1) // multiple) * multiple
    if batch_max_len > max_len:
        raise ValueError(
            f"Batch max length {batch_max_len} exceeds pad_to_length={max_len}"
        )

    batch_size = len(batch)
    image_tokens, latent_dim = batch[0]["image_latents"].shape
    latent_dtype = batch[0]["image_latents"].dtype
    input_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
    token_types = torch.full(
        (batch_size, max_len), 3, dtype=torch.uint8
    )
    sigma = torch.full(
        (batch_size, max_len), max_len, dtype=torch.long
    )
    labels = torch.full(
        (batch_size, max_len), -100, dtype=torch.long
    )
    image_latents = torch.zeros(
        batch_size,
        max_len,
        latent_dim,
        dtype=latent_dtype,
    )
    image_local_positions = torch.full(
        (batch_size, max_len), -1, dtype=torch.long
    )
    image_span_rows: list[list[int]] = []

    for row_index, item in enumerate(batch):
        if tuple(item["image_latents"].shape) != (
            image_tokens,
            latent_dim,
        ):
            raise ValueError(
                "all samples in a batch must share the same latent shape"
            )
        length = int(item["input_ids"].shape[0])
        image_start = scalar_int(item, "image_start")
        image_end = image_start + image_tokens

        input_ids[row_index, :length] = item["input_ids"]
        token_types[row_index, :length] = item["token_types"]
        labels[row_index, :length] = item["labels"]
        image_latents[row_index, image_start:image_end] = item[
            "image_latents"
        ]
        image_local_positions[
            row_index, image_start:image_end
        ] = torch.arange(image_tokens, dtype=torch.long)
        image_span_rows.append(
            [
                row_index,
                0,
                image_start,
                image_end,
                scalar_int(item, "img_id"),
            ]
        )
        sigma[row_index, :length] = build_selfless_sigma(
            item,
            image_tokens=image_tokens,
        )

    valid_tokens = (token_types != 3).sum()
    image_token_count = (token_types == 1).sum()
    padding_tokens = (token_types == 3).sum()
    return {
        "input_ids": input_ids,
        "token_types": token_types,
        "sigma": sigma,
        "position_ids": build_row_col_position_ids(
            token_types, image_tokens
        ),
        "image_local_positions": image_local_positions,
        "image_span_table": torch.tensor(
            image_span_rows, dtype=torch.long
        ),
        "labels": labels,
        "image_latents": image_latents,
        "pack_stats": (
            int(valid_tokens),
            int(image_token_count),
            int(padding_tokens),
            int(max_len),
        ),
    }
