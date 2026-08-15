"""Attention and masking contracts for the Show-o2-style ablation."""

from __future__ import annotations

import math

import torch
from torch.nn.attention.flex_attention import create_block_mask


def image_span_ids(token_types: torch.Tensor) -> torch.Tensor:
    """Assign a positive id to each contiguous image-latent span."""
    image = token_types.eq(1)
    previous_image = torch.zeros_like(image)
    previous_image[:, 1:] = image[:, :-1]
    starts = image & ~previous_image
    return torch.cumsum(starts.to(torch.long), dim=1) * image.to(torch.long)


def build_showo2_position_ids(
    token_types: torch.Tensor,
    segment_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return standard 1D Qwen positions, resetting packed segments."""
    batch_size, seq_len = token_types.shape
    positions = torch.arange(
        seq_len,
        device=token_types.device,
        dtype=torch.long,
    ).unsqueeze(0).expand(batch_size, -1)
    if segment_ids is None:
        return positions
    if tuple(segment_ids.shape) != tuple(token_types.shape):
        raise ValueError("segment_ids must align with token_types")
    segment_ids = segment_ids.to(device=token_types.device, dtype=torch.long)
    starts = torch.ones_like(segment_ids, dtype=torch.bool)
    starts[:, 1:] = segment_ids[:, 1:].ne(segment_ids[:, :-1])
    starts &= segment_ids.ge(0)
    start_positions = torch.where(starts, positions, torch.zeros_like(positions))
    latest_start = torch.cummax(start_positions, dim=1).values
    return torch.where(
        segment_ids.ge(0),
        positions - latest_start,
        torch.zeros_like(positions),
    )


def build_showo2_allowed_mask(
    token_types: torch.Tensor,
    *,
    segment_ids: torch.Tensor | None = None,
    image_uncond_rows: torch.Tensor | None = None,
    image_uncond_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return Show-o2 visibility: causal diagonal plus full image blocks.

    Text and special tokens use ordinary autoregressive ``kv <= q``
    visibility.  Image-latent queries additionally see every latent in their
    own image span.  Conditioning dropout restricts selected image queries to
    that same image span, removing preceding text/class context.
    """
    if token_types.ndim != 2:
        raise ValueError(
            f"token_types must have shape [B,L], got {tuple(token_types.shape)}"
        )
    batch_size, seq_len = token_types.shape
    device = token_types.device
    valid = token_types.ne(3)
    indices = torch.arange(seq_len, device=device)
    causal = indices.view(1, 1, seq_len) <= indices.view(1, seq_len, 1)

    if segment_ids is None:
        same_segment = torch.ones(
            batch_size,
            seq_len,
            seq_len,
            device=device,
            dtype=torch.bool,
        )
    else:
        if tuple(segment_ids.shape) != tuple(token_types.shape):
            raise ValueError("segment_ids must align with token_types")
        segment_ids = segment_ids.to(device=device, dtype=torch.long)
        same_segment = (
            segment_ids.unsqueeze(1).eq(segment_ids.unsqueeze(2))
            & segment_ids.unsqueeze(1).ge(0)
            & segment_ids.unsqueeze(2).ge(0)
        )

    spans = image_span_ids(token_types)
    query_is_image = token_types.eq(1).unsqueeze(-1)
    key_is_image = token_types.eq(1).unsqueeze(1)
    same_image = (
        query_is_image
        & key_is_image
        & spans.unsqueeze(-1).eq(spans.unsqueeze(1))
        & spans.unsqueeze(-1).gt(0)
    )
    allowed = (
        valid.unsqueeze(-1)
        & valid.unsqueeze(1)
        & same_segment
        & (causal | same_image)
    )

    if image_uncond_rows is not None and image_uncond_mask is not None:
        raise ValueError("provide image_uncond_rows or image_uncond_mask, not both")
    if image_uncond_rows is not None:
        if tuple(image_uncond_rows.shape) != (batch_size,):
            raise ValueError(
                f"image_uncond_rows must have shape {(batch_size,)}, got "
                f"{tuple(image_uncond_rows.shape)}"
            )
        uncond_queries = (
            image_uncond_rows.to(device=device, dtype=torch.bool).unsqueeze(1)
            & token_types.eq(1)
        )
    elif image_uncond_mask is not None:
        if tuple(image_uncond_mask.shape) != tuple(token_types.shape):
            raise ValueError("image_uncond_mask must align with token_types")
        uncond_queries = image_uncond_mask.to(device=device, dtype=torch.bool)
    else:
        uncond_queries = torch.zeros_like(token_types, dtype=torch.bool)
    allowed = allowed & (~uncond_queries.unsqueeze(-1) | same_image)
    return allowed


def get_showo2_attention_mask(
    token_types: torch.Tensor,
    *,
    segment_ids: torch.Tensor | None = None,
    image_uncond_rows: torch.Tensor | None = None,
    image_uncond_mask: torch.Tensor | None = None,
):
    """Build the fused-NPU or reference-flex attention representation."""
    allowed = build_showo2_allowed_mask(
        token_types,
        segment_ids=segment_ids,
        image_uncond_rows=image_uncond_rows,
        image_uncond_mask=image_uncond_mask,
    )
    if token_types.device.type == "npu":
        return (~allowed).unsqueeze(1)

    batch_size, seq_len = token_types.shape

    def mask_mod(b, h, q_idx, kv_idx):
        del h
        return allowed[b, q_idx, kv_idx]

    return create_block_mask(
        mask_mod,
        B=batch_size,
        H=None,
        Q_LEN=seq_len,
        KV_LEN=seq_len,
        device=token_types.device,
    )


def sample_maskgit_training_masks(
    token_types: torch.Tensor,
    image_span_table: torch.Tensor,
    *,
    image_tokens_per_img: int,
    validation_mask_ratio: float | None = None,
    flow_sigma: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vectorized per-image cosine masking.

    Returns ``(visible_latents, loss_mask, mask_ratio_per_image)``.  Training
    samples one cosine-schedule ratio per image.  Validation uses a fixed ratio
    and the dataset's fixed random sigma ordering for deterministic masks.
    """
    if image_span_table.ndim != 2 or image_span_table.shape[1] < 4:
        raise ValueError("image_span_table must have shape [num_images, >=4]")
    device = token_types.device
    table = image_span_table.to(device=device, dtype=torch.long)
    rows = table[:, 0]
    starts = table[:, 2]
    if table.shape[0] == 0:
        raise ValueError("MaskGIT batches must contain at least one image")
    image_tokens = int(image_tokens_per_img)
    if image_tokens <= 0:
        raise ValueError("image_tokens_per_img must be positive")
    # The collator and the parent flow-loss path already enforce the canonical
    # fixed span length.  Reading a device scalar on the host here would insert
    # a device-to-host synchronization in every training step.
    offsets = torch.arange(image_tokens, device=device, dtype=torch.long)
    token_indices = starts.unsqueeze(1) + offsets.unsqueeze(0)
    num_images = table.shape[0]

    if validation_mask_ratio is None:
        schedule_t = torch.rand(num_images, device=device, dtype=torch.float32)
        ratios = torch.cos(schedule_t * (math.pi / 2.0))
        ordering_scores = torch.rand(
            num_images,
            image_tokens,
            device=device,
            dtype=torch.float32,
        )
    else:
        ratio = float(validation_mask_ratio)
        if not 0.0 < ratio <= 1.0:
            raise ValueError("validation_mask_ratio must be in (0, 1]")
        ratios = torch.full(
            (num_images,),
            ratio,
            device=device,
            dtype=torch.float32,
        )
        if flow_sigma is None:
            ordering_scores = offsets.float().unsqueeze(0).expand(num_images, -1)
        else:
            flow_sigma = flow_sigma.to(device=device, dtype=torch.float32)
            ordering_scores = flow_sigma[rows.unsqueeze(1), token_indices]

    masked_counts = torch.ceil(ratios * image_tokens).long().clamp_(1, image_tokens)
    order = ordering_scores.argsort(dim=1)
    ranks = torch.empty_like(order)
    ranks.scatter_(
        1,
        order,
        offsets.unsqueeze(0).expand(num_images, -1),
    )
    local_loss_mask = ranks < masked_counts.unsqueeze(1)
    loss_mask = torch.zeros_like(token_types, dtype=torch.bool)
    loss_mask[rows.unsqueeze(1), token_indices] = local_loss_mask
    visible_latents = token_types.eq(1) & ~loss_mask
    return visible_latents, loss_mask, ratios


__all__ = [
    "build_showo2_position_ids",
    "build_showo2_allowed_mask",
    "get_showo2_attention_mask",
    "image_span_ids",
    "sample_maskgit_training_masks",
]
