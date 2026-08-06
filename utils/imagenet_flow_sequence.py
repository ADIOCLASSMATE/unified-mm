"""Shared sequence-order utilities for ImageNet flow batches."""

from __future__ import annotations

from typing import Any, Mapping

import torch


def scalar_int(item: Mapping[str, Any], key: str) -> int:
    """Read scalar sample metadata without coupling callers to its container."""

    value = item[key]
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(
                f"sample field {key!r} must be scalar, got shape={tuple(value.shape)}"
            )
        value = value.item()
    return int(value)


def build_selfless_sigma(
    item: Mapping[str, Any],
    *,
    image_tokens: int,
) -> torch.Tensor:
    """Build one sample's strict generation order.

    The physical sequence is ``prefix, BOI, image, EOI, suffix, EOS`` while
    sigma deliberately orders EOI before every image token. Consequently EOI
    cannot see image content, and image queries can see prefix/BOI/EOI plus
    only image tokens revealed earlier by the deterministic permutation.
    """

    length = int(item["input_ids"].shape[0])
    prompt_len = scalar_int(item, "prompt_len")
    suffix_len = scalar_int(item, "suffix_len")
    image_start = scalar_int(item, "image_start")
    eoi_pos = image_start + int(image_tokens)
    suffix_start = eoi_pos + 1
    eos_pos = length - 1
    if image_start != prompt_len + 1:
        raise ValueError(
            "image_start must immediately follow prefix and BOI: "
            f"image_start={image_start}, prompt_len={prompt_len}"
        )
    if eoi_pos >= length or eos_pos < eoi_pos:
        raise ValueError(
            f"invalid image span for segment length={length}: "
            f"start={image_start}, image_tokens={image_tokens}"
        )
    expected_length = prompt_len + int(image_tokens) + suffix_len + 3
    if length != expected_length:
        raise ValueError(
            "serialized sample metadata does not match sequence length: "
            f"length={length}, expected={expected_length}"
        )

    sigma = torch.empty(length, dtype=torch.long)
    if prompt_len:
        sigma[:prompt_len] = torch.arange(prompt_len, dtype=torch.long)
    sigma[prompt_len] = prompt_len
    sigma[eoi_pos] = prompt_len + 1

    generator = torch.Generator(device="cpu")
    generator.manual_seed(scalar_int(item, "reveal_seed"))
    reveal_order = torch.rand(
        int(image_tokens), generator=generator
    ).argsort()
    sigma[image_start:eoi_pos] = prompt_len + 2 + reveal_order

    suffix_sigma_start = prompt_len + int(image_tokens) + 2
    if suffix_len:
        sigma[suffix_start:eos_pos] = suffix_sigma_start + torch.arange(
            suffix_len, dtype=torch.long
        )
    sigma[eos_pos] = suffix_sigma_start + suffix_len
    return sigma
