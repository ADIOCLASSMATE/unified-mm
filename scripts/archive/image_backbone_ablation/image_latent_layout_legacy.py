"""Historical S2D implementation retained only to audit archived experiments.

Active training/model code must not import this module.
"""

from __future__ import annotations

import math

import torch


CANONICAL_IMAGE_GRID_SIDE = 16
CANONICAL_IMAGE_LATENT_DIM = 16
SUPPORTED_SPACE_TO_DEPTH_FACTORS = (1, 2)


def validate_space_to_depth_factor(factor: int) -> int:
    factor = int(factor)
    if factor not in SUPPORTED_SPACE_TO_DEPTH_FACTORS:
        raise ValueError(
            f"image_space_to_depth_factor must be one of "
            f"{SUPPORTED_SPACE_TO_DEPTH_FACTORS}, got {factor}"
        )
    return factor


def derived_image_layout(
    factor: int,
    *,
    canonical_grid_side: int = CANONICAL_IMAGE_GRID_SIDE,
    canonical_latent_dim: int = CANONICAL_IMAGE_LATENT_DIM,
) -> dict[str, int]:
    factor = validate_space_to_depth_factor(factor)
    canonical_grid_side = int(canonical_grid_side)
    canonical_latent_dim = int(canonical_latent_dim)
    if canonical_grid_side <= 0 or canonical_grid_side % factor:
        raise ValueError(
            f"canonical_grid_side={canonical_grid_side} must be positive and divisible by factor={factor}"
        )
    if canonical_latent_dim <= 0:
        raise ValueError(f"canonical_latent_dim must be positive, got {canonical_latent_dim}")
    image_grid_side = canonical_grid_side // factor
    return {
        "factor": factor,
        "canonical_grid_side": canonical_grid_side,
        "canonical_latent_dim": canonical_latent_dim,
        "canonical_tokens_per_img": canonical_grid_side * canonical_grid_side,
        "image_grid_side": image_grid_side,
        "image_tokens_per_img": image_grid_side * image_grid_side,
        "image_latent_dim": canonical_latent_dim * factor * factor,
        "position_stride": factor,
    }


def validate_flattened_layout(
    image_tokens_per_img: int,
    image_latent_dim: int,
    factor: int,
    *,
    canonical_grid_side: int = CANONICAL_IMAGE_GRID_SIDE,
    canonical_latent_dim: int = CANONICAL_IMAGE_LATENT_DIM,
) -> dict[str, int]:
    layout = derived_image_layout(
        factor,
        canonical_grid_side=canonical_grid_side,
        canonical_latent_dim=canonical_latent_dim,
    )
    actual = (int(image_tokens_per_img), int(image_latent_dim))
    expected = (layout["image_tokens_per_img"], layout["image_latent_dim"])
    if actual != expected:
        raise ValueError(
            "image layout fields must be derived from image_space_to_depth_factor: "
            f"factor={factor} requires image_tokens_per_img={expected[0]} and "
            f"image_latent_dim={expected[1]}, got {actual[0]} and {actual[1]}"
        )
    return layout


def space_to_depth_2d(latents: torch.Tensor, factor: int) -> torch.Tensor:
    """Losslessly pack channels-last ``[..., H, W, C]`` latents into spatial blocks."""

    factor = validate_space_to_depth_factor(factor)
    if latents.ndim < 3:
        raise ValueError(f"latents must have shape [...,H,W,C], got {tuple(latents.shape)}")
    if factor == 1:
        return latents
    height, width, channels = map(int, latents.shape[-3:])
    if height % factor or width % factor:
        raise ValueError(
            f"spatial shape {(height, width)} must be divisible by factor={factor}"
        )
    prefix = latents.shape[:-3]
    packed = latents.reshape(
        *prefix,
        height // factor,
        factor,
        width // factor,
        factor,
        channels,
    )
    prefix_dims = len(prefix)
    packed = packed.permute(
        *range(prefix_dims),
        prefix_dims,
        prefix_dims + 2,
        prefix_dims + 4,
        prefix_dims + 1,
        prefix_dims + 3,
    )
    return packed.reshape(
        *prefix,
        height // factor,
        width // factor,
        channels * factor * factor,
    )


def depth_to_space_2d(latents: torch.Tensor, factor: int) -> torch.Tensor:
    """Invert :func:`space_to_depth_2d` for channels-last latents."""

    factor = validate_space_to_depth_factor(factor)
    if latents.ndim < 3:
        raise ValueError(f"latents must have shape [...,H,W,C], got {tuple(latents.shape)}")
    if factor == 1:
        return latents
    height, width, packed_channels = map(int, latents.shape[-3:])
    block_area = factor * factor
    if packed_channels % block_area:
        raise ValueError(
            f"packed channel count {packed_channels} must be divisible by factor^2={block_area}"
        )
    channels = packed_channels // block_area
    prefix = latents.shape[:-3]
    unpacked = latents.reshape(
        *prefix,
        height,
        width,
        channels,
        factor,
        factor,
    )
    prefix_dims = len(prefix)
    unpacked = unpacked.permute(
        *range(prefix_dims),
        prefix_dims,
        prefix_dims + 3,
        prefix_dims + 1,
        prefix_dims + 4,
        prefix_dims + 2,
    )
    return unpacked.reshape(
        *prefix,
        height * factor,
        width * factor,
        channels,
    )


def restore_canonical_latents_chw(latents: torch.Tensor, factor: int) -> torch.Tensor:
    """Restore packed ``[..., C, H, W]`` tensors to canonical channels-first latents."""

    factor = validate_space_to_depth_factor(factor)
    if latents.ndim < 3:
        raise ValueError(f"latents must have shape [...,C,H,W], got {tuple(latents.shape)}")
    if factor == 1:
        return latents
    prefix_dims = latents.ndim - 3
    channels_last = latents.permute(
        *range(prefix_dims),
        prefix_dims + 1,
        prefix_dims + 2,
        prefix_dims,
    )
    restored = depth_to_space_2d(channels_last, factor)
    return restored.permute(
        *range(prefix_dims),
        prefix_dims + 2,
        prefix_dims,
        prefix_dims + 1,
    )


def infer_canonical_layout_from_cache(latents: torch.Tensor) -> tuple[int, int]:
    if latents.ndim != 3:
        raise ValueError(f"Expected cache latents [N,T,C], got {tuple(latents.shape)}")
    tokens = int(latents.shape[1])
    side = int(math.isqrt(tokens))
    if side * side != tokens:
        raise ValueError(f"canonical cache token count must form a square grid, got {tokens}")
    return side, int(latents.shape[2])


def resolve_image_layout_config(config) -> dict[str, int]:
    """Make the S2D factor authoritative across model and dataset config fields."""

    factor = int(config.model.get("image_space_to_depth_factor", 1))
    canonical_side = int(config.model.get("image_canonical_grid_side", CANONICAL_IMAGE_GRID_SIDE))
    canonical_dim = int(config.model.get("image_canonical_latent_dim", CANONICAL_IMAGE_LATENT_DIM))
    layout = derived_image_layout(
        factor,
        canonical_grid_side=canonical_side,
        canonical_latent_dim=canonical_dim,
    )
    expected_model_fields = {
        "image_tokens_per_img": layout["image_tokens_per_img"],
        "image_latent_dim": layout["image_latent_dim"],
    }
    for key, expected in expected_model_fields.items():
        configured = config.model.get(key, None)
        if configured is not None and int(configured) != int(expected):
            raise ValueError(
                f"model.{key}={configured} conflicts with image_space_to_depth_factor={factor}; "
                f"expected {expected}"
            )
        config.model[key] = int(expected)
    config.model.image_space_to_depth_factor = factor
    config.model.image_canonical_grid_side = canonical_side
    config.model.image_canonical_latent_dim = canonical_dim

    dataset = config.get("dataset", None)
    if dataset is not None and dataset.get("params", None) is not None:
        params = dataset.params
        for key, expected in expected_model_fields.items():
            configured = params.get(key, None)
            if configured is not None and int(configured) != int(expected):
                raise ValueError(
                    f"dataset.params.{key}={configured} conflicts with "
                    f"image_space_to_depth_factor={factor}; expected {expected}"
                )
            params[key] = int(expected)
        configured_factor = params.get("image_space_to_depth_factor", factor)
        if int(configured_factor) != factor:
            raise ValueError(
                "dataset.params.image_space_to_depth_factor must match "
                f"model.image_space_to_depth_factor={factor}, got {configured_factor}"
            )
        params.image_space_to_depth_factor = factor
    return layout
