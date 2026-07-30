"""Shared invariants for the fixed pure-2D Selfless-Flow image backbone."""

from __future__ import annotations

import math
from typing import Any


CANONICAL_IMAGE_GRID_SIDE = 16
CANONICAL_IMAGE_LATENT_DIM = 16
CANONICAL_IMAGE_TOKENS = CANONICAL_IMAGE_GRID_SIDE**2


def _get(container: Any, key: str, default: Any = None) -> Any:
    if isinstance(container, dict):
        return container.get(key, default)
    getter = getattr(container, "get", None)
    if callable(getter):
        try:
            return getter(key, default)
        except TypeError:
            pass
    return getattr(container, key, default)


def validate_model_image_layout(model_config: Any) -> tuple[int, int]:
    """Validate the direct square latent layout used by the fixed architecture."""

    tokens = int(_get(model_config, "image_tokens_per_img", CANONICAL_IMAGE_TOKENS))
    latent_dim = int(
        _get(model_config, "image_latent_dim", CANONICAL_IMAGE_LATENT_DIM)
    )
    side = int(math.isqrt(tokens))
    if tokens <= 0 or side * side != tokens or latent_dim <= 0:
        raise ValueError(
            "Selfless Flow image latents must use a direct square [T,C] layout "
            f"with positive C; got [{tokens},{latent_dim}]."
        )
    return tokens, latent_dim


def validate_image_data_layout(config: Any) -> tuple[int, int]:
    """Ensure model and dataset describe the same direct latent layout."""

    model = _get(config, "model")
    if model is None:
        raise ValueError("config.model is required")
    model_tokens, model_latent_dim = validate_model_image_layout(model)

    dataset = _get(config, "dataset")
    params = _get(dataset, "params") if dataset is not None else None
    if params is None:
        return model_tokens, model_latent_dim

    data_tokens = int(
        _get(params, "image_tokens_per_img", model_tokens)
    )
    data_latent_dim = int(
        _get(params, "image_latent_dim", model_latent_dim)
    )
    if (data_tokens, data_latent_dim) != (model_tokens, model_latent_dim):
        raise ValueError(
            "Dataset image layout must match the model direct layout; "
            f"model=[{model_tokens},{model_latent_dim}], "
            f"dataset=[{data_tokens},{data_latent_dim}]."
        )
    return model_tokens, model_latent_dim


def pure_2d_position_contract() -> dict[str, object]:
    """Machine-readable description of the only supported position design."""

    return {
        "schema": "selfless_pure_2d_position_v1",
        "backbone": {
            "image_qk_rotary": "row_col_2d",
            "text_qk_rotary": "qwen_sequence_1d",
            "additive_image_position": False,
        },
        "flow_head": {
            "architecture": "dynamic_dual_stream",
            "image_qk_rotary": "row_col_2d",
            "rotate_value": False,
            "additive_image_position": False,
        },
    }


def validate_direct_latent_cache_shape(
    latents: Any,
    *,
    image_tokens_per_img: int,
    image_latent_dim: int,
) -> int:
    if getattr(latents, "ndim", None) != 3:
        raise ValueError(
            f"Expected cache latents [N,T,C], got {tuple(latents.shape)}"
        )
    tokens, channels = int(latents.shape[1]), int(latents.shape[2])
    expected = (int(image_tokens_per_img), int(image_latent_dim))
    if (tokens, channels) != expected:
        raise ValueError(
            "Image latent cache shape must match the model direct layout; "
            f"expected [N,{expected[0]},{expected[1]}], got {tuple(latents.shape)}."
        )
    side = int(math.isqrt(tokens))
    if side * side != tokens:
        raise ValueError(f"Image token count must form a square grid, got {tokens}")
    return side
