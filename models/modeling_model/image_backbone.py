"""Closed image-backbone interface for Selfless Flow.

The backbone study is complete.  Runtime code intentionally exposes only the
three retained variants below; stage embeddings, 1D image RoPE, and packed
space-to-depth layouts are historical experiments rather than supported
architecture knobs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


CANONICAL_IMAGE_GRID_SIDE = 16
CANONICAL_IMAGE_LATENT_DIM = 16
CANONICAL_IMAGE_TOKENS = CANONICAL_IMAGE_GRID_SIDE**2
DEFAULT_IMAGE_BACKBONE_VARIANT = "E2-Q0"


@dataclass(frozen=True)
class ImageBackboneSpec:
    variant: str
    observed_position_mode: str
    mask_position_mode: str


IMAGE_BACKBONE_SPECS = {
    "E2-Q1": ImageBackboneSpec(
        variant="E2-Q1",
        observed_position_mode="none",
        mask_position_mode="additive_2d",
    ),
    "E2-Q0": ImageBackboneSpec(
        variant="E2-Q0",
        observed_position_mode="none",
        mask_position_mode="none",
    ),
    "E2b-Q0": ImageBackboneSpec(
        variant="E2b-Q0",
        observed_position_mode="additive_2d",
        mask_position_mode="none",
    ),
}
SUPPORTED_IMAGE_BACKBONE_VARIANTS = tuple(IMAGE_BACKBONE_SPECS)

# These keys are read only to migrate already-produced E2/E2b checkpoints.
# They are removed immediately and are never emitted by new configs.
LEGACY_IMAGE_ARCHITECTURE_KEYS = (
    "image_query_stage_mode",
    "image_observed_position_mode",
    "image_mask_position_mode",
    "image_rope_mode",
    "image_space_to_depth_factor",
    "image_canonical_grid_side",
    "image_canonical_latent_dim",
)


def normalize_image_backbone_variant(value: str | None) -> str:
    if value is None or not str(value).strip():
        return DEFAULT_IMAGE_BACKBONE_VARIANT
    normalized = str(value).strip().lower().replace("_", "-")
    aliases = {key.lower(): key for key in SUPPORTED_IMAGE_BACKBONE_VARIANTS}
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported image_backbone_variant={value!r}; expected one of "
            f"{SUPPORTED_IMAGE_BACKBONE_VARIANTS}."
        ) from exc


def image_backbone_spec(value: str | None) -> ImageBackboneSpec:
    return IMAGE_BACKBONE_SPECS[normalize_image_backbone_variant(value)]


def _contains(container: Any, key: str) -> bool:
    if isinstance(container, dict):
        return key in container
    try:
        return key in container
    except (TypeError, AttributeError):
        return key in vars(container)


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


def _set(container: Any, key: str, value: Any) -> None:
    if isinstance(container, dict):
        container[key] = value
    else:
        try:
            container[key] = value
        except (TypeError, AttributeError):
            setattr(container, key, value)


def _delete(container: Any, key: str) -> None:
    if not _contains(container, key):
        return
    if isinstance(container, dict):
        del container[key]
        return
    try:
        del container[key]
    except (TypeError, AttributeError):
        delattr(container, key)


def _infer_legacy_variant(model_config: Any) -> str:
    """Map only the exact retained historical configs to the new enum."""

    stage = str(_get(model_config, "image_query_stage_mode", "none")).lower()
    observed = str(
        _get(model_config, "image_observed_position_mode", "additive_2d")
    ).lower()
    mask = str(_get(model_config, "image_mask_position_mode", "additive_2d")).lower()
    rope = str(_get(model_config, "image_rope_mode", "sequence_1d")).lower()
    factor = int(_get(model_config, "image_space_to_depth_factor", 1))
    canonical_side = int(
        _get(model_config, "image_canonical_grid_side", CANONICAL_IMAGE_GRID_SIDE)
    )
    canonical_dim = int(
        _get(model_config, "image_canonical_latent_dim", CANONICAL_IMAGE_LATENT_DIM)
    )

    fixed_contract = (
        stage == "none"
        and rope == "row_col_2d"
        and factor == 1
        and canonical_side == CANONICAL_IMAGE_GRID_SIDE
        and canonical_dim == CANONICAL_IMAGE_LATENT_DIM
    )
    mapping = {
        ("none", "additive_2d"): "E2-Q1",
        ("none", "none"): "E2-Q0",
        ("additive_2d", "none"): "E2b-Q0",
    }
    variant = mapping.get((observed, mask)) if fixed_contract else None
    if variant is None:
        raise ValueError(
            "This checkpoint/config uses a retired image-backbone architecture. "
            "Only E2-Q1, E2-Q0, and E2b-Q0 are supported; stage embeddings, "
            "sequence-1D image RoPE, S2D, and E2b-Q1 are intentionally removed."
        )
    return variant


def _validate_direct_layout(model_config: Any) -> tuple[int, int]:
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
    _set(model_config, "image_tokens_per_img", tokens)
    _set(model_config, "image_latent_dim", latent_dim)
    return tokens, latent_dim


def resolve_model_image_backbone(model_config: Any) -> ImageBackboneSpec:
    """Resolve/migrate a model config and leave only ``image_backbone_variant``."""

    has_variant = _contains(model_config, "image_backbone_variant")
    has_legacy = any(_contains(model_config, key) for key in LEGACY_IMAGE_ARCHITECTURE_KEYS)
    if has_variant:
        variant = normalize_image_backbone_variant(
            _get(model_config, "image_backbone_variant")
        )
        if has_legacy:
            raise ValueError(
                "image_backbone_variant cannot be combined with retired architecture "
                f"fields: {[key for key in LEGACY_IMAGE_ARCHITECTURE_KEYS if _contains(model_config, key)]}"
            )
    elif has_legacy:
        variant = _infer_legacy_variant(model_config)
    else:
        variant = DEFAULT_IMAGE_BACKBONE_VARIANT

    _validate_direct_layout(model_config)
    _set(model_config, "image_backbone_variant", variant)
    for key in LEGACY_IMAGE_ARCHITECTURE_KEYS:
        _delete(model_config, key)
    return IMAGE_BACKBONE_SPECS[variant]


def resolve_image_backbone_config(config: Any) -> ImageBackboneSpec:
    """Resolve an OmegaConf-style training config to the closed backbone enum."""

    model = _get(config, "model")
    if model is None:
        raise ValueError("config.model is required")
    had_explicit_variant = _contains(model, "image_backbone_variant")
    spec = resolve_model_image_backbone(model)

    dataset = _get(config, "dataset")
    params = _get(dataset, "params") if dataset is not None else None
    if params is not None:
        retired_dataset_keys = (
            "image_space_to_depth_factor",
            "image_canonical_grid_side",
            "image_canonical_latent_dim",
        )
        present_retired = [
            key for key in retired_dataset_keys if _contains(params, key)
        ]
        if present_retired and had_explicit_variant:
            raise ValueError(
                "Dataset S2D/canonical-layout overrides are retired; remove "
                f"{present_retired}."
            )
        if present_retired:
            legacy_values = {
                "image_space_to_depth_factor": 1,
                "image_canonical_grid_side": CANONICAL_IMAGE_GRID_SIDE,
                "image_canonical_latent_dim": CANONICAL_IMAGE_LATENT_DIM,
            }
            invalid = {
                key: _get(params, key)
                for key in present_retired
                if int(_get(params, key)) != legacy_values[key]
            }
            if invalid:
                raise ValueError(
                    "This legacy dataset config uses a retired packed layout: "
                    f"{invalid}."
                )
            for key in present_retired:
                _delete(params, key)
        model_tokens = int(_get(model, "image_tokens_per_img"))
        model_latent_dim = int(_get(model, "image_latent_dim"))
        tokens = int(_get(params, "image_tokens_per_img", model_tokens))
        latent_dim = int(_get(params, "image_latent_dim", model_latent_dim))
        if (tokens, latent_dim) != (model_tokens, model_latent_dim):
            raise ValueError(
                "Dataset image layout must match the model direct layout; "
                f"model=[{model_tokens},{model_latent_dim}], "
                f"dataset=[{tokens},{latent_dim}]."
            )
        _set(params, "image_tokens_per_img", model_tokens)
        _set(params, "image_latent_dim", model_latent_dim)
    return spec


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
