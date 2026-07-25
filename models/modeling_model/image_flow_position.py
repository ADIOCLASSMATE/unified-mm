"""Position contracts for the retained DF1 Selfless Flow baselines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FlowHeadPositionSpec:
    variant: str
    query_position_mode: str
    context_position_mode: str
    rope_mode: str

    @property
    def query_additive(self) -> bool:
        return self.query_position_mode == "additive_2d"

    @property
    def context_additive(self) -> bool:
        return self.context_position_mode == "additive_2d"

    @property
    def uses_rope(self) -> bool:
        return self.rope_mode == "row_col_2d"

    def as_contract(self, axis_dims: tuple[int, int]) -> dict[str, object]:
        return {
            "schema": "selfless_flow_head_position_v1",
            "variant": self.variant,
            "A_q": int(self.query_additive),
            "A_c": int(self.context_additive),
            "R_f": int(self.uses_rope),
            "query_position_mode": self.query_position_mode,
            "context_position_mode": self.context_position_mode,
            "rope_mode": self.rope_mode,
            "rope_axis_dims": [int(axis_dims[0]), int(axis_dims[1])],
            "rotate_value": False,
        }


FLOW_HEAD_POSITION_SPECS = {
    "FH0": FlowHeadPositionSpec("FH0", "additive_2d", "additive_2d", "none"),
    "FH4": FlowHeadPositionSpec("FH4", "none", "none", "row_col_2d"),
}
SUPPORTED_FLOW_HEAD_POSITION_VARIANTS = tuple(FLOW_HEAD_POSITION_SPECS)
BASELINE_FLOW_HEAD_POSITION_VARIANTS = SUPPORTED_FLOW_HEAD_POSITION_VARIANTS
DEFAULT_FLOW_HEAD_POSITION_VARIANT = "FH0"

_SPEC_BY_MODES = {
    (
        spec.query_position_mode,
        spec.context_position_mode,
        spec.rope_mode,
    ): spec
    for spec in FLOW_HEAD_POSITION_SPECS.values()
}

FLOW_HEAD_POSITION_CONFIG_KEYS = (
    "image_flow_position_variant",
    "image_flow_query_position_mode",
    "image_flow_context_position_mode",
    "image_flow_rope_mode",
    "image_flow_rope_axis_dims",
    "image_flow_rope_rotate_value",
)


def _contains(container: Any, key: str) -> bool:
    if isinstance(container, dict):
        return key in container
    try:
        return key in container
    except (TypeError, AttributeError):
        return hasattr(container, key)


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
        return
    try:
        container[key] = value
    except (TypeError, AttributeError):
        setattr(container, key, value)


def normalize_flow_head_position_variant(value: str | None) -> str:
    if value is None or not str(value).strip():
        return DEFAULT_FLOW_HEAD_POSITION_VARIANT
    normalized = str(value).strip().upper().replace("_", "")
    if normalized in FLOW_HEAD_POSITION_SPECS:
        return normalized
    raise ValueError(
        f"Unsupported image_flow_position_variant={value!r}; expected one of "
        f"{SUPPORTED_FLOW_HEAD_POSITION_VARIANTS}."
    )


def _normalize_position_mode(value: Any, *, label: str) -> str:
    normalized = str(value).strip().lower().replace("-", "_")
    if normalized not in {"none", "additive_2d"}:
        raise ValueError(f"{label} must be none or additive_2d, got {value!r}.")
    return normalized


def _normalize_rope_mode(value: Any) -> str:
    normalized = str(value).strip().lower().replace("-", "_")
    if normalized not in {"none", "row_col_2d"}:
        raise ValueError(
            "image_flow_rope_mode must be none or row_col_2d, "
            f"got {value!r}."
        )
    return normalized


def _head_dim(model_config: Any) -> int:
    width = int(_get(model_config, "image_flow_width", 1280))
    heads = int(_get(model_config, "image_flow_latent_mixer_heads", 8))
    if width <= 0 or heads <= 0 or width % heads:
        raise ValueError(
            "Contextual flow-head width must be positive and divisible by heads; "
            f"got width={width}, heads={heads}."
        )
    return width // heads


def validate_flow_rope_axis_dims(
    axis_dims: Any,
    *,
    head_dim: int,
) -> tuple[int, int]:
    if axis_dims is None:
        if head_dim % 4:
            raise ValueError(
                "Default row/column flow RoPE split requires head_dim divisible "
                f"by 4, got {head_dim}."
            )
        dims = (head_dim // 2, head_dim // 2)
    else:
        try:
            dims = tuple(int(item) for item in axis_dims)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "image_flow_rope_axis_dims must contain two integers."
            ) from exc
        if len(dims) != 2:
            raise ValueError(
                "image_flow_rope_axis_dims must contain exactly [row_dim, col_dim]."
            )
    if any(dim <= 0 or dim % 2 for dim in dims):
        raise ValueError(
            "Each image_flow_rope_axis_dims entry must be a positive even "
            f"dimension, got {dims}."
        )
    if sum(dims) != int(head_dim):
        raise ValueError(
            "image_flow_rope_axis_dims must sum to the per-head dimension; "
            f"got dims={dims}, head_dim={head_dim}."
        )
    return dims


def resolve_model_flow_head_position(
    model_config: Any,
) -> tuple[FlowHeadPositionSpec | None, tuple[int, int] | None]:
    """Resolve one of the two retained DF1 baseline position contracts."""

    head_arch = str(_get(model_config, "image_flow_head_arch", "contextual"))
    head_arch = head_arch.strip().lower().replace("-", "_")
    is_contextual = head_arch in {
        "contextual",
        "latent_mixer",
        "cross_attention",
        "cross_attn",
    }
    present = [key for key in FLOW_HEAD_POSITION_CONFIG_KEYS if _contains(model_config, key)]
    if not is_contextual:
        if present:
            raise ValueError(
                "Flow-head position modes apply only to image_flow_head_arch=contextual; "
                f"remove {present}."
            )
        return None, None

    head_dim = _head_dim(model_config)
    # Legacy/third-party additive-only configs may use toy head dimensions that
    # cannot be split into two non-empty rotary axes.  Preserve that inactive
    # FH0 path, while every explicitly declared FH contract (and every RoPE
    # system) remains subject to the strict dimension validator below.
    explicit_position_contract = bool(present)
    axis_dims = (
        validate_flow_rope_axis_dims(
            _get(model_config, "image_flow_rope_axis_dims", None),
            head_dim=head_dim,
        )
        if explicit_position_contract or head_dim % 4 == 0
        else None
    )
    rotate_value = bool(_get(model_config, "image_flow_rope_rotate_value", False))
    if rotate_value:
        raise ValueError(
            "image_flow_rope_rotate_value=true is forbidden; flow-head RoPE "
            "never rotates V."
        )

    has_variant = _contains(model_config, "image_flow_position_variant")
    has_modes = any(
        _contains(model_config, key)
        for key in (
            "image_flow_query_position_mode",
            "image_flow_context_position_mode",
            "image_flow_rope_mode",
        )
    )
    if has_variant:
        variant = normalize_flow_head_position_variant(
            _get(model_config, "image_flow_position_variant")
        )
        spec = FLOW_HEAD_POSITION_SPECS[variant]
        if has_modes:
            modes = (
                _normalize_position_mode(
                    _get(model_config, "image_flow_query_position_mode"),
                    label="image_flow_query_position_mode",
                ),
                _normalize_position_mode(
                    _get(model_config, "image_flow_context_position_mode"),
                    label="image_flow_context_position_mode",
                ),
                _normalize_rope_mode(_get(model_config, "image_flow_rope_mode")),
            )
            expected_modes = (
                spec.query_position_mode,
                spec.context_position_mode,
                spec.rope_mode,
            )
            if modes != expected_modes:
                raise ValueError(
                    f"{variant} conflicts with explicit flow-position modes: "
                    f"expected={expected_modes}, actual={modes}."
                )
    elif has_modes:
        modes = (
            _normalize_position_mode(
                _get(
                    model_config,
                    "image_flow_query_position_mode",
                    "additive_2d",
                ),
                label="image_flow_query_position_mode",
            ),
            _normalize_position_mode(
                _get(
                    model_config,
                    "image_flow_context_position_mode",
                    "additive_2d",
                ),
                label="image_flow_context_position_mode",
            ),
            _normalize_rope_mode(
                _get(model_config, "image_flow_rope_mode", "none")
            ),
        )
        spec = _SPEC_BY_MODES.get(modes)
        if spec is None:
            raise ValueError(
                "Flow-head position flags do not match a retained baseline "
                f"system: {modes}."
            )
    else:
        spec = FLOW_HEAD_POSITION_SPECS[DEFAULT_FLOW_HEAD_POSITION_VARIANT]

    if spec.uses_rope and axis_dims is None:
        raise ValueError(
            "row_col_2d flow-head RoPE requires a head dimension that can be "
            "split into two positive even axes."
        )
    if not explicit_position_contract and axis_dims is None:
        # Do not turn an implicit additive-only toy config into an explicit,
        # subsequently invalid rotary contract when the same config object is
        # reused or saved and reloaded.
        return spec, None
    _set(model_config, "image_flow_position_variant", spec.variant)
    _set(
        model_config,
        "image_flow_query_position_mode",
        spec.query_position_mode,
    )
    _set(
        model_config,
        "image_flow_context_position_mode",
        spec.context_position_mode,
    )
    _set(model_config, "image_flow_rope_mode", spec.rope_mode)
    if axis_dims is not None:
        _set(model_config, "image_flow_rope_axis_dims", list(axis_dims))
    _set(model_config, "image_flow_rope_rotate_value", False)
    return spec, axis_dims


def resolve_flow_head_position_config(
    config: Any,
) -> tuple[FlowHeadPositionSpec | None, tuple[int, int] | None]:
    model = _get(config, "model")
    if model is None:
        raise ValueError("config.model is required")
    spec, axis_dims = resolve_model_flow_head_position(model)
    if spec is None:
        return None, None
    return spec, axis_dims
