"""Strict image-flow adapter loading shared by training and gradient probes."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import torch


def is_disabled_path(value: object) -> bool:
    return value is None or (
        isinstance(value, str) and value.lower() in {"none", "null", "false", ""}
    )


def should_load_configured_adapter(
    *,
    adapter_path: str | Path | None,
    configured_model_path: str | Path,
    probe_model_path: str | Path,
    ema_dir: str | Path | None = None,
) -> bool:
    """Return whether a probe still needs the configured split-init adapter.

    A configured adapter belongs on top of the original text-only checkpoint.
    A later full-model checkpoint (including a sharded EMA checkpoint) already
    contains the trained image modules, so loading the original adapter again
    would silently overwrite the state that the probe is supposed to measure.
    """

    if is_disabled_path(adapter_path) or not is_disabled_path(ema_dir):
        return False
    configured = Path(configured_model_path).expanduser().resolve(strict=False)
    probed = Path(probe_model_path).expanduser().resolve(strict=False)
    return configured == probed


def _special_token_ids(config) -> dict[str, int]:
    ids = {
        "mask": int(config.model.mask_token_id),
        "boi": int(config.model.boi_token_id),
        "eoi": int(config.model.eoi_token_id),
    }
    image_mask_token_id = config.model.get("image_mask_token_id", None)
    if image_mask_token_id is not None:
        ids["image_mask"] = int(image_mask_token_id)
    return ids


def _module_summary(state: dict[str, torch.Tensor]) -> dict[str, Any]:
    return {
        "tensor_count": len(state),
        "numel": sum(int(value.numel()) for value in state.values()),
        "keys": sorted(state),
    }


def load_image_flow_adapter(
    model,
    adapter_path: str | Path | None,
    config,
    *,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any] | None:
    """Load only image modules and multimodal special-token rows.

    The language backbone, ordinary vocabulary rows, tied LM head, and all
    transformer normalization parameters are deliberately left untouched.
    ``strict=True`` is used for every image module so an architecture mismatch
    cannot silently fall back to partially random weights.
    """

    if is_disabled_path(adapter_path):
        return None

    resolved = Path(adapter_path)
    if resolved.is_dir():
        resolved = resolved / "model.safetensors"
    if not resolved.is_file():
        raise FileNotFoundError(f"image-flow adapter does not exist: {resolved}")

    module_states: dict[str, dict[str, torch.Tensor]]
    special_token_ids = _special_token_ids(config)
    special_token_rows: dict[str, torch.Tensor] = {}

    if resolved.suffix == ".safetensors":
        from safetensors import safe_open

        module_states = {
            "image_flow_head": {},
            "image_flow_condition_proj": {},
            "image_token_embedder": {},
            "backbone_flow_time_embedder": {},
        }
        with safe_open(str(resolved), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key.startswith("image_flow_head."):
                    module_states["image_flow_head"][
                        key.removeprefix("image_flow_head.")
                    ] = handle.get_tensor(key)
                elif key.startswith("image_flow_condition_proj."):
                    module_states["image_flow_condition_proj"][
                        key.removeprefix("image_flow_condition_proj.")
                    ] = handle.get_tensor(key)
                elif key.startswith("model.image_token_embedder."):
                    module_states["image_token_embedder"][
                        key.removeprefix("model.image_token_embedder.")
                    ] = handle.get_tensor(key)
                elif key.startswith("model.backbone_flow_time_embedder."):
                    module_states["backbone_flow_time_embedder"][
                        key.removeprefix("model.backbone_flow_time_embedder.")
                    ] = handle.get_tensor(key)
                elif key == "model.embed_tokens.weight":
                    embedding = handle.get_tensor(key)
                    special_token_rows = {
                        name: embedding[token_id].clone()
                        for name, token_id in special_token_ids.items()
                    }
        adapter_format = "hf_safetensors_image_modules"
    else:
        state = torch.load(resolved, map_location="cpu", weights_only=True)
        required = {
            "image_flow_head",
            "image_flow_condition_proj",
            "image_token_embedder",
            "special_token_ids",
            "special_token_embeddings",
        }
        if hasattr(model.model, "backbone_flow_time_embedder"):
            required.add("backbone_flow_time_embedder")
        missing = required - set(state)
        if missing:
            raise ValueError(
                f"Final image-flow adapter {resolved} is missing {sorted(missing)}"
            )
        recorded_token_ids = {
            str(name): int(token_id)
            for name, token_id in state["special_token_ids"].items()
        }
        if recorded_token_ids != special_token_ids:
            raise ValueError(
                "Adapter special-token ids do not match the text tokenizer: "
                f"adapter={recorded_token_ids}, model={special_token_ids}"
            )
        special_token_rows = state["special_token_embeddings"]
        if set(special_token_rows) != set(special_token_ids):
            raise ValueError(
                "Adapter special-token set does not match the finalized model: "
                f"adapter={sorted(special_token_rows)}, "
                f"model={sorted(special_token_ids)}"
            )
        module_states = {
            "image_flow_head": state["image_flow_head"],
            "image_flow_condition_proj": state["image_flow_condition_proj"],
            "image_token_embedder": state["image_token_embedder"],
            "backbone_flow_time_embedder": state.get(
                "backbone_flow_time_embedder", {}
            ),
        }
        adapter_format = "finalized_pt_adapter"

    required_nonempty = (
        "image_flow_head",
        "image_flow_condition_proj",
        "image_token_embedder",
    )
    empty = [name for name in required_nonempty if not module_states[name]]
    if empty:
        raise ValueError(
            f"Image-flow adapter {resolved} has empty required modules: {empty}"
        )

    model.image_flow_head.load_state_dict(module_states["image_flow_head"], strict=True)
    model.image_flow_condition_proj.load_state_dict(
        module_states["image_flow_condition_proj"], strict=True
    )
    model.image_token_embedder.load_state_dict(
        module_states["image_token_embedder"], strict=True
    )
    backbone_flow_time_embedder = getattr(
        model.model, "backbone_flow_time_embedder", None
    )
    if backbone_flow_time_embedder is not None:
        backbone_flow_time_embedder.load_state_dict(
            module_states["backbone_flow_time_embedder"], strict=True
        )

    embed = model.model.embed_tokens.weight
    with torch.no_grad():
        for name, token_id in special_token_ids.items():
            value = special_token_rows[name]
            if tuple(value.shape) != tuple(embed[token_id].shape):
                raise ValueError(
                    f"Adapter special-token row {name!r} shape mismatch: "
                    f"adapter={tuple(value.shape)}, model={tuple(embed[token_id].shape)}"
                )
            embed[token_id].copy_(value.to(device=embed.device, dtype=embed.dtype))

    report = {
        "schema": "selfless_image_flow_adapter_load_v1",
        "path": str(resolved),
        "format": adapter_format,
        "loaded_modules": {
            name: _module_summary(state)
            for name, state in module_states.items()
            if state
        },
        "special_token_ids": special_token_ids,
        "special_token_rows_loaded": sorted(special_token_ids),
        "preserved_from_text_initialization": [
            "model.layers",
            "model.norm",
            "ordinary model.embed_tokens rows",
            "tied lm_head",
        ],
    }
    if log is not None:
        log(
            "Loaded image-flow adapter only from "
            f"{resolved}; modules={sorted(report['loaded_modules'])}, "
            f"special_tokens={sorted(special_token_ids)}"
        )
    return report
