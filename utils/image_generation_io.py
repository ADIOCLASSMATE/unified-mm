"""Shared model/adapter/VAE loading for image-generation entry points."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
from safetensors import safe_open


def _load_compatible_state(module, state: dict[str, torch.Tensor]) -> dict:
    target = module.state_dict()
    compatible = {}
    skipped = {}
    for key, value in state.items():
        target_value = target.get(key)
        if target_value is None:
            skipped[key] = [list(value.shape), None]
        elif tuple(value.shape) == tuple(target_value.shape):
            compatible[key] = value
        else:
            skipped[key] = [
                list(value.shape),
                list(target_value.shape),
            ]
    missing, unexpected = module.load_state_dict(
        compatible,
        strict=False,
    )
    return {
        "loaded": len(compatible),
        "missing": list(missing),
        "unexpected": list(unexpected),
        "skipped": skipped,
    }


def load_adapter(model, adapter_path: str | Path) -> dict:
    """Load a legacy flow adapter without adding generation behavior."""

    raw_path = str(adapter_path)
    if raw_path.lower() in {"none", "null", "false", ""}:
        return {"adapter": None}
    path = Path(raw_path)
    if path.is_dir():
        path = path / "model.safetensors"
    if not path.is_file():
        raise FileNotFoundError(path)

    report = {"adapter": str(path)}
    if path.suffix == ".safetensors":
        states = {
            "image_flow_head": {},
            "image_flow_condition_proj": {},
            "image_token_embedder": {},
            "backbone_flow_time_embedder": {},
        }
        projector_target = model.image_token_embedder.state_dict()
        with safe_open(
            str(path),
            framework="pt",
            device="cpu",
        ) as handle:
            for key in handle.keys():
                value = handle.get_tensor(key)
                if key.startswith("image_flow_head."):
                    states["image_flow_head"][
                        key.removeprefix("image_flow_head.")
                    ] = value
                elif key.startswith("image_flow_condition_proj."):
                    states["image_flow_condition_proj"][
                        key.removeprefix("image_flow_condition_proj.")
                    ] = value
                elif key.startswith("model.image_flow_condition_proj."):
                    states["image_flow_condition_proj"][
                        key.removeprefix("model.image_flow_condition_proj.")
                    ] = value
                elif key.startswith("model.image_token_embedder."):
                    name = key.removeprefix(
                        "model.image_token_embedder."
                    )
                    if (
                        name in projector_target
                        and tuple(value.shape)
                        == tuple(projector_target[name].shape)
                    ):
                        states["image_token_embedder"][name] = value
                elif key.startswith(
                    "model.backbone_flow_time_embedder."
                ):
                    states["backbone_flow_time_embedder"][
                        key.removeprefix(
                            "model.backbone_flow_time_embedder."
                        )
                    ] = value
    else:
        payload = torch.load(path, map_location="cpu")
        states = {
            name: payload.get(name, {})
            for name in (
                "image_flow_head",
                "image_flow_condition_proj",
                "image_token_embedder",
                "backbone_flow_time_embedder",
            )
        }
        special_embeddings = payload.get("special_token_embeddings")
        special_ids = payload.get("special_token_ids")
        if special_embeddings and special_ids:
            with torch.no_grad():
                embedding = model.model.embed_tokens.weight
                for name, token_id in special_ids.items():
                    if name in special_embeddings:
                        embedding[int(token_id)].copy_(
                            special_embeddings[name].to(
                                dtype=embedding.dtype
                            )
                        )
            report["loaded_special_token_embeddings"] = sorted(
                special_embeddings
            )

    if states["image_flow_head"]:
        report["image_flow_head"] = _load_compatible_state(
            model.image_flow_head,
            states["image_flow_head"],
        )
    projector = getattr(model, "image_flow_condition_proj", None)
    if projector is not None and states["image_flow_condition_proj"]:
        report["image_flow_condition_proj"] = _load_compatible_state(
            projector,
            states["image_flow_condition_proj"],
        )
    if states["image_token_embedder"]:
        missing, unexpected = model.image_token_embedder.load_state_dict(
            states["image_token_embedder"],
            strict=False,
        )
        report["image_token_embedder_missing"] = list(missing)
        report["image_token_embedder_unexpected"] = list(unexpected)

    time_embedder = getattr(
        model.model,
        "backbone_flow_time_embedder",
        None,
    )
    if time_embedder is not None:
        if not states["backbone_flow_time_embedder"]:
            raise ValueError(
                "Dynamic-XT adapter is missing "
                "backbone_flow_time_embedder"
            )
        time_embedder.load_state_dict(
            states["backbone_flow_time_embedder"],
            strict=True,
        )
        report["backbone_flow_time_embedder"] = len(
            states["backbone_flow_time_embedder"]
        )
    return report


def load_model_state(model, model_state_path: str | Path) -> dict:
    """Load an optional full-model state used by legacy evaluators."""

    if not model_state_path:
        return {"model_state": None}
    path = Path(model_state_path)
    if path.is_dir():
        candidate = (
            path / "pytorch_model" / "mp_rank_00_model_states.pt"
        )
        if candidate.is_file():
            path = candidate
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu")
    state = (
        payload["module"]
        if isinstance(payload, dict) and "module" in payload
        else payload
    )
    missing, unexpected = model.load_state_dict(state, strict=False)
    return {
        "model_state": str(path),
        "keys": len(state),
        "missing": list(missing),
        "unexpected": list(unexpected),
    }


def load_vae(config, device, dtype_name: str):
    module_root = Path(
        config.experiment.get(
            "validation_vae_module_root",
            "/inspire/hdd/global_user/wanjiaxin-253108030048/code/mar",
        )
    )
    checkpoint = Path(config.experiment.validation_vae_path)
    module_path = module_root / "models" / "vae.py"
    spec = importlib.util.spec_from_file_location("kl16_vae", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load KL16 VAE module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    vae = module.AutoencoderKL(
        embed_dim=16,
        ch_mult=(1, 1, 2, 2, 4),
        ckpt_path=str(checkpoint),
    )
    dtype = (
        torch.float16
        if dtype_name == "fp16" and device.type in {"cuda", "npu"}
        else torch.float32
    )
    vae = vae.to(device=device, dtype=dtype).eval()
    for parameter in vae.parameters():
        parameter.requires_grad_(False)
    return vae


def decode_latents(vae, latents, scaling_factor: float):
    vae_dtype = next(vae.parameters()).dtype
    decoded = (
        vae.decode(latents.to(dtype=vae_dtype) / scaling_factor)
        .float()
        .clamp(-1, 1)
    )
    return (decoded + 1.0) / 2.0
