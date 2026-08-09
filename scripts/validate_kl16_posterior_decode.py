#!/usr/bin/env python3
"""Check one cached KL16 posterior and decode its mean on an accelerator."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path

import torch
from PIL import Image


def load_vae_class(module_root: Path):
    module_path = module_root / "models" / "vae.py"
    spec = importlib.util.spec_from_file_location("mar_kl16_decode_vae", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load VAE module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.AutoencoderKL


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--vae_module_root", default="public/code/mar")
    parser.add_argument("--vae_path", default="public/vae/mar-kl16/kl16.ckpt")
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--vae_dtype", choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--row", type=int, default=0)
    parser.add_argument("--scaling_factor", type=float, default=0.2325)
    parser.add_argument("--output_image", required=True)
    parser.add_argument("--output_json", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "npu":
        import torch_npu  # noqa: F401

        torch.npu.set_device(0 if device.index is None else device.index)
    dtype = torch.float16 if args.vae_dtype == "fp16" else torch.float32

    payload = torch.load(
        args.cache,
        map_location="cpu",
        mmap=True,
        weights_only=True,
    )
    stats = payload["posterior_stats"][args.row].float()
    image_id = int(payload["img_ids"][args.row].item())
    mean = stats[..., :16]
    std = stats[..., 16:]
    if not bool(torch.isfinite(stats).all()) or bool((std < 0).any()):
        raise RuntimeError("posterior stats are non-finite or contain negative std")

    AutoencoderKL = load_vae_class(Path(args.vae_module_root))
    vae = AutoencoderKL(
        embed_dim=16,
        ch_mult=(1, 1, 2, 2, 4),
        ckpt_path=args.vae_path,
    )
    vae = vae.to(device=device, dtype=dtype).eval()
    for parameter in vae.parameters():
        parameter.requires_grad_(False)
    latent = (
        mean.reshape(16, 16, 16)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device=device, dtype=dtype)
        / args.scaling_factor
    )
    with torch.inference_mode():
        decoded = vae.decode(latent).float()
    if device.type == "npu":
        torch.npu.synchronize(device)
    if tuple(decoded.shape) != (1, 3, 256, 256) or not bool(
        torch.isfinite(decoded).all()
    ):
        raise RuntimeError(f"invalid VAE decode: shape={tuple(decoded.shape)}")

    pixels = ((decoded[0].clamp(-1, 1) + 1) * 127.5).round().byte()
    pixels = pixels.permute(1, 2, 0).cpu().numpy()
    output_image = Path(args.output_image)
    output_image.parent.mkdir(parents=True, exist_ok=True)
    temporary_image = output_image.with_suffix(output_image.suffix + ".tmp")
    Image.fromarray(pixels).save(temporary_image, format="PNG")
    os.replace(temporary_image, output_image)

    report = {
        "status": "ok",
        "cache": str(args.cache),
        "row": args.row,
        "img_id": image_id,
        "device": str(device),
        "vae_dtype": str(dtype).removeprefix("torch."),
        "posterior_mean": {
            "min": float(mean.min().item()),
            "max": float(mean.max().item()),
            "mean": float(mean.mean().item()),
            "std": float(mean.std().item()),
            "finite": bool(torch.isfinite(mean).all()),
        },
        "posterior_std": {
            "min": float(std.min().item()),
            "max": float(std.max().item()),
            "mean": float(std.mean().item()),
            "std": float(std.std().item()),
            "finite": bool(torch.isfinite(std).all()),
            "non_negative": bool((std >= 0).all()),
        },
        "decode": {
            "shape": list(decoded.shape),
            "min": float(decoded.min().item()),
            "max": float(decoded.max().item()),
            "mean": float(decoded.mean().item()),
            "std": float(decoded.std().item()),
            "finite": bool(torch.isfinite(decoded).all()),
            "output_image": str(output_image),
        },
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    temporary_json = output_json.with_suffix(output_json.suffix + ".tmp")
    temporary_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_json, output_json)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
