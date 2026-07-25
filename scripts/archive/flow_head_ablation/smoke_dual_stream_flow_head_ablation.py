#!/usr/bin/env python3
"""Full-size CUDA forward/backward and incremental-cache smoke for DF1/DF2."""

from __future__ import annotations

import argparse
import json

import torch

from models.modeling_model.image_flow_loss import FlowLoss
from models.modeling_model.image_flow_position import FLOW_HEAD_POSITION_SPECS


def _head(
    variant: str,
    position: str,
    device: torch.device,
) -> FlowLoss:
    spec = FLOW_HEAD_POSITION_SPECS[position]
    torch.manual_seed(42)
    return FlowLoss(
        target_channels=16,
        z_channels=1024,
        depth=8,
        width=1280,
        num_sampling_steps=2,
        grad_checkpointing=False,
        time_scale=1000.0,
        time_sampling="logit_normal",
        uniform_mix=0.1,
        solver="heun",
        mlp_ratio=1.0,
        image_tokens_per_img=256,
        latent_mixer_heads=8,
        latent_mixer_dropout=0.0,
        latent_mixer_zero_init_gate=True,
        head_arch="contextual",
        position_variant=position,
        query_position_mode=spec.query_position_mode,
        context_position_mode=spec.context_position_mode,
        rope_mode=spec.rope_mode,
        rope_axis_dims=(80, 80),
        flow_head_variant=variant,
    ).to(device=device, dtype=torch.bfloat16)


def run_cell(variant: str, position: str, device: torch.device) -> dict:
    flow = _head(variant, position, device)
    torch.cuda.reset_peak_memory_stats()
    flow.train()
    generator = torch.Generator(device=device).manual_seed(1234)
    target = torch.randn(
        1, 256, 16, generator=generator, device=device, dtype=torch.bfloat16
    )
    context = target + 0.01 * torch.randn(
        target.shape, generator=generator, device=device, dtype=torch.bfloat16
    )
    condition = torch.randn(
        1, 256, 1024, generator=generator, device=device, dtype=torch.bfloat16
    )
    sigma = torch.randperm(256, generator=generator, device=device).view(1, 256)
    positions = torch.arange(256, device=device).view(1, 256)
    loss = flow(
        target=target,
        z=condition,
        sigma=sigma,
        image_positions=positions,
        context_latents=context,
    )
    if not bool(torch.isfinite(loss).item()):
        raise FloatingPointError(f"{variant} forward loss is non-finite")
    loss.backward()
    nonfinite_gradients = sum(
        int((~torch.isfinite(parameter.grad)).sum().item())
        for parameter in flow.parameters()
        if parameter.grad is not None
    )
    if nonfinite_gradients:
        raise FloatingPointError(
            f"{variant} has {nonfinite_gradients} non-finite gradients"
        )

    flow.eval()
    cache = flow.empty_latent_mixer_cache()
    cache = flow.append_latent_mixer_cache(
        cache,
        context_latents=context[:, 0],
        context_conditions=condition[:, 0],
        context_positions=positions[:, 0],
    )
    velocity = flow.velocity(
        target[:, 1],
        torch.full((1,), 0.5, device=device),
        condition[:, 1],
        query_positions=positions[:, 1],
        latent_mixer_cache=cache,
    )
    if not bool(torch.isfinite(velocity).all().item()):
        raise FloatingPointError(f"{variant} cached velocity is non-finite")

    return {
        "cell_id": f"{variant}-{position}",
        "variant": variant,
        "position_variant": position,
        "loss": float(loss.detach().float().item()),
        "flow_head_parameters": int(sum(p.numel() for p in flow.parameters())),
        "nonfinite_gradients": nonfinite_gradients,
        "cached_velocity_rms": float(
            velocity.detach().float().pow(2).mean().sqrt().item()
        ),
        "peak_cuda_allocated_mib": float(
            torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
        ),
        "cache_contract": flow.net.cache_contract(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the DF1/DF2 full-size smoke")
    device = torch.device("cuda", 0)
    rows = [
        run_cell(variant, position, device)
        for variant in ("DF1", "DF2")
        for position in ("FH0", "FH1", "FH4")
    ]
    if {row["flow_head_parameters"] for row in rows} != {164_072_976}:
        raise RuntimeError("full-size flow-head parameter count drifted")
    payload = {
        "schema": "selfless_dual_stream_flow_head_cuda_smoke_v1",
        "torch_version": torch.__version__,
        "device": torch.cuda.get_device_name(device),
        "rows": rows,
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        from pathlib import Path

        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
        print(path.resolve())
    else:
        print(text)


if __name__ == "__main__":
    main()
