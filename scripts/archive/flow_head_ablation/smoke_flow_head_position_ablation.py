#!/usr/bin/env python3
"""CUDA smoke for every full-size FH0--FH4 flow-head position path."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from models.modeling_model.image_flow_loss import ContextualFlowTransformerHead
from models.modeling_model.image_flow_position import FLOW_HEAD_POSITION_SPECS


def run_variant(variant: str) -> dict[str, object]:
    spec = FLOW_HEAD_POSITION_SPECS[variant]
    torch.manual_seed(42)
    head = ContextualFlowTransformerHead(
        in_channels=16,
        model_channels=1280,
        out_channels=16,
        z_channels=1024,
        num_res_blocks=8,
        mlp_ratio=1.0,
        latent_mixer_heads=8,
        image_tokens_per_img=256,
        query_position_mode=spec.query_position_mode,
        context_position_mode=spec.context_position_mode,
        rope_mode=spec.rope_mode,
        rope_axis_dims=(80, 80),
        position_variant=variant,
    ).to(device="cuda", dtype=torch.bfloat16)
    head.train()
    generator = torch.Generator(device="cuda").manual_seed(17)
    x = torch.randn(1, 8, 16, generator=generator, device="cuda", dtype=torch.bfloat16)
    timestep = torch.rand(1, 8, generator=generator, device="cuda")
    condition = torch.randn(
        1, 8, 1024, generator=generator, device="cuda", dtype=torch.bfloat16
    )
    context = torch.randn(
        1, 16, 16, generator=generator, device="cuda", dtype=torch.bfloat16
    )
    query_positions = torch.tensor(
        [[0, 1, 17, 18, 85, 86, 254, 255]], device="cuda"
    )
    context_positions = torch.arange(16, device="cuda").unsqueeze(0)
    context_mask = torch.ones(1, 8, 16, device="cuda", dtype=torch.bool)
    context_mask[:, 0] = False
    started = time.perf_counter()
    output = head(
        x,
        timestep,
        condition,
        context_latents=context,
        context_mask=context_mask,
        query_positions=query_positions,
        context_positions=context_positions,
    )
    if not bool(torch.isfinite(output).all().item()):
        raise FloatingPointError(f"{variant} produced non-finite CUDA output")
    output.float().square().mean().backward()
    cache = head.prepare_latent_mixer_cache(
        context,
        context_mask,
        context_positions,
    )
    expected_rotations = int(spec.uses_rope)
    if any(
        int(layer["k_rotation_count"]) != expected_rotations
        for layer in cache["layers"]
    ):
        raise AssertionError(f"{variant} K rotation count drifted")
    if any(
        layer["k"].shape != layer["v"].shape for layer in cache["layers"]
    ):
        raise AssertionError(f"{variant} K/V cache shapes differ")
    elapsed = time.perf_counter() - started
    result = {
        "variant": variant,
        "position_contract": head.position_contract(),
        "parameter_count": sum(value.numel() for value in head.parameters()),
        "finite": True,
        "k_rotation_count": expected_rotations,
        "elapsed_seconds": elapsed,
        "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated() / (1024**2),
    }
    del head, output, cache
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("FH smoke requires CUDA")
    torch.backends.cuda.matmul.allow_tf32 = True
    results = [run_variant(variant) for variant in FLOW_HEAD_POSITION_SPECS]
    counts = {result["parameter_count"] for result in results}
    if len(counts) != 1:
        raise AssertionError(f"FH parameter counts differ: {counts}")
    payload = {
        "schema": "selfless_flow_head_position_cuda_smoke_v1",
        "torch_version": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(),
        "results": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(output.resolve())


if __name__ == "__main__":
    main()
