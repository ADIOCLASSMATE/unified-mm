#!/usr/bin/env python3
"""Tiny CUDA forward/backward smoke for all six joint-ablation cells."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from scripts.archive.backbone_flow_head_joint_ablation.backbone_flow_head_joint_ablation import (
    BACKBONE_VARIANTS,
    FLOW_POSITION_VARIANTS,
    cell_id,
)
from scripts.unified_smoke import (
    patch_cpu_runtime,
    run_train_step,
    synthetic_batch,
    tiny_config,
)
from models.modeling_model.modeling_selfless_flow import Qwen3ForCausalLM
from pretrain.train_selfless_flow import (
    _validation_flat_query_mixer_context,
    _validation_sequence_mixer_context,
)


@torch.no_grad()
def run_joint_validation(model, batch) -> dict:
    model.eval()
    output = model(
        X0_input_ids=batch["input_ids"],
        labels=batch["labels"],
        attention_mask=object(),
        token_types=batch["token_types"],
        image_latents=batch["image_latents"],
        calculate_likelihood=True,
    )
    if not torch.isfinite(output.loss):
        raise RuntimeError("Joint smoke validation loss is not finite.")
    condition = model._prepare_image_flow_condition(
        output.last_hidden_state[0, 2:6]
    )
    target_tokens = batch["image_latents"][0, 2:6]
    span_sigma = batch["sigma"][0, 2:6].float()
    local_positions = torch.arange(4, device=target_tokens.device)
    full_sample = model.sample_image_flow_with_cfg(
        condition,
        temperature=1.0,
        cfg=1.0,
        **_validation_flat_query_mixer_context(
            target_tokens,
            span_sigma,
            local_positions,
            condition,
        ),
    )
    probe_velocity = model.image_flow_head.velocity(
        target_tokens.unsqueeze(0),
        torch.full((1, 4), 0.5, device=target_tokens.device),
        condition.unsqueeze(0),
        **_validation_sequence_mixer_context(
            target_tokens,
            span_sigma,
            local_positions,
            condition,
        ),
    )
    single_stream, trace = model.sample_image_latents_single_stream(
        input_ids=batch["input_ids"],
        token_types=batch["token_types"],
        sigma=batch["sigma"],
        spans=[(0, 2, 6)],
        image_latent_dim=4,
        flow_temperature=1.0,
        flow_cfg=1.5,
        flow_cfg_schedule="linear",
        parallel_rate=1,
        order_strategy="sigma",
        return_trace=True,
    )
    if tuple(full_sample.shape) != (4, 4):
        raise RuntimeError(
            f"Unexpected full sample shape: {tuple(full_sample.shape)}"
        )
    if tuple(single_stream.shape) != (1, 4, 2, 2):
        raise RuntimeError(
            f"Unexpected single-stream shape: {tuple(single_stream.shape)}"
        )
    if not torch.isfinite(probe_velocity).all():
        raise RuntimeError("Validation probe velocity is not finite.")
    target = (
        batch["image_latents"][0, 2:6]
        .view(2, 2, 4)
        .permute(2, 0, 1)
        .unsqueeze(0)
    )
    return {
        "loss": float(output.loss.item()),
        "full_sample_rms": float(
            full_sample.float().pow(2).mean().sqrt().item()
        ),
        "probe_velocity_rms": float(
            probe_velocity.float().pow(2).mean().sqrt().item()
        ),
        "single_stream_mse_to_target": float(
            F.mse_loss(single_stream.float(), target.float()).item()
        ),
        "single_stream_steps": int(trace["generation_step"].max().item()),
    }


def run_cell(backbone: str, position: str, device: torch.device) -> dict:
    torch.manual_seed(42)
    config = tiny_config()
    config.image_backbone_variant = backbone
    config.image_flow_head_arch = "contextual"
    config.image_flow_head_variant = "DF1"
    config.image_flow_position_variant = position
    config.image_flow_latent_mixer_heads = 2
    model = Qwen3ForCausalLM(config).to(device)
    batch = synthetic_batch(device)
    train_loss = run_train_step(model, batch)
    validation = run_joint_validation(model, batch)
    return {
        "cell_id": cell_id(backbone, position),
        "train_loss": train_loss,
        "validation": validation,
        "flow_head_parameters": sum(
            parameter.numel()
            for parameter in model.image_flow_head.parameters()
        ),
        "finite": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA smoke requested but CUDA is unavailable.")
    device = torch.device(args.device)
    patch_cpu_runtime()
    rows = [
        run_cell(backbone, position, device)
        for backbone in BACKBONE_VARIANTS
        for position in FLOW_POSITION_VARIANTS
    ]
    parameter_counts = {row["flow_head_parameters"] for row in rows}
    if len(parameter_counts) != 1:
        raise RuntimeError(
            f"Flow-head parameter counts differ: {sorted(parameter_counts)}"
        )
    report = {
        "schema": "selfless_backbone_flow_head_joint_smoke_v1",
        "device": str(device),
        "cells": rows,
    }
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite smoke: {output}")
        output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
