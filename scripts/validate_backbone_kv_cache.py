#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import torch
import torch_npu  # noqa: F401
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.generate_flow_validation_images import (
    load_sharded_ema_checkpoint,
)
from utils.dataset_utils import get_dataloaders
from utils.utils import load_model_tokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare full-sequence and incremental-backbone-cache image "
            "generation using the same checkpoint and initial noise."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--ema_checkpoint", required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--model_dtype",
        choices=("bf16", "fp32"),
        default="fp32",
        help=(
            "Use FP32 for semantic cache/full equivalence; BF16 may diverge "
            "because Q=full and Q=1/2 FlexAttention kernels have different "
            "reduction numerics."
        ),
    )
    parser.add_argument("--sampling_steps", type=int, default=1)
    parser.add_argument("--max_generation_steps", type=int, default=0)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--cfg", type=float, default=3.5)
    parser.add_argument(
        "--mode", choices=("both", "full", "cache"), default="both"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--atol", type=float, default=2.0e-3)
    parser.add_argument("--rtol", type=float, default=2.0e-3)
    parser.add_argument("--output", default="")
    return parser.parse_args()


def image_spans(token_types: torch.Tensor, image_tokens: int):
    spans = []
    for row in range(token_types.shape[0]):
        positions = (token_types[row] == 1).nonzero(as_tuple=True)[0]
        if positions.numel() != image_tokens:
            raise ValueError(
                f"row {row} has {positions.numel()} image tokens; "
                f"expected {image_tokens}"
            )
        start = int(positions[0].item())
        end = int(positions[-1].item()) + 1
        if end - start != image_tokens:
            raise ValueError(f"row {row} image span is not contiguous")
        spans.append((row, start, end))
    return spans


@torch.no_grad()
def main():
    args = parse_args()
    if not torch.npu.is_available():
        raise RuntimeError("This checkpoint validation requires Ascend NPU.")
    device = torch.device("npu", 0)
    torch.npu.set_device(device)
    config = OmegaConf.load(args.config)
    config.training.batch_size = int(args.batch_size)
    config.training.dataloader_workers = 0
    config.model.image_flow_num_sampling_steps = str(args.sampling_steps)

    model, tokenizer = load_model_tokenizer(
        config,
        model_dtype={
            "bf16": torch.bfloat16,
            "fp32": torch.float32,
        }[args.model_dtype],
    )
    checkpoint_report = load_sharded_ema_checkpoint(
        model,
        args.ema_checkpoint,
    )
    model = model.to(device=device).eval()
    _, val_loader = get_dataloaders(config, tokenizer)
    batch = next(iter(val_loader))
    input_ids = batch["input_ids"].to(device)
    token_types = batch["token_types"].to(device)
    sigma = batch["sigma"].to(device)
    image_tokens = int(config.model.image_tokens_per_img)
    latent_dim = int(config.model.image_latent_dim)
    spans = image_spans(token_types, image_tokens)
    noise_generator = torch.Generator(device="cpu")
    noise_generator.manual_seed(int(args.seed))
    initial_noise = torch.randn(
        len(spans),
        image_tokens,
        latent_dim,
        generator=noise_generator,
        dtype=torch.float32,
    ).to(device)

    common = {
        "input_ids": input_ids,
        "token_types": token_types,
        "sigma": sigma,
        "spans": spans,
        "initial_noise_bank": initial_noise,
        "flow_temperature": 1.0,
        "flow_cfg": float(args.cfg),
        "flow_cfg_schedule": "constant",
        "flow_solver": "euler",
        "flow_num_steps": int(args.sampling_steps),
        "parallel_rate": 1,
        "order_strategy": "spatial_halton",
        "return_trace": True,
        "_debug_max_generation_steps": (
            int(args.max_generation_steps)
            if int(args.max_generation_steps) > 0
            else None
        ),
    }

    def run(use_cache: bool, *, measured: bool = True):
        torch.npu.synchronize(device)
        baseline = torch.npu.memory_allocated(device)
        if measured:
            torch.npu.reset_peak_memory_stats(device)
        started = time.perf_counter()
        generated, trace = model.sample_image_latents_single_stream(
            **common,
            use_backbone_cache=use_cache,
        )
        torch.npu.synchronize(device)
        elapsed = time.perf_counter() - started
        peak_delta = (
            torch.npu.max_memory_allocated(device) - baseline
            if measured
            else 0
        )
        return generated, trace, elapsed, peak_delta

    if int(args.warmup_steps) > 0:
        previous_limit = common["_debug_max_generation_steps"]
        common["_debug_max_generation_steps"] = int(args.warmup_steps)
        if args.mode in {"both", "full"}:
            run(False, measured=False)
        if args.mode in {"both", "cache"}:
            run(True, measured=False)
        common["_debug_max_generation_steps"] = previous_limit

    if args.mode != "both":
        use_cache = args.mode == "cache"
        generated, trace, elapsed, peak_delta = run(use_cache)
        report = {
            "schema": "backbone_kv_cache_probe_v1",
            "config": str(Path(args.config).resolve()),
            "checkpoint": checkpoint_report,
            "device": torch.npu.get_device_name(device),
            "dtype": str(next(model.parameters()).dtype),
            "mode": str(args.mode),
            "batch_size": len(spans),
            "sampling_steps": int(args.sampling_steps),
            "max_generation_steps": int(args.max_generation_steps),
            "warmup_steps": int(args.warmup_steps),
            "cfg": float(args.cfg),
            "seconds": float(elapsed),
            "peak_delta_mib": float(peak_delta / (1024.0**2)),
            "generated_rms": float(
                generated.float().pow(2).mean().sqrt().item()
            ),
            "trace": {
                key: value
                for key, value in trace.items()
                if key.startswith("backbone_kv_cache")
            },
        }
        payload = json.dumps(report, indent=2, sort_keys=True)
        print(payload)
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(payload + "\n", encoding="utf-8")
        return

    full, full_trace, full_seconds, full_peak_delta = run(False)
    cached, cached_trace, cached_seconds, cached_peak_delta = run(True)
    difference = (cached.float() - full.float()).abs()
    hidden_errors = {}
    for stream in ("conditional", "unconditional"):
        trace_key = f"debug_{stream}_backbone_hidden"
        full_hidden = full_trace.get(trace_key)
        cached_hidden = cached_trace.get(trace_key)
        if isinstance(full_hidden, torch.Tensor) and isinstance(
            cached_hidden, torch.Tensor
        ):
            hidden_difference = (cached_hidden - full_hidden).abs()
            hidden_errors[f"{stream}_hidden_max_abs_error"] = float(
                hidden_difference.max().item()
            )
            hidden_errors[f"{stream}_hidden_mean_abs_error"] = float(
                hidden_difference.mean().item()
            )
            hidden_errors[f"{stream}_hidden_reference_rms"] = float(
                full_hidden.float().pow(2).mean().sqrt().item()
            )
    close = torch.allclose(
        cached.float(),
        full.float(),
        atol=float(args.atol),
        rtol=float(args.rtol),
    )
    report = {
        "schema": "backbone_kv_cache_validation_v1",
        "config": str(Path(args.config).resolve()),
        "checkpoint": checkpoint_report,
        "device": torch.npu.get_device_name(device),
        "dtype": str(next(model.parameters()).dtype),
        "batch_size": len(spans),
        "sampling_steps": int(args.sampling_steps),
        "max_generation_steps": int(args.max_generation_steps),
        "warmup_steps": int(args.warmup_steps),
        "cfg": float(args.cfg),
        "atol": float(args.atol),
        "rtol": float(args.rtol),
        "allclose": bool(close),
        "max_abs_error": float(difference.max().item()),
        "mean_abs_error": float(difference.mean().item()),
        **hidden_errors,
        "full_seconds": float(full_seconds),
        "cached_seconds": float(cached_seconds),
        "speedup": float(full_seconds / cached_seconds),
        "full_peak_delta_mib": float(full_peak_delta / (1024.0**2)),
        "cached_peak_delta_mib": float(cached_peak_delta / (1024.0**2)),
        "full_trace": {
            key: value
            for key, value in full_trace.items()
            if key.startswith("backbone_kv_cache")
        },
        "cached_trace": {
            key: value
            for key, value in cached_trace.items()
            if key.startswith("backbone_kv_cache")
        },
    }
    payload = json.dumps(report, indent=2, sort_keys=True)
    print(payload)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload + "\n", encoding="utf-8")
    if not close:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
