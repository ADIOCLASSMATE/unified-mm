#!/usr/bin/env python3
"""Time the actual training validation runner, including model load and EMA swap."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.distributed as dist
from omegaconf import OmegaConf

from utils.evaluation.multimodal_likelihood import atomic_write_text, initialize_device
from utils.evaluation_model_source import (
    add_model_source_argument, configure_model_source, load_model_source_weights, model_source_from_args,
)
from utils.sharded_ema import RankShardedEMA, build_sharded_ema_layout
from utils.training_downstream_validation import ValidationProfile, run_downstream_validation
from utils.utils import load_model_tokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/selfless/unified_baseline_100b_ascend_64npu.yaml"))
    add_model_source_argument(parser)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "npu", "cuda"), default="npu")
    args = parser.parse_args()
    rank, world, _, device = initialize_device(args.device)
    if args.device == "npu" and world != 16:
        raise ValueError("timing acceptance must use exactly 16 NPUs")
    started = time.monotonic()
    if device.type in {"npu", "cuda"}:
        getattr(torch, device.type).reset_peak_memory_stats(device)
    source = model_source_from_args(args)
    config = OmegaConf.load(args.config)
    configure_model_source(config, source)
    model, tokenizer = load_model_tokenizer(config, model_dtype=torch.bfloat16)
    load_model_source_weights(model, source)
    # DeepSpeed BF16 calls module.bfloat16(), including nonpersistent RoPE
    # buffers. from_pretrained(dtype=bf16) alone leaves those buffers FP32.
    model.to(device=device, dtype=torch.bfloat16).train()
    layout = build_sharded_ema_layout(model, world_size=world)
    ema = RankShardedEMA(layout, rank=rank, decay=0.9999, update_after_step=0)
    ema.bind(model)
    ema.initialize_from_model(global_step=source.global_step)
    # The EMA contains the real checkpoint; make live weights observably different.
    parameter = next(model.parameters())
    with torch.no_grad():
        parameter.flatten()[0].add_(0.125)
    live_value = parameter.flatten()[0].clone()
    summary = run_downstream_validation(
        model, tokenizer, device=device, output_dir=args.output_dir, step=source.global_step,
        ema=ema, profile=ValidationProfile.from_config(config), started=started,
        weight_source=str(source.path),
    )
    assert torch.equal(parameter.flatten()[0], live_value), "EMA validation changed training weights"
    assert model.training, "validation failed to restore training mode"
    benchmark = {"ema_restore_verified": True, "world_size": world}
    if device.type in {"npu", "cuda"}:
        backend = getattr(torch, device.type)
        memory = torch.tensor([backend.max_memory_allocated(device), backend.max_memory_reserved(device)],
                              device=device, dtype=torch.float32)
        if dist.is_initialized():
            dist.all_reduce(memory, op=dist.ReduceOp.MAX)
        benchmark.update(zip(("peak_allocated_bytes", "peak_reserved_bytes"), memory.cpu().tolist()))
    summary["benchmark"] = benchmark
    if rank == 0:
        atomic_write_text(args.output_dir / "summary.json", json.dumps(summary, indent=2) + "\n")
        print(json.dumps({"event": "timing_acceptance", "passed": summary["complete"] and summary["within_time_budget"],
                          "wall_seconds": summary["wall_seconds"], **benchmark}), flush=True)
    if dist.is_initialized():
        dist.destroy_process_group()
    if not summary["complete"] or not summary["within_time_budget"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
