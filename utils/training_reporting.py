"""Distributed runtime accounting and atomic reports for one training run."""
import json
from pathlib import Path

import torch

from utils.atomic_io import atomic_write_text
from utils.distributed_io import run_io_phase


def write_training_runtime_report(
    *,
    config,
    accelerator,
    model,
    training_runtime_elapsed,
    global_step,
    training_runtime_start_step,
    total_batch_size,
    finite_loss_microbatches_checked,
    last_logged_loss,
    trainability,
    ema_layout,
    cumulative_wall_seconds_before_run,
    cumulative_loss_checks_before_run,
):
    if accelerator.device.type == "npu":
        memory_backend = "npu"
        local_memory = torch.tensor(
            [
                int(torch.npu.max_memory_allocated(accelerator.device)),
                int(torch.npu.max_memory_reserved(accelerator.device)),
            ],
            device=accelerator.device,
            dtype=torch.int64,
        )
    elif accelerator.device.type == "cuda":
        memory_backend = "cuda"
        local_memory = torch.tensor(
            [
                int(torch.cuda.max_memory_allocated(accelerator.device)),
                int(torch.cuda.max_memory_reserved(accelerator.device)),
            ],
            device=accelerator.device,
            dtype=torch.int64,
        )
    else:
        memory_backend = accelerator.device.type
        local_memory = torch.zeros(
            2,
            device=accelerator.device,
            dtype=torch.int64,
        )
    local_elapsed = torch.tensor(
        [float(training_runtime_elapsed)],
        device=accelerator.device,
        dtype=torch.float32,
    )
    gathered_memory = accelerator.gather(local_memory).reshape(-1, 2)
    gathered_elapsed = accelerator.gather(local_elapsed).reshape(-1)
    memory_max = gathered_memory.max(dim=0).values
    elapsed_max = gathered_elapsed.max()
    def write_report():
        runtime_payload = {
            "schema": "selfless_training_runtime_metrics_v1",
            "global_step": int(global_step),
            "run_start_global_step": int(training_runtime_start_step),
            "world_size": int(accelerator.num_processes),
            "total_batch_size": int(total_batch_size),
            "steps_this_run": int(global_step - training_runtime_start_step),
            "finite_loss_microbatches_checked": int(
                finite_loss_microbatches_checked
            ),
            "last_logged_loss": last_logged_loss,
            "training_wall_seconds": float(elapsed_max.item()),
            "cumulative_training_wall_seconds": float(
                cumulative_wall_seconds_before_run + elapsed_max.item()
            ),
            "cumulative_finite_loss_microbatches_checked": int(
                cumulative_loss_checks_before_run
                + finite_loss_microbatches_checked
            ),
            "train_samples_per_second": float(
                (global_step - training_runtime_start_step)
                * total_batch_size
                / max(float(elapsed_max.item()), 1e-12)
            ),
            "memory_backend": memory_backend,
            "peak_memory_allocated_bytes_per_rank": int(
                memory_max[0].item()
            ),
            "peak_memory_reserved_bytes_per_rank": int(
                memory_max[1].item()
            ),
            "trainability": trainability,
        }
        flow_net = getattr(
            getattr(accelerator.unwrap_model(model), "image_flow_head", None),
            "net", None,
        )
        flow_batch_layout = getattr(flow_net, "last_training_batch_layout", None)
        if flow_batch_layout is not None:
            runtime_payload["flow_training_batch_layout"] = flow_batch_layout
        if ema_layout is not None:
            full_ema_bytes = int(
                sum(chunk["bytes"] for chunk in ema_layout["chunks"].values())
            )
            max_shard_bytes = int(max(ema_layout["rank_bytes"]))
            runtime_payload["ema"] = {
                "full_fp32_replica_bytes": full_ema_bytes,
                "shard_bytes_by_rank": ema_layout["rank_bytes"],
                "max_shard_bytes": max_shard_bytes,
                "minimum_bytes_saved_per_rank": full_ema_bytes
                - max_shard_bytes,
                "minimum_fraction_saved_per_rank": (
                    (full_ema_bytes - max_shard_bytes) / full_ema_bytes
                    if full_ema_bytes
                    else 0.0
                ),
            }
        runtime_root = Path(config.experiment.output_dir)
        runtime_paths = (
            runtime_root / "training_runtime_metrics.json",
            runtime_root
            / (
                "training_runtime_metrics_"
                f"step-{training_runtime_start_step}-to-{global_step}.json"
            ),
        )
        for runtime_path in runtime_paths:
            atomic_write_text(runtime_path, json.dumps(runtime_payload, indent=2, sort_keys=True) + "\n")

    run_io_phase(accelerator, write_report, description="training runtime report", main_process_only=True)
