"""Targeted distributed checks for Ascend training-runtime bookkeeping."""

from __future__ import annotations

import os

import tbe  # noqa: F401
import torch
import torch.distributed as dist
import torch_npu  # noqa: F401
from accelerate import Accelerator

from utils.selfless_training_runtime import TrainingWindow


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    accelerator = Accelerator()
    rank = accelerator.process_index
    memory = torch.tensor(
        [10_000_000_000 + rank, 20_000_000_000 + rank],
        device=accelerator.device,
        dtype=torch.int64,
    )
    elapsed = torch.tensor(
        [1.25 + rank],
        device=accelerator.device,
        dtype=torch.float32,
    )
    gathered_memory = accelerator.gather(memory).reshape(-1, 2).cpu()
    gathered_elapsed = accelerator.gather(elapsed).reshape(-1).cpu()
    expected_memory = torch.tensor(
        [[10_000_000_000 + i, 20_000_000_000 + i] for i in range(accelerator.num_processes)],
        dtype=torch.int64,
    )
    expected_elapsed = torch.tensor(
        [1.25 + i for i in range(accelerator.num_processes)],
        dtype=torch.float32,
    )
    if not torch.equal(gathered_memory, expected_memory):
        raise AssertionError((gathered_memory, expected_memory))
    if not torch.equal(gathered_elapsed, expected_elapsed):
        raise AssertionError((gathered_elapsed, expected_elapsed))

    all_reduce_value = torch.tensor(
        [rank + 1.0], device=accelerator.device, dtype=torch.float32
    )
    dist.all_reduce(all_reduce_value, op=dist.ReduceOp.SUM)
    expected_sum = accelerator.num_processes * (accelerator.num_processes + 1) / 2
    if float(all_reduce_value.item()) != expected_sum:
        raise AssertionError((float(all_reduce_value.item()), expected_sum))

    window = TrainingWindow()
    window.record_batch(
        rows=rank + 1,
        sequence_length=320,
        logical_images=rank + 2,
        pack_stats=(100 + rank, 64 + rank, 20 + rank, 320),
        data_wait_seconds=0.001 * (rank + 1),
    )
    window.record_optimizer_step()
    window_tensor = window.as_tensor(accelerator.device)
    if window_tensor.dtype != torch.float32:
        raise AssertionError(window_tensor.dtype)
    gathered_windows = accelerator.gather(window_tensor).reshape(-1, 10).cpu()
    if not torch.equal(
        gathered_windows[:, 0],
        torch.ones(accelerator.num_processes, dtype=torch.float32),
    ):
        raise AssertionError(gathered_windows[:, 0])
    if not torch.equal(
        gathered_windows[:, 3],
        torch.arange(
            1,
            accelerator.num_processes + 1,
            dtype=torch.float32,
        ),
    ):
        raise AssertionError(gathered_windows[:, 3])

    torch.npu.manual_seed(123 + rank)
    state_before = torch.npu.get_rng_state().clone()
    with torch.random.fork_rng(devices=[local_rank], device_type="npu"):
        torch.npu.manual_seed(999 + rank)
        torch.rand(8, device=accelerator.device)
    if not torch.equal(state_before.cpu(), torch.npu.get_rng_state().cpu()):
        raise AssertionError("NPU RNG state was not restored")
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print(
            f"PASS world={accelerator.num_processes} "
            f"memory_dtype={gathered_memory.dtype} "
            f"elapsed_dtype={gathered_elapsed.dtype} "
            f"window_dtype={gathered_windows.dtype} "
            f"hccl_all_reduce_sum={float(all_reduce_value.item()):g}"
        )


if __name__ == "__main__":
    main()
