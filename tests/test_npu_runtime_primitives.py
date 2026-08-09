"""Targeted distributed checks for Ascend training-runtime bookkeeping."""

from __future__ import annotations

import os

import torch
import torch_npu  # noqa: F401
from accelerate import Accelerator


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
            f"memory_dtype={gathered_memory.dtype} elapsed_dtype={gathered_elapsed.dtype}"
        )


if __name__ == "__main__":
    main()
