#!/usr/bin/env python3
"""Distributed Ascend smoke test for the FID/IS sufficient statistics."""

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401

from scripts.evaluate_single_stream_fid_is import (
    distributed_barrier,
    init_distributed,
    reduce_max,
    reduce_sum,
)
from scripts.image_evaluation_metrics import (
    FeatureMoments,
    InceptionScoreMoments,
)


def main() -> None:
    distributed, rank, world_size, _, device = init_distributed("npu")
    assert distributed
    assert device.type == "npu"
    assert dist.get_backend() == "hccl"

    features = FeatureMoments.zeros(4, device)
    features.update(
        torch.full((2, 4), float(rank + 1), device=device)
    )
    scores = InceptionScoreMoments.zeros(world_size, 3, device)
    scores.update(
        torch.tensor([[1.0, 0.0, -1.0]], device=device),
        [rank],
        world_size,
    )
    features.all_reduce_()
    scores.all_reduce_()

    expected_count = 2 * world_size
    expected_sum = world_size * (world_size + 1)
    assert int(features.count.item()) == expected_count
    assert torch.allclose(
        features.sum,
        torch.full((4,), float(expected_sum), device=device),
    )
    assert int(scores.count.sum().item()) == world_size
    assert reduce_sum(rank + 1, device) == float(
        world_size * (world_size + 1) // 2
    )
    assert reduce_max(rank + 1, device) == float(world_size)
    distributed_barrier(distributed, device)
    if rank == 0:
        print(
            "PASS npu_fid_is_metrics "
            f"world={world_size} backend={dist.get_backend()} "
            f"dtype={features.sum.dtype}",
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
