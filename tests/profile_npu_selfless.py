"""Short Level-1 torch_npu profile of the test-only production benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
import torch_npu


sys.path.insert(0, str(Path(__file__).resolve().parent))
import cross_device_npu_benchmark as benchmark  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--component", choices=("attention", "flow", "model"), default="model")
    parser.add_argument("--flow-mul", type=int, default=4)
    args = parser.parse_args()

    torch.npu.set_device(0)
    device = torch.device("npu", 0)
    fixture = torch.load(args.fixture, map_location="cpu", weights_only=True)
    if args.component == "attention":
        step, _capture = benchmark._build_attention_step(fixture, device)
    elif args.component == "flow":
        step, _capture = benchmark._build_flow_step(fixture, device)
    else:
        step, _capture = benchmark._build_model_step(
            fixture,
            device,
            args.flow_mul,
            distributed=False,
            local_rank=0,
        )

    # Initialize lazy kernels and optimizer state outside the profiling window.
    step()
    torch.npu.synchronize()
    args.output.mkdir(parents=True, exist_ok=True)
    experimental_config = torch_npu.profiler._ExperimentalConfig(
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        l2_cache=False,
        data_simplification=False,
    )
    schedule = torch_npu.profiler.schedule(wait=0, warmup=1, active=2, repeat=1)
    handler = torch_npu.profiler.tensorboard_trace_handler(
        str(args.output),
        analyse_flag=True,
        async_mode=False,
    )
    with torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU,
        ],
        schedule=schedule,
        on_trace_ready=handler,
        record_shapes=True,
        profile_memory=False,
        with_stack=False,
        experimental_config=experimental_config,
    ) as profiler:
        for _ in range(3):
            step()
            profiler.step()
    torch.npu.synchronize()
    print(f"profile={args.output}")


if __name__ == "__main__":
    main()
