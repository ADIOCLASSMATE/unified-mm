"""Run with torchrun on the development Notebook, for Gloo or HCCL I/O phases."""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.distributed as dist

from utils.distributed_io import run_io_phase


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("hccl", "gloo"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.backend == "hccl":
        import torch_npu  # noqa: F401
        torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group(args.backend, timeout=timedelta(seconds=90))
    rank, world = dist.get_rank(), dist.get_world_size()
    accelerator = SimpleNamespace(process_index=rank, num_processes=world, is_main_process=rank == 0)
    checks = []
    try:
        for main_only, failure_rank in ((False, world - 1), (True, 0)):
            def fail():
                if rank == failure_rank:
                    raise OSError("injected rank write failure")
            try:
                run_io_phase(accelerator, fail, description="failure injection", main_process_only=main_only)
            except RuntimeError as exc:
                checks.append("injected rank write failure" in str(exc))
            else:
                checks.append(False)
        # The group must remain usable after a filesystem failure.
        result = run_io_phase(accelerator, lambda: "published", description="recovered publication", main_process_only=True)
        checks.append(result == "published")
        reports = [None] * world
        dist.all_gather_object(reports, {"rank": rank, "passed": all(checks)})
        assert all(row["passed"] for row in reports)
        if rank == 0:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps({"passed": True, "backend": args.backend, "world_size": world,
                                              "rank_reports": reports}, indent=2) + "\n")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
