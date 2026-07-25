#!/usr/bin/env python3
"""Run the formal evaluator with the missing ``Mapping`` name available.

The formal DF checkpoints bind the byte-for-byte evaluator source in their
runtime provenance.  The bound evaluator completed generation but referenced
``Mapping`` without importing it while aggregating guidance diagnostics.
Keeping the bound file unchanged preserves that provenance; this narrow
launcher supplies the missing built-in name for evaluation-only recovery.
"""

from collections.abc import Mapping
import builtins
from datetime import timedelta
import os
from pathlib import Path
import runpy
import sys

import torch.distributed as dist


DEFAULT_PROCESS_GROUP_TIMEOUT_SECONDS = 2 * 60 * 60


def install_process_group_timeout() -> int:
    """Allow slow evaluation ranks to rendezvous after long 10K sampling."""

    timeout_seconds = int(
        os.environ.get(
            "EVAL_PROCESS_GROUP_TIMEOUT_SECONDS",
            DEFAULT_PROCESS_GROUP_TIMEOUT_SECONDS,
        )
    )
    if timeout_seconds <= 0:
        raise ValueError(
            "EVAL_PROCESS_GROUP_TIMEOUT_SECONDS must be positive, "
            f"got {timeout_seconds}"
        )
    original_init_process_group = dist.init_process_group

    def init_process_group_with_evaluation_timeout(*args, **kwargs):
        kwargs.setdefault("timeout", timedelta(seconds=timeout_seconds))
        return original_init_process_group(*args, **kwargs)

    dist.init_process_group = init_process_group_with_evaluation_timeout
    return timeout_seconds


def main() -> None:
    evaluator = Path(__file__).with_name("evaluate_single_stream_fid_is.py")
    builtins.Mapping = Mapping
    install_process_group_timeout()
    sys.argv[0] = str(evaluator)
    runpy.run_path(str(evaluator), run_name="__main__")


if __name__ == "__main__":
    main()
