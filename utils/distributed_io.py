"""Small filesystem phases with a shared outcome before the next collective."""
from __future__ import annotations

import torch.distributed as dist


def run_io_phase(accelerator, operation, *, description: str, main_process_only=False):
    """Propagate rank-local I/O failures to every participant.

    The operation must not contain unmatched collectives. Backend-internal
    collectives (e.g. DeepSpeed save_state) still depend on backend timeouts
    and the launcher to handle a killed process.
    """
    result = None
    error = None
    try:
        if not main_process_only or accelerator.is_main_process:
            result = operation()
    except Exception as exc:
        if int(getattr(accelerator, "num_processes", 1)) == 1:
            raise
        error = f"rank {accelerator.process_index}: {type(exc).__name__}: {exc}"
    world_size = int(getattr(accelerator, "num_processes", 1))
    if world_size == 1:
        return result
    outcomes = [None] * world_size
    # Only main-only phases share a result. Rank-local operations may return
    # tensors or large objects which must not be copied to every rank.
    dist.all_gather_object(outcomes, {"error": error, "result": result if main_process_only else None})
    errors = [item["error"] for item in outcomes if item["error"] is not None]
    if errors:
        raise RuntimeError(f"{description} failed: " + "; ".join(errors))
    return outcomes[0]["result"] if main_process_only else result
