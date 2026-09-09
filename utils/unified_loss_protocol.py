"""The shared statistical contract of mixed-source training and loss validation."""
from __future__ import annotations

from collections import Counter
import math

import torch

LOSS_PROTOCOL = "unified_schedule_microbatch_mean_v1"


def loss_protocol(config):
    schedule = [str(source) for source in config.dataset.params.schedule]
    accumulation = int(config.training.gradient_accumulation_steps)
    if not schedule or accumulation != len(schedule):
        raise ValueError("loss protocol requires one complete source schedule per accumulation window")
    counts = Counter(schedule)
    if set(counts) - {"t2i", "i2t", "climbmix"}:
        raise ValueError("unsupported loss source")
    weights = {"t2i": float(config.model.lambda_image),
               "i2t": float(config.model.lambda_text), "climbmix": float(config.model.lambda_text)}
    if any(not math.isfinite(weights[s]) or weights[s] <= 0 for s in counts):
        raise ValueError("every active loss source must have a positive finite model weight")
    batches = {s: int(config.dataset.params.sources[s].micro_batch_size) for s in counts}
    if any(size <= 0 for size in batches.values()):
        raise ValueError("loss protocol requires positive micro_batch_sizes")
    return {"name": LOSS_PROTOCOL, "schedule": schedule, "gradient_accumulation_steps": accumulation,
            "model_weights": {s: weights[s] for s in counts},
            "source_fractions": {s: n / accumulation for s, n in counts.items()},
            "micro_batch_sizes": batches,
            "raw_aggregation": "sum(task_loss * valid_targets) / sum(valid_targets)",
            "total_aggregation": "sum(source_fraction * mean(model.loss per source microbatch))"}


def source_loss_metric_payload(reduced_source_totals: torch.Tensor, *, num_processes: int,
                               gradient_accumulation_steps: int, active_sources, prefix: str):
    """Rows: target-weighted raw sum, targets, sum of weighted batch losses.

    Training passes one optimizer window reduced across ranks. Validation
    converts each source's measured batch mean to its number of schedule slots,
    then uses this same reducer with a single logical window.
    """
    if reduced_source_totals.numel() != len(active_sources) * 3:
        raise ValueError("source loss totals must contain three values per source")
    denominator = int(num_processes) * int(gradient_accumulation_steps)
    if denominator <= 0:
        raise ValueError("num_processes * gradient_accumulation_steps must be positive")
    rows = reduced_source_totals.reshape(len(active_sources), 3)
    logs, display = {}, {}
    for index, source in enumerate(active_sources):
        targets = float(rows[index, 1].item())
        if not math.isfinite(targets) or targets <= 0:
            raise RuntimeError(f"mixed source {source!r} produced no optimization targets")
        raw = float((rows[index, 0] / rows[index, 1]).item())
        contribution = float((rows[index, 2] / denominator).item())
        if not math.isfinite(raw) or not math.isfinite(contribution):
            raise FloatingPointError(f"non-finite source metric for {source}")
        logs[f"{prefix}/loss_{source}"] = raw
        logs[f"{prefix}/weighted_contribution_{source}"] = contribution
        logs[f"{prefix}/{source}_target_tokens"] = targets
        display[source] = (raw, contribution)
    return logs, display
