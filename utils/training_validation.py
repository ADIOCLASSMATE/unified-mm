"""Current-weight loss/images and EMA downstream validation."""
from __future__ import annotations

import json
from pathlib import Path
import time

import torch.distributed as dist

from utils.training_downstream_validation import (
    DownstreamValidationState, ValidationProfile, _distributed, _local_phase,
    _reduce, _synchronize, run_downstream_validation,
)
from utils.training_unified_loss_validation import UnifiedLossValidator, UnifiedLossValidationProfile
from utils.training_image_generation import TrainingImageGenerator


class TrainingValidator:
    """Own the fixed sample plan and CPU caches for the lifetime of a run."""

    def __init__(self, config, *, image_loader=None, image_collators=None, imagenet_subset=None):
        self.loss = None
        if UnifiedLossValidationProfile.from_config(config).enabled:
            self.loss = UnifiedLossValidator(config, image_loader=image_loader,
                image_collators=image_collators, imagenet_subset=imagenet_subset)
        self.downstream = DownstreamValidationState(ValidationProfile.from_config(config),
            imagenet_subset=self.loss.imagenet_subset if self.loss is not None else imagenet_subset)
        self.generation = TrainingImageGenerator(config)

    def run(self, model, tokenizer, *, device, step, output_dir, forward_batch,
            ema=None, training_exclusion=None, training_shards=()):
        from utils.atomic_io import atomic_write_text

        _synchronize(device)
        started = time.monotonic()
        output_dir = Path(output_dir)
        loss_result = None
        metrics = {}
        if self.loss is not None:
            loss_result = self.loss.run(model, tokenizer, device=device, step=step,
                output_dir=output_dir, forward_batch=forward_batch,
                training_exclusion=training_exclusion, training_shards=training_shards)
            metrics.update(loss_result["metrics"])
            metrics["val/unified_loss_seconds"] = loss_result["wall_seconds"]
            metrics.update({f"val/unified_loss/{source}_seconds": row["wall_seconds"]
                            for source, row in loss_result["timings"].items()})
        generation = self.generation.run(model, tokenizer, device=device, step=step,
            output_dir=output_dir, ema=ema)
        if generation is not None:
            metrics.update({"val/generation_seconds": generation["wall_seconds"],
                            "val/generation_images": generation["samples"],
                            "val/generation_complete": int(generation["complete"])})
        downstream = run_downstream_validation(model, tokenizer, device=device, step=step,
            output_dir=output_dir / "downstream_validation" / f"step-{step}",
            ema=ema, state=self.downstream)
        seconds = float(_reduce([time.monotonic() - started], device, dist.ReduceOp.MAX)[0])
        complete = downstream["complete"] and (generation is None or generation["complete"])
        metrics.update({"val/validation_seconds": seconds,
                        "val/validation_complete": int(complete),
                        "val/downstream_complete": int(downstream["complete"]),
                        "val/downstream_seconds": downstream["wall_seconds"],
                        "val/downstream_within_budget": int(downstream["within_time_budget"]),
                        "val/downstream_prepare_seconds": downstream["prepare_seconds"],
                        "val/downstream_prepare_cache_hit": int(downstream["prepare_cache_hit"])})
        metrics.update({f"val/downstream/{task}": row["primary"]
                        for task, row in downstream["tasks"].items() if row["complete"]})
        if "text_mean" in downstream:
            metrics["val/downstream/text_mean"] = downstream["text_mean"]
        # Reference the detailed artifacts instead of copying all sample IDs
        # and loss records into another large per-step file.
        payload = {"schema": "training_validation_summary_v1", "step": int(step),
                   "complete": bool(complete), "wall_seconds": seconds,
                   "loss": None if loss_result is None else {
                       "path": f"validation_unified_loss_metrics_step_{step}.json",
                       "complete": loss_result["complete"], "wall_seconds": loss_result["wall_seconds"]},
                   "downstream": {"path": f"downstream_validation/step-{step}/summary.json",
                       "complete": downstream["complete"], "wall_seconds": downstream["wall_seconds"],
                       "within_time_budget": downstream["within_time_budget"],
                       "prepare_cache_hit": downstream["prepare_cache_hit"]},
                   "generation": None if generation is None else {
                       "path": f"validation_generation/step-{step}/summary.json",
                       "gallery": f"validation_generation/step-{step}/index.html",
                       "overview": f"validation_generation/step-{step}/overview.png",
                       "complete": generation["complete"], "samples": generation["samples"],
                       "weight_source": generation["weight_source"], "wall_seconds": generation["wall_seconds"]},
                   "metrics": metrics}
        with _local_phase(device):
            if not _distributed() or dist.get_rank() == 0:
                atomic_write_text(output_dir / f"validation_summary_step_{step}.json",
                                  json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        return payload
