"""Fixed validation batches evaluated with the training loss protocol."""
from __future__ import annotations

from collections import Counter
import copy
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Subset

from utils.training_climbmix_validation import (
    ClimbMixValidationProfile, encode_documents, load_documents, text_batch,
)
from utils.training_downstream_validation import (
    ValidationProfile, _coverage_status, _distributed, _local_phase, _reduce,
    evaluation_state, prepare_imagenet_subset, rank_indices,
)
from utils.unified_loss_protocol import loss_protocol, source_loss_metric_payload


@dataclass(frozen=True)
class UnifiedLossValidationProfile:
    enabled: bool = True
    seed: int = 424242

    def __post_init__(self):
        if self.seed < 0:
            raise ValueError("invalid unified loss validation seed")

    @classmethod
    def from_config(cls, config):
        options = dict(config.experiment.get("loss_validation", {}))
        if "image_samples" in options:
            raise ValueError("loss_validation.image_samples was replaced by the shared downstream_validation.imagenet_per_class")
        return cls(**options)


def validation_metrics(totals, protocol):
    """Reduce true token means separately from schedule-weighted batch means.

    Each source stores [raw*targets, targets, sum(model.loss), batch_count].
    Unequal target counts must not turn a token mean into a batch mean.
    """
    sources = tuple(protocol["source_fractions"])
    rows = torch.as_tensor(totals).reshape(len(sources), 4).clone()
    if not torch.isfinite(rows).all() or (rows[:, 1] <= 0).any() or (rows[:, 3] <= 0).any():
        raise ValueError("complete finite validation is required for every active training source")
    slots = Counter(protocol["schedule"])
    weighted = rows[:, :3].clone()
    for index, source in enumerate(sources):
        weighted[index, 2] = rows[index, 2] / rows[index, 3] * slots[source]
    metrics, display = source_loss_metric_payload(
        weighted, num_processes=1, gradient_accumulation_steps=protocol["gradient_accumulation_steps"],
        active_sources=sources, prefix="val",
    )
    metrics["val/loss"] = sum(contribution for _, contribution in display.values())
    metrics["val/weighted_contribution_total"] = metrics["val/loss"]
    for index, source in enumerate(sources):
        metrics[f"val/mean_microbatch_loss_{source}"] = float(rows[index, 2] / rows[index, 3])
        metrics[f"val/{source}_microbatches"] = int(rows[index, 3])
        if source != "t2i":
            metrics[f"val/ppl_{source}"] = math.exp(min(metrics[f"val/loss_{source}"], 100))
    # Keep explicit legacy count aliases, never the ambiguous loss_text alias.
    metrics["val/image_target_tokens"] = metrics.get("val/t2i_target_tokens", 0)
    metrics["val/text_target_tokens"] = metrics.get("val/i2t_target_tokens", 0)
    return metrics


def _seed_batch(seed, device):
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if device.type in {"npu", "cuda"}:
        getattr(torch, device.type).manual_seed(seed)


class UnifiedLossValidator:
    def __init__(self, config, *, image_loader=None, image_collators=None, imagenet_subset=None):
        self.config = config
        self.profile = UnifiedLossValidationProfile.from_config(config)
        self.protocol = loss_protocol(config)
        self.sources = tuple(self.protocol["source_fractions"])
        self.image_datasets = {}
        self.image_collators = copy.deepcopy(image_collators or {})
        self.imagenet_subset = imagenet_subset
        self.image_indices = {}
        self._image_rows = {}
        self._rank_layout = None
        self.text_rows = None
        self.text_source = None
        self.text_profile = None
        if "climbmix" in self.sources:
            self.text_profile = ClimbMixValidationProfile.from_config(config)
            if not self.text_profile.enabled:
                raise ValueError("unified validation requires ClimbMix; disable loss_validation as a whole to skip it")
            source = config.dataset.params.sources.climbmix
            if self.text_profile.sequence_length != int(source.sequence_length):
                raise ValueError("validation text sequence_length must match training")
            if self.text_profile.batch_size != int(source.micro_batch_size):
                raise ValueError("validation text batch_size must match training micro_batch_size")
        for source in self.sources:
            if source == "climbmix":
                continue
            if image_loader is None:
                raise ValueError("active image loss needs an independent validation loader")
            dataset = image_loader.dataset
            if hasattr(dataset, "datasets"):
                # PairedImageTaskValidationDataset shares posterior tensors.
                subset = Subset(dataset.datasets[source], dataset.source_indices)
            else:
                subset = dataset
            if not isinstance(subset, Subset) or subset.dataset.dataset_split != "val":
                raise ValueError("unified loss must use the independent ImageNet-val dataset")
            downstream_profile = ValidationProfile.from_config(config)
            if self.imagenet_subset is None:
                self.imagenet_subset = prepare_imagenet_subset(downstream_profile)
            self.imagenet_subset.validate_profile(downstream_profile)
            self.image_datasets[source] = subset
            self.image_collators.setdefault(source, copy.deepcopy(image_loader.collate_fn))
            self.image_indices[source] = self.imagenet_subset.indices_for(subset)

    def _image_batch(self, source, indices, batch_id):
        # Only this rank's selected, immutable val rows live in RAM. The full
        # posterior stays mmap-backed; stochastic model inputs are never cached.
        rows = self._image_rows.setdefault(source, {})
        for index in indices:
            if index not in rows:
                rows[index] = self.image_datasets[source][index]
        collator = self.image_collators[source]
        if hasattr(collator, "reset") and hasattr(collator, "schedule"):
            collator.reset(batch_id % len(collator.schedule))
        # Training collators copy/pack rows into fresh batch tensors.
        return collator([rows[index] for index in indices])

    def _text_batch(self, indices, tokenizer):
        encoded = text_batch([self.text_rows[i] for i in indices], int(tokenizer.eos_token_id),
                             self.text_profile.sequence_length, torch.device("cpu"))
        renames = {"X0_input_ids": "input_ids", "flow_sigma": "sigma", "_text_segment_ids": "segment_ids"}
        return {renames.get(key, key): value for key, value in encoded.items() if isinstance(value, torch.Tensor)}

    def run(self, model, tokenizer, *, device, step, output_dir, forward_batch,
            training_exclusion=None, training_shards=()):
        started = time.monotonic()
        rank, world = (dist.get_rank(), dist.get_world_size()) if _distributed() else (0, 1)
        if self._rank_layout != (rank, world):
            self._image_rows.clear()
            self._rank_layout = (rank, world)
        totals = torch.zeros((len(self.sources), 4), dtype=torch.float32, device=device)
        coverage, subsets, timings = {}, {}, {}
        # Use training-mode stochastic loss (including image input noise), with
        # no gradients. Keep all training RNG streams and module modes intact.
        with evaluation_state(model, device):
            model.train()
            with _local_phase(device):
                if self.text_profile is not None:
                    if self.text_profile.jsonl and Path(self.text_profile.jsonl).resolve() in {Path(p).resolve() for p in training_shards}:
                        raise ValueError("external validation JSONL is a training shard")
                    if self.text_rows is None:
                        documents, self.text_source = load_documents(self.text_profile)
                        self.text_rows = encode_documents(documents, tokenizer, self.text_profile)
                for source in self.sources:
                    attr = "lambda_image" if source == "t2i" else "lambda_text"
                    if float(getattr(model, attr)) != self.protocol["model_weights"][source]:
                        raise ValueError("validation model loss weights differ from the training protocol")
            for source_index, source in enumerate(self.sources):
                source_started = time.monotonic()
                data_seconds = 0.0
                indices = list(range(len(self.text_rows))) if source == "climbmix" else self.image_indices[source]
                batch_size = self.protocol["micro_batch_sizes"][source]
                batches = [indices[i:i + batch_size] for i in range(0, len(indices), batch_size)]
                seen = torch.zeros(len(batches))
                invalid = torch.zeros((), dtype=torch.bool, device=device)
                with _local_phase(device):
                    for batch_id in rank_indices(len(batches), rank, world):
                        # Partition fixed GLOBAL microbatches, not rows: changing
                        # world size must not change the loss averaging groups.
                        source_seed = {"climbmix": 0, "t2i": 1000003, "i2t": 2000006}[source]
                        _seed_batch(self.profile.seed + source_seed + batch_id, device)
                        selected = batches[batch_id]
                        data_started = time.monotonic()
                        if source == "climbmix":
                            batch = self._text_batch(selected, tokenizer)
                        else:
                            batch = self._image_batch(source, selected, batch_id)
                        data_seconds += time.monotonic() - data_started
                        output = forward_batch(model, batch, source)
                        key = "image" if source == "t2i" else "text"
                        raw = output.per_modality_loss[key + "_loss"].detach().float()
                        targets = output.per_modality_count[key + "_tokens"].detach().float()
                        weighted = output.loss.detach().float()
                        values = torch.stack((raw * targets, targets, weighted, torch.ones_like(weighted)))
                        # Check on device, synchronize once per source instead
                        # of forcing two host/device waits per microbatch.
                        invalid |= ~torch.isfinite(values).all() | (targets <= 0)
                        totals[source_index] += values
                        seen[batch_id] = 1
                        del output, batch
                    if bool(invalid):
                        raise ValueError(f"invalid unified validation loss/targets: {source}")
                status = _coverage_status(seen, len(batches), device)
                elapsed = _reduce([time.monotonic() - source_started, data_seconds], device, dist.ReduceOp.MAX)
                timings[source] = {"wall_seconds": float(elapsed[0]), "data_prepare_seconds": float(elapsed[1])}
                coverage[source] = {"complete": status["complete"], "microbatches": status["samples"],
                                    "expected_microbatches": status["expected_samples"]}
                subsets[source] = {"samples": len(indices), "microbatches": len(batches),
                                   "micro_batch_size": batch_size, "last_batch_size": len(batches[-1]),
                                   "indices": indices}
            reduced = _reduce(totals, device)
        if not all(value["complete"] for value in coverage.values()):
            raise RuntimeError("unified loss validation did not complete every source")
        metrics = validation_metrics(reduced, self.protocol)
        text = None
        if self.text_profile is not None:
            independent = self.text_profile.external_independent if self.text_profile.jsonl else training_exclusion == self.text_source
            text = {"source": self.text_source, "protocol": "climbmix_fixed_document_ce_v1",
                    "sequence_length": self.text_profile.sequence_length, "seed": self.text_profile.seed,
                    "independence": "external_data_declared_independent" if independent and self.text_profile.jsonl else
                        "source_rows_excluded_since_training_start" if independent else "may_have_been_seen_in_training"}
        payload = {"schema": "selfless_unified_loss_validation_metrics_v1", "complete": True,
                   "global_step": int(step), "metrics": metrics, "loss_protocol": self.protocol,
                   "training_seed": int(self.config.training.seed), "validation_seed": self.profile.seed,
                   "model_weights": "current", "model_mode": "train_no_grad",
                   "image_data": dict(self.config.dataset.params.image.validation) if self.image_datasets else None,
                   "imagenet_subset": self.imagenet_subset.metadata() if self.image_datasets else None,
                   "pure_text": text, "subsets": subsets, "coverage": coverage,
                   "timings": timings, "world_size": world,
                   "wall_seconds": float(_reduce([time.monotonic() - started], device, dist.ReduceOp.MAX)[0])}
        with _local_phase(device):
            if rank == 0:
                path = Path(output_dir) / f"validation_unified_loss_metrics_step_{int(step)}.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix(".tmp")
                temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
                temporary.replace(path)
        return payload
