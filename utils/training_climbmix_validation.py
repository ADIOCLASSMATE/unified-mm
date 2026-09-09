"""Finite, token-weighted pure-text CE alongside the downstream benchmarks."""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
import time

import torch
import torch.distributed as dist

from utils.climbmix_validation_data import DEFAULT_MANIFEST, validation_documents
from utils.training_downstream_validation import (
    _coverage_status, _distributed, _local_phase, _reduce, evaluation_state, rank_indices,
)


@dataclass(frozen=True)
class ClimbMixValidationProfile:
    manifest: str = DEFAULT_MANIFEST
    jsonl: str | None = None
    # For an external JSONL this declaration is preserved as a user assertion;
    # source-path checks alone cannot establish absence of duplicate text.
    external_independent: bool = False
    sequence_length: int = 2048
    batch_size: int = 4
    max_document_chars: int = 32768
    seed: int = 424242
    enabled: bool = True

    def __post_init__(self):
        if self.sequence_length < 2 or self.batch_size < 1 or self.max_document_chars < 1:
            raise ValueError("invalid pure-text validation dimensions")

    @classmethod
    def from_config(cls, config):
        source = config.dataset.params.sources.climbmix
        params = dict(config.experiment.get("climbmix_validation", {}))
        params.setdefault("sequence_length", int(source.sequence_length))
        params.setdefault("batch_size", int(source.micro_batch_size))
        params.setdefault("max_document_chars", int(source.get("max_document_chars", 32768)))
        return cls(**params)


def load_documents(profile):
    if profile.jsonl:
        path = Path(profile.jsonl).resolve()
        rows = []
        with path.open() as handle:
            for index, line in enumerate(handle):
                if not line.strip():
                    continue
                record = json.loads(line)
                if not isinstance(record.get("text"), str) or not record["text"].strip():
                    raise ValueError(f"validation JSONL needs nonempty text: {path}:{index+1}")
                rows.append({"id": str(index), "text": record["text"]})
        stat = path.stat()
        return rows, {"jsonl": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    return validation_documents(profile.manifest)


def encode_documents(documents, tokenizer, profile):
    """One fixed document fragment per row; never join unrelated documents.

    Text is capped before tokenization as in training. A fixed random token
    window covers long fragments; the first token is context and is not scored.
    Short documents are padded, with all padding labels ignored.
    """
    rows = []
    rng = random.Random(profile.seed)
    eos = int(tokenizer.eos_token_id)
    for document in documents:
        text = document["text"]
        start = rng.randrange(max(1, len(text) - profile.max_document_chars + 1))
        fragment = text[start:start + profile.max_document_chars]
        ids = list(tokenizer.encode(fragment, add_special_tokens=False))
        # Only a real document/chunk boundary gets EOS, not a token truncation.
        ids.append(eos)
        if len(ids) > profile.sequence_length:
            offset = rng.randrange(len(ids) - profile.sequence_length + 1)
            ids = ids[offset:offset + profile.sequence_length]
        if len(ids) < 2:
            raise ValueError(f"validation document has no scored targets: {document['id']}")
        rows.append((document["id"], ids))
    if not rows:
        raise ValueError("pure-text validation dataset is empty")
    return rows


def text_batch(rows, eos, length, device):
    size = len(rows)
    ids = torch.zeros((size, length), dtype=torch.long, device=device)
    labels = torch.full_like(ids, -100)
    types = torch.full((size, length), 3, dtype=torch.uint8, device=device)
    sigma = torch.full_like(ids, length)
    segments = torch.full_like(ids, -1)
    for i, (_, tokens) in enumerate(rows):
        n = len(tokens)
        ids[i, :n] = torch.tensor(tokens, device=device)
        labels[i, 1:n] = ids[i, 1:n]
        types[i, :n] = torch.where(ids[i, :n] == eos, 2, 0).to(torch.uint8)
        sigma[i, :n] = torch.arange(n, device=device)
        segments[i, :n] = 0
    positions = torch.arange(length, device=device).view(1, 1, length).expand(2, size, length)
    return {"X0_input_ids": ids, "labels": labels, "token_types": types, "flow_sigma": sigma,
            "position_ids": positions, "_text_segment_ids": segments,
            "image_loss_mask": torch.zeros_like(ids, dtype=torch.bool),
            "image_span_table": torch.empty((0, 5), dtype=torch.long, device=device),
            "compute_text_loss": True, "compute_image_loss": False, "calculate_likelihood": True,
            "return_logits": False}


class ClimbMixLossValidator:
    """Cache CPU tokenization once; evaluate the same global rows on every step."""

    def __init__(self, profile):
        self.profile = profile
        self.rows = None
        self.source = None

    def run(self, model, tokenizer, *, device, step, output_dir, mask_builder,
            training_seed, training_exclusion=None, text_only=False, training_shards=()):
        started = time.monotonic()
        rank, world = (dist.get_rank(), dist.get_world_size()) if _distributed() else (0, 1)
        profile = self.profile
        totals = torch.zeros(2, dtype=torch.float32, device=device)
        # Validation neither advances training's data cursor nor changes RNG,
        # optimizer/EMA state, training/eval mode or persistent inference caches.
        with evaluation_state(model, device):
            with _local_phase(device):
                if self.rows is None:
                    documents, self.source = load_documents(profile)
                    self.rows = encode_documents(documents, tokenizer, profile)
                if profile.jsonl and Path(profile.jsonl).resolve() in {Path(p).resolve() for p in training_shards}:
                    raise ValueError("external validation JSONL is a training shard")
            count = len(self.rows)
            seen = torch.zeros(count)
            local = list(rank_indices(count, rank, world))
            with _local_phase(device):
                # Empty rank shards perform no model forward and still reduce.
                for offset in range(0, len(local), profile.batch_size):
                    indices = local[offset:offset + profile.batch_size]
                    batch = text_batch([self.rows[i] for i in indices], int(tokenizer.eos_token_id), profile.sequence_length, device)
                    query, content = mask_builder(input_ids=batch["X0_input_ids"], token_types=batch["token_types"],
                                                  sigma=batch["flow_sigma"], segment_ids=batch["_text_segment_ids"])
                    batch["attention_mask"] = query
                    if content is not None:
                        batch["content_attention_mask"] = content
                    result = model(**batch)
                    loss = result.per_modality_loss["text_loss"].detach().float()
                    targets = result.per_modality_count["text_tokens"].detach().float()
                    expected = batch["labels"].ne(-100).sum()
                    if not torch.isfinite(loss) or targets <= 0 or targets != expected:
                        raise ValueError("invalid pure-text validation loss or target count")
                    totals += torch.stack((loss * targets, targets))
                    seen[indices] = 1
            status = _coverage_status(seen, count, device)
            sums = _reduce(totals, device)
        if not status["complete"] or sums[1] <= 0:
            raise RuntimeError("pure-text validation did not cover its fixed global subset")
        mean = float(sums[0] / sums[1])
        weight = float(model.lambda_text)
        independent = bool(profile.external_independent) if profile.jsonl else training_exclusion == self.source
        metrics = {"val/loss_climbmix": mean, "val/ppl_climbmix": math.exp(min(mean, 100)),
                   "val/climbmix_target_tokens": int(sums[1]), "val/weighted_contribution_climbmix": weight * mean}
        if text_only:
            metrics["val/loss"] = weight * mean
        payload = {"schema": "selfless_climbmix_validation_metrics_v1", "global_step": int(step),
                   "metrics": metrics, "training_seed": training_seed, "validation_seed": profile.seed,
                   "protocol": "climbmix_fixed_document_ce_v1", "model_weights": "current",
                   "source": self.source, "excluded_from_training_stream": independent if not profile.jsonl else None,
                   "independence": "external_data_declared_independent" if profile.jsonl and independent else
                       "source_rows_excluded_since_training_start" if independent else "may_have_been_seen_in_training",
                   "independence_note": "Record exclusion does not remove duplicate text elsewhere or establish absence from base-model pretraining.",
                   "samples": count, "sequence_length": profile.sequence_length,
                   "max_document_chars": profile.max_document_chars, "world_size": world,
                   "aggregation": "sum(unweighted CE * valid target tokens) / sum(valid target tokens); no caption targets",
                   "complete": True, "wall_seconds": time.monotonic() - started}
        with _local_phase(device):
            if rank == 0:
                path = Path(output_dir) / f"validation_climbmix_metrics_step_{int(step)}.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix(".tmp")
                temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
                temporary.replace(path)
        return payload
