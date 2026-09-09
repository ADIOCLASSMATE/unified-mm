"""Fixed global downstream probes, evaluated by every training rank.

The same runner is used in training and by the 16-device timing acceptance.
No padded DataLoader shards, rank-dependent random seeds or prefix limits.
The 540-second cooperative work deadline reserves 60 seconds for reduction,
EMA restoration and logging. An unfinished task never produces a score.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path
import random
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist


PROFILE = "training_downstream_v1"
TEXT_TASKS = (
    "arc_easy", "arc_challenge", "hellaswag", "piqa", "winogrande",
    "boolq", "openbookqa", "mmlu",
)
GROUNDING_TASKS = ("aro_vg_relation", "sugarcrepe")


@dataclass(frozen=True)
class ValidationProfile:
    seed: int = 424242
    imagenet_per_class: int = 2
    grounding_samples: int = 512
    mc_samples: int = 16
    text_batch_size: int = 8
    image_batch_size: int = 32
    grounding_batch_size: int = 64
    lm_head_chunk_tokens: int = 256
    work_seconds: float = 540.0
    max_seconds: float = 600.0
    text_root: str = "public/benchmarks/selfless_text_v1"
    image_manifest: str = "public/datasets/imagenet_full/manifest_val.jsonl"
    image_classes: str = "public/datasets/imagenet1k_synthetic_v1/t2i/classes.json"
    image_classnames: str = "scripts/assets/imagenet1k_openai_clip_classnames.json"
    image_cache: str = "public/datasets/imagenet_full/vae_posterior_mar_kl16/val_shards"
    grounding_root: str = "public/benchmarks/selfless_multimodal_likelihood_v1"
    grounding_cache: str = "public/benchmarks/selfless_multimodal_likelihood_v1/vae_posterior_mar_kl16_v2/shards"

    def __post_init__(self):
        for name in ("imagenet_per_class", "grounding_samples", "mc_samples",
                     "text_batch_size", "image_batch_size", "grounding_batch_size",
                     "lm_head_chunk_tokens"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0 < self.work_seconds < self.max_seconds <= 600:
            raise ValueError("require 0 < work_seconds < max_seconds <= 600")

    @classmethod
    def from_config(cls, config):
        return cls(**dict(config.experiment.get("downstream_validation", {})))


def stratified_sample(items, count, *, category, identity, seed):
    """Proportional fixed sampling with at least one item per nonempty stratum."""
    if not 0 < count <= len(items):
        raise ValueError(f"invalid subset size {count} for {len(items)} records")
    if len({identity(item) for item in items}) != len(items):
        raise ValueError("duplicate sample identities")
    groups = defaultdict(list)
    for item in items:
        groups[str(category(item))].append(item)
    keys = sorted(groups)
    if count < len(keys):
        raise ValueError("subset is too small to cover every stratum")
    # Assign one per stratum, then proportionally allocate remaining capacity.
    remaining = count - len(keys)
    capacity = len(items) - len(keys)
    quotas = {key: (remaining * (len(groups[key]) - 1) / capacity if capacity else 0)
              for key in keys}
    sizes = {key: 1 + math.floor(quotas[key]) for key in keys}
    for key in sorted(keys, key=lambda key: (-(quotas[key] % 1), key))[:count - sum(sizes.values())]:
        sizes[key] += 1
    rng = random.Random(seed)
    selected = []
    for key in keys:
        pool = sorted(groups[key], key=identity)
        rng.shuffle(pool)
        selected.extend(pool[:sizes[key]])
    rng.shuffle(selected)
    return selected


@dataclass(frozen=True)
class ImageNetValidationSubset:
    """One ordered, class-balanced list shared by loss and downstream scoring."""

    records: tuple
    seed: int
    per_class: int
    image_manifest: str
    image_classes: str

    def __post_init__(self):
        counts = Counter(record.class_index for record in self.records)
        if not counts or set(counts.values()) != {self.per_class}:
            raise ValueError("ImageNet validation subset must be balanced per class")
        for key in ("image_id", "img_id"):
            if len({getattr(record, key) for record in self.records}) != len(self.records):
                raise ValueError(f"duplicate ImageNet subset {key}")

    def validate_profile(self, profile):
        if (self.seed != profile.seed or self.per_class != profile.imagenet_per_class or
                Path(self.image_manifest).resolve() != Path(profile.image_manifest).resolve() or
                Path(self.image_classes).resolve() != Path(profile.image_classes).resolve()):
            raise ValueError("shared ImageNet subset differs from downstream_validation configuration")

    def metadata(self):
        return {"schema": "training_imagenet_subset_v1", "seed": self.seed,
                "per_class": self.per_class, "samples": len(self.records),
                "classes": len({record.class_index for record in self.records}),
                "image_manifest": self.image_manifest, "image_classes": self.image_classes,
                "sample_ids": [record.image_id for record in self.records],
                "img_ids": [record.img_id for record in self.records],
                "class_indices": [record.class_index for record in self.records]}

    def indices_for(self, subset):
        """Resolve official IDs to this Subset's offsets, never assume row order."""
        dataset = subset.dataset
        if dataset.dataset_split != "val":
            raise ValueError("shared ImageNet subset requires an independent val dataset")
        image_ids = dataset.img_ids.tolist()
        positions = {int(image_ids[int(row)]): offset for offset, row in enumerate(subset.indices)}
        if len(positions) != len(subset):
            raise ValueError("duplicate image identities in validation loader")
        indices = []
        for record in self.records:
            if record.img_id not in positions:
                raise ValueError(f"shared ImageNet sample absent from loss dataset: {record.image_id}")
            actual_id = f"val/{Path(dataset.source_paths_full[record.img_id]).stem}"
            if actual_id != record.image_id or dataset.synsets[record.img_id] != record.synset:
                raise ValueError(f"ImageNet loss/downstream identity mismatch: {record.image_id}")
            offset = positions[record.img_id]
            if dataset._is_training_index(int(subset.indices[offset])):
                raise ValueError("ImageNet loss samples must use fixed validation serialization")
            indices.append(offset)
        return indices


def prepare_imagenet_subset(profile):
    from scripts.evaluate_imagenet_pretraining_native import load_imagenet_records

    records = load_imagenet_records(Path(profile.image_manifest), Path(profile.image_classes))
    # Preserve the exact historical downstream selection AND order.
    selected = stratified_sample(records, 1000 * profile.imagenet_per_class,
                                 category=lambda r: r.class_index, identity=lambda r: r.image_id,
                                 seed=profile.seed)
    return ImageNetValidationSubset(tuple(selected), profile.seed, profile.imagenet_per_class,
                                    profile.image_manifest, profile.image_classes)


class DownstreamValidationState:
    """Per-training-process CPU data/token cache; never retain model scores."""

    def __init__(self, profile, *, imagenet_subset=None):
        self.profile = profile
        self.imagenet_subset = imagenet_subset
        self.prepared = None
        self.tokenizer = None
        self._shape = None

    def prepare(self, model, tokenizer):
        from scripts.evaluate_imagenet_pretraining_native import CachedTokenizer

        shape = (int(model.config.image_tokens_per_img), int(model.config.image_latent_dim))
        if self.prepared is not None:
            if self._shape != shape or self.tokenizer.tokenizer is not tokenizer:
                raise ValueError("validation cache cannot be reused with different image dimensions or tokenizer")
            return True
        if self.imagenet_subset is None:
            self.imagenet_subset = prepare_imagenet_subset(self.profile)
        self.imagenet_subset.validate_profile(self.profile)
        prepared = _prepare(model, self.profile, self.imagenet_subset)
        self.prepared = prepared
        self.tokenizer = CachedTokenizer(tokenizer)
        self._shape = shape
        return False


def rank_indices(count, rank, world_size):
    if world_size <= 0 or not 0 <= rank < world_size:
        raise ValueError("invalid distributed rank")
    return range(rank, count, world_size)


def _distributed():
    return dist.is_available() and dist.is_initialized()


def _reduce(value, device, op=dist.ReduceOp.SUM):
    tensor = torch.as_tensor(value, dtype=torch.float32, device=device)
    if _distributed():
        dist.all_reduce(tensor, op=op)
    return tensor.cpu()


def _synchronize(device):
    if device.type in {"npu", "cuda"}:
        getattr(torch, device.type).synchronize(device)


@contextmanager
def _local_phase(device):
    """Finish local-only work before all ranks agree whether reduction is safe."""
    error = None
    try:
        yield
    except Exception as exc:
        error = exc
    failed = _reduce([int(error is not None)], device, dist.ReduceOp.MAX)
    if bool(failed[0]):
        raise RuntimeError("downstream validation failed on a training rank") from error


@contextmanager
def evaluation_state(model, device, ema=None):
    """Restore weights, module modes and all training RNG streams on failure too."""
    python_state, numpy_state = random.getstate(), np.random.get_state()
    modes = [(module, module.training) for module in model.modules()]
    rng_args = {"devices": []}
    if device.type in {"npu", "cuda"}:
        rng_args = {"devices": [device.index or 0], "device_type": device.type}
    try:
        with torch.random.fork_rng(**rng_args):
            with ema.applied_to(model) if ema is not None else nullcontext():
                model.eval()
                with torch.no_grad():
                    yield
    finally:
        # train() also invalidates weight-dependent inference caches.
        model.train(modes[0][1])
        for module, training in modes:
            module.training = training
        random.setstate(python_state)
        np.random.set_state(numpy_state)


class Deadline:
    def __init__(self, seconds, started=None):
        self.started = time.monotonic() if started is None else started
        self.stop_at = self.started + seconds

    def available(self):
        return time.monotonic() < self.stop_at


def _coverage_status(seen, expected, device):
    counts = _reduce(seen, device)
    if bool((counts > 1).any()):
        raise RuntimeError("validation sampled duplicate records across ranks")
    completed = int(counts.sum().item())
    return {"complete": completed == expected, "samples": completed,
            "expected_samples": expected, "status": "complete" if completed == expected else "time_budget_exhausted"}


def _text_task(model, tokenizer, examples, task, profile, device, rank, world, deadline):
    from scripts.evaluate_selfless_text_benchmarks import encode_choice, score_choice_requests, primary_metric

    categories = sorted({str(example.category or "all") for example in examples})
    category_index = {name: i for i, name in enumerate(categories)}
    totals = torch.zeros((len(categories), 3))
    seen = torch.zeros(len(examples))
    local = list(rank_indices(len(examples), rank, world))
    with _local_phase(device):
        for offset in range(0, len(local), 32):
            if not deadline.available():
                break
            indices = local[offset:offset + 32]
            requests, spans = [], []
            for index in indices:
                example = examples[index]
                start = len(requests)
                requests.extend(encode_choice(tokenizer, example, j, 4096) for j in range(len(example.choices)))
                spans.append((start, len(requests)))
            scores = score_choice_requests(model, requests, batch_size=profile.text_batch_size,
                                           lm_head_chunk_tokens=profile.lm_head_chunk_tokens, device=device)
            for index, (start, stop) in zip(indices, spans):
                example = examples[index]
                raw = [score.loglikelihood for score in scores[start:stop]]
                norm = [score.normalized_loglikelihood for score in scores[start:stop]]
                totals[category_index[str(example.category or "all")]] += torch.tensor([
                    1, max(range(len(raw)), key=raw.__getitem__) == example.label,
                    max(range(len(norm)), key=norm.__getitem__) == example.label,
                ])
                seen[index] = 1
    status = _coverage_status(seen, len(examples), device)
    totals = _reduce(totals, device)
    if status["complete"]:
        count, correct, normalized = totals.sum(0).tolist()
        metrics = {"accuracy": correct / count, "accuracy_normalized": normalized / count,
                   "accuracy_macro": float((totals[:, 1] / totals[:, 0]).mean()),
                   "by_category": {name: {"samples": int(row[0]), "accuracy": float(row[1] / row[0])}
                                   for name, row in zip(categories, totals)}}
        status.update(metrics=metrics, primary=primary_metric(task, metrics))
    return status


def _imagenet_task(model, tokenizer, records, class_names, cache, profile, device, rank, world, deadline):
    from scripts.evaluate_imagenet_pretraining_native import (
        CLASS_TEXT_TEMPLATE, CLASSIFICATION_TASK, RETRIEVAL_PROMPT,
        score_candidates_with_backend, classification_metrics,
    )
    from scripts.language_prior_calibration import language_prior_debiased_scores

    candidates = [CLASS_TEXT_TEMPLATE.format(class_name=name) for name in class_names]
    matrix = torch.zeros((len(records), len(candidates)))
    seen = torch.zeros(len(records))
    args = SimpleNamespace(batch_size_per_rank=profile.image_batch_size,
                           request_chunk_size=128, max_length=2048,
                           lm_head_chunk_tokens=profile.lm_head_chunk_tokens,
                           seed=profile.seed, scoring_backend="cached_prefix")
    with _local_phase(device):
        for index in rank_indices(len(records), rank, world):
            if not deadline.available():
                break
            record = records[index]
            _, matrix[index] = score_candidates_with_backend(
                model=model, tokenizer=tokenizer, cache=cache, image_id=record.img_id,
                item_id=f"{CLASSIFICATION_TASK}/{record.image_id}", prompt=RETRIEVAL_PROMPT,
                candidates=candidates, args=args, device=device,
                image_sigma_order=model.config.training_image_sigma_order,
                attention_contract=model.config.dual_stream_attention_contract,
                evaluation_task=CLASSIFICATION_TASK,
            )
            seen[index] = 1
    status = _coverage_status(seen, len(records), device)
    if status["complete"]:
        matrix = _reduce(matrix, device)
        calibrated, _ = language_prior_debiased_scores(matrix)
        metrics = classification_metrics(calibrated, torch.tensor([r.class_index for r in records]))
        status.update(metrics=metrics, primary=metrics["top_1_accuracy"],
                      language_prior_alpha=1.0, language_prior_image_count=len(records))
    return status


def _grounding_task(model, tokenizer, examples, cache, null_ids, profile, device, rank, world, deadline):
    from scripts.evaluate_multimodal_likelihood_benchmarks import (
        encode_candidate_mc, score_candidate_requests, build_prediction_rows, DEBIASED_SCORE,
    )

    categories = sorted({str(example.category or "all") for example in examples})
    category_index = {name: i for i, name in enumerate(categories)}
    totals = torch.zeros((len(categories), 4))  # count, strict wins, ties, margin sum
    seen = torch.zeros(len(examples))
    order = model.config.training_image_sigma_order
    mc_samples = profile.mc_samples if order == "random" else 1
    encode_kwargs = dict(image_tokens=int(model.config.image_tokens_per_img),
                         boi_token_id=int(model.config.boi_token_id), eoi_token_id=int(model.config.eoi_token_id),
                         image_mask_token_id=int(model.config.image_mask_token_id), max_length=2048,
                         image_sigma_order=order, seed=profile.seed, mc_samples=mc_samples)
    score_kwargs = dict(batch_size=profile.grounding_batch_size,
                        lm_head_chunk_tokens=profile.lm_head_chunk_tokens,
                        attention_contract=model.config.dual_stream_attention_contract, device=device)
    with _local_phase(device):
        for index in rank_indices(len(examples), rank, world):
            if not deadline.available():
                break
            example = examples[index]
            requests = [r for j in range(len(example.candidates))
                        for r in encode_candidate_mc(tokenizer, example, j, **encode_kwargs)]
            prior_requests = [r for j in range(len(example.candidates)) for image_id in null_ids
                              for r in encode_candidate_mc(tokenizer, replace(example, image_id=image_id), j, **encode_kwargs)]
            scores = score_candidate_requests(model, requests, cache, **score_kwargs)
            priors = score_candidate_requests(model, prior_requests, cache, **score_kwargs)
            row = build_prediction_rows([example], requests, scores, prior_requests=prior_requests,
                                        prior_scores=priors, null_image_ids=null_ids, mc_samples=mc_samples)[0]
            margin = (row["candidate_scores"][example.label][DEBIASED_SCORE]
                      - row["candidate_scores"][1 - example.label][DEBIASED_SCORE])
            totals[category_index[str(example.category or "all")]] += torch.tensor([1, margin > 0, margin == 0, margin])
            seen[index] = 1
    status = _coverage_status(seen, len(examples), device)
    totals = _reduce(totals, device)
    if status["complete"]:
        count, wins, ties, margin = totals.sum(0).tolist()
        status.update(primary=wins / count, metrics={"win_rate": wins / count, "tie_rate": ties / count,
                      "mean_margin": margin / count,
                      "by_category": {name: {"samples": int(row[0]), "win_rate": float(row[1] / row[0])}
                                      for name, row in zip(categories, totals)}},
                      mc_samples=mc_samples, language_prior_alpha=1.0, null_image_ids=list(null_ids))
    return status


def _prepare(model, profile, imagenet_subset):
    from scripts import evaluate_selfless_text_benchmarks as text_eval
    from scripts import evaluate_imagenet_pretraining_native as image_eval
    from scripts import evaluate_multimodal_likelihood_benchmarks as mm_eval

    text = {task: text_eval.load_multiple_choice_task(task, Path(profile.text_root)) for task in TEXT_TASKS}
    records = imagenet_subset.records
    class_names, _ = image_eval.load_openai_clip_class_names(Path(profile.image_classnames))
    manifest = json.loads((Path(profile.grounding_root) / "manifest.json").read_text())
    grounding = {}
    for index, task in enumerate(GROUNDING_TASKS):
        examples = mm_eval.load_examples(task, manifest, 0)
        grounding[task] = stratified_sample(examples, profile.grounding_samples,
                                            category=lambda r: r.category, identity=lambda r: r.item_id,
                                            seed=profile.seed + index + 1)
        if any(len(example.candidates) != 2 or example.label not in (0, 1) for example in grounding[task]):
            raise ValueError(f"{task} must contain labeled caption pairs")
    cache_args = dict(expected_image_tokens=int(model.config.image_tokens_per_img),
                      expected_latent_dim=int(model.config.image_latent_dim), seed=profile.seed)
    image_cache = mm_eval.PosteriorCache(Path(profile.image_cache), **cache_args)
    grounding_cache = mm_eval.PosteriorCache(Path(profile.grounding_cache), **cache_args)
    null_ids = mm_eval.language_prior_null_image_ids(manifest)
    if any(record.img_id not in image_cache for record in records):
        raise ValueError("ImageNet subset is absent from the posterior cache")
    if any(example.image_id not in grounding_cache for rows in grounding.values() for example in rows) or any(i not in grounding_cache for i in null_ids):
        raise ValueError("grounding subset or null images are absent from the posterior cache")
    ids = {task: [example.item_id for example in examples] for task, examples in {**text, **grounding}.items()}
    ids["imagenet"] = [record.image_id for record in records]
    return text, records, class_names, grounding, image_cache, grounding_cache, null_ids, ids


def run_downstream_validation(model, tokenizer, *, device, output_dir, step=0,
                              ema=None, profile=None, started=None, weight_source=None, state=None):
    """Run on the unwrapped, replicated model on ALL ranks of the training group."""
    started = time.monotonic() if started is None else started
    from scripts.evaluate_multimodal_likelihood_benchmarks import atomic_write_text

    profile = profile or (state.profile if state is not None else ValidationProfile())
    state = state or DownstreamValidationState(profile)
    rank, world = (dist.get_rank(), dist.get_world_size()) if _distributed() else (0, 1)
    _synchronize(device)
    if _distributed():
        dist.barrier()
    deadline = Deadline(profile.work_seconds, started=started)
    output_dir = Path(output_dir)
    summary = {"schema": PROFILE, "profile": asdict(profile), "step": int(step),
               "world_size": world, "weight_source": weight_source or ("ema" if ema else "model"),
               "complete": False, "runtime_hashing_enabled": False, "tasks": {}}
    summary["model_contract"] = {
        key: getattr(model.config, key, None) for key in (
            "architecture_variant", "training_objective", "dual_stream_attention_contract",
            "flow_head_attention_contract", "flow_condition_contract", "training_image_sigma_order",
        )
    }
    summary["floating_buffer_dtypes"] = {
        name: str(buffer.dtype) for name, buffer in model.named_buffers() if buffer.is_floating_point()
    }
    if ema is not None:
        summary["ema"] = {"step": ema.global_step, "decay": ema.decay, "dtype": "fp32_shards"}
    with evaluation_state(model, device, ema):
        prepare_started = time.monotonic()
        with _local_phase(device):
            if state.profile != profile:
                raise ValueError("downstream validation cache profile changed")
            cache_hit = state.prepare(model, tokenizer)
            text, records, names, grounding, image_cache, grounding_cache, null_ids, ids = state.prepared
        summary["prepare_seconds"] = float(_reduce([time.monotonic() - prepare_started], device, dist.ReduceOp.MAX)[0])
        summary["prepare_cache_hit"] = bool(_reduce([int(cache_hit)], device, dist.ReduceOp.MIN)[0])
        with _local_phase(device):
            if rank == 0:
                atomic_write_text(output_dir / "subset.json", json.dumps({"schema": PROFILE, "seed": profile.seed,
                                  "sample_ids": ids, "imagenet_subset": state.imagenet_subset.metadata(),
                                  "profile": asdict(profile)}, ensure_ascii=False) + "\n")
                atomic_write_text(output_dir / "summary.json", json.dumps(summary, indent=2) + "\n")
        tokenizer = state.tokenizer
        tasks = [(task, lambda task=task: _text_task(model, tokenizer, text[task], task, profile,
                                                   device, rank, world, deadline)) for task in TEXT_TASKS]
        tasks.append(("imagenet", lambda: _imagenet_task(model, tokenizer, records, names, image_cache,
                                                       profile, device, rank, world, deadline)))
        tasks.extend((task, lambda task=task: _grounding_task(model, tokenizer, grounding[task], grounding_cache,
                                                            null_ids, profile, device, rank, world, deadline))
                     for task in GROUNDING_TASKS)
        for task, evaluate in tasks:
            began = time.monotonic()
            result = evaluate()
            result["wall_seconds"] = float(_reduce([time.monotonic() - began], device, dist.ReduceOp.MAX)[0])
            summary["tasks"][task] = result
            with _local_phase(device):
                if rank == 0:
                    print(json.dumps({"event": "training_downstream_task", "step": step, "task": task,
                                      **{k: v for k, v in result.items() if k != "metrics"}}), flush=True)
                    atomic_write_text(output_dir / "summary.json", json.dumps(summary, indent=2) + "\n")
    _synchronize(device)
    summary["wall_seconds"] = float(_reduce([time.monotonic() - deadline.started], device, dist.ReduceOp.MAX)[0])
    summary["within_time_budget"] = summary["wall_seconds"] <= profile.max_seconds
    summary["complete"] = all(item["complete"] for item in summary["tasks"].values())
    if all(summary["tasks"][task]["complete"] for task in TEXT_TASKS):
        summary["text_mean"] = sum(summary["tasks"][task]["primary"] for task in TEXT_TASKS) / len(TEXT_TASKS)
    with _local_phase(device):
        if rank == 0:
            atomic_write_text(output_dir / "summary.json", json.dumps(summary, indent=2) + "\n")
            print(json.dumps({"event": "training_downstream_complete", "step": step,
                              "complete": summary["complete"], "wall_seconds": summary["wall_seconds"],
                              "within_time_budget": summary["within_time_budget"]}), flush=True)
    return summary
