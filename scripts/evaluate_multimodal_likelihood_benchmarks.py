#!/usr/bin/env python3
"""Evaluate image-conditioned text likelihood without free-form QA generation.

Selfless predicts a token at the same query-stream position, so generic
causal-LM benchmark adapters (which apply a one-token shift) are incorrect for
this model.  This runner constructs the exact multimodal sequence used by I2T
training and scores only candidate suffix positions from the Selfless query
stream.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import statistics
import sys
import tempfile
import time
from typing import Any, Iterable, Sequence

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import torch
import torch.distributed as dist
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.evaluation_model_source import (  # noqa: E402
    add_model_source_argument,
    configure_model_source,
    load_model_source_weights,
    model_source_from_args,
    resolve_evaluation_model_source,
)
from utils.utils import get_selfless_mask, load_model_tokenizer  # noqa: E402


DEFAULT_CONFIG = Path("configs/selfless/unified_baseline_100b_ascend_64npu.yaml")
DEFAULT_ASSET_ROOT = Path("public/benchmarks/selfless_multimodal_likelihood_v1")
ASSET_SCHEMA = "selfless_multimodal_likelihood_assets_v1"
TASK_SCHEMA = "selfless_multimodal_likelihood_task_v1"
CACHE_FORMAT = "imagenet_kl16_scaled_posterior_v1"
CACHE_LAYOUT = "scaled_mean_then_scaled_std"
LIKELIHOOD_SCORING_CONTRACT = "selfless_same_position_dual_stream_v2"
DEFAULT_TASKS = (
    "mmbench_dev_en",
    "seed_bench_image",
    "sugarcrepe",
    "aro_vg_relation",
    "aro_vg_attribution",
)


@dataclass(frozen=True)
class LikelihoodExample:
    item_index: int
    item_id: str
    task: str
    kind: str
    image_id: int
    prompt: str
    candidates: tuple[str, ...]
    label: int | None
    category: str | None
    metadata: dict[str, Any]


@dataclass(frozen=True)
class CandidateRequest:
    example_index: int
    candidate_index: int
    image_id: int
    input_ids: tuple[int, ...]
    token_types: tuple[int, ...]
    sigma: tuple[int, ...]
    target_start: int
    image_start: int
    truncated_prompt_tokens: int


@dataclass(frozen=True)
class CandidateScore:
    loglikelihood: float
    normalized_loglikelihood: float
    token_count: int
    greedy: bool


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def readable_file_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def atomic_write_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".partial",
            delete=False,
        ) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def jsonl_text(rows: Iterable[dict[str, Any]]) -> str:
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    )


def initialize_device(kind: str) -> tuple[int, int, int, torch.device]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    if kind == "npu":
        import torch_npu  # noqa: F401

        device = torch.device(f"npu:{local_rank}")
        torch.npu.set_device(device)
        backend = "hccl"
    elif kind == "cuda":
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend=backend)
    return rank, world_size, local_rank, device


def barrier(device: torch.device) -> None:
    if not (dist.is_available() and dist.is_initialized()):
        return
    if device.type in {"npu", "cuda"}:
        dist.barrier(device_ids=[int(device.index or 0)])
    else:
        dist.barrier()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    add_model_source_argument(parser)
    parser.add_argument("--asset_root", type=Path, default=DEFAULT_ASSET_ROOT)
    parser.add_argument("--cache_shard_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    parser.add_argument("--batch_size_per_rank", type=int, default=4)
    parser.add_argument("--lm_head_chunk_tokens", type=int, default=256)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=424242)
    parser.add_argument("--device", choices=("npu", "cuda", "cpu"), default="npu")
    parser.add_argument("--model_dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument(
        "--image_sigma_order",
        choices=("auto", "random", "sequential"),
        default="auto",
    )
    parser.add_argument("--progress_every", type=int, default=50)
    return parser.parse_args()


def read_checkpoint_step(checkpoint: Path) -> int:
    return resolve_evaluation_model_source(checkpoint).global_step


def validate_args(
    args: argparse.Namespace, source
) -> tuple[tuple[str, ...], dict[str, Any]]:
    for path in (
        args.config,
        source.path,
        args.asset_root,
        args.cache_shard_dir,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    for value in (
        args.batch_size_per_rank,
        args.lm_head_chunk_tokens,
        args.max_length,
        args.progress_every,
    ):
        if int(value) <= 0:
            raise ValueError(f"positive integer required, got {value}")
    if args.limit < 0:
        raise ValueError("--limit must be non-negative")
    manifest_path = args.asset_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != ASSET_SCHEMA:
        raise ValueError(f"unexpected asset schema in {manifest_path}")
    if bool(manifest.get("runtime_hashing_enabled", True)):
        raise ValueError("multimodal likelihood assets violate the no-hash contract")
    tasks = tuple(value.strip() for value in str(args.tasks).split(",") if value.strip())
    if not tasks:
        raise ValueError("at least one task is required")
    missing = sorted(set(tasks) - set(manifest.get("tasks", {})))
    if missing:
        unavailable = manifest.get("unavailable", {})
        details = {task: unavailable.get(task, "not present") for task in missing}
        raise ValueError(f"requested tasks are unavailable: {details}")
    return tasks, manifest


def arithmetic_seed(seed: int, image_id: int, offset: int) -> int:
    return (int(seed) + 97_409 * int(image_id) + int(offset)) & ((1 << 63) - 1)


class PosteriorCache:
    def __init__(
        self,
        shard_dir: Path,
        *,
        expected_image_tokens: int,
        expected_latent_dim: int,
        seed: int,
    ) -> None:
        shard_paths = sorted(shard_dir.glob("shard-*-of-*.pt"))
        if not shard_paths:
            shard_paths = sorted(shard_dir.glob("*.pt"))
        if not shard_paths:
            raise FileNotFoundError(f"no posterior cache shards in {shard_dir}")
        self.seed = int(seed)
        self.image_tokens = int(expected_image_tokens)
        self.latent_dim = int(expected_latent_dim)
        self._payloads: list[dict[str, Any]] = []
        self._locations: dict[int, tuple[int, int]] = {}
        self._sampled: dict[int, torch.Tensor] = {}
        for shard_index, path in enumerate(shard_paths):
            payload = torch.load(
                str(path), map_location="cpu", mmap=True, weights_only=True
            )
            stats = payload.get("posterior_stats")
            image_ids = payload.get("img_ids")
            metadata = payload.get("metadata", {})
            expected_shape = (self.image_tokens, 2 * self.latent_dim)
            if not torch.is_tensor(stats) or tuple(stats.shape[1:]) != expected_shape:
                raise ValueError(f"invalid posterior shape in {path}: {getattr(stats, 'shape', None)}")
            if not torch.is_tensor(image_ids) or int(image_ids.numel()) != int(stats.shape[0]):
                raise ValueError(f"invalid img_ids in {path}")
            if metadata.get("format") != CACHE_FORMAT:
                raise ValueError(f"unexpected cache format in {path}")
            if metadata.get("stats_layout") != CACHE_LAYOUT:
                raise ValueError(f"unexpected cache layout in {path}")
            if bool(metadata.get("runtime_hashing_enabled", True)):
                raise ValueError(f"cache shard violates the no-hash contract: {path}")
            self._payloads.append(payload)
            for row_index, value in enumerate(image_ids.tolist()):
                image_id = int(value)
                if image_id in self._locations:
                    raise ValueError(f"duplicate image_id={image_id} across cache shards")
                self._locations[image_id] = (shard_index, row_index)

    def __contains__(self, image_id: int) -> bool:
        return int(image_id) in self._locations

    def sample(self, image_id: int) -> torch.Tensor:
        image_id = int(image_id)
        cached = self._sampled.get(image_id)
        if cached is not None:
            return cached
        try:
            shard_index, row_index = self._locations[image_id]
        except KeyError as exc:
            raise KeyError(f"image_id={image_id} is absent from posterior cache") from exc
        stats = self._payloads[shard_index]["posterior_stats"][row_index]
        mean = stats[..., : self.latent_dim].float()
        std = stats[..., self.latent_dim :].float()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(arithmetic_seed(self.seed, image_id, 11))
        noise = torch.randn(mean.shape, generator=generator, dtype=torch.float32)
        sampled = (mean + std * noise).to(dtype=stats.dtype)
        self._sampled[image_id] = sampled
        return sampled

    def __len__(self) -> int:
        return len(self._locations)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("schema") != TASK_SCHEMA:
                raise ValueError(f"unexpected task schema at {path}:{line_number}")
            rows.append(row)
    return rows


def load_examples(
    task: str,
    asset_manifest: dict[str, Any],
    limit: int,
) -> list[LikelihoodExample]:
    task_info = asset_manifest["tasks"][task]
    rows = read_jsonl(Path(task_info["manifest_jsonl"]))
    if int(task_info["records"]) != len(rows):
        raise ValueError(f"task record count mismatch for {task}")
    if limit > 0:
        rows = rows[:limit]
    examples: list[LikelihoodExample] = []
    for index, row in enumerate(rows):
        candidates = tuple(str(value).strip() for value in row["candidates"])
        label = row.get("label")
        if not candidates or any(not value for value in candidates):
            raise ValueError(f"{task}/{row.get('item_id')} contains empty candidates")
        if label is not None and not 0 <= int(label) < len(candidates):
            raise ValueError(f"{task}/{row.get('item_id')} has invalid label")
        examples.append(
            LikelihoodExample(
                item_index=index,
                item_id=str(row["item_id"]),
                task=str(row["task"]),
                kind=str(row["kind"]),
                image_id=int(row["image_id"]),
                prompt=str(row["prompt"]),
                candidates=candidates,
                label=int(label) if label is not None else None,
                category=(str(row["category"]) if row.get("category") else None),
                metadata=dict(row.get("metadata") or {}),
            )
        )
    kinds = {example.kind for example in examples}
    if len(kinds) != 1:
        raise ValueError(f"task {task} mixes kinds: {sorted(kinds)}")
    return examples


def build_image_sigma(
    image_tokens: int,
    *,
    order: str,
    seed: int,
) -> list[int]:
    if order == "sequential":
        return list(range(int(image_tokens)))
    if order != "random":
        raise ValueError(f"unknown image sigma order: {order}")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return [
        int(value)
        for value in torch.rand(int(image_tokens), generator=generator).argsort().tolist()
    ]


def encode_candidate(
    tokenizer,
    example: LikelihoodExample,
    candidate_index: int,
    *,
    image_tokens: int,
    boi_token_id: int,
    eoi_token_id: int,
    image_mask_token_id: int,
    max_length: int,
    image_sigma_order: str,
    seed: int,
) -> CandidateRequest:
    prompt_ids = [
        int(value)
        for value in tokenizer.encode(example.prompt.strip(), add_special_tokens=False)
    ]
    candidate_ids = [
        int(value)
        for value in tokenizer.encode(
            example.candidates[candidate_index].strip(), add_special_tokens=False
        )
    ]
    if not prompt_ids:
        eos_id = tokenizer.eos_token_id
        if eos_id is None:
            raise ValueError("tokenizer needs eos_token_id for an empty prompt")
        prompt_ids = [int(eos_id)]
    if not candidate_ids:
        raise ValueError(
            f"candidate tokenized to empty text: {example.task}/{example.item_id}"
        )
    fixed = int(image_tokens) + 2 + len(candidate_ids)
    if fixed >= int(max_length):
        raise ValueError(
            f"candidate is too long for max_length={max_length}: "
            f"{example.task}/{example.item_id}"
        )
    retained_prompt = int(max_length) - fixed
    truncated = max(0, len(prompt_ids) - retained_prompt)
    prompt_ids = prompt_ids[-retained_prompt:]
    prompt_len = len(prompt_ids)
    image_start = prompt_len + 1
    input_ids = (
        prompt_ids
        + [int(boi_token_id)]
        + [int(image_mask_token_id)] * int(image_tokens)
        + [int(eoi_token_id)]
        + candidate_ids
    )
    token_types = (
        [0] * prompt_len
        + [2]
        + [1] * int(image_tokens)
        + [2]
        + [0] * len(candidate_ids)
    )
    eoi_position = image_start + int(image_tokens)
    sigma = [0] * len(input_ids)
    sigma[:prompt_len] = list(range(prompt_len))
    sigma[prompt_len] = prompt_len
    sigma[eoi_position] = prompt_len + 1
    reveal = build_image_sigma(
        image_tokens,
        order=image_sigma_order,
        seed=arithmetic_seed(seed, example.image_id, 53),
    )
    for local_index, order_value in enumerate(reveal):
        sigma[image_start + local_index] = prompt_len + 2 + int(order_value)
    candidate_start = eoi_position + 1
    suffix_sigma_start = prompt_len + int(image_tokens) + 2
    for local_index in range(len(candidate_ids)):
        sigma[candidate_start + local_index] = suffix_sigma_start + local_index
    return CandidateRequest(
        example_index=int(example.item_index),
        candidate_index=int(candidate_index),
        image_id=int(example.image_id),
        input_ids=tuple(input_ids),
        token_types=tuple(token_types),
        sigma=tuple(sigma),
        target_start=candidate_start,
        image_start=image_start,
        truncated_prompt_tokens=truncated,
    )


def build_attention_masks(
    *,
    sigma: torch.Tensor,
    segment_ids: torch.Tensor,
    token_types: torch.Tensor,
    input_ids: torch.Tensor,
    boi_token_id: int,
    attention_contract: str,
    device: torch.device,
) -> tuple[Any, Any | None]:
    kwargs = {
        "sigma": sigma,
        "seq_len": int(sigma.shape[1]),
        "device": device,
        "input_ids": input_ids,
        "token_types": token_types,
        "boi_token_id": int(boi_token_id),
        "segment_ids": segment_ids,
    }
    query_mask = get_selfless_mask(**kwargs)
    content_mask = (
        get_selfless_mask(**kwargs, include_diagonal=True)
        if attention_contract == "xlnet_content_diagonal"
        else None
    )
    return query_mask, content_mask


@torch.inference_mode()
def score_candidate_requests(
    model,
    requests: Sequence[CandidateRequest],
    cache: PosteriorCache,
    *,
    batch_size: int,
    lm_head_chunk_tokens: int,
    attention_contract: str,
    device: torch.device,
) -> list[CandidateScore]:
    if not requests:
        return []
    pad_id = int(getattr(model.config, "eos_token_id", 0) or 0)
    latent_dim = int(model.config.image_latent_dim)
    image_tokens = int(model.config.image_tokens_per_img)
    ordered = sorted(
        enumerate(requests), key=lambda pair: len(pair[1].input_ids), reverse=True
    )
    scores: list[CandidateScore | None] = [None] * len(requests)
    for offset in range(0, len(ordered), int(batch_size)):
        batch_pairs = ordered[offset : offset + int(batch_size)]
        batch = [request for _, request in batch_pairs]
        length = max(len(request.input_ids) for request in batch)
        rows = len(batch)
        input_ids = torch.full((rows, length), pad_id, device=device, dtype=torch.long)
        token_types = torch.full((rows, length), 3, device=device, dtype=torch.uint8)
        segment_ids = torch.full((rows, length), -1, device=device, dtype=torch.long)
        sigma = torch.full((rows, length), length, device=device, dtype=torch.long)
        target_mask = torch.zeros((rows, length), device=device, dtype=torch.bool)
        latent_dtype = cache.sample(batch[0].image_id).dtype
        image_latents = torch.zeros(
            rows,
            length,
            latent_dim,
            device=device,
            dtype=latent_dtype,
        )
        image_latent_mask = torch.zeros(
            (rows, length), device=device, dtype=torch.bool
        )
        image_span_rows: list[list[int]] = []
        for row_index, request in enumerate(batch):
            item_length = len(request.input_ids)
            input_ids[row_index, :item_length] = torch.tensor(
                request.input_ids, device=device, dtype=torch.long
            )
            token_types[row_index, :item_length] = torch.tensor(
                request.token_types, device=device, dtype=torch.uint8
            )
            segment_ids[row_index, :item_length] = 0
            sigma[row_index, :item_length] = torch.tensor(
                request.sigma, device=device, dtype=torch.long
            )
            target_mask[row_index, request.target_start:item_length] = True
            image_end = request.image_start + image_tokens
            image_latents[row_index, request.image_start:image_end] = cache.sample(
                request.image_id
            ).to(device=device, dtype=image_latents.dtype)
            image_latent_mask[row_index, request.image_start:image_end] = True
            image_span_rows.append(
                [row_index, 0, request.image_start, image_end, request.image_id]
            )
        query_mask, content_mask = build_attention_masks(
            sigma=sigma,
            segment_ids=segment_ids,
            token_types=token_types,
            input_ids=input_ids,
            boi_token_id=int(model.config.boi_token_id),
            attention_contract=attention_contract,
            device=device,
        )
        forward_kwargs = {
            "X0_input_ids": input_ids,
            "attention_mask": query_mask,
            "token_types": token_types,
            "image_latents": image_latents,
            "image_latent_mask": image_latent_mask,
            "image_span_table": torch.tensor(
                image_span_rows, device=device, dtype=torch.long
            ),
            "calculate_likelihood": True,
        }
        if content_mask is not None:
            forward_kwargs["content_attention_mask"] = content_mask
        outputs = model.model(**forward_kwargs)
        hidden = outputs.last_hidden_state
        selected_hidden = hidden[target_mask]
        selected_targets = input_ids[target_mask]
        selected_rows = target_mask.nonzero(as_tuple=False)[:, 0]
        token_logprobs: list[torch.Tensor] = []
        token_greedy: list[torch.Tensor] = []
        for start in range(0, selected_hidden.shape[0], int(lm_head_chunk_tokens)):
            stop = min(
                start + int(lm_head_chunk_tokens), selected_hidden.shape[0]
            )
            logits = model.lm_head(selected_hidden[start:stop])
            targets = selected_targets[start:stop]
            gold = logits.gather(1, targets.unsqueeze(1)).squeeze(1).float()
            token_logprobs.append(gold - torch.logsumexp(logits.float(), dim=-1))
            token_greedy.append(logits.argmax(dim=-1).eq(targets))
        logprobs = torch.cat(token_logprobs)
        greedy = torch.cat(token_greedy)
        for row_index, (original_index, _) in enumerate(batch_pairs):
            selection = selected_rows.eq(row_index)
            row_logprobs = logprobs[selection]
            row_greedy = greedy[selection]
            count = int(row_logprobs.numel())
            if count <= 0:
                raise RuntimeError("candidate request has no target tokens")
            total = float(row_logprobs.sum().item())
            scores[original_index] = CandidateScore(
                loglikelihood=total,
                normalized_loglikelihood=total / count,
                token_count=count,
                greedy=bool(row_greedy.all().item()),
            )
        del (
            outputs,
            hidden,
            selected_hidden,
            selected_targets,
            query_mask,
            content_mask,
            image_latents,
        )
    if any(score is None for score in scores):
        raise RuntimeError("candidate scoring left incomplete results")
    return [score for score in scores if score is not None]


def argmax(values: Sequence[float]) -> int:
    return max(range(len(values)), key=lambda index: (float(values[index]), -index))


def finite_perplexity(normalized_loglikelihood: float) -> float:
    exponent = -float(normalized_loglikelihood)
    return math.exp(exponent) if exponent < 700.0 else float("inf")


def build_prediction_rows(
    examples: Sequence[LikelihoodExample],
    requests: Sequence[CandidateRequest],
    scores: Sequence[CandidateScore],
) -> list[dict[str, Any]]:
    grouped: dict[int, list[tuple[CandidateRequest, CandidateScore]]] = defaultdict(list)
    for request, score in zip(requests, scores):
        grouped[int(request.example_index)].append((request, score))
    rows: list[dict[str, Any]] = []
    for example in examples:
        pairs = sorted(grouped[example.item_index], key=lambda pair: pair[0].candidate_index)
        if len(pairs) != len(example.candidates):
            raise RuntimeError(f"incomplete candidate scores for {example.task}/{example.item_id}")
        candidate_scores = []
        for candidate, (request, score) in zip(example.candidates, pairs):
            candidate_scores.append(
                {
                    "candidate_index": int(request.candidate_index),
                    "text": candidate,
                    "loglikelihood": float(score.loglikelihood),
                    "normalized_loglikelihood": float(score.normalized_loglikelihood),
                    "perplexity": finite_perplexity(score.normalized_loglikelihood),
                    "token_count": int(score.token_count),
                    "greedy": bool(score.greedy),
                    "truncated_prompt_tokens": int(request.truncated_prompt_tokens),
                }
            )
        raw_prediction = argmax([value["loglikelihood"] for value in candidate_scores])
        normalized_prediction = argmax(
            [value["normalized_loglikelihood"] for value in candidate_scores]
        )
        rows.append(
            {
                "item_index": int(example.item_index),
                "item_id": example.item_id,
                "task": example.task,
                "kind": example.kind,
                "image_id": int(example.image_id),
                "label": example.label,
                "category": example.category,
                "metadata": example.metadata,
                "candidate_scores": candidate_scores,
                "prediction_raw": raw_prediction,
                "prediction_normalized": normalized_prediction,
                "correct_raw": (
                    bool(raw_prediction == example.label)
                    if example.label is not None
                    else None
                ),
                "correct_normalized": (
                    bool(normalized_prediction == example.label)
                    if example.label is not None
                    else None
                ),
            }
        )
    return rows


def mean(values: Sequence[float]) -> float:
    return float(sum(float(value) for value in values) / len(values)) if values else 0.0


def classification_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    raw_correct = [float(bool(row["correct_raw"])) for row in rows]
    norm_correct = [float(bool(row["correct_normalized"])) for row in rows]
    margins: list[float] = []
    gold_ll = 0.0
    gold_tokens = 0
    for row in rows:
        label = int(row["label"])
        scores = row["candidate_scores"]
        gold = float(scores[label]["normalized_loglikelihood"])
        best_other = max(
            float(value["normalized_loglikelihood"])
            for index, value in enumerate(scores)
            if index != label
        )
        margins.append(gold - best_other)
        gold_ll += float(scores[label]["loglikelihood"])
        gold_tokens += int(scores[label]["token_count"])
    token_nll = -gold_ll / max(1, gold_tokens)
    return {
        "records": len(rows),
        "accuracy_raw_loglikelihood": mean(raw_correct),
        "accuracy_normalized_loglikelihood": mean(norm_correct),
        "primary_metric": "accuracy_normalized_loglikelihood",
        "mean_normalized_margin": mean(margins),
        "gold_token_nll": token_nll,
        "gold_token_perplexity": math.exp(token_nll) if token_nll < 700.0 else float("inf"),
        "gold_tokens": gold_tokens,
    }


def pairwise_ranking_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Summarize positive-vs-negative text ranking, including strict ties.

    The ordinary classification accuracy uses the repository's deterministic
    candidate-index tie break.  Pairwise grounding probes additionally need a
    strict win rate and signed margin so that a textual-prior tie cannot be
    mistaken for visual discrimination.
    """

    if any(len(row["candidate_scores"]) != 2 for row in rows):
        raise ValueError("pairwise ranking rows must contain exactly two candidates")
    result = classification_metrics(rows)
    variants: dict[str, dict[str, Any]] = {}
    for score_name in ("loglikelihood", "normalized_loglikelihood"):
        margins: list[float] = []
        for row in rows:
            label = int(row["label"])
            scores = row["candidate_scores"]
            margins.append(
                float(scores[label][score_name])
                - float(scores[1 - label][score_name])
            )
        wins = sum(value > 0.0 for value in margins)
        ties = sum(value == 0.0 for value in margins)
        variants[score_name] = {
            "win_rate": wins / max(1, len(margins)),
            "tie_rate": ties / max(1, len(margins)),
            "loss_rate": (len(margins) - wins - ties) / max(1, len(margins)),
            "mean_margin": mean(margins),
            "median_margin": float(statistics.median(margins)) if margins else 0.0,
        }
    result.update(
        {
            "raw_pairwise": variants["loglikelihood"],
            "normalized_pairwise": variants["normalized_loglikelihood"],
            "primary_metric": "normalized_pairwise.win_rate",
        }
    )
    return result


def category_metrics(
    rows: Sequence[dict[str, Any]],
    metric_fn=classification_metrics,
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("category") or "uncategorized")].append(row)
    return {
        category: metric_fn(category_rows)
        for category, category_rows in sorted(grouped.items())
    }


def binary_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    result = classification_metrics(rows)
    true_positive = false_positive = false_negative = 0
    predicted_yes = 0
    for row in rows:
        prediction = int(row["prediction_normalized"])
        label = int(row["label"])
        predicted_yes += int(prediction == 0)
        true_positive += int(prediction == 0 and label == 0)
        false_positive += int(prediction == 0 and label == 1)
        false_negative += int(prediction == 1 and label == 0)
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    result.update(
        {
            "yes_precision": precision,
            "yes_recall": recall,
            "yes_f1": 2 * precision * recall / max(1.0e-12, precision + recall),
            "yes_ratio": predicted_yes / max(1, len(rows)),
        }
    )
    return result


def mmbench_circular_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Report both per-row and strict official circular grouping metrics.

    The model is scored by answer-text likelihood rather than free-form answer
    extraction.  The grouping itself follows MMBench CircularEval: an original
    question passes only when every supplied option rotation is correct.
    """

    result = classification_metrics(rows)
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    originals: list[dict[str, Any]] = []
    for row in rows:
        metadata = row.get("metadata") or {}
        original_index = int(metadata["original_index"])
        grouped[original_index].append(row)
        if bool(metadata.get("is_original", False)):
            originals.append(row)
    if len(originals) != len(grouped):
        raise ValueError(
            "MMBench circular rows must contain exactly one original row per group"
        )

    def accuracy(selected: Sequence[dict[str, Any]], field: str) -> float:
        return mean([float(bool(row[field])) for row in selected])

    circular_raw = mean(
        [float(all(bool(row["correct_raw"]) for row in group)) for group in grouped.values()]
    )
    circular_normalized = mean(
        [
            float(all(bool(row["correct_normalized"]) for row in group))
            for group in grouped.values()
        ]
    )
    group_size_counts: dict[str, int] = defaultdict(int)
    for group in grouped.values():
        group_size_counts[str(len(group))] += 1
    result.update(
        {
            "original_questions": len(grouped),
            "circular_rows": len(rows),
            "circular_group_size_counts": dict(sorted(group_size_counts.items())),
            "vanilla_accuracy_raw_loglikelihood": accuracy(originals, "correct_raw"),
            "vanilla_accuracy_normalized_loglikelihood": accuracy(
                originals, "correct_normalized"
            ),
            "circular_accuracy_raw_loglikelihood": circular_raw,
            "circular_accuracy_normalized_loglikelihood": circular_normalized,
            "primary_metric": "circular_accuracy_normalized_loglikelihood",
            "answer_scoring": "semantic_candidate_same_position_likelihood",
            "official_free_form_answer_extraction": False,
        }
    )
    return result


def caption_perplexity_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    total_ll = 0.0
    total_tokens = 0
    sequence_ppl: list[float] = []
    for row in rows:
        if len(row["candidate_scores"]) != 1:
            raise ValueError("caption_perplexity rows must contain one candidate")
        score = row["candidate_scores"][0]
        total_ll += float(score["loglikelihood"])
        total_tokens += int(score["token_count"])
        sequence_ppl.append(float(score["perplexity"]))
    token_nll = -total_ll / max(1, total_tokens)
    return {
        "records": len(rows),
        "tokens": total_tokens,
        "token_nll": token_nll,
        "token_perplexity": math.exp(token_nll) if token_nll < 700.0 else float("inf"),
        "mean_sequence_perplexity": mean(sequence_ppl),
        "primary_metric": "token_perplexity",
    }


def winoground_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        metadata = row.get("metadata") or {}
        grouped[str(metadata["group_id"])][int(metadata["image_slot"])] = row
    results: dict[str, dict[str, float]] = {}
    for score_name in ("loglikelihood", "normalized_loglikelihood"):
        text_correct: list[float] = []
        image_correct: list[float] = []
        group_correct: list[float] = []
        for group_id, pair in grouped.items():
            if set(pair) != {0, 1}:
                raise ValueError(f"incomplete Winoground group {group_id}")
            s00 = float(pair[0]["candidate_scores"][0][score_name])
            s01 = float(pair[0]["candidate_scores"][1][score_name])
            s10 = float(pair[1]["candidate_scores"][0][score_name])
            s11 = float(pair[1]["candidate_scores"][1][score_name])
            text_ok = s00 > s01 and s11 > s10
            image_ok = s00 > s10 and s11 > s01
            text_correct.append(float(text_ok))
            image_correct.append(float(image_ok))
            group_correct.append(float(text_ok and image_ok))
        results[score_name] = {
            "text_score": mean(text_correct),
            "image_score": mean(image_correct),
            "group_score": mean(group_correct),
        }
    return {
        "records": len(rows),
        "groups": len(grouped),
        "raw": results["loglikelihood"],
        "normalized": results["normalized_loglikelihood"],
        "primary_metric": "normalized.group_score",
    }


def paired_image_ranking_metrics(
    rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Compare the same caption on a positive and a negative image.

    This is the native SVO-Probes adaptation for a conditional generative
    model: every official pair is represented by two rows, and the positive
    image must assign the shared sentence a strictly higher likelihood.
    """

    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        metadata = row.get("metadata") or {}
        pair_id = str(metadata["pair_id"])
        role = str(metadata["image_role"])
        if role not in {"positive", "negative"}:
            raise ValueError(f"invalid paired-image role for {pair_id}: {role}")
        if role in grouped[pair_id]:
            raise ValueError(f"duplicate {role} row for paired-image item {pair_id}")
        if len(row["candidate_scores"]) != 1:
            raise ValueError("paired-image rows must contain one shared caption")
        grouped[pair_id][role] = row

    results: dict[str, dict[str, Any]] = {}
    for score_name in ("loglikelihood", "normalized_loglikelihood"):
        margins: list[float] = []
        category_margins: dict[str, list[float]] = defaultdict(list)
        for pair_id, pair in grouped.items():
            if set(pair) != {"positive", "negative"}:
                raise ValueError(f"incomplete paired-image item {pair_id}")
            positive = pair["positive"]
            negative = pair["negative"]
            positive_score = float(positive["candidate_scores"][0][score_name])
            negative_score = float(negative["candidate_scores"][0][score_name])
            margin = positive_score - negative_score
            margins.append(margin)
            category = str(
                (positive.get("metadata") or {}).get("negative_type")
                or positive.get("category")
                or "uncategorized"
            )
            category_margins[category].append(margin)

        def summarize_margins(values: Sequence[float]) -> dict[str, float]:
            wins = sum(value > 0.0 for value in values)
            ties = sum(value == 0.0 for value in values)
            return {
                "win_rate": wins / max(1, len(values)),
                "tie_rate": ties / max(1, len(values)),
                "loss_rate": (len(values) - wins - ties) / max(1, len(values)),
                "mean_margin": mean(values),
                "median_margin": (
                    float(statistics.median(values)) if values else 0.0
                ),
            }

        results[score_name] = {
            **summarize_margins(margins),
            "categories": {
                category: {
                    "pairs": len(values),
                    **summarize_margins(values),
                }
                for category, values in sorted(category_margins.items())
            },
        }
    return {
        "records": len(rows),
        "pairs": len(grouped),
        "raw": results["loglikelihood"],
        "normalized": results["normalized_loglikelihood"],
        "primary_metric": "normalized.win_rate",
        "scoring_direction": "shared_caption_positive_image_over_negative_image",
    }


def whatsup_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Compute official What’sUp individual, pair, and set accuracies."""

    grouped: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        metadata = row.get("metadata") or {}
        subset = str(metadata["subset"]).upper()
        set_id = str(metadata["set_id"])
        relation = str(metadata["relation"])
        if subset not in {"A", "B"}:
            raise ValueError(f"invalid What’sUp subset: {subset}")
        key = (subset, set_id)
        if relation in grouped[key]:
            raise ValueError(f"duplicate What’sUp relation {relation} in {key}")
        grouped[key][relation] = row

    expected_relations = {
        "A": ({"left", "right"}, {"on", "under"}),
        "B": ({"left", "right"}, {"in-front", "behind"}),
    }
    variants: dict[str, dict[str, Any]] = {}
    for prediction_field, score_key in (
        ("prediction_raw", "raw"),
        ("prediction_normalized", "normalized"),
    ):
        individual: list[float] = []
        pair_correct: list[float] = []
        set_correct: list[float] = []
        subset_values: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for (subset, set_id), relation_rows in grouped.items():
            relation_pairs = expected_relations[subset]
            expected = set().union(*relation_pairs)
            if set(relation_rows) != expected:
                raise ValueError(
                    f"incomplete What’sUp set {(subset, set_id)}: "
                    f"{sorted(relation_rows)} != {sorted(expected)}"
                )
            correctness = {
                relation: int(row[prediction_field]) == int(row["label"])
                for relation, row in relation_rows.items()
            }
            individual.extend(float(value) for value in correctness.values())
            subset_values[subset]["individual"].extend(
                float(value) for value in correctness.values()
            )
            current_pair_values = [
                float(all(correctness[relation] for relation in relation_pair))
                for relation_pair in relation_pairs
            ]
            pair_correct.extend(current_pair_values)
            subset_values[subset]["pair"].extend(current_pair_values)
            current_set = float(all(correctness.values()))
            set_correct.append(current_set)
            subset_values[subset]["set"].append(current_set)
        variants[score_key] = {
            "individual_accuracy": mean(individual),
            "pair_accuracy": mean(pair_correct),
            "set_accuracy": mean(set_correct),
            "subsets": {
                subset: {
                    "individual_accuracy": mean(values["individual"]),
                    "pair_accuracy": mean(values["pair"]),
                    "set_accuracy": mean(values["set"]),
                }
                for subset, values in sorted(subset_values.items())
            },
        }
    return {
        "records": len(rows),
        "sets": len(grouped),
        "raw": variants["raw"],
        "normalized": variants["normalized"],
        "primary_metric": "normalized.individual_accuracy",
    }


def summarize_task(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize an empty task")
    kind = str(rows[0]["kind"])
    if any(str(row["kind"]) != kind for row in rows):
        raise ValueError("task rows mix kinds")
    if kind == "caption_perplexity":
        metrics = caption_perplexity_metrics(rows)
    elif kind == "binary_classification":
        metrics = binary_metrics(rows)
    elif kind == "mmbench_circular_multiple_choice":
        metrics = mmbench_circular_metrics(rows)
    elif kind == "winoground_pair":
        metrics = winoground_metrics(rows)
    elif kind == "svo_image_pair":
        metrics = paired_image_ranking_metrics(rows)
    elif kind == "whatsup_controlled_spatial":
        metrics = whatsup_metrics(rows)
    elif kind == "pairwise_caption_ranking":
        metrics = pairwise_ranking_metrics(rows)
    else:
        metrics = classification_metrics(rows)
    if kind == "mmbench_circular_multiple_choice":
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[str(row.get("category") or "uncategorized")].append(row)
        metrics["categories"] = {
            category: mmbench_circular_metrics(category_rows)
            for category, category_rows in sorted(grouped.items())
        }
    elif kind == "pairwise_caption_ranking":
        metrics["categories"] = category_metrics(rows, pairwise_ranking_metrics)
    elif kind not in {
        "caption_perplexity",
        "winoground_pair",
        "svo_image_pair",
        "whatsup_controlled_spatial",
    }:
        metrics["categories"] = category_metrics(rows)
    return {"kind": kind, "metrics": metrics}


def shard_path(output_dir: Path, task: str, rank: int, world_size: int) -> Path:
    return output_dir / "shards" / task / f"rank-{rank:05d}-of-{world_size:05d}.jsonl"


def validate_prediction_shard(
    rows: Sequence[dict[str, Any]],
    examples: Sequence[LikelihoodExample],
    *,
    task: str,
) -> None:
    expected_indices = [int(example.item_index) for example in examples]
    actual_indices = [int(row["item_index"]) for row in rows]
    if actual_indices != expected_indices:
        raise ValueError(f"resumable prediction shard has wrong indices for {task}")
    for row, example in zip(rows, examples):
        if str(row.get("task")) != task or str(row.get("item_id")) != example.item_id:
            raise ValueError(f"resumable prediction shard identity mismatch for {task}")
        if len(row.get("candidate_scores", [])) != len(example.candidates):
            raise ValueError(f"resumable prediction shard candidate mismatch for {task}")


def evaluate_task(
    *,
    task: str,
    examples: Sequence[LikelihoodExample],
    model,
    tokenizer,
    cache: PosteriorCache,
    args: argparse.Namespace,
    rank: int,
    world_size: int,
    device: torch.device,
    image_sigma_order: str,
    attention_contract: str,
) -> dict[str, Any] | None:
    local_examples = list(examples[rank::world_size])
    started = time.monotonic()
    local_shard = shard_path(args.output_dir, task, rank, world_size)
    if local_shard.is_file():
        predictions = read_jsonl_predictions(local_shard)
        validate_prediction_shard(predictions, local_examples, task=task)
        print(
            json.dumps(
                {
                    "event": "multimodal_likelihood_shard_resumed",
                    "task": task,
                    "rank": rank,
                    "records": len(predictions),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    else:
        requests = [
            encode_candidate(
                tokenizer,
                example,
                candidate_index,
                image_tokens=int(model.config.image_tokens_per_img),
                boi_token_id=int(model.config.boi_token_id),
                eoi_token_id=int(model.config.eoi_token_id),
                image_mask_token_id=int(model.config.image_mask_token_id),
                max_length=int(args.max_length),
                image_sigma_order=image_sigma_order,
                seed=int(args.seed),
            )
            for example in local_examples
            for candidate_index in range(len(example.candidates))
        ]
        scores: list[CandidateScore] = []
        for offset in range(0, len(requests), int(args.progress_every)):
            chunk = requests[offset : offset + int(args.progress_every)]
            scores.extend(
                score_candidate_requests(
                    model,
                    chunk,
                    cache,
                    batch_size=int(args.batch_size_per_rank),
                    lm_head_chunk_tokens=int(args.lm_head_chunk_tokens),
                    attention_contract=attention_contract,
                    device=device,
                )
            )
            completed = min(offset + len(chunk), len(requests))
            elapsed = time.monotonic() - started
            print(
                json.dumps(
                    {
                        "event": "multimodal_likelihood_progress",
                        "task": task,
                        "rank": rank,
                        "candidate_requests": completed,
                        "candidate_requests_total": len(requests),
                        "elapsed_seconds": elapsed,
                        "requests_per_second": completed / max(elapsed, 1.0e-9),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        predictions = build_prediction_rows(local_examples, requests, scores)
        atomic_write_text(local_shard, jsonl_text(predictions))
    barrier(device)
    if rank != 0:
        barrier(device)
        return None
    merged: list[dict[str, Any]] = []
    for shard_rank in range(world_size):
        merged.extend(read_jsonl_predictions(shard_path(args.output_dir, task, shard_rank, world_size)))
    merged.sort(key=lambda row: int(row["item_index"]))
    if len(merged) != len(examples):
        raise RuntimeError(f"merged {len(merged)} rows for {task}, expected {len(examples)}")
    if [int(row["item_index"]) for row in merged] != list(range(len(examples))):
        raise RuntimeError(f"merged item indices are not complete for {task}")
    atomic_write_text(args.output_dir / "predictions" / f"{task}.jsonl", jsonl_text(merged))
    summary = summarize_task(merged)
    summary.update(
        {
            "task": task,
            "candidate_requests": sum(len(example.candidates) for example in examples),
            "elapsed_seconds_max_rank_approx": time.monotonic() - started,
        }
    )
    atomic_write_text(
        args.output_dir / "summaries" / f"{task}.json",
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    barrier(device)
    return summary


def read_jsonl_predictions(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    args = parse_args()
    source = model_source_from_args(args)
    tasks, asset_manifest = validate_args(args, source)
    checkpoint_step = source.global_step
    rank, world_size, local_rank, device = initialize_device(args.device)
    config = OmegaConf.load(args.config)
    configure_model_source(config, source)
    objective = str(config.model.get("training_objective", "selfless_dual_stream"))
    if objective != "selfless_dual_stream":
        raise ValueError(
            "likelihood benchmark runner currently requires selfless_dual_stream, "
            f"got {objective!r}"
        )
    attention_contract = str(
        config.model.get("dual_stream_attention_contract", "selfless_strict")
    ).strip().lower()
    if attention_contract not in {"selfless_strict", "xlnet_content_diagonal"}:
        raise ValueError(f"unknown dual-stream attention contract: {attention_contract}")
    configured_order = str(
        config.dataset.params.image.get("image_sigma_order", "random")
    ).strip().lower()
    image_sigma_order = (
        configured_order if args.image_sigma_order == "auto" else args.image_sigma_order
    )
    if image_sigma_order not in {"random", "sequential"}:
        raise ValueError(f"unknown image sigma order: {image_sigma_order}")

    model_dtype = torch.bfloat16 if args.model_dtype == "bf16" else torch.float32
    model, tokenizer = load_model_tokenizer(config, model_dtype=model_dtype)
    loaded_attention_contract = str(
        getattr(
            model.config,
            "dual_stream_attention_contract",
            "selfless_strict",
        )
    ).strip().lower()
    if loaded_attention_contract != attention_contract:
        raise ValueError(
            "loaded model attention contract does not match evaluation config: "
            f"model={loaded_attention_contract!r}, config={attention_contract!r}"
        )
    load_report = load_model_source_weights(model, source)
    model = model.to(device=device, dtype=model_dtype).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    cache = PosteriorCache(
        args.cache_shard_dir,
        expected_image_tokens=int(model.config.image_tokens_per_img),
        expected_latent_dim=int(model.config.image_latent_dim),
        seed=int(args.seed),
    )
    examples_by_task = {
        task: load_examples(task, asset_manifest, int(args.limit)) for task in tasks
    }
    missing_cache_ids = sorted(
        {
            example.image_id
            for examples in examples_by_task.values()
            for example in examples
            if example.image_id not in cache
        }
    )
    if missing_cache_ids:
        raise ValueError(
            f"posterior cache is missing {len(missing_cache_ids)} image IDs; "
            f"examples={missing_cache_ids[:8]}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        asset_manifest_path = (args.asset_root / "manifest.json").resolve()
        run_identity = {
            "schema": "selfless_multimodal_likelihood_evaluation_v2",
            "runtime_hashing_enabled": False,
            "checkpoint": str(source.path),
            "checkpoint_step": checkpoint_step,
            "weight_source": source.kind,
            "config": str(args.config.resolve()),
            "config_readable_identity": readable_file_identity(args.config),
            "asset_manifest": str(asset_manifest_path),
            "asset_manifest_readable_identity": readable_file_identity(
                asset_manifest_path
            ),
            "cache_shard_dir": str(args.cache_shard_dir.resolve()),
            "cache_images": len(cache),
            "tasks": list(tasks),
            "records": {
                task: len(examples) for task, examples in examples_by_task.items()
            },
            "world_size": world_size,
            "device": args.device,
            "model_dtype": args.model_dtype,
            "batch_size_per_rank": int(args.batch_size_per_rank),
            "max_length": int(args.max_length),
            "seed": int(args.seed),
            "image_sigma_order": image_sigma_order,
            "dual_stream_attention_contract": attention_contract,
            "scoring_contract": LIKELIHOOD_SCORING_CONTRACT,
            "query_stream_diagonal": False,
            "content_stream_diagonal": (
                attention_contract == "xlnet_content_diagonal"
            ),
            "primary_candidate_score": "normalized_loglikelihood",
            "candidate_target": "answer_or_caption_text_only_without_eos",
            "free_form_generation_required": False,
        }
        manifest_path = args.output_dir / "manifest.json"
        if manifest_path.is_file():
            run_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            mismatches = [
                key
                for key, expected in run_identity.items()
                if run_manifest.get(key) != expected
            ]
            if mismatches:
                raise ValueError(
                    "existing multimodal output has a different readable run "
                    f"identity: {mismatches}"
                )
            run_manifest["complete"] = False
            run_manifest["resumed"] = True
            run_manifest["resumed_at"] = utc_now()
        else:
            run_manifest = {
                **run_identity,
                "created_at": utc_now(),
                "complete": False,
                "resumed": False,
            }
        run_manifest["checkpoint_load"] = load_report
        atomic_write_text(
            manifest_path,
            json.dumps(run_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )
    barrier(device)

    summaries: dict[str, Any] = {}
    for task in tasks:
        summary = evaluate_task(
            task=task,
            examples=examples_by_task[task],
            model=model,
            tokenizer=tokenizer,
            cache=cache,
            args=args,
            rank=rank,
            world_size=world_size,
            device=device,
            image_sigma_order=image_sigma_order,
            attention_contract=attention_contract,
        )
        if rank == 0 and summary is not None:
            summaries[task] = summary

    if rank == 0:
        manifest_path = args.output_dir / "manifest.json"
        run_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        run_manifest["complete"] = True
        run_manifest["completed_at"] = utc_now()
        run_manifest["summaries"] = summaries
        atomic_write_text(
            manifest_path,
            json.dumps(run_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )
        atomic_write_text(
            args.output_dir / "summary.json",
            json.dumps(
                {
                    "schema": "selfless_multimodal_likelihood_summary_v2",
                    "checkpoint_step": checkpoint_step,
                    "runtime_hashing_enabled": False,
                    "scoring": {
                        "contract": LIKELIHOOD_SCORING_CONTRACT,
                        "dual_stream_attention_contract": attention_contract,
                        "query_stream_diagonal": False,
                        "content_stream_diagonal": (
                            attention_contract == "xlnet_content_diagonal"
                        ),
                    },
                    "tasks": summaries,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
        )
        print(json.dumps(summaries, ensure_ascii=False, indent=2, sort_keys=True))
    barrier(device)


if __name__ == "__main__":
    main()
