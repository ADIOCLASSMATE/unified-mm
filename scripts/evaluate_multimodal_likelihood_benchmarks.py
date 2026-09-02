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
from dataclasses import dataclass, replace
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
ASSET_SCHEMA = "selfless_multimodal_likelihood_assets_v2"
TASK_SCHEMA = "selfless_multimodal_likelihood_task_v1"
CACHE_FORMAT = "imagenet_kl16_scaled_posterior_v1"
CACHE_LAYOUT = "scaled_mean_then_scaled_std"
LIKELIHOOD_SCORING_CONTRACT = "selfless_same_position_dual_stream_v3"
IMAGE_ORDER_MC_CONTRACT = "shared_image_order_expected_loglikelihood_v1"
IMAGE_ORDER_MC_SEED_STRIDE = 1_000_003
LANGUAGE_PRIOR_ALPHA = 1.0
LANGUAGE_PRIOR_ESTIMATOR = "content_free_gaussian_image_logmeanexp"
LANGUAGE_PRIOR_NULL_IMAGE_COUNT = 3
LANGUAGE_PRIOR_NULL_IMAGE_IDS = tuple(
    9_000_000_000 + index for index in range(LANGUAGE_PRIOR_NULL_IMAGE_COUNT)
)
LANGUAGE_PRIOR_NULL_IMAGE_SEEDS = (17_071, 29_129, 43_231)
LANGUAGE_PRIOR_NULL_PIXEL_SPACE = "vae_preprocess_normalized_minus1_to_plus1"
LANGUAGE_PRIOR_NULL_CLAMP = [-1.0, 1.0]
LANGUAGE_PRIOR_NULL_STORAGE = "lossless_rgb_png"
DEBIASED_SCORE = "language_prior_debiased_mean_token_loglikelihood"
DEFAULT_TASKS = (
    "mmbench_dev_en",
    "seed_bench_image",
    "sugarcrepe",
    "aro_vg_relation",
    "aro_vg_attribution",
)
FORMAL_TASK_RECORDS = {
    "mmbench_dev_en": 4_329,
    "seed_bench_image": 14_233,
    "sugarcrepe": 7_511,
    "aro_vg_relation": 23_937,
    "aro_vg_attribution": 28_748,
}


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
    mc_sample_index: int = 0


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
    parser.add_argument(
        "--mc",
        type=int,
        default=1,
        help=(
            "Number of random image-token orders per candidate. The effective "
            "forward batch is batch_size_per_rank * mc."
        ),
    )
    parser.add_argument("--lm_head_chunk_tokens", type=int, default=256)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--require_formal_protocol", action="store_true")
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


def validate_language_prior_contract(prior: Any) -> None:
    if not isinstance(prior, dict):
        raise ValueError("asset manifest has no language-prior null-image contract")
    if prior.get("estimator") != LANGUAGE_PRIOR_ESTIMATOR:
        raise ValueError("asset manifest uses an unexpected language-prior estimator")
    if int(prior.get("count", -1)) != LANGUAGE_PRIOR_NULL_IMAGE_COUNT:
        raise ValueError("formal protocol requires exactly three null images")
    if bool(prior.get("uses_benchmark_labels", True)):
        raise ValueError("language-prior estimation must not use benchmark labels")
    if prior.get("pixel_space") != LANGUAGE_PRIOR_NULL_PIXEL_SPACE:
        raise ValueError("null images use an unexpected pixel space")
    if float(prior.get("normalized_gaussian_mean", math.inf)) != 0.0:
        raise ValueError("null images must have normalized Gaussian mean zero")
    if float(prior.get("normalized_gaussian_std", math.inf)) != 0.25:
        raise ValueError("null images must have normalized Gaussian std 0.25")
    if prior.get("clamp") != LANGUAGE_PRIOR_NULL_CLAMP:
        raise ValueError("null images must be clamped to the normalized [-1, 1] range")
    if prior.get("storage") != LANGUAGE_PRIOR_NULL_STORAGE:
        raise ValueError("formal null images must use lossless RGB PNG storage")
    null_rows = prior.get("images")
    if not isinstance(null_rows, list) or len(null_rows) != 3:
        raise ValueError("language-prior null-image records are incomplete")
    null_ids = tuple(int(row["image_id"]) for row in null_rows)
    if null_ids != LANGUAGE_PRIOR_NULL_IMAGE_IDS:
        raise ValueError("language-prior null-image IDs do not match the formal contract")
    null_seeds = tuple(int(row["seed"]) for row in null_rows)
    if null_seeds != LANGUAGE_PRIOR_NULL_IMAGE_SEEDS:
        raise ValueError("language-prior null-image seeds do not match the formal contract")
    for row in null_rows:
        path = Path(str(row.get("path", "")))
        if not path.is_file() or path.suffix.lower() != ".png":
            raise ValueError(f"language-prior null image is missing or not PNG: {path}")


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
        args.mc,
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
    validate_language_prior_contract(manifest.get("language_prior_null_images"))
    tasks = tuple(value.strip() for value in str(args.tasks).split(",") if value.strip())
    if not tasks:
        raise ValueError("at least one task is required")
    if bool(args.require_formal_protocol):
        if int(args.limit) != 0:
            raise ValueError("formal multimodal evaluation forbids row limiting")
        if int(args.mc) != 64:
            raise ValueError("formal multimodal evaluation requires MC=64")
        if tasks != DEFAULT_TASKS:
            raise ValueError(
                "formal multimodal evaluation requires the complete ordered task suite"
            )
    missing = sorted(set(tasks) - set(manifest.get("tasks", {})))
    if missing:
        unavailable = manifest.get("unavailable", {})
        details = {task: unavailable.get(task, "not present") for task in missing}
        raise ValueError(f"requested tasks are unavailable: {details}")
    return tasks, manifest


def language_prior_null_image_ids(asset_manifest: dict[str, Any]) -> tuple[int, ...]:
    return tuple(
        int(row["image_id"])
        for row in asset_manifest["language_prior_null_images"]["images"]
    )


def arithmetic_seed(seed: int, image_id: int, offset: int) -> int:
    return (int(seed) + 97_409 * int(image_id) + int(offset)) & ((1 << 63) - 1)


def image_order_mc_seed(seed: int, image_id: int, mc_sample_index: int) -> int:
    if int(mc_sample_index) < 0:
        raise ValueError("mc_sample_index must be non-negative")
    return arithmetic_seed(
        seed,
        image_id,
        53 + IMAGE_ORDER_MC_SEED_STRIDE * int(mc_sample_index),
    )


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
    mc_sample_index: int = 0,
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
        seed=image_order_mc_seed(seed, example.image_id, mc_sample_index),
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
        mc_sample_index=int(mc_sample_index),
    )


def encode_candidate_mc(
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
    mc_samples: int,
) -> list[CandidateRequest]:
    """Encode text once, then expand random image orders along the batch axis."""

    if int(mc_samples) <= 0:
        raise ValueError("mc_samples must be positive")
    first = encode_candidate(
        tokenizer,
        example,
        candidate_index,
        image_tokens=image_tokens,
        boi_token_id=boi_token_id,
        eoi_token_id=eoi_token_id,
        image_mask_token_id=image_mask_token_id,
        max_length=max_length,
        image_sigma_order=image_sigma_order,
        seed=seed,
        mc_sample_index=0,
    )
    requests = [first]
    prompt_length = int(first.image_start) - 1
    for mc_sample_index in range(1, int(mc_samples)):
        reveal = build_image_sigma(
            image_tokens,
            order=image_sigma_order,
            seed=image_order_mc_seed(seed, example.image_id, mc_sample_index),
        )
        sigma = list(first.sigma)
        for local_index, order_value in enumerate(reveal):
            sigma[first.image_start + local_index] = (
                prompt_length + 2 + int(order_value)
            )
        requests.append(
            replace(
                first,
                sigma=tuple(sigma),
                mc_sample_index=mc_sample_index,
            )
        )
    return requests


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


def logmeanexp(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("logmeanexp requires at least one value")
    if any(not math.isfinite(float(value)) for value in values):
        raise ValueError("logmeanexp requires finite values")
    maximum = max(float(value) for value in values)
    return maximum + math.log(
        sum(math.exp(float(value) - maximum) for value in values) / len(values)
    )


def build_prediction_rows(
    examples: Sequence[LikelihoodExample],
    requests: Sequence[CandidateRequest],
    scores: Sequence[CandidateScore],
    *,
    prior_requests: Sequence[CandidateRequest],
    prior_scores: Sequence[CandidateScore],
    null_image_ids: Sequence[int],
    mc_samples: int = 1,
) -> list[dict[str, Any]]:
    if len(requests) != len(scores):
        raise ValueError("candidate requests and scores must have equal lengths")
    if len(prior_requests) != len(prior_scores):
        raise ValueError("prior requests and scores must have equal lengths")
    if int(mc_samples) <= 0:
        raise ValueError("mc_samples must be positive")
    null_image_ids = tuple(int(value) for value in null_image_ids)
    if len(null_image_ids) != LANGUAGE_PRIOR_NULL_IMAGE_COUNT:
        raise ValueError("formal scoring requires exactly three null images")
    if len(set(null_image_ids)) != len(null_image_ids):
        raise ValueError("null image IDs must be unique")
    grouped: dict[
        tuple[int, int], list[tuple[CandidateRequest, CandidateScore]]
    ] = defaultdict(list)
    for request, score in zip(requests, scores):
        grouped[(int(request.example_index), int(request.candidate_index))].append(
            (request, score)
        )
    grouped_prior: dict[
        tuple[int, int, int], list[tuple[CandidateRequest, CandidateScore]]
    ] = defaultdict(list)
    for request, score in zip(prior_requests, prior_scores):
        grouped_prior[
            (
                int(request.example_index),
                int(request.candidate_index),
                int(request.image_id),
            )
        ].append((request, score))
    rows: list[dict[str, Any]] = []
    for example in examples:
        candidate_scores = []
        for candidate_index, candidate in enumerate(example.candidates):
            pairs = sorted(
                grouped[(int(example.item_index), candidate_index)],
                key=lambda pair: pair[0].mc_sample_index,
            )
            if len(pairs) != int(mc_samples):
                raise RuntimeError(
                    f"expected {mc_samples} MC scores for "
                    f"{example.task}/{example.item_id}/candidate-{candidate_index}, "
                    f"got {len(pairs)}"
                )
            sample_indices = [pair[0].mc_sample_index for pair in pairs]
            if sample_indices != list(range(int(mc_samples))):
                raise RuntimeError(
                    f"invalid MC sample indices for "
                    f"{example.task}/{example.item_id}/candidate-{candidate_index}: "
                    f"{sample_indices}"
                )
            token_counts = {pair[1].token_count for pair in pairs}
            truncated_counts = {
                pair[0].truncated_prompt_tokens for pair in pairs
            }
            if len(token_counts) != 1 or len(truncated_counts) != 1:
                raise RuntimeError(
                    f"MC samples disagree on candidate shape for "
                    f"{example.task}/{example.item_id}/candidate-{candidate_index}"
                )
            request = pairs[0][0]
            conditional_values = [
                pair[1].normalized_loglikelihood for pair in pairs
            ]
            null_means: list[float] = []
            for null_image_id in null_image_ids:
                null_pairs = sorted(
                    grouped_prior[
                        (
                            int(example.item_index),
                            candidate_index,
                            null_image_id,
                        )
                    ],
                    key=lambda pair: pair[0].mc_sample_index,
                )
                if len(null_pairs) != int(mc_samples):
                    raise RuntimeError(
                        f"expected {mc_samples} prior MC scores for "
                        f"{example.task}/{example.item_id}/candidate-{candidate_index}/"
                        f"null-{null_image_id}, got {len(null_pairs)}"
                    )
                if [pair[0].mc_sample_index for pair in null_pairs] != list(
                    range(int(mc_samples))
                ):
                    raise RuntimeError("invalid prior MC sample indices")
                prior_token_counts = {pair[1].token_count for pair in null_pairs}
                if prior_token_counts != token_counts:
                    raise RuntimeError("conditional and prior token counts disagree")
                null_means.append(
                    mean(
                        [
                            pair[1].normalized_loglikelihood
                            for pair in null_pairs
                        ]
                    )
                )
            conditional_score = mean(conditional_values)
            language_prior = logmeanexp(null_means)
            debiased_score = conditional_score - language_prior
            candidate_scores.append(
                {
                    "candidate_index": candidate_index,
                    "text": candidate,
                    DEBIASED_SCORE: debiased_score,
                    "estimated_language_prior_log_score": language_prior,
                    "token_count": int(next(iter(token_counts))),
                    "truncated_prompt_tokens": int(request.truncated_prompt_tokens),
                    "mc_samples": int(mc_samples),
                    "conditional_mc_mean_token_loglikelihood_std": float(
                        statistics.pstdev(conditional_values)
                    ),
                    "null_image_mean_token_loglikelihood_std": float(
                        statistics.pstdev(null_means)
                    ),
                    "language_prior_alpha": LANGUAGE_PRIOR_ALPHA,
                    "language_prior_null_images": len(null_image_ids),
                }
            )
        prediction = argmax(
            [value[DEBIASED_SCORE] for value in candidate_scores]
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
                "prediction_language_prior_debiased": prediction,
                "correct_language_prior_debiased": (
                    bool(prediction == example.label)
                    if example.label is not None
                    else None
                ),
            }
        )
    return rows


def chunk_examples_by_candidate_count(
    examples: Sequence[LikelihoodExample],
    max_candidates: int,
) -> Iterable[list[LikelihoodExample]]:
    """Yield whole-example chunks while bounding logical candidate count."""

    if int(max_candidates) <= 0:
        raise ValueError("max_candidates must be positive")
    chunk: list[LikelihoodExample] = []
    candidates = 0
    for example in examples:
        example_candidates = len(example.candidates)
        if chunk and candidates + example_candidates > int(max_candidates):
            yield chunk
            chunk = []
            candidates = 0
        chunk.append(example)
        candidates += example_candidates
    if chunk:
        yield chunk


def mean(values: Sequence[float]) -> float:
    return float(sum(float(value) for value in values) / len(values)) if values else 0.0


def classification_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    correct = [
        float(bool(row["correct_language_prior_debiased"])) for row in rows
    ]
    margins: list[float] = []
    for row in rows:
        label = int(row["label"])
        scores = row["candidate_scores"]
        gold = float(scores[label][DEBIASED_SCORE])
        best_other = max(
            float(value[DEBIASED_SCORE])
            for index, value in enumerate(scores)
            if index != label
        )
        margins.append(gold - best_other)
    return {
        "records": len(rows),
        "accuracy_language_prior_debiased": mean(correct),
        "primary_metric": "accuracy_language_prior_debiased",
        "mean_language_prior_debiased_margin": mean(margins),
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
    margins: list[float] = []
    for row in rows:
        label = int(row["label"])
        scores = row["candidate_scores"]
        margins.append(
            float(scores[label][DEBIASED_SCORE])
            - float(scores[1 - label][DEBIASED_SCORE])
        )
    wins = sum(value > 0.0 for value in margins)
    ties = sum(value == 0.0 for value in margins)
    pairwise = {
        "win_rate": wins / max(1, len(margins)),
        "tie_rate": ties / max(1, len(margins)),
        "loss_rate": (len(margins) - wins - ties) / max(1, len(margins)),
        "mean_margin": mean(margins),
        "median_margin": float(statistics.median(margins)) if margins else 0.0,
    }
    result.update(
        {
            "language_prior_debiased_pairwise": pairwise,
            "primary_metric": "language_prior_debiased_pairwise.win_rate",
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
        prediction = int(row["prediction_language_prior_debiased"])
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

    circular = mean(
        [
            float(
                all(
                    bool(row["correct_language_prior_debiased"])
                    for row in group
                )
            )
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
            "vanilla_accuracy_language_prior_debiased": mean(
                [
                    float(bool(row["correct_language_prior_debiased"]))
                    for row in originals
                ]
            ),
            "circular_accuracy_language_prior_debiased": circular,
            "primary_metric": "circular_accuracy_language_prior_debiased",
            "answer_scoring": "semantic_candidate_same_position_likelihood",
            "official_free_form_answer_extraction": False,
        }
    )
    return result


def winoground_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        metadata = row.get("metadata") or {}
        grouped[str(metadata["group_id"])][int(metadata["image_slot"])] = row
    text_correct: list[float] = []
    image_correct: list[float] = []
    group_correct: list[float] = []
    for group_id, pair in grouped.items():
        if set(pair) != {0, 1}:
            raise ValueError(f"incomplete Winoground group {group_id}")
        s00 = float(pair[0]["candidate_scores"][0][DEBIASED_SCORE])
        s01 = float(pair[0]["candidate_scores"][1][DEBIASED_SCORE])
        s10 = float(pair[1]["candidate_scores"][0][DEBIASED_SCORE])
        s11 = float(pair[1]["candidate_scores"][1][DEBIASED_SCORE])
        text_ok = s00 > s01 and s11 > s10
        image_ok = s00 > s10 and s11 > s01
        text_correct.append(float(text_ok))
        image_correct.append(float(image_ok))
        group_correct.append(float(text_ok and image_ok))
    result = {
        "text_score": mean(text_correct),
        "image_score": mean(image_correct),
        "group_score": mean(group_correct),
    }
    return {
        "records": len(rows),
        "groups": len(grouped),
        "language_prior_debiased": result,
        "primary_metric": "language_prior_debiased.group_score",
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

    margins: list[float] = []
    category_margins: dict[str, list[float]] = defaultdict(list)
    for pair_id, pair in grouped.items():
        if set(pair) != {"positive", "negative"}:
            raise ValueError(f"incomplete paired-image item {pair_id}")
        positive = pair["positive"]
        negative = pair["negative"]
        positive_score = float(positive["candidate_scores"][0][DEBIASED_SCORE])
        negative_score = float(negative["candidate_scores"][0][DEBIASED_SCORE])
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

    result = {
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
        "language_prior_debiased": result,
        "primary_metric": "language_prior_debiased.win_rate",
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
            relation: int(row["prediction_language_prior_debiased"])
            == int(row["label"])
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
    result = {
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
        "language_prior_debiased": result,
        "primary_metric": "language_prior_debiased.individual_accuracy",
    }


def summarize_task(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize an empty task")
    kind = str(rows[0]["kind"])
    if any(str(row["kind"]) != kind for row in rows):
        raise ValueError("task rows mix kinds")
    if kind == "caption_perplexity":
        raise ValueError(
            "caption perplexity was removed from the formal protocol; it is not "
            "a language-prior-debiased image-text matching metric"
        )
    if kind == "binary_classification":
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
    mc_samples: int,
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
        candidate_mc = {
            int(score.get("mc_samples", 1)) for score in row["candidate_scores"]
        }
        if candidate_mc != {int(mc_samples)}:
            raise ValueError(
                f"resumable prediction shard MC mismatch for {task}: "
                f"{sorted(candidate_mc)}"
            )
        if any(DEBIASED_SCORE not in score for score in row["candidate_scores"]):
            raise ValueError(
                f"resumable prediction shard uses the removed score protocol for {task}"
            )
        prior_counts = {
            int(score.get("language_prior_null_images", -1))
            for score in row["candidate_scores"]
        }
        if prior_counts != {LANGUAGE_PRIOR_NULL_IMAGE_COUNT}:
            raise ValueError(f"resumable prediction shard has wrong prior for {task}")


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
    null_image_ids: Sequence[int],
) -> dict[str, Any] | None:
    local_examples = list(examples[rank::world_size])
    started = time.monotonic()
    local_shard = shard_path(args.output_dir, task, rank, world_size)
    if local_shard.is_file():
        predictions = read_jsonl_predictions(local_shard)
        validate_prediction_shard(
            predictions,
            local_examples,
            task=task,
            mc_samples=int(args.mc),
        )
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
        predictions = []
        logical_total = sum(len(example.candidates) for example in local_examples)
        logical_completed = 0
        for example_chunk in chunk_examples_by_candidate_count(
            local_examples,
            int(args.progress_every),
        ):
            requests = [
                request
                for example in example_chunk
                for candidate_index in range(len(example.candidates))
                for request in encode_candidate_mc(
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
                    mc_samples=int(args.mc),
                )
            ]
            scores = score_candidate_requests(
                model,
                requests,
                cache,
                batch_size=int(args.batch_size_per_rank) * int(args.mc),
                lm_head_chunk_tokens=int(args.lm_head_chunk_tokens),
                attention_contract=attention_contract,
                device=device,
            )
            prior_requests = [
                request
                for example in example_chunk
                for candidate_index in range(len(example.candidates))
                for null_image_id in null_image_ids
                for request in encode_candidate_mc(
                    tokenizer,
                    replace(example, image_id=int(null_image_id)),
                    candidate_index,
                    image_tokens=int(model.config.image_tokens_per_img),
                    boi_token_id=int(model.config.boi_token_id),
                    eoi_token_id=int(model.config.eoi_token_id),
                    image_mask_token_id=int(model.config.image_mask_token_id),
                    max_length=int(args.max_length),
                    image_sigma_order=image_sigma_order,
                    seed=int(args.seed),
                    mc_samples=int(args.mc),
                )
            ]
            prior_scores = score_candidate_requests(
                model,
                prior_requests,
                cache,
                batch_size=int(args.batch_size_per_rank) * int(args.mc),
                lm_head_chunk_tokens=int(args.lm_head_chunk_tokens),
                attention_contract=attention_contract,
                device=device,
            )
            predictions.extend(
                build_prediction_rows(
                    example_chunk,
                    requests,
                    scores,
                    prior_requests=prior_requests,
                    prior_scores=prior_scores,
                    null_image_ids=null_image_ids,
                    mc_samples=int(args.mc),
                )
            )
            logical_completed += sum(
                len(example.candidates) for example in example_chunk
            )
            elapsed = time.monotonic() - started
            print(
                json.dumps(
                    {
                        "event": "multimodal_likelihood_progress",
                        "task": task,
                        "rank": rank,
                        "candidate_groups": logical_completed,
                        "candidate_groups_total": logical_total,
                        "mc_samples": int(args.mc),
                        "forward_sequences": logical_completed
                        * int(args.mc)
                        * (1 + len(null_image_ids)),
                        "forward_sequences_total": logical_total
                        * int(args.mc)
                        * (1 + len(null_image_ids)),
                        "elapsed_seconds": elapsed,
                        "sequences_per_second": (
                            logical_completed
                            * int(args.mc)
                            * (1 + len(null_image_ids))
                        )
                        / max(elapsed, 1.0e-9),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
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
            "candidate_groups": sum(len(example.candidates) for example in examples),
            "candidate_requests": int(args.mc)
            * (1 + len(null_image_ids))
            * sum(len(example.candidates) for example in examples),
            "mc_samples": int(args.mc),
            "language_prior_null_images": len(null_image_ids),
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
    if int(args.mc) > 1 and image_sigma_order != "random":
        raise ValueError(
            "--mc > 1 requires random image_sigma_order; a sequential order has "
            "no image-order distribution to sample"
        )

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
    if bool(args.require_formal_protocol):
        actual_records = {
            task: len(examples) for task, examples in examples_by_task.items()
        }
        if actual_records != FORMAL_TASK_RECORDS:
            raise ValueError(
                "formal multimodal task cardinality mismatch: "
                f"{actual_records} != {FORMAL_TASK_RECORDS}"
            )
    null_image_ids = language_prior_null_image_ids(asset_manifest)
    missing_cache_ids = sorted(
        {
            example.image_id
            for examples in examples_by_task.values()
            for example in examples
            if example.image_id not in cache
        }
        | {image_id for image_id in null_image_ids if image_id not in cache}
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
            "schema": "selfless_multimodal_likelihood_evaluation_v5",
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
            "mc_samples": int(args.mc),
            "project_formal_protocol": bool(args.require_formal_protocol),
            "effective_forward_batch_size_per_rank": (
                int(args.batch_size_per_rank) * int(args.mc)
            ),
            "mc_aggregation": "mean_loglikelihood",
            "mc_common_random_numbers_across_candidates": True,
            "image_order_mc_contract": IMAGE_ORDER_MC_CONTRACT,
            "language_prior_alpha": LANGUAGE_PRIOR_ALPHA,
            "language_prior_estimator": LANGUAGE_PRIOR_ESTIMATOR,
            "language_prior_null_image_ids": list(null_image_ids),
            "language_prior_null_image_count": len(null_image_ids),
            "language_prior_uses_labels": False,
            "max_length": int(args.max_length),
            "seed": int(args.seed),
            "image_sigma_order": image_sigma_order,
            "dual_stream_attention_contract": attention_contract,
            "scoring_contract": LIKELIHOOD_SCORING_CONTRACT,
            "query_stream_diagonal": False,
            "content_stream_diagonal": (
                attention_contract == "xlnet_content_diagonal"
            ),
            "primary_candidate_score": DEBIASED_SCORE,
            "reported_score_variant": "language_prior_debiased_only",
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
            null_image_ids=null_image_ids,
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
                    "schema": "selfless_multimodal_likelihood_summary_v5",
                    "project_formal_protocol": bool(args.require_formal_protocol),
                    "checkpoint_step": checkpoint_step,
                    "runtime_hashing_enabled": False,
                    "accuracy_and_rate_unit": "unit_interval",
                    "scoring": {
                        "contract": LIKELIHOOD_SCORING_CONTRACT,
                        "dual_stream_attention_contract": attention_contract,
                        "query_stream_diagonal": False,
                        "content_stream_diagonal": (
                            attention_contract == "xlnet_content_diagonal"
                        ),
                        "mc_samples": int(args.mc),
                        "mc_aggregation": "mean_loglikelihood",
                        "mc_common_random_numbers_across_candidates": True,
                        "image_order_mc_contract": IMAGE_ORDER_MC_CONTRACT,
                        "primary_candidate_score": DEBIASED_SCORE,
                        "reported_score_variant": "language_prior_debiased_only",
                        "language_prior_alpha": LANGUAGE_PRIOR_ALPHA,
                        "language_prior_estimator": LANGUAGE_PRIOR_ESTIMATOR,
                        "language_prior_null_image_count": len(null_image_ids),
                        "language_prior_uses_labels": False,
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
