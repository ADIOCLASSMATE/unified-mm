#!/usr/bin/env python3
"""Evaluate the custom ImageNet-val image-text retrieval protocol.

This evaluator uses the Selfless same-position image-conditioned text
likelihood directly.  It does not fine-tune a classifier, use a CLIP proxy, or
apply a causal-LM one-token shift.  The reference backend streams candidate
requests and is correctness-first: the 1K x 1K and 5K x 5K retrieval
protocols are intentionally reported as high-cost evaluations.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Iterable, Sequence

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import torch
from torch.nn.attention.flex_attention import create_block_mask
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_multimodal_likelihood_benchmarks import (  # noqa: E402
    CandidateScore,
    LIKELIHOOD_SCORING_CONTRACT,
    LikelihoodExample,
    PosteriorCache,
    atomic_write_text,
    barrier,
    encode_candidate,
    initialize_device,
    jsonl_text,
    score_candidate_requests,
    utc_now,
)
from models.modeling_model.image_position_utils import (  # noqa: E402
    build_row_col_position_ids,
)
from models.modeling_model.modeling_selfless_flow import (  # noqa: E402
    SelflessStaticCache,
)
from utils.evaluation_model_source import (  # noqa: E402
    add_model_source_argument,
    configure_model_source,
    load_model_source_weights,
    model_source_from_args,
)
from utils.utils import load_model_tokenizer  # noqa: E402


DEFAULT_CONFIG = Path("configs/selfless/unified_baseline_100b_ascend_64npu.yaml")
DEFAULT_MANIFEST = Path("public/datasets/imagenet_full/manifest_val.jsonl")
DEFAULT_CAPTIONS = Path(
    "public/datasets/imagenet1k_synthetic_v1/captions/"
    "imagenet1k_val_visual_descriptions.jsonl"
)
DEFAULT_CLASSES = Path("public/datasets/imagenet1k_synthetic_v1/t2i/classes.json")
DEFAULT_CACHE = Path(
    "public/datasets/imagenet_full/vae_posterior_mar_kl16/val_shards"
)
DEFAULT_TASKS = ("retrieval_1k", "retrieval_5k")
RETRIEVAL_PROMPT = "Describe this image in one detailed caption:"


@dataclass(frozen=True)
class ImageNetRecord:
    index: int
    img_id: int
    source_path: str
    image_id: str
    synset: str
    class_index: int
    caption: str


class CachedTokenizer:
    """Cache the small fixed evaluation candidate pool within each rank."""

    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer
        self._encoded: dict[tuple[str, bool], tuple[int, ...]] = {}

    @property
    def eos_token_id(self):
        return self.tokenizer.eos_token_id

    def encode(self, text, add_special_tokens=False):
        key = (str(text), bool(add_special_tokens))
        value = self._encoded.get(key)
        if value is None:
            value = tuple(
                int(token)
                for token in self.tokenizer.encode(
                    str(text), add_special_tokens=bool(add_special_tokens)
                )
            )
            self._encoded[key] = value
        return list(value)

    def __getattr__(self, name: str):
        return getattr(self.tokenizer, name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    add_model_source_argument(parser)
    parser.add_argument("--image_manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--captions", type=Path, default=DEFAULT_CAPTIONS)
    parser.add_argument("--classes", type=Path, default=DEFAULT_CLASSES)
    parser.add_argument("--cache_shard_dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    parser.add_argument("--batch_size_per_rank", type=int, default=32)
    parser.add_argument("--request_chunk_size", type=int, default=128)
    parser.add_argument("--lm_head_chunk_tokens", type=int, default=256)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=424242)
    parser.add_argument("--device", choices=("npu", "cuda", "cpu"), default="npu")
    parser.add_argument("--model_dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument(
        "--scoring_backend",
        choices=("cached_prefix", "repeated_full_sequence"),
        default="cached_prefix",
    )
    parser.add_argument(
        "--image_sigma_order",
        choices=("auto", "random", "sequential"),
        default="auto",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_imagenet_records(
    manifest_path: Path,
    captions_path: Path,
    classes_path: Path,
) -> list[ImageNetRecord]:
    images = read_jsonl(manifest_path)
    captions = read_jsonl(captions_path)
    class_payload = json.loads(classes_path.read_text(encoding="utf-8"))
    class_rows = list(class_payload["classes"])
    if len(images) != 50_000 or len(captions) != 50_000:
        raise ValueError("formal ImageNet-val evaluation requires exactly 50,000 rows")
    if len(class_rows) != 1_000:
        raise ValueError("ImageNet class metadata has unexpected cardinality")
    synset_to_class = {
        str(row["synset"]): int(row["class_index"]) for row in class_rows
    }
    if len(synset_to_class) != 1_000:
        raise ValueError("invalid ImageNet class metadata")

    records: list[ImageNetRecord] = []
    class_counts: dict[int, int] = defaultdict(int)
    for index, (image, caption) in enumerate(zip(images, captions)):
        caption_values = caption.get("captions") or []
        if len(caption_values) != 1:
            raise ValueError(f"expected one ImageNet caption at row {index}")
        if (
            int(image["manifest_index"]) != index
            or int(caption["manifest_index"]) != index
            or int(image["img_id"]) != int(caption["img_id"])
            or str(image["synset"]) != str(caption["synset"])
            or str(image.get("split")) != "val"
            or str(caption.get("split")) != "val"
        ):
            raise ValueError(f"ImageNet val alignment mismatch at row {index}")
        synset = str(image["synset"])
        class_index = synset_to_class[synset]
        class_counts[class_index] += 1
        records.append(
            ImageNetRecord(
                index=index,
                img_id=int(image["img_id"]),
                source_path=str(image["source_path"]),
                image_id=str(image["image_id"]),
                synset=synset,
                class_index=class_index,
                caption=str(caption_values[0]["text"]).strip(),
            )
        )
    if set(class_counts) != set(range(1_000)) or set(class_counts.values()) != {50}:
        raise ValueError("ImageNet val must contain 50 examples for every class")
    return records


def arithmetic_order_key(record: ImageNetRecord, seed: int) -> tuple[int, int]:
    value = (
        int(record.img_id) * 1_103_515_245
        + int(seed) * 12_345
        + int(record.class_index) * 97_409
    ) & ((1 << 63) - 1)
    return value, int(record.img_id)


def stratified_records(
    records: Sequence[ImageNetRecord],
    per_class: int,
    seed: int,
) -> list[ImageNetRecord]:
    if not 1 <= int(per_class) <= 50:
        raise ValueError("per_class must be in [1, 50]")
    grouped: dict[int, list[ImageNetRecord]] = defaultdict(list)
    for record in records:
        grouped[int(record.class_index)].append(record)
    selected: list[ImageNetRecord] = []
    for class_index in range(1_000):
        ordered = sorted(
            grouped[class_index],
            key=lambda record: arithmetic_order_key(record, int(seed)),
        )
        selected.extend(ordered[: int(per_class)])
    return selected


def score_text_candidates(
    *,
    model,
    tokenizer,
    cache: PosteriorCache,
    image_id: int,
    item_id: str,
    prompt: str,
    candidates: Sequence[str],
    args: argparse.Namespace,
    device: torch.device,
    image_sigma_order: str,
    attention_contract: str,
    evaluation_task: str = "imagenet_pretraining_native",
) -> tuple[torch.Tensor, torch.Tensor]:
    example = LikelihoodExample(
        item_index=0,
        item_id=str(item_id),
        task=str(evaluation_task),
        kind="candidate_matrix_row",
        image_id=int(image_id),
        prompt=str(prompt),
        candidates=tuple(str(value) for value in candidates),
        label=None,
        category=None,
        metadata={},
    )
    scores: list[CandidateScore] = []
    for offset in range(0, len(candidates), int(args.request_chunk_size)):
        stop = min(offset + int(args.request_chunk_size), len(candidates))
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
            for candidate_index in range(offset, stop)
        ]
        scores.extend(
            score_candidate_requests(
                model,
                requests,
                cache,
                batch_size=int(args.batch_size_per_rank),
                lm_head_chunk_tokens=int(args.lm_head_chunk_tokens),
                attention_contract=attention_contract,
                device=device,
            )
        )
    raw = torch.tensor([value.loglikelihood for value in scores], dtype=torch.float32)
    normalized = torch.tensor(
        [value.normalized_loglikelihood for value in scores], dtype=torch.float32
    )
    return raw, normalized


def cross_sigma_attention_mask(
    *,
    query_sigma: torch.Tensor,
    key_sigma: torch.Tensor,
    query_valid: torch.Tensor,
    key_valid: torch.Tensor,
    include_diagonal: bool,
    device: torch.device,
):
    """Build a QxKV Selfless mask for a static prefix cache."""

    if query_sigma.ndim != 2 or key_sigma.ndim != 2:
        raise ValueError("cached sigma tensors must be rank two")
    if query_valid.shape != query_sigma.shape or key_valid.shape != key_sigma.shape:
        raise ValueError("cached sigma/valid shapes must align")
    comparison = (
        key_sigma.unsqueeze(1) <= query_sigma.unsqueeze(-1)
        if include_diagonal
        else key_sigma.unsqueeze(1) < query_sigma.unsqueeze(-1)
    )
    allowed = (
        query_valid.unsqueeze(-1)
        & key_valid.unsqueeze(1)
        & comparison
    )
    if device.type == "npu":
        return (~allowed).unsqueeze(1)

    def mask_mod(batch, head, query_index, key_index):
        del head
        return allowed[batch, query_index, key_index]

    return create_block_mask(
        mask_mod,
        B=int(allowed.shape[0]),
        H=None,
        Q_LEN=int(allowed.shape[1]),
        KV_LEN=int(allowed.shape[2]),
        device=device,
    )


def clone_prefix_cache(
    prefix_cache: SelflessStaticCache,
    *,
    config,
    batch_size: int,
) -> SelflessStaticCache:
    cloned = SelflessStaticCache(config, prefix_cache.get_max_cache_shape())
    for source, destination in zip(prefix_cache.layers, cloned.layers):
        if not source.is_initialized:
            raise RuntimeError("cannot clone an uninitialized prefix cache")
        destination.keys = source.keys.expand(batch_size, -1, -1, -1).clone()
        destination.values = source.values.expand(batch_size, -1, -1, -1).clone()
        destination.is_initialized = True
    cached_text_source = getattr(
        prefix_cache, "_single_stream_text_ar_last_hidden", None
    )
    if cached_text_source is not None:
        cloned._single_stream_text_ar_last_hidden = cached_text_source.expand(
            batch_size, -1
        ).clone()
    return cloned


@torch.inference_mode()
def score_text_candidates_cached_prefix(
    *,
    model,
    tokenizer,
    cache: PosteriorCache,
    image_id: int,
    item_id: str,
    prompt: str,
    candidates: Sequence[str],
    args: argparse.Namespace,
    device: torch.device,
    image_sigma_order: str,
    attention_contract: str,
    evaluation_task: str = "imagenet_pretraining_native",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score many suffixes while computing the shared image prefix once."""

    example = LikelihoodExample(
        item_index=0,
        item_id=str(item_id),
        task=str(evaluation_task),
        kind="candidate_matrix_row",
        image_id=int(image_id),
        prompt=str(prompt),
        candidates=tuple(str(value) for value in candidates),
        label=None,
        category=None,
        metadata={},
    )
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
        for candidate_index in range(len(candidates))
    ]
    if not requests:
        return torch.empty(0), torch.empty(0)
    prefix_length = int(requests[0].target_start)
    prefix_identity = (
        requests[0].input_ids[:prefix_length],
        requests[0].token_types[:prefix_length],
        requests[0].sigma[:prefix_length],
        requests[0].image_start,
    )
    if any(
        (
            request.input_ids[: request.target_start],
            request.token_types[: request.target_start],
            request.sigma[: request.target_start],
            request.image_start,
        )
        != prefix_identity
        for request in requests[1:]
    ):
        # Candidate-dependent prompt truncation destroys the shared prefix.
        return score_text_candidates(
            model=model,
            tokenizer=tokenizer,
            cache=cache,
            image_id=image_id,
            item_id=item_id,
            prompt=prompt,
            candidates=candidates,
            args=args,
            device=device,
            image_sigma_order=image_sigma_order,
            attention_contract=attention_contract,
            evaluation_task=evaluation_task,
        )

    max_cache_length = max(len(request.input_ids) for request in requests)
    prefix_input_ids = torch.tensor(
        [requests[0].input_ids[:prefix_length]], device=device, dtype=torch.long
    )
    prefix_token_types = torch.tensor(
        [requests[0].token_types[:prefix_length]],
        device=device,
        dtype=torch.uint8,
    )
    prefix_sigma = torch.tensor(
        [requests[0].sigma[:prefix_length]], device=device, dtype=torch.long
    )
    prefix_valid = torch.ones_like(prefix_sigma, dtype=torch.bool)
    prefix_key_sigma = torch.zeros(
        (1, max_cache_length), device=device, dtype=torch.long
    )
    prefix_key_sigma[:, :prefix_length] = prefix_sigma
    prefix_key_valid = torch.zeros(
        (1, max_cache_length), device=device, dtype=torch.bool
    )
    prefix_key_valid[:, :prefix_length] = True
    prefix_query_mask = cross_sigma_attention_mask(
        query_sigma=prefix_sigma,
        key_sigma=prefix_key_sigma,
        query_valid=prefix_valid,
        key_valid=prefix_key_valid,
        include_diagonal=False,
        device=device,
    )
    prefix_content_mask = (
        cross_sigma_attention_mask(
            query_sigma=prefix_sigma,
            key_sigma=prefix_key_sigma,
            query_valid=prefix_valid,
            key_valid=prefix_key_valid,
            include_diagonal=True,
            device=device,
        )
        if attention_contract == "xlnet_content_diagonal"
        else None
    )
    latent_dim = int(model.config.image_latent_dim)
    image_tokens = int(model.config.image_tokens_per_img)
    image_start = int(requests[0].image_start)
    image_end = image_start + image_tokens
    prefix_latents = torch.zeros(
        (1, prefix_length, latent_dim),
        device=device,
        dtype=cache.sample(image_id).dtype,
    )
    prefix_latents[:, image_start:image_end] = cache.sample(image_id).to(device)
    prefix_latent_mask = torch.zeros(
        (1, prefix_length), device=device, dtype=torch.bool
    )
    prefix_latent_mask[:, image_start:image_end] = True
    prefix_cache = SelflessStaticCache(model.config, max_cache_length)
    model.model(
        X0_input_ids=prefix_input_ids,
        attention_mask=prefix_query_mask,
        content_attention_mask=prefix_content_mask,
        position_ids=build_row_col_position_ids(
            prefix_token_types, image_tokens
        ),
        past_key_values=prefix_cache,
        use_cache=True,
        cache_position=torch.arange(prefix_length, device=device).unsqueeze(0),
        calculate_likelihood=False,
        _text_ar_mode=True,
        token_types=prefix_token_types,
        image_latents=prefix_latents,
        image_latent_mask=prefix_latent_mask,
        image_span_table=torch.tensor(
            [[0, 0, image_start, image_end, int(image_id)]],
            device=device,
            dtype=torch.long,
        ),
    )

    grouped: dict[int, list[tuple[int, Any]]] = defaultdict(list)
    for request_index, request in enumerate(requests):
        grouped[len(request.input_ids) - prefix_length].append(
            (request_index, request)
        )
    ordered_scores: list[CandidateScore | None] = [None] * len(requests)
    for candidate_length, group in sorted(grouped.items()):
        for offset in range(0, len(group), int(args.batch_size_per_rank)):
            batch_pairs = group[offset : offset + int(args.batch_size_per_rank)]
            batch_size = len(batch_pairs)
            batch_requests = [request for _, request in batch_pairs]
            input_ids = torch.tensor(
                [request.input_ids[prefix_length:] for request in batch_requests],
                device=device,
                dtype=torch.long,
            )
            token_types = torch.zeros(
                (batch_size, candidate_length), device=device, dtype=torch.uint8
            )
            query_sigma = torch.tensor(
                [request.sigma[prefix_length:] for request in batch_requests],
                device=device,
                dtype=torch.long,
            )
            query_valid = torch.ones_like(query_sigma, dtype=torch.bool)
            key_sigma = torch.zeros(
                (batch_size, max_cache_length), device=device, dtype=torch.long
            )
            key_sigma[:, :prefix_length] = prefix_sigma.expand(batch_size, -1)
            key_sigma[:, prefix_length : prefix_length + candidate_length] = (
                query_sigma
            )
            key_valid = torch.zeros(
                (batch_size, max_cache_length), device=device, dtype=torch.bool
            )
            key_valid[:, : prefix_length + candidate_length] = True
            query_mask = cross_sigma_attention_mask(
                query_sigma=query_sigma,
                key_sigma=key_sigma,
                query_valid=query_valid,
                key_valid=key_valid,
                include_diagonal=False,
                device=device,
            )
            content_mask = (
                cross_sigma_attention_mask(
                    query_sigma=query_sigma,
                    key_sigma=key_sigma,
                    query_valid=query_valid,
                    key_valid=key_valid,
                    include_diagonal=True,
                    device=device,
                )
                if attention_contract == "xlnet_content_diagonal"
                else None
            )
            full_token_types = torch.cat(
                [prefix_token_types.expand(batch_size, -1), token_types], dim=1
            )
            position_ids = build_row_col_position_ids(
                full_token_types, image_tokens
            )[:, :, prefix_length:]
            candidate_cache = clone_prefix_cache(
                prefix_cache, config=model.config, batch_size=batch_size
            )
            outputs = model.model(
                X0_input_ids=input_ids,
                attention_mask=query_mask,
                content_attention_mask=content_mask,
                position_ids=position_ids,
                past_key_values=candidate_cache,
                use_cache=True,
                cache_position=torch.arange(
                    prefix_length,
                    prefix_length + candidate_length,
                    device=device,
                ).unsqueeze(0).expand(batch_size, -1),
                calculate_likelihood=True,
                token_types=token_types,
            )
            hidden = outputs.last_hidden_state.reshape(-1, outputs.last_hidden_state.shape[-1])
            targets = input_ids.reshape(-1)
            token_logprobs: list[torch.Tensor] = []
            token_greedy: list[torch.Tensor] = []
            for chunk_start in range(0, hidden.shape[0], int(args.lm_head_chunk_tokens)):
                chunk_stop = min(
                    chunk_start + int(args.lm_head_chunk_tokens), hidden.shape[0]
                )
                logits = model.lm_head(hidden[chunk_start:chunk_stop])
                chunk_targets = targets[chunk_start:chunk_stop]
                gold = logits.gather(1, chunk_targets.unsqueeze(1)).squeeze(1).float()
                token_logprobs.append(gold - torch.logsumexp(logits.float(), dim=-1))
                token_greedy.append(logits.argmax(dim=-1).eq(chunk_targets))
            logprobs = torch.cat(token_logprobs).view(batch_size, candidate_length)
            greedy = torch.cat(token_greedy).view(batch_size, candidate_length)
            for row_index, (request_index, _) in enumerate(batch_pairs):
                total = float(logprobs[row_index].sum().item())
                ordered_scores[request_index] = CandidateScore(
                    loglikelihood=total,
                    normalized_loglikelihood=total / candidate_length,
                    token_count=candidate_length,
                    greedy=bool(greedy[row_index].all().item()),
                )
            del candidate_cache, outputs, hidden, logits
    if any(value is None for value in ordered_scores):
        raise RuntimeError("cached-prefix candidate scoring is incomplete")
    scores = [value for value in ordered_scores if value is not None]
    return (
        torch.tensor([value.loglikelihood for value in scores], dtype=torch.float32),
        torch.tensor(
            [value.normalized_loglikelihood for value in scores], dtype=torch.float32
        ),
    )


def score_candidates_with_backend(**kwargs) -> tuple[torch.Tensor, torch.Tensor]:
    args = kwargs["args"]
    if args.scoring_backend == "cached_prefix":
        return score_text_candidates_cached_prefix(**kwargs)
    return score_text_candidates(**kwargs)


def retrieval_direction_metrics(
    score_matrix: torch.Tensor,
    class_indices: torch.Tensor,
) -> dict[str, Any]:
    count = int(score_matrix.shape[0])
    order = torch.argsort(score_matrix, dim=1, descending=True, stable=True)
    positives = torch.arange(count, dtype=torch.long).unsqueeze(1)
    ranks = (order == positives).nonzero(as_tuple=False)[:, 1] + 1
    query_classes = class_indices.unsqueeze(1)
    retrieved_classes = class_indices[order]
    output: dict[str, Any] = {
        "median_rank": float(ranks.float().median().item()),
        "mean_rank": float(ranks.float().mean().item()),
    }
    recalls: list[float] = []
    for k in (1, 5, 10):
        effective = min(k, count)
        instance = float((ranks <= effective).float().mean().item())
        class_relevant = float(
            (retrieved_classes[:, :effective] == query_classes)
            .any(dim=1)
            .float()
            .mean()
            .item()
        )
        output[f"instance_recall_at_{k}"] = instance
        output[f"class_relevance_recall_at_{k}"] = class_relevant
        recalls.append(instance)
    output["instance_mean_recall_at_1_5_10"] = sum(recalls) / len(recalls)
    return output


def retrieval_metrics(
    scores: torch.Tensor,
    class_indices: torch.Tensor,
) -> dict[str, Any]:
    if scores.ndim != 2 or scores.shape[0] != scores.shape[1]:
        raise ValueError("retrieval score matrix must be square")
    image_to_text = retrieval_direction_metrics(scores, class_indices)
    text_to_image = retrieval_direction_metrics(scores.T, class_indices)
    result: dict[str, Any] = {
        "records": int(scores.shape[0]),
        "exact_instance_positive": "matrix_diagonal",
        "same_class_relevance_reported": True,
        "primary_metric": (
            "normalized_loglikelihood.mean_bidirectional_instance_recall_at_1"
        ),
        "normalized_loglikelihood": {
            "image_to_text": image_to_text,
            "text_to_image": text_to_image,
            "mean_bidirectional_instance_recall_at_1": (
                image_to_text["instance_recall_at_1"]
                + text_to_image["instance_recall_at_1"]
            )
            / 2.0,
            "mean_bidirectional_instance_recall_at_1_5_10": (
                image_to_text["instance_mean_recall_at_1_5_10"]
                + text_to_image["instance_mean_recall_at_1_5_10"]
            )
            / 2.0,
        },
    }
    return result


def atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
        torch.save(payload, temporary)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def evaluate_retrieval(
    *,
    name: str,
    records: Sequence[ImageNetRecord],
    per_class: int,
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
    selected = stratified_records(records, per_class, int(args.seed))
    if int(args.limit) > 0:
        selected = selected[: int(args.limit)]
    candidates = [record.caption for record in selected]
    local_pairs = list(enumerate(selected))[rank::world_size]
    local_scores = torch.empty((len(local_pairs), len(selected)), dtype=torch.float32)
    query_indices: list[int] = []
    started = time.monotonic()
    for local_index, (query_index, record) in enumerate(local_pairs):
        _, scores = score_candidates_with_backend(
            model=model,
            tokenizer=tokenizer,
            cache=cache,
            image_id=record.img_id,
            item_id=f"{name}/{record.image_id}",
            prompt=RETRIEVAL_PROMPT,
            candidates=candidates,
            args=args,
            device=device,
            image_sigma_order=image_sigma_order,
            attention_contract=attention_contract,
        )
        local_scores[local_index] = scores
        query_indices.append(query_index)
        print(
            json.dumps(
                {
                    "event": "imagenet_retrieval_progress",
                    "task": name,
                    "rank": rank,
                    "query_rows": local_index + 1,
                    "query_rows_total": len(local_pairs),
                    "candidate_captions": len(candidates),
                    "elapsed_seconds": time.monotonic() - started,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    shard = args.output_dir / name / "shards" / (
        f"rank-{rank:05d}-of-{world_size:05d}.pt"
    )
    atomic_torch_save(
        shard,
        {
            "query_indices": torch.tensor(query_indices, dtype=torch.long),
            "normalized_loglikelihood": local_scores,
            "scoring_contract": LIKELIHOOD_SCORING_CONTRACT,
            "dual_stream_attention_contract": attention_contract,
            "runtime_hashing_enabled": False,
        },
    )
    barrier(device)
    if rank != 0:
        return None
    matrix = torch.empty((len(selected), len(selected)), dtype=torch.float32)
    seen = torch.zeros(len(selected), dtype=torch.bool)
    for shard_rank in range(world_size):
        path = args.output_dir / name / "shards" / (
            f"rank-{shard_rank:05d}-of-{world_size:05d}.pt"
        )
        payload = torch.load(str(path), map_location="cpu", weights_only=True)
        if payload.get("runtime_hashing_enabled", True) is not False:
            raise ValueError("retrieval shard violates the no-hash contract")
        if payload.get("scoring_contract") != LIKELIHOOD_SCORING_CONTRACT:
            raise ValueError("retrieval shard uses the wrong scoring contract")
        if payload.get("dual_stream_attention_contract") != attention_contract:
            raise ValueError("retrieval shard uses the wrong attention contract")
        indices = payload["query_indices"].long()
        if bool(seen[indices].any()):
            raise ValueError("duplicate retrieval query rows across shards")
        matrix[indices] = payload["normalized_loglikelihood"].float()
        seen[indices] = True
    if not bool(seen.all()):
        raise RuntimeError("retrieval matrix is incomplete")
    class_indices = torch.tensor(
        [record.class_index for record in selected], dtype=torch.long
    )
    summary = retrieval_metrics(matrix, class_indices)
    summary.update(
        {
            "task": name,
            "formal_target_records": per_class * 1_000,
            "complete_formal_target": len(selected) == per_class * 1_000,
            "selection": "deterministic_class_balanced_arithmetic_order",
            "images_per_class": per_class,
            "prompt": RETRIEVAL_PROMPT,
        }
    )
    atomic_write_text(
        args.output_dir / name / "summary.json",
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    atomic_torch_save(
        args.output_dir / name / "score_matrix.pt",
        {
            "normalized_loglikelihood": matrix,
            "img_ids": torch.tensor([record.img_id for record in selected]),
            "class_indices": class_indices,
            "scoring_contract": LIKELIHOOD_SCORING_CONTRACT,
            "dual_stream_attention_contract": attention_contract,
            "runtime_hashing_enabled": False,
        },
    )
    return summary


def validate_args(args: argparse.Namespace, source) -> tuple[str, ...]:
    for path in (
        args.config,
        source.path,
        args.image_manifest,
        args.captions,
        args.classes,
        args.cache_shard_dir,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    for value in (
        args.batch_size_per_rank,
        args.request_chunk_size,
        args.lm_head_chunk_tokens,
        args.max_length,
    ):
        if int(value) <= 0:
            raise ValueError("batch/chunk/length arguments must be positive")
    if int(args.limit) < 0:
        raise ValueError("limit must be non-negative")
    tasks = tuple(value.strip() for value in str(args.tasks).split(",") if value.strip())
    allowed = set(DEFAULT_TASKS)
    unknown = sorted(set(tasks) - allowed)
    if not tasks or unknown:
        raise ValueError(f"invalid ImageNet pretraining-native tasks: {unknown}")
    return tasks


def main() -> None:
    args = parse_args()
    source = model_source_from_args(args)
    tasks = validate_args(args, source)
    checkpoint_step = source.global_step
    rank, world_size, _, device = initialize_device(args.device)
    config = OmegaConf.load(args.config)
    configure_model_source(config, source)
    objective = str(config.model.get("training_objective", "selfless_dual_stream"))
    if objective != "selfless_dual_stream":
        raise ValueError(f"pretraining-native likelihood requires Selfless, got {objective}")
    attention_contract = str(
        config.model.get("dual_stream_attention_contract", "selfless_strict")
    ).strip().lower()
    configured_order = str(
        config.dataset.params.image.get("image_sigma_order", "random")
    ).strip().lower()
    image_sigma_order = (
        configured_order if args.image_sigma_order == "auto" else args.image_sigma_order
    )
    if attention_contract not in {"selfless_strict", "xlnet_content_diagonal"}:
        raise ValueError(f"unknown attention contract: {attention_contract}")
    if image_sigma_order not in {"random", "sequential"}:
        raise ValueError(f"unknown image sigma order: {image_sigma_order}")

    records = load_imagenet_records(
        args.image_manifest, args.captions, args.classes
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
    tokenizer = CachedTokenizer(tokenizer)
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
    missing = [record.img_id for record in records if record.img_id not in cache]
    if missing:
        raise ValueError(f"ImageNet posterior cache is missing IDs: {missing[:8]}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    if rank == 0:
        manifest = {
            "schema": "selfless_imagenet_retrieval_evaluation_v3",
            "complete": False,
            "created_at": utc_now(),
            "runtime_hashing_enabled": False,
            "checkpoint": str(source.path),
            "checkpoint_step": checkpoint_step,
            "weight_source": source.kind,
            "config": str(args.config.resolve()),
            "tasks": list(tasks),
            "world_size": world_size,
            "dataset_split": "imagenet_val",
            "dataset_records": len(records),
            "training_overlap_allowed": False,
            "scoring": {
                "model_contract": "selfless_same_position_query_stream",
                "contract": LIKELIHOOD_SCORING_CONTRACT,
                "dual_stream_attention_contract": attention_contract,
                "query_stream_diagonal": False,
                "content_stream_diagonal": (
                    attention_contract == "xlnet_content_diagonal"
                ),
                "causal_lm_one_token_shift": False,
                "primary_candidate_score": "mean_token_loglikelihood",
                "scoring_backend": str(args.scoring_backend),
                "reference_backend": "streamed_repeated_full_sequence_likelihood",
            },
            "checkpoint_load": load_report,
        }
        atomic_write_text(
            manifest_path,
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )
    barrier(device)

    summaries: dict[str, Any] = {}
    for task_name, per_class in (("retrieval_1k", 1), ("retrieval_5k", 5)):
        if task_name not in tasks:
            continue
        summary = evaluate_retrieval(
            name=task_name,
            records=records,
            per_class=per_class,
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
            summaries[task_name] = summary
    if rank == 0:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["complete"] = True
        manifest["completed_at"] = utc_now()
        manifest["summaries"] = summaries
        atomic_write_text(
            manifest_path,
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )
        atomic_write_text(
            args.output_dir / "summary.json",
            json.dumps(
                {
                    "schema": "selfless_imagenet_retrieval_summary_v3",
                    "checkpoint_step": checkpoint_step,
                    "runtime_hashing_enabled": False,
                    "scoring": {
                        "contract": LIKELIHOOD_SCORING_CONTRACT,
                        "dual_stream_attention_contract": attention_contract,
                        "query_stream_diagonal": False,
                        "content_stream_diagonal": (
                            attention_contract == "xlnet_content_diagonal"
                        ),
                        "backend": str(args.scoring_backend),
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
