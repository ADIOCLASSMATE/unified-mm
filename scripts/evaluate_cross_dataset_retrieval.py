#!/usr/bin/env python3
"""Evaluate calibrated COCO/Flickr Karpathy-test image-text retrieval."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Sequence

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.evaluation.native_understanding import (  # noqa: E402
    CachedTokenizer,
    RETRIEVAL_PROMPT,
    atomic_torch_save,
    score_candidates_with_backend,
)
from utils.evaluation.multimodal_likelihood import (  # noqa: E402
    LIKELIHOOD_SCORING_CONTRACT,
    PosteriorCache,
    atomic_write_text,
    barrier,
    initialize_device,
    utc_now,
)
from utils.evaluation.calibration import (  # noqa: E402
    LANGUAGE_PRIOR_ALPHA,
    LANGUAGE_PRIOR_ESTIMATOR,
    language_prior_debiased_scores,
)
from utils.evaluation_model_source import (  # noqa: E402
    add_model_source_argument,
    configure_model_source,
    load_model_source_weights,
    model_source_from_args,
)
from utils.utils import load_model_tokenizer  # noqa: E402


DEFAULT_CONFIG = Path("configs/selfless/unified_baseline_100b_ascend_64npu.yaml")
SUPPORTED_TASKS = {
    "mscoco_karpathy_test_5k": (5_000, 25_010),
    "flickr30k_karpathy_test_1k": (1_000, 5_000),
}
EXPECTED_CAPTION_COUNT_DISTRIBUTIONS = {
    "mscoco_karpathy_test_5k": {5: 4_990, 6: 10},
    "flickr30k_karpathy_test_1k": {5: 1_000},
}


@dataclass(frozen=True)
class RetrievalRecord:
    image_index: int
    img_id: int
    source_image_id: str
    source_path: str
    captions: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    add_model_source_argument(parser)
    parser.add_argument("--asset_root", type=Path, required=True)
    parser.add_argument("--cache_shard_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--batch_size_per_rank", type=int, default=32)
    parser.add_argument("--request_chunk_size", type=int, default=128)
    parser.add_argument("--lm_head_chunk_tokens", type=int, default=256)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--require_formal_protocol", action="store_true")
    parser.add_argument("--query_partition_index", type=int, default=0)
    parser.add_argument("--query_partition_count", type=int, default=1)
    parser.add_argument("--seed", type=int, default=424242)
    parser.add_argument("--progress_every", type=int, default=1)
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


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def load_records(asset_root: Path) -> tuple[dict[str, Any], list[RetrievalRecord]]:
    manifest = read_json(asset_root / "manifest.json")
    if manifest.get("schema") != "selfless_cross_dataset_retrieval_assets_v1":
        raise ValueError("cross-dataset retrieval assets use an obsolete schema")
    if manifest.get("complete") is not True:
        raise ValueError("cross-dataset retrieval assets are incomplete")
    if manifest.get("runtime_hashing_enabled", True) is not False:
        raise ValueError("cross-dataset retrieval assets violate the no-hash contract")
    task = str(manifest.get("task"))
    if task not in SUPPORTED_TASKS:
        raise ValueError(f"unsupported cross-dataset retrieval task: {task}")
    expected_dataset = {
        "mscoco_karpathy_test_5k": "mscoco",
        "flickr30k_karpathy_test_1k": "flickr30k",
    }[task]
    if manifest.get("dataset") != expected_dataset or manifest.get("split") != (
        "karpathy_test"
    ):
        raise ValueError(f"formal {task} asset dataset/split identity is invalid")
    protocol = manifest.get("protocol") or {}
    if (
        protocol.get("bidirectional") is not True
        or protocol.get("coco_five_fold_1k_average") is not False
        or protocol.get("recall_at") != [1, 5, 10]
    ):
        raise ValueError(f"formal {task} retrieval protocol metadata is invalid")
    records: list[RetrievalRecord] = []
    path = asset_root / str(manifest["files"]["retrieval"])
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            captions = tuple(str(value).strip() for value in row["captions"])
            if not captions or any(not value for value in captions):
                raise ValueError("retrieval captions must be non-empty")
            records.append(
                RetrievalRecord(
                    image_index=int(row["image_index"]),
                    img_id=int(row["img_id"]),
                    source_image_id=str(row["source_image_id"]),
                    source_path=str(row["source_path"]),
                    captions=captions,
                )
            )
    expected_images, expected_captions = SUPPORTED_TASKS[task]
    captions = sum(len(record.captions) for record in records)
    if (
        len(records) != expected_images
        or captions != expected_captions
        or int(manifest.get("images", -1)) != expected_images
        or int(manifest.get("captions", -1)) != expected_captions
    ):
        raise ValueError(f"formal {task} asset cardinality is invalid")
    caption_count_distribution = Counter(len(record.captions) for record in records)
    if caption_count_distribution != Counter(
        EXPECTED_CAPTION_COUNT_DISTRIBUTIONS[task]
    ):
        raise ValueError(f"formal {task} caption-count distribution is invalid")
    manifest_distribution = {
        int(count): int(images)
        for count, images in (manifest.get("caption_count_distribution") or {}).items()
    }
    if manifest_distribution != EXPECTED_CAPTION_COUNT_DISTRIBUTIONS[task]:
        raise ValueError(f"formal {task} manifest caption counts are invalid")
    if [record.image_index for record in records] != list(range(len(records))):
        raise ValueError("retrieval image indices are not contiguous")
    if len({record.img_id for record in records}) != len(records):
        raise ValueError("retrieval image IDs are not unique")
    if len({record.source_image_id for record in records}) != len(records):
        raise ValueError("retrieval source image IDs are not unique")
    if len({record.source_path for record in records}) != len(records):
        raise ValueError("retrieval source image paths are not unique")
    return manifest, records


def stable_positive_ranks(
    scores: torch.Tensor,
    positive_indices: torch.Tensor,
    *,
    chunk_size: int = 256,
) -> torch.Tensor:
    """Return one-based ranks, matching descending stable sort without sorting."""

    if scores.ndim != 2 or positive_indices.ndim != 2:
        raise ValueError("scores and positive_indices must be rank two")
    if scores.shape[0] != positive_indices.shape[0]:
        raise ValueError("positive rows do not align with score queries")
    ranks = torch.empty(scores.shape[0], dtype=torch.long)
    candidate_indices = torch.arange(scores.shape[1], dtype=torch.long)
    for start in range(0, scores.shape[0], int(chunk_size)):
        stop = min(start + int(chunk_size), scores.shape[0])
        block = scores[start:stop]
        positives = positive_indices[start:stop]
        valid_positives = positives.ge(0)
        if not bool(valid_positives.any(dim=1).all()):
            raise ValueError("each retrieval query must have at least one positive")
        safe_positives = positives.clamp_min(0)
        positive_scores = block.gather(1, safe_positives).masked_fill(
            ~valid_positives, -torch.inf
        )
        best_scores, best_offsets = positive_scores.max(dim=1)
        winners = safe_positives.gather(1, best_offsets.unsqueeze(1)).squeeze(1)
        strictly_better = (block > best_scores.unsqueeze(1)).sum(dim=1)
        earlier_ties = (
            (block == best_scores.unsqueeze(1))
            & (candidate_indices.unsqueeze(0) < winners.unsqueeze(1))
        ).sum(dim=1)
        ranks[start:stop] = 1 + strictly_better + earlier_ties
    return ranks


def direction_metrics(ranks: torch.Tensor, candidates: int) -> dict[str, Any]:
    output: dict[str, Any] = {
        "queries": int(ranks.numel()),
        "candidates": int(candidates),
        "mean_rank": float(ranks.float().mean().item()),
        "median_rank": float(ranks.float().median().item()),
    }
    recalls = []
    for k in (1, 5, 10):
        value = float((ranks <= min(k, candidates)).float().mean().item())
        output[f"recall_at_{k}"] = value
        recalls.append(value)
    output["mean_recall_at_1_5_10"] = sum(recalls) / len(recalls)
    return output


def retrieval_metrics(
    prior_debiased_loglikelihood: torch.Tensor,
    caption_counts: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Compute the six canonical recalls from an already calibrated matrix."""

    scores = prior_debiased_loglikelihood
    if scores.ndim != 2:
        raise ValueError("retrieval scores must be a matrix")
    images, captions = map(int, scores.shape)
    if caption_counts is None:
        if images <= 0 or captions % images:
            raise ValueError("caption counts are required for a ragged retrieval matrix")
        caption_counts = [captions // images] * images
    caption_counts = [int(value) for value in caption_counts]
    if (
        len(caption_counts) != images
        or any(value <= 0 for value in caption_counts)
        or sum(caption_counts) != captions
    ):
        raise ValueError("caption counts do not align with the retrieval matrix")
    caption_to_image = torch.arange(images, dtype=torch.long).repeat_interleave(
        torch.tensor(caption_counts, dtype=torch.long)
    )
    image_positives = torch.full(
        (images, max(caption_counts)), -1, dtype=torch.long
    )
    offset = 0
    for image_index, count in enumerate(caption_counts):
        image_positives[image_index, :count] = torch.arange(
            offset, offset + count, dtype=torch.long
        )
        offset += count
    i2t_ranks = stable_positive_ranks(scores, image_positives)
    t2i_ranks = stable_positive_ranks(
        scores.T.contiguous(), caption_to_image.unsqueeze(1)
    )
    i2t = direction_metrics(i2t_ranks, captions)
    t2i = direction_metrics(t2i_ranks, images)
    return {
        "images": images,
        "captions": captions,
        "recall_unit": "unit_interval",
        "rank_unit": "one_based_candidate_rank",
        "caption_count_distribution": {
            str(count): image_count
            for count, image_count in sorted(Counter(caption_counts).items())
        },
        "primary_metric": "mean_recall_at_1_5_10",
        "image_to_text": i2t,
        "text_to_image": t2i,
        "mean_recall_at_1": (
            i2t["recall_at_1"] + t2i["recall_at_1"]
        )
        / 2.0,
        "mean_recall_at_1_5_10": (
            i2t["mean_recall_at_1_5_10"] + t2i["mean_recall_at_1_5_10"]
        )
        / 2.0,
    }


def validate_args(args: argparse.Namespace, source) -> None:
    for path in (args.config, source.path, args.asset_root, args.cache_shard_dir):
        if not path.exists():
            raise FileNotFoundError(path)
    for name in (
        "batch_size_per_rank",
        "request_chunk_size",
        "lm_head_chunk_tokens",
        "max_length",
        "progress_every",
    ):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name} must be positive")
    if int(args.limit) < 0:
        raise ValueError("--limit must be non-negative")
    if bool(args.require_formal_protocol) and int(args.limit) != 0:
        raise ValueError("formal retrieval forbids row limiting")
    if int(args.query_partition_count) <= 0:
        raise ValueError("--query_partition_count must be positive")
    if not 0 <= int(args.query_partition_index) < int(args.query_partition_count):
        raise ValueError(
            "--query_partition_index must be in [0, query_partition_count)"
        )


def main() -> None:
    args = parse_args()
    source = model_source_from_args(args)
    validate_args(args, source)
    asset_manifest, records = load_records(args.asset_root)
    formal_images, formal_captions = SUPPORTED_TASKS[str(asset_manifest["task"])]
    if int(args.limit) > 0:
        records = records[: int(args.limit)]
    captions = [caption for record in records for caption in record.captions]
    caption_counts = [len(record.captions) for record in records]
    checkpoint_step = source.global_step
    rank, world_size, _, device = initialize_device(args.device)
    config = OmegaConf.load(args.config)
    configure_model_source(config, source)
    objective = str(config.model.get("training_objective", "selfless_dual_stream"))
    if objective != "selfless_dual_stream":
        raise ValueError(f"retrieval likelihood requires Selfless, got {objective}")
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
        raise ValueError(f"posterior cache is missing retrieval image IDs: {missing[:8]}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    if rank == 0:
        manifest = {
            "schema": "selfless_cross_dataset_retrieval_evaluation_v3",
            "complete": False,
            "created_at": utc_now(),
            "runtime_hashing_enabled": False,
            "checkpoint": str(source.path),
            "checkpoint_step": checkpoint_step,
            "weight_source": source.kind,
            "config": str(args.config.resolve()),
            "task": asset_manifest["task"],
            "split": "karpathy_test",
            "images": len(records),
            "captions": len(captions),
            "caption_count_distribution": {
                str(count): images
                for count, images in sorted(Counter(caption_counts).items())
            },
            "formal_target_images": formal_images,
            "formal_target_captions": formal_captions,
            "project_formal_protocol": bool(args.require_formal_protocol),
            "world_size": world_size,
            "query_partition": {
                "method": "image_index_modulo",
                "index": int(args.query_partition_index),
                "count": int(args.query_partition_count),
                "query_rows": sum(
                    query_index % int(args.query_partition_count)
                    == int(args.query_partition_index)
                    for query_index in range(len(records))
                ),
            },
            "scoring": {
                "model_contract": "selfless_same_position_query_stream",
                "contract": LIKELIHOOD_SCORING_CONTRACT,
                "dual_stream_attention_contract": attention_contract,
                "query_stream_diagonal": False,
                "content_stream_diagonal": (
                    attention_contract == "xlnet_content_diagonal"
                ),
                "causal_lm_one_token_shift": False,
                "conditional_candidate_score": "mean_token_loglikelihood",
                "primary_candidate_score": (
                    "language_prior_debiased_mean_token_loglikelihood"
                ),
                "language_prior_alpha": LANGUAGE_PRIOR_ALPHA,
                "language_prior_estimator": LANGUAGE_PRIOR_ESTIMATOR,
                "language_prior_image_set": "all_candidate_images",
                "language_prior_uses_labels": False,
                "calibration_deferred_until_partition_merge": (
                    int(args.query_partition_count) > 1
                ),
                "scoring_backend": str(args.scoring_backend),
            },
            "checkpoint_load": load_report,
        }
        atomic_write_text(
            manifest_path,
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )
    barrier(device)

    partition_pairs = [
        (query_index, record)
        for query_index, record in enumerate(records)
        if query_index % int(args.query_partition_count)
        == int(args.query_partition_index)
    ]
    local_pairs = partition_pairs[rank::world_size]
    local_scores = torch.empty((len(local_pairs), len(captions)), dtype=torch.float32)
    query_indices: list[int] = []
    started = time.monotonic()
    for local_index, (query_index, record) in enumerate(local_pairs):
        _, scores = score_candidates_with_backend(
            model=model,
            tokenizer=tokenizer,
            cache=cache,
            image_id=record.img_id,
            item_id=f"{asset_manifest['task']}/{record.source_image_id}",
            prompt=RETRIEVAL_PROMPT,
            candidates=captions,
            args=args,
            device=device,
            image_sigma_order=image_sigma_order,
            attention_contract=attention_contract,
            evaluation_task=str(asset_manifest["task"]),
        )
        local_scores[local_index] = scores
        query_indices.append(query_index)
        if (local_index + 1) % int(args.progress_every) == 0 or local_index + 1 == len(local_pairs):
            print(
                json.dumps(
                    {
                        "event": "cross_dataset_retrieval_progress",
                        "task": asset_manifest["task"],
                        "rank": rank,
                        "query_rows": local_index + 1,
                        "query_rows_total": len(local_pairs),
                        "candidate_captions": len(captions),
                        "elapsed_seconds": time.monotonic() - started,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    shard_path = args.output_dir / "shards" / (
        f"rank-{rank:05d}-of-{world_size:05d}.pt"
    )
    atomic_torch_save(
        shard_path,
        {
            "query_indices": torch.tensor(query_indices, dtype=torch.long),
            "conditional_mean_token_loglikelihood": local_scores,
            "scoring_contract": LIKELIHOOD_SCORING_CONTRACT,
            "dual_stream_attention_contract": attention_contract,
            "runtime_hashing_enabled": False,
        },
    )
    barrier(device)
    if rank == 0:
        if int(args.query_partition_count) > 1:
            expected = torch.zeros(len(records), dtype=torch.bool)
            expected[
                int(args.query_partition_index) :: int(args.query_partition_count)
            ] = True
            seen = torch.zeros(len(records), dtype=torch.bool)
            for shard_rank in range(world_size):
                path = args.output_dir / "shards" / (
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
                scores = payload["conditional_mean_token_loglikelihood"]
                if scores.shape != (indices.numel(), len(captions)):
                    raise ValueError("partition retrieval shard shape mismatch")
                if indices.unique().numel() != indices.numel():
                    raise ValueError("duplicate retrieval rows within a partition shard")
                if indices.numel() and (
                    int(indices.min()) < 0 or int(indices.max()) >= len(records)
                ):
                    raise ValueError("partition retrieval shard has out-of-range rows")
                if bool(seen[indices].any()):
                    raise ValueError("duplicate retrieval rows across partition shards")
                seen[indices] = True
            if not torch.equal(seen, expected):
                raise RuntimeError("cross-dataset retrieval partition is incomplete")
            partition_summary = {
                "schema": "selfless_cross_dataset_retrieval_partition_v3",
                "complete": True,
                "runtime_hashing_enabled": False,
                "checkpoint": str(source.path),
                "checkpoint_step": checkpoint_step,
                "task": asset_manifest["task"],
                "split": "karpathy_test",
                "images": len(records),
                "captions": len(captions),
                "project_formal_protocol": bool(args.require_formal_protocol),
                "query_partition": {
                    "method": "image_index_modulo",
                    "index": int(args.query_partition_index),
                    "count": int(args.query_partition_count),
                    "query_rows": int(seen.sum().item()),
                },
                "world_size": world_size,
                "scoring": {
                    "contract": LIKELIHOOD_SCORING_CONTRACT,
                    "dual_stream_attention_contract": attention_contract,
                    "query_stream_diagonal": False,
                    "content_stream_diagonal": (
                        attention_contract == "xlnet_content_diagonal"
                    ),
                    "backend": str(args.scoring_backend),
                    "conditional_candidate_score": "mean_token_loglikelihood",
                    "primary_candidate_score": (
                        "language_prior_debiased_mean_token_loglikelihood"
                    ),
                    "language_prior_alpha": LANGUAGE_PRIOR_ALPHA,
                    "language_prior_estimator": LANGUAGE_PRIOR_ESTIMATOR,
                    "calibration_deferred_until_partition_merge": True,
                },
                "completed_at": utc_now(),
            }
            atomic_write_text(
                args.output_dir / "partition_summary.json",
                json.dumps(
                    partition_summary, ensure_ascii=False, indent=2, sort_keys=True
                )
                + "\n",
            )
            manifest = read_json(manifest_path)
            manifest["partition_complete"] = True
            manifest["completed_at"] = partition_summary["completed_at"]
            atomic_write_text(
                manifest_path,
                json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
                + "\n",
            )
            print(json.dumps(partition_summary, ensure_ascii=False, indent=2, sort_keys=True))
        else:
            matrix = torch.empty((len(records), len(captions)), dtype=torch.float32)
            seen = torch.zeros(len(records), dtype=torch.bool)
            for shard_rank in range(world_size):
                path = args.output_dir / "shards" / (
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
                if indices.unique().numel() != indices.numel():
                    raise ValueError("duplicate retrieval rows within a shard")
                if indices.numel() and (
                    int(indices.min()) < 0 or int(indices.max()) >= len(records)
                ):
                    raise ValueError("retrieval shard contains out-of-range rows")
                if bool(seen[indices].any()):
                    raise ValueError("duplicate retrieval rows across shards")
                matrix[indices] = payload[
                    "conditional_mean_token_loglikelihood"
                ].float()
                seen[indices] = True
            if not bool(seen.all()):
                raise RuntimeError("cross-dataset retrieval matrix is incomplete")
            calibrated, text_log_prior = language_prior_debiased_scores(matrix)
            summary = retrieval_metrics(calibrated, caption_counts)
            summary.update(
                {
                    "schema": "selfless_cross_dataset_retrieval_summary_v3",
                    "task": asset_manifest["task"],
                    "split": "karpathy_test",
                    "checkpoint": str(source.path),
                    "checkpoint_step": checkpoint_step,
                    "runtime_hashing_enabled": False,
                    "project_formal_protocol": bool(args.require_formal_protocol),
                    "formal_target_images": formal_images,
                    "formal_target_captions": formal_captions,
                    "complete_formal_target": (
                        len(records) == formal_images and len(captions) == formal_captions
                    ),
                    "coco_five_fold_1k_average": False,
                    "scoring": {
                        "contract": LIKELIHOOD_SCORING_CONTRACT,
                        "dual_stream_attention_contract": attention_contract,
                        "query_stream_diagonal": False,
                        "content_stream_diagonal": (
                            attention_contract == "xlnet_content_diagonal"
                        ),
                        "backend": str(args.scoring_backend),
                        "conditional_candidate_score": "mean_token_loglikelihood",
                        "primary_candidate_score": (
                            "language_prior_debiased_mean_token_loglikelihood"
                        ),
                        "language_prior_alpha": LANGUAGE_PRIOR_ALPHA,
                        "language_prior_estimator": LANGUAGE_PRIOR_ESTIMATOR,
                        "language_prior_image_count": len(records),
                        "language_prior_uses_labels": False,
                        "text_to_image_ranking_invariant_to_correction": True,
                    },
                }
            )
            atomic_write_text(
                args.output_dir / "summary.json",
                json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            )
            atomic_torch_save(
                args.output_dir / "score_matrix.pt",
                {
                    "prior_debiased_loglikelihood": calibrated,
                    "text_log_prior": text_log_prior,
                    "img_ids": torch.tensor([record.img_id for record in records]),
                    "caption_to_image": torch.arange(len(records)).repeat_interleave(
                        torch.tensor(caption_counts, dtype=torch.long)
                    ),
                    "scoring_contract": LIKELIHOOD_SCORING_CONTRACT,
                    "dual_stream_attention_contract": attention_contract,
                    "language_prior_alpha": LANGUAGE_PRIOR_ALPHA,
                    "language_prior_estimator": LANGUAGE_PRIOR_ESTIMATOR,
                    "runtime_hashing_enabled": False,
                },
            )
            manifest = read_json(manifest_path)
            manifest["complete"] = True
            manifest["complete_formal_target"] = summary["complete_formal_target"]
            manifest["completed_at"] = utc_now()
            atomic_write_text(
                manifest_path,
                json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            )
            print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    barrier(device)


if __name__ == "__main__":
    main()
