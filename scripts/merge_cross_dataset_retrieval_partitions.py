#!/usr/bin/env python3
"""Merge independently evaluated COCO/Flickr query partitions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import torch

from scripts.evaluate_cross_dataset_retrieval import (
    SUPPORTED_TASKS,
    load_records,
    retrieval_metrics,
)
from scripts.evaluate_imagenet_pretraining_native import atomic_torch_save
from scripts.evaluate_multimodal_likelihood_benchmarks import (
    LIKELIHOOD_SCORING_CONTRACT,
    atomic_write_text,
    utc_now,
)
from scripts.language_prior_calibration import (
    LANGUAGE_PRIOR_ALPHA,
    LANGUAGE_PRIOR_ESTIMATOR,
    language_prior_debiased_scores,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset_root", type=Path, required=True)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Prefix size for an end-to-end smoke merge; zero requires full formal data.",
    )
    parser.add_argument(
        "--partition_root",
        type=Path,
        action="append",
        required=True,
        help="Complete partition output root; repeat once per partition.",
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def merge_score_shards(
    partition_specs: Sequence[tuple[Path, int, int, int]],
    *,
    images: int,
    captions: int,
    dual_stream_attention_contract: str | None = None,
) -> torch.Tensor:
    """Load partition rank shards and require exact, duplicate-free coverage."""

    matrix = torch.empty((images, captions), dtype=torch.float32)
    seen = torch.zeros(images, dtype=torch.bool)
    for root, partition_index, partition_count, world_size in partition_specs:
        expected = torch.zeros(images, dtype=torch.bool)
        expected[partition_index::partition_count] = True
        partition_seen = torch.zeros(images, dtype=torch.bool)
        for shard_rank in range(world_size):
            shard_path = root / "shards" / (
                f"rank-{shard_rank:05d}-of-{world_size:05d}.pt"
            )
            payload = torch.load(
                str(shard_path), map_location="cpu", weights_only=True
            )
            if payload.get("runtime_hashing_enabled", True) is not False:
                raise ValueError(f"partition shard violates no-hash contract: {shard_path}")
            indices = payload["query_indices"].long()
            if payload.get("scoring_contract") != LIKELIHOOD_SCORING_CONTRACT:
                raise ValueError(f"partition shard scoring mismatch: {shard_path}")
            if (
                dual_stream_attention_contract is not None
                and payload.get("dual_stream_attention_contract")
                != dual_stream_attention_contract
            ):
                raise ValueError(
                    f"partition shard attention contract mismatch: {shard_path}"
                )
            scores = payload["conditional_mean_token_loglikelihood"].float()
            if indices.ndim != 1 or scores.shape != (indices.numel(), captions):
                raise ValueError(f"invalid partition shard shape: {shard_path}")
            if indices.unique().numel() != indices.numel():
                raise ValueError(f"duplicate rows within partition shard: {shard_path}")
            if indices.numel() and (
                int(indices.min()) < 0 or int(indices.max()) >= images
            ):
                raise ValueError(f"partition shard has out-of-range rows: {shard_path}")
            if bool(partition_seen[indices].any()) or bool(seen[indices].any()):
                raise ValueError(f"duplicate retrieval rows: {shard_path}")
            if indices.numel() and bool((indices % partition_count != partition_index).any()):
                raise ValueError(f"partition shard contains rows from another partition: {shard_path}")
            matrix[indices] = scores
            partition_seen[indices] = True
            seen[indices] = True
        if not torch.equal(partition_seen, expected):
            raise ValueError(f"incomplete query partition: {root}")
    if not bool(seen.all()):
        missing = torch.nonzero(~seen).flatten().tolist()[:8]
        raise ValueError(f"merged retrieval matrix is incomplete; missing rows {missing}")
    return matrix


def main() -> None:
    args = parse_args()
    asset_manifest, records = load_records(args.asset_root)
    task = str(asset_manifest["task"])
    formal_images, formal_captions = SUPPORTED_TASKS[task]
    if int(args.limit) < 0:
        raise ValueError("--limit must be non-negative")
    if int(args.limit) > 0:
        records = records[: int(args.limit)]
    captions = [caption for record in records for caption in record.captions]
    caption_counts = [len(record.captions) for record in records]
    images = len(records)
    caption_total = len(captions)
    roots = [root.resolve() for root in args.partition_root]
    if len(set(roots)) != len(roots):
        raise ValueError("partition roots must be unique")
    manifests = [read_json(root / "manifest.json") for root in roots]
    summaries = [read_json(root / "partition_summary.json") for root in roots]
    first = manifests[0]
    partition_count = int(first["query_partition"]["count"])
    if partition_count <= 1 or len(roots) != partition_count:
        raise ValueError("provide exactly one root for every independent partition")

    expected_indices = set(range(partition_count))
    actual_indices: set[int] = set()
    specs: list[tuple[Path, int, int, int]] = []
    checkpoint = str(first["checkpoint"])
    checkpoint_step = int(first["checkpoint_step"])
    for root, manifest, summary in zip(roots, manifests, summaries):
        if manifest.get("schema") != "selfless_cross_dataset_retrieval_evaluation_v3":
            raise ValueError(f"obsolete partition manifest protocol: {root}")
        if summary.get("schema") != "selfless_cross_dataset_retrieval_partition_v3":
            raise ValueError(f"obsolete partition summary protocol: {root}")
        if manifest.get("project_formal_protocol") is not True:
            raise ValueError(f"partition manifest is not a formal run: {root}")
        if summary.get("project_formal_protocol") is not True:
            raise ValueError(f"partition summary is not a formal run: {root}")
        partition = manifest.get("query_partition", {})
        index = int(partition.get("index", -1))
        count = int(partition.get("count", -1))
        if index in actual_indices or index not in expected_indices or count != partition_count:
            raise ValueError(f"invalid or duplicate query partition: {root}")
        actual_indices.add(index)
        if manifest.get("partition_complete") is not True:
            raise ValueError(f"partition manifest is incomplete: {root}")
        if manifest.get("complete") is not False:
            raise ValueError(f"partition root unexpectedly claims a full result: {root}")
        if summary.get("complete") is not True:
            raise ValueError(f"partition summary is incomplete: {root}")
        for payload in (manifest, summary):
            if payload.get("runtime_hashing_enabled", True) is not False:
                raise ValueError(f"partition violates the no-hash contract: {root}")
            if str(payload.get("checkpoint")) != checkpoint:
                raise ValueError(f"partition checkpoint mismatch: {root}")
            if int(payload.get("checkpoint_step", -1)) != checkpoint_step:
                raise ValueError(f"partition checkpoint step mismatch: {root}")
            if str(payload.get("task")) != task:
                raise ValueError(f"partition task mismatch: {root}")
            if int(payload.get("images", -1)) != images:
                raise ValueError(f"partition image cardinality mismatch: {root}")
            if int(payload.get("captions", -1)) != caption_total:
                raise ValueError(f"partition caption cardinality mismatch: {root}")
        if manifest.get("scoring") != first.get("scoring"):
            raise ValueError(f"partition scoring contract mismatch: {root}")
        specs.append((root, index, count, int(manifest["world_size"])))
    if actual_indices != expected_indices:
        raise ValueError("query partition set is incomplete")

    output_dir = args.output_dir.resolve()
    if output_dir in roots:
        raise ValueError("merged output must differ from partition roots")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"merged output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix = merge_score_shards(
        specs,
        images=images,
        captions=caption_total,
        dual_stream_attention_contract=str(
            first["scoring"]["dual_stream_attention_contract"]
        ),
    )
    calibrated, text_log_prior = language_prior_debiased_scores(matrix)
    summary = retrieval_metrics(calibrated, caption_counts)
    scoring = dict(first["scoring"])
    scoring.update(
        {
            "calibration_deferred_until_partition_merge": False,
            "language_prior_image_count": images,
            "text_to_image_ranking_invariant_to_correction": True,
        }
    )
    summary.update(
        {
            "schema": "selfless_cross_dataset_retrieval_summary_v3",
            "task": task,
            "split": "karpathy_test",
            "checkpoint": checkpoint,
            "checkpoint_step": checkpoint_step,
            "runtime_hashing_enabled": False,
            "formal_target_images": formal_images,
            "formal_target_captions": formal_captions,
            "complete_formal_target": (
                images == formal_images and caption_total == formal_captions
            ),
            "coco_five_fold_1k_average": False,
            "project_formal_protocol": True,
            "scoring": scoring,
        }
    )
    manifest = {
        key: value
        for key, value in first.items()
        if key
        not in {
            "complete",
            "completed_at",
            "created_at",
            "partition_complete",
            "query_partition",
            "world_size",
        }
    }
    manifest.update(
        {
            "schema": "selfless_cross_dataset_retrieval_evaluation_v3",
            "complete": True,
            "complete_formal_target": summary["complete_formal_target"],
            "runtime_hashing_enabled": False,
            "project_formal_protocol": True,
            "created_at": min(str(value["created_at"]) for value in manifests),
            "completed_at": utc_now(),
            "world_size": sum(int(value["world_size"]) for value in manifests),
            "scoring": scoring,
            "query_partitioning": {
                "method": "image_index_modulo",
                "partitions": partition_count,
                "independent_workers": [
                    {
                        "index": int(value["query_partition"]["index"]),
                        "world_size": int(value["world_size"]),
                    }
                    for value in sorted(
                        manifests, key=lambda item: int(item["query_partition"]["index"])
                    )
                ],
            },
        }
    )
    atomic_write_text(
        output_dir / "summary.json",
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    atomic_torch_save(
        output_dir / "score_matrix.pt",
        {
            "prior_debiased_loglikelihood": calibrated,
            "text_log_prior": text_log_prior,
            "img_ids": torch.tensor([record.img_id for record in records]),
            "caption_to_image": torch.arange(images).repeat_interleave(
                torch.tensor(caption_counts, dtype=torch.long)
            ),
            "scoring_contract": LIKELIHOOD_SCORING_CONTRACT,
            "language_prior_alpha": LANGUAGE_PRIOR_ALPHA,
            "language_prior_estimator": LANGUAGE_PRIOR_ESTIMATOR,
            "runtime_hashing_enabled": False,
        },
    )
    atomic_write_text(
        output_dir / "manifest.json",
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
