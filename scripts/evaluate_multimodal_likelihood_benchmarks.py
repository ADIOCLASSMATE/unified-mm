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

from utils.evaluation.model_contracts import scoring_contract
from utils.evaluation.aro import ARO_TASKS, CONDITIONAL_SCORE, ARO_SCORE_VARIANT

from utils.evaluation_model_source import (  # noqa: E402
    add_model_source_argument,
    configure_model_source,
    load_model_source_weights,
    model_source_from_args,
    resolve_evaluation_model_source,
)
from utils.utils import get_selfless_mask, load_model_tokenizer  # noqa: E402


from utils.evaluation.multimodal_likelihood import (  # noqa: F401 -- compatibility exports
    DEFAULT_CONFIG,
    DEFAULT_ASSET_ROOT,
    ASSET_SCHEMA,
    TASK_SCHEMA,
    CACHE_FORMAT,
    CACHE_LAYOUT,
    LIKELIHOOD_SCORING_CONTRACT,
    IMAGE_ORDER_MC_CONTRACT,
    IMAGE_ORDER_MC_SEED_STRIDE,
    LANGUAGE_PRIOR_ALPHA,
    LANGUAGE_PRIOR_ESTIMATOR,
    LANGUAGE_PRIOR_NULL_IMAGE_COUNT,
    LANGUAGE_PRIOR_NULL_IMAGE_IDS,
    LANGUAGE_PRIOR_NULL_IMAGE_SEEDS,
    LANGUAGE_PRIOR_NULL_PIXEL_SPACE,
    LANGUAGE_PRIOR_NULL_CLAMP,
    LANGUAGE_PRIOR_NULL_STORAGE,
    DEBIASED_SCORE,
    DEFAULT_TASKS,
    FORMAL_TASK_RECORDS,
    LikelihoodExample,
    CandidateRequest,
    CandidateScore,
    utc_now,
    readable_file_identity,
    atomic_write_text,
    jsonl_text,
    initialize_device,
    barrier,
    read_checkpoint_step,
    validate_language_prior_contract,
    validate_args,
    language_prior_null_image_ids,
    arithmetic_seed,
    image_order_mc_seed,
    PosteriorCache,
    read_jsonl,
    load_examples,
    build_image_sigma,
    encode_candidate,
    encode_candidate_mc,
    build_attention_masks,
    score_candidate_requests,
    argmax,
    logmeanexp,
    build_prediction_rows,
    chunk_examples_by_candidate_count,
    mean,
    classification_metrics,
    pairwise_ranking_metrics,
    category_metrics,
    binary_metrics,
    mmbench_circular_metrics,
    winoground_metrics,
    paired_image_ranking_metrics,
    whatsup_metrics,
    summarize_task,
    shard_path,
    validate_prediction_shard,
    evaluate_task,
    read_jsonl_predictions,
)


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
        choices=("auto", "random", "sequential", "spatial_halton", "spatial_halton_shifted"),
        default="auto",
    )
    parser.add_argument("--progress_every", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = model_source_from_args(args)
    tasks, asset_manifest = validate_args(args, source)
    checkpoint_step = source.global_step
    rank, world_size, local_rank, device = initialize_device(args.device)
    config = OmegaConf.load(args.config)
    configure_model_source(config, source)
    objective = str(config.model.get("training_objective", "selfless_dual_stream"))
    if objective not in {"selfless_dual_stream", "showo2_full_image_flow"}:
        raise ValueError(
            "likelihood benchmark runner currently requires selfless_dual_stream, "
            f"got {objective!r}"
        )
    attention_contract = str(
        config.model.get("dual_stream_attention_contract", "selfless_strict")
    ).strip().lower()
    if attention_contract not in {"selfless_strict", "xlnet_content_diagonal", "showo2_omni_attention"}:
        raise ValueError(f"unknown dual-stream attention contract: {attention_contract}")
    configured_order = str(
        config.dataset.params.image.get("image_sigma_order", "random")
    ).strip().lower()
    image_sigma_order = (
        configured_order if args.image_sigma_order == "auto" else args.image_sigma_order
    )
    if attention_contract == "showo2_omni_attention":
        image_sigma_order = "sequential"
        args.mc = 1  # Exact AR score; no image-order Monte Carlo distribution.
    if image_sigma_order not in {"random", "sequential", "spatial_halton", "spatial_halton_shifted"}:
        raise ValueError(f"unknown image sigma order: {image_sigma_order}")
    if int(args.mc) > 1 and image_sigma_order not in {"random", "spatial_halton_shifted"}:
        raise ValueError(
            "--mc > 1 requires random or shifted Halton image_sigma_order; a fixed order has "
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
            "image_order_mc_contract": ("not_applicable_full_image_ar" if attention_contract == "showo2_omni_attention" else IMAGE_ORDER_MC_CONTRACT),
            "language_prior_alpha": LANGUAGE_PRIOR_ALPHA,
            "language_prior_estimator": LANGUAGE_PRIOR_ESTIMATOR,
            "language_prior_null_image_ids": list(null_image_ids),
            "language_prior_null_image_count": len(null_image_ids),
            "language_prior_uses_labels": False,
            "max_length": int(args.max_length),
            "seed": int(args.seed),
            "image_sigma_order": image_sigma_order,
            "image_order_distribution": ("halton_base2_base3_uniform_torus_shift_v1" if image_sigma_order == "spatial_halton_shifted" else "uniform_random_permutation" if image_sigma_order == "random" else "fixed"),
            "halton_shift_rng": ("torch_cpu_float64_uniform_2d_image_order_mc_seed" if image_sigma_order == "spatial_halton_shifted" else None),
            "dual_stream_attention_contract": attention_contract,
            "scoring_contract": scoring_contract(attention_contract, LIKELIHOOD_SCORING_CONTRACT),
            "query_stream_diagonal": False,
            "content_stream_diagonal": (
                attention_contract == "xlnet_content_diagonal"
            ),
            "primary_candidate_score": DEBIASED_SCORE,
            "reported_score_variant": ARO_SCORE_VARIANT,
            "task_primary_candidate_scores": {task: CONDITIONAL_SCORE for task in sorted(ARO_TASKS)},
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
                        "contract": scoring_contract(attention_contract, LIKELIHOOD_SCORING_CONTRACT),
                        "dual_stream_attention_contract": attention_contract,
                        "query_stream_diagonal": False,
                        "content_stream_diagonal": (
                            attention_contract == "xlnet_content_diagonal"
                        ),
                        "image_sigma_order": image_sigma_order,
                        "image_order_distribution": ("halton_base2_base3_uniform_torus_shift_v1" if image_sigma_order == "spatial_halton_shifted" else "uniform_random_permutation" if image_sigma_order == "random" else "fixed"),
                        "mc_samples": int(args.mc),
                        "mc_aggregation": "mean_loglikelihood",
                        "mc_common_random_numbers_across_candidates": True,
                        "image_order_mc_contract": ("not_applicable_full_image_ar" if attention_contract == "showo2_omni_attention" else IMAGE_ORDER_MC_CONTRACT),
                        "primary_candidate_score": DEBIASED_SCORE,
                        "reported_score_variant": ARO_SCORE_VARIANT,
                        "task_primary_candidate_scores": {task: CONDITIONAL_SCORE for task in sorted(ARO_TASKS)},
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
