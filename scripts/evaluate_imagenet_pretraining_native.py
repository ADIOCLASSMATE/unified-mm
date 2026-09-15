#!/usr/bin/env python3
"""Evaluate calibrated zero-shot ImageNet-1K classification on all 50K val images.

The 1,000 candidate texts use OpenAI CLIP's curated ImageNet class names and
one frozen template.  The Selfless model itself supplies the length-normalized
conditional token likelihoods; no external CLIP model scores the checkpoint.
The formal score subtracts each class text's log-mean-exp marginal over all
50,000 validation images with a fixed language-prior alpha of one.
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
from typing import Any, Sequence

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import torch
from torch.nn.attention.flex_attention import create_block_mask
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.evaluation.multimodal_likelihood import (  # noqa: E402
    CandidateScore,
    LIKELIHOOD_SCORING_CONTRACT,
    LikelihoodExample,
    PosteriorCache,
    atomic_write_text,
    barrier,
    encode_candidate,
    initialize_device,
    score_candidate_requests,
    utc_now,
)
from utils.evaluation.calibration import (  # noqa: E402
    LANGUAGE_PRIOR_ALPHA,
    LANGUAGE_PRIOR_ESTIMATOR,
    language_prior_debiased_scores,
)
from models.modeling_model.image_position_utils import (  # noqa: E402
    build_row_col_position_ids,
)
from models.modeling_model.modeling_selfless_flow import (  # noqa: E402
    SelflessStaticCache,
)
from utils.evaluation.model_contracts import scoring_contract

from utils.evaluation_model_source import (  # noqa: E402
    add_model_source_argument,
    configure_model_source,
    load_model_source_weights,
    model_source_from_args,
)
from utils.utils import load_model_tokenizer  # noqa: E402


from utils.evaluation.native_understanding import (  # noqa: F401 -- compatibility exports
    DEFAULT_CONFIG,
    DEFAULT_MANIFEST,
    DEFAULT_CLASSES,
    DEFAULT_CLASSNAMES,
    DEFAULT_CACHE,
    RETRIEVAL_PROMPT,
    CLASS_TEXT_TEMPLATE,
    CLASSIFICATION_TASK,
    OPENAI_CLIP_CLASSNAME_COMMIT,
    OPENAI_CLIP_CLASSNAME_NOTEBOOK,
    CLASS_IDENTITY_CORRECTIONS,
    ImageNetRecord,
    CachedTokenizer,
    read_jsonl,
    load_imagenet_records,
    load_openai_clip_class_names,
    score_text_candidates,
    cross_sigma_attention_mask,
    clone_prefix_cache,
    score_text_candidates_cached_prefix,
    score_candidates_with_backend,
    classification_metrics,
    atomic_torch_save,
    evaluate_classification,
    validate_args,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    add_model_source_argument(parser)
    parser.add_argument("--image_manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--classes", type=Path, default=DEFAULT_CLASSES)
    parser.add_argument("--class_names", type=Path, default=DEFAULT_CLASSNAMES)
    parser.add_argument("--cache_shard_dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--batch_size_per_rank", type=int, default=32)
    parser.add_argument("--request_chunk_size", type=int, default=128)
    parser.add_argument("--lm_head_chunk_tokens", type=int, default=256)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--require_formal_protocol", action="store_true")
    parser.add_argument("--seed", type=int, default=424242)
    parser.add_argument("--progress_every", type=int, default=10)
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


def main() -> None:
    args = parse_args()
    source = model_source_from_args(args)
    validate_args(args, source)
    checkpoint_step = source.global_step
    rank, world_size, _, device = initialize_device(args.device)
    config = OmegaConf.load(args.config)
    configure_model_source(config, source)
    objective = str(config.model.get("training_objective", "selfless_dual_stream"))
    if objective not in {"selfless_dual_stream", "showo2_full_image_flow"}:
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
    if attention_contract not in {"selfless_strict", "xlnet_content_diagonal", "showo2_omni_attention"}:
        raise ValueError(f"unknown attention contract: {attention_contract}")
    if attention_contract == "showo2_omni_attention":
        image_sigma_order = "sequential"
        args.scoring_backend = "repeated_full_sequence"
    if config.model.get("architecture_variant") == "selfless_joint_dit":
        image_sigma_order = "joint"
    if image_sigma_order not in {"random", "sequential", "joint"}:
        raise ValueError(f"unknown image sigma order: {image_sigma_order}")

    records = load_imagenet_records(args.image_manifest, args.classes)
    class_names, class_name_provenance = load_openai_clip_class_names(
        args.class_names
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
            "schema": "selfless_imagenet1k_zeroshot_classification_evaluation_v2",
            "complete": False,
            "created_at": utc_now(),
            "runtime_hashing_enabled": False,
            "checkpoint": str(source.path),
            "checkpoint_step": checkpoint_step,
            "weight_source": source.kind,
            "config": str(args.config.resolve()),
            "task": CLASSIFICATION_TASK,
            "world_size": world_size,
            "dataset_split": "imagenet_val",
            "dataset_records": len(records),
            "formal_target_records": 50_000,
            "project_formal_protocol": bool(args.require_formal_protocol),
            "classes": len(class_names),
            "training_overlap_allowed": False,
            "class_text": {
                "template": CLASS_TEXT_TEMPLATE,
                "prompt": RETRIEVAL_PROMPT,
                "class_names": str(args.class_names.resolve()),
                "provenance": class_name_provenance,
                "template_selection_used_imagenet_val_labels": False,
            },
            "scoring": {
                "model_contract": scoring_contract(attention_contract, "selfless_same_position_query_stream"),
                "contract": scoring_contract(attention_contract, LIKELIHOOD_SCORING_CONTRACT),
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
                "language_prior_image_set": "all_evaluation_images",
                "language_prior_uses_labels": False,
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

    summary = evaluate_classification(
        records=records,
        class_names=class_names,
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
    if rank == 0:
        if summary is None:
            raise RuntimeError("rank zero did not produce a classification summary")
        summary.update(
            {
                "checkpoint": str(source.path),
                "checkpoint_step": checkpoint_step,
                "weight_source": source.kind,
                "runtime_hashing_enabled": False,
                "project_formal_protocol": bool(args.require_formal_protocol),
                "scoring": {
                    "contract": scoring_contract(attention_contract, LIKELIHOOD_SCORING_CONTRACT),
                    "dual_stream_attention_contract": attention_contract,
                    "backend": str(args.scoring_backend),
                    "conditional_candidate_score": "mean_token_loglikelihood",
                    "primary_candidate_score": (
                        "language_prior_debiased_mean_token_loglikelihood"
                    ),
                    "language_prior_alpha": LANGUAGE_PRIOR_ALPHA,
                    "language_prior_estimator": LANGUAGE_PRIOR_ESTIMATOR,
                },
            }
        )
        atomic_write_text(
            args.output_dir / "summary.json",
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["complete"] = True
        manifest["complete_formal_target"] = summary["complete_formal_target"]
        if bool(args.require_formal_protocol) and not bool(
            summary["complete_formal_target"]
        ):
            raise RuntimeError("formal ImageNet classification did not cover 50K images")
        manifest["completed_at"] = utc_now()
        manifest["reported_metrics"] = ["top_1_accuracy", "top_5_accuracy"]
        atomic_write_text(
            manifest_path,
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    barrier(device)


if __name__ == "__main__":
    main()
