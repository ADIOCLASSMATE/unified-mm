#!/usr/bin/env python3
"""Validate the formal 64-NPU ImageNet-1K 800-epoch training contract."""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter
from pathlib import Path

import torch
from omegaconf import OmegaConf

from utils.dataset_imagenet_flow_cache import (
    POSTERIOR_CACHE_FORMAT,
    POSTERIOR_STATS_LAYOUT,
)
from utils.selfless_training_runtime import validate_wsd_contract

TRAIN_IMAGES = 1_281_167
TRAIN_SAMPLES_PER_EPOCH = 1_281_024
GLOBAL_BATCH = 1_024
STEPS_PER_EPOCH = 1_251
EPOCHS = 800
MAX_STEPS = 1_000_800
VALIDATION_IMAGES = 50_000
DEFAULT_GENERATION_STEPS = 10


def require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    return path


def validate_config(config, *, world_size: int) -> dict[str, object]:
    validate_wsd_contract(config)
    params = config.dataset.params
    architecture_variant = str(
        config.model.get("architecture_variant", "selfless_contextual")
    ).lower()
    image_sigma_order = str(params.get("image_sigma_order", "random")).lower()
    project_by_variant = {
        ("selfless_contextual", "random"): (
            "selfless-flow-imagenet1k-class-ascend64-b1024-800ep"
        ),
        ("selfless_contextual", "sequential"): (
            "selfless-flow-imagenet1k-class-ascend64-b1024-800ep-seq-sigma"
        ),
        ("positionwise_selfless", "random"): (
            "selfless-flow-imagenet1k-class-ascend64-b1024-800ep-"
            "positionwise-head"
        ),
    }
    variant_key = (architecture_variant, image_sigma_order)
    if variant_key not in project_by_variant:
        raise RuntimeError(
            "unsupported formal architecture/sigma pair: "
            f"architecture_variant={architecture_variant!r}, "
            f"image_sigma_order={image_sigma_order!r}"
        )
    expected_project = project_by_variant[variant_key]
    if image_sigma_order == "sequential":
        expected_generation_strategy = "sequential"
    else:
        expected_generation_strategy = "spatial_halton"
    required = {
        "project": (
            str(config.experiment.project),
            expected_project,
        ),
        "conditioning_mode": (str(params.conditioning_mode), "class"),
        "training_split": (str(params.expected_split), "train"),
        "training_records": (int(params.expected_records), TRAIN_IMAGES),
        "validation_split": (str(params.validation.expected_split), "val"),
        "validation_records": (
            int(params.validation.expected_records),
            VALIDATION_IMAGES,
        ),
        "total_batch_size": (int(config.training.total_batch_size), GLOBAL_BATCH),
        "samples_per_epoch": (int(config.training.samples_per_epoch), TRAIN_SAMPLES_PER_EPOCH),
        "optimizer_steps_per_epoch": (int(config.training.optimizer_steps_per_epoch), STEPS_PER_EPOCH),
        "num_train_epochs": (int(config.training.num_train_epochs), EPOCHS),
        "max_train_steps": (int(config.training.max_train_steps), MAX_STEPS),
        "warmup_steps": (int(config.lr_scheduler.params.warmup_steps), 6_255),
        "decay_steps": (int(config.lr_scheduler.params.decay_steps), 250_200),
        "save_every": (int(config.experiment.save_every), 12_510),
        "save_ema_eval_every": (
            int(config.experiment.save_ema_eval_every),
            20 * STEPS_PER_EPOCH,
        ),
        "save_model_with_ema_eval": (
            bool(config.experiment.save_model_with_ema_eval),
            True,
        ),
        "val_every": (int(config.experiment.val_every), 12_510),
        "validation_image_every": (int(config.experiment.validation_image_every), 12_510),
        "checkpoints_total_limit": (int(config.experiment.checkpoints_total_limit), 3),
        "checkpoint_milestone_every": (
            int(config.experiment.checkpoint_milestone_every),
            125_100,
        ),
        "learning_rate": (float(config.optimizer.params.learning_rate), 4e-5),
        "backbone_learning_rate": (float(config.optimizer.params.backbone_learning_rate), 30e-5),
        "projector_learning_rate": (float(config.optimizer.params.projector_learning_rate), 4e-5),
        "flow_learning_rate": (float(config.optimizer.params.flow_learning_rate), 4e-5),
        "special_token_learning_rate": (float(config.optimizer.params.special_token_learning_rate), 30e-5),
        "ema_decay": (float(config.training.ema_decay), 0.9999),
        "ema_update_after_step": (int(config.training.ema_update_after_step), 0),
        "ema_save_hf_model": (bool(config.training.ema_save_hf_model), True),
        "ema_eval_dtype": (str(config.experiment.ema_eval_dtype).lower(), "bf16"),
        "save_image_flow_adapter": (
            bool(config.experiment.save_image_flow_adapter),
            False,
        ),
        "save_final_image_flow_adapter": (
            bool(config.experiment.save_final_image_flow_adapter),
            True,
        ),
        "model_sampling_steps": (
            int(config.model.image_flow_num_sampling_steps),
            DEFAULT_GENERATION_STEPS,
        ),
        "evaluation_samples": (int(config.evaluation.samples), VALIDATION_IMAGES),
        "evaluation_sampling_steps": (
            int(config.evaluation.sampling_steps),
            DEFAULT_GENERATION_STEPS,
        ),
        "evaluation_strategies": (
            str(config.evaluation.strategies),
            expected_generation_strategy,
        ),
        "validation_single_stream_order_strategies": (
            list(config.experiment.validation_single_stream_order_strategies),
            [expected_generation_strategy],
        ),
    }
    for label, (actual, expected) in required.items():
        if actual != expected:
            raise RuntimeError(f"config {label} mismatch: {actual!r} != {expected!r}")
    if str(config.model.model_path) != "public/models/Qwen--Qwen3-0.6B-Base":
        raise RuntimeError("formal pretraining must initialize from Qwen3-0.6B-Base")
    expected_evaluation_checkpoint = f"output/{expected_project}/hf_model-final-ema"
    if str(config.evaluation.checkpoint) != expected_evaluation_checkpoint:
        raise RuntimeError(
            "formal evaluation checkpoint mismatch: "
            f"{config.evaluation.checkpoint!r} != "
            f"{expected_evaluation_checkpoint!r}"
        )
    if bool(config.training.from_scratch) or not bool(config.training.use_ema):
        raise RuntimeError("formal pretraining requires pretrained Qwen initialization and EMA")
    if bool(config.training.use_gradient_checkpointing):
        raise RuntimeError("gradient checkpointing is outside the measured BS1024 contract")
    if str(config.training.mixed_precision).lower() != "bf16":
        raise RuntimeError("formal pretraining requires BF16")
    if str(config.training.gradient_accumulation_dtype).lower() != "fp32":
        raise RuntimeError("formal pretraining requires FP32 gradient accumulation")
    microbatch = int(config.training.batch_size)
    denominator = microbatch * int(world_size)
    if GLOBAL_BATCH % denominator or GLOBAL_BATCH // denominator != 1:
        raise RuntimeError("formal 64-NPU contract requires gradient accumulation 1")
    decay = float(config.training.ema_decay)
    half_life_steps = math.log(0.5) / math.log(decay)
    return {
        "world_size": int(world_size),
        "microbatch_per_rank": microbatch,
        "gradient_accumulation_steps": 1,
        "global_batch": GLOBAL_BATCH,
        "train_images": TRAIN_IMAGES,
        "samples_per_epoch": TRAIN_SAMPLES_PER_EPOCH,
        "dropped_samples_per_epoch": TRAIN_IMAGES - TRAIN_SAMPLES_PER_EPOCH,
        "optimizer_steps_per_epoch": STEPS_PER_EPOCH,
        "epochs": EPOCHS,
        "max_optimizer_steps": MAX_STEPS,
        "architecture_variant": architecture_variant,
        "image_sigma_order": image_sigma_order,
        "wsd_epochs": {"warmup": 5, "stable": 595, "decay": 200},
        "ema_decay": decay,
        "ema_half_life_steps": half_life_steps,
        "ema_half_life_epochs": half_life_steps / STEPS_PER_EPOCH,
    }


def validate_manifest(path: Path) -> dict[str, object]:
    counts: Counter[str] = Counter()
    expected_id = 1
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            image_id = int(row["img_id"])
            if image_id != expected_id:
                raise RuntimeError(
                    f"manifest img_id mismatch at line {line_number}: "
                    f"{image_id} != {expected_id}"
                )
            synset = str(row["synset"])
            if not synset:
                raise RuntimeError(f"empty synset at manifest line {line_number}")
            counts[synset] += 1
            expected_id += 1
    rows = expected_id - 1
    if rows != TRAIN_IMAGES or len(counts) != 1_000:
        raise RuntimeError(
            f"ImageNet-1K manifest failed: rows={rows}, classes={len(counts)}"
        )
    return {
        "rows": rows,
        "classes": len(counts),
        "minimum_images_per_class": min(counts.values()),
        "maximum_images_per_class": max(counts.values()),
    }


def validate_cache(path: Path, *, deep_scan: bool) -> dict[str, object]:
    payload = torch.load(
        str(path), map_location="cpu", mmap=True, weights_only=True
    )
    stats = payload.get("posterior_stats")
    image_ids = payload.get("img_ids")
    metadata = payload.get("metadata", {})
    if not torch.is_tensor(stats) or tuple(stats.shape) != (TRAIN_IMAGES, 256, 32):
        raise RuntimeError(
            f"posterior_stats shape mismatch: {getattr(stats, 'shape', None)}"
        )
    if stats.dtype != torch.float16:
        raise RuntimeError(f"posterior_stats must be FP16, got {stats.dtype}")
    if not torch.is_tensor(image_ids) or image_ids.dtype != torch.int64:
        raise RuntimeError("cache img_ids must be an int64 tensor")
    expected_ids = torch.arange(1, TRAIN_IMAGES + 1, dtype=torch.int64)
    if not torch.equal(image_ids, expected_ids):
        raise RuntimeError("cache img_ids do not match the canonical manifest")
    expected_metadata = {
        "format": POSTERIOR_CACHE_FORMAT,
        "stats_layout": POSTERIOR_STATS_LAYOUT,
        "stats_are_scaled": True,
        "num_images": TRAIN_IMAGES,
        "image_tokens_per_img": 256,
        "image_latent_dim": 16,
        "posterior_stats_dim": 32,
        "storage_dtype": "float16",
        "scaling_factor": 0.2325,
    }
    for field, expected in expected_metadata.items():
        if metadata.get(field) != expected:
            raise RuntimeError(
                f"cache metadata {field} mismatch: "
                f"{metadata.get(field)!r} != {expected!r}"
            )
    if deep_scan:
        for start in range(0, TRAIN_IMAGES, 512):
            chunk = stats[start : start + 512]
            if not bool(torch.isfinite(chunk).all()):
                raise RuntimeError(f"posterior cache contains NaN/Inf at row {start}")
            if bool((chunk[..., 16:] < 0).any()):
                raise RuntimeError(f"posterior cache contains negative std at row {start}")
    return {
        "shape": list(stats.shape),
        "dtype": str(stats.dtype),
        "deep_scan": bool(deep_scan),
    }


def validate_real_stats(path: Path) -> dict[str, object]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    schema = payload.get("schema")
    if schema not in {
        "imagenet_inception_feature_moments_v1",
        "imagenet_val_inception_feature_moments_v2",
    }:
        raise RuntimeError("unsupported ImageNet validation real-stat schema")
    stats = payload.get("stats", {})
    feature_sum = stats.get("sum")
    outer_sum = stats.get("outer_sum")
    if int(stats.get("count", -1)) != VALIDATION_IMAGES:
        raise RuntimeError("ImageNet validation real-stat count must be 50000")
    if not torch.is_tensor(feature_sum) or tuple(feature_sum.shape) != (2048,):
        raise RuntimeError("real-stat feature sum shape mismatch")
    if not torch.is_tensor(outer_sum) or tuple(outer_sum.shape) != (2048, 2048):
        raise RuntimeError("real-stat outer sum shape mismatch")
    if feature_sum.dtype != torch.float32 or outer_sum.dtype != torch.float32:
        raise RuntimeError("Ascend real-stat tensors must be FP32")
    if not bool(torch.isfinite(feature_sum).all()) or not bool(torch.isfinite(outer_sum).all()):
        raise RuntimeError("real-stat tensors contain NaN/Inf")
    metadata = payload.get("metadata", {})
    source = metadata.get("source", {})
    feature = metadata.get("feature", {})
    split = source.get("split")
    if split not in {"val", "validation"}:
        raise RuntimeError(
            f"real-stat metadata split mismatch: {split!r} is not ImageNet val"
        )
    expected = {
        "classes": (source.get("classes"), 1_000),
        "samples_per_class": (source.get("samples_per_class"), 50),
        "feature": (feature.get("feature"), 2048),
        "accumulation_dtype": (feature.get("accumulation_dtype"), "torch.float32"),
    }
    if schema == "imagenet_val_inception_feature_moments_v2":
        expected.update(
            {
                "dataset": (source.get("dataset"), "ImageNet-1K"),
                "records": (source.get("records"), VALIDATION_IMAGES),
            }
        )
    for label, (actual, required) in expected.items():
        if actual != required:
            raise RuntimeError(
                f"real-stat metadata {label} mismatch: {actual!r} != {required!r}"
            )
    return {
        "schema": schema,
        "count": VALIDATION_IMAGES,
        "classes": 1_000,
        "samples_per_class": 50,
        "feature": 2048,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=(
            "configs/selfless/"
            "imagenet1k_class_pretrain_800ep_ascend_64npu_bs1024.yaml"
        ),
    )
    parser.add_argument("--world_size", type=int, default=64)
    parser.add_argument("--require_npu_count", type=int, default=None)
    parser.add_argument("--require_hccl_intra_roce", action="store_true")
    parser.add_argument("--config_only", action="store_true")
    parser.add_argument("--deep_cache_scan", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = require_file(Path(args.config), "training config")
    config = OmegaConf.load(config_path)
    report: dict[str, object] = {
        "status": "ok",
        "config": str(config_path),
        "training": validate_config(config, world_size=args.world_size),
    }
    if not args.config_only:
        model_root = Path(config.model.model_path)
        vae_root = Path(config.experiment.validation_vae_module_root)
        manifest_path = Path(config.dataset.params.manifest_jsonl)
        cache_path = Path(config.dataset.params.cache_path)
        inception_path = Path(config.evaluation.inception_weights_path)
        real_stats_path = Path(config.evaluation.real_stats_path)
        report["assets"] = {
            "qwen_weights": str(
                require_file(
                    model_root / "model.safetensors",
                    "Qwen3-0.6B-Base weights",
                )
            ),
            "vae_module": str(
                require_file(vae_root / "models" / "vae.py", "MAR KL16 VAE module")
            ),
            "vae_checkpoint": str(
                require_file(
                    Path(config.experiment.validation_vae_path),
                    "MAR KL16 checkpoint",
                )
            ),
            "manifest": str(
                require_file(manifest_path, "canonical ImageNet-1K train manifest")
            ),
            "inception_weights": str(
                require_file(inception_path, "torch-fidelity Inception weights")
            ),
        }
        report["manifest"] = validate_manifest(manifest_path)
        report["cache"] = validate_cache(
            require_file(cache_path, "ImageNet-1K KL16 posterior cache"),
            deep_scan=args.deep_cache_scan,
        )
        report["real_stats"] = validate_real_stats(
            require_file(real_stats_path, "ImageNet validation FID real stats")
        )
        for filename in ("config.json", "tokenizer.json", "tokenizer_config.json"):
            require_file(model_root / filename, f"offline Qwen asset {filename}")
        require_file(
            Path(config.dataset.params.synset_mapping_path),
            "ImageNet synset mapping",
        )
    if args.require_npu_count is not None:
        import torch_npu  # noqa: F401

        available = bool(torch.npu.is_available())
        count = int(torch.npu.device_count())
        if not available or count != int(args.require_npu_count):
            raise RuntimeError(
                f"NPU contract failed: available={available}, count={count}, "
                f"expected={args.require_npu_count}"
            )
        report["hardware"] = {"npu_available": available, "npu_count": count}
    if (
        args.require_hccl_intra_roce
        and os.environ.get("HCCL_INTRA_ROCE_ENABLE") != "1"
    ):
        raise RuntimeError("HCCL_INTRA_ROCE_ENABLE must equal 1")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
