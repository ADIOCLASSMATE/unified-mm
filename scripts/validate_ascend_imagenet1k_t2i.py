#!/usr/bin/env python3
"""Validate the formal 64-NPU ImageNet-1K T2I-only training variants."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from omegaconf import OmegaConf

from scripts.validate_ascend_imagenet1k_caption_joint import (
    require_file,
    validate_synthetic_assets,
)
from utils.selfless_training_runtime import validate_wsd_contract


TRAIN_IMAGES = 1_281_167
VALIDATION_IMAGES = 50_000
TRAIN_SAMPLES_PER_EPOCH = 1_230_848
GLOBAL_BATCH = 1_024
STEPS_PER_EPOCH = 1_202
EPOCH_CONTRACTS = {
    80: {"warmup": 8, "stable": 48, "decay": 24},
    400: {"warmup": 40, "stable": 240, "decay": 120},
}
SELECTED_LR = 2e-5
T2I_PROMPTS_PER_IMAGE = 12
DEFAULT_GENERATION_STEPS = 10

VARIANT_CONTRACTS = {
    "baseline": {
        "project": "selfless-flow-imagenet1k-t2i-baseline-ascend64-b1024-80ep",
        "model_path": (
            "output/selfless-flow-imagenet1k-class-ascend64-b1024-800ep/"
            "hf_model-final-ema"
        ),
        "architecture_variant": "selfless_contextual",
        "image_sigma_order": "random",
        "generation_strategy": "spatial_halton",
    },
    "positionwise_head": {
        "project": (
            "selfless-flow-imagenet1k-t2i-positionwise-head-"
            "ascend64-b1024-80ep"
        ),
        "model_path": (
            "output/selfless-flow-imagenet1k-class-ascend64-b1024-800ep-"
            "positionwise-head/hf_model-final-ema"
        ),
        "architecture_variant": "positionwise_selfless",
        "image_sigma_order": "random",
        "generation_strategy": "spatial_halton",
    },
    "seq_sigma": {
        "project": "selfless-flow-imagenet1k-t2i-seq-sigma-ascend64-b1024-80ep",
        "model_path": (
            "output/selfless-flow-imagenet1k-class-ascend64-b1024-800ep-"
            "seq-sigma/hf_model-final-ema"
        ),
        "architecture_variant": "selfless_contextual",
        "image_sigma_order": "sequential",
        "generation_strategy": "sequential",
    },
}


def validate_config(
    config,
    *,
    world_size: int,
    variant: str = "baseline",
) -> dict[str, object]:
    try:
        contract = VARIANT_CONTRACTS[variant]
    except KeyError as exc:
        raise RuntimeError(f"unsupported T2I variant: {variant!r}") from exc
    epochs = int(config.training.num_train_epochs)
    try:
        epoch_contract = EPOCH_CONTRACTS[epochs]
    except KeyError as exc:
        raise RuntimeError(
            f"unsupported T2I epoch contract: {epochs}; "
            f"expected one of {sorted(EPOCH_CONTRACTS)}"
        ) from exc
    expected_project = str(contract["project"]).replace(
        "-80ep", f"-{epochs}ep"
    )
    max_steps = STEPS_PER_EPOCH * epochs
    validate_wsd_contract(config)
    params = config.dataset.params
    architecture_variant = str(
        config.model.get("architecture_variant", "selfless_contextual")
    )
    image_sigma_order = str(params.get("image_sigma_order", "random"))
    required = {
        "experiment_project": (str(config.experiment.project), expected_project),
        "model_path": (str(config.model.model_path), contract["model_path"]),
        "architecture_variant": (
            architecture_variant,
            contract["architecture_variant"],
        ),
        "image_sigma_order": (image_sigma_order, contract["image_sigma_order"]),
        "validation_order_strategies": (
            list(config.experiment.validation_single_stream_order_strategies),
            [contract["generation_strategy"]],
        ),
        "evaluation_strategy": (
            str(config.evaluation.strategies),
            contract["generation_strategy"],
        ),
        "evaluation_checkpoint": (
            str(config.evaluation.checkpoint),
            f"output/{expected_project}/hf_model-final-ema",
        ),
        "backbone_attention_output_gate": (
            str(config.model.backbone_attention_output_gate),
            "none",
        ),
        "conditioning_mode": (str(params.conditioning_mode), "caption"),
        "caption_include_original": (bool(params.caption_include_original), False),
        "caption_sequence_modes": (list(params.caption_sequence_modes), ["t2i"]),
        "caption_jsonl": (
            str(params.caption_jsonl),
            "public/datasets/imagenet1k_synthetic_v1/captions/"
            "imagenet1k_train_7captions.jsonl",
        ),
        "synthetic_text_index_manifest": (
            str(params.synthetic_text_index_manifest),
            "public/datasets/imagenet1k_synthetic_v1/indexed/train/manifest.json",
        ),
        "lambda_text": (float(config.model.lambda_text), 0.0),
        "lambda_image": (float(config.model.lambda_image), 1.0),
        "pretrained_image_flow_adapter": (
            str(config.model.pretrained_image_flow_adapter).lower(),
            "none",
        ),
        "model_sampling_steps": (
            int(config.model.image_flow_num_sampling_steps),
            DEFAULT_GENERATION_STEPS,
        ),
        "evaluation_sampling_steps": (
            int(config.evaluation.sampling_steps),
            DEFAULT_GENERATION_STEPS,
        ),
        "learning_rate": (
            float(config.optimizer.params.learning_rate),
            SELECTED_LR,
        ),
        "backbone_learning_rate": (
            float(config.optimizer.params.backbone_learning_rate),
            SELECTED_LR,
        ),
        "special_token_learning_rate": (
            float(config.optimizer.params.special_token_learning_rate),
            SELECTED_LR,
        ),
        "projector_learning_rate": (
            float(config.optimizer.params.projector_learning_rate),
            SELECTED_LR,
        ),
        "flow_learning_rate": (
            float(config.optimizer.params.flow_learning_rate),
            SELECTED_LR,
        ),
        "total_batch_size": (int(config.training.total_batch_size), GLOBAL_BATCH),
        "batch_size": (int(config.training.batch_size), 16),
        "samples_per_epoch": (
            int(config.training.samples_per_epoch),
            TRAIN_SAMPLES_PER_EPOCH,
        ),
        "optimizer_steps_per_epoch": (
            int(config.training.optimizer_steps_per_epoch),
            STEPS_PER_EPOCH,
        ),
        "num_train_epochs": (int(config.training.num_train_epochs), epochs),
        "max_train_steps": (int(config.training.max_train_steps), max_steps),
        "stop_after_steps": (int(config.training.stop_after_steps), max_steps),
        "warmup_steps": (
            int(config.lr_scheduler.params.warmup_steps),
            epoch_contract["warmup"] * STEPS_PER_EPOCH,
        ),
        "decay_steps": (
            int(config.lr_scheduler.params.decay_steps),
            epoch_contract["decay"] * STEPS_PER_EPOCH,
        ),
        "save_every": (int(config.experiment.save_every), 10 * STEPS_PER_EPOCH),
        "val_every": (int(config.experiment.val_every), 10 * STEPS_PER_EPOCH),
        "log_every": (int(config.experiment.log_every), 50),
        "log_grad_norm_every": (
            int(config.experiment.log_grad_norm_every),
            1_200,
        ),
        "max_seq_length": (int(params.max_seq_length), 512),
        "pad_to_length": (int(params.pad_to_length), 512),
        "training_split": (str(params.expected_split), "train"),
        "training_records": (int(params.expected_records), TRAIN_IMAGES),
        "validation_split": (str(params.validation.expected_split), "val"),
        "validation_records": (
            int(params.validation.expected_records),
            VALIDATION_IMAGES,
        ),
        "trainable_scope": (str(config.training.trainable_scope), "full"),
        "ema_decay": (float(config.training.ema_decay), 0.999),
        "mixed_precision": (str(config.training.mixed_precision).lower(), "bf16"),
        "gradient_accumulation_dtype": (
            str(config.training.gradient_accumulation_dtype).lower(),
            "fp32",
        ),
    }
    for label, (actual, expected) in required.items():
        if actual != expected:
            raise RuntimeError(f"config {label} mismatch: {actual!r} != {expected!r}")

    denominator = int(config.training.batch_size) * int(world_size)
    if GLOBAL_BATCH % denominator:
        raise RuntimeError(
            "global batch is not divisible by per-rank batch * world size"
        )
    accumulation = GLOBAL_BATCH // denominator
    if accumulation != 1:
        raise RuntimeError(
            f"64-NPU T2I training requires gradient accumulation 1, got {accumulation}"
        )
    if bool(config.training.from_scratch):
        raise RuntimeError("T2I training must load the completed baseline EMA")
    if not bool(config.training.use_ema):
        raise RuntimeError("T2I training requires EMA")
    if int(config.experiment.log_grad_norm_every) % int(config.experiment.log_every):
        raise RuntimeError("gradient norm interval must align with log interval")

    return {
        "variant": variant,
        "architecture_variant": architecture_variant,
        "image_sigma_order": image_sigma_order,
        "generation_strategy": contract["generation_strategy"],
        "world_size": int(world_size),
        "microbatch_per_rank": int(config.training.batch_size),
        "gradient_accumulation_steps": accumulation,
        "global_batch": GLOBAL_BATCH,
        "samples_per_epoch": TRAIN_SAMPLES_PER_EPOCH,
        "optimizer_steps_per_epoch": STEPS_PER_EPOCH,
        "epochs": epochs,
        "max_optimizer_steps": max_steps,
        "wsd_epochs": dict(epoch_contract),
        "task_modes": ["t2i"],
        "training_images_available": TRAIN_IMAGES,
        "validation_images": VALIDATION_IMAGES,
        "t2i_source": "twelve_synthetic_prompts",
        "prompt_schedule": (
            "deterministic_per_image_random_offset_without_replacement; "
            "one full 12-prompt cycle before reuse"
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=(
            "configs/selfless/"
            "imagenet1k_t2i_baseline_80ep_ascend_64npu_bs1024.yaml"
        ),
    )
    parser.add_argument("--world_size", type=int, default=64)
    parser.add_argument(
        "--variant",
        choices=tuple(VARIANT_CONTRACTS),
        default="baseline",
    )
    parser.add_argument("--require_npu_count", type=int, default=None)
    parser.add_argument("--require_hccl_intra_roce", action="store_true")
    parser.add_argument("--config_only", action="store_true")
    parser.add_argument("--deep_caption_scan", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = require_file(Path(args.config), "T2I training config")
    config = OmegaConf.load(config_path)
    report: dict[str, object] = {
        "status": "ok",
        "config": str(config_path),
        "training": validate_config(
            config,
            world_size=args.world_size,
            variant=args.variant,
        ),
    }
    if not args.config_only:
        model_root = Path(config.model.model_path)
        report["initialization"] = {
            "path": str(model_root),
            "weights": str(
                require_file(
                    model_root / "model.safetensors",
                    f"completed ImageNet-1K {args.variant} EMA weights",
                )
            ),
            "config": str(
                require_file(
                    model_root / "config.json",
                    f"completed ImageNet-1K {args.variant} EMA config",
                )
            ),
        }
        for filename in (
            "tokenizer.json",
            "tokenizer_config.json",
            "generation_config.json",
        ):
            require_file(
                model_root / filename,
                f"{args.variant} EMA model asset {filename}",
            )
        report["synthetic_data"] = validate_synthetic_assets(
            Path(config.dataset.params.caption_jsonl),
            Path(config.dataset.params.synthetic_text_index_manifest),
            deep_scan=args.deep_caption_scan,
        )
        require_file(
            Path(config.dataset.params.cache_path),
            "ImageNet-1K posterior cache",
        )
        require_file(
            Path(config.dataset.params.manifest_jsonl),
            "ImageNet-1K manifest",
        )
    if args.require_npu_count is not None:
        import torch
        import torch_npu  # noqa: F401

        available = bool(torch.npu.is_available())
        count = int(torch.npu.device_count())
        if not available or count != int(args.require_npu_count):
            raise RuntimeError(
                f"NPU contract failed: available={available}, count={count}, "
                f"expected={args.require_npu_count}"
            )
        report["hardware"] = {
            "npu_available": available,
            "npu_count": count,
        }
    if (
        args.require_hccl_intra_roce
        and os.environ.get("HCCL_INTRA_ROCE_ENABLE") != "1"
    ):
        raise RuntimeError("HCCL_INTRA_ROCE_ENABLE must equal 1")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
