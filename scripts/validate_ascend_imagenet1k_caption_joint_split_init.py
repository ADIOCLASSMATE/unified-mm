#!/usr/bin/env python3
"""Validate Qwen-text + class-adapter joint-training initialization."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import torch
from omegaconf import OmegaConf

from scripts.validate_ascend_imagenet1k_caption_joint_sweep import (
    CAPTION_SHA256,
    GLOBAL_BATCH,
    MAX_STEPS,
    STEPS_PER_EPOCH,
    SYNTHETIC_CAPTIONS_PER_IMAGE,
    T2I_PROMPTS_PER_IMAGE,
    TRAIN_IMAGES,
    TRAIN_SAMPLES_PER_EPOCH,
    VALIDATION_IMAGES,
    validate_synthetic_assets,
)
from utils.selfless_training_runtime import validate_wsd_contract


QWEN_ROOT = Path("public/models/Qwen--Qwen3-0.6B-Base")
QWEN_WEIGHTS_SHA256 = (
    "cd2a512003e2f9f3cd3c32a9c3573f820bb28c940f73c57b1ddaa983d9223eba"
)
QWEN_CONFIG_SHA256 = (
    "504a6b58c4271583724e66584b6b7698aea18450209df6b2f7582df0e89cee59"
)
ADAPTER_PATH = Path(
    "output/selfless-flow-imagenet1k-class-ascend64-b1024-800ep/"
    "image_flow_adapter-final.pt"
)
ADAPTER_SHA256 = (
    "3a20383c73070a9e1db607d5b1b211acc1f51c37ad05ea41c215d19902a15d4c"
)
EXPECTED_ADAPTER_NUMEL = {
    "image_flow_head": 164_072_976,
    "image_flow_condition_proj": 1_049_600,
    "image_token_embedder": 17_408,
    "special_token_embeddings": 4_096,
}
EXPECTED_SPECIAL_TOKEN_IDS = {
    "mask": 151_669,
    "boi": 151_670,
    "eoi": 151_671,
    "image_mask": 151_672,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    return path


def require_hash(path: Path, expected: str, label: str) -> str:
    actual = sha256_file(require_file(path, label))
    if actual != expected:
        raise RuntimeError(
            f"{label} SHA256 mismatch: expected={expected}, actual={actual}, path={path}"
        )
    return actual


def _require_equal(actual, expected, label: str) -> None:
    if actual != expected:
        raise RuntimeError(f"config {label} mismatch: {actual!r} != {expected!r}")


def validate_config(config, *, world_size: int) -> dict[str, object]:
    validate_wsd_contract(config)
    params = config.dataset.params
    required = {
        "model_path": (str(config.model.model_path), str(QWEN_ROOT)),
        "pretrained_image_flow_adapter": (
            str(config.model.pretrained_image_flow_adapter),
            str(ADAPTER_PATH),
        ),
        "conditioning_mode": (str(params.conditioning_mode), "caption"),
        "caption_include_original": (bool(params.caption_include_original), False),
        "caption_sequence_modes": (
            list(params.caption_sequence_modes),
            ["t2i", "i2t"],
        ),
        "caption_manifest_sha256": (
            str(params.caption_manifest_sha256),
            CAPTION_SHA256,
        ),
        "lambda_image": (float(config.model.lambda_image), 1.0),
        "batch_size": (int(config.training.batch_size), 16),
        "total_batch_size": (int(config.training.total_batch_size), GLOBAL_BATCH),
        "samples_per_epoch": (
            int(config.training.samples_per_epoch),
            TRAIN_SAMPLES_PER_EPOCH,
        ),
        "optimizer_steps_per_epoch": (
            int(config.training.optimizer_steps_per_epoch),
            STEPS_PER_EPOCH,
        ),
        "max_train_steps": (int(config.training.max_train_steps), MAX_STEPS),
        "warmup_steps": (int(config.lr_scheduler.params.warmup_steps), 1_202),
        "decay_steps": (int(config.lr_scheduler.params.decay_steps), 3_606),
        "save_every": (int(config.experiment.save_every), 1_202),
        "val_every": (int(config.experiment.val_every), 1_202),
        "validation_seed": (int(config.experiment.validation_seed), 424_242),
        "split_seed": (int(params.split_seed), 42),
        "val_samples_per_class": (int(params.val_samples_per_class), 50),
        "validation_overlap_train": (bool(params.validation_overlap_train), False),
        "mixed_precision": (str(config.training.mixed_precision).lower(), "bf16"),
        "gradient_accumulation_dtype": (
            str(config.training.gradient_accumulation_dtype).lower(),
            "fp32",
        ),
        "from_scratch": (bool(config.training.from_scratch), False),
        "use_ema": (bool(config.training.use_ema), True),
    }
    for label, (actual, expected) in required.items():
        _require_equal(actual, expected, label)

    denominator = int(config.training.batch_size) * int(world_size)
    if GLOBAL_BATCH % denominator:
        raise RuntimeError("global batch is not divisible by rank batch * world size")
    gradient_accumulation = GLOBAL_BATCH // denominator
    if gradient_accumulation != 4:
        raise RuntimeError(
            f"16-NPU split-init training requires GA=4, got {gradient_accumulation}"
        )
    return {
        "world_size": int(world_size),
        "microbatch_per_rank": int(config.training.batch_size),
        "gradient_accumulation_steps": gradient_accumulation,
        "global_batch": GLOBAL_BATCH,
        "samples_per_epoch": TRAIN_SAMPLES_PER_EPOCH,
        "optimizer_steps_per_epoch": STEPS_PER_EPOCH,
        "max_optimizer_steps": MAX_STEPS,
        "validation_every_steps": 1_202,
        "task_modes": ["t2i", "i2t"],
        "train_images": TRAIN_IMAGES - VALIDATION_IMAGES,
        "validation_images": VALIDATION_IMAGES,
        "synthetic_captions_per_image": SYNTHETIC_CAPTIONS_PER_IMAGE,
        "t2i_prompts_per_image": T2I_PROMPTS_PER_IMAGE,
    }


def validate_adapter(path: Path) -> dict[str, object]:
    require_hash(path, ADAPTER_SHA256, "class-trained image-flow adapter")
    state = torch.load(path, map_location="cpu", weights_only=True)
    required = set(EXPECTED_ADAPTER_NUMEL) | {
        "special_token_ids",
        "special_token_embeddings",
    }
    missing = required - set(state)
    if missing:
        raise RuntimeError(f"adapter is missing {sorted(missing)}")
    actual_numel = {
        name: sum(int(tensor.numel()) for tensor in state[name].values())
        for name in EXPECTED_ADAPTER_NUMEL
    }
    if actual_numel != EXPECTED_ADAPTER_NUMEL:
        raise RuntimeError(
            f"adapter parameter inventory mismatch: {actual_numel} != "
            f"{EXPECTED_ADAPTER_NUMEL}"
        )
    token_ids = {
        str(name): int(token_id)
        for name, token_id in state["special_token_ids"].items()
    }
    if token_ids != EXPECTED_SPECIAL_TOKEN_IDS:
        raise RuntimeError(
            f"adapter special-token ids mismatch: {token_ids} != "
            f"{EXPECTED_SPECIAL_TOKEN_IDS}"
        )
    return {
        "path": str(path),
        "sha256": ADAPTER_SHA256,
        "parameter_numel": actual_numel,
        "special_token_ids": token_ids,
        "source": "completed ImageNet-1K class-conditioned FP32 EMA adapter export",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--backbone_lr", type=float, required=True)
    parser.add_argument("--flow_lr", type=float, required=True)
    parser.add_argument("--lambda_text", type=float, required=True)
    parser.add_argument(
        "--stop_after_steps", type=int, choices=(1_202, 2_404, 4_808, 12_020), required=True
    )
    parser.add_argument("--world_size", type=int, default=16)
    parser.add_argument("--require_npu_count", type=int, default=None)
    parser.add_argument("--require_hccl_intra_roce", action="store_true")
    parser.add_argument("--config_only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 1e-6 <= args.backbone_lr <= 2e-5:
        raise ValueError(f"backbone_lr outside conservative range: {args.backbone_lr}")
    if not 1e-5 <= args.flow_lr <= 8e-5:
        raise ValueError(f"flow_lr outside adapter-alignment range: {args.flow_lr}")
    if not 0.005 <= args.lambda_text <= 0.4:
        raise ValueError(f"lambda_text outside supported range: {args.lambda_text}")
    config = OmegaConf.load(require_file(args.config, "split-init config"))
    report: dict[str, object] = {
        "schema": "selfless_caption_t2i_split_initialization_preflight_v1",
        "status": "ok",
        "config": str(args.config),
        "candidate": {
            "backbone_lr": args.backbone_lr,
            "special_token_lr": args.backbone_lr,
            "flow_lr": args.flow_lr,
            "projector_lr": args.flow_lr,
            "lambda_text": args.lambda_text,
            "lambda_image": 1.0,
            "stop_after_steps": args.stop_after_steps,
        },
        "training": validate_config(config, world_size=args.world_size),
    }
    if not args.config_only:
        report["text_initialization"] = {
            "path": str(QWEN_ROOT),
            "weights_sha256": require_hash(
                QWEN_ROOT / "model.safetensors",
                QWEN_WEIGHTS_SHA256,
                "Qwen text weights",
            ),
            "config_sha256": require_hash(
                QWEN_ROOT / "config.json",
                QWEN_CONFIG_SHA256,
                "Qwen text config",
            ),
            "preserved_scopes": [
                "transformer layers",
                "final norm",
                "ordinary vocabulary embeddings",
                "tied LM head",
            ],
        }
        for filename in ("tokenizer.json", "tokenizer_config.json"):
            require_file(QWEN_ROOT / filename, f"Qwen tokenizer asset {filename}")
        report["adapter_initialization"] = validate_adapter(ADAPTER_PATH)
        report["synthetic_data"] = validate_synthetic_assets(
            Path(config.dataset.params.caption_jsonl),
            Path(config.dataset.params.synthetic_text_index_manifest),
            deep_scan=False,
        )
        require_file(Path(config.dataset.params.cache_path), "posterior cache")
        require_file(Path(config.dataset.params.manifest_jsonl), "ImageNet manifest")
    if args.require_npu_count is not None:
        import torch_npu  # noqa: F401

        available = bool(torch.npu.is_available())
        count = int(torch.npu.device_count())
        if not available or count != args.require_npu_count:
            raise RuntimeError(
                f"NPU contract failed: available={available}, count={count}, "
                f"expected={args.require_npu_count}"
            )
        report["hardware"] = {"npu_available": available, "npu_count": count}
    if args.require_hccl_intra_roce and os.environ.get("HCCL_INTRA_ROCE_ENABLE") != "1":
        raise RuntimeError("HCCL_INTRA_ROCE_ENABLE must equal 1")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
