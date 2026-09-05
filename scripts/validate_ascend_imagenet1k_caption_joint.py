#!/usr/bin/env python3
"""Validate the selected 16-NPU ImageNet-1K Caption + T2I configuration."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from omegaconf import OmegaConf

from utils.imagenet_synthetic_text_index import ImageNetSyntheticTextIndex
from utils.selfless_training_runtime import validate_wsd_contract


TRAIN_IMAGES = 1_281_167
VALIDATION_IMAGES = 50_000
TRAIN_SAMPLES_PER_EPOCH = 1_230_848
GLOBAL_BATCH = 1_024
STEPS_PER_EPOCH = 1_202
EPOCHS = 10
MAX_STEPS = 12_020
PUBLISHED_CAPTIONS_PER_IMAGE = 7
SYNTHETIC_CAPTIONS_PER_IMAGE = 6
T2I_PROMPTS_PER_IMAGE = 12
SELECTED_LR = 2e-5
SELECTED_LAMBDA_TEXT = 0.05
DEFAULT_GENERATION_STEPS = 10


def require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    return path


def validate_config(config, *, world_size: int) -> dict[str, object]:
    validate_wsd_contract(config)
    params = config.dataset.params
    required = {
        "model_path": (
            str(config.model.model_path),
            "output/selfless-flow-imagenet1k-class-ascend64-b1024-800ep/"
            "hf_model-final-ema",
        ),
        "conditioning_mode": (str(params.conditioning_mode), "caption"),
        "caption_include_original": (
            bool(params.caption_include_original),
            False,
        ),
        "caption_sequence_modes": (
            list(params.caption_sequence_modes),
            ["t2i", "i2t"],
        ),
        "caption_jsonl": (
            str(params.caption_jsonl),
            "public/datasets/imagenet1k_synthetic_v1/captions/"
            "imagenet1k_train_7captions.jsonl",
        ),
        "synthetic_text_index_manifest": (
            str(params.synthetic_text_index_manifest),
            "public/datasets/imagenet1k_synthetic_v1/indexed/train/manifest.json",
        ),
        "lambda_text": (float(config.model.lambda_text), SELECTED_LAMBDA_TEXT),
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
        "num_train_epochs": (int(config.training.num_train_epochs), EPOCHS),
        "max_train_steps": (int(config.training.max_train_steps), MAX_STEPS),
        "stop_after_steps": (int(config.training.stop_after_steps), MAX_STEPS),
        "warmup_steps": (int(config.lr_scheduler.params.warmup_steps), 1_202),
        "decay_steps": (int(config.lr_scheduler.params.decay_steps), 3_606),
        "save_every": (int(config.experiment.save_every), 2_404),
        "checkpoints_total_limit": (int(config.experiment.checkpoints_total_limit), 3),
        "checkpoint_milestone_every": (
            int(config.experiment.checkpoint_milestone_every),
            100 * STEPS_PER_EPOCH,
        ),
        "save_ema_eval_every": (
            int(config.experiment.save_ema_eval_every),
            20 * STEPS_PER_EPOCH,
        ),
        "save_model_with_ema_eval": (
            bool(config.experiment.save_model_with_ema_eval),
            True,
        ),
        "val_every": (int(config.experiment.val_every), 2_404),
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
    if accumulation != 4:
        raise RuntimeError(
            "16-NPU training requires gradient accumulation 4, "
            f"got {accumulation}"
        )
    if bool(config.training.from_scratch):
        raise RuntimeError("joint training must load the completed EMA initialization")
    if not bool(config.training.use_ema):
        raise RuntimeError("joint training requires EMA")
    if (
        int(config.experiment.log_grad_norm_every)
        % int(config.experiment.log_every)
    ):
        raise RuntimeError("gradient norm interval must align with log interval")
    return {
        "world_size": int(world_size),
        "microbatch_per_rank": int(config.training.batch_size),
        "gradient_accumulation_steps": accumulation,
        "global_batch": GLOBAL_BATCH,
        "samples_per_epoch": TRAIN_SAMPLES_PER_EPOCH,
        "optimizer_steps_per_epoch": STEPS_PER_EPOCH,
        "epochs": EPOCHS,
        "max_optimizer_steps": MAX_STEPS,
        "wsd_epochs": {"warmup": 1, "stable": 6, "decay": 3},
        "task_modes": ["t2i", "i2t"],
        "training_images_available": TRAIN_IMAGES,
        "validation_images": VALIDATION_IMAGES,
        "caption_source": "six_synthetic_only",
        "t2i_source": "twelve_synthetic_prompts",
    }


def _require_equal(actual, expected, label: str) -> None:
    if actual != expected:
        raise RuntimeError(f"{label} mismatch: {actual!r} != {expected!r}")


def _validate_caption_row(row: dict, expected_index: int) -> None:
    _require_equal(
        int(row.get("manifest_index", -1)), expected_index, "caption row index"
    )
    captions = row.get("captions")
    if not isinstance(captions, list):
        raise RuntimeError(f"caption list missing at row {expected_index}")
    _require_equal(
        len(captions), PUBLISHED_CAPTIONS_PER_IMAGE, "published caption count"
    )
    source_counts: dict[str, int] = {}
    for caption in captions:
        source = str(caption.get("source", ""))
        source_counts[source] = source_counts.get(source, 0) + 1
        if not str(caption.get("text", "")).strip():
            raise RuntimeError(f"empty caption text at row {expected_index}")
    _require_equal(source_counts.get("original", 0), 1, "original caption count")
    _require_equal(source_counts.get("local_qwen", 0), 3, "Qwen caption count")
    _require_equal(
        source_counts.get("api_distilled", 0), 3, "MiniMax caption count"
    )


def validate_synthetic_assets(
    caption_path: Path,
    index_manifest_path: Path,
    *,
    deep_scan: bool,
) -> dict[str, object]:
    caption_path = require_file(caption_path, "published synthetic caption JSONL")
    dataset_root = caption_path.parent.parent
    dataset_manifest_path = require_file(
        dataset_root / "dataset_manifest.json", "synthetic dataset manifest"
    )
    dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    expected_dataset_fields = {
        "schema": "imagenet1k_synthetic_dataset_v1",
        "dataset": "imagenet1k",
    }
    for key, expected in expected_dataset_fields.items():
        _require_equal(dataset_manifest.get(key), expected, f"dataset manifest {key}")
    _require_equal(
        dataset_manifest["captions"]["split_records"],
        {"train": TRAIN_IMAGES},
        "caption split records",
    )
    _require_equal(
        int(dataset_manifest["captions"]["captions_per_image"]),
        PUBLISHED_CAPTIONS_PER_IMAGE,
        "captions per image",
    )
    _require_equal(
        dataset_manifest["t2i"]["split_records"],
        {"train": TRAIN_IMAGES, "val": VALIDATION_IMAGES},
        "T2I split records",
    )
    _require_equal(
        int(dataset_manifest["t2i"]["prompts_per_image"]),
        T2I_PROMPTS_PER_IMAGE,
        "T2I prompts per image",
    )
    _require_equal(
        dataset_manifest["alignment"]["status"], "passed", "alignment status"
    )
    _require_equal(
        int(dataset_manifest["alignment"]["train_caption_without_t2i"]),
        0,
        "train caption without T2I",
    )
    _require_equal(
        int(dataset_manifest["alignment"]["train_t2i_without_caption"]),
        0,
        "train T2I without caption",
    )

    caption_manifest_path = require_file(
        dataset_root / dataset_manifest["captions"]["manifest"],
        "caption manifest",
    )
    caption_manifest = json.loads(
        caption_manifest_path.read_text(encoding="utf-8")
    )
    _require_equal(int(caption_manifest["records"]), TRAIN_IMAGES, "caption rows")
    _require_equal(
        int(caption_manifest["captions_per_image"]),
        PUBLISHED_CAPTIONS_PER_IMAGE,
        "published captions per image",
    )

    alignment_audit = json.loads(
        require_file(
            dataset_root / "alignment" / "audit_report.json", "alignment audit"
        ).read_text(encoding="utf-8")
    )
    _require_equal(alignment_audit.get("status"), "passed", "alignment audit")
    _require_equal(
        int(alignment_audit["alignment"]["train_rows"]),
        TRAIN_IMAGES,
        "aligned train rows",
    )
    _require_equal(
        int(alignment_audit["alignment"]["val_rows"]),
        VALIDATION_IMAGES,
        "aligned val rows",
    )

    index_manifest_path = require_file(
        index_manifest_path, "seekable synthetic text index"
    )
    index = ImageNetSyntheticTextIndex(index_manifest_path)
    try:
        _require_equal(index.row_count, TRAIN_IMAGES, "indexed train rows")
        sample_rows = []
        for row_index in (0, TRAIN_IMAGES // 2, TRAIN_IMAGES - 1):
            caption_row = index.read_caption(row_index)
            _validate_caption_row(caption_row, row_index)
            t2i_row = index.read_t2i(row_index)
            prompts = t2i_row.get("model_result", {}).get("prompts")
            if not isinstance(prompts, list):
                raise RuntimeError(f"T2I prompts missing at index row {row_index}")
            _require_equal(
                len(prompts), T2I_PROMPTS_PER_IMAGE, "indexed T2I prompt count"
            )
            expected_image_id = f"train/{caption_row['id']}"
            _require_equal(
                t2i_row.get("image_id"), expected_image_id, "indexed image identity"
            )
            sample_rows.append(
                {
                    "row": row_index,
                    "image_id": expected_image_id,
                    "published_captions": len(caption_row["captions"]),
                    "synthetic_captions_used": SYNTHETIC_CAPTIONS_PER_IMAGE,
                    "t2i_prompts": len(prompts),
                }
            )
    finally:
        index.close()

    checked_rows = 0
    if deep_scan:
        with caption_path.open(encoding="utf-8") as handle:
            for checked_rows, line in enumerate(handle, start=1):
                _validate_caption_row(json.loads(line), checked_rows - 1)
        _require_equal(checked_rows, TRAIN_IMAGES, "deep-scanned caption rows")

    t2i_manifest_path = require_file(
        dataset_root / dataset_manifest["t2i"]["manifest"], "T2I manifest"
    )
    t2i_manifest = json.loads(t2i_manifest_path.read_text(encoding="utf-8"))
    train_shards = [
        shard for shard in t2i_manifest["shards"] if shard["split"] == "train"
    ]
    val_shards = [
        shard for shard in t2i_manifest["shards"] if shard["split"] == "val"
    ]
    _require_equal(len(train_shards), 64, "T2I train shard count")
    _require_equal(len(val_shards), 8, "T2I val shard count")
    _require_equal(
        sum(int(shard["records"]) for shard in train_shards),
        TRAIN_IMAGES,
        "T2I train rows",
    )
    _require_equal(
        sum(int(shard["records"]) for shard in val_shards),
        VALIDATION_IMAGES,
        "T2I val rows",
    )
    return {
        "dataset_root": str(dataset_root),
        "dataset_manifest": str(dataset_manifest_path),
        "caption_path": str(caption_path),
        "caption_rows": TRAIN_IMAGES,
        "published_captions_per_image": PUBLISHED_CAPTIONS_PER_IMAGE,
        "synthetic_captions_used_per_image": SYNTHETIC_CAPTIONS_PER_IMAGE,
        "t2i_train_rows": TRAIN_IMAGES,
        "t2i_val_rows": VALIDATION_IMAGES,
        "t2i_prompts_per_image": T2I_PROMPTS_PER_IMAGE,
        "seek_index": str(index_manifest_path),
        "sample_rows": sample_rows,
        "deep_scan": bool(deep_scan),
        "checked_caption_rows": checked_rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=(
            "configs/selfless/"
            "imagenet1k_caption_joint_10ep_ascend16_b1024.yaml"
        ),
    )
    parser.add_argument("--world_size", type=int, default=16)
    parser.add_argument("--require_npu_count", type=int, default=None)
    parser.add_argument("--require_hccl_intra_roce", action="store_true")
    parser.add_argument("--config_only", action="store_true")
    parser.add_argument("--deep_caption_scan", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = require_file(Path(args.config), "joint training config")
    config = OmegaConf.load(config_path)
    report: dict[str, object] = {
        "status": "ok",
        "config": str(config_path),
        "selection": {
            "backbone_lr": SELECTED_LR,
            "special_token_lr": SELECTED_LR,
            "flow_lr": SELECTED_LR,
            "projector_lr": SELECTED_LR,
            "stop_after_steps": MAX_STEPS,
            "lambda_text": SELECTED_LAMBDA_TEXT,
            "lambda_image": 1.0,
        },
        "training": validate_config(config, world_size=args.world_size),
    }
    if not args.config_only:
        model_root = Path(config.model.model_path)
        report["initialization"] = {
            "path": str(model_root),
            "weights": str(
                require_file(
                    model_root / "model.safetensors",
                    "completed ImageNet-1K EMA weights",
                )
            ),
            "config": str(
                require_file(
                    model_root / "config.json",
                    "completed ImageNet-1K EMA config",
                )
            ),
        }
        for filename in (
            "tokenizer.json",
            "tokenizer_config.json",
            "generation_config.json",
        ):
            require_file(model_root / filename, f"EMA model asset {filename}")
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
