#!/usr/bin/env python3
"""Validate assets and immutable training semantics for the Ascend run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import torch
from omegaconf import OmegaConf

EXPECTED_HASHES = {
    "qwen_weights": "cd2a512003e2f9f3cd3c32a9c3573f820bb28c940f73c57b1ddaa983d9223eba",
    "vae_module": "95e9d47d017817cd86858d78587786c931a9ba9596fe3eb6d6dce4136580112b",
    "vae_checkpoint": "34ce001bcfffb7af67ec8af1e683a30d7bd45760855ddc7deedc1330f2cfd38f",
    "membership": "6c6fc84e6ec9cb8b92421659d44abbb24d1cc34558213d9fb9db7e4ec44f1c3a",
    "split": "02e5c67c058f95bcca46c82f3c1fc81086f61dcec62ce25049843f09d930a5ba",
}
POSTERIOR_FORMAT = "imagenet_kl16_scaled_posterior_v1"
POSTERIOR_LAYOUT = "scaled_mean_then_scaled_std"


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


def load_membership(path: Path) -> tuple[dict[int, str], Counter]:
    rows: dict[int, str] = {}
    counts: Counter = Counter()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            image_id = int(row["img_id"])
            synset = str(row["synset"])
            if image_id in rows:
                raise ValueError(f"duplicate img_id={image_id} in {path}:{line_number}")
            rows[image_id] = synset
            counts[synset] += 1
    if len(rows) != 125_000 or len(counts) != 100 or set(counts.values()) != {1250}:
        raise RuntimeError(
            "membership contract failed: "
            f"rows={len(rows)}, classes={len(counts)}, counts={sorted(set(counts.values()))}"
        )
    return rows, counts


def validate_split(path: Path, membership: dict[int, str]) -> Counter:
    seen: set[int] = set()
    counts: Counter = Counter()
    class_counts: dict[str, Counter] = {}
    expected_index = {"train": 0, "validation": 0}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            image_id = int(row["img_id"])
            split = str(row["split"]).lower()
            if split == "val":
                split = "validation"
            if split not in expected_index:
                raise ValueError(f"unsupported split={split!r} in {path}:{line_number}")
            if image_id in seen or image_id not in membership:
                raise ValueError(
                    f"invalid or duplicate img_id={image_id} in {path}:{line_number}"
                )
            if str(row.get("synset")) != membership[image_id]:
                raise ValueError(f"synset mismatch for img_id={image_id}")
            if int(row.get("split_index", -1)) != expected_index[split]:
                raise ValueError(
                    f"non-contiguous split_index for {split} in {path}:{line_number}"
                )
            expected_index[split] += 1
            seen.add(image_id)
            counts[split] += 1
            class_counts.setdefault(membership[image_id], Counter())[split] += 1
    if seen != set(membership):
        raise RuntimeError(
            f"split does not cover membership: {len(seen)} != {len(membership)}"
        )
    if counts != Counter({"train": 115_000, "validation": 10_000}):
        raise RuntimeError(f"split counts mismatch: {dict(counts)}")
    expected_per_class = Counter({"train": 1150, "validation": 100})
    invalid = {
        key: value for key, value in class_counts.items() if value != expected_per_class
    }
    if invalid:
        raise RuntimeError(f"non-stratified split classes: {list(invalid.items())[:4]}")
    return counts


def validate_cache(
    path: Path, membership: dict[int, str], *, scan_values: bool
) -> dict:
    payload = torch.load(
        str(path),
        map_location="cpu",
        mmap=True,
        weights_only=True,
    )
    posterior_stats = payload.get("posterior_stats")
    image_ids = payload.get("img_ids")
    metadata = payload.get("metadata", {})
    if not torch.is_tensor(posterior_stats) or tuple(posterior_stats.shape) != (
        125_000,
        256,
        32,
    ):
        raise RuntimeError(
            f"posterior_stats shape mismatch: {getattr(posterior_stats, 'shape', None)}"
        )
    if posterior_stats.dtype != torch.float16:
        raise RuntimeError(f"posterior_stats dtype mismatch: {posterior_stats.dtype}")
    if not torch.is_tensor(image_ids) or image_ids.dtype != torch.int64:
        raise RuntimeError(
            f"img_ids dtype mismatch: {getattr(image_ids, 'dtype', None)}"
        )
    if tuple(image_ids.shape) != (125_000,):
        raise RuntimeError(f"img_ids shape mismatch: {tuple(image_ids.shape)}")
    if bool((image_ids[1:] <= image_ids[:-1]).any()):
        raise RuntimeError("cache img_ids must be unique and strictly increasing")
    manifest_ids = torch.tensor(sorted(membership), dtype=torch.int64)
    if not torch.equal(image_ids, manifest_ids):
        raise RuntimeError("cache img_ids do not exactly match membership")
    expected_metadata = {
        "format": POSTERIOR_FORMAT,
        "stats_layout": POSTERIOR_LAYOUT,
        "stats_are_scaled": True,
        "num_images": 125_000,
        "image_tokens_per_img": 256,
        "image_latent_dim": 16,
        "posterior_stats_dim": 32,
        "storage_dtype": "float16",
        "scaling_factor": 0.2325,
    }
    for field, expected in expected_metadata.items():
        if metadata.get(field) != expected:
            raise RuntimeError(
                f"cache metadata mismatch for {field}: {metadata.get(field)!r} != {expected!r}"
            )
    if metadata.get("vae_checkpoint_sha256") != EXPECTED_HASHES["vae_checkpoint"]:
        raise RuntimeError("cache was not encoded with the canonical KL16 checkpoint")
    if scan_values:
        for start in range(0, posterior_stats.shape[0], 512):
            chunk = posterior_stats[start : start + 512]
            if not bool(torch.isfinite(chunk).all()):
                raise RuntimeError(f"posterior cache contains NaN/Inf at row {start}")
            if bool((chunk[..., 16:] < 0).any()):
                raise RuntimeError(
                    f"posterior cache contains negative std at row {start}"
                )
    return metadata


def validate_config(config, world_size: int, split_counts: Counter) -> dict:
    params = config.dataset.params
    required_values = {
        "conditioning_mode": (str(params.conditioning_mode), "class"),
        "model_path": (
            str(config.model.model_path),
            "public/models/Qwen--Qwen3-0.6B-Base",
        ),
        "mixed_precision": (str(config.training.mixed_precision).lower(), "bf16"),
        "gradient_accumulation_dtype": (
            str(config.training.gradient_accumulation_dtype).lower(),
            "fp32",
        ),
        "max_seq_length": (int(params.max_seq_length), 320),
        "image_tokens_per_img": (int(config.model.image_tokens_per_img), 256),
        "image_latent_dim": (int(config.model.image_latent_dim), 16),
        "image_flow_batch_mul": (int(config.model.image_flow_batch_mul), 4),
        "total_batch_size": (int(config.training.total_batch_size), 512),
        "max_train_steps": (int(config.training.max_train_steps), 17_920),
        "warmup_steps": (int(config.lr_scheduler.params.warmup_steps), 1000),
        "decay_steps": (int(config.lr_scheduler.params.decay_steps), 4480),
        "save_every": (int(config.experiment.save_every), 4480),
        "val_every": (int(config.experiment.val_every), 2240),
        "backbone_lr": (float(config.optimizer.params.backbone_learning_rate), 4e-5),
        "flow_lr": (float(config.optimizer.params.flow_learning_rate), 1e-4),
    }
    for label, (actual, expected) in required_values.items():
        if actual != expected:
            raise RuntimeError(f"config {label} mismatch: {actual!r} != {expected!r}")
    if bool(config.training.from_scratch):
        raise RuntimeError("training.from_scratch must be false")
    if not bool(config.training.use_ema):
        raise RuntimeError("training.use_ema must be true")
    if bool(config.training.use_gradient_checkpointing):
        raise RuntimeError(
            "activation checkpointing is not authorized without measured OOM"
        )
    if str(config.model.backbone_attention_output_gate) != "none":
        raise RuntimeError(
            "canonical backbone attention output gate must remain disabled"
        )

    microbatch = int(config.training.batch_size)
    if microbatch not in {16, 32}:
        raise RuntimeError(f"unsupported Ascend microbatch: {microbatch}")
    global_microbatch = microbatch * world_size
    global_batch = int(config.training.total_batch_size)
    if global_batch % global_microbatch:
        raise RuntimeError(
            f"global batch {global_batch} is not divisible by {microbatch} * {world_size}"
        )
    accumulation = global_batch // global_microbatch
    steps_per_epoch = split_counts["train"] // global_batch
    if steps_per_epoch != 224:
        raise RuntimeError(f"optimizer steps/epoch mismatch: {steps_per_epoch}")
    epochs = int(config.training.max_train_steps) // steps_per_epoch
    if epochs != 80:
        raise RuntimeError(f"epoch count mismatch: {epochs}")
    return {
        "world_size": world_size,
        "microbatch_per_rank": microbatch,
        "gradient_accumulation_steps": accumulation,
        "global_batch": global_batch,
        "optimizer_steps_per_epoch": steps_per_epoch,
        "epochs": epochs,
        "max_optimizer_steps": int(config.training.max_train_steps),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/selfless/imagenet100_class_base_80ep_ascend_16npu.yaml",
    )
    parser.add_argument("--world_size", type=int, default=16)
    parser.add_argument("--require_npu_count", type=int, default=None)
    parser.add_argument("--require_hccl_intra_roce", action="store_true")
    parser.add_argument("--skip_cache_value_scan", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = require_file(Path(args.config), "training config")
    config = OmegaConf.load(config_path)
    qwen_root = Path(config.model.model_path)
    vae_root = Path(config.experiment.validation_vae_module_root)
    vae_checkpoint = Path(config.experiment.validation_vae_path)
    membership_path = Path(config.dataset.params.manifest_jsonl)
    split_path = Path(config.dataset.params.split_manifest_jsonl)
    cache_path = Path(config.dataset.params.cache_path)

    hashes = {
        "qwen_weights": require_hash(
            qwen_root / "model.safetensors",
            EXPECTED_HASHES["qwen_weights"],
            "Qwen3-0.6B-Base weights",
        ),
        "vae_module": require_hash(
            vae_root / "models" / "vae.py",
            EXPECTED_HASHES["vae_module"],
            "MAR KL16 VAE module",
        ),
        "vae_checkpoint": require_hash(
            vae_checkpoint,
            EXPECTED_HASHES["vae_checkpoint"],
            "MAR KL16 checkpoint",
        ),
        "membership": require_hash(
            membership_path,
            EXPECTED_HASHES["membership"],
            "ImageNet-100 membership",
        ),
        "split": require_hash(
            split_path,
            EXPECTED_HASHES["split"],
            "ImageNet-100 split",
        ),
    }
    for filename in (
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ):
        require_file(qwen_root / filename, f"offline Qwen asset {filename}")
    require_file(
        Path(config.dataset.params.synset_mapping_path),
        "ImageNet synset mapping",
    )

    membership, membership_counts = load_membership(membership_path)
    split_counts = validate_split(split_path, membership)
    cache_metadata = validate_cache(
        require_file(cache_path, "KL16 posterior cache"),
        membership,
        scan_values=not args.skip_cache_value_scan,
    )
    training = validate_config(config, args.world_size, split_counts)

    hardware = None
    if args.require_npu_count is not None:
        import torch_npu  # noqa: F401

        available = bool(torch.npu.is_available())
        count = int(torch.npu.device_count())
        if not available or count != args.require_npu_count:
            raise RuntimeError(
                f"NPU contract failed: available={available}, count={count}, "
                f"expected={args.require_npu_count}"
            )
        hardware = {"npu_available": available, "npu_count": count}
    if args.require_hccl_intra_roce and os.environ.get("HCCL_INTRA_ROCE_ENABLE") != "1":
        raise RuntimeError("HCCL_INTRA_ROCE_ENABLE must equal 1")

    report = {
        "status": "ok",
        "config": str(config_path),
        "hashes": hashes,
        "membership": {
            "images": len(membership),
            "classes": len(membership_counts),
            "samples_per_class": sorted(set(membership_counts.values())),
        },
        "split": dict(split_counts),
        "cache": {
            "path": str(cache_path),
            "sha256": sha256_file(cache_path),
            "format": cache_metadata["format"],
            "stats_layout": cache_metadata["stats_layout"],
        },
        "training": training,
        "hardware": hardware,
        "hccl_intra_roce_enable": os.environ.get("HCCL_INTRA_ROCE_ENABLE"),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
