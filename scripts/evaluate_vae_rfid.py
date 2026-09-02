#!/usr/bin/env python3
"""Evaluate KL-VAE reconstruction FID from cached ImageNet posteriors."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

from scripts.image_evaluation_metrics import (
    FeatureMoments,
    build_inception_extractor,
    extract_inception_features,
    frechet_distance,
)
from utils.image_generation_io import decode_latents


CACHE_FORMAT = "imagenet_kl16_scaled_posterior_v1"
CACHE_LAYOUT = "scaled_mean_then_scaled_std"
REAL_STATS_SCHEMA = "imagenet_inception_feature_moments_v1"
REAL_STATS_SCHEMAS = {
    REAL_STATS_SCHEMA,
    "imagenet_val_inception_feature_moments_v2",
}
RESULT_SCHEMA = "mar_kl16_vae_rfid_v2"
PRIMARY_METRIC_PATH = "metrics.sample.rfid"
EXPECTED_SCALING_FACTOR = 0.2325
IMAGE_TOKENS = 256
LATENT_CHANNELS = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache_shard_dir",
        default=(
            "public/datasets/imagenet_full/vae_posterior_mar_kl16/"
            "val_shards"
        ),
    )
    parser.add_argument("--cache_shards", type=int, default=16)
    parser.add_argument("--vae_module_root", default="public/code/mar")
    parser.add_argument(
        "--vae_path", default="public/vae/mar-kl16/kl16.ckpt"
    )
    parser.add_argument(
        "--inception_weights_path",
        default=(
            "public/models/torch-fidelity/"
            "weights-inception-2015-12-05-6726825d.pth"
        ),
    )
    parser.add_argument(
        "--real_stats_path",
        default=(
            "public/datasets/imagenet_full/fid_stats/"
            "inception_v3_2048_imagenet_val50000_256.pt"
        ),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=["npu", "cuda"], default="npu")
    parser.add_argument(
        "--vae_dtype", choices=["fp16", "fp32"], default="fp32"
    )
    parser.add_argument(
        "--posterior_modes",
        nargs="+",
        choices=["sample", "mean"],
        default=["sample", "mean"],
    )
    parser.add_argument("--samples", type=int, default=50_000)
    parser.add_argument("--batch_size_per_rank", type=int, default=16)
    parser.add_argument("--feature", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--require_full_protocol", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def init_distributed(device_type: str):
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if device_type == "npu":
        import torch_npu  # noqa: F401

        device = torch.device("npu", local_rank)
        torch.npu.set_device(device)
        torch.npu.set_compile_mode(jit_compile=False)
        backend = "hccl"
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
        backend = "nccl"
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(
            backend=backend,
            timeout=timedelta(
                seconds=int(
                    os.environ.get(
                        "EVAL_PROCESS_GROUP_TIMEOUT_SECONDS", "3600"
                    )
                )
            ),
        )
    return rank, world_size, device


def barrier(device: torch.device) -> None:
    if not dist.is_initialized():
        return
    dist.barrier(device_ids=[int(device.index or 0)])


def load_vae(
    module_root: Path,
    checkpoint: Path,
    device: torch.device,
    dtype_name: str,
):
    module_path = module_root / "models" / "vae.py"
    spec = importlib.util.spec_from_file_location("mar_kl16_vae", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load KL16 VAE module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    vae = module.AutoencoderKL(
        embed_dim=LATENT_CHANNELS,
        ch_mult=(1, 1, 2, 2, 4),
        ckpt_path=str(checkpoint),
    )
    dtype = torch.float16 if dtype_name == "fp16" else torch.float32
    vae = vae.to(device=device, dtype=dtype).eval()
    for parameter in vae.parameters():
        parameter.requires_grad_(False)
    return vae


def load_rank_cache(
    shard_dir: Path,
    *,
    rank: int,
    cache_shards: int,
    samples: int,
):
    path = shard_dir / f"shard-{rank:05d}-of-{cache_shards:05d}.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(
        str(path),
        map_location="cpu",
        mmap=True,
        weights_only=True,
    )
    stats = payload.get("posterior_stats")
    image_ids = payload.get("img_ids")
    metadata = payload.get("metadata", {})
    if metadata.get("format") != CACHE_FORMAT:
        raise ValueError(f"invalid cache format in {path}")
    if metadata.get("stats_layout") != CACHE_LAYOUT:
        raise ValueError(f"invalid cache layout in {path}")
    if int(metadata.get("num_shards", -1)) != int(cache_shards):
        raise ValueError(f"cache shard-count mismatch in {path}")
    if int(metadata.get("shard_index", -1)) != int(rank):
        raise ValueError(f"cache rank mismatch in {path}")
    if not torch.is_tensor(stats) or not torch.is_tensor(image_ids):
        raise ValueError(f"cache tensors are missing from {path}")
    if stats.ndim != 3 or tuple(stats.shape[1:]) != (
        IMAGE_TOKENS,
        2 * LATENT_CHANNELS,
    ):
        raise ValueError(f"invalid posterior shape in {path}: {stats.shape}")
    if stats.dtype != torch.float16:
        raise ValueError(f"posterior cache must be float16, got {stats.dtype}")
    if image_ids.ndim != 1 or int(image_ids.shape[0]) != int(stats.shape[0]):
        raise ValueError(f"cache image-id coverage is invalid in {path}")
    if int(metadata.get("num_images", -1)) != int(stats.shape[0]):
        raise ValueError(f"cache metadata image count is invalid in {path}")
    local_count = int((image_ids <= int(samples)).sum().item())
    stats = stats[:local_count]
    image_ids = image_ids[:local_count]
    expected_ids = torch.arange(
        rank + 1,
        int(samples) + 1,
        cache_shards,
        dtype=torch.long,
    )
    if not torch.equal(image_ids, expected_ids):
        raise ValueError(
            f"cache coverage mismatch in {path}: found {image_ids.tolist()[:8]}, "
            f"expected {expected_ids.tolist()[:8]}"
        )
    return path, stats, image_ids, metadata


def validate_full_cache_metadata(metadata: dict, path: Path) -> None:
    expected = {
        "stats_are_scaled": True,
        "source_mode": "manifest_jsonl",
        "vae": "mar-kl16",
        "vae_dtype": "float16",
        "storage_dtype": "float16",
        "image_size": 256,
        "posterior_shape": [16, 16, 32],
        "token_shape": [256, 32],
        "runtime_hashing_enabled": False,
    }
    for key, expected_value in expected.items():
        if metadata.get(key) != expected_value:
            raise ValueError(
                f"full-protocol cache metadata mismatch for {key} in "
                f"{path}: {metadata.get(key)!r} != {expected_value!r}"
            )
    source_manifest = Path(str(metadata.get("source_manifest_jsonl", "")))
    if source_manifest.name != "manifest_val.jsonl":
        raise ValueError(
            f"full protocol requires the ImageNet validation manifest, got "
            f"{source_manifest}"
        )
    scaling_factor = float(metadata.get("scaling_factor", 0.0))
    if scaling_factor != EXPECTED_SCALING_FACTOR:
        raise ValueError(
            f"full protocol requires KL16 scaling factor "
            f"{EXPECTED_SCALING_FACTOR}, got {scaling_factor}"
        )


def canonical_cache_metadata(metadata: dict) -> dict:
    """Return the reportable, readable cache contract without digests."""
    return {
        "format": metadata.get("format"),
        "stats_layout": metadata.get("stats_layout"),
        "stats_are_scaled": metadata.get("stats_are_scaled"),
        "source_mode": metadata.get("source_mode"),
        "source_manifest_jsonl": metadata.get("source_manifest_jsonl"),
        "vae": metadata.get("vae"),
        "vae_dtype": metadata.get("vae_dtype"),
        "storage_dtype": metadata.get("storage_dtype"),
        "scaling_factor": metadata.get("scaling_factor"),
        "image_size": metadata.get("image_size"),
        "posterior_shape": metadata.get("posterior_shape"),
        "token_shape": metadata.get("token_shape"),
        "runtime_hashing_enabled": metadata.get(
            "runtime_hashing_enabled"
        ),
    }


def stable_posterior_sample(
    mean: torch.Tensor,
    std: torch.Tensor,
    image_ids: torch.Tensor,
    seed: int,
    storage_dtype: torch.dtype,
) -> torch.Tensor:
    noise_rows = []
    for image_id in image_ids.tolist():
        global_index = int(image_id) - 1
        posterior_seed = (
            int(seed) + 97_409 * global_index + 11
        ) & ((1 << 63) - 1)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(posterior_seed)
        noise_rows.append(
            torch.randn(
                (IMAGE_TOKENS, LATENT_CHANNELS),
                generator=generator,
                dtype=torch.float32,
            )
        )
    noise = torch.stack(noise_rows)
    return (mean + std * noise).to(dtype=storage_dtype).float()


def tokens_to_chw(tokens: torch.Tensor) -> torch.Tensor:
    side = int(IMAGE_TOKENS**0.5)
    return tokens.reshape(-1, side, side, LATENT_CHANNELS).permute(
        0, 3, 1, 2
    )


def load_real_moments(path: Path, feature: int):
    payload = torch.load(str(path), map_location="cpu", weights_only=True)
    if payload.get("schema") not in REAL_STATS_SCHEMAS:
        raise ValueError(
            f"invalid real-stat schema in {path}: {payload.get('schema')!r}"
        )
    stats = payload.get("stats", {})
    count = int(stats.get("count", 0))
    feature_sum = torch.as_tensor(stats.get("sum"))
    outer_sum = torch.as_tensor(stats.get("outer_sum"))
    if count < 2:
        raise ValueError(f"invalid real-stat count in {path}: {count}")
    if tuple(feature_sum.shape) != (feature,):
        raise ValueError(f"invalid real-stat sum shape: {feature_sum.shape}")
    if tuple(outer_sum.shape) != (feature, feature):
        raise ValueError(
            f"invalid real-stat outer-sum shape: {outer_sum.shape}"
        )
    moments = FeatureMoments(
        count=torch.tensor(count, dtype=torch.long),
        sum=feature_sum,
        outer_sum=outer_sum,
    )
    metadata = payload.get("metadata", {})
    source = metadata.get("source", {})
    feature_metadata = metadata.get("feature", {})
    image_transform = metadata.get("image_transform", {})
    expected_metadata = {
        "classes": (source.get("classes"), 1000),
        "samples_per_class": (source.get("samples_per_class"), 50),
        "extractor": (
            feature_metadata.get("extractor"),
            "torch-fidelity-inception-v3-compat",
        ),
        "feature": (feature_metadata.get("feature"), int(feature)),
        "resize": (image_transform.get("resize"), 256),
        "interpolation": (
            image_transform.get("interpolation"),
            "bicubic",
        ),
        "center_crop": (image_transform.get("center_crop"), 256),
        "color_mode": (image_transform.get("color_mode"), "RGB"),
    }
    if source.get("split") not in {"val", "validation"}:
        raise ValueError(
            f"real-stat metadata mismatch for split in {path}: "
            f"{source.get('split')!r} is not ImageNet validation"
        )
    for key, (actual, expected) in expected_metadata.items():
        if actual != expected:
            raise ValueError(
                f"real-stat metadata mismatch for {key} in {path}: "
                f"{actual!r} != {expected!r}"
            )
    report_metadata = {
        "schema": payload.get("schema"),
        "source": {
            "split": source.get("split"),
            "classes": source.get("classes"),
            "samples_per_class": source.get("samples_per_class"),
        },
        "feature": {
            "extractor": feature_metadata.get("extractor"),
            "feature": feature_metadata.get("feature"),
            "accumulation_dtype": feature_metadata.get(
                "accumulation_dtype"
            ),
        },
        "image_transform": {
            "resize": image_transform.get("resize"),
            "interpolation": image_transform.get("interpolation"),
            "center_crop": image_transform.get("center_crop"),
            "color_mode": image_transform.get("color_mode"),
        },
    }
    return moments, report_metadata


def validate_full_protocol_args(args: argparse.Namespace, modes: list[str]) -> None:
    requirements = {
        "samples": (int(args.samples), 50_000),
        "cache_shards": (int(args.cache_shards), 16),
        "feature": (int(args.feature), 2048),
        "posterior_modes": (modes, ["sample", "mean"]),
        "posterior_seed": (int(args.seed), 42),
        "vae_dtype": (args.vae_dtype, "fp32"),
    }
    for name, (actual, expected) in requirements.items():
        if actual != expected:
            raise ValueError(
                f"full rFID protocol requires {name}={expected!r}, "
                f"got {actual!r}"
            )


def write_json_atomic(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.samples < 2 or args.batch_size_per_rank < 1:
        raise ValueError("samples must be >= 2 and batch size must be >= 1")
    modes = list(dict.fromkeys(args.posterior_modes))
    if args.require_full_protocol:
        validate_full_protocol_args(args, modes)
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise FileExistsError(output)

    rank, world_size, device = init_distributed(args.device)
    try:
        if world_size != int(args.cache_shards):
            raise ValueError(
                f"world size must match cache shards: {world_size} != "
                f"{args.cache_shards}"
            )
        shard_path, stats, image_ids, cache_metadata = load_rank_cache(
            Path(args.cache_shard_dir),
            rank=rank,
            cache_shards=int(args.cache_shards),
            samples=int(args.samples),
        )
        scaling_factor = float(cache_metadata.get("scaling_factor", 0.0))
        if scaling_factor <= 0:
            raise ValueError(f"invalid cache scaling factor: {scaling_factor}")
        if args.require_full_protocol:
            validate_full_cache_metadata(cache_metadata, shard_path)

        real_moments = real_metadata = None
        if rank == 0:
            real_moments, real_metadata = load_real_moments(
                Path(args.real_stats_path), int(args.feature)
            )
            if args.require_full_protocol:
                if int(real_moments.count.item()) != int(args.samples):
                    raise ValueError(
                        "full protocol requires matching 50K real statistics"
                    )
        barrier(device)

        vae = load_vae(
            Path(args.vae_module_root),
            Path(args.vae_path),
            device,
            args.vae_dtype,
        )
        inception = build_inception_extractor(
            int(args.feature),
            args.inception_weights_path,
            device,
        )
        reconstructed = {
            mode: FeatureMoments.zeros(int(args.feature), device)
            for mode in modes
        }

        started = time.perf_counter()
        total_batches = (
            len(stats) + int(args.batch_size_per_rank) - 1
        ) // int(args.batch_size_per_rank)
        for batch_index, start in enumerate(
            range(0, len(stats), int(args.batch_size_per_rank))
        ):
            end = min(start + int(args.batch_size_per_rank), len(stats))
            batch_stats = stats[start:end].float()
            batch_ids = image_ids[start:end]
            mean = batch_stats[..., :LATENT_CHANNELS]
            std = batch_stats[..., LATENT_CHANNELS:]
            if not bool(torch.isfinite(batch_stats).all()) or bool(
                (std < 0).any()
            ):
                raise FloatingPointError(
                    f"invalid posterior stats in {shard_path} rows {start}:{end}"
                )

            latent_batches = []
            for mode in modes:
                tokens = (
                    stable_posterior_sample(
                        mean,
                        std,
                        batch_ids,
                        int(args.seed),
                        stats.dtype,
                    )
                    if mode == "sample"
                    else mean
                )
                latent_batches.append(tokens_to_chw(tokens))
            batch_size = end - start
            latents = torch.cat(latent_batches, dim=0).to(device=device)
            images = decode_latents(vae, latents, scaling_factor)
            features, _ = extract_inception_features(inception, images)
            if not bool(torch.isfinite(features).all()):
                raise FloatingPointError(
                    f"non-finite Inception features at rank={rank}, "
                    f"batch={batch_index}"
                )
            for mode_index, mode in enumerate(modes):
                offset = mode_index * batch_size
                reconstructed[mode].update(
                    features[offset : offset + batch_size]
                )
            if rank == 0 and (
                (batch_index + 1) % 10 == 0
                or batch_index + 1 == total_batches
            ):
                print(
                    f"rFID progress: {batch_index + 1}/{total_batches} "
                    f"local batches",
                    flush=True,
                )

        for moments in reconstructed.values():
            moments.all_reduce_()
            if int(moments.count.item()) != int(args.samples):
                raise RuntimeError(
                    f"distributed reconstruction count is "
                    f"{int(moments.count.item())}, expected {args.samples}"
                )

        if device.type == "npu":
            torch.npu.synchronize(device)
        else:
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started

        if rank == 0:
            assert real_moments is not None
            real_mean, real_cov = real_moments.mean_cov()
            metrics = {}
            for mode, moments in reconstructed.items():
                reconstruction_mean, reconstruction_cov = moments.mean_cov()
                metrics[mode] = {
                    "rfid": frechet_distance(
                        real_mean,
                        real_cov,
                        reconstruction_mean,
                        reconstruction_cov,
                    ),
                    "samples": int(moments.count.item()),
                }
            result = {
                "schema": RESULT_SCHEMA,
                "primary_metric": {
                    "path": PRIMARY_METRIC_PATH,
                    "value": metrics["sample"]["rfid"],
                    "direction": "lower_is_better",
                },
                "secondary_diagnostics": ["metrics.mean.rfid"],
                "metrics": metrics,
                "protocol": {
                    "dataset": "ImageNet-1K validation",
                    "samples": int(args.samples),
                    "resolution": 256,
                    "posterior_source": "cached scaled mean/std",
                    "posterior_storage_dtype": str(stats.dtype).removeprefix(
                        "torch."
                    ),
                    "posterior_encoder_dtype": cache_metadata.get(
                        "vae_dtype"
                    ),
                    "posterior_modes": modes,
                    "posterior_sample_role": "paper_primary",
                    "posterior_mean_role": "diagnostic_only",
                    "posterior_seed": int(args.seed),
                    "vae_decoder_dtype": args.vae_dtype,
                    "scaling_factor": scaling_factor,
                    "inception_feature": int(args.feature),
                    "world_size": int(world_size),
                    "batch_size_per_rank": int(args.batch_size_per_rank),
                    "elapsed_seconds": elapsed,
                    "full_protocol": bool(args.require_full_protocol),
                    "runtime_hashing_enabled": False,
                },
                "artifacts": {
                    "cache_shard_dir": str(
                        Path(args.cache_shard_dir).resolve()
                    ),
                    "vae_module_root": str(
                        Path(args.vae_module_root).resolve()
                    ),
                    "vae_checkpoint": str(Path(args.vae_path).resolve()),
                    "inception_weights": str(
                        Path(args.inception_weights_path).resolve()
                    ),
                    "real_stats": str(Path(args.real_stats_path).resolve()),
                },
                "cache_metadata": canonical_cache_metadata(cache_metadata),
                "real_stats_metadata": real_metadata,
            }
            write_json_atomic(result, output)
            print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
        barrier(device)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
