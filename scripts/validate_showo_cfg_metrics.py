#!/usr/bin/env python3
"""Validate one formal Qwen-Show-o ImageNet-100 CFG evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_PROTOCOL = "imagenet100-balanced-val100-per-class-class-name-v1"
EXPECTED_CONFIG = "configs/ablation/qwen_showo_vq_100c_80ep.yaml"
EXPECTED_CHECKPOINT = (
    "output/qwen-showo-vq-ablation-imagenet100-80ep/hf_model-final"
)
EXPECTED_CHECKPOINT_SHA256 = (
    "2eaf3c5958c36be4f2554ce88f67082cc6e40d67924df945c8b35a3efdec1806"
)
EXPECTED_SAMPLES_SHA256 = (
    "b4d3dd73da722a7367118d9604d4b7beab53baf9de0329e2af8d18218bf45190"
)
EXPECTED_REAL_STATS = (
    REPO_ROOT
    / "public/datasets/imagenet_ablation_100c_balanced/fid_stats/"
    "inception_v3_2048_original_256.pt"
).resolve()
EXPECTED_MAGVIT = (REPO_ROOT / "public/models/showlab/magvitv2").resolve()
EXPECTED_MANIFEST_SHA256 = (
    "6c6fc84e6ec9cb8b92421659d44abbb24d1cc34558213d9fb9db7e4ec44f1c3a"
)
EXPECTED_SPLIT_MANIFEST_SHA256 = (
    "02e5c67c058f95bcca46c82f3c1fc81086f61dcec62ce25049843f09d930a5ba"
)
EXPECTED_SELECTION_SHA256 = (
    "f862b4d0bc1c48e089b97f75f651a91d23224fa26bb367347531f545f8162845"
)
EXPECTED_INCEPTION_SHA256 = (
    "6726825d0af5f729cebd5821db510b11b1cfad8faad88a03f1befd49fb9129b2"
)
EXPECTED_IMAGE_NAMES = tuple(f"{index:08d}.png" for index in range(10_000))
EXPECTED_SPLIT = {
    "source": "authoritative_split_manifest",
    "order": "validation_split_index",
    "strategy": "stratified",
    "seed": 42,
    "val_samples_per_class": 100,
}
EXPECTED_PROMPT = {
    "type": "class_name",
    "mapping": "first comma-separated LOC_synset_mapping name",
}
EXPECTED_TRANSFORM = {
    "input_color": "RGB",
    "resize": {
        "type": "shorter_side",
        "size": 256,
        "interpolation": "bicubic",
        "antialias": True,
    },
    "crop": {"type": "center", "height": 256, "width": 256},
    "tensor_range": [0.0, 1.0],
    "inception_input": "uint8_0_255",
    "quantization": "multiply_255_then_uint8_truncate",
    "inception_resize": {
        "size": 299,
        "interpolation": "bilinear",
        "antialias": True,
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def nested(payload: Any, *keys: str) -> Any:
    current = payload
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def finite(value: Any, *, minimum: float | None = None) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    numeric = float(value)
    return math.isfinite(numeric) and (
        minimum is None or numeric >= minimum
    )


def path_matches(value: Any, expected: Path) -> bool:
    if not isinstance(value, str) or not value:
        return False
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return candidate.resolve() == expected.resolve()


def validate_metrics(
    metrics_path: Path,
    expected_guidance_scale: float,
    require_images: bool,
    *,
    expected_checkpoint: str = EXPECTED_CHECKPOINT,
    expected_checkpoint_sha256: str | None = EXPECTED_CHECKPOINT_SHA256,
    expected_samples_sha256: str | None = EXPECTED_SAMPLES_SHA256,
    expected_config: str = EXPECTED_CONFIG,
    expected_seed: int = 42,
) -> tuple[dict[str, Any], list[str], int | None]:
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))

    checkpoint_root = Path(expected_checkpoint)
    if not checkpoint_root.is_absolute():
        checkpoint_root = REPO_ROOT / checkpoint_root
    checkpoint_hash_matches = True
    if expected_checkpoint_sha256 is not None:
        try:
            checkpoint_hash_matches = (
                sha256_file(checkpoint_root / "model.safetensors")
                == expected_checkpoint_sha256.lower()
            )
        except OSError:
            checkpoint_hash_matches = False

    config_path = Path(expected_config)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    expected_common_cfg = 1.0 + float(expected_guidance_scale)
    sampling = payload.get("sampling")
    if not isinstance(sampling, dict):
        sampling = {}
    real_metadata = nested(payload, "real_stats", "metadata")
    if not isinstance(real_metadata, dict):
        real_metadata = {}
    class_counts = real_metadata.get("class_counts")
    feature = real_metadata.get("feature")
    if not isinstance(feature, dict):
        feature = {}
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        metrics = {}
    split_scores = metrics.get("inception_score_splits")

    checks = {
        "protocol": payload.get("protocol") == EXPECTED_PROTOCOL,
        "official_protocol": payload.get("official_protocol") is True,
        "config": path_matches(payload.get("config"), config_path),
        "checkpoint": path_matches(
            payload.get("checkpoint"), checkpoint_root
        ),
        "checkpoint_sha256": checkpoint_hash_matches,
        "samples": payload.get("samples") == 10_000,
        "seed": payload.get("seed") == int(expected_seed),
        "sampling_method": sampling.get("method") == "maskgit",
        "timesteps": sampling.get("timesteps") == 12,
        "guidance_scale": finite(sampling.get("guidance_scale"))
        and math.isclose(
            float(sampling["guidance_scale"]),
            float(expected_guidance_scale),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ),
        "common_cfg_scale": finite(sampling.get("common_cfg_scale"))
        and math.isclose(
            float(sampling["common_cfg_scale"]),
            expected_common_cfg,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ),
        "guidance_formula": sampling.get("guidance_formula")
        == "(1+s)*conditional-s*unconditional",
        "temperature": sampling.get("temperature") == 1.0,
        "temperature_schedule": sampling.get("temperature_schedule")
        == "official_showo_cumulative_one_minus_ratio",
        "mask_schedule": sampling.get("mask_schedule") == "cosine",
        "tokenizer_type": nested(payload, "tokenizer", "type")
        == "official_showo_magvitv2",
        "tokenizer_path": path_matches(
            nested(payload, "tokenizer", "path"), EXPECTED_MAGVIT
        ),
        "image_vocab_size": nested(
            payload, "tokenizer", "image_vocab_size"
        )
        == 8192,
        "tokens_per_image": nested(
            payload, "tokenizer", "tokens_per_image"
        )
        == 256,
        "decode_dtype": nested(payload, "tokenizer", "decode_dtype")
        == "float32",
        "real_stats_path": path_matches(
            nested(payload, "real_stats", "path"), EXPECTED_REAL_STATS
        ),
        "real_stats_schema": real_metadata.get("schema")
        == "qwen_showo_imagenet100_real_stats_v1",
        "real_stats_protocol": real_metadata.get("protocol")
        == EXPECTED_PROTOCOL,
        "real_stats_source": real_metadata.get("real_source")
        == "original_imagenet",
        "manifest_sha256": real_metadata.get("manifest_sha256")
        == EXPECTED_MANIFEST_SHA256,
        "split_manifest_sha256": real_metadata.get(
            "split_manifest_sha256"
        )
        == EXPECTED_SPLIT_MANIFEST_SHA256,
        "selection_sha256": real_metadata.get("selection_sha256")
        == EXPECTED_SELECTION_SHA256,
        "real_samples": real_metadata.get("num_samples") == 10_000,
        "real_classes": real_metadata.get("num_classes") == 100,
        "real_class_counts": isinstance(class_counts, dict)
        and len(class_counts) == 100
        and set(class_counts.values()) == {100},
        "split_protocol": real_metadata.get("split") == EXPECTED_SPLIT,
        "prompt_protocol": real_metadata.get("prompt") == EXPECTED_PROMPT,
        "transform_protocol": real_metadata.get("transform")
        == EXPECTED_TRANSFORM,
        "fid_backend": feature.get("backend")
        == "torchmetrics.NoTrainInceptionV3/torch-fidelity",
        "fid_feature_metadata": feature.get("feature") == 2048,
        "fid_feature_name": feature.get("feature_name") == "2048",
        "fid_logits_name": feature.get("logits_name")
        == "logits_unbiased",
        "inception_sha256": feature.get("weights_sha256")
        == EXPECTED_INCEPTION_SHA256,
        "world_size": nested(payload, "distributed", "world_size") == 8,
        "local_batch_size": nested(
            payload, "distributed", "local_batch_size"
        )
        == 8,
        "fid": finite(metrics.get("fid"), minimum=0.0),
        "fid_feature": metrics.get("fid_feature") == 2048,
        "is_mean": finite(
            metrics.get("inception_score_mean"), minimum=0.0
        ),
        "is_std": finite(
            metrics.get("inception_score_std"), minimum=0.0
        ),
        "is_splits": isinstance(split_scores, list)
        and len(split_scores) == 10
        and all(finite(value, minimum=0.0) for value in split_scores),
    }

    if expected_samples_sha256 is not None:
        samples_path = metrics_path.parent / "samples.jsonl"
        try:
            samples_hash_matches = (
                sha256_file(samples_path) == expected_samples_sha256.lower()
            )
        except OSError:
            samples_hash_matches = False
        checks["samples_manifest_sha256"] = samples_hash_matches

    image_count = None
    if require_images:
        checks["saved_images"] = payload.get("saved_images") is True
        generated_dir = metrics_path.parent / "generated"
        names = (
            tuple(sorted(path.name for path in generated_dir.glob("*.png")))
            if generated_dir.is_dir()
            else ()
        )
        image_count = len(names)
        checks["generated_png_sequence"] = names == EXPECTED_IMAGE_NAMES

    errors = [name for name, passed in checks.items() if not passed]
    return payload, errors, image_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate one formal Qwen-Show-o ImageNet-100 CFG point."
    )
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--guidance-scale", type=float, required=True)
    parser.add_argument("--expected-checkpoint", default=EXPECTED_CHECKPOINT)
    parser.add_argument(
        "--expected-checkpoint-sha256",
        default=EXPECTED_CHECKPOINT_SHA256,
    )
    parser.add_argument(
        "--expected-samples-sha256",
        default=EXPECTED_SAMPLES_SHA256,
    )
    parser.add_argument("--expected-config", default=EXPECTED_CONFIG)
    parser.add_argument("--expected-seed", type=int, default=42)
    parser.add_argument("--require-images", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload, errors, image_count = validate_metrics(
        args.metrics,
        expected_guidance_scale=float(args.guidance_scale),
        require_images=bool(args.require_images),
        expected_checkpoint=args.expected_checkpoint,
        expected_checkpoint_sha256=args.expected_checkpoint_sha256,
        expected_samples_sha256=args.expected_samples_sha256,
        expected_config=args.expected_config,
        expected_seed=int(args.expected_seed),
    )
    result = {
        "metrics": str(args.metrics),
        "valid": not errors,
        "errors": errors,
        "guidance_scale": nested(
            payload, "sampling", "guidance_scale"
        ),
        "common_cfg_scale": nested(
            payload, "sampling", "common_cfg_scale"
        ),
        "fid": nested(payload, "metrics", "fid"),
        "inception_score_mean": nested(
            payload, "metrics", "inception_score_mean"
        ),
        "image_count": image_count,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
