#!/usr/bin/env python3
import argparse
import hashlib
import json
import math
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SELECTION_SHA256 = (
    "f862b4d0bc1c48e089b97f75f651a91d23224fa26bb367347531f545f8162845"
)
EXPECTED_INCEPTION_SHA256 = (
    "6726825d0af5f729cebd5821db510b11b1cfad8faad88a03f1befd49fb9129b2"
)
EXPECTED_MANIFEST_SHA256 = (
    "6c6fc84e6ec9cb8b92421659d44abbb24d1cc34558213d9fb9db7e4ec44f1c3a"
)
EXPECTED_SPLIT_MANIFEST_SHA256 = (
    "02e5c67c058f95bcca46c82f3c1fc81086f61dcec62ce25049843f09d930a5ba"
)
EXPECTED_MODEL_PATH = (
    "output/selfless-flow-ablation-imagenet100-80ep/hf_model-final-ema"
)
EXPECTED_EMA_MODEL_SHA256 = (
    "81f86d1805d732f8c8e377a08cef6a6aad285eb533677405d4867bda90a86203"
)
EXPECTED_NON_EMA_MODEL_SHA256 = (
    "1af7302e4498a8bf4b50c8bd0d8fe3b008487ab2b82f1504eb34b9ac21b2dab1"
)
EXPECTED_CHECKPOINT_WEIGHT_DTYPES = {
    EXPECTED_MODEL_PATH: ["fp32"],
    "output/selfless-flow-ablation-imagenet100-80ep/hf_model-final": ["bf16"],
}
EXPECTED_CONFIG = "configs/ablation/imagenet_flow_100c_80ep.yaml"
EXPECTED_REAL_STATS_PATH = (REPO_ROOT / (
    "public/datasets/imagenet_ablation_100c_balanced/fid_stats/"
    "inception_v3_2048_original_256.pt"
)).resolve()
EXPECTED_INCEPTION_WEIGHTS_PATH = (REPO_ROOT / (
    "output/cache/inception/weights-inception-2015-12-05-6726825d.pth"
)).resolve()
EXPECTED_IMAGE_NAMES = tuple(f"{index:08d}.png" for index in range(10_000))
EXPECTED_METRIC_PROTOCOL = {
    "fid_reducer": "symmetric_eigendecomposition",
    "is_split_assignment": "contiguous_by_global_sample_index",
    "is_std": "population",
    "is_splits": 10,
}
EXPECTED_SPLIT_PROTOCOL = {
    "source": "authoritative_split_manifest",
    "order": "validation_split_index",
    "strategy": "stratified",
    "seed": 42,
    "val_samples_per_class": 100,
}
EXPECTED_PROMPT_PROTOCOL = {
    "type": "class_name",
    "mapping": "first comma-separated LOC_synset_mapping name",
}
EXPECTED_TRANSFORM_PROTOCOL = {
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
EXPECTED_FEATURE_PROTOCOL = {
    "backend": "torchmetrics.NoTrainInceptionV3/torch-fidelity",
    "extractor_antialias": True,
    "feature": 2048,
    "feature_name": "2048",
    "logits_name": "logits_unbiased",
    "weights_sha256": EXPECTED_INCEPTION_SHA256,
    "weights_filename": "weights-inception-2015-12-05-6726825d.pth",
}


def _nested(payload, *keys):
    current = payload
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _finite(value, *, minimum=None) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    numeric = float(value)
    return math.isfinite(numeric) and (minimum is None or numeric >= minimum)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path_matches(value, expected: Path) -> bool:
    if not isinstance(value, str) or not value:
        return False
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return candidate.resolve() == expected


def _real_stats_contract_checks(payload):
    metadata = _nested(payload, "real_stats_metadata") or {}
    feature = metadata.get("feature") if isinstance(metadata, dict) else None
    class_counts = metadata.get("class_counts") if isinstance(metadata, dict) else None
    critical_feature = {
        key: feature.get(key) if isinstance(feature, dict) else None
        for key in EXPECTED_FEATURE_PROTOCOL
    }
    return {
        "real_source": payload.get("real_source") == "cached_original_imagenet",
        "real_stats_path": _path_matches(
            payload.get("real_stats_path"), EXPECTED_REAL_STATS_PATH
        ),
        "real_stats_schema": metadata.get("schema")
        == "qwen_showo_imagenet100_real_stats_v1",
        "real_stats_protocol": metadata.get("protocol")
        == "imagenet100-balanced-val100-per-class-class-name-v1",
        "real_stats_source": metadata.get("real_source") == "original_imagenet",
        "manifest_sha256": metadata.get("manifest_sha256")
        == EXPECTED_MANIFEST_SHA256,
        "split_manifest_sha256": metadata.get("split_manifest_sha256")
        == EXPECTED_SPLIT_MANIFEST_SHA256,
        "selection_sha256": metadata.get("selection_sha256")
        == EXPECTED_SELECTION_SHA256,
        "real_samples": metadata.get("num_samples") == 10_000,
        "real_classes": metadata.get("num_classes") == 100,
        "real_class_counts": isinstance(class_counts, dict)
        and len(class_counts) == 100
        and set(class_counts.values()) == {100},
        "real_split_protocol": metadata.get("split") == EXPECTED_SPLIT_PROTOCOL,
        "real_prompt_protocol": metadata.get("prompt") == EXPECTED_PROMPT_PROTOCOL,
        "real_transform_protocol": metadata.get("transform")
        == EXPECTED_TRANSFORM_PROTOCOL,
        "real_feature_protocol": critical_feature == EXPECTED_FEATURE_PROTOCOL,
        "inception_weights_path": _path_matches(
            payload.get("inception_weights_path"),
            EXPECTED_INCEPTION_WEIGHTS_PATH,
        ),
    }


def validate_metrics(
    metrics_path: Path,
    expected_cfg: float,
    require_images: bool,
    *,
    expected_model_path: str = EXPECTED_MODEL_PATH,
    expected_model_sha256: str | None = None,
    expected_config: str = EXPECTED_CONFIG,
    expected_seed: int = 42,
    expected_model_dtype: str = "bf16",
):
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    errors = []

    model_sha256_matches = True
    if expected_model_sha256 is not None:
        model_root = Path(expected_model_path)
        if not model_root.is_absolute():
            model_root = REPO_ROOT / model_root
        model_file = model_root / "model.safetensors"
        try:
            model_sha256_matches = (
                _sha256_file(model_file) == expected_model_sha256.lower()
            )
        except OSError:
            model_sha256_matches = False

    precision_protocol = payload.get("precision_protocol")
    legacy_bf16_metrics = precision_protocol is None
    if legacy_bf16_metrics:
        precision_protocol = {}
    elif not isinstance(precision_protocol, dict):
        precision_protocol = {}
    recorded_model_dtype = precision_protocol.get("model_dtype")
    normalized_model_dtype = "bf16" if legacy_bf16_metrics else recorded_model_dtype
    expected_parameter_dtypes = {
        "bf16": ["torch.bfloat16"],
        "fp32": ["torch.float32"],
    }
    recorded_parameter_dtypes = precision_protocol.get("model_parameter_dtypes")
    parameter_dtypes_match = (
        legacy_bf16_metrics and expected_model_dtype == "bf16"
    ) or recorded_parameter_dtypes == expected_parameter_dtypes.get(
        expected_model_dtype
    )
    explicit_precision_matches = legacy_bf16_metrics or {
        "schema": precision_protocol.get("schema"),
        "vae_dtype": precision_protocol.get("vae_dtype"),
        "flow_integrator_dtype": precision_protocol.get("flow_integrator_dtype"),
        "autocast_enabled": precision_protocol.get("autocast_enabled"),
        "cuda_matmul_allow_tf32": precision_protocol.get(
            "cuda_matmul_allow_tf32"
        ),
        "cudnn_allow_tf32": precision_protocol.get("cudnn_allow_tf32"),
        "float32_matmul_precision": precision_protocol.get(
            "float32_matmul_precision"
        ),
    } == {
        "schema": "flow_eval_precision_v1",
        "vae_dtype": "fp32",
        "flow_integrator_dtype": "fp32",
        "autocast_enabled": False,
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": True,
        "float32_matmul_precision": "highest",
    }
    recorded_checkpoint_dtypes = precision_protocol.get(
        "checkpoint_weight_dtypes"
    )
    expected_checkpoint_dtypes = EXPECTED_CHECKPOINT_WEIGHT_DTYPES.get(
        expected_model_path
    )
    checkpoint_weight_dtypes_valid = legacy_bf16_metrics or (
        recorded_checkpoint_dtypes == expected_checkpoint_dtypes
        if expected_checkpoint_dtypes is not None
        else (
            isinstance(recorded_checkpoint_dtypes, list)
            and bool(recorded_checkpoint_dtypes)
            and all(
                isinstance(value, str) and bool(value)
                for value in recorded_checkpoint_dtypes
            )
        )
    )

    strategies = payload.get("strategies")
    checks = {
        "official_protocol": payload.get("official_protocol") is True,
        "config": payload.get("config") == expected_config,
        "model_path": payload.get("model_path") == expected_model_path,
        "model_sha256": model_sha256_matches,
        "model_dtype": normalized_model_dtype == expected_model_dtype,
        "model_parameter_dtypes": parameter_dtypes_match,
        "precision_protocol": explicit_precision_matches,
        "checkpoint_weight_dtypes": checkpoint_weight_dtypes_valid,
        "adapter": payload.get("adapter") == {"adapter": None},
        "model_state": payload.get("model_state") == {"model_state": None},
        "ema_state": payload.get("ema_state") == {"ema_state": None},
        "split": payload.get("split") == "val",
        "seed": payload.get("seed") == expected_seed,
        "batch_size": payload.get("batch_size") == 512,
        "samples_requested": payload.get("samples_requested") == 10_000,
        "samples_evaluated": payload.get("samples_evaluated") == 10_000,
        "distributed_enabled": _nested(payload, "distributed", "enabled") is True,
        "world_size": _nested(payload, "distributed", "world_size") == 8,
        "rank": _nested(payload, "distributed", "rank") == 0,
        "local_rank": _nested(payload, "distributed", "local_rank") == 0,
        "peak_cuda_allocated": _finite(
            _nested(payload, "distributed", "peak_cuda_allocated_mib"), minimum=0.0
        ),
        "peak_cuda_reserved": _finite(
            _nested(payload, "distributed", "peak_cuda_reserved_mib"), minimum=0.0
        ),
        "cfg": _finite(payload.get("cfg"))
        and math.isclose(float(payload["cfg"]), expected_cfg, abs_tol=1.0e-12),
        "cfg_schedule": payload.get("cfg_schedule") == "constant",
        "sampling_steps": str(payload.get("sampling_steps")) == "100",
        "temperature": payload.get("temperature") == 1.0,
        "flow_solver": payload.get("flow_solver") == "heun",
        "parallel_rate": payload.get("parallel_rate") == 1,
        "metric_protocol": payload.get("metric_protocol")
        == EXPECTED_METRIC_PROTOCOL,
        "strategies": isinstance(strategies, dict)
        and set(strategies) == {"spatial_halton"},
    }
    checks.update(_real_stats_contract_checks(payload))
    errors.extend(name for name, passed in checks.items() if not passed)

    strategy = _nested(payload, "strategies", "spatial_halton") or {}
    if not isinstance(strategy, dict):
        strategy = {}
    split_values = strategy.get("inception_score_splits")
    strategy_checks = {
        "strategy_count": strategy.get("count") == 10_000,
        "fid": _finite(strategy.get("fid"), minimum=0.0),
        "is_mean": _finite(strategy.get("inception_score_mean"), minimum=0.0),
        "is_std": _finite(strategy.get("inception_score_std"), minimum=0.0),
        "is_split_values": isinstance(split_values, list)
        and len(split_values) == 10
        and all(
            _finite(value, minimum=0.0)
            for value in split_values
        ),
        "latent_mse": _finite(strategy.get("latent_mse_to_target"), minimum=0.0),
        "latent_rms": _finite(strategy.get("latent_rms"), minimum=0.0),
        "generation_step_max": strategy.get("generation_step_max") == 256.0,
    }
    errors.extend(name for name, passed in strategy_checks.items() if not passed)

    image_counts = None
    if require_images:
        output_dir = metrics_path.parent
        image_counts = {}
        for directory_name in ("spatial_halton", "target_decoded"):
            directory = output_dir / directory_name
            names = tuple(
                sorted(path.name for path in directory.glob("*.png"))
            ) if directory.is_dir() else ()
            image_counts[directory_name] = len(names)
            if names != EXPECTED_IMAGE_NAMES:
                errors.append(f"{directory_name}_png_sequence")

    return payload, errors, image_counts


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate one formal ImageNet-100C Flow CFG evaluation."
    )
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--cfg", type=float, required=True)
    parser.add_argument(
        "--expected-model-path",
        default=EXPECTED_MODEL_PATH,
        help="Exact model_path expected in metrics.json.",
    )
    parser.add_argument(
        "--expected-model-sha256",
        default=None,
        help="Optional exact SHA256 for MODEL_PATH/model.safetensors.",
    )
    parser.add_argument(
        "--expected-config",
        default=EXPECTED_CONFIG,
        help="Exact config path expected in metrics.json.",
    )
    parser.add_argument("--expected-seed", type=int, default=42)
    parser.add_argument(
        "--expected-model-dtype",
        choices=("bf16", "fp32"),
        default="bf16",
    )
    parser.add_argument("--require-images", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    payload, errors, image_counts = validate_metrics(
        args.metrics,
        expected_cfg=float(args.cfg),
        require_images=bool(args.require_images),
        expected_model_path=args.expected_model_path,
        expected_model_sha256=args.expected_model_sha256,
        expected_config=args.expected_config,
        expected_seed=int(args.expected_seed),
        expected_model_dtype=args.expected_model_dtype,
    )
    result = {
        "metrics": str(args.metrics),
        "cfg": float(args.cfg),
        "valid": not errors,
        "model_dtype": (
            "bf16"
            if payload.get("precision_protocol") is None
            else _nested(payload, "precision_protocol", "model_dtype")
        ),
        "model_dtype_source": (
            "legacy_inferred"
            if payload.get("precision_protocol") is None
            else "explicit"
        ),
        "errors": errors,
        "fid": _nested(payload, "strategies", "spatial_halton", "fid"),
        "inception_score_mean": _nested(
            payload, "strategies", "spatial_halton", "inception_score_mean"
        ),
        "image_counts": image_counts,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
