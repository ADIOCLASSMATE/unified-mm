#!/usr/bin/env python3
"""Create or verify the immutable contract for a Show-o CFG sweep root."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "qwen_showo_cfg_sweep_contract_v1"
CHECKPOINT_SIDECARS = (
    "chat_template.jinja",
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
)
SOURCE_FILES = (
    "models/modeling_model/modeling_qwen_showo.py",
    "pretrain/train_qwen_showo.py",
    "scripts/evaluate_qwen_showo_fid_is.py",
    "scripts/ensure_showo_cfg_sweep_contract.py",
    "scripts/validate_showo_cfg_metrics.py",
    "scripts/summarize_showo_cfg_sweep.py",
    "script/ablation/evaluate_qwen_showo_vq_100c.sh",
    "script/ablation/evaluate_qwen_showo_vq_cfg_sweep_100c.sh",
)
FIXED_ARTIFACTS = {
    "manifest": (
        "public/datasets/imagenet_ablation_100c_balanced/manifest.jsonl"
    ),
    "split_manifest": (
        "public/datasets/imagenet_ablation_100c_balanced/"
        "split_seed42_val100.jsonl"
    ),
    "synset_mapping": "public/datasets/imagenet/LOC_synset_mapping.txt",
    "real_stats": (
        "public/datasets/imagenet_ablation_100c_balanced/fid_stats/"
        "inception_v3_2048_original_256.pt"
    ),
    "inception_weights": (
        "output/cache/inception/"
        "weights-inception-2015-12-05-6726825d.pth"
    ),
    "magvit_config": "public/models/showlab/magvitv2/config.json",
    "magvit_weights": (
        "public/models/showlab/magvitv2/pytorch_model.safetensors"
    ),
    "official_magvit_source": (
        "/inspire/hdd/global_user/wanjiaxin-253108030048/code/"
        "Show-o/models/modeling_magvitv2.py"
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_path(raw: str | Path) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def require_file(raw: str | Path) -> Path:
    path = resolve_path(raw)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def checkpoint_sidecars(checkpoint: Path) -> dict[str, str]:
    return {
        name: sha256_file(require_file(checkpoint / name))
        for name in CHECKPOINT_SIDECARS
    }


def source_hashes() -> dict[str, str]:
    return {
        path: sha256_file(require_file(path))
        for path in SOURCE_FILES
    }


def artifact_hashes() -> dict[str, dict[str, str]]:
    result = {}
    for name, raw_path in FIXED_ARTIFACTS.items():
        path = require_file(raw_path)
        result[name] = {
            "path": str(path),
            "sha256": sha256_file(path),
        }
    return result


def build_contract(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = resolve_path(args.checkpoint)
    config = require_file(args.config)
    model_weights = require_file(checkpoint / "model.safetensors")
    actual_model_hash = sha256_file(model_weights)
    if actual_model_hash != args.checkpoint_sha256.lower():
        raise ValueError(
            "checkpoint SHA256 mismatch: "
            f"expected={args.checkpoint_sha256.lower()}, actual={actual_model_hash}"
        )
    if int(args.num_gpus) != 8 or int(args.local_batch_size) != 8:
        raise ValueError(
            "formal Show-o sweep requires 8 GPUs and local batch size 8"
        )
    if int(args.timesteps) != 12 or float(args.temperature) != 1.0:
        raise ValueError(
            "formal Show-o sweep requires 12 timesteps and temperature 1.0"
        )
    if int(args.seed) != 42:
        raise ValueError("formal Show-o sweep requires seed 42")

    return {
        "schema": SCHEMA,
        "checkpoint": {
            "path": str(checkpoint),
            "model_sha256": actual_model_hash,
            "sidecar_sha256": checkpoint_sidecars(checkpoint),
        },
        "config": {
            "path": str(config),
            "sha256": sha256_file(config),
        },
        "source_sha256": source_hashes(),
        "artifact_sha256": artifact_hashes(),
        "protocol": {
            "name": "imagenet100-balanced-val100-per-class-class-name-v1",
            "samples": 10_000,
            "seed": int(args.seed),
            "method": "maskgit",
            "timesteps": int(args.timesteps),
            "temperature": float(args.temperature),
            "temperature_schedule": (
                "official_showo_cumulative_one_minus_ratio"
            ),
            "mask_schedule": "cosine",
            "guidance_formula": "(1+s)*conditional-s*unconditional",
            "common_cfg_mapping": "w=1+s",
            "num_gpus": int(args.num_gpus),
            "local_batch_size": int(args.local_batch_size),
            "is_splits": 10,
            "fid_feature": 2048,
            "save_images": bool(args.save_images),
        },
    }


def differences(left: Any, right: Any, prefix: str = "") -> list[str]:
    if isinstance(left, dict) and isinstance(right, dict):
        result = []
        for key in sorted(set(left) | set(right)):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in left or key not in right:
                result.append(path)
            else:
                result.extend(differences(left[key], right[key], path))
        return result
    return [] if left == right else [prefix or "<root>"]


def ensure_contract(root: Path, expected: dict[str, Any]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / ".sweep_contract.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        mismatches = differences(existing, expected)
        if mismatches:
            raise ValueError(
                "Show-o sweep contract mismatch: " + ", ".join(mismatches)
            )
        return path

    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(expected, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create or verify a Show-o CFG sweep contract."
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument("--local-batch-size", type=int, default=8)
    parser.add_argument("--timesteps", type=int, default=12)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-images", type=int, choices=(0, 1), default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    contract = build_contract(args)
    path = ensure_contract(args.root, contract)
    print(
        json.dumps(
            {"contract": str(path), "schema": SCHEMA, "valid": True},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
