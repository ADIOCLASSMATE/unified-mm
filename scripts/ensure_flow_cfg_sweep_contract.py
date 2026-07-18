#!/usr/bin/env python3
"""Create or verify the immutable protocol contract for one CFG sweep root."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "selfless_flow_cfg_sweep_contract_v4"
SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
FIXED_ARTIFACTS = {
    "real_stats": REPO_ROOT
    / "public/datasets/imagenet_ablation_100c_balanced/fid_stats/"
    "inception_v3_2048_original_256.pt",
    "inception_weights": REPO_ROOT
    / "output/cache/inception/weights-inception-2015-12-05-6726825d.pth",
    "vae_checkpoint": REPO_ROOT / "public/vae/mar-kl16/kl16.ckpt",
    "flow_latent_cache": REPO_ROOT
    / "public/datasets/imagenet_ablation_100c_balanced/vae_latents_mar_kl16/"
    "flow_latents_100c_1250pc_fp16.pt",
    "manifest": REPO_ROOT
    / "public/datasets/imagenet_ablation_100c_balanced/manifest.jsonl",
    "split_manifest": REPO_ROOT
    / "public/datasets/imagenet_ablation_100c_balanced/split_seed42_val100.jsonl",
    "synset_mapping": REPO_ROOT / "public/datasets/imagenet/LOC_synset_mapping.txt",
}
REQUIRED_CHECKPOINT_SIDECARS = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
)
SOURCE_FILES = (
    "scripts/evaluate_single_stream_fid_is.py",
    "scripts/evaluate_qwen_showo_fid_is.py",
    "scripts/generate_flow_validation_images.py",
    "script/ablation/evaluate_imagenet_flow_100c.sh",
    "script/ablation/evaluate_imagenet_flow_cfg_sweep_100c.sh",
    "scripts/ensure_flow_cfg_sweep_contract.py",
    "scripts/validate_flow_cfg_metrics.py",
    "models/modeling_model/modeling_selfless_flow.py",
    "models/modeling_model/image_flow_loss.py",
    "utils/utils.py",
    "utils/dataset_utils.py",
    "utils/dataset_imagenet_flow_cache.py",
)
EXTERNAL_SOURCE_FILES = {
    "mar/models/vae.py": Path(
        "/inspire/hdd/global_user/wanjiaxin-253108030048/code/mar/models/vae.py"
    )
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def checked_sha256(label: str, path: Path) -> str:
    try:
        return sha256_file(path)
    except OSError as exc:
        raise SystemExit(f"cannot hash {label} at {path}: {exc}") from exc


def checkpoint_sidecar_sha256(checkpoint_dir: Path) -> dict[str, str]:
    """Hash every checkpoint file other than the separately bound model weights."""
    for filename in REQUIRED_CHECKPOINT_SIDECARS:
        path = checkpoint_dir / filename
        if not path.is_file():
            raise SystemExit(f"missing required checkpoint sidecar: {path}")

    sidecars = sorted(
        path
        for path in checkpoint_dir.rglob("*")
        if path.is_file()
        and path.relative_to(checkpoint_dir).as_posix() != "model.safetensors"
    )
    return {
        path.relative_to(checkpoint_dir).as_posix(): checked_sha256(
            f"checkpoint sidecar {path.relative_to(checkpoint_dir).as_posix()}", path
        )
        for path in sidecars
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--model-dtype", choices=("bf16", "fp32"), required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument("--sampling-steps", required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--cfg-schedule", required=True)
    parser.add_argument("--flow-solver", required=True)
    parser.add_argument("--parallel-rate", type=int, required=True)
    parser.add_argument("--strategies", required=True)
    parser.add_argument("--save-images", type=int, choices=(0, 1), required=True)
    parser.add_argument(
        "--allow-legacy-root",
        action="store_true",
        help="Initialize a non-empty root that predates sweep contracts.",
    )
    return parser.parse_args()


def atomic_write_json(path: Path, payload):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    args = parse_args()
    if not SHA256_RE.fullmatch(args.model_sha256):
        raise SystemExit("--model-sha256 must contain exactly 64 hexadecimal characters")
    checkpoint_dir = resolve_repo_path(args.model_path)
    model_file = checkpoint_dir / "model.safetensors"
    actual_model_sha256 = checked_sha256("checkpoint weights", model_file)
    if actual_model_sha256 != args.model_sha256.lower():
        raise SystemExit(
            f"checkpoint SHA256 mismatch for {model_file}: "
            f"{actual_model_sha256} != {args.model_sha256.lower()}"
        )
    strategies = sorted(
        value for value in re.split(r"[\s,]+", args.strategies.strip()) if value
    )
    if not strategies:
        raise SystemExit("--strategies must not be empty")

    config_file = resolve_repo_path(args.config)
    artifact_sha256 = {
        "config": checked_sha256("config", config_file),
        **{
            label: checked_sha256(label, path.resolve())
            for label, path in FIXED_ARTIFACTS.items()
        },
    }
    source_sha256 = {
        relative_path: checked_sha256(
            f"source {relative_path}", (REPO_ROOT / relative_path).resolve()
        )
        for relative_path in SOURCE_FILES
    }
    source_sha256.update(
        {
            label: checked_sha256(f"source {label}", path.resolve())
            for label, path in EXTERNAL_SOURCE_FILES.items()
        }
    )
    sidecar_sha256 = checkpoint_sidecar_sha256(checkpoint_dir)

    contract = {
        "schema": SCHEMA,
        "model_path": args.model_path,
        "model_sha256": args.model_sha256.lower(),
        "model_dtype": args.model_dtype,
        "precision_policy": {
            "schema": "flow_eval_precision_v1",
            "vae_dtype": "fp32",
            "flow_integrator_dtype": "fp32",
            "autocast_enabled": False,
            "cuda_matmul_allow_tf32": False,
            "cudnn_allow_tf32": True,
            "float32_matmul_precision": "highest",
        },
        "checkpoint_sidecar_sha256": sidecar_sha256,
        "config": args.config,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "samples": args.samples,
        "sampling_steps": str(args.sampling_steps),
        "temperature": args.temperature,
        "cfg_schedule": args.cfg_schedule,
        "flow_solver": args.flow_solver,
        "parallel_rate": args.parallel_rate,
        "strategies": strategies,
        "save_images": bool(args.save_images),
        "artifact_sha256": artifact_sha256,
        "source_sha256": source_sha256,
    }

    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    path = root / ".sweep_contract.json"
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit(f"invalid existing sweep contract {path}: {exc}") from exc
        if existing != contract:
            changed = sorted(
                key
                for key in set(existing) | set(contract)
                if existing.get(key) != contract.get(key)
            )
            raise SystemExit(
                f"sweep contract mismatch at {path}; changed fields: "
                + ", ".join(changed)
            )
        print(f"Validated sweep contract: {path}")
        return

    legacy_entries = sorted(
        entry.name
        for entry in root.iterdir()
        if entry.name not in {".sweep_contract.lock"}
    )
    if legacy_entries and not args.allow_legacy_root:
        preview = ", ".join(legacy_entries[:8])
        raise SystemExit(
            f"refusing to initialize non-empty legacy sweep root {root}; "
            f"existing entries: {preview}. Use a new root or explicitly pass "
            "--allow-legacy-root after validating its contents."
        )

    atomic_write_json(path, contract)
    print(f"Created sweep contract: {path}")


if __name__ == "__main__":
    main()
