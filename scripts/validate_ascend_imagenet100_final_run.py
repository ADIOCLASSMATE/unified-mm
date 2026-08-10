#!/usr/bin/env python3
"""Strict final acceptance for the canonical 16-NPU ImageNet-100 run."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from models.modeling_model.modeling_selfless_flow import Qwen3ForCausalLM
from scripts.validate_ascend_imagenet100_assets import (
    EXPECTED_HASHES,
    load_membership,
    require_hash,
    validate_cache,
    validate_config,
    validate_split,
)

EXPECTED_PARAMETERS = 761_189_904
EXPECTED_TOKENIZER_SIZE = 151_673
EXPECTED_FINAL_STEP = 17_920
EXPECTED_WORLD_SIZE = 16
EXPECTED_GRADIENT_ACCUMULATION = 2
EXPECTED_GLOBAL_BATCH = 512
EXPECTED_LOSS_MICROBATCH_CHECKS = EXPECTED_FINAL_STEP * EXPECTED_GRADIENT_ACCUMULATION


def ordered_rng_state_paths(checkpoint: Path) -> list[Path]:
    """Return RNG shards in numeric rank order after validating their names."""
    ranked_paths = []
    for path in checkpoint.glob("random_states_*.pkl"):
        match = re.fullmatch(r"random_states_(0|[1-9]\d*)\.pkl", path.name)
        if match is None:
            raise RuntimeError(f"invalid RNG shard name: {path.name}")
        ranked_paths.append((int(match.group(1)), path))
    ranked_paths.sort(key=lambda item: item[0])
    actual_ranks = [rank for rank, _ in ranked_paths]
    expected_ranks = list(range(EXPECTED_WORLD_SIZE))
    if actual_ranks != expected_ranks:
        raise RuntimeError(
            "non-contiguous RNG shard ranks: "
            f"{actual_ranks} != {expected_ranks}"
        )
    return [path for _, path in ranked_paths]


def require_file(path: Path, label: str, *, minimum_bytes: int = 1) -> Path:
    if not path.is_file() or path.stat().st_size < minimum_bytes:
        raise RuntimeError(f"missing or truncated {label}: {path}")
    return path


def load_json(path: Path, label: str) -> dict:
    return json.loads(require_file(path, label).read_text(encoding="utf-8"))


def validate_assets_and_config(run_root: Path) -> dict:
    membership_path = Path(
        "public/datasets/imagenet_ablation_100c_balanced/manifest.jsonl"
    )
    split_path = Path(
        "public/datasets/imagenet_ablation_100c_balanced/split_seed42_val100.jsonl"
    )
    cache_path = Path(
        "public/datasets/imagenet_ablation_100c_balanced/"
        "vae_posterior_mar_kl16/posterior_stats_100c_1250pc_fp16.pt"
    )
    hashes = {
        "qwen_weights": require_hash(
            Path("public/models/Qwen--Qwen3-0.6B-Base/model.safetensors"),
            EXPECTED_HASHES["qwen_weights"],
            "Qwen weights",
        ),
        "vae_module": require_hash(
            Path("public/code/mar/models/vae.py"),
            EXPECTED_HASHES["vae_module"],
            "MAR VAE module",
        ),
        "vae_checkpoint": require_hash(
            Path("public/vae/mar-kl16/kl16.ckpt"),
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
    membership, _ = load_membership(membership_path)
    split_counts = validate_split(split_path, membership)
    cache_metadata = validate_cache(cache_path, membership, scan_values=True)
    config = OmegaConf.load(require_file(run_root / "config.yaml", "run config"))
    training_contract = validate_config(config, EXPECTED_WORLD_SIZE, split_counts)
    return {
        "hashes": hashes,
        "cache_format": cache_metadata["format"],
        "training_contract": training_contract,
    }


def validate_runtime(run_root: Path) -> dict:
    runtime = load_json(
        run_root / "training_runtime_metrics.json",
        "latest runtime metrics",
    )
    expected = {
        "global_step": EXPECTED_FINAL_STEP,
        "world_size": EXPECTED_WORLD_SIZE,
        "total_batch_size": EXPECTED_GLOBAL_BATCH,
        "memory_backend": "npu",
    }
    for field, value in expected.items():
        if runtime.get(field) != value:
            raise RuntimeError(
                f"runtime {field} mismatch: {runtime.get(field)!r} != {value!r}"
            )
    required_positive = (
        "cumulative_training_wall_seconds",
        "train_samples_per_second",
        "peak_memory_allocated_bytes_per_rank",
        "peak_memory_reserved_bytes_per_rank",
    )
    for field in required_positive:
        value = float(runtime.get(field, 0))
        if not math.isfinite(value) or value <= 0:
            raise RuntimeError(f"runtime {field} must be finite and positive: {value}")
    final_loss = float(runtime.get("last_logged_loss", math.nan))
    if not math.isfinite(final_loss):
        raise RuntimeError(f"final logged loss is non-finite: {final_loss}")
    loss_checks = int(runtime.get("cumulative_finite_loss_microbatches_checked", -1))
    if loss_checks != EXPECTED_LOSS_MICROBATCH_CHECKS:
        raise RuntimeError(
            "finite loss coverage mismatch: "
            f"{loss_checks} != {EXPECTED_LOSS_MICROBATCH_CHECKS}"
        )
    trainability = runtime.get("trainability", {})
    if (
        int(trainability.get("total_numel", -1)) != EXPECTED_PARAMETERS
        or int(trainability.get("trainable_numel", -1)) != EXPECTED_PARAMETERS
        or int(trainability.get("frozen_numel", -1)) != 0
    ):
        raise RuntimeError(f"unexpected final trainability: {trainability}")
    return runtime


def validate_checkpoint(run_root: Path) -> dict:
    checkpoint = run_root / f"checkpoint-{EXPECTED_FINAL_STEP}"
    completion = load_json(
        checkpoint / "checkpoint_complete.json",
        "final checkpoint completion marker",
    )
    expected_completion = {
        "schema": "selfless_caption_checkpoint_complete_v1",
        "global_step": EXPECTED_FINAL_STEP,
    }
    if completion != expected_completion:
        raise RuntimeError(
            f"checkpoint completion mismatch: {completion} != {expected_completion}"
        )
    metadata = load_json(checkpoint / "metadata.json", "final checkpoint metadata")
    required_metadata = {
        "schema": "selfless_caption_training_checkpoint_v3",
        "global_step": EXPECTED_FINAL_STEP,
        "world_size": EXPECTED_WORLD_SIZE,
        "gradient_accumulation_steps": EXPECTED_GRADIENT_ACCUMULATION,
        "config_signature_version": 3,
        "cumulative_finite_loss_microbatches_checked": (
            EXPECTED_LOSS_MICROBATCH_CHECKS
        ),
    }
    for field, value in required_metadata.items():
        if metadata.get(field) != value:
            raise RuntimeError(
                f"checkpoint metadata {field} mismatch: "
                f"{metadata.get(field)!r} != {value!r}"
            )
    if float(metadata.get("cumulative_training_wall_seconds", 0)) <= 0:
        raise RuntimeError("checkpoint has no cumulative training wall time")
    if len(str(metadata.get("config_signature", ""))) != 64:
        raise RuntimeError("checkpoint config signature is missing")
    if len(str(metadata.get("ema_layout_fingerprint", ""))) != 64:
        raise RuntimeError("checkpoint EMA layout fingerprint is missing")

    model_root = checkpoint / "pytorch_model"
    require_file(
        model_root / "mp_rank_00_model_states.pt",
        "DeepSpeed model state",
        minimum_bytes=1_000_000_000,
    )
    optimizer_shards = sorted(model_root.glob("*optim_states.pt"))
    if len(optimizer_shards) != EXPECTED_WORLD_SIZE:
        raise RuntimeError(
            f"optimizer shard count mismatch: {len(optimizer_shards)} != 16"
        )
    for path in optimizer_shards:
        require_file(path, "DeepSpeed optimizer shard", minimum_bytes=100_000_000)
    require_file(checkpoint / "scheduler.bin", "scheduler state")

    rng_files = ordered_rng_state_paths(checkpoint)
    for path in rng_files:
        state = torch.load(str(path), map_location="cpu", weights_only=False)
        npu_state = state.get("torch_npu_manual_seed")
        required_keys = {
            "random_state",
            "numpy_random_seed",
            "torch_manual_seed",
            "torch_npu_manual_seed",
        }
        if not required_keys.issubset(state):
            raise RuntimeError(f"RNG state keys missing in {path}")
        if not isinstance(npu_state, torch.Tensor) or npu_state.dtype != torch.uint8:
            raise RuntimeError(f"invalid NPU RNG state in {path}")

    ema_manifest = load_json(checkpoint / "ema_manifest.json", "EMA manifest")
    if (
        ema_manifest.get("schema") != "selfless_rank_sharded_fp32_ema_v1"
        or int(ema_manifest.get("world_size", -1)) != EXPECTED_WORLD_SIZE
        or ema_manifest.get("layout_fingerprint")
        != metadata.get("ema_layout_fingerprint")
    ):
        raise RuntimeError("final checkpoint EMA manifest contract failed")
    ema_shards = sorted(checkpoint.glob("ema_shard_rank_*.safetensors"))
    if len(ema_shards) != EXPECTED_WORLD_SIZE:
        raise RuntimeError(f"EMA shard count mismatch: {len(ema_shards)} != 16")
    for path in ema_shards:
        require_file(path, "EMA shard", minimum_bytes=100_000_000)
    return {
        "path": str(checkpoint.resolve()),
        "optimizer_shards": len(optimizer_shards),
        "rng_shards": len(rng_files),
        "ema_shards": len(ema_shards),
        "metadata": metadata,
    }


def validate_logs(run_root: Path) -> dict:
    audit_dirs = sorted(
        path for path in run_root.glob("prelaunch_audit*") if path.is_dir()
    )
    if not audit_dirs:
        raise RuntimeError("no prelaunch audit directories found")
    log_paths = []
    combined = ""
    for directory in audit_dirs:
        require_file(directory / "asset_preflight.json", "asset preflight")
        require_file(directory / "git_commit.txt", "git commit snapshot")
        require_file(
            directory / "git_diff.patch",
            "git diff snapshot",
            minimum_bytes=0,
        )
        require_file(directory / "npu_smi.txt", "npu-smi snapshot")
        require_file(directory / "software_versions.json", "software versions")
        require_file(directory / "launch_command.sh", "launch command")
        log_path = require_file(directory / "training.log", "training log")
        log_paths.append(str(log_path.resolve()))
        combined += log_path.read_text(encoding="utf-8", errors="replace")
    required_log_fragments = (
        "DistributedType.DEEPSPEED  Backend: hccl",
        f"Num processes: {EXPECTED_WORLD_SIZE}",
        "'grad_accum_dtype': 'fp32'",
        "Exact epoch contract: samples=114688, "
        "prepared_microbatches/rank=448, gradient_accumulation=2, "
        "optimizer_steps/epoch=224, epochs=80",
        f"Step: {EXPECTED_FINAL_STEP} | Loss:",
    )
    missing = [
        fragment for fragment in required_log_fragments if fragment not in combined
    ]
    if missing:
        raise RuntimeError(f"formal training logs are missing: {missing}")
    return {"audit_directories": len(audit_dirs), "training_logs": log_paths}


def validate_formal_gate(run_root: Path, audit_root: Path) -> dict:
    gate_parent = audit_root / "formal-gates"
    candidates = sorted(gate_parent.glob(f"{run_root.name}-*"))
    if not candidates:
        raise RuntimeError(f"no formal 16-rank gate audit found under {gate_parent}")
    expected = (
        "PASS world=16 memory_dtype=torch.int64 elapsed_dtype=torch.float32 "
        "window_dtype=torch.float32 hccl_all_reduce_sum=136"
    )
    for gate_dir in reversed(candidates):
        log_path = gate_dir / "hccl_gate.log"
        if not log_path.is_file():
            continue
        content = log_path.read_text(encoding="utf-8", errors="replace")
        if expected in content:
            require_file(gate_dir / "npu_smi_before_hccl.txt", "formal NPU snapshot")
            require_file(gate_dir / "hccl_gate_command.sh", "formal HCCL command")
            require_file(gate_dir / "tmux_launch_command.sh", "formal tmux command")
            require_file(gate_dir / "tmux_session.txt", "formal tmux session")
            return {
                "path": str(gate_dir.resolve()),
                "hccl_contract": expected,
            }
    raise RuntimeError("no formal gate audit contains the exact 16-rank HCCL pass")


def offline_reload(path: Path, expected_dtype: torch.dtype) -> dict:
    required_names = (
        "config.json",
        "generation_config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
    )
    for name in required_names:
        require_file(
            path / name,
            f"HF artifact {name}",
            minimum_bytes=1_000_000_000 if name == "model.safetensors" else 1,
        )
    config = load_json(path / "config.json", "HF model config")
    expected_name = str(expected_dtype).removeprefix("torch.")
    if config.get("dtype") != expected_name:
        raise RuntimeError(
            f"HF dtype mismatch for {path}: {config.get('dtype')} != {expected_name}"
        )
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    if len(tokenizer) != EXPECTED_TOKENIZER_SIZE:
        raise RuntimeError(f"tokenizer size mismatch for {path}: {len(tokenizer)}")
    model = Qwen3ForCausalLM.from_pretrained(
        path,
        local_files_only=True,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
    )
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    dtypes = {parameter.dtype for parameter in model.parameters()}
    if total_parameters != EXPECTED_PARAMETERS or dtypes != {expected_dtype}:
        raise RuntimeError(
            f"offline model contract failed for {path}: "
            f"parameters={total_parameters}, dtypes={dtypes}"
        )
    for name, parameter in model.named_parameters():
        if not bool(torch.isfinite(parameter).all()):
            raise RuntimeError(f"non-finite offline parameter: {path}/{name}")
    result = {
        "path": str(path.resolve()),
        "parameters": total_parameters,
        "dtype": expected_name,
        "tokenizer_size": len(tokenizer),
        "all_parameters_finite": True,
    }
    del model, tokenizer
    gc.collect()
    return result


def write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run_root",
        default=("output/selfless-flow-im100-class-ascend16-b16ga2-b4e5-f1e4"),
    )
    parser.add_argument("--output_json", default=None)
    parser.add_argument(
        "--audit_root",
        default=(
            "/inspire/sj-ssd3/project/high-dimensionaldata/"
            "wanjiaxin-253108030048/npu-parity-audit"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root)
    if not run_root.is_dir():
        raise FileNotFoundError(f"formal run root does not exist: {run_root}")
    report = {
        "schema": "ascend_imagenet100_16npu_final_acceptance_v1",
        "status": "ok",
        "run_root": str(run_root.resolve()),
        "assets_and_config": validate_assets_and_config(run_root),
        "runtime": validate_runtime(run_root),
        "checkpoint": validate_checkpoint(run_root),
        "formal_gate": validate_formal_gate(run_root, Path(args.audit_root)),
        "logs": validate_logs(run_root),
        "final_model": offline_reload(run_root / "hf_model-final", torch.bfloat16),
        "ema_model": offline_reload(run_root / "hf_model-final-ema", torch.float32),
    }
    if args.output_json:
        write_report(Path(args.output_json), report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
