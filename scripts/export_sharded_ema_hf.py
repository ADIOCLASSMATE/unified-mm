#!/usr/bin/env python3
"""Export a rank-sharded FP32 EMA checkpoint as an HF model on CPU."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.sharded_ema import (  # noqa: E402
    load_ema_manifest,
    mark_hf_ema_config_fp32,
    merge_sharded_ema_state_dict,
)
from utils.utils import load_model_tokenizer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge rank-sharded EMA files on CPU and save a compatible HF model. "
            "The output weights remain FP32."
        )
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--ema_dir", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.config.is_file():
        raise FileNotFoundError(args.config)
    if args.output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing EMA HF export: {args.output_dir}"
        )

    manifest = load_ema_manifest(args.ema_dir)
    config = OmegaConf.load(args.config)
    model, tokenizer = load_model_tokenizer(config, model_dtype=torch.bfloat16)

    model_keys = list(model.state_dict().keys())
    manifest_keys = list(manifest["state_keys"])
    if model_keys != manifest_keys:
        missing = sorted(set(manifest_keys) - set(model_keys))
        unexpected = sorted(set(model_keys) - set(manifest_keys))
        raise RuntimeError(
            "EMA/model state mismatch before export: "
            f"missing_from_model={missing}, unexpected_in_model={unexpected}, "
            "order_matches=False"
        )

    print(
        f"Merging {len(manifest_keys)} EMA state entries from "
        f"world_size={manifest['world_size']} on CPU...",
        flush=True,
    )
    merged_state = merge_sharded_ema_state_dict(args.ema_dir)
    if any(tensor.device.type != "cpu" for tensor in merged_state.values()):
        raise RuntimeError("EMA merge unexpectedly created a non-CPU tensor")
    floating_dtypes = {
        str(tensor.dtype) for tensor in merged_state.values() if tensor.is_floating_point()
    }
    if floating_dtypes != {"torch.float32"}:
        raise RuntimeError(f"EMA floating dtypes are not strictly FP32: {floating_dtypes}")

    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(
        args.output_dir,
        state_dict=merged_state,
        safe_serialization=True,
    )
    mark_hf_ema_config_fp32(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    runtime = manifest.get("runtime") or {}
    metadata = {
        "source_ema_dir": str(args.ema_dir),
        "source_global_step": runtime.get("global_step"),
        "source_world_size": manifest["world_size"],
        "layout_fingerprint": manifest["layout_fingerprint"],
        "state_key_count": len(merged_state),
        "floating_dtype": "float32",
        "merge_device": "cpu",
    }
    (args.output_dir / "ema_export_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"Exported step {runtime.get('global_step')} FP32 EMA HF model to "
        f"{args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
