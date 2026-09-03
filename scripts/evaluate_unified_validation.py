#!/usr/bin/env python3
"""Run held-out validation from a final HF EMA or retained sharded EMA."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import sys

os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("DIFFUSERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import torch
from accelerate import Accelerator, DataLoaderConfiguration
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pretrain.train_selfless_flow import validate  # noqa: E402
from utils.combined_dataloaders import (  # noqa: E402
    build_unified_image_validation_dataloader,
)
from utils.evaluation_model_source import (  # noqa: E402
    add_model_source_argument,
    configure_model_source,
    load_model_source_weights,
    model_source_from_args,
    resolve_evaluation_model_source,
)
from utils.flow_head_contract import (  # noqa: E402
    flow_head_attention_report,
    validate_flow_head_attention_contract,
)
from utils.utils import load_model_tokenizer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/selfless/unified_baseline_100b_ascend_64npu.yaml"),
    )
    add_model_source_argument(parser)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--validation_max_batches", type=int, default=4)
    parser.add_argument("--batch_size_per_rank", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--image_samples", type=int, default=2)
    parser.add_argument("--i2t_samples", type=int, default=2)
    parser.add_argument("--i2t_max_new_tokens", type=int, default=64)
    parser.add_argument("--model_dtype", choices=("bf16", "fp32"), default="bf16")
    return parser.parse_args()


def read_checkpoint_step(checkpoint: Path) -> int:
    return resolve_evaluation_model_source(checkpoint).global_step


def main() -> None:
    args = parse_args()
    if args.validation_max_batches < 0:
        raise ValueError("--validation_max_batches must be non-negative")
    if args.batch_size_per_rank <= 0 or args.num_workers < 0:
        raise ValueError("invalid dataloader settings")
    source = model_source_from_args(args)
    global_step = source.global_step
    config = OmegaConf.load(args.config)
    if str(config.dataset.class_name) != "UnifiedMixedDataset":
        raise ValueError("unified validation requires UnifiedMixedDataset config")
    image_contract = config.dataset.params.image
    if str(image_contract.get("expected_split", "")) != "train":
        raise ValueError("evaluation config training split is not ImageNet train")
    if str(image_contract.validation.get("expected_split", "")) != "val":
        raise ValueError("evaluation config validation split is not ImageNet val")
    config.training.runtime_hashing_enabled = False
    configure_model_source(config, source)
    attention_contract = str(
        config.model.get(
            "dual_stream_attention_contract",
            "selfless_strict",
        )
    ).strip().lower()
    if attention_contract not in {
        "selfless_strict",
        "xlnet_content_diagonal",
    }:
        raise ValueError(
            "unsupported dual-stream attention contract: "
            f"{attention_contract!r}"
        )
    flow_head_attention_contract = validate_flow_head_attention_contract(
        config.model,
        label="evaluation config",
    )
    config.experiment.output_dir = str(args.output_dir)
    config.experiment.validation_max_batches = int(args.validation_max_batches)
    config.experiment.validation_image_every = 1
    config.experiment.validation_i2t_every = 1
    config.experiment.validation_image_samples = int(args.image_samples)
    config.experiment.validation_i2t_samples = int(args.i2t_samples)
    config.experiment.validation_i2t_max_new_tokens = int(args.i2t_max_new_tokens)

    accelerator = Accelerator(
        mixed_precision=("bf16" if args.model_dtype == "bf16" else "no"),
        dataloader_config=DataLoaderConfiguration(non_blocking=True),
    )
    model_dtype = torch.bfloat16 if args.model_dtype == "bf16" else torch.float32
    model, tokenizer = load_model_tokenizer(config, model_dtype=model_dtype)
    loaded_attention_contract = str(
        getattr(
            model.config,
            "dual_stream_attention_contract",
            "selfless_strict",
        )
    ).strip().lower()
    if loaded_attention_contract != attention_contract:
        raise ValueError(
            "loaded model attention contract does not match evaluation config: "
            f"model={loaded_attention_contract!r}, config={attention_contract!r}"
        )
    loaded_flow_head_attention_contract = validate_flow_head_attention_contract(
        model.config,
        label="loaded model",
    )
    if loaded_flow_head_attention_contract != flow_head_attention_contract:
        raise ValueError(
            "loaded model flow-head attention contract does not match "
            "evaluation config: "
            f"model={loaded_flow_head_attention_contract!r}, "
            f"config={flow_head_attention_contract!r}"
        )
    weight_report = load_model_source_weights(model, source)
    if int(weight_report["global_step"]) != global_step:
        raise ValueError(
            f"loaded model step={weight_report['global_step']} != source step={global_step}"
        )
    model.to(accelerator.device).eval()
    val_loader = build_unified_image_validation_dataloader(
        config,
        tokenizer,
        task_modes=("t2i", "i2t"),
        batch_size=int(args.batch_size_per_rank),
        num_workers=int(args.num_workers),
    )
    val_loader = accelerator.prepare_data_loader(val_loader)
    accelerator.wait_for_everyone()
    validate(
        model,
        val_loader,
        accelerator,
        global_step,
        config=config,
        tokenizer=tokenizer,
    )
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        metrics_path = args.output_dir / f"validation_metrics_step_{global_step}.json"
        if not metrics_path.is_file():
            raise FileNotFoundError(metrics_path)
        generation_path = (
            args.output_dir
            / f"validation_generation_step_{global_step}.json"
        )
        if args.image_samples > 0:
            if not generation_path.is_file():
                raise FileNotFoundError(generation_path)
            generation = json.loads(
                generation_path.read_text(encoding="utf-8")
            )
            if (
                generation.get("generation_entry") != "model.generate"
                or generation.get("use_cache") is not True
            ):
                raise RuntimeError(
                    "held-out image generation did not use the unified "
                    "cached entry"
                )
        else:
            generation = None
        payload = {
            "schema": "unified_checkpoint_validation_run_v2",
            "complete": True,
            "runtime_hashing_enabled": False,
            "checkpoint": str(source.path),
            "weight_source": source.kind,
            "model_source": weight_report,
            "global_step": global_step,
            "dual_stream_attention_contract": attention_contract,
            "flow_head_attention_contract": flow_head_attention_contract,
            "backbone_attention": {
                "dual_stream_attention_contract": attention_contract,
                "query_stream_diagonal": False,
                "content_stream_diagonal": (
                    attention_contract == "xlnet_content_diagonal"
                ),
                "single_stream_visible_content_diagonal": (
                    attention_contract == "xlnet_content_diagonal"
                ),
                "single_stream_current_query_diagonal": False,
            },
            "flow_head_attention": flow_head_attention_report(config.model),
            "imagenet_split": "val",
            "world_size": int(accelerator.num_processes),
            "batch_size_per_rank": int(args.batch_size_per_rank),
            "validation_max_batches": int(args.validation_max_batches),
            "metrics": str(metrics_path.resolve()),
            "generation": (
                str(generation_path.resolve())
                if generation is not None
                else None
            ),
            "completed_at": datetime.now(timezone.utc).isoformat().replace(
                "+00:00", "Z"
            ),
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "evaluation_run.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
