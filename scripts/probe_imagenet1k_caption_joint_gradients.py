#!/usr/bin/env python3
"""Measure task-separated caption/T2I gradients on a fixed holdout prefix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import Counter
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("DIFFUSERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

try:
    import tbe  # noqa: F401
except ImportError:
    pass

import torch
import torch_npu  # noqa: F401
from omegaconf import OmegaConf

from utils.dataset_utils import get_dataloaders
from utils.joint_gradient_probe import (
    measure_gradient_probe_batch,
    summarize_probe_batches,
)
from utils.selfless_flow_optimizer import (
    learning_rate_for_parameter,
    optimizer_parameter_role,
)
from utils.utils import get_selfless_mask, load_model_tokenizer


DEFAULT_CONFIG = Path(
    "configs/selfless/imagenet1k_caption_joint_sweep_10ep_ascend16_b1024.yaml"
)
DEFAULT_CHECKPOINT = Path(
    "output/selfless-flow-imagenet1k-class-ascend64-b1024-800ep/"
    "hf_model-final-ema"
)
DEFAULT_OUTPUT = Path(
    "output/selfless-flow-imagenet1k-caption-joint-gradient-probe/"
    "init/probe.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num_batches", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=424242)
    parser.add_argument("--device", default="npu:0")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _reset_seed(seed: int, device: torch.device) -> None:
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if device.type == "npu":
        torch.npu.manual_seed_all(int(seed))


def _optimizer_inventory(model, config) -> dict[str, Any]:
    params = config.optimizer.params
    backbone_lr = float(params.backbone_learning_rate)
    flow_lr = float(params.flow_learning_rate)
    projector_lr = float(params.projector_learning_rate)
    special_token_lr = float(params.special_token_learning_rate)
    roles: dict[str, dict[str, Any]] = {}
    seen: set[int] = set()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or id(parameter) in seen:
            continue
        seen.add(id(parameter))
        role = optimizer_parameter_role(name)
        row = roles.setdefault(
            role,
            {
                "parameter_count": 0,
                "numel": 0,
                "learning_rates": set(),
                "example_names": [],
            },
        )
        row["parameter_count"] += 1
        row["numel"] += int(parameter.numel())
        row["learning_rates"].add(
            learning_rate_for_parameter(
                name,
                backbone_lr=backbone_lr,
                flow_lr=flow_lr,
                projector_lr=projector_lr,
                special_token_lr=special_token_lr,
            )
        )
        if len(row["example_names"]) < 4:
            row["example_names"].append(name)
    for row in roles.values():
        row["learning_rates"] = sorted(row["learning_rates"])
    tied = model.lm_head.weight is model.model.embed_tokens.weight
    return {
        "roles": roles,
        "lm_head_embed_tokens_tied": bool(tied),
        "special_token_lr_scope": (
            "complete_tied_lm_head_embedding_matrix" if tied else "embedding_matrix"
        ),
    }


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=False)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in batch.items()
    }


def main() -> None:
    args = parse_args()
    if not 16 <= int(args.num_batches) <= 32:
        raise ValueError(f"--num_batches must be in [16, 32], got {args.num_batches}")
    if not args.config.is_file():
        raise FileNotFoundError(args.config)
    for filename in ("model.safetensors", "config.json", "tokenizer.json"):
        if not (args.checkpoint / filename).is_file():
            raise FileNotFoundError(args.checkpoint / filename)
    if not torch.npu.is_available():
        raise RuntimeError("Ascend NPU is required for the formal gradient probe")

    device = torch.device(args.device)
    torch.npu.set_device(device)
    config = OmegaConf.load(args.config)
    config.model.model_path = str(args.checkpoint)
    if args.batch_size is not None:
        if args.batch_size <= 0:
            raise ValueError("--batch_size must be positive")
        config.training.batch_size = int(args.batch_size)
    batch_size = int(config.training.batch_size)
    if str(config.training.mixed_precision).lower() != "bf16":
        raise RuntimeError("formal probe requires the training bf16 contract")
    if str(config.training.gradient_accumulation_dtype).lower() != "fp32":
        raise RuntimeError("formal probe requires the training FP32 accumulation contract")

    _reset_seed(args.seed, device)
    model, tokenizer = load_model_tokenizer(
        config=config,
        model_dtype=torch.bfloat16,
    )
    model = model.to(device=device).train()
    if float(model.lambda_text) <= 0.0 or float(model.lambda_image) <= 0.0:
        raise RuntimeError("probe requires both model task losses to be enabled")
    _, val_dataloader = get_dataloaders(config, tokenizer)
    val_dataset = val_dataloader.dataset
    val_indices = list(getattr(val_dataset, "indices", []))
    split_digest = hashlib.sha256()
    for index in val_indices:
        split_digest.update(f"{int(index)}\n".encode("ascii"))

    special_token_ids = [
        int(config.model.mask_token_id),
        int(config.model.boi_token_id),
        int(config.model.eoi_token_id),
    ]
    if config.model.get("image_mask_token_id", None) is not None:
        special_token_ids.append(int(config.model.image_mask_token_id))

    report: dict[str, Any] = {
        "schema": "selfless_caption_t2i_gradient_probe_v1",
        "status": "running",
        "config": str(args.config),
        "checkpoint": {
            "path": str(args.checkpoint),
            "model_sha256": sha256_file(args.checkpoint / "model.safetensors"),
            "config_sha256": sha256_file(args.checkpoint / "config.json"),
        },
        "contract": {
            "optimizer_steps": 0,
            "model_dtype": "bfloat16",
            "gradient_metric_accumulation_dtype": "float32",
            "loss_normalization": {
                "text": (
                    "sum token cross-entropy / valid I2T target-token count; "
                    "token_types in {text,special} and labels != -100"
                ),
                "image": (
                    "mean latent-channel flow MSE / valid T2I image-token "
                    "count after image_flow_batch_mul"
                ),
            },
            "seed": int(args.seed),
            "num_batches": int(args.num_batches),
            "batch_size": batch_size,
            "holdout_split_size": len(val_indices),
            "holdout_indices_sha256": split_digest.hexdigest(),
            "caption_t2i_sampling": list(config.dataset.params.caption_sequence_modes),
            "image_flow_batch_mul": int(config.model.image_flow_batch_mul),
            "image_uncond_prob": float(config.model.image_uncond_prob),
            "special_token_ids": special_token_ids,
        },
        "optimizer_parameter_inventory": _optimizer_inventory(model, config),
        "batches": [],
    }
    _atomic_write(args.output, report)

    for loader_batch_index, host_batch in enumerate(val_dataloader):
        if len(report["batches"]) >= args.num_batches:
            break
        batch = _move_batch(host_batch, device)
        task_counts = Counter(map(str, batch["task_modes"]))
        # The fixed holdout order overwhelmingly yields both modes per batch;
        # skip a pathological single-task batch because both task gradients
        # must be measured on identical examples.
        if not {"i2t", "t2i"}.issubset(task_counts):
            continue
        input_ids = batch["input_ids"].contiguous()
        token_types = batch["token_types"]
        sigma = batch["sigma"]
        labels = batch["labels"]
        image_loss_mask = batch["image_loss_mask"].bool()
        image_span_table = batch["image_span_table"]
        image_latents = batch["image_latents"]
        position_ids = batch.get("position_ids")
        image_local_positions = batch.get("image_local_positions")
        batch_seed = int(args.seed) + loader_batch_index
        cfg_generator = torch.Generator(device="cpu")
        cfg_generator.manual_seed(batch_seed + 10_000_000)
        image_uncond_rows = (
            torch.rand(input_ids.shape[0], generator=cfg_generator)
            < float(config.model.image_uncond_prob)
        ).to(device=device)
        attention_mask = get_selfless_mask(
            sigma=sigma,
            seq_len=input_ids.shape[1],
            device=device,
            input_ids=input_ids,
            token_types=token_types,
            boi_token_id=int(config.model.boi_token_id),
            image_uncond_rows=image_uncond_rows,
        )

        def forward_losses():
            output = model(
                X0_input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask,
                position_ids=position_ids,
                token_types=token_types,
                image_latents=image_latents,
                image_local_positions=image_local_positions,
                image_span_table=image_span_table,
                image_loss_mask=image_loss_mask,
                flow_sigma=sigma,
                calculate_likelihood=True,
                record_flow_stats=False,
                return_per_modality_loss_graph=True,
                use_cache=False,
            )
            return output.per_modality_loss_graph, output.per_modality_count

        metrics = measure_gradient_probe_batch(
            model,
            forward_losses,
            special_token_ids=special_token_ids,
            reset_seed=lambda: _reset_seed(batch_seed, device),
        )
        metrics.update(
            {
                "probe_batch_index": len(report["batches"]),
                "loader_batch_index": loader_batch_index,
                "seed": batch_seed,
                "task_counts": dict(sorted(task_counts.items())),
                "image_uncond_rows": int(image_uncond_rows.sum().cpu()),
                "img_ids": [
                    int(value)
                    for value in image_span_table[:, 4].detach().cpu().tolist()
                ],
            }
        )
        report["batches"].append(metrics)
        _atomic_write(args.output, report)

    if len(report["batches"]) != args.num_batches:
        raise RuntimeError(
            f"holdout loader yielded only {len(report['batches'])} joint batches"
        )
    report["summary"] = summarize_probe_batches(report["batches"])
    report["status"] = "complete"
    report["gradients_cleared"] = all(
        parameter.grad is None for parameter in model.parameters()
    )
    if not report["gradients_cleared"]:
        raise RuntimeError("gradient probe left model gradients populated")
    torch.npu.synchronize()
    report["npu"] = {
        "device": str(device),
        "name": torch.npu.get_device_name(device.index or 0),
        "max_memory_allocated": int(torch.npu.max_memory_allocated(device)),
        "max_memory_reserved": int(torch.npu.max_memory_reserved(device)),
    }
    _atomic_write(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
