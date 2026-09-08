#!/usr/bin/env python3
"""Replay in-training validation from a raw resumable checkpoint.

The training metric is produced from the current (non-EMA) model.  This
evaluator therefore loads DeepSpeed's raw ``module`` state and replays the
checkpoint's original logical data-parallel topology.  A smaller physical
world can emulate a larger saved world in waves without changing sample-to-
rank assignment or per-rank validation seeds.

Training-time T2I generation happens after the first loss batch and advances
the NPU RNG before later batches.  Full image generation is irrelevant to the
loss but its Gaussian draws are not.  The replay replaces generation with an
RNG-only implementation that consumes the same per-token FP32 draws.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import MethodType
from typing import Any

os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("DIFFUSERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import torch
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.data_loader import prepare_data_loader
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pretrain.train_selfless_flow import validate
from utils.combined_dataloaders import (
    build_unified_image_validation_dataloader,
)
from utils.evaluation_model_source import (
    configure_model_source,
    resolve_evaluation_model_source,
)
from utils.utils import load_model_tokenizer

CORE_METRICS = (
    "val/loss",
    "val/loss_text",
    "val/loss_image_flow",
    "val/text_target_tokens",
    "val/image_target_tokens",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--recorded_metrics", type=Path)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--absolute_tolerance",
        type=float,
        default=1.0e-4,
        help=(
            "Tolerance for weighted val/loss and per-modality weighted "
            "contributions (default: 1e-4)."
        ),
    )
    return parser.parse_args()


class LogicalRankAccelerator:
    """Override only the validation seed rank; collectives stay physical."""

    def __init__(self, accelerator: Accelerator, logical_rank: int) -> None:
        self._accelerator = accelerator
        self._logical_rank = int(logical_rank)

    @property
    def process_index(self) -> int:
        return self._logical_rank

    def __getattr__(self, name: str):
        return getattr(self._accelerator, name)


def active_validation_modes(config) -> tuple[str, ...]:
    schedule = tuple(
        str(source).strip().lower()
        for source in config.dataset.params.schedule
    )
    active_sources = tuple(dict.fromkeys(schedule))
    modes = tuple(
        source for source in active_sources if source in {"t2i", "i2t"}
    )
    if not modes:
        raise ValueError(
            "the checkpoint schedule has no image-backed validation source"
        )
    if modes not in {("t2i", "i2t"), ("t2i",), ("i2t",)}:
        raise ValueError(f"unsupported validation mode order: {modes}")
    return modes


def load_raw_training_weights(model, checkpoint: Path) -> dict[str, Any]:
    model_state = (
        checkpoint / "pytorch_model" / "mp_rank_00_model_states.pt"
    )
    if not model_state.is_file():
        raise FileNotFoundError(model_state)
    payload = torch.load(
        str(model_state),
        map_location="cpu",
        mmap=True,
        weights_only=False,
    )
    if not isinstance(payload, dict) or not isinstance(
        payload.get("module"), dict
    ):
        raise TypeError(f"invalid DeepSpeed model state: {model_state}")
    state = payload["module"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    report = {
        "kind": "raw_training_checkpoint",
        "path": str(model_state.resolve()),
        "state_key_count": len(state),
        "missing": list(missing),
        "unexpected": list(unexpected),
        "global_steps": int(payload.get("global_steps", -1)),
        "dp_world_size": int(payload.get("dp_world_size", -1)),
    }
    del state, payload
    gc.collect()
    if report["missing"] or report["unexpected"]:
        raise RuntimeError(f"raw model state is incompatible: {report}")
    return report


def install_rng_only_generation(model, config) -> dict[str, Any]:
    strategies = tuple(
        str(item).strip().lower()
        for item in config.experiment.get(
            "validation_single_stream_order_strategies", []
        )
    )
    unsupported = sorted(set(strategies) - {"sequential", "spatial_halton"})
    if unsupported:
        raise ValueError(
            "RNG-only generation has not proven equivalent for strategies: "
            f"{unsupported}"
        )
    image_tokens = int(config.model.image_tokens_per_img)
    latent_dim = int(config.model.image_latent_dim)
    side = math.isqrt(image_tokens)
    if side * side != image_tokens:
        raise ValueError(f"image token count is not square: {image_tokens}")

    def rng_only_generate(self, task: str, /, **kwargs):
        task = str(task).strip().lower()
        if task != "t2i":
            raise RuntimeError(
                "RNG-only replay expects validation_i2t_every=0, got "
                f"generation task {task!r}"
            )
        spans = kwargs["spans"]
        input_ids = kwargs["input_ids"]
        sample_count = len(spans)
        requested_latent_dim = int(kwargs.get("image_latent_dim", latent_dim))
        if requested_latent_dim != latent_dim:
            raise ValueError(
                f"generation latent dim changed: {requested_latent_dim}"
            )
        # Serialized generation invokes one independent FP32 flow-noise draw
        # for every image position.  CFG duplicates this state afterwards and
        # therefore does not change the draw shape.
        for _ in range(image_tokens):
            torch.randn(
                sample_count,
                latent_dim,
                device=input_ids.device,
                dtype=torch.float32,
            )
        dtype = next(self.image_flow_head.parameters()).dtype
        prediction = torch.zeros(
            sample_count,
            latent_dim,
            side,
            side,
            device=input_ids.device,
            dtype=dtype,
        )
        generation_step = torch.arange(
            1,
            image_tokens + 1,
            device=input_ids.device,
            dtype=torch.long,
        ).unsqueeze(0).expand(sample_count, -1)
        attention_contract = str(
            getattr(
                self.config,
                "dual_stream_attention_contract",
                "selfless_strict",
            )
        ).strip().lower()
        return prediction, {
            "attention_contract": attention_contract,
            "single_stream_content_self_diagonal": (
                attention_contract == "xlnet_content_diagonal"
            ),
            "backbone_kv_cache_enabled": True,
            "backbone_kv_cache_peak_bytes": 0,
            "generation_step": generation_step,
            "rng_only_replay": True,
        }

    model.generate = MethodType(rng_only_generate, model)
    return {
        "kind": "per_token_flow_noise_only",
        "strategies": list(strategies),
        "draw_calls_per_strategy": image_tokens,
        "draw_shape": [
            int(config.experiment.validation_image_samples),
            latent_dim,
        ],
        "draw_dtype": "float32",
    }


def aggregate_wave_metrics(
    wave_metrics: list[dict[str, Any]],
    *,
    lambda_text: float,
    lambda_image: float,
) -> dict[str, float]:
    text_count = sum(
        float(metrics["val/text_target_tokens"]) for metrics in wave_metrics
    )
    image_count = sum(
        float(metrics["val/image_target_tokens"]) for metrics in wave_metrics
    )
    text_weighted = sum(
        float(metrics["val/loss_text"])
        * float(metrics["val/text_target_tokens"])
        for metrics in wave_metrics
    )
    image_weighted = sum(
        float(metrics["val/loss_image_flow"])
        * float(metrics["val/image_target_tokens"])
        for metrics in wave_metrics
    )
    loss_text = text_weighted / max(text_count, 1.0)
    loss_image = image_weighted / max(image_count, 1.0)
    weighted_text = float(lambda_text) * loss_text
    weighted_image = float(lambda_image) * loss_image
    return {
        "val/loss": weighted_text + weighted_image,
        "val/loss_text": loss_text,
        "val/loss_i2t": loss_text,
        "val/loss_image_flow": loss_image,
        "val/loss_t2i": loss_image,
        "val/weighted_contribution_i2t": weighted_text,
        "val/weighted_contribution_t2i": weighted_image,
        "val/weighted_contribution_total": weighted_text + weighted_image,
        "val/text_target_tokens": text_count,
        "val/image_target_tokens": image_count,
    }


def git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def effective_flow_condition_contract(model, config) -> str:
    architecture = str(config.model.architecture_variant).strip().lower()
    if architecture == "dynamic_xt":
        return str(model.dynamic_xt_flow_condition_contract)
    if architecture == "positionwise_flow_head_on_b":
        return "not_applicable"
    return str(model.flow_condition_contract)


def main() -> None:
    args = parse_args()
    if args.num_workers < 0:
        raise ValueError("--num_workers must be non-negative")
    if args.absolute_tolerance < 0.0:
        raise ValueError("--absolute_tolerance must be non-negative")

    checkpoint = args.checkpoint.resolve()
    source = resolve_evaluation_model_source(checkpoint)
    if source.kind != "rank_sharded_ema":
        raise ValueError("a raw resumable checkpoint is required")
    metadata = json.loads(
        (checkpoint / "metadata.json").read_text(encoding="utf-8")
    )
    source_world_size = int(metadata["world_size"])
    global_step = int(metadata["global_step"])
    if global_step != source.global_step:
        raise ValueError("checkpoint step metadata disagree")

    recorded_path = (
        args.recorded_metrics.resolve()
        if args.recorded_metrics is not None
        else checkpoint.parent
        / f"validation_metrics_step_{global_step}.json"
    )
    recorded_payload = json.loads(recorded_path.read_text(encoding="utf-8"))
    if int(recorded_payload["global_step"]) != global_step:
        raise ValueError("recorded validation step does not match checkpoint")
    recorded = recorded_payload["metrics"]

    config = OmegaConf.load(args.config)
    if str(config.dataset.class_name) != "UnifiedMixedDataset":
        raise ValueError("replay requires UnifiedMixedDataset")
    config.training.runtime_hashing_enabled = False
    configure_model_source(config, source)
    modes = active_validation_modes(config)
    validation_max_batches = int(config.experiment.validation_max_batches)
    if validation_max_batches <= 0:
        raise ValueError("validation_max_batches must be positive")
    batch_size = int(config.dataset.params.sources[modes[0]].micro_batch_size)

    output_dir = args.output_dir.resolve()
    config.experiment.validation_i2t_every = 0
    # Keep T2I generation enabled so the RNG-only replacement runs after the
    # first loss batch, but prevent the main rank from loading the VAE.
    config.experiment.validation_vae_path = str(
        output_dir / "intentionally_missing_vae.ckpt"
    )

    accelerator = Accelerator(
        mixed_precision="bf16",
        dataloader_config=DataLoaderConfiguration(non_blocking=True),
    )
    physical_world_size = int(accelerator.num_processes)
    if source_world_size % physical_world_size:
        raise ValueError(
            "saved world size must be divisible by physical replay world: "
            f"{source_world_size} % {physical_world_size}"
        )
    wave_count = source_world_size // physical_world_size

    model, tokenizer = load_model_tokenizer(
        config,
        model_dtype=torch.bfloat16,
    )
    weight_report = load_raw_training_weights(model, checkpoint)
    if weight_report["global_steps"] != global_step:
        raise ValueError(
            "DeepSpeed model state step does not match checkpoint metadata: "
            f"{weight_report['global_steps']} != {global_step}"
        )
    rng_replay = install_rng_only_generation(model, config)
    model.to(accelerator.device).eval()

    wave_reports: list[dict[str, Any]] = []
    for wave_index in range(wave_count):
        logical_rank = (
            wave_index * physical_world_size + accelerator.process_index
        )
        wave_dir = output_dir / f"wave-{wave_index:02d}"
        config.experiment.output_dir = str(wave_dir)
        config.experiment.validation_output_dir = str(wave_dir)
        base_loader = build_unified_image_validation_dataloader(
            config,
            tokenizer,
            task_modes=modes,
            batch_size=batch_size,
            num_workers=int(args.num_workers),
        )
        logical_loader = prepare_data_loader(
            base_loader,
            device=accelerator.device,
            num_processes=source_world_size,
            process_index=logical_rank,
            split_batches=False,
            put_on_device=False,
            dispatch_batches=False,
            even_batches=True,
            non_blocking=True,
        )
        logical_accelerator = LogicalRankAccelerator(
            accelerator,
            logical_rank,
        )
        validate(
            model,
            logical_loader,
            logical_accelerator,
            global_step,
            config=config,
            tokenizer=tokenizer,
        )
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            metrics_path = (
                wave_dir / f"validation_metrics_step_{global_step}.json"
            )
            wave_payload = json.loads(
                metrics_path.read_text(encoding="utf-8")
            )
            wave_reports.append(
                {
                    "wave_index": wave_index,
                    "logical_rank_start": wave_index * physical_world_size,
                    "logical_rank_end": (
                        (wave_index + 1) * physical_world_size - 1
                    ),
                    "metrics_path": str(metrics_path),
                    "metrics": wave_payload["metrics"],
                }
            )
        accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        replayed = aggregate_wave_metrics(
            [wave["metrics"] for wave in wave_reports],
            lambda_text=float(model.lambda_text),
            lambda_image=float(model.lambda_image),
        )
        comparison = {}
        for metric in CORE_METRICS:
            recorded_value = float(recorded[metric])
            replayed_value = float(replayed[metric])
            absolute_delta = abs(replayed_value - recorded_value)
            comparison[metric] = {
                "recorded": recorded_value,
                "replayed": replayed_value,
                "signed_delta": replayed_value - recorded_value,
                "absolute_delta": absolute_delta,
                "within_tolerance": (
                    absolute_delta <= float(args.absolute_tolerance)
                ),
            }
        count_metrics = {
            "val/text_target_tokens",
            "val/image_target_tokens",
        }
        weighted_component_deltas = {
            "i2t": abs(
                float(replayed["val/loss_text"])
                - float(recorded["val/loss_text"])
            )
            * float(model.lambda_text),
            "t2i": abs(
                float(replayed["val/loss_image_flow"])
                - float(recorded["val/loss_image_flow"])
            )
            * float(model.lambda_image),
        }
        # The requested compatibility target is the weighted training
        # ``val/loss``.  Also require each modality's weighted contribution to
        # agree so opposite-sign component drift cannot cancel accidentally.
        # Raw component losses remain in ``comparison`` as diagnostics.
        matches = (
            comparison["val/loss"]["within_tolerance"]
            and all(
                comparison[metric]["absolute_delta"] == 0.0
                for metric in count_metrics
            )
            and all(
                delta <= float(args.absolute_tolerance)
                for delta in weighted_component_deltas.values()
            )
        )
        payload = {
            "schema": "raw_training_validation_compatibility_audit_v1",
            "complete": True,
            "matches_recorded": matches,
            "absolute_tolerance": float(args.absolute_tolerance),
            "current_git_revision": git_revision(),
            "completed_at": datetime.now(UTC)
            .isoformat()
            .replace("+00:00", "Z"),
            "config": str(args.config.resolve()),
            "checkpoint": str(checkpoint),
            "recorded_metrics_path": str(recorded_path),
            "global_step": global_step,
            "architecture_variant": str(config.model.architecture_variant),
            "dual_stream_attention_contract": str(
                config.model.dual_stream_attention_contract
            ),
            "flow_head_attention_contract": str(
                config.model.flow_head_attention_contract
            ),
            "flow_condition_contract": effective_flow_condition_contract(
                model,
                config,
            ),
            "resolved_config_flow_condition_contract": str(
                config.model.flow_condition_contract
            ),
            "task_modes": list(modes),
            "batch_size_per_logical_rank": batch_size,
            "validation_max_batches_per_logical_rank": (
                validation_max_batches
            ),
            "source_world_size": source_world_size,
            "physical_world_size": physical_world_size,
            "logical_replay_waves": wave_count,
            "weight_load": weight_report,
            "rng_replay": rng_replay,
            "recorded": recorded,
            "replayed": replayed,
            "comparison": comparison,
            "weighted_component_absolute_deltas": (
                weighted_component_deltas
            ),
            "waves": wave_reports,
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / "audit.json"
        report_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
        print(f"Wrote {report_path}", flush=True)
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
