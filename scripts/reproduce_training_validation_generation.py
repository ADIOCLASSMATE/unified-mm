#!/usr/bin/env python3
"""Reproduce the rank-0 in-training validation image generation exactly.

This diagnostic intentionally executes the validation loss forward before
generation because that forward advances the NPU RNG used by the subsequent
per-token Gaussian draws.  It also builds the interleaved T2I/I2T batch used by
the training job and selects T2I spans through the original image-loss mask.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("DIFFUSERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import numpy as np
import torch
import torch_npu  # noqa: F401
from omegaconf import OmegaConf
from PIL import Image
from torchvision.utils import make_grid, save_image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pretrain.train_selfless_flow import (  # noqa: E402
    _build_backbone_attention_masks,
)
from utils.combined_dataloaders import (  # noqa: E402
    build_unified_image_validation_dataloader,
)
from utils.image_generation_io import (  # noqa: E402
    decode_latents,
    load_model_state,
    load_vae,
)
from utils.utils import load_model_tokenizer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model_state", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--reference_dir", default="")
    parser.add_argument("--global_step", type=int, default=20000)
    parser.add_argument("--model_dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--vae_dtype", choices=("fp16", "fp32"), default="fp32")
    return parser.parse_args()


def _span_records(table: torch.Tensor) -> list[tuple[int, int, int, int]]:
    return [
        (int(row), int(start), int(end), int(image_id))
        for row, _, start, end, image_id, *_ in table.detach().cpu().tolist()
    ]


def _image_arrays(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def _comparison_metrics(left: Path, right: Path) -> dict[str, float | bool]:
    a = _image_arrays(left)
    b = _image_arrays(right)
    if a.shape != b.shape:
        return {
            "same_shape": False,
            "left_height": int(a.shape[0]),
            "left_width": int(a.shape[1]),
            "right_height": int(b.shape[0]),
            "right_width": int(b.shape[1]),
        }
    difference = np.abs(a - b)
    return {
        "same_shape": True,
        "pixel_mse": float(np.square(a - b).mean()),
        "pixel_max_abs": float(difference.max()),
        "pixel_exact_fraction": float((difference == 0).mean()),
    }


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if not torch.npu.is_available():
        raise RuntimeError("This diagnostic requires an Ascend NPU")
    device = torch.device("npu", 0)
    torch.npu.set_device(device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = OmegaConf.load(args.config)
    model_dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[
        args.model_dtype
    ]
    model, tokenizer = load_model_tokenizer(config, model_dtype=model_dtype)
    weight_report = load_model_state(model, args.model_state)
    model = model.to(device=device).eval()
    vae = load_vae(config, device, args.vae_dtype)

    batch_size = int(config.dataset.params.sources.t2i.micro_batch_size)
    loader = build_unified_image_validation_dataloader(
        config,
        tokenizer,
        task_modes=("t2i", "i2t"),
        batch_size=batch_size,
        num_workers=0,
    )
    seed = int(config.experiment.validation_seed)
    started = time.perf_counter()
    with torch.random.fork_rng(devices=[0], device_type="npu"):
        torch.default_generator.manual_seed(seed)
        with torch.npu.device(device):
            torch.npu.manual_seed(seed)

        batch = next(iter(loader))
        input_ids = batch["input_ids"].contiguous().to(device)
        token_types = batch["token_types"].to(device)
        sigma = batch["sigma"].to(device)
        labels = batch["labels"].to(device)
        host_image_loss_mask = batch["image_loss_mask"]
        image_loss_mask = host_image_loss_mask.to(device=device, dtype=torch.bool)
        position_ids = batch["position_ids"].to(device)
        image_local_positions = batch["image_local_positions"].to(device)
        host_image_span_table = batch["image_span_table"]
        image_span_table = host_image_span_table.to(device)
        image_latents = batch["image_latents"].to(device)

        attention_mask, content_attention_mask = _build_backbone_attention_masks(
            config=config,
            input_ids=input_ids,
            token_types=token_types,
            sigma=sigma,
        )
        forward_kwargs = dict(
            X0_input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            token_types=token_types,
            position_ids=position_ids,
            image_local_positions=image_local_positions,
            image_span_table=image_span_table,
            image_loss_mask=image_loss_mask,
            image_latents=image_latents,
            flow_sigma=sigma,
            calculate_likelihood=True,
            record_flow_stats=False,
            compute_text_loss=True,
            compute_image_loss=True,
        )
        if content_attention_mask is not None:
            forward_kwargs["content_attention_mask"] = content_attention_mask
        print("Running the validation-loss forward to advance NPU RNG...", flush=True)
        model(**forward_kwargs)

        active = []
        active_image_ids = []
        for row, start, end, image_id in _span_records(host_image_span_table):
            if bool(host_image_loss_mask[row, start:end].any()):
                active.append((row, start, end))
                active_image_ids.append(image_id)
        sample_count = min(
            int(config.experiment.validation_image_samples), len(active)
        )
        spans = active[:sample_count]
        active_image_ids = active_image_ids[:sample_count]
        image_tokens = int(config.model.image_tokens_per_img)
        side = int(image_tokens**0.5)
        target_latents = torch.stack(
            [
                image_latents[row, start:end]
                .view(side, side, -1)
                .permute(2, 0, 1)
                for row, start, end in spans
            ]
        )

        captured_draws: list[torch.Tensor] = []
        original_randn = torch.randn

        def traced_randn(*shape, **kwargs):
            value = original_randn(*shape, **kwargs)
            if tuple(value.shape) == (sample_count, int(config.model.image_latent_dim)):
                captured_draws.append(value.detach().float().cpu())
            return value

        torch.randn = traced_randn
        try:
            print("Running production model.generate without supplied noise...", flush=True)
            generated_latents, trace = model.generate(
                "t2i",
                input_ids=input_ids,
                token_types=token_types,
                sigma=sigma,
                spans=spans,
                image_latent_dim=int(config.model.image_latent_dim),
                flow_temperature=float(config.experiment.validation_flow_temperature),
                flow_cfg=float(config.experiment.validation_flow_cfg),
                flow_cfg_schedule=str(config.experiment.validation_flow_cfg_schedule),
                flow_solver=str(config.experiment.validation_flow_solver),
                parallel_rate=int(
                    config.experiment.validation_single_stream_parallel_rate
                ),
                order_strategy=str(
                    config.experiment.validation_single_stream_order_strategies[0]
                ),
                use_cache=True,
                return_trace=True,
                debug_finite=True,
            )
        finally:
            torch.randn = original_randn

    if len(captured_draws) != image_tokens:
        raise RuntimeError(
            f"expected {image_tokens} per-token noise draws, got {len(captured_draws)}"
        )
    generation_order = trace["generation_order"].reshape(sample_count, -1)
    initial_noise_bank = torch.zeros(
        sample_count,
        image_tokens,
        int(config.model.image_latent_dim),
        dtype=torch.float32,
    )
    for step_index, draw in enumerate(captured_draws, start=1):
        positions = generation_order.eq(step_index).to(torch.int64).argmax(dim=1)
        initial_noise_bank[
            torch.arange(sample_count), positions.cpu()
        ] = draw
    torch.save(initial_noise_bank, output_dir / "initial_noise_bank.pt")

    original_contract_check = model._uses_backbone_x0_flow_content_condition
    model._uses_backbone_x0_flow_content_condition = lambda: False
    try:
        print(
            "Replaying the same noise with the historical XT-shared content path...",
            flush=True,
        )
        legacy_latents, legacy_trace = model.generate(
            "t2i",
            input_ids=input_ids,
            token_types=token_types,
            sigma=sigma,
            spans=spans,
            image_latent_dim=int(config.model.image_latent_dim),
            initial_noise_bank=initial_noise_bank.to(device),
            flow_temperature=float(config.experiment.validation_flow_temperature),
            flow_cfg=float(config.experiment.validation_flow_cfg),
            flow_cfg_schedule=str(config.experiment.validation_flow_cfg_schedule),
            flow_solver=str(config.experiment.validation_flow_solver),
            parallel_rate=int(
                config.experiment.validation_single_stream_parallel_rate
            ),
            order_strategy=str(
                config.experiment.validation_single_stream_order_strategies[0]
            ),
            use_cache=True,
            return_trace=True,
            debug_finite=True,
        )
    finally:
        model._uses_backbone_x0_flow_content_condition = original_contract_check

    scaling = float(config.experiment.validation_vae_scaling_factor)
    target_images = decode_latents(vae, target_latents.float(), scaling)
    generated_images = decode_latents(vae, generated_latents.float(), scaling)
    legacy_images = decode_latents(vae, legacy_latents.float(), scaling)
    target_path = output_dir / "target.png"
    generated_path = output_dir / "generated.png"
    overview_path = output_dir / "overview.png"
    legacy_path = output_dir / "generated_legacy_xt_content.png"
    contract_comparison_path = output_dir / "content_contract_comparison.png"
    save_image(target_images, target_path)
    save_image(generated_images, generated_path)
    save_image(legacy_images, legacy_path)
    comparison = torch.stack([target_images, generated_images], dim=1).flatten(0, 1)
    save_image(make_grid(comparison, nrow=2), overview_path)
    contract_comparison = torch.stack(
        [target_images, generated_images, legacy_images], dim=1
    ).flatten(0, 1)
    save_image(
        make_grid(contract_comparison, nrow=3), contract_comparison_path
    )

    report: dict[str, object] = {
        "schema": "training_validation_generation_reproduction_v1",
        "config": str(Path(args.config).resolve()),
        "model_state": str(Path(args.model_state).resolve()),
        "weight_report": weight_report,
        "seed": seed,
        "batch_size": batch_size,
        "task_modes": list(batch.get("task_modes", [])),
        "active_t2i_spans": [list(item) for item in spans],
        "active_image_ids": active_image_ids,
        "image_flow_batch_mul": int(model.image_flow_batch_mul),
        "initial_noise_bank_shape": list(initial_noise_bank.shape),
        "initial_noise_rms": float(initial_noise_bank.square().mean().sqrt()),
        "target_latent_rms": float(target_latents.float().square().mean().sqrt()),
        "generated_latent_rms": float(
            generated_latents.float().square().mean().sqrt()
        ),
        "latent_mse_to_target": float(
            torch.nn.functional.mse_loss(
                generated_latents.float(), target_latents.float()
            )
        ),
        "legacy_xt_content_latent_rms": float(
            legacy_latents.float().square().mean().sqrt()
        ),
        "legacy_xt_content_latent_mse_to_target": float(
            torch.nn.functional.mse_loss(
                legacy_latents.float(), target_latents.float()
            )
        ),
        "legacy_flow_content_condition": legacy_trace.get(
            "flow_content_condition"
        ),
        "generation_step_max": int(trace["generation_step"].max().item()),
        "backbone_kv_cache_enabled": bool(
            trace.get("backbone_kv_cache_enabled", False)
        ),
        "seconds": float(time.perf_counter() - started),
    }
    reference_dir = Path(args.reference_dir) if args.reference_dir else None
    if reference_dir is not None:
        reference_target = (
            reference_dir / f"step-{args.global_step:08d}-target.png"
        )
        reference_generated = (
            reference_dir
            / f"step-{args.global_step:08d}-single_stream_pred_spatial_halton.png"
        )
        report["reference_target_comparison"] = _comparison_metrics(
            reference_target, target_path
        )
        report["reference_generated_comparison"] = _comparison_metrics(
            reference_generated, generated_path
        )
        report["reference_legacy_generated_comparison"] = _comparison_metrics(
            reference_generated, legacy_path
        )

    report_path = output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    print(f"Wrote {report_path}", flush=True)


if __name__ == "__main__":
    main()
