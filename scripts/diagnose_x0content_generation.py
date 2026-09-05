#!/usr/bin/env python3
"""Diagnose oracle denoising and free-running generation for static X0-content B.

The oracle experiment keeps the exact training-time backbone/flow context, adds
controlled rectified-flow noise to the target latent, and integrates from that
time to data time.  The free-running experiment uses the production serialized
generation path from Gaussian noise.  Keeping these two paths in one report
separates local denoising quality from autoregressive error accumulation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("DIFFUSERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401
from omegaconf import OmegaConf
from PIL import Image, ImageDraw

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.modeling_model.image_flow_loss import FlowLoss  # noqa: E402
from utils.combined_dataloaders import (  # noqa: E402
    build_unified_image_validation_dataloader,
)
from utils.image_generation_io import (  # noqa: E402
    decode_latents,
    load_model_state,
    load_vae,
)
from utils.sharded_ema import load_sharded_ema_checkpoint  # noqa: E402
from utils.utils import get_selfless_mask, load_model_tokenizer  # noqa: E402


def parse_csv_floats(raw: str, *, label: str) -> list[float]:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError(f"{label} must not be empty")
    if any(not math.isfinite(value) for value in values):
        raise ValueError(f"{label} must contain only finite values")
    return list(dict.fromkeys(values))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare oracle-history RF denoising at controlled noise fractions "
            "with production free-running generation from Gaussian noise."
        )
    )
    parser.add_argument("--config", required=True)
    weights = parser.add_mutually_exclusive_group(required=True)
    weights.add_argument("--model_state", default="")
    weights.add_argument("--ema_checkpoint", default="")
    weights.add_argument(
        "--hf_model",
        default="",
        help="Load a complete Hugging Face model export directly.",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=424242)
    parser.add_argument(
        "--noise_fractions",
        default="0.10,0.25,0.50,0.75,1.00",
        help="RF noise fractions; data time is 1-noise_fraction.",
    )
    parser.add_argument(
        "--temperatures",
        default="1.0",
        help="Initial Gaussian-noise amplitudes for free-running generation.",
    )
    parser.add_argument(
        "--cfg_values",
        default="1.0,3.5",
        help="CFG values for free-running generation.",
    )
    parser.add_argument("--sampling_steps", type=int, default=10)
    parser.add_argument("--model_dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--vae_dtype", choices=("fp16", "fp32"), default="fp32")
    parser.add_argument(
        "--order_strategy",
        choices=("spatial_halton", "sequential", "spatial_uniform", "random"),
        default="spatial_halton",
    )
    parser.add_argument("--flow_solver", choices=("heun", "euler"), default="heun")
    return parser.parse_args()


def image_spans(image_span_table: torch.Tensor, image_tokens: int):
    spans = []
    for raw in image_span_table.detach().cpu().tolist():
        row, _, start, end, *_ = map(int, raw)
        if end - start != image_tokens:
            raise ValueError(
                f"image span [{start}, {end}) has {end-start} tokens, "
                f"expected {image_tokens}"
            )
        spans.append((row, start, end))
    if not spans:
        raise ValueError("validation batch contains no image spans")
    return spans


def span_latents(image_latents: torch.Tensor, spans, side: int) -> torch.Tensor:
    return torch.stack(
        [
            image_latents[row, start:end]
            .view(side, side, -1)
            .permute(2, 0, 1)
            for row, start, end in spans
        ]
    )


def flat_to_chw(latents: torch.Tensor, side: int) -> torch.Tensor:
    return latents.view(latents.shape[0], side, side, latents.shape[-1]).permute(
        0, 3, 1, 2
    )


def metric_record(
    prediction: torch.Tensor,
    target: torch.Tensor,
    decoded_prediction: torch.Tensor,
    decoded_target: torch.Tensor,
) -> dict[str, float]:
    latent_mse = F.mse_loss(prediction.float(), target.float()).item()
    pixel_mse = F.mse_loss(
        decoded_prediction.float(), decoded_target.float()
    ).item()
    return {
        "latent_mse_to_target": float(latent_mse),
        "latent_rmse_to_target": float(math.sqrt(latent_mse)),
        "latent_rms": float(prediction.float().square().mean().sqrt().item()),
        "pixel_mse_to_target": float(pixel_mse),
        "pixel_psnr_to_target": float(
            -10.0 * math.log10(max(pixel_mse, 1.0e-12))
        ),
    }


def tensor_to_pil(image: torch.Tensor) -> Image.Image:
    array = (
        image.detach()
        .float()
        .clamp(0, 1)
        .mul(255)
        .round()
        .to(torch.uint8)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    return Image.fromarray(array)


def save_contact_sheet(
    path: Path,
    columns: list[tuple[str, torch.Tensor]],
    *,
    row_labels: list[str],
) -> None:
    if not columns:
        return
    batch = int(columns[0][1].shape[0])
    if any(int(images.shape[0]) != batch for _, images in columns):
        raise ValueError("contact-sheet columns have inconsistent batch sizes")
    if len(row_labels) != batch:
        raise ValueError("row labels do not match contact-sheet batch")

    tiles = [[tensor_to_pil(images[row]) for _, images in columns] for row in range(batch)]
    tile_width, tile_height = tiles[0][0].size
    label_width = 105
    header_height = 34
    gap = 4
    sheet = Image.new(
        "RGB",
        (
            label_width + len(columns) * (tile_width + gap),
            header_height + batch * (tile_height + gap),
        ),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    for column, (label, _) in enumerate(columns):
        draw.text(
            (label_width + column * (tile_width + gap) + 4, 8),
            label,
            fill="black",
        )
    for row in range(batch):
        y = header_height + row * (tile_height + gap)
        draw.text((4, y + 4), row_labels[row], fill="black")
        for column, tile in enumerate(tiles[row]):
            x = label_width + column * (tile_width + gap)
            sheet.paste(tile, (x, y))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def decode_flat(vae, flat: torch.Tensor, side: int, scaling_factor: float):
    return decode_latents(vae, flat_to_chw(flat.float(), side), scaling_factor)


def capture_training_flow_inputs(
    *,
    model,
    config,
    batch: dict,
    device: torch.device,
) -> dict[str, torch.Tensor | None]:
    input_ids = batch["input_ids"].contiguous().to(device)
    token_types = batch["token_types"].to(device)
    sigma = batch["sigma"].to(device)
    labels = batch["labels"].to(device)
    position_ids = batch["position_ids"].to(device)
    image_local_positions = batch["image_local_positions"].to(device)
    image_span_table = batch["image_span_table"].to(device)
    image_loss_mask = batch["image_loss_mask"].to(device=device, dtype=torch.bool)
    image_latents = batch["image_latents"].to(device)

    mask_kwargs = {
        "sigma": sigma,
        "seq_len": int(input_ids.shape[1]),
        "device": device,
        "input_ids": input_ids,
        "token_types": token_types,
        "boi_token_id": int(config.model.boi_token_id),
    }
    attention_mask = get_selfless_mask(**mask_kwargs)
    attention_contract = str(
        config.model.get("dual_stream_attention_contract", "selfless_strict")
    ).strip().lower()
    content_attention_mask = (
        get_selfless_mask(**mask_kwargs, include_diagonal=True)
        if attention_contract == "xlnet_content_diagonal"
        else None
    )

    captured: dict[str, torch.Tensor | None] = {}

    def capture_hook(_module, _args, kwargs):
        for key in (
            "target",
            "z",
            "sigma",
            "image_positions",
            "context_latents",
            "context_mask",
            "content_attention_mask",
            "context_conditions",
        ):
            value = kwargs.get(key)
            captured[key] = value.detach() if isinstance(value, torch.Tensor) else value

    previous_batch_mul = int(model.image_flow_batch_mul)
    model.image_flow_batch_mul = 1
    handle = model.image_flow_head.register_forward_pre_hook(
        capture_hook,
        with_kwargs=True,
    )
    try:
        model(
            X0_input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            content_attention_mask=content_attention_mask,
            token_types=token_types,
            position_ids=position_ids,
            image_local_positions=image_local_positions,
            image_span_table=image_span_table,
            image_loss_mask=image_loss_mask,
            image_latents=image_latents,
            flow_sigma=sigma,
            calculate_likelihood=True,
            compute_text_loss=False,
            compute_image_loss=True,
            record_flow_stats=False,
            return_logits=False,
        )
    finally:
        handle.remove()
        model.image_flow_batch_mul = previous_batch_mul

    required = {
        "target",
        "z",
        "sigma",
        "image_positions",
        "context_latents",
    }
    if model._uses_backbone_x0_flow_content_condition():
        required.add("context_conditions")
    missing = sorted(key for key in required if captured.get(key) is None)
    if missing:
        raise RuntimeError(f"failed to capture training flow inputs: {missing}")
    return captured


def prepare_oracle_context(flow, captured: dict[str, torch.Tensor | None]):
    target = captured["target"]
    assert isinstance(target, torch.Tensor)
    raw_context = FlowLoss._training_context(
        flow,
        target,
        captured["sigma"],
        captured["image_positions"],
        context_latents=captured["context_latents"],
        context_mask=captured["context_mask"],
        content_attention_mask=captured["content_attention_mask"],
    )
    # The historical shared-query contract omitted an explicit content
    # condition.  Its direct training forward falls back to the query
    # condition (`z`) inside ContextualFlowTransformerHead.forward.  Cache
    # construction has no such implicit fallback, so reproduce it here.
    context_conditions = captured["context_conditions"]
    if context_conditions is None:
        context_conditions = captured["z"]
    raw_context["context_conditions"] = context_conditions
    latent_mixer_cache = flow.prepare_latent_mixer_cache(
        context_latents=raw_context["context_latents"],
        context_mask=raw_context["context_mask"],
        content_attention_mask=raw_context["content_attention_mask"],
        context_positions=raw_context["context_positions"],
        context_conditions=raw_context["context_conditions"],
    )
    z = captured["z"]
    assert isinstance(z, torch.Tensor)
    query_positions = flow.net._positions(
        raw_context["query_positions"],
        int(z.shape[0]),
        int(z.shape[1]),
        z.device,
    )
    prepared = {
        "query_positions": query_positions,
        "query_rope": flow.net._build_rope(query_positions, z.dtype),
        "condition_embedding": flow.net.cond_embed(z),
    }
    return prepared, latent_mixer_cache


def oracle_integrate(
    *,
    flow,
    target: torch.Tensor,
    z: torch.Tensor,
    noise: torch.Tensor,
    noise_fraction: float,
    steps: int,
    solver: str,
    prepared_context: dict,
    latent_mixer_cache: dict,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not 0.0 <= noise_fraction <= 1.0:
        raise ValueError(f"noise fraction must lie in [0,1], got {noise_fraction}")
    data_time = 1.0 - float(noise_fraction)
    x = data_time * target.float() + noise_fraction * noise.float()
    initial_x = x.clone()
    times = torch.linspace(
        data_time,
        1.0,
        int(steps) + 1,
        device=target.device,
        dtype=torch.float32,
    )
    direct_estimate = None

    def velocity(state: torch.Tensor, scalar_time: torch.Tensor) -> torch.Tensor:
        t = scalar_time.expand(target.shape[:-1])
        context = dict(prepared_context)
        context["time_embedding"] = flow.net._shape_time(
            flow._scale_time(t), tuple(z.shape[:-1])
        )
        return flow._velocity_prepared(
            state.to(dtype=z.dtype),
            t,
            z,
            context,
            latent_mixer_cache,
        ).float()

    for index in range(int(steps)):
        t = times[index]
        t_next = times[index + 1]
        dt = t_next - t
        v = velocity(x, t)
        if direct_estimate is None:
            direct_estimate = x + (1.0 - t) * v
        if solver == "euler":
            x = x + dt * v
        elif solver == "heun":
            predictor = x + dt * v
            x = x + 0.5 * dt * (v + velocity(predictor, t_next))
        else:
            raise ValueError(f"unsupported solver: {solver}")
    if direct_estimate is None:
        direct_estimate = x
    return x, initial_x, direct_estimate


def prompt_records(tokenizer, batch: dict, spans) -> list[dict[str, object]]:
    records = []
    input_ids = batch["input_ids"]
    token_types = batch["token_types"]
    table = batch["image_span_table"].detach().cpu().tolist()
    image_id_by_row = {int(row[0]): int(row[4]) for row in table}
    for sample_index, (row, _, _) in enumerate(spans):
        keep = token_types[row].ne(1) & token_types[row].ne(3)
        ids = input_ids[row][keep].detach().cpu().tolist()
        records.append(
            {
                "sample_index": sample_index,
                "image_id": image_id_by_row.get(int(row)),
                "prompt": tokenizer.decode(ids, skip_special_tokens=True).strip(),
            }
        )
    return records


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if args.samples <= 0 or args.sampling_steps <= 0:
        raise ValueError("samples and sampling_steps must be positive")
    noise_fractions = parse_csv_floats(
        args.noise_fractions, label="noise_fractions"
    )
    if any(value < 0.0 or value > 1.0 for value in noise_fractions):
        raise ValueError("noise_fractions must lie in [0,1]")
    temperatures = parse_csv_floats(args.temperatures, label="temperatures")
    if any(value < 0.0 for value in temperatures):
        raise ValueError("temperatures must be nonnegative")
    cfg_values = parse_csv_floats(args.cfg_values, label="cfg_values")

    if not torch.npu.is_available():
        raise RuntimeError("This diagnostic requires an Ascend NPU")
    device = torch.device("npu", 0)
    torch.npu.set_device(device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = OmegaConf.load(args.config)
    config.training.batch_size = int(args.samples)
    config.training.dataloader_workers = 0
    config.model.image_flow_num_sampling_steps = str(args.sampling_steps)
    model_dtype = {
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[args.model_dtype]

    if args.hf_model:
        config.model.model_path = str(Path(args.hf_model).resolve())

    started = time.perf_counter()
    print("Loading model/tokenizer...", flush=True)
    model, tokenizer = load_model_tokenizer(config, model_dtype=model_dtype)
    if args.model_state:
        weight_report = load_model_state(model, args.model_state)
        weight_kind = "raw_model_state"
    elif args.ema_checkpoint:
        weight_report = load_sharded_ema_checkpoint(model, args.ema_checkpoint)
        weight_kind = "rank_sharded_ema"
    else:
        weight_report = {"hf_model": str(Path(args.hf_model).resolve())}
        weight_kind = "hf_model_export"
    model = model.to(device=device).eval()

    print("Loading T2I validation rows and VAE...", flush=True)
    loader = build_unified_image_validation_dataloader(
        config,
        tokenizer,
        task_modes=("t2i",),
        batch_size=int(args.samples),
        num_workers=0,
    )
    batch = next(iter(loader))
    image_tokens = int(config.model.image_tokens_per_img)
    side = math.isqrt(image_tokens)
    if side * side != image_tokens:
        raise ValueError(f"image token count is not square: {image_tokens}")
    spans = image_spans(batch["image_span_table"], image_tokens)
    if len(spans) != int(args.samples):
        raise ValueError(f"expected {args.samples} image spans, got {len(spans)}")
    prompts = prompt_records(tokenizer, batch, spans)
    vae = load_vae(config, device, args.vae_dtype)
    scaling_factor = float(config.experiment.validation_vae_scaling_factor)

    input_ids = batch["input_ids"].to(device)
    token_types = batch["token_types"].to(device)
    sequence_sigma = batch["sigma"].to(device)
    image_latents = batch["image_latents"].to(device)
    target_flat = torch.stack(
        [image_latents[row, start:end] for row, start, end in spans]
    ).float()
    target_chw = span_latents(image_latents, spans, side).float()
    decoded_target = decode_latents(vae, target_chw, scaling_factor)

    print("Capturing exact training-time X0/XT flow conditions...", flush=True)
    captured = capture_training_flow_inputs(
        model=model,
        config=config,
        batch=batch,
        device=device,
    )
    captured_target = captured["target"]
    if not isinstance(captured_target, torch.Tensor):
        raise RuntimeError("captured target is missing")
    if not torch.allclose(captured_target.float(), target_flat, rtol=0, atol=0):
        raise RuntimeError("captured flow targets do not match validation latents")

    flow = model.image_flow_head
    prepared_context, latent_mixer_cache = prepare_oracle_context(flow, captured)
    captured_z = captured["z"]
    assert isinstance(captured_z, torch.Tensor)

    noise_generator = torch.Generator(device="cpu")
    noise_generator.manual_seed(int(args.seed))
    shared_noise = torch.randn(
        target_flat.shape,
        generator=noise_generator,
        device="cpu",
        dtype=torch.float32,
    ).to(device)

    oracle_metrics = {}
    oracle_columns = [("target", decoded_target)]
    for fraction in noise_fractions:
        print(f"Oracle denoise: noise_fraction={fraction:.2f}", flush=True)
        torch.npu.synchronize(device)
        stage_started = time.perf_counter()
        denoised, noised, direct_estimate = oracle_integrate(
            flow=flow,
            target=captured_target,
            z=captured_z,
            noise=shared_noise,
            noise_fraction=fraction,
            steps=int(args.sampling_steps),
            solver=str(args.flow_solver),
            prepared_context=prepared_context,
            latent_mixer_cache=latent_mixer_cache,
        )
        torch.npu.synchronize(device)
        decoded_denoised = decode_flat(vae, denoised, side, scaling_factor)
        decoded_noised = decode_flat(vae, noised, side, scaling_factor)
        decoded_direct = decode_flat(vae, direct_estimate, side, scaling_factor)
        label = f"noise_{fraction:.2f}".replace(".", "p")
        integrated = metric_record(
            denoised, target_flat, decoded_denoised, decoded_target
        )
        direct = metric_record(
            direct_estimate, target_flat, decoded_direct, decoded_target
        )
        oracle_metrics[f"{fraction:.6g}"] = {
            "noise_fraction": float(fraction),
            "data_time_start": float(1.0 - fraction),
            "seconds": float(time.perf_counter() - stage_started),
            "integrated": integrated,
            "direct_x0_estimate": direct,
        }
        save_contact_sheet(
            output_dir / f"oracle_{label}.png",
            [
                ("target", decoded_target),
                (f"x_t n={fraction:.2f}", decoded_noised),
                ("Heun denoised", decoded_denoised),
                ("direct x0", decoded_direct),
            ],
            row_labels=[f"sample {index}" for index in range(len(spans))],
        )
        oracle_columns.append((f"n={fraction:.2f}", decoded_denoised))

    save_contact_sheet(
        output_dir / "oracle_denoising_overview.png",
        oracle_columns,
        row_labels=[f"sample {index}" for index in range(len(spans))],
    )

    free_metrics = {}
    free_columns = [("target", decoded_target)]
    for temperature in temperatures:
        for cfg in cfg_values:
            print(
                f"Free generation: temperature={temperature:.2f}, CFG={cfg:.2f}",
                flush=True,
            )
            torch.npu.synchronize(device)
            stage_started = time.perf_counter()
            generated_chw, trace = model.generate(
                "t2i",
                input_ids=input_ids,
                token_types=token_types,
                sigma=sequence_sigma,
                spans=spans,
                image_latent_dim=int(config.model.image_latent_dim),
                initial_noise_bank=shared_noise,
                flow_temperature=float(temperature),
                flow_cfg=float(cfg),
                flow_cfg_schedule="constant",
                flow_solver=str(args.flow_solver),
                flow_num_steps=int(args.sampling_steps),
                parallel_rate=1,
                order_strategy=str(args.order_strategy),
                use_cache=True,
                return_trace=True,
                debug_finite=True,
            )
            torch.npu.synchronize(device)
            if tuple(generated_chw.shape) != tuple(target_chw.shape):
                raise RuntimeError(
                    "free-running generation shape mismatch: "
                    f"generated={tuple(generated_chw.shape)}, "
                    f"target={tuple(target_chw.shape)}"
                )
            decoded_generated = decode_latents(
                vae,
                generated_chw.float(),
                scaling_factor,
            )
            label = f"temp_{temperature:.2f}_cfg_{cfg:.2f}".replace(".", "p")
            record = metric_record(
                generated_chw, target_chw, decoded_generated, decoded_target
            )
            record.update(
                {
                    "temperature": float(temperature),
                    "cfg": float(cfg),
                    "seconds": float(time.perf_counter() - stage_started),
                    "generation_steps_completed": int(
                        trace["generation_step"].max().item()
                    ),
                    "backbone_kv_cache_enabled": bool(
                        trace.get("backbone_kv_cache_enabled", False)
                    ),
                }
            )
            free_metrics[label] = record
            save_contact_sheet(
                output_dir / f"from_scratch_{label}.png",
                [("target", decoded_target), (label, decoded_generated)],
                row_labels=[f"sample {index}" for index in range(len(spans))],
            )
            free_columns.append((f"T={temperature:g} C={cfg:g}", decoded_generated))
            del generated_chw, decoded_generated, trace
            torch.npu.empty_cache()

    save_contact_sheet(
        output_dir / "from_scratch_overview.png",
        free_columns,
        row_labels=[f"sample {index}" for index in range(len(spans))],
    )

    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        git_commit = None
    report = {
        "schema": "x0content_generation_diagnostic_v1",
        "config": str(Path(args.config).resolve()),
        "weight_kind": weight_kind,
        "weight_report": weight_report,
        "model_contract": {
            "architecture_variant": str(
                getattr(model.config, "architecture_variant", "")
            ),
            "dual_stream_attention_contract": str(
                getattr(model.config, "dual_stream_attention_contract", "")
            ),
            "flow_head_attention_contract": str(
                getattr(model.config, "flow_head_attention_contract", "")
            ),
            "flow_condition_contract": str(
                getattr(model.config, "flow_condition_contract", "")
            ),
        },
        "git_commit": git_commit,
        "device": torch.npu.get_device_name(device),
        "model_dtype": str(next(model.parameters()).dtype),
        "vae_dtype": str(next(vae.parameters()).dtype),
        "samples": int(args.samples),
        "seed": int(args.seed),
        "sampling_steps": int(args.sampling_steps),
        "flow_solver": str(args.flow_solver),
        "order_strategy": str(args.order_strategy),
        "rf_convention": "t=0 noise, t=1 data",
        "prompts": prompts,
        "target_latent_rms": float(target_flat.square().mean().sqrt().item()),
        "oracle_history_denoising": oracle_metrics,
        "free_running_from_gaussian_noise": free_metrics,
        "total_seconds": float(time.perf_counter() - started),
    }
    report_path = output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    print(f"Wrote diagnostic report to {report_path}", flush=True)


if __name__ == "__main__":
    main()
