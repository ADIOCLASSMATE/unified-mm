#!/usr/bin/env python3
"""Generate fixed-caption samples and audit backbone KV-cache inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import textwrap
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("DIFFUSERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import torch
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont
from torchvision.utils import save_image

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.generate_flow_validation_images import decode_latents, load_vae
from utils.dataset_imagenet_flow_cache import DEFAULT_CAPTION_PREFIX
from utils.utils import load_model_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate arbitrary caption samples with an auditable backbone KV-cache mode."
    )
    parser.add_argument(
        "--config",
        default="configs/selfless/imagenet100_caption_base_80ep.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        default="output/selfless-flow-base-imagenet100-caption-80ep/hf_model-final-ema",
    )
    parser.add_argument("--prompts_jsonl", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model_dtype", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--vae_dtype", choices=["fp32", "fp16"], default="fp32")
    parser.add_argument("--sampling_steps", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--cfg", type=float, default=3.5)
    parser.add_argument("--cfg_schedule", choices=["constant", "linear"], default="constant")
    parser.add_argument("--flow_solver", choices=["heun", "euler"], default="heun")
    parser.add_argument("--order_strategy", default="spatial_halton")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache_mode", choices=["on", "off"], default="on")
    parser.add_argument(
        "--warmup_generation_steps",
        type=int,
        default=2,
        help=(
            "Run this many autoregressive image-token steps before timing; "
            "the cached path needs 2 to cover both prefill/query and commit/query shapes. "
            "0 disables warm-up."
        ),
    )
    parser.add_argument("--decode_batch_size", type=int, default=4)
    return parser.parse_args()


def read_prompts(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if not isinstance(record.get("caption"), str) or not record["caption"].strip():
                raise ValueError(f"{path}:{line_number}: non-empty string 'caption' is required")
            record = dict(record)
            record.setdefault("id", f"prompt_{len(records):03d}")
            record.setdefault("category", "prompt")
            record["caption"] = record["caption"].strip()
            records.append(record)
    if not records:
        raise ValueError(f"No prompts found in {path}")
    ids = [str(record["id"]) for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Prompt ids must be unique")
    return records


def build_caption_batch(tokenizer, model_config, records: list[dict], device: torch.device):
    image_tokens = int(model_config.image_tokens_per_img)
    sequences = []
    token_type_rows = []
    prompt_lengths = []
    for record in records:
        prompt = f"{DEFAULT_CAPTION_PREFIX} {record['caption']}"
        prompt_ids = torch.tensor(
            tokenizer.encode(prompt, add_special_tokens=False),
            dtype=torch.long,
        )
        prompt_len = int(prompt_ids.numel())
        input_ids = torch.cat(
            [
                prompt_ids,
                torch.tensor(
                    [int(model_config.boi_token_id)]
                    + [int(model_config.image_mask_token_id)] * image_tokens
                    + [int(model_config.eoi_token_id), int(tokenizer.eos_token_id)],
                    dtype=torch.long,
                ),
            ]
        )
        token_types = torch.cat(
            [
                torch.zeros(prompt_len, dtype=torch.uint8),
                torch.tensor([2] + [1] * image_tokens + [2, 2], dtype=torch.uint8),
            ]
        )
        sequences.append(input_ids)
        token_type_rows.append(token_types)
        prompt_lengths.append(prompt_len)

    max_len = max(int(row.numel()) for row in sequences)
    batch_size = len(records)
    input_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
    token_types = torch.full((batch_size, max_len), 3, dtype=torch.uint8)
    sigma = torch.full((batch_size, max_len), max_len, dtype=torch.long)
    spans = []
    for batch_idx, (ids, types, prompt_len) in enumerate(
        zip(sequences, token_type_rows, prompt_lengths)
    ):
        length = int(ids.numel())
        image_start = prompt_len + 1
        image_end = image_start + image_tokens
        input_ids[batch_idx, :length] = ids
        token_types[batch_idx, :length] = types
        sigma[batch_idx, :prompt_len] = torch.arange(prompt_len)
        sigma[batch_idx, prompt_len] = prompt_len
        sigma[batch_idx, image_end] = prompt_len + 1
        sigma[batch_idx, image_start:image_end] = (
            prompt_len + 2 + torch.arange(image_tokens)
        )
        sigma[batch_idx, length - 1] = prompt_len + image_tokens + 2
        spans.append((batch_idx, image_start, image_end))

    return (
        input_ids.to(device),
        token_types.to(device),
        sigma.to(device),
        spans,
    )


def make_initial_noise(
    records: list[dict], image_tokens: int, latent_dim: int, seed: int
) -> tuple[torch.Tensor, list[int]]:
    rows = []
    seeds = []
    for index, record in enumerate(records):
        noise_seed = int(record.get("seed", seed + index))
        generator = torch.Generator(device="cpu")
        generator.manual_seed(noise_seed)
        rows.append(
            torch.randn(
                image_tokens,
                latent_dim,
                generator=generator,
                dtype=torch.float32,
            )
        )
        seeds.append(noise_seed)
    return torch.stack(rows), seeds


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def cache_trace_summary(trace: dict) -> dict:
    keys = (
        "backbone_kv_cache_enabled",
        "backbone_kv_cache_fallback_reason",
        "backbone_kv_cache_context_tokens",
        "backbone_kv_cache_tokens_committed",
        "backbone_kv_cache_peak_bytes",
        "flow_content_cache_peak_bytes_per_sample",
    )
    return {key: trace.get(key) for key in keys}


def load_font(size: int):
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ):
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def make_contact_sheet(
    records: list[dict], image_paths: list[Path], output_path: Path, cache_mode: str
) -> None:
    columns = 4
    panel_width = 320
    image_size = 320
    caption_height = 126
    header_height = 46
    rows = (len(records) + columns - 1) // columns
    sheet = Image.new(
        "RGB",
        (columns * panel_width, header_height + rows * (image_size + caption_height)),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    title_font = load_font(18)
    body_font = load_font(13)
    draw.text(
        (12, 11),
        f"Final EMA caption samples | backbone KV cache: {cache_mode.upper()}",
        fill="black",
        font=title_font,
    )
    for index, (record, image_path) in enumerate(zip(records, image_paths)):
        row, column = divmod(index, columns)
        x = column * panel_width
        y = header_height + row * (image_size + caption_height)
        image = Image.open(image_path).convert("RGB").resize(
            (image_size, image_size), Image.Resampling.LANCZOS
        )
        sheet.paste(image, (x, y))
        label = f"[{record['category']}] {record['id']}\n{record['caption']}"
        wrapped = "\n".join(textwrap.wrap(label, width=43))
        draw.multiline_text(
            (x + 7, y + image_size + 7),
            wrapped,
            fill="black",
            font=body_font,
            spacing=3,
        )
    sheet.save(output_path)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    records = read_prompts(Path(args.prompts_jsonl))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    model_dtype = torch.bfloat16 if args.model_dtype == "bf16" else torch.float32
    use_backbone_cache = args.cache_mode == "on"
    torch.manual_seed(args.seed)

    config = OmegaConf.load(args.config)
    config.model.model_path = args.checkpoint
    config.model.image_flow_num_sampling_steps = str(args.sampling_steps)
    config.training.from_scratch = False
    config.training.use_gradient_checkpointing = False

    load_started = time.perf_counter()
    print(f"Loading final model from {args.checkpoint}", flush=True)
    model, tokenizer = load_model_tokenizer(config, model_dtype=model_dtype)
    model = model.to(device).eval()
    print("Loading KL16 VAE", flush=True)
    vae = load_vae(config, device, args.vae_dtype)
    synchronize(device)
    load_seconds = time.perf_counter() - load_started

    input_ids, token_types, sigma, spans = build_caption_batch(
        tokenizer, model.config, records, device
    )
    image_tokens = int(model.config.image_tokens_per_img)
    latent_dim = int(model.config.image_latent_dim)
    initial_noise, noise_seeds = make_initial_noise(
        records, image_tokens, latent_dim, args.seed
    )

    sample_kwargs = {
        "input_ids": input_ids,
        "token_types": token_types,
        "sigma": sigma,
        "spans": spans,
        "image_latent_dim": latent_dim,
        "initial_noise_bank": initial_noise,
        "flow_temperature": args.temperature,
        "flow_cfg": args.cfg,
        "flow_cfg_schedule": args.cfg_schedule,
        "flow_solver": args.flow_solver,
        "flow_num_steps": args.sampling_steps,
        "parallel_rate": 1,
        "order_strategy": args.order_strategy,
        "use_backbone_cache": use_backbone_cache,
        "return_trace": True,
    }

    warmup_seconds = 0.0
    if args.warmup_generation_steps > 0:
        print(
            f"Warming up {args.cache_mode} mode for "
            f"{args.warmup_generation_steps} image-token step(s)",
            flush=True,
        )
        synchronize(device)
        warmup_started = time.perf_counter()
        with torch.inference_mode():
            model.sample_image_latents_single_stream(
                **sample_kwargs,
                _debug_max_generation_steps=args.warmup_generation_steps,
            )
        synchronize(device)
        warmup_seconds = time.perf_counter() - warmup_started

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    print(
        f"Generating {len(records)} images with backbone KV cache {args.cache_mode}",
        flush=True,
    )
    synchronize(device)
    generation_started = time.perf_counter()
    with torch.inference_mode():
        latents, trace = model.sample_image_latents_single_stream(**sample_kwargs)
    synchronize(device)
    generation_seconds = time.perf_counter() - generation_started
    peak_allocated = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
    )
    peak_reserved = (
        int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None
    )

    expected_cache_state = use_backbone_cache
    actual_cache_state = bool(trace.get("backbone_kv_cache_enabled"))
    if actual_cache_state != expected_cache_state:
        raise RuntimeError(
            "Backbone cache state mismatch: "
            f"requested={expected_cache_state}, trace={cache_trace_summary(trace)}"
        )

    decode_started = time.perf_counter()
    scaling_factor = float(config.experiment.validation_vae_scaling_factor)
    decoded_chunks = []
    with torch.inference_mode():
        for chunk in latents.split(args.decode_batch_size):
            decoded_chunks.append(decode_latents(vae, chunk.float(), scaling_factor).cpu())
    synchronize(device)
    decode_seconds = time.perf_counter() - decode_started
    images = torch.cat(decoded_chunks)

    image_paths = []
    for index, (record, image) in enumerate(zip(records, images)):
        safe_id = "".join(
            character if character.isalnum() or character in "-_" else "_"
            for character in str(record["id"])
        )
        image_path = image_dir / f"{index:02d}_{safe_id}.png"
        save_image(image, image_path)
        image_paths.append(image_path)

    latents_cpu = latents.detach().float().cpu()
    torch.save(latents_cpu, output_dir / "latents.pt")
    contact_sheet_path = output_dir / "contact_sheet.png"
    make_contact_sheet(records, image_paths, contact_sheet_path, args.cache_mode)

    checkpoint_path = Path(args.checkpoint).resolve()
    model_file = checkpoint_path / "model.safetensors"
    checkpoint_sha256 = None
    if model_file.exists():
        digest = hashlib.sha256()
        with model_file.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        checkpoint_sha256 = digest.hexdigest()

    prompt_metadata = []
    for record, noise_seed, image_path in zip(records, noise_seeds, image_paths):
        item = dict(record)
        item["noise_seed"] = noise_seed
        item["image"] = str(image_path.resolve())
        prompt_metadata.append(item)
    total_image_tokens = len(records) * image_tokens
    metrics = {
        "schema": "caption_kv_cache_probe_v1",
        "config": str(Path(args.config).resolve()),
        "checkpoint": str(checkpoint_path),
        "checkpoint_model_sha256": checkpoint_sha256,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "model_dtype": args.model_dtype,
        "vae_dtype": args.vae_dtype,
        "batch_size": len(records),
        "sequence_length": int(input_ids.shape[1]),
        "image_tokens_per_sample": image_tokens,
        "sampling_steps": args.sampling_steps,
        "temperature": args.temperature,
        "cfg": args.cfg,
        "cfg_schedule": args.cfg_schedule,
        "flow_solver": args.flow_solver,
        "order_strategy": args.order_strategy,
        "cache_mode": args.cache_mode,
        "cache_trace": cache_trace_summary(trace),
        "timing": {
            "model_and_vae_load_seconds": load_seconds,
            "warmup_seconds": warmup_seconds,
            "generation_seconds": generation_seconds,
            "generation_seconds_per_image": generation_seconds / len(records),
            "image_tokens_per_second": total_image_tokens / generation_seconds,
            "vae_decode_seconds": decode_seconds,
        },
        "cuda_peak_allocated_bytes": peak_allocated,
        "cuda_peak_reserved_bytes": peak_reserved,
        "latent": {
            "shape": list(latents_cpu.shape),
            "mean": float(latents_cpu.mean().item()),
            "std": float(latents_cpu.std(unbiased=False).item()),
            "rms": float(latents_cpu.square().mean().sqrt().item()),
            "finite": bool(torch.isfinite(latents_cpu).all().item()),
        },
        "contact_sheet": str(contact_sheet_path.resolve()),
        "prompts": prompt_metadata,
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics["timing"], indent=2), flush=True)
    print(json.dumps(metrics["cache_trace"], indent=2), flush=True)
    print(f"Saved contact sheet to {contact_sheet_path}", flush=True)


if __name__ == "__main__":
    main()
