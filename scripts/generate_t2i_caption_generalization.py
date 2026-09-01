#!/usr/bin/env python3
"""Generate a paired caption-generalization gallery from the three T2I models.

The script deliberately follows the formal ImageNet-1K T2I generation contract:
the training-time T2I prefix, 10-step Heun sampling, CFG 3.5, one-token
single-stream generation, the variant-specific reveal strategy, and KL16 VAE
decoding.  Every model receives the same per-caption initial noise.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import textwrap
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.image_generation_io import decode_latents, load_vae
from utils.imagenet_flow_batching import collate_imagenet_flow_cache
from utils.utils import load_model_tokenizer

VARIANTS: dict[str, dict[str, str]] = {
    "baseline": {
        "label": "Baseline contextual",
        "config": (
            "configs/selfless/imagenet1k_t2i_baseline_80ep_ascend_64npu_bs1024.yaml"
        ),
        "checkpoint": (
            "output/selfless-flow-imagenet1k-t2i-baseline-"
            "ascend64-b1024-80ep/hf_model-final-ema"
        ),
        "image_sigma_order": "random",
        "generation_strategy": "spatial_halton",
    },
    "positionwise_head": {
        "label": "Position-wise flow head",
        "config": (
            "configs/selfless/imagenet1k_t2i_positionwise_head_80ep_"
            "ascend_64npu_bs1024.yaml"
        ),
        "checkpoint": (
            "output/selfless-flow-imagenet1k-t2i-positionwise-head-"
            "ascend64-b1024-80ep/hf_model-final-ema"
        ),
        "image_sigma_order": "random",
        "generation_strategy": "spatial_halton",
    },
    "seq_sigma": {
        "label": "Sequential sigma",
        "config": (
            "configs/selfless/imagenet1k_t2i_seq_sigma_80ep_ascend_64npu_bs1024.yaml"
        ),
        "checkpoint": (
            "output/selfless-flow-imagenet1k-t2i-seq-sigma-"
            "ascend64-b1024-80ep/hf_model-final-ema"
        ),
        "image_sigma_order": "sequential",
        "generation_strategy": "sequential",
    },
}


DEFAULT_PROMPTS: list[dict[str, str]] = [
    {
        "id": "seen_objects_unseen_binding",
        "axis": "attribute_binding",
        "caption": (
            "A bright blue monarch butterfly perched on a ripe red strawberry, "
            "macro wildlife photograph with a softly blurred green background."
        ),
    },
    {
        "id": "two_subject_color_binding",
        "axis": "multi_subject_binding",
        "caption": (
            "Two golden retrievers sit beside a yellow bicycle; the dog on the "
            "left wears a red bandana and the dog on the right wears a blue bandana."
        ),
    },
    {
        "id": "exact_counting",
        "axis": "counting",
        "caption": (
            "Exactly three white ceramic teacups arranged in a straight row on a "
            "dark wooden table, studio photograph."
        ),
    },
    {
        "id": "left_right_relation",
        "axis": "spatial_relation",
        "caption": (
            "A glossy blue ceramic sphere is to the left of a matte red wooden "
            "cube on a plain gray background."
        ),
    },
    {
        "id": "panda_astronaut",
        "axis": "novel_composition",
        "caption": (
            "A tiny panda wearing a silver astronaut suit rides a skateboard on "
            "the moon, with Earth visible in the black sky."
        ),
    },
    {
        "id": "transparent_material",
        "axis": "material_and_containment",
        "caption": (
            "A transparent glass teapot filled with glowing orange flowers on a "
            "marble windowsill, morning sunlight and realistic reflections."
        ),
    },
    {
        "id": "overhead_scene",
        "axis": "scene_composition",
        "caption": (
            "Overhead photograph of a breakfast table with one croissant, a bowl "
            "of blueberries, a green notebook, and a silver camera."
        ),
    },
    {
        "id": "weather_action",
        "axis": "action_and_weather",
        "caption": (
            "A red fox leaping across a narrow stream during heavy snowfall at "
            "blue hour, crisp nature photography."
        ),
    },
    {
        "id": "watercolor_style",
        "axis": "style_transfer",
        "caption": (
            "A loose watercolor painting of a quiet coastal village beneath "
            "purple storm clouds, visible paper texture and soft pigment blooms."
        ),
    },
    {
        "id": "pixel_art_style",
        "axis": "style_transfer",
        "caption": (
            "Colorful 16-bit pixel art of a small robot watering sunflowers in a "
            "rooftop garden at sunset."
        ),
    },
    {
        "id": "rendered_text",
        "axis": "text_rendering",
        "caption": (
            "A black street sign with the clearly readable white words NORTH STAR, "
            "photographed at night under a warm lamp."
        ),
    },
    {
        "id": "futuristic_architecture",
        "axis": "out_of_domain_scene",
        "caption": (
            "A futuristic glass greenhouse floating above a desert canyon, filled "
            "with lush tropical plants, cinematic wide-angle photograph."
        ),
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("output/t2i-caption-generalization-20260826"),
    )
    parser.add_argument(
        "--variants",
        default=",".join(VARIANTS),
        help="Comma-separated subset of baseline,positionwise_head,seq_sigma.",
    )
    parser.add_argument(
        "--prompts_json",
        type=Path,
        default=None,
        help="Optional JSON list of {id, axis, caption} objects.",
    )
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--seed", type=int, default=424242)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--sampling_steps", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--cfg", type=float, default=3.5)
    parser.add_argument(
        "--cfg_schedule", choices=("constant", "linear"), default="constant"
    )
    parser.add_argument("--flow_solver", choices=("heun", "euler"), default="heun")
    parser.add_argument(
        "--debug_finite",
        action="store_true",
        help="Enable expensive per-layer finite checks inside generation.",
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def load_prompts(path: Path | None) -> list[dict[str, str]]:
    rows: Any = DEFAULT_PROMPTS if path is None else json.loads(path.read_text())
    if not isinstance(rows, list) or not rows:
        raise ValueError("prompt JSON must be a non-empty list")
    prompts: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise TypeError(f"prompt row {index} must be an object")
        prompt_id = str(row.get("id", "")).strip()
        axis = str(row.get("axis", "")).strip()
        caption = str(row.get("caption", "")).strip()
        if not prompt_id or not axis or not caption:
            raise ValueError(f"prompt row {index} has an empty id, axis, or caption")
        if prompt_id in seen:
            raise ValueError(f"duplicate prompt id: {prompt_id}")
        if any(
            character not in "abcdefghijklmnopqrstuvwxyz0123456789_"
            for character in prompt_id
        ):
            raise ValueError(f"unsafe prompt id: {prompt_id!r}")
        seen.add(prompt_id)
        prompts.append({"id": prompt_id, "axis": axis, "caption": caption})
    return prompts


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    tensor = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
    header = json.dumps(
        {"dtype": str(tensor.dtype), "shape": list(tensor.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(header + b"\n")
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def initial_noise_for_prompt(
    prompt_index: int,
    *,
    seed: int,
    image_tokens: int,
    latent_dim: int,
) -> torch.Tensor:
    side = math.isqrt(image_tokens)
    if side * side != image_tokens:
        raise ValueError(f"image token count is not square: {image_tokens}")
    generator = torch.Generator(device="cpu")
    generator.manual_seed((int(seed) + int(prompt_index)) % (2**63))
    return torch.randn(
        (side, side, latent_dim),
        generator=generator,
        device="cpu",
        dtype=torch.float32,
    ).reshape(image_tokens, latent_dim)


def build_prompt_item(
    *,
    tokenizer: Any,
    model: Any,
    caption: str,
    prompt_index: int,
    prefix: str,
    image_sigma_order: str,
    seed: int,
) -> tuple[dict[str, Any], int]:
    serialized = f"{prefix} {caption}"
    prefix_ids = torch.tensor(
        tokenizer.encode(serialized, add_special_tokens=False), dtype=torch.long
    )
    image_tokens = int(model.config.image_tokens_per_img)
    latent_dim = int(model.config.image_latent_dim)
    boi_id = int(model.config.boi_token_id)
    eoi_id = int(model.config.eoi_token_id)
    mask_id = int(model.config.mask_token_id)
    eos_id = int(tokenizer.eos_token_id)
    image_block = torch.full((image_tokens + 2,), mask_id, dtype=torch.long)
    image_block[0] = boi_id
    image_block[-1] = eoi_id
    input_ids = torch.cat([prefix_ids, image_block, torch.tensor([eos_id])])
    token_types = torch.cat(
        [
            torch.zeros(prefix_ids.numel(), dtype=torch.uint8),
            torch.tensor([2], dtype=torch.uint8),
            torch.ones(image_tokens, dtype=torch.uint8),
            torch.tensor([2, 2], dtype=torch.uint8),
        ]
    )
    image_start = int(prefix_ids.numel()) + 1
    image_loss_mask = torch.zeros(input_ids.numel(), dtype=torch.bool)
    image_loss_mask[image_start : image_start + image_tokens] = True
    return (
        {
            "input_ids": input_ids,
            "token_types": token_types,
            "labels": torch.full_like(input_ids, -100),
            "image_loss_mask": image_loss_mask,
            "image_latents": torch.zeros(image_tokens, latent_dim),
            "prompt_len": torch.tensor(prefix_ids.numel()),
            "suffix_len": torch.tensor(0),
            "image_start": torch.tensor(image_start),
            "img_id": torch.tensor(prompt_index + 1),
            "task_mode": "t2i",
            "reveal_seed": torch.tensor((int(seed) + prompt_index) % (2**63)),
            "image_sigma_order": image_sigma_order,
        },
        int(prefix_ids.numel()),
    )


def synchronize(device: torch.device) -> None:
    if device.type == "npu":
        torch.npu.synchronize(device)
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def tensor_to_png(image: torch.Tensor, path: Path) -> None:
    pixels = (
        image.detach()
        .float()
        .clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels, mode="RGB").save(path, format="PNG", optimize=False)


def checkpoint_metadata(checkpoint: Path) -> dict[str, Any]:
    export_metadata_path = checkpoint / "ema_export_metadata.json"
    export_metadata = (
        json.loads(export_metadata_path.read_text())
        if export_metadata_path.is_file()
        else None
    )
    weights_path = checkpoint / "model.safetensors"
    return {
        "path": str(checkpoint.resolve()),
        "weights_bytes": weights_path.stat().st_size,
        "config_sha256": file_sha256(checkpoint / "config.json"),
        "ema_export_metadata": export_metadata,
    }


def load_font(size: int) -> ImageFont.ImageFont:
    candidates = (
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/dejavu/DejaVuSans.ttf"),
    )
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def draw_wrapped_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    *,
    font: ImageFont.ImageFont,
    width_chars: int,
    fill: str = "black",
    spacing: int = 4,
) -> None:
    draw.multiline_text(
        xy,
        textwrap.fill(text, width=width_chars),
        font=font,
        fill=fill,
        spacing=spacing,
    )


def make_variant_sheet(
    output_dir: Path,
    variant: str,
    prompts: list[dict[str, str]],
) -> Path:
    columns = 4
    image_size = 256
    label_height = 86
    rows = math.ceil(len(prompts) / columns)
    sheet = Image.new(
        "RGB", (columns * image_size, rows * (image_size + label_height)), "white"
    )
    draw = ImageDraw.Draw(sheet)
    font = load_font(15)
    for index, prompt in enumerate(prompts):
        x = (index % columns) * image_size
        y = (index // columns) * (image_size + label_height)
        image_path = output_dir / variant / "images" / f"{index:02d}_{prompt['id']}.png"
        with Image.open(image_path) as image:
            sheet.paste(image.convert("RGB"), (x, y))
        draw.rectangle(
            (x, y + image_size, x + image_size, y + image_size + label_height),
            fill="white",
        )
        draw_wrapped_text(
            draw,
            (x + 6, y + image_size + 5),
            f"{index:02d} [{prompt['axis']}] {prompt['caption']}",
            font=font,
            width_chars=34,
        )
    path = output_dir / f"contact_sheet_{variant}.png"
    sheet.save(path, format="PNG", optimize=False)
    return path


def make_comparison_sheet(
    output_dir: Path,
    variants: list[str],
    prompts: list[dict[str, str]],
) -> Path:
    caption_width = 360
    image_size = 256
    header_height = 54
    sheet = Image.new(
        "RGB",
        (
            caption_width + len(variants) * image_size,
            header_height + len(prompts) * image_size,
        ),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    header_font = load_font(18)
    body_font = load_font(15)
    draw.text((10, 15), "Caption / generalization axis", fill="black", font=header_font)
    for column, variant in enumerate(variants):
        x = caption_width + column * image_size
        draw.text(
            (x + 8, 15), VARIANTS[variant]["label"], fill="black", font=header_font
        )
    for row, prompt in enumerate(prompts):
        y = header_height + row * image_size
        fill = "#f3f3f3" if row % 2 else "white"
        draw.rectangle((0, y, caption_width, y + image_size), fill=fill)
        draw_wrapped_text(
            draw,
            (10, y + 10),
            f"{row:02d} [{prompt['axis']}]\n{prompt['caption']}",
            font=body_font,
            width_chars=44,
        )
        for column, variant in enumerate(variants):
            image_path = (
                output_dir / variant / "images" / f"{row:02d}_{prompt['id']}.png"
            )
            with Image.open(image_path) as image:
                sheet.paste(
                    image.convert("RGB"),
                    (caption_width + column * image_size, y),
                )
    path = output_dir / "comparison_all_models.png"
    sheet.save(path, format="PNG", optimize=False)
    return path


def generate_variant(
    *,
    variant: str,
    prompts: list[dict[str, str]],
    args: argparse.Namespace,
    device: torch.device,
    vae: Any | None,
) -> tuple[Any, dict[str, Any]]:
    spec = VARIANTS[variant]
    config_path = Path(spec["config"])
    checkpoint = Path(spec["checkpoint"])
    for required in (
        config_path,
        checkpoint / "config.json",
        checkpoint / "model.safetensors",
        checkpoint / "tokenizer.json",
    ):
        if not required.is_file():
            raise FileNotFoundError(required)

    print(f"[{variant}] loading {checkpoint}", flush=True)
    config = OmegaConf.load(config_path)
    config.model.model_path = str(checkpoint)
    config.model.image_flow_num_sampling_steps = str(args.sampling_steps)
    config.training.from_scratch = False
    config.training.use_gradient_checkpointing = False
    model, tokenizer = load_model_tokenizer(config, model_dtype=torch.bfloat16)
    model = model.to(device).eval()
    if vae is None:
        vae = load_vae(config, device, "fp32")
    scaling_factor = float(config.experiment.validation_vae_scaling_factor)

    image_tokens = int(model.config.image_tokens_per_img)
    latent_dim = int(model.config.image_latent_dim)
    prefix = str(config.dataset.params.caption_t2i_prefix).strip()
    pad_to_length = int(config.dataset.params.pad_to_length)
    variant_dir = args.output_dir / variant
    image_dir = variant_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any] | None] = [None] * len(prompts)
    pending_indices: list[int] = []
    for index, prompt in enumerate(prompts):
        image_path = image_dir / f"{index:02d}_{prompt['id']}.png"
        if args.resume and image_path.is_file():
            with Image.open(image_path) as image:
                width, height = image.size
            records[index] = {
                "prompt_index": index,
                "id": prompt["id"],
                "axis": prompt["axis"],
                "caption": prompt["caption"],
                "image": str(image_path.resolve()),
                "image_sha256": file_sha256(image_path),
                "width": width,
                "height": height,
                "resumed": True,
            }
        else:
            pending_indices.append(index)

    started = time.perf_counter()
    with torch.inference_mode():
        for batch_start in range(0, len(pending_indices), int(args.batch_size)):
            indices = pending_indices[batch_start : batch_start + int(args.batch_size)]
            items = []
            token_lengths = []
            noise_rows = []
            for index in indices:
                item, prompt_tokens = build_prompt_item(
                    tokenizer=tokenizer,
                    model=model,
                    caption=prompts[index]["caption"],
                    prompt_index=index,
                    prefix=prefix,
                    image_sigma_order=spec["image_sigma_order"],
                    seed=args.seed,
                )
                if int(item["input_ids"].numel()) > pad_to_length:
                    raise ValueError(
                        f"prompt {index} serializes to {item['input_ids'].numel()} "
                        f"tokens, exceeding pad_to_length={pad_to_length}"
                    )
                items.append(item)
                token_lengths.append(prompt_tokens)
                noise_rows.append(
                    initial_noise_for_prompt(
                        index,
                        seed=args.seed,
                        image_tokens=image_tokens,
                        latent_dim=latent_dim,
                    )
                )
            batch = collate_imagenet_flow_cache(items, pad_to_length=pad_to_length)
            input_ids = batch["input_ids"].to(device)
            token_types = batch["token_types"].to(device)
            sigma = batch["sigma"].to(device)
            spans = [
                (row, int(item["image_start"]), int(item["image_start"]) + image_tokens)
                for row, item in enumerate(items)
            ]
            initial_noise = torch.stack(noise_rows)

            synchronize(device)
            batch_started = time.perf_counter()
            latents, trace = model.generate(
                "t2i",
                input_ids=input_ids,
                token_types=token_types,
                sigma=sigma,
                spans=spans,
                image_latent_dim=latent_dim,
                initial_noise_bank=initial_noise,
                flow_temperature=float(args.temperature),
                flow_cfg=float(args.cfg),
                flow_cfg_schedule=str(args.cfg_schedule),
                flow_solver=str(args.flow_solver),
                flow_num_steps=int(args.sampling_steps),
                parallel_rate=1,
                order_strategy=spec["generation_strategy"],
                use_cache=True,
                return_trace=True,
                debug_finite=bool(args.debug_finite),
            )
            synchronize(device)
            generation_seconds = time.perf_counter() - batch_started
            if latents is None or not bool(torch.isfinite(latents).all().item()):
                raise FloatingPointError(f"{variant} generated invalid latents")
            decoded = decode_latents(vae, latents.float(), scaling_factor)
            if not bool(torch.isfinite(decoded).all().item()):
                raise FloatingPointError(f"{variant} VAE decode is non-finite")

            seconds_per_image = generation_seconds / len(indices)
            for row, index in enumerate(indices):
                prompt = prompts[index]
                image_path = image_dir / f"{index:02d}_{prompt['id']}.png"
                tensor_to_png(decoded[row], image_path)
                records[index] = {
                    "prompt_index": index,
                    "id": prompt["id"],
                    "axis": prompt["axis"],
                    "caption": prompt["caption"],
                    "serialized_prompt": f"{prefix} {prompt['caption']}",
                    "prompt_tokens": token_lengths[row],
                    "initial_noise_seed": int(args.seed) + index,
                    "initial_noise_sha256": tensor_sha256(noise_rows[row]),
                    "image": str(image_path.resolve()),
                    "image_sha256": file_sha256(image_path),
                    "width": 256,
                    "height": 256,
                    "latent_mean": float(latents[row].float().mean().item()),
                    "latent_std": float(
                        latents[row].float().std(unbiased=False).item()
                    ),
                    "pixel_mean": float(decoded[row].float().mean().item()),
                    "pixel_std": float(decoded[row].float().std(unbiased=False).item()),
                    "generation_seconds_per_image_in_batch": seconds_per_image,
                    "resumed": False,
                }
            print(
                json.dumps(
                    {
                        "variant": variant,
                        "completed": indices,
                        "generation_seconds": generation_seconds,
                        "backbone_kv_cache_enabled": trace.get(
                            "backbone_kv_cache_enabled"
                        ),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    if any(record is None for record in records):
        raise RuntimeError(f"{variant} did not produce every requested image")
    variant_manifest = {
        "variant": variant,
        "label": spec["label"],
        "config": str(config_path.resolve()),
        "checkpoint": checkpoint_metadata(checkpoint),
        "architecture_variant": str(
            config.model.get("architecture_variant", "selfless_contextual")
        ),
        "image_sigma_order": spec["image_sigma_order"],
        "generation_strategy": spec["generation_strategy"],
        "elapsed_seconds": time.perf_counter() - started,
        "samples": records,
    }
    (variant_dir / "manifest.json").write_text(
        json.dumps(variant_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"[{variant}] saved {len(records)} raw PNGs", flush=True)
    del model, tokenizer
    if device.type == "npu":
        torch.npu.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()
    return vae, variant_manifest


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive")
    if args.sampling_steps <= 0:
        raise ValueError("--sampling_steps must be positive")
    variants = [item.strip() for item in args.variants.split(",") if item.strip()]
    if not variants or len(variants) != len(set(variants)):
        raise ValueError("--variants must contain unique variant names")
    unknown = sorted(set(variants) - set(VARIANTS))
    if unknown:
        raise ValueError(f"unknown variants: {unknown}")
    prompts = load_prompts(args.prompts_json)
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.resume:
        raise FileExistsError(
            f"output directory is not empty: {args.output_dir}; pass --resume to reuse PNGs"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "prompts.json").write_text(
        json.dumps(prompts, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    if args.device.startswith("npu"):
        import torch_npu  # noqa: F401

    device = torch.device(args.device)
    if device.type == "npu":
        if not torch.npu.is_available():
            raise RuntimeError("Ascend NPU is unavailable")
        torch.npu.set_device(device)
    started_at = datetime.now(UTC)
    vae = None
    variant_manifests = []
    for variant in variants:
        vae, manifest = generate_variant(
            variant=variant,
            prompts=prompts,
            args=args,
            device=device,
            vae=vae,
        )
        variant_manifests.append(manifest)

    variant_sheets = [
        str(make_variant_sheet(args.output_dir, variant, prompts).resolve())
        for variant in variants
    ]
    comparison_sheet = make_comparison_sheet(args.output_dir, variants, prompts)
    completed_at = datetime.now(UTC)
    manifest = {
        "schema": "t2i_caption_generalization_gallery_v1",
        "created_at_utc": started_at.isoformat(),
        "completed_at_utc": completed_at.isoformat(),
        "protocol": {
            "caption_prefix": "Generate an image matching this description:",
            "sampling_steps": int(args.sampling_steps),
            "temperature": float(args.temperature),
            "cfg": float(args.cfg),
            "cfg_schedule": str(args.cfg_schedule),
            "flow_solver": str(args.flow_solver),
            "parallel_rate": 1,
            "seed": int(args.seed),
            "paired_initial_noise": True,
            "model_dtype": "bfloat16",
            "vae_dtype": "float32",
            "image_size": [256, 256],
        },
        "prompts": prompts,
        "variants": variant_manifests,
        "artifacts": {
            "comparison_sheet": str(comparison_sheet.resolve()),
            "variant_sheets": variant_sheets,
        },
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "manifest": str(manifest_path.resolve()),
                "comparison_sheet": str(comparison_sheet.resolve()),
                "raw_pngs": len(prompts) * len(variants),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
