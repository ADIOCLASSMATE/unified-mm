#!/usr/bin/env python3
import argparse
import json
import math
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("DIFFUSERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchvision import transforms
from torchvision.utils import save_image
from tqdm.auto import tqdm

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.generate_flow_validation_images import (  # noqa: E402
    decode_latents,
    load_adapter,
    load_ema_state,
    load_model_state,
    load_vae,
)
from utils.dataset_utils import get_dataloaders  # noqa: E402
from utils.utils import load_model_tokenizer  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Offline FID/IS evaluation for selfless single-stream image generation."
    )
    parser.add_argument("--config", default="configs/selfless/imagenet_flow_full_from_qwen3base.yaml")
    parser.add_argument("--model_path_override", default="")
    parser.add_argument("--adapter", default="none")
    parser.add_argument("--model_state", default="")
    parser.add_argument("--ema_state", default="")
    parser.add_argument("--output_dir", default="output/single_stream_fid_is")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--split", choices=["val", "train"], default="val")
    parser.add_argument("--sampling_steps", default="50")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--cfg", type=float, default=1.0)
    parser.add_argument("--cfg_schedule", choices=["constant", "linear"], default="constant")
    parser.add_argument("--flow_solver", choices=["heun", "euler"], default="heun")
    parser.add_argument("--parallel_rate", type=int, default=4)
    parser.add_argument("--strategies", default="causal_sigma,spatial_halton")
    parser.add_argument("--vae_dtype", choices=["fp32", "fp16"], default="fp32")
    parser.add_argument("--fid_feature", type=int, default=2048)
    parser.add_argument("--is_splits", type=int, default=10)
    parser.add_argument(
        "--real_source",
        choices=["vae_decoded_target_latents", "imagenet_original"],
        default="vae_decoded_target_latents",
        help=(
            "Reference distribution for FID. The default compares against VAE-decoded target latents; "
            "imagenet_original loads original ImageNet files from the manifest/source_path mapping."
        ),
    )
    parser.add_argument(
        "--imagenet_train_dir",
        default="/inspire/dataset/imagenet/v1/ILSVRC/Data/CLS-LOC/train",
        help="Root used to resolve manifest source paths when --real_source=imagenet_original.",
    )
    parser.add_argument(
        "--real_image_size",
        type=int,
        default=256,
        help="Resize/center-crop size for original ImageNet real images.",
    )
    parser.add_argument("--save_images", action="store_true")
    parser.add_argument("--no_progress", action="store_true")
    return parser.parse_args()


def image_spans(token_types: torch.Tensor, image_tokens_per_img: int):
    spans = []
    bsz, seq_len = token_types.shape
    for b in range(bsz):
        pos = 0
        while pos < seq_len:
            if int(token_types[b, pos].item()) != 1:
                pos += 1
                continue
            start = pos
            while pos < seq_len and int(token_types[b, pos].item()) == 1:
                pos += 1
            end = pos
            if end - start == image_tokens_per_img:
                spans.append((b, start, end))
    return spans


def span_latents_to_chw(
    image_latents: torch.Tensor,
    spans: list[tuple[int, int, int]],
    side: int,
) -> torch.Tensor:
    latents = []
    for b, start, end in spans:
        latents.append(image_latents[b, start:end].view(side, side, -1).permute(2, 0, 1))
    return torch.stack(latents)


def metric_images(x: torch.Tensor) -> torch.Tensor:
    return x.detach().float().clamp(0.0, 1.0)


def save_batch_images(images: torch.Tensor, directory: Path, offset: int):
    directory.mkdir(parents=True, exist_ok=True)
    for idx, image in enumerate(images):
        save_image(image, directory / f"{offset + idx:08d}.png")


def get_base_dataset_and_indices(loader_dataset):
    if hasattr(loader_dataset, "dataset") and hasattr(loader_dataset, "indices"):
        return loader_dataset.dataset, list(loader_dataset.indices)
    return loader_dataset, None


def source_paths_for_spans(loader_dataset, batch_offset: int, spans, imagenet_train_dir: Path):
    base_dataset, subset_indices = get_base_dataset_and_indices(loader_dataset)
    if not hasattr(base_dataset, "img_ids") or not hasattr(base_dataset, "source_paths"):
        raise ValueError("--real_source=imagenet_original requires an ImageNetFlowCacheDataset or Subset of it.")

    paths = []
    for batch_row, _, _ in spans:
        dataset_row = batch_offset + int(batch_row)
        if subset_indices is not None:
            dataset_row = int(subset_indices[dataset_row])
        img_id = int(base_dataset.img_ids[dataset_row].item())
        source_path = base_dataset.source_paths.get(img_id)
        if not source_path:
            raise KeyError(f"No source_path in manifest for img_id={img_id}")
        path = Path(source_path)
        if not path.is_absolute():
            path = imagenet_train_dir / path
        if not path.exists():
            raise FileNotFoundError(path)
        paths.append(path)
    return paths


def build_real_image_transform(image_size: int):
    return transforms.Compose(
        [
            transforms.Resize(int(image_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(int(image_size)),
            transforms.ToTensor(),
        ]
    )


def load_real_images(paths, transform, device):
    images = []
    for path in paths:
        with Image.open(path) as image:
            images.append(transform(image.convert("RGB")))
    return torch.stack(images, dim=0).to(device=device, dtype=torch.float32)


def finite_or_none(value: float) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


@torch.no_grad()
def main():
    args = parse_args()
    progress = not args.no_progress
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    strategies = [item.strip() for item in args.strategies.split(",") if item.strip()]
    if not strategies:
        raise ValueError("--strategies must contain at least one strategy")

    config = OmegaConf.load(args.config)
    if args.model_path_override:
        config.model.model_path = args.model_path_override
    config.training.batch_size = int(args.batch_size)
    config.training.dataloader_workers = 0
    config.model.image_flow_num_sampling_steps = str(args.sampling_steps)

    print("Loading model/tokenizer...")
    model, tokenizer = load_model_tokenizer(config)
    print(f"Loading adapter: {args.adapter}")
    adapter_report = load_adapter(model, args.adapter)
    print(f"Loading model state: {args.model_state or 'none'}")
    model_state_report = load_model_state(model, args.model_state)
    print(f"Loading EMA state: {args.ema_state or 'none'}")
    ema_state_report = load_ema_state(model, args.ema_state)
    model = model.to(device).eval()

    print("Loading KL16 VAE...")
    vae = load_vae(config, device, args.vae_dtype)
    scaling_factor = float(config.experiment.validation_vae_scaling_factor)

    print(f"Loading {args.split} dataloader...")
    train_loader, val_loader = get_dataloaders(config, tokenizer)
    loader = val_loader if args.split == "val" else train_loader
    if args.real_source == "imagenet_original" and args.split != "val":
        raise ValueError("--real_source=imagenet_original currently expects --split val because train loader is shuffled.")
    real_transform = build_real_image_transform(args.real_image_size)
    imagenet_train_dir = Path(args.imagenet_train_dir)

    image_tokens = int(config.model.image_tokens_per_img)
    side = int(image_tokens ** 0.5)
    if side * side != image_tokens:
        raise ValueError(f"image_tokens_per_img={image_tokens} is not a square grid")

    metrics = {
        strategy: {
            "fid": FrechetInceptionDistance(
                feature=int(args.fid_feature),
                normalize=True,
            ).to(device),
            "is": InceptionScore(
                normalize=True,
                splits=int(args.is_splits),
            ).to(device),
            "latent_mse_sum": 0.0,
            "latent_rms_sum": 0.0,
            "count": 0,
        }
        for strategy in strategies
    }

    generated = 0
    iterator = tqdm(loader, desc="single-stream FID/IS", dynamic_ncols=True, disable=not progress)
    batch_offset = 0
    for batch_idx, batch in enumerate(iterator):
        if generated >= args.samples:
            break

        input_ids = batch["input_ids"].to(device)
        token_types = batch["token_types"].to(device)
        sigma = batch["sigma"].to(device)
        image_latents = batch["image_latents"].to(device)
        current_batch_size = int(input_ids.shape[0])
        spans = image_spans(token_types, image_tokens)
        if not spans:
            batch_offset += current_batch_size
            continue
        remaining = int(args.samples) - generated
        spans = spans[:remaining]

        target_latents = span_latents_to_chw(image_latents, spans, side)
        target_images = metric_images(decode_latents(vae, target_latents.float(), scaling_factor))
        if args.real_source == "imagenet_original":
            real_paths = source_paths_for_spans(loader.dataset, batch_offset, spans, imagenet_train_dir)
            real_images = load_real_images(real_paths, real_transform, device)
        else:
            real_images = target_images
        if args.save_images:
            save_batch_images(target_images.cpu(), out_dir / "target_decoded", generated)
            if args.real_source == "imagenet_original":
                save_batch_images(real_images.cpu(), out_dir / "imagenet_original_real", generated)

        for strategy_idx, strategy in enumerate(strategies):
            torch.manual_seed(int(args.seed) + batch_idx * 1009 + strategy_idx * 131071)
            single_latents, trace = model.sample_image_latents_single_stream(
                input_ids=input_ids,
                token_types=token_types,
                sigma=sigma,
                spans=spans,
                image_latent_dim=image_latents.shape[-1],
                flow_temperature=float(args.temperature),
                flow_cfg=float(args.cfg),
                flow_cfg_schedule=str(args.cfg_schedule),
                flow_solver=args.flow_solver,
                parallel_rate=int(args.parallel_rate),
                order_strategy=str(strategy),
                return_trace=True,
            )
            generated_images = metric_images(decode_latents(vae, single_latents.float(), scaling_factor))
            state = metrics[strategy]
            state["fid"].update(real_images, real=True)
            state["fid"].update(generated_images, real=False)
            state["is"].update(generated_images)
            count = int(generated_images.shape[0])
            state["latent_mse_sum"] += float(F.mse_loss(single_latents.float(), target_latents.float()).item()) * count
            state["latent_rms_sum"] += float(single_latents.float().pow(2).mean().sqrt().item()) * count
            state["count"] += count
            if trace and isinstance(trace.get("generation_step"), torch.Tensor):
                state["generation_step_max"] = float(trace["generation_step"].float().max().item())
            if args.save_images:
                save_batch_images(generated_images.cpu(), out_dir / str(strategy), generated)

        generated += len(spans)
        batch_offset += current_batch_size
        iterator.set_postfix_str(f"{generated}/{args.samples}", refresh=False)

    if generated == 0:
        raise RuntimeError("No complete image spans were evaluated.")

    results = {
        "config": args.config,
        "model_path": str(config.model.model_path),
        "adapter": adapter_report,
        "model_state": model_state_report,
        "ema_state": ema_state_report,
        "split": args.split,
        "samples_requested": int(args.samples),
        "samples_evaluated": int(generated),
        "real_source": str(args.real_source),
        "imagenet_train_dir": str(imagenet_train_dir) if args.real_source == "imagenet_original" else None,
        "real_image_size": int(args.real_image_size) if args.real_source == "imagenet_original" else None,
        "cfg": float(args.cfg),
        "cfg_schedule": str(args.cfg_schedule),
        "sampling_steps": str(args.sampling_steps),
        "temperature": float(args.temperature),
        "flow_solver": str(args.flow_solver),
        "parallel_rate": int(args.parallel_rate),
        "strategies": {},
    }

    for strategy, state in metrics.items():
        count = max(int(state["count"]), 1)
        is_mean, is_std = state["is"].compute()
        results["strategies"][strategy] = {
            "count": int(state["count"]),
            "fid": finite_or_none(state["fid"].compute().item()),
            "inception_score_mean": finite_or_none(is_mean.item()),
            "inception_score_std": finite_or_none(is_std.item()),
            "latent_mse_to_target": state["latent_mse_sum"] / count,
            "latent_rms": state["latent_rms_sum"] / count,
            "generation_step_max": state.get("generation_step_max"),
        }

    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results["strategies"], indent=2))
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
