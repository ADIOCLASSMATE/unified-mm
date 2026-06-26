#!/usr/bin/env python3
import argparse
import importlib.util
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from safetensors import safe_open
from tqdm.auto import tqdm
from torchvision.utils import make_grid, save_image

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.dataset_utils import get_dataloaders
from utils.utils import get_selfless_mask, load_model_tokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Manual validation image generation for MAR DiffLoss adapters.")
    parser.add_argument("--config", default="configs/selfless/imagenet_diffusion_warmup_full.yaml")
    parser.add_argument(
        "--adapter",
        default="output/selfless-diffusion-0.6B-imagenet-warmup-from-textft/image_diffusion_adapter-12000.pt",
        help="Warmup adapter .pt, MAR .safetensors, or 'none' to use randomly initialized current head.",
    )
    parser.add_argument("--output_dir", default="output/manual_diffusion_validation")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--split", choices=["val", "train"], default="val")
    parser.add_argument("--sampling_steps", default="100")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--cfg", type=float, default=1.0)
    parser.add_argument("--denoise_timestep", type=int, default=500)
    parser.add_argument("--full_clip", action="store_true", help="Save full diffusion sample with clip_denoised=True.")
    parser.add_argument("--full_no_clip", action="store_true", help="Save full diffusion sample with clip_denoised=False.")
    parser.add_argument("--single_stream", action="store_true", help="Also run single-stream iterative generation.")
    parser.add_argument("--single_stream_clip", action="store_true", help="Use clip_denoised=True for single-stream.")
    parser.add_argument("--parallel_rate", type=int, default=1)
    parser.add_argument(
        "--strategies",
        default="sigma,hidden_norm",
        help="Comma-separated single-stream order strategies.",
    )
    parser.add_argument("--vae_dtype", choices=["fp32", "fp16"], default="fp32")
    parser.add_argument("--save_individual", action="store_true")
    parser.add_argument("--no_progress", action="store_true", help="Disable tqdm progress bars.")
    return parser.parse_args()


def tensor_stats(x):
    x = x.detach().float()
    return {
        "finite": bool(torch.isfinite(x).all().item()),
        "min": float(x.min().item()),
        "max": float(x.max().item()),
        "mean": float(x.mean().item()),
        "std": float(x.std(unbiased=False).item()),
        "rms": float(x.pow(2).mean().sqrt().item()),
    }


def load_adapter(model, adapter_path: str):
    if adapter_path.lower() in {"none", "null", "false", ""}:
        return {"adapter": None}

    path = Path(adapter_path)
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix == ".safetensors":
        head_state = {}
        projector_state = {}
        projector_target = model.model.image_latent_proj.state_dict()
        projector_skipped = {}
        mar_mask_token = None

        def maybe_add_projector_key(name, value):
            if name not in projector_target:
                projector_skipped[name] = (tuple(value.shape), None)
                return
            target_shape = tuple(projector_target[name].shape)
            if tuple(value.shape) != target_shape:
                projector_skipped[name] = (tuple(value.shape), target_shape)
                return
            projector_state[name] = value

        with safe_open(str(path), framework="pt", device="cpu") as f:
            for key in f.keys():
                if key.startswith("diffloss."):
                    head_state[key[len("diffloss."):]] = f.get_tensor(key)
                elif key.startswith("image_diffusion_head."):
                    head_state[key[len("image_diffusion_head."):]] = f.get_tensor(key)
                elif key in {"z_proj.weight", "z_proj.bias", "z_proj_ln.weight", "z_proj_ln.bias"}:
                    maybe_add_projector_key(key, f.get_tensor(key))
                elif key == "encoder_pos_embed_learned":
                    pos = f.get_tensor(key).squeeze(0)
                    maybe_add_projector_key(
                        "image_pos_embed.weight",
                        pos[-model.model.image_latent_proj.image_tokens_per_img:],
                    )
                elif key == "diffusion_pos_embed_learned":
                    maybe_add_projector_key("diffusion_pos_embed.weight", f.get_tensor(key).squeeze(0))
                elif key == "mask_token":
                    mar_mask_token = f.get_tensor(key).reshape(-1)
        missing, unexpected = model.image_diffusion_head.load_state_dict(head_state, strict=False)
        report = {
            "adapter": str(path),
            "image_diffusion_head_keys": len(head_state),
            "missing": list(missing),
            "unexpected": list(unexpected),
            "image_latent_proj_keys": len(projector_state),
            "image_latent_proj_skipped": projector_skipped,
        }
        if projector_state:
            missing, unexpected = model.model.image_latent_proj.load_state_dict(projector_state, strict=False)
            report["image_latent_proj_missing"] = list(missing)
            report["image_latent_proj_unexpected"] = list(unexpected)
        image_mask_token_id = getattr(model.config, "image_mask_token_id", None)
        if mar_mask_token is not None and image_mask_token_id is not None:
            with torch.no_grad():
                embed = model.model.embed_tokens.weight
                if mar_mask_token.numel() == embed.shape[1]:
                    embed[int(image_mask_token_id)].copy_(mar_mask_token.to(device=embed.device, dtype=embed.dtype))
                    report["loaded_mar_mask_token_id"] = int(image_mask_token_id)
                else:
                    report["skipped_mar_mask_token_shape"] = [int(mar_mask_token.numel()), int(embed.shape[1])]
        return report

    state = torch.load(path, map_location="cpu")
    report = {"adapter": str(path)}
    if "image_diffusion_head" in state:
        missing, unexpected = model.image_diffusion_head.load_state_dict(state["image_diffusion_head"], strict=False)
        report["image_diffusion_head_missing"] = list(missing)
        report["image_diffusion_head_unexpected"] = list(unexpected)
    if "image_latent_proj" in state:
        missing, unexpected = model.model.image_latent_proj.load_state_dict(state["image_latent_proj"], strict=False)
        report["image_latent_proj_missing"] = list(missing)
        report["image_latent_proj_unexpected"] = list(unexpected)
    if "image_condition_null" in state and hasattr(model.model, "image_condition_null"):
        with torch.no_grad():
            model.model.image_condition_null.copy_(
                state["image_condition_null"].to(
                    device=model.model.image_condition_null.device,
                    dtype=model.model.image_condition_null.dtype,
                )
            )
        report["loaded_image_condition_null"] = True
    if "special_token_embeddings" in state and "special_token_ids" in state:
        with torch.no_grad():
            embed = model.model.embed_tokens.weight
            for name, token_id in state["special_token_ids"].items():
                if name in state["special_token_embeddings"]:
                    embed[int(token_id)].copy_(state["special_token_embeddings"][name].to(dtype=embed.dtype))
        report["loaded_special_token_embeddings"] = sorted(state["special_token_embeddings"].keys())
    return report


def load_vae(config, device, dtype_name):
    mar_root = Path(config.experiment.validation_mar_root)
    vae_path = Path(config.experiment.validation_vae_path)
    spec = importlib.util.spec_from_file_location("mar_vae", mar_root / "models" / "vae.py")
    mar_vae = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mar_vae)
    vae = mar_vae.AutoencoderKL(embed_dim=16, ch_mult=(1, 1, 2, 2, 4), ckpt_path=str(vae_path))
    dtype = torch.float16 if dtype_name == "fp16" and device.type == "cuda" else torch.float32
    vae = vae.to(device=device, dtype=dtype).eval()
    for param in vae.parameters():
        param.requires_grad_(False)
    return vae


def decode_latents(vae, latents, scaling_factor):
    vae_dtype = next(vae.parameters()).dtype
    decoded = vae.decode(latents.to(dtype=vae_dtype) / scaling_factor).float().clamp(-1, 1)
    return (decoded + 1.0) / 2.0


def main():
    args = parse_args()
    progress = not args.no_progress
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = OmegaConf.load(args.config)
    config.training.batch_size = args.batch_size
    config.training.dataloader_workers = 0
    config.model.image_diffusion_num_sampling_steps = str(args.sampling_steps)

    print("Loading model/tokenizer...")
    model, tokenizer = load_model_tokenizer(config)
    print(f"Loading adapter: {args.adapter}")
    adapter_report = load_adapter(model, args.adapter)
    model = model.to(device).eval()
    print("Loading MAR VAE...")
    vae = load_vae(config, device, args.vae_dtype)
    scaling_factor = float(config.experiment.validation_vae_scaling_factor)

    print(f"Loading {args.split} batch...")
    train_loader, val_loader = get_dataloaders(config, tokenizer)
    loader = val_loader if args.split == "val" else train_loader
    batch = next(iter(loader))
    input_ids = batch["input_ids"].to(device)
    token_types = batch["token_types"].to(device)
    labels = batch["labels"].to(device)
    sigma = batch["sigma"].to(device)
    image_latents = batch["image_latents"].to(device)

    attention_mask = get_selfless_mask(sigma=sigma, seq_len=input_ids.shape[1], device=device)
    print("Running teacher-forced forward pass...")
    with torch.no_grad():
        output = model(
            X0_input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            token_types=token_types,
            image_latents=image_latents,
            calculate_likelihood=True,
        )
        uncond_output = None
        if args.cfg != 1.0:
            uncond_output = model(
                X0_input_ids=input_ids,
                attention_mask=attention_mask,
                token_types=token_types,
                image_latents=image_latents,
                image_condition_drop=True,
                calculate_likelihood=True,
                return_logits=False,
            )

    image_tokens = int(config.model.image_tokens_per_img)
    side = int(image_tokens ** 0.5)
    spans = []
    for batch_idx in range(token_types.shape[0]):
        pos = (token_types[batch_idx] == 1).nonzero(as_tuple=True)[0]
        if pos.numel() == image_tokens:
            spans.append((batch_idx, int(pos[0].item()), int(pos[-1].item()) + 1))
    spans = spans[: args.samples]
    if not spans:
        raise RuntimeError("No complete image spans found in the selected batch.")

    strategies = [item.strip() for item in args.strategies.split(",") if item.strip()]
    metrics = {
        "adapter": adapter_report,
        "config": args.config,
        "sampling_steps": str(args.sampling_steps),
        "temperature": args.temperature,
        "cfg": args.cfg,
        "denoise_timestep": args.denoise_timestep,
        "loss": float(output.loss.detach().float().item()),
        "diffusion_stats": {
            key: float(value.detach().float().item())
            for key, value in getattr(output, "diffusion_debug_stats", {}).items()
        },
        "samples": [],
    }

    target_latents = []
    denoise_latents = []
    full_clip_latents = []
    full_no_clip_latents = []

    sample_iter = tqdm(
        list(enumerate(spans)),
        desc="Denoise/full sampling",
        dynamic_ncols=True,
        disable=not progress,
    )
    with torch.no_grad():
        for sample_idx, (batch_idx, start, end) in sample_iter:
            local_positions = torch.arange(end - start, device=device, dtype=torch.long)
            z = model._prepare_image_diffusion_condition(
                output.last_hidden_state[batch_idx, start:end],
                local_positions,
            )
            z_uncond = None
            if uncond_output is not None:
                z_uncond = model._prepare_image_diffusion_condition(
                    uncond_output.last_hidden_state[batch_idx, start:end],
                    local_positions,
                )
            target = image_latents[batch_idx, start:end].to(dtype=z.dtype)
            sample_iter.set_postfix_str("denoise_x0", refresh=False)

            timestep = max(0, min(args.denoise_timestep, model.image_diffusion_head.train_diffusion.num_timesteps - 1))
            t = torch.full((target.shape[0],), timestep, device=device, dtype=torch.long)
            torch.manual_seed(args.seed + 1000 + sample_idx)
            noise = torch.randn_like(target)
            noisy = model.image_diffusion_head.train_diffusion.q_sample(target, t, noise=noise)
            denoise = model.image_diffusion_head.train_diffusion.p_mean_variance(
                model.image_diffusion_head.net,
                noisy,
                t,
                clip_denoised=False,
                model_kwargs={"c": z},
            )["pred_xstart"].to(dtype=target.dtype)

            sample_metrics = {
                "index": sample_idx,
                "batch_index": batch_idx,
                "target": tensor_stats(target),
                "denoise_x0": tensor_stats(denoise),
                "denoise_x0_mse_to_target": float(F.mse_loss(denoise.float(), target.float()).item()),
            }

            target_latents.append(target.view(side, side, -1).permute(2, 0, 1))
            denoise_latents.append(denoise.view(side, side, -1).permute(2, 0, 1))

            if args.full_clip:
                sample_iter.set_postfix_str("full_clip=True", refresh=False)
                torch.manual_seed(args.seed + 2000 + sample_idx)
                full_clip = model.sample_image_diffusion_with_cfg(
                    z,
                    z_uncond=z_uncond,
                    temperature=args.temperature,
                    cfg=args.cfg,
                    clip_denoised=True,
                ).to(dtype=target.dtype)
                full_clip_latents.append(full_clip.view(side, side, -1).permute(2, 0, 1))
                sample_metrics["full_sample_clip_true"] = tensor_stats(full_clip)
                sample_metrics["full_sample_clip_true_mse_to_target"] = float(
                    F.mse_loss(full_clip.float(), target.float()).item()
                )

            if args.full_no_clip:
                sample_iter.set_postfix_str("full_clip=False", refresh=False)
                torch.manual_seed(args.seed + 2000 + sample_idx)
                full_no_clip = model.sample_image_diffusion_with_cfg(
                    z,
                    z_uncond=z_uncond,
                    temperature=args.temperature,
                    cfg=args.cfg,
                    clip_denoised=False,
                ).to(dtype=target.dtype)
                full_no_clip_latents.append(full_no_clip.view(side, side, -1).permute(2, 0, 1))
                sample_metrics["full_sample_clip_false"] = tensor_stats(full_no_clip)
                sample_metrics["full_sample_clip_false_mse_to_target"] = float(
                    F.mse_loss(full_no_clip.float(), target.float()).item()
                )

            metrics["samples"].append(sample_metrics)

    target_img = decode_latents(vae, torch.stack(target_latents).float(), scaling_factor)
    denoise_img = decode_latents(vae, torch.stack(denoise_latents).float(), scaling_factor)
    overview_columns = [("target", target_img), (f"denoise_x0_t{args.denoise_timestep}", denoise_img)]

    if full_no_clip_latents:
        full_no_clip_img = decode_latents(vae, torch.stack(full_no_clip_latents).float(), scaling_factor)
        overview_columns.append(("full_clip_false", full_no_clip_img))
        save_image(full_no_clip_img, out_dir / "full_sample_clip_false.png")
    if full_clip_latents:
        full_clip_img = decode_latents(vae, torch.stack(full_clip_latents).float(), scaling_factor)
        overview_columns.append(("full_clip_true", full_clip_img))
        save_image(full_clip_img, out_dir / "full_sample_clip_true.png")

    if args.single_stream and strategies:
        strategy_iter = tqdm(
            strategies,
            desc="Single-stream strategies",
            dynamic_ncols=True,
            disable=not progress,
        )
        for strategy in strategy_iter:
            strategy_iter.set_postfix_str(strategy, refresh=False)
            with torch.no_grad():
                single_latents, trace = model.sample_image_latents_single_stream(
                    input_ids=input_ids,
                    token_types=token_types,
                    sigma=sigma,
                    spans=spans,
                    image_latent_dim=image_latents.shape[-1],
                    diffusion_temperature=args.temperature,
                    diffusion_cfg=args.cfg,
                    diffusion_clip_denoised=args.single_stream_clip,
                    parallel_rate=args.parallel_rate,
                    order_strategy=strategy,
                    return_trace=True,
                )
            single_img = decode_latents(vae, single_latents.float(), scaling_factor)
            tag = strategy.replace("/", "_")
            save_image(make_grid(torch.stack([target_img, single_img], dim=1).flatten(0, 1), nrow=2), out_dir / f"strategy_{tag}.png")
            overview_columns.append((f"strategy_{tag}", single_img))
            metrics[f"single_stream_{tag}"] = {
                "latent_rms": float(single_latents.float().pow(2).mean().sqrt().item()),
                "latent_mse_to_target": float(
                    F.mse_loss(single_latents.float(), torch.stack(target_latents).float()).item()
                ),
                "generation_step_max": float(trace["generation_step"].float().max().item()) if trace else None,
            }

    if args.save_individual:
        save_image(target_img, out_dir / "target.png")
        save_image(denoise_img, out_dir / f"denoise_x0_t{args.denoise_timestep}.png")

    grid = make_grid(torch.stack([img for _, img in overview_columns], dim=1).flatten(0, 1), nrow=len(overview_columns))
    overview_path = out_dir / "overview.png"
    save_image(grid, overview_path)

    metrics["overview_columns"] = [name for name, _ in overview_columns]
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved overview: {overview_path}")
    print(f"Saved metrics: {metrics_path}")
    print(json.dumps({k: metrics[k] for k in ["loss", "diffusion_stats", "overview_columns"]}, indent=2))


if __name__ == "__main__":
    main()
