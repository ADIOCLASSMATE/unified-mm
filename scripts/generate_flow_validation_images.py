#!/usr/bin/env python3
import argparse
import importlib.util
import json
import math
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
    parser = argparse.ArgumentParser(description="Manual validation image generation for image rectified-flow heads.")
    parser.add_argument("--config", default="configs/selfless/imagenet_flow_full_from_qwen3base.yaml")
    parser.add_argument("--model_path_override", default="")
    parser.add_argument(
        "--adapter",
        default="none",
        help="Flow adapter, current flow checkpoint safetensors, or 'none'.",
    )
    parser.add_argument(
        "--model_state",
        default="",
        help=(
            "Optional full model state to load after initialization. Supports a DeepSpeed "
            "mp_rank_00_model_states.pt file with a 'module' key or a plain state_dict."
        ),
    )
    parser.add_argument(
        "--ema_state",
        default="",
        help=(
            "Optional EMA state to load after model_state/adapter. Supports checkpoint "
            "ema_state.pt files with a 'state_dict' key or a plain state_dict."
        ),
    )
    parser.add_argument("--output_dir", default="output/manual_flow_validation")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--split", choices=["val", "train"], default="val")
    parser.add_argument("--sampling_steps", default="50")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--cfg", type=float, default=1.0)
    parser.add_argument("--cfg_schedule", choices=["constant", "linear"], default="constant")
    parser.add_argument("--flow_solver", choices=["heun", "euler"], default="heun")
    parser.add_argument("--probe_times", default="0.25,0.5,0.75,0.95")
    parser.add_argument("--single_stream", action="store_true")
    parser.add_argument(
        "--oracle_reveal_ratios",
        default="",
        help="Comma-separated fractions of image tokens seeded with ground-truth latents before single-stream generation.",
    )
    parser.add_argument(
        "--oracle_reveal_order",
        default="same",
        help="Order for oracle seeding: same, sigma, random, spatial_halton, spatial_uniform, or prefix.",
    )
    parser.add_argument("--parallel_rate", type=int, default=1)
    parser.add_argument("--strategies", default="sigma,hidden_norm")
    parser.add_argument(
        "--refine_ratios",
        default="",
        help=(
            "Comma-separated remask ratios for inference-only refinement, "
            "for example 0.5,0.25,0.125,0.0625."
        ),
    )
    parser.add_argument("--refine_seed_offset", type=int, default=9000)
    parser.add_argument("--vae_dtype", choices=["fp32", "fp16"], default="fp32")
    parser.add_argument("--save_individual", action="store_true")
    parser.add_argument("--no_progress", action="store_true")
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


def parse_float_list(value: str) -> list[float]:
    if not value:
        return []
    out = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        value_float = float(item)
        if value_float < 0.0 or value_float > 1.0:
            raise ValueError(f"values must be in [0, 1], got {value_float}")
        out.append(value_float)
    return out


def _halton(index: int, base: int) -> float:
    value = 0.0
    scale = 1.0 / float(base)
    while index > 0:
        value += (index % base) * scale
        index //= base
        scale /= float(base)
    return value


def _halton_order(side: int, device) -> torch.Tensor:
    seen = set()
    order = []
    idx = 1
    image_tokens = side * side
    while len(order) < image_tokens and idx < image_tokens * 32:
        row = min(side - 1, int(_halton(idx, 2) * side))
        col = min(side - 1, int(_halton(idx, 3) * side))
        flat = row * side + col
        if flat not in seen:
            seen.add(flat)
            order.append(flat)
        idx += 1
    if len(order) < image_tokens:
        order.extend([flat for flat in range(image_tokens) if flat not in seen])
    return torch.tensor(order, device=device, dtype=torch.long)


def _spatial_uniform_order(side: int, device) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.arange(side, device=device),
        torch.arange(side, device=device),
        indexing="ij",
    )
    center = (side - 1) / 2.0
    ring = torch.maximum((yy.float() - center).abs(), (xx.float() - center).abs())
    checker = (yy % 2) * 2 + (xx % 2)
    return torch.argsort((ring * 4.0 + checker.float()).flatten())


def _oracle_order(strategy: str, sigma_row: torch.Tensor, start: int, end: int, side: int, seed: int, sample_idx: int):
    strategy = str(strategy or "sigma").lower()
    image_tokens = end - start
    device = sigma_row.device
    if strategy in {"sigma", "sigma_replay", "causal_sigma"}:
        return torch.argsort(sigma_row[start:end].to(device=device, dtype=torch.float32))
    if strategy == "random":
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed) + 1009 * int(sample_idx))
        return torch.randperm(image_tokens, device=device, generator=generator)
    if strategy in {"spatial_halton", "halton"}:
        return _halton_order(side, device)
    if strategy in {"spatial_uniform", "uniform"}:
        return _spatial_uniform_order(side, device)
    if strategy in {"prefix", "raster", "row_major"}:
        return torch.arange(image_tokens, device=device, dtype=torch.long)
    return torch.argsort(sigma_row[start:end].to(device=device, dtype=torch.float32))


def build_oracle_initial_mask(
    token_types: torch.Tensor,
    sigma: torch.Tensor,
    spans: list[tuple[int, int, int]],
    image_tokens: int,
    ratio: float,
    generation_strategy: str,
    reveal_order: str,
    seed: int,
) -> torch.Tensor:
    mask = torch.zeros_like(token_types, dtype=torch.bool)
    reveal_count = max(0, min(image_tokens, int(math.floor(float(ratio) * image_tokens))))
    if reveal_count == 0:
        return mask
    side = int(image_tokens ** 0.5)
    order_strategy = generation_strategy if str(reveal_order).lower() == "same" else reveal_order
    for sample_idx, (batch_idx, start, end) in enumerate(spans):
        order = _oracle_order(order_strategy, sigma[batch_idx], start, end, side, seed, sample_idx)
        mask[batch_idx, start + order[:reveal_count]] = True
    return mask


def masked_mse_and_rms(pred: torch.Tensor, target: torch.Tensor, mask_hw: torch.Tensor) -> dict:
    pred = pred.float()
    target = target.float()
    mask = mask_hw.to(device=pred.device, dtype=torch.bool).unsqueeze(1)
    denom = mask.sum().item() * pred.shape[1]
    if denom <= 0:
        return {"latent_mse": None, "latent_rms": None}
    mask_f = mask.to(dtype=pred.dtype)
    return {
        "latent_mse": float((((pred - target) ** 2) * mask_f).sum().item() / denom),
        "latent_rms": float(((pred.pow(2) * mask_f).sum().item() / denom) ** 0.5),
    }


def sequence_mixer_context(
    target: torch.Tensor,
    span_sigma: torch.Tensor,
    local_positions: torch.Tensor,
) -> dict[str, torch.Tensor]:
    sigma_row = span_sigma.to(device=target.device, dtype=torch.float32).unsqueeze(0)
    positions = local_positions.to(device=target.device, dtype=torch.long).unsqueeze(0)
    return {
        "context_latents": target.unsqueeze(0),
        "context_mask": sigma_row.unsqueeze(1) < sigma_row.unsqueeze(2),
        "query_positions": positions,
        "context_positions": positions,
    }


def flat_query_mixer_context(
    target: torch.Tensor,
    span_sigma: torch.Tensor,
    local_positions: torch.Tensor,
) -> dict[str, torch.Tensor]:
    query_count = target.shape[0]
    sigma_values = span_sigma.to(device=target.device, dtype=torch.float32)
    positions = local_positions.to(device=target.device, dtype=torch.long)
    return {
        "context_latents": target.unsqueeze(0).expand(query_count, -1, -1).contiguous(),
        "context_mask": (sigma_values.unsqueeze(0) < sigma_values.unsqueeze(1)).unsqueeze(1),
        "query_positions": positions,
        "context_positions": positions.unsqueeze(0).expand(query_count, -1).contiguous(),
    }


def _migrate_head_state(model, head_state: dict[str, torch.Tensor]):
    target = model.image_flow_head.state_dict()
    load_state = {}
    skipped = {}
    for key, value in head_state.items():
        if key not in target:
            skipped[key] = [list(value.shape), None]
            continue
        target_shape = tuple(target[key].shape)
        value_shape = tuple(value.shape)
        if value_shape == target_shape:
            load_state[key] = value
        else:
            skipped[key] = [list(value_shape), list(target_shape)]
    missing, unexpected = model.image_flow_head.load_state_dict(load_state, strict=False)
    return {
        "loaded": len(load_state),
        "missing": list(missing),
        "unexpected": list(unexpected),
        "skipped": skipped,
    }


def _migrate_condition_proj_state(model, projector_state: dict[str, torch.Tensor]):
    projector = getattr(model, "image_flow_condition_proj", None)
    if projector is None:
        return {"loaded": 0, "missing": [], "unexpected": [], "skipped": {"_module": "missing"}}
    target = projector.state_dict()
    if not target:
        return {"loaded": 0, "missing": [], "unexpected": [], "skipped": {"_module": "empty"}}

    load_state = {}
    skipped = {}
    for key, value in projector_state.items():
        if key not in target:
            skipped[key] = [list(value.shape), None]
            continue
        target_shape = tuple(target[key].shape)
        value_shape = tuple(value.shape)
        if value_shape == target_shape:
            load_state[key] = value
        else:
            skipped[key] = [list(value_shape), list(target_shape)]
    missing, unexpected = projector.load_state_dict(load_state, strict=False)
    return {
        "loaded": len(load_state),
        "missing": list(missing),
        "unexpected": list(unexpected),
        "skipped": skipped,
    }


def load_adapter(model, adapter_path: str):
    if adapter_path.lower() in {"none", "null", "false", ""}:
        return {"adapter": None}

    path = Path(adapter_path)
    if path.is_dir():
        path = path / "model.safetensors"
    if not path.exists():
        raise FileNotFoundError(path)

    report = {"adapter": str(path)}
    if path.suffix == ".safetensors":
        head_state = {}
        condition_proj_state = {}
        projector_state = {}
        projector_target = model.image_token_embedder.state_dict()

        def maybe_add_projector_key(name, value):
            if name in projector_target and tuple(value.shape) == tuple(projector_target[name].shape):
                projector_state[name] = value

        with safe_open(str(path), framework="pt", device="cpu") as f:
            for key in f.keys():
                if key.startswith("image_flow_head."):
                    head_state[key[len("image_flow_head."):]] = f.get_tensor(key)
                elif key.startswith("image_flow_condition_proj."):
                    condition_proj_state[key[len("image_flow_condition_proj."):]] = f.get_tensor(key)
                elif key.startswith("model.image_flow_condition_proj."):
                    condition_proj_state[key[len("model.image_flow_condition_proj."):]] = f.get_tensor(key)
                elif key.startswith("model.image_token_embedder."):
                    maybe_add_projector_key(key[len("model.image_token_embedder."):], f.get_tensor(key))

        report["image_flow_head"] = _migrate_head_state(model, head_state)
        if condition_proj_state:
            report["image_flow_condition_proj"] = _migrate_condition_proj_state(model, condition_proj_state)
        if projector_state:
            missing, unexpected = model.image_token_embedder.load_state_dict(projector_state, strict=False)
            report["image_token_embedder_missing"] = list(missing)
            report["image_token_embedder_unexpected"] = list(unexpected)
        return report

    state = torch.load(path, map_location="cpu")
    if "image_flow_head" in state:
        report["image_flow_head"] = _migrate_head_state(model, state["image_flow_head"])
    if "image_flow_condition_proj" in state:
        report["image_flow_condition_proj"] = _migrate_condition_proj_state(model, state["image_flow_condition_proj"])
    if "image_token_embedder" in state:
        missing, unexpected = model.image_token_embedder.load_state_dict(state["image_token_embedder"], strict=False)
        report["image_token_embedder_missing"] = list(missing)
        report["image_token_embedder_unexpected"] = list(unexpected)
    if "special_token_embeddings" in state and "special_token_ids" in state:
        with torch.no_grad():
            embed = model.model.embed_tokens.weight
            for name, token_id in state["special_token_ids"].items():
                if name in state["special_token_embeddings"]:
                    embed[int(token_id)].copy_(state["special_token_embeddings"][name].to(dtype=embed.dtype))
        report["loaded_special_token_embeddings"] = sorted(state["special_token_embeddings"].keys())
    return report


def load_model_state(model, model_state_path: str):
    if not model_state_path:
        return {"model_state": None}
    path = Path(model_state_path)
    if path.is_dir():
        candidate = path / "pytorch_model" / "mp_rank_00_model_states.pt"
        if candidate.exists():
            path = candidate
    if not path.exists():
        raise FileNotFoundError(path)
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "module" in state:
        state_dict = state["module"]
    else:
        state_dict = state
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    return {
        "model_state": str(path),
        "keys": len(state_dict),
        "missing": list(missing),
        "unexpected": list(unexpected),
    }


def load_ema_state(model, ema_state_path: str):
    if not ema_state_path:
        return {"ema_state": None}
    path = Path(ema_state_path)
    if path.is_dir():
        candidate = path / "ema_state.pt"
        if candidate.exists():
            path = candidate
    if not path.exists():
        raise FileNotFoundError(path)
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state_dict = state["state_dict"]
        global_step = state.get("global_step")
        decay = state.get("decay")
    else:
        state_dict = state
        global_step = None
        decay = None
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    return {
        "ema_state": str(path),
        "global_step": global_step,
        "decay": decay,
        "keys": len(state_dict),
        "missing": list(missing),
        "unexpected": list(unexpected),
    }


def load_vae(config, device, dtype_name):
    vae_module_root = Path(
        config.experiment.get(
            "validation_vae_module_root",
            "/inspire/hdd/global_user/wanjiaxin-253108030048/code/mar",
        )
    )
    vae_path = Path(config.experiment.validation_vae_path)
    spec = importlib.util.spec_from_file_location("kl16_vae", vae_module_root / "models" / "vae.py")
    vae_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vae_module)
    vae = vae_module.AutoencoderKL(embed_dim=16, ch_mult=(1, 1, 2, 2, 4), ckpt_path=str(vae_path))
    dtype = torch.float16 if dtype_name == "fp16" and device.type == "cuda" else torch.float32
    vae = vae.to(device=device, dtype=dtype).eval()
    for param in vae.parameters():
        param.requires_grad_(False)
    return vae


def decode_latents(vae, latents, scaling_factor):
    vae_dtype = next(vae.parameters()).dtype
    decoded = vae.decode(latents.to(dtype=vae_dtype) / scaling_factor).float().clamp(-1, 1)
    return (decoded + 1.0) / 2.0


@torch.no_grad()
def refine_single_stream_latents(
    model,
    input_ids: torch.Tensor,
    token_types: torch.Tensor,
    sigma: torch.Tensor,
    spans: list[tuple[int, int, int]],
    draft_latents: torch.Tensor,
    refine_ratios: list[float],
    *,
    temperature: float,
    cfg: float,
    cfg_schedule: str,
    solver: str,
    seed: int,
) -> tuple[torch.Tensor, dict]:
    if not refine_ratios:
        return draft_latents, {"rounds": []}

    device = input_ids.device
    image_tokens = int(getattr(model.config, "image_tokens_per_img", 256))
    latent_dim = int(getattr(model.config, "image_latent_dim", draft_latents.shape[1]))
    side = int(image_tokens ** 0.5)
    if side * side != image_tokens:
        raise ValueError(f"image_tokens_per_img={image_tokens} is not square")

    selected_input_ids = torch.stack([input_ids[b] for b, _, _ in spans]).to(device=device)
    selected_token_types = torch.stack([token_types[b] for b, _, _ in spans]).to(device=device)
    selected_sigma = torch.stack([sigma[b] for b, _, _ in spans]).to(device=device, dtype=torch.float32)
    work_latents = torch.zeros(
        selected_input_ids.shape[0],
        selected_input_ids.shape[1],
        latent_dim,
        device=device,
        dtype=model.image_flow_head.net.final_layer.linear.weight.dtype,
    )
    current = draft_latents.to(device=device, dtype=work_latents.dtype)
    current_seq = current.permute(0, 2, 3, 1).reshape(len(spans), image_tokens, latent_dim)
    for sample_idx, (_, start, end) in enumerate(spans):
        work_latents[sample_idx, start:end] = current_seq[sample_idx]

    all_positions = torch.arange(image_tokens, device=device, dtype=torch.long)
    use_cfg = cfg != 1.0
    boi_token_id = getattr(model.config, "boi_token_id", None)
    if use_cfg and boi_token_id is None:
        raise ValueError("cfg != 1.0 requires model.config.boi_token_id")

    trace = {"rounds": []}
    for round_idx, ratio in enumerate(refine_ratios):
        ratio = float(ratio)
        if ratio <= 0.0:
            continue
        remask_count = max(1, min(image_tokens, int(round(ratio * image_tokens))))
        target_masks = []
        for sample_idx in range(len(spans)):
            generator = torch.Generator(device=device).manual_seed(seed + round_idx * 1_000_003 + sample_idx * 97)
            perm = torch.randperm(image_tokens, device=device, generator=generator)
            mask = torch.zeros(image_tokens, device=device, dtype=torch.bool)
            mask[perm[:remask_count]] = True
            target_masks.append(mask)
        target_masks = torch.stack(target_masks)
        keep_masks = ~target_masks

        image_latent_mask = torch.zeros_like(selected_token_types, dtype=torch.bool)
        current_sigma = selected_sigma.clone()
        for sample_idx, (_, start, end) in enumerate(spans):
            image_latent_mask[sample_idx, start:end] = keep_masks[sample_idx]
            non_padding = selected_token_types[sample_idx] != 3
            base = float(selected_sigma[sample_idx, non_padding].min().item())
            kept = all_positions[keep_masks[sample_idx]]
            targets = all_positions[target_masks[sample_idx]]
            if kept.numel() > 0:
                current_sigma[sample_idx, start + kept] = base + torch.arange(
                    kept.numel(),
                    device=device,
                    dtype=current_sigma.dtype,
                )
            current_sigma[sample_idx, start + targets] = base + image_tokens + 1.0

        attention_mask = get_selfless_mask(
            sigma=current_sigma,
            seq_len=selected_input_ids.shape[1],
            device=device,
        )
        hidden = model.model(
            X0_input_ids=selected_input_ids,
            attention_mask=attention_mask,
            token_types=selected_token_types,
            image_latents=work_latents,
            image_latent_mask=image_latent_mask,
            calculate_likelihood=False,
        ).last_hidden_state

        uncond_hidden = None
        if use_cfg:
            uncond_attention_mask = get_selfless_mask(
                sigma=current_sigma,
                seq_len=selected_input_ids.shape[1],
                device=device,
                input_ids=selected_input_ids,
                token_types=selected_token_types,
                boi_token_id=int(boi_token_id),
                image_uncond_rows=torch.ones(selected_input_ids.shape[0], device=device, dtype=torch.bool),
            )
            uncond_hidden = model.model(
                X0_input_ids=selected_input_ids,
                attention_mask=uncond_attention_mask,
                token_types=selected_token_types,
                image_latents=work_latents,
                image_latent_mask=image_latent_mask,
                calculate_likelihood=False,
            ).last_hidden_state

        sample_indices = []
        seq_positions = []
        local_positions = []
        for sample_idx, (_, start, _) in enumerate(spans):
            positions = all_positions[target_masks[sample_idx]]
            sample_indices.append(torch.full_like(positions, sample_idx))
            seq_positions.append(start + positions)
            local_positions.append(positions)
        sample_indices = torch.cat(sample_indices)
        seq_positions = torch.cat(seq_positions)
        local_positions = torch.cat(local_positions)

        z = model._prepare_image_flow_condition(hidden[sample_indices, seq_positions], local_positions)
        z_uncond = None
        if uncond_hidden is not None:
            z_uncond = model._prepare_image_flow_condition(
                uncond_hidden[sample_indices, seq_positions],
                local_positions,
            )
        torch.manual_seed(seed + 17_171 + round_idx)
        pred = model.sample_image_flow_with_cfg(
            z,
            z_uncond=z_uncond,
            temperature=temperature,
            cfg=cfg,
            cfg_schedule=cfg_schedule,
            solver=solver,
        ).to(dtype=work_latents.dtype)

        cursor = 0
        for sample_idx, (_, start, _) in enumerate(spans):
            positions = all_positions[target_masks[sample_idx]]
            count = positions.numel()
            work_latents[sample_idx, start + positions] = pred[cursor : cursor + count]
            cursor += count

        trace["rounds"].append(
            {
                "round": int(round_idx),
                "ratio": ratio,
                "remask_count": int(remask_count),
                "cfg": float(cfg),
                "cfg_schedule": str(cfg_schedule),
            }
        )

    refined = []
    for sample_idx, (_, start, end) in enumerate(spans):
        refined.append(work_latents[sample_idx, start:end].view(side, side, latent_dim).permute(2, 0, 1))
    return torch.stack(refined), trace


def main():
    args = parse_args()
    progress = not args.no_progress
    oracle_reveal_ratios = parse_float_list(args.oracle_reveal_ratios)
    refine_ratios = parse_float_list(args.refine_ratios)
    probe_times = parse_float_list(args.probe_times)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = OmegaConf.load(args.config)
    if args.model_path_override:
        config.model.model_path = args.model_path_override
    config.training.batch_size = args.batch_size
    config.training.dataloader_workers = 0
    config.model.image_flow_num_sampling_steps = str(args.sampling_steps)

    print("Loading model/tokenizer...")
    model, tokenizer = load_model_tokenizer(config)
    print(f"Loading adapter: {args.adapter}")
    adapter_report = load_adapter(
        model,
        args.adapter,
    )
    print(f"Loading model state: {args.model_state or 'none'}")
    model_state_report = load_model_state(model, args.model_state)
    print(f"Loading EMA state: {args.ema_state or 'none'}")
    ema_state_report = load_ema_state(model, args.ema_state)
    model = model.to(device).eval()
    print("Loading KL16 VAE...")
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
            image_uncond_rows = torch.ones(
                input_ids.shape[0],
                device=device,
                dtype=torch.bool,
            )
            uncond_attention_mask = get_selfless_mask(
                sigma=sigma,
                seq_len=input_ids.shape[1],
                device=device,
                input_ids=input_ids,
                token_types=token_types,
                boi_token_id=int(config.model.boi_token_id),
                image_uncond_rows=image_uncond_rows,
            )
            uncond_output = model(
                X0_input_ids=input_ids,
                attention_mask=uncond_attention_mask,
                token_types=token_types,
                image_latents=image_latents,
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
        "model_state": model_state_report,
        "ema_state": ema_state_report,
        "config": args.config,
        "model_path": str(config.model.model_path),
        "sampling_steps": str(args.sampling_steps),
        "temperature": args.temperature,
        "cfg": args.cfg,
        "cfg_schedule": args.cfg_schedule,
        "flow_solver": args.flow_solver,
        "probe_times": probe_times,
        "oracle_reveal_ratios": oracle_reveal_ratios,
        "oracle_reveal_order": args.oracle_reveal_order,
        "refine_ratios": refine_ratios,
        "loss": float(output.loss.detach().float().item()),
        "flow_stats": {
            key: float(value.detach().float().item())
            for key, value in getattr(output, "flow_debug_stats", {}).items()
        },
        "samples": [],
    }

    target_latents = []
    full_sample_latents = []
    probe_x0_latents = {time_value: [] for time_value in probe_times}

    sample_iter = tqdm(list(enumerate(spans)), desc="Flow sampling", dynamic_ncols=True, disable=not progress)
    with torch.no_grad():
        for sample_idx, (batch_idx, start, end) in sample_iter:
            local_positions = torch.arange(end - start, device=device, dtype=torch.long)
            z = model._prepare_image_flow_condition(output.last_hidden_state[batch_idx, start:end], local_positions)
            z_uncond = None
            if uncond_output is not None:
                z_uncond = model._prepare_image_flow_condition(
                    uncond_output.last_hidden_state[batch_idx, start:end],
                    local_positions,
                )
            target = image_latents[batch_idx, start:end].to(dtype=z.dtype)
            span_sigma = sigma[batch_idx, start:end].to(device=device, dtype=torch.float32)
            torch.manual_seed(args.seed + 2000 + sample_idx)
            full_sample = model.sample_image_flow_with_cfg(
                z,
                z_uncond=z_uncond,
                temperature=args.temperature,
                cfg=args.cfg,
                cfg_schedule=args.cfg_schedule,
                solver=args.flow_solver,
                **flat_query_mixer_context(target, span_sigma, local_positions),
            ).to(dtype=target.dtype)

            sample_metrics = {
                "index": sample_idx,
                "batch_index": batch_idx,
                "target": tensor_stats(target),
                "full_sample": tensor_stats(full_sample),
                "full_sample_mse_to_target": float(F.mse_loss(full_sample.float(), target.float()).item()),
                "probes": {},
            }
            for time_value in probe_times:
                t = torch.full((target.shape[0],), time_value, device=device, dtype=torch.float32)
                noise = torch.randn_like(target)
                t_view = t.view(-1, 1).to(dtype=target.dtype)
                x_t = (1.0 - t_view) * noise + t_view * target
                v_target = target - noise
                v_pred = model.image_flow_head.velocity(
                    x_t.unsqueeze(0),
                    t.unsqueeze(0),
                    z.unsqueeze(0),
                    **sequence_mixer_context(target, span_sigma, local_positions),
                ).squeeze(0).to(dtype=target.dtype)
                x0_est = x_t + (1.0 - t_view) * v_pred
                sample_metrics["probes"][str(time_value)] = {
                    "v_mse": float(F.mse_loss(v_pred.float(), v_target.float()).item()),
                    "x0_est_mse": float(F.mse_loss(x0_est.float(), target.float()).item()),
                    "x0_est": tensor_stats(x0_est),
                }
                probe_x0_latents[time_value].append(x0_est.view(side, side, -1).permute(2, 0, 1))

            target_latents.append(target.view(side, side, -1).permute(2, 0, 1))
            full_sample_latents.append(full_sample.view(side, side, -1).permute(2, 0, 1))
            metrics["samples"].append(sample_metrics)

    target_chw = torch.stack(target_latents).float()
    target_img = decode_latents(vae, target_chw, scaling_factor)
    full_sample_img = decode_latents(vae, torch.stack(full_sample_latents).float(), scaling_factor)
    overview_columns = [("target", target_img)]
    for time_value, latents in probe_x0_latents.items():
        if latents:
            probe_img = decode_latents(vae, torch.stack(latents).float(), scaling_factor)
            tag = str(time_value).replace(".", "p")
            overview_columns.append((f"flow_x0_est_{tag}", probe_img))
            save_image(probe_img, out_dir / f"flow_x0_est_{tag}.png")
    overview_columns.append(("full_sample", full_sample_img))
    save_image(full_sample_img, out_dir / "full_sample.png")

    if args.single_stream and strategies:
        strategy_iter = tqdm(strategies, desc="Single-stream strategies", dynamic_ncols=True, disable=not progress)
        for strategy in strategy_iter:
            strategy_iter.set_postfix_str(strategy, refresh=False)
            with torch.no_grad():
                single_latents, trace = model.sample_image_latents_single_stream(
                    input_ids=input_ids,
                    token_types=token_types,
                    sigma=sigma,
                    spans=spans,
                    image_latent_dim=image_latents.shape[-1],
                    flow_temperature=args.temperature,
                    flow_cfg=args.cfg,
                    flow_cfg_schedule=args.cfg_schedule,
                    flow_solver=args.flow_solver,
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
                "latent_mse_to_target": float(F.mse_loss(single_latents.float(), target_chw).item()),
                "generation_step_max": float(trace["generation_step"].float().max().item()) if trace else None,
            }
            if refine_ratios:
                with torch.no_grad():
                    refined_latents, refine_trace = refine_single_stream_latents(
                        model=model,
                        input_ids=input_ids,
                        token_types=token_types,
                        sigma=sigma,
                        spans=spans,
                        draft_latents=single_latents,
                        refine_ratios=refine_ratios,
                        temperature=args.temperature,
                        cfg=args.cfg,
                        cfg_schedule=args.cfg_schedule,
                        solver=args.flow_solver,
                        seed=args.seed + args.refine_seed_offset,
                    )
                refined_img = decode_latents(vae, refined_latents.float(), scaling_factor)
                save_image(
                    make_grid(torch.stack([target_img, single_img, refined_img], dim=1).flatten(0, 1), nrow=3),
                    out_dir / f"strategy_{tag}_refined.png",
                )
                overview_columns.append((f"strategy_{tag}_refined", refined_img))
                metrics[f"single_stream_{tag}_refined"] = {
                    "latent_rms": float(refined_latents.float().pow(2).mean().sqrt().item()),
                    "latent_mse_to_target": float(F.mse_loss(refined_latents.float(), target_chw).item()),
                    "latent_mse_to_draft": float(F.mse_loss(refined_latents.float(), single_latents.float()).item()),
                    "trace": refine_trace,
                }

            for ratio in oracle_reveal_ratios:
                oracle_mask = build_oracle_initial_mask(
                    token_types=token_types,
                    sigma=sigma,
                    spans=spans,
                    image_tokens=image_tokens,
                    ratio=ratio,
                    generation_strategy=strategy,
                    reveal_order=args.oracle_reveal_order,
                    seed=args.seed,
                )
                with torch.no_grad():
                    oracle_latents, oracle_trace = model.sample_image_latents_single_stream(
                        input_ids=input_ids,
                        token_types=token_types,
                        sigma=sigma,
                        spans=spans,
                        image_latent_dim=image_latents.shape[-1],
                        initial_image_latents=image_latents,
                        initial_image_latent_mask=oracle_mask,
                        flow_temperature=args.temperature,
                        flow_cfg=args.cfg,
                        flow_cfg_schedule=args.cfg_schedule,
                        flow_solver=args.flow_solver,
                        parallel_rate=args.parallel_rate,
                        order_strategy=strategy,
                        return_trace=True,
                    )
                oracle_grid = torch.stack(
                    [oracle_mask[batch_idx, start:end].view(side, side) for batch_idx, start, end in spans]
                )
                remaining_grid = ~oracle_grid
                oracle_img = decode_latents(vae, oracle_latents.float(), scaling_factor)
                ratio_tag = str(ratio).replace(".", "p")
                save_image(
                    make_grid(torch.stack([target_img, oracle_img], dim=1).flatten(0, 1), nrow=2),
                    out_dir / f"strategy_{tag}_oracle_{ratio_tag}.png",
                )
                overview_columns.append((f"strategy_{tag}_oracle_{ratio_tag}", oracle_img))
                metrics[f"single_stream_{tag}_oracle_{ratio_tag}"] = {
                    "oracle_reveal_ratio": float(ratio),
                    "oracle_revealed_tokens_per_image": int(oracle_grid[0].sum().item()) if oracle_grid.numel() else 0,
                    "latent_rms": float(oracle_latents.float().pow(2).mean().sqrt().item()),
                    "latent_mse_to_target": float(F.mse_loss(oracle_latents.float(), target_chw).item()),
                    "remaining": masked_mse_and_rms(oracle_latents, target_chw, remaining_grid),
                    "known": masked_mse_and_rms(oracle_latents, target_chw, oracle_grid),
                    "baseline_remaining": masked_mse_and_rms(single_latents, target_chw, remaining_grid),
                    "baseline_known": masked_mse_and_rms(single_latents, target_chw, oracle_grid),
                    "generation_step_max": (
                        float(oracle_trace["generation_step"].float().max().item()) if oracle_trace else None
                    ),
                }

    if args.save_individual:
        save_image(target_img, out_dir / "target.png")

    grid = make_grid(torch.stack([img for _, img in overview_columns], dim=1).flatten(0, 1), nrow=len(overview_columns))
    overview_path = out_dir / "overview.png"
    save_image(grid, overview_path)

    metrics["overview_columns"] = [name for name, _ in overview_columns]
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved overview: {overview_path}")
    print(f"Saved metrics: {metrics_path}")
    print(json.dumps({k: metrics[k] for k in ["loss", "flow_stats", "overview_columns"]}, indent=2))


if __name__ == "__main__":
    main()
