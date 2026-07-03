#!/usr/bin/env python3
import argparse
import importlib.util
import json
import math
import os
import sys
import types
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("DIFFUSERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torchvision.utils import make_grid, save_image
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.dataset_utils import get_dataloaders
from utils.utils import get_selfless_mask


def _load_helper_module():
    helper_path = ROOT / "scripts" / "generate_flow_validation_images.py"
    spec = importlib.util.spec_from_file_location("flow_validation_helpers", helper_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


HELPERS = _load_helper_module()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate validation images for archived legacy latent-mix flow checkpoints."
    )
    parser.add_argument(
        "--config",
        default="output/selfless-flow-stage0-imagenet-full-from-qwen3base-latentmix/config.yaml",
    )
    parser.add_argument(
        "--legacy_root",
        default="archive/legacy_selfless_flow_latentmix_4a354f7",
    )
    parser.add_argument("--model_path_override", default="")
    parser.add_argument("--adapter", default="none")
    parser.add_argument("--model_state", default="")
    parser.add_argument("--ema_state", default="")
    parser.add_argument("--output_dir", default="output/manual_legacy_latentmix_flow_validation")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--split", choices=["val", "train"], default="val")
    parser.add_argument("--sampling_steps", default="50")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--cfg", type=float, default=5.0)
    parser.add_argument("--cfg_schedule", choices=["constant", "linear"], default="constant")
    parser.add_argument("--flow_solver", choices=["heun", "euler"], default="heun")
    parser.add_argument("--probe_times", default="0.25,0.5,0.75,0.95")
    parser.add_argument("--single_stream", action="store_true")
    parser.add_argument("--parallel_rate", type=int, default=4)
    parser.add_argument("--strategies", default="causal_sigma,spatial_halton,spatial_uniform,random")
    parser.add_argument("--vae_dtype", choices=["fp32", "fp16"], default="fp32")
    parser.add_argument("--save_individual", action="store_true")
    parser.add_argument("--no_progress", action="store_true")
    return parser.parse_args()


def _ensure_package(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    sys.modules[name] = module
    return module


def _load_module(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_legacy_modules(legacy_root: Path):
    legacy_root = legacy_root.resolve()
    pkg = "legacy_latentmix_4a354f7"
    _ensure_package(pkg, legacy_root)
    _ensure_package(f"{pkg}.models", legacy_root / "models")
    _ensure_package(f"{pkg}.models.modeling_model", legacy_root / "models" / "modeling_model")
    module_root = legacy_root / "models" / "modeling_model"
    _load_module(
        f"{pkg}.models.modeling_model.image_position_utils",
        module_root / "image_position_utils.py",
    )
    mar_module = _load_module(
        f"{pkg}.models.modeling_model.mar_flow_latentmix",
        module_root / "mar_flow_latentmix.py",
    )
    model_module = _load_module(
        f"{pkg}.models.modeling_model.modeling_selfless_flow_latentmix",
        module_root / "modeling_selfless_flow_latentmix.py",
    )
    return model_module, mar_module


def _scheduled_cfg(cfg: float, schedule: str | None, progress: float) -> float:
    cfg = float(cfg)
    if cfg == 1.0:
        return 1.0
    schedule = str(schedule or "constant").lower()
    if schedule in {"constant", "none", "off", ""}:
        return cfg
    progress = max(0.0, min(1.0, float(progress)))
    if schedule == "linear":
        return 1.0 + (cfg - 1.0) * progress
    raise ValueError(f"Unknown cfg_schedule={schedule!r}; expected constant or linear.")


def install_corrected_cfg_runtime(model_module, mar_module):
    def identity_image_fill_cfg(cfg, schedule, progress):
        return float(cfg)

    def legacy_flow_sample_corrected(
        self,
        z,
        temperature=1.0,
        cfg=1.0,
        cfg_schedule="constant",
        solver=None,
        num_steps=None,
        return_trace=False,
        *,
        context_latents=None,
        context_mask=None,
        query_positions=None,
        context_positions=None,
    ):
        model_dtype = self.net.input_proj.weight.dtype
        model_device = self.net.input_proj.weight.device
        z = z.to(device=model_device, dtype=model_dtype)
        if cfg != 1.0 and z.shape[0] % 2 != 0:
            raise ValueError(f"cfg != 1.0 requires paired conditional/unconditional conditions; got batch {z.shape[0]}")

        steps = int(num_steps or self.num_sampling_steps)
        if steps <= 0:
            raise ValueError(f"num_steps must be positive, got {steps}")
        solver_name = str(solver or self.solver).lower()
        x_shape = (z.shape[0] // 2, *z.shape[1:-1]) if cfg != 1.0 else z.shape[:-1]
        x = torch.randn(*x_shape, self.in_channels, device=z.device, dtype=torch.float32) * float(temperature)

        raw_context_kwargs = self._context_to_device(
            {
                "context_latents": context_latents,
                "context_mask": context_mask,
                "query_positions": query_positions,
                "context_positions": context_positions,
            },
            z.device,
            z.dtype,
        )
        latent_mixer_cache = self.prepare_latent_mixer_cache(
            context_latents=raw_context_kwargs.get("context_latents"),
            context_mask=raw_context_kwargs.get("context_mask"),
            context_positions=raw_context_kwargs.get("context_positions"),
        )
        context_kwargs = {
            "query_positions": raw_context_kwargs.get("query_positions"),
            "latent_mixer_cache": latent_mixer_cache,
        }
        context_is_paired = False
        if cfg != 1.0:
            context_kwargs = self._duplicate_context(context_kwargs)
            context_is_paired = True

        times = torch.linspace(1.0, 0.0, steps + 1, device=z.device, dtype=torch.float32)
        for idx in range(steps):
            t = times[idx].expand(x_shape)
            t_next = times[idx + 1].expand(x_shape)
            dt = (times[idx + 1] - times[idx]).float()
            cfg_t = _scheduled_cfg(cfg, cfg_schedule, 1.0 - float(times[idx].item()))
            v = self._guided_velocity(
                x.to(dtype=model_dtype),
                t,
                z,
                cfg_t,
                context_kwargs,
                context_is_paired=context_is_paired,
            ).float()
            if solver_name == "euler":
                x = x + dt * v
            elif solver_name == "heun":
                x_euler = x + dt * v
                cfg_t_next = _scheduled_cfg(cfg, cfg_schedule, 1.0 - float(times[idx + 1].item()))
                v_next = self._guided_velocity(
                    x_euler.to(dtype=model_dtype),
                    t_next,
                    z,
                    cfg_t_next,
                    context_kwargs,
                    context_is_paired=context_is_paired,
                ).float()
                x = x + 0.5 * dt * (v + v_next)
            else:
                raise ValueError(f"Unknown image_flow_solver={solver_name!r}; expected heun or euler.")

        if return_trace:
            return x.to(dtype=model_dtype), {
                "solver": solver_name,
                "num_steps": steps,
                "cfg_schedule": str(cfg_schedule or "constant"),
                "time_convention": "legacy_target_to_noise",
            }
        return x.to(dtype=model_dtype)

    def sample_image_flow_with_corrected_cfg(
        self,
        z,
        z_uncond=None,
        temperature=1.0,
        cfg=1.0,
        cfg_schedule=None,
        solver=None,
        num_steps=None,
        context_latents=None,
        context_mask=None,
        query_positions=None,
        context_positions=None,
    ):
        if cfg_schedule is None:
            cfg_schedule = getattr(self, "_legacy_corrected_flow_cfg_schedule", "constant")
        if cfg == 1.0:
            return self.image_flow_head.sample(
                z,
                temperature=temperature,
                cfg=1.0,
                cfg_schedule=cfg_schedule,
                solver=solver,
                num_steps=num_steps,
                context_latents=context_latents,
                context_mask=context_mask,
                query_positions=query_positions,
                context_positions=context_positions,
            )
        if z_uncond is None:
            raise ValueError("cfg != 1.0 requires z_uncond; pass paired conditional/unconditional conditions.")
        if z.shape != z_uncond.shape:
            raise ValueError(f"z and z_uncond must have the same shape, got {tuple(z.shape)} vs {tuple(z_uncond.shape)}")
        paired = torch.cat([z, z_uncond], dim=0)
        return self.image_flow_head.sample(
            paired,
            temperature=temperature,
            cfg=cfg,
            cfg_schedule=cfg_schedule,
            solver=solver,
            num_steps=num_steps,
            context_latents=context_latents,
            context_mask=context_mask,
            query_positions=query_positions,
            context_positions=context_positions,
        )

    model_module._scheduled_flow_cfg = identity_image_fill_cfg
    mar_module.FlowLoss.sample = legacy_flow_sample_corrected
    model_module.Qwen3ForCausalLM.sample_image_flow_with_cfg = sample_image_flow_with_corrected_cfg


def load_legacy_model_tokenizer(config, model_class):
    def tokenizer_from(path):
        try:
            return AutoTokenizer.from_pretrained(path, fix_mistral_regex=True)
        except TypeError:
            return AutoTokenizer.from_pretrained(path)

    tokenizer = tokenizer_from(config.model.model_path)
    mask_token = "<|mdm_mask|>"
    if mask_token in tokenizer.get_vocab():
        mask_token_id = tokenizer.convert_tokens_to_ids(mask_token)
        added_mask_token = False
    else:
        tokenizer.add_special_tokens({"mask_token": mask_token})
        mask_token_id = tokenizer.convert_tokens_to_ids(mask_token)
        added_mask_token = True
    config.model.mask_token_id = mask_token_id

    boi_token = "<|boi|>"
    eoi_token = "<|eoi|>"
    image_mask_token = "<|img_mask|>"
    tokens_to_add = [
        token
        for token in (boi_token, eoi_token, image_mask_token)
        if token not in tokenizer.get_vocab()
    ]
    added_image_mask_token = image_mask_token in tokens_to_add
    if tokens_to_add:
        tokenizer.add_tokens(tokens_to_add, special_tokens=True)

    config.model.boi_token_id = tokenizer.convert_tokens_to_ids(boi_token)
    config.model.eoi_token_id = tokenizer.convert_tokens_to_ids(eoi_token)
    config.model.image_mask_token_id = tokenizer.convert_tokens_to_ids(image_mask_token)
    config.model.image_offset = config.model.get("image_offset", None) or 200000

    model_config = AutoConfig.from_pretrained(config.model.model_path, trust_remote_code=True)
    model_config.mask_token_id = config.model.mask_token_id
    model_config.use_flex_attention = config.model.use_flex_attention
    model_config.eos_token_id = tokenizer.eos_token_id
    for key, value in config.model.items():
        if key == "model_path":
            continue
        setattr(model_config, key, OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value)

    model = model_class.from_pretrained(
        pretrained_model_name_or_path=config.model.model_path,
        config=model_config,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    if len(tokenizer) > model.config.vocab_size:
        model.resize_token_embeddings(len(tokenizer))

    if added_mask_token or added_image_mask_token:
        with torch.no_grad():
            embed = model.model.embed_tokens.weight
            mask_token_id = int(model.config.mask_token_id)
            image_mask_token_id = int(model.config.image_mask_token_id)
            if (
                0 <= mask_token_id < embed.shape[0]
                and 0 <= image_mask_token_id < embed.shape[0]
                and mask_token_id != image_mask_token_id
            ):
                embed[image_mask_token_id].copy_(embed[mask_token_id])
    model.config.use_cache = False
    return model, tokenizer


def tensor_stats(x):
    return HELPERS.tensor_stats(x)


def main():
    args = parse_args()
    progress = not args.no_progress
    probe_times = HELPERS.parse_float_list(args.probe_times)
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

    print("Loading archived legacy modules...")
    model_module, mar_module = load_legacy_modules(Path(args.legacy_root))
    install_corrected_cfg_runtime(model_module, mar_module)

    print("Loading legacy model/tokenizer...")
    model, tokenizer = load_legacy_model_tokenizer(config, model_module.Qwen3ForCausalLM)
    print(f"Loading adapter: {args.adapter}")
    adapter_report = HELPERS.load_adapter(model, args.adapter)
    print(f"Loading model state: {args.model_state or 'none'}")
    model_state_report = HELPERS.load_model_state(model, args.model_state)
    print(f"Loading EMA state: {args.ema_state or 'none'}")
    ema_state_report = HELPERS.load_ema_state(model, args.ema_state)
    model._legacy_corrected_flow_cfg_schedule = args.cfg_schedule
    model = model.to(device).eval()

    print("Loading KL16 VAE...")
    vae = HELPERS.load_vae(config, device, args.vae_dtype)
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
            image_uncond_rows = torch.ones(input_ids.shape[0], device=device, dtype=torch.bool)
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
    if side * side != image_tokens:
        raise ValueError(f"image_tokens_per_img={image_tokens} is not square")
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
        "legacy_root": args.legacy_root,
        "model_path": str(config.model.model_path),
        "sampling_steps": str(args.sampling_steps),
        "temperature": args.temperature,
        "cfg": args.cfg,
        "cfg_schedule": args.cfg_schedule,
        "flow_solver": args.flow_solver,
        "time_convention": "legacy_target_to_noise",
        "probe_times": probe_times,
        "loss": float(output.loss.detach().float().item()),
        "flow_stats": {
            key: float(value.detach().float().item())
            for key, value in getattr(output, "flow_debug_stats", {}).items()
            if value.numel() == 1
        },
        "samples": [],
    }

    target_latents = []
    full_sample_latents = []
    probe_x0_latents = {time_value: [] for time_value in probe_times}

    sample_iter = tqdm(list(enumerate(spans)), desc="Legacy flow sampling", dynamic_ncols=True, disable=not progress)
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
                **HELPERS.flat_query_mixer_context(target, span_sigma, local_positions),
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
                x_t = (1.0 - t_view) * target + t_view * noise
                v_target = noise - target
                v_pred = model.image_flow_head.velocity(
                    x_t.unsqueeze(0),
                    t.unsqueeze(0),
                    z.unsqueeze(0),
                    **HELPERS.sequence_mixer_context(target, span_sigma, local_positions),
                ).squeeze(0).to(dtype=target.dtype)
                x0_est = x_t - t_view * v_pred
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
    target_img = HELPERS.decode_latents(vae, target_chw, scaling_factor)
    full_sample_img = HELPERS.decode_latents(vae, torch.stack(full_sample_latents).float(), scaling_factor)
    overview_columns = [("target", target_img)]
    for time_value, latents in probe_x0_latents.items():
        if latents:
            probe_img = HELPERS.decode_latents(vae, torch.stack(latents).float(), scaling_factor)
            tag = str(time_value).replace(".", "p")
            overview_columns.append((f"legacy_x0_est_{tag}", probe_img))
            save_image(probe_img, out_dir / f"legacy_x0_est_{tag}.png")
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
            single_img = HELPERS.decode_latents(vae, single_latents.float(), scaling_factor)
            tag = strategy.replace("/", "_")
            save_image(
                make_grid(torch.stack([target_img, single_img], dim=1).flatten(0, 1), nrow=2),
                out_dir / f"strategy_{tag}.png",
            )
            overview_columns.append((f"strategy_{tag}", single_img))
            metrics[f"single_stream_{tag}"] = {
                "latent_rms": float(single_latents.float().pow(2).mean().sqrt().item()),
                "latent_mse_to_target": float(F.mse_loss(single_latents.float(), target_chw).item()),
                "generation_step_max": float(trace["generation_step"].float().max().item()) if trace else None,
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
