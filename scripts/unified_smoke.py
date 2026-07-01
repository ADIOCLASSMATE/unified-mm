#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from transformers import Qwen3Config

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.modeling_model import modeling_selfless_flow as selfless_flow
from models.modeling_model.modeling_selfless_flow import Qwen3ForCausalLM
from pretrain import train_selfless_flow as train_flow
from pretrain.train_selfless_flow import (
    _create_ema_model,
    _load_ema_state_if_available,
    _load_image_flow_adapter,
    _maybe_update_ema_model,
    _save_ema_state,
    _save_image_flow_adapter,
    _sync_ema_model,
    _update_ema_model,
)


class SmokeLogger:
    @staticmethod
    def info(message):
        print(f"[smoke] {message}")

    @staticmethod
    def warning(message):
        print(f"[smoke][warning] {message}")

    @staticmethod
    def error(message):
        print(f"[smoke][error] {message}")


class SmokeAccelerator:
    def __init__(self, device):
        self.is_main_process = True
        self.device = device

    @staticmethod
    def unwrap_model(model):
        return model


def patch_cpu_runtime():
    def rms_norm_forward(self, hidden_states):
        dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight.float() * hidden_states).to(dtype)

    selfless_flow.Qwen3RMSNorm.forward = rms_norm_forward

    def no_mask(*args, **kwargs):
        sigma = kwargs.get("sigma", args[0] if args else None)
        seq_len = kwargs.get("seq_len")
        if seq_len is None and len(args) > 1:
            seq_len = args[1]
        device = kwargs.get("device")
        if device is None and len(args) > 2:
            device = args[2]
        return SimpleNamespace(sigma=sigma, seq_len=seq_len, device=device)

    import utils.utils as utils_mod

    utils_mod.get_selfless_mask = no_mask


def tiny_config() -> Qwen3Config:
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=0,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=9,
    )
    config.mask_token_id = 7
    config.image_mask_token_id = 8
    config.boi_token_id = 11
    config.eoi_token_id = 12
    config.image_latent_dim = 4
    config.image_tokens_per_img = 4
    config.image_projector_width = 16
    config.image_flow_width = 8
    config.image_flow_depth = 1
    config.image_flow_num_sampling_steps = "2"
    config.image_flow_batch_mul = 1
    config.image_flow_condition_norm = "rms"
    config.image_flow_condition_norm_eps = 1e-6
    config.image_flow_time_scale = 1000.0
    config.image_flow_time_sampling = "uniform"
    config.image_flow_time_eps = 1e-4
    config.image_flow_time_uniform_mix = 0.0
    config.image_flow_solver = "heun"
    config.image_uncond_prob = 0.0
    config.lambda_text = 0.1
    config.lambda_image = 1.0
    config.use_flex_attention = False
    return config


def smoke_train_config(output_dir: Path) -> OmegaConf:
    return OmegaConf.create(
        {
            "experiment": {"output_dir": str(output_dir)},
            "training": {
                "use_ema": True,
                "ema_decay": 0.5,
                "ema_save_hf_model": False,
                "ema_save_adapter": True,
                "save_image_flow_adapter": True,
            },
            "model": {
                "mask_token_id": 7,
                "boi_token_id": 11,
                "eoi_token_id": 12,
                "image_mask_token_id": 8,
            },
        }
    )


def synthetic_batch(device):
    input_ids = torch.tensor([[3, 11, 8, 8, 8, 8, 12, 9]], device=device)
    token_types = torch.tensor([[0, 2, 1, 1, 1, 1, 2, 0]], device=device, dtype=torch.uint8)
    labels = input_ids.clone()
    labels[token_types == 1] = -100
    labels[:, 6] = -100
    sigma = torch.tensor([[0.0, 1.0, 4.0, 5.0, 6.0, 7.0, 2.0, 3.0]], device=device)
    image_latents = torch.zeros(input_ids.shape[0], input_ids.shape[1], 4, device=device)
    image_latents[:, 2:6] = torch.tensor(
        [
            [
                [0.10, -0.20, 0.30, -0.40],
                [-0.50, 0.60, -0.70, 0.80],
                [0.15, 0.25, -0.35, -0.45],
                [-0.55, -0.65, 0.75, 0.85],
            ]
        ],
        device=device,
    )
    return {
        "input_ids": input_ids,
        "token_types": token_types,
        "labels": labels,
        "sigma": sigma,
        "image_latents": image_latents,
    }


def run_train_step(model, batch):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    output = model(
        X0_input_ids=batch["input_ids"],
        labels=batch["labels"],
        attention_mask=object(),
        token_types=batch["token_types"],
        image_latents=batch["image_latents"],
    )
    loss = output.loss
    if not torch.isfinite(loss):
        raise RuntimeError(f"training loss is not finite: {loss.item()}")
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return float(loss.detach().item())


@torch.no_grad()
def run_validation(model, batch):
    model.eval()
    device = batch["input_ids"].device
    output = model(
        X0_input_ids=batch["input_ids"],
        labels=batch["labels"],
        attention_mask=object(),
        token_types=batch["token_types"],
        image_latents=batch["image_latents"],
        calculate_likelihood=True,
    )
    loss = output.loss
    if not torch.isfinite(loss):
        raise RuntimeError(f"validation loss is not finite: {loss.item()}")

    z = model._prepare_image_flow_condition(
        output.last_hidden_state[0, 2:6],
        torch.arange(4, device=device),
    )
    sampled = model.sample_image_flow_with_cfg(z, temperature=1.0, cfg=1.0)
    if tuple(sampled.shape) != (4, 4):
        raise RuntimeError(f"unexpected flow sample shape: {tuple(sampled.shape)}")

    single_stream, trace = model.sample_image_latents_single_stream(
        input_ids=batch["input_ids"],
        token_types=batch["token_types"],
        sigma=batch["sigma"],
        spans=[(0, 2, 6)],
        image_latent_dim=4,
        flow_temperature=1.0,
        flow_cfg=1.5,
        flow_cfg_schedule="linear",
        parallel_rate=2,
        order_strategy="sigma",
        return_trace=True,
    )
    if tuple(single_stream.shape) != (1, 4, 2, 2):
        raise RuntimeError(f"unexpected single-stream latent shape: {tuple(single_stream.shape)}")
    if trace.get("flow_cfg_schedule") != "linear":
        raise RuntimeError(f"missing cfg schedule trace: {trace}")

    target = batch["image_latents"][0, 2:6].view(2, 2, 4).permute(2, 0, 1)
    return {
        "loss": float(loss.detach().item()),
        "full_sample_rms": float(sampled.float().pow(2).mean().sqrt().item()),
        "single_stream_mse_to_target": float(F.mse_loss(single_stream.float(), target.unsqueeze(0).float()).item()),
        "single_stream_steps": int(trace["generation_step"].max().item()),
    }


@torch.no_grad()
def verify_ema_precision_and_delay(device):
    config = smoke_train_config(Path("unused"))
    config.training.ema_decay = 0.5

    model = Qwen3ForCausalLM(tiny_config()).to(device=device, dtype=torch.bfloat16)
    ema_model = _create_ema_model(model, config).to(device)
    accelerator = SmokeAccelerator(device)

    floating_dtypes = {
        value.dtype
        for value in ema_model.state_dict().values()
        if torch.is_floating_point(value)
    }
    if floating_dtypes != {torch.float32}:
        raise RuntimeError(f"EMA floating state must be fp32, got {sorted(str(dtype) for dtype in floating_dtypes)}")

    _sync_ema_model(ema_model, model, accelerator)
    started = False
    with torch.no_grad():
        ema_model.model.embed_tokens.weight.fill_(1.0)
        model.model.embed_tokens.weight.fill_(2.0)

    started = _maybe_update_ema_model(
        ema_model,
        model,
        accelerator,
        decay=0.5,
        next_step=4,
        update_after_step=5,
        ema_started=started,
    )
    if started:
        raise RuntimeError("EMA should not start before ema_update_after_step.")
    if not torch.allclose(ema_model.model.embed_tokens.weight, torch.ones_like(ema_model.model.embed_tokens.weight)):
        raise RuntimeError("EMA changed before ema_update_after_step.")

    started = _maybe_update_ema_model(
        ema_model,
        model,
        accelerator,
        decay=0.5,
        next_step=5,
        update_after_step=5,
        ema_started=started,
    )
    if not started:
        raise RuntimeError("EMA did not start at ema_update_after_step.")
    expected_sync = torch.full_like(ema_model.model.embed_tokens.weight, 2.0)
    if not torch.allclose(ema_model.model.embed_tokens.weight, expected_sync):
        raise RuntimeError("EMA start should sync current weights before averaging.")

    with torch.no_grad():
        model.model.embed_tokens.weight.fill_(4.0)
    _maybe_update_ema_model(
        ema_model,
        model,
        accelerator,
        decay=0.5,
        next_step=6,
        update_after_step=5,
        ema_started=started,
    )
    expected_ema = torch.full_like(ema_model.model.embed_tokens.weight, 3.0)
    if not torch.allclose(ema_model.model.embed_tokens.weight, expected_ema):
        raise RuntimeError("EMA did not average after the delayed sync step.")

    class TiedSmokeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(2, 2)
            self.head = torch.nn.Linear(2, 2, bias=False)
            self.head.weight = self.embed.weight

    tied_model = TiedSmokeModel().to(device)
    tied_ema = _create_ema_model(tied_model, config).to(device)
    if tied_ema.embed.weight.data_ptr() != tied_ema.head.weight.data_ptr():
        raise RuntimeError("Tied smoke model lost weight tying in EMA copy.")
    with torch.no_grad():
        tied_model.embed.weight.fill_(10.0)
        tied_ema.embed.weight.zero_()
    _update_ema_model(tied_ema, tied_model, accelerator, decay=0.9)
    tied_value = float(tied_ema.embed.weight.flatten()[0].item())
    if abs(tied_value - 1.0) > 1.0e-6:
        raise RuntimeError(f"Tied EMA weight was updated more than once: got {tied_value}")

    return {
        "floating_dtype": "torch.float32",
        "delay_sync_value": float(expected_sync.flatten()[0].item()),
        "post_start_ema_value": float(expected_ema.flatten()[0].item()),
        "tied_single_update_value": tied_value,
    }


def save_and_reload(model, ema_model, config, output_dir, batch, device):
    accelerator = SmokeAccelerator(device)
    hf_dir = output_dir / "hf_model-smoke"
    model.save_pretrained(hf_dir, safe_serialization=True)

    _save_image_flow_adapter(ema_model, config, accelerator, "smoke")
    _save_ema_state(ema_model, config, accelerator, 1)

    loaded = Qwen3ForCausalLM.from_pretrained(hf_dir).to(device)
    loaded.eval()
    loaded_metrics = run_validation(loaded, batch)

    adapter_loaded = Qwen3ForCausalLM(tiny_config()).to(device)
    _load_image_flow_adapter(adapter_loaded, output_dir / "image_flow_adapter-smoke.pt", config)
    adapter_metrics = run_validation(adapter_loaded, batch)

    ema_loaded = Qwen3ForCausalLM(tiny_config()).to(device)
    if not _load_ema_state_if_available(ema_loaded, output_dir / "checkpoint-1"):
        raise RuntimeError("EMA state was not saved or could not be loaded.")
    ema_loaded.to(device)
    ema_metrics = run_validation(ema_loaded, batch)

    return {
        "hf_model_dir": str(hf_dir),
        "adapter_path": str(output_dir / "image_flow_adapter-smoke.pt"),
        "ema_state_path": str(output_dir / "checkpoint-1" / "ema_state.pt"),
        "loaded_validation": loaded_metrics,
        "adapter_validation": adapter_metrics,
        "ema_validation": ema_metrics,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Tiny unified model smoke: train, validate, save, and reload.")
    parser.add_argument("--output_dir", default="output/unified_smoke")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cuda", help="Use cuda for GPU smoke, or cpu for local fallback.")
    parser.add_argument("--no_clean", action="store_true", help="Do not remove an existing output_dir first.")
    return parser.parse_args()


def main():
    args = parse_args()
    train_flow.logger = SmokeLogger()
    patch_cpu_runtime()
    torch.manual_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for unified smoke, but torch.cuda.is_available() is false.")
    device = torch.device(args.device)

    output_dir = Path(args.output_dir)
    if output_dir.exists() and not args.no_clean:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = smoke_train_config(output_dir)
    model = Qwen3ForCausalLM(tiny_config()).to(device)
    batch = synthetic_batch(device)

    ema_model = _create_ema_model(model, config)
    _sync_ema_model(ema_model, model, SmokeAccelerator(device))
    ema_regression = verify_ema_precision_and_delay(device)

    train_loss = run_train_step(model, batch)
    _update_ema_model(ema_model, model, SmokeAccelerator(device), decay=float(config.training.ema_decay))
    ema_model.eval()

    validation = run_validation(ema_model, batch)
    reload_metrics = save_and_reload(model, ema_model, config, output_dir, batch, device)

    report = {
        "device": str(device),
        "train_loss": train_loss,
        "ema_regression": ema_regression,
        "ema_validation": validation,
        "reload": reload_metrics,
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Unified smoke passed. Report: {report_path}")


if __name__ == "__main__":
    main()
