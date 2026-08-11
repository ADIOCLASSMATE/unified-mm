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

from models.modeling_model import modeling_selfless_flow as selfless_flow  # noqa: E402
from models.modeling_model.modeling_selfless_flow import Qwen3ForCausalLM  # noqa: E402
from pretrain import train_selfless_flow as train_flow  # noqa: E402
from pretrain.train_selfless_flow import (  # noqa: E402
    _load_image_flow_adapter,
    _save_ema_image_flow_adapter,
)
from utils.sharded_ema import (  # noqa: E402
    RankShardedEMA,
    build_sharded_ema_layout,
    mark_hf_ema_config_fp32,
    merge_sharded_ema_state_dict,
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
    def __init__(self, device, *, rank=0):
        self.process_index = int(rank)
        self.is_main_process = self.process_index == 0
        self.device = device

    @staticmethod
    def unwrap_model(model):
        return model

    @staticmethod
    def wait_for_everyone():
        return None


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
    config.image_flow_width = 32
    config.image_flow_depth = 1
    config.image_flow_num_sampling_steps = "2"
    config.image_flow_batch_mul = 1
    config.image_flow_time_scale = 1000.0
    config.image_flow_time_sampling = "uniform"
    config.image_flow_time_eps = 1e-4
    config.image_flow_time_uniform_mix = 0.0
    config.image_flow_solver = "heun"
    config.image_uncond_prob = 0.0
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
        "image_local_positions": torch.tensor(
            [[-1, -1, 0, 1, 2, 3, -1, -1]],
            device=device,
            dtype=torch.long,
        ),
        "image_span_table": torch.tensor(
            [[0, 0, 2, 6, 0]],
            device=device,
            dtype=torch.long,
        ),
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
        image_local_positions=batch["image_local_positions"],
        image_span_table=batch["image_span_table"],
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
    output = model(
        X0_input_ids=batch["input_ids"],
        labels=batch["labels"],
        attention_mask=object(),
        token_types=batch["token_types"],
        image_latents=batch["image_latents"],
        image_local_positions=batch["image_local_positions"],
        image_span_table=batch["image_span_table"],
        calculate_likelihood=True,
    )
    loss = output.loss
    if not torch.isfinite(loss):
        raise RuntimeError(f"validation loss is not finite: {loss.item()}")

    z = model._prepare_image_flow_condition(output.last_hidden_state[0, 2:6])
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
        parallel_rate=1,
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


def _in_memory_merged_state(emas, model):
    layout = emas[0].layout
    source_state = model.state_dict(keep_vars=True)
    canonical = {}
    for name, metadata in layout["tensors"].items():
        source = source_state[name]
        dtype = torch.float32 if source.dtype.is_floating_point else source.dtype
        canonical[name] = torch.empty(metadata["shape"], dtype=dtype, device=source.device)
    seen = set()
    for ema in emas:
        for chunk_id, value in ema.shards.items():
            if chunk_id in seen:
                raise RuntimeError(f"EMA chunk {chunk_id} is owned by more than one rank")
            seen.add(chunk_id)
            chunk = layout["chunks"][chunk_id]
            canonical[chunk["tensor"]].view(-1).narrow(
                0, int(chunk["offset"]), int(chunk["numel"])
            ).copy_(value)
    if seen != set(layout["chunks"]):
        raise RuntimeError(
            f"EMA chunks are incomplete: missing={sorted(set(layout['chunks']) - seen)}"
        )
    return {
        name: canonical[layout["canonical_for_name"][name]]
        for name in layout["state_keys"]
    }


def _make_virtual_emas(model, *, world_size, decay, update_after_step, chunk_numel):
    layout = build_sharded_ema_layout(
        model,
        world_size=world_size,
        chunk_numel=chunk_numel,
    )
    emas = []
    for rank in range(world_size):
        ema = RankShardedEMA(
            layout,
            rank=rank,
            decay=decay,
            update_after_step=update_after_step,
        )
        ema.bind(model)
        ema.initialize_from_model(global_step=0)
        emas.append(ema)
    return layout, emas


@torch.no_grad()
def verify_ema_precision_and_delay(device, output_dir):
    model = Qwen3ForCausalLM(tiny_config()).to(device=device, dtype=torch.bfloat16)
    layout, emas = _make_virtual_emas(
        model,
        world_size=2,
        decay=0.5,
        update_after_step=5,
        chunk_numel=17,
    )

    owned = [chunk_id for ema in emas for chunk_id in ema.local_chunk_ids]
    if len(owned) != len(set(owned)) or set(owned) != set(layout["chunks"]):
        raise RuntimeError("EMA sharding has missing or duplicate chunks")
    floating_dtypes = {
        value.dtype
        for ema in emas
        for value in ema.shards.values()
        if value.dtype.is_floating_point
    }
    if floating_dtypes != {torch.float32}:
        raise RuntimeError(f"EMA floating shards must be fp32, got {floating_dtypes}")

    initial = _in_memory_merged_state(emas, model)
    with torch.no_grad():
        model.model.embed_tokens.weight.fill_(2.0)
    for ema in emas:
        ema.maybe_update(4)
    before_start = _in_memory_merged_state(emas, model)
    for name in initial:
        torch.testing.assert_close(before_start[name], initial[name], rtol=0, atol=0)

    for ema in emas:
        ema.maybe_update(5)
    after_sync = _in_memory_merged_state(emas, model)
    torch.testing.assert_close(
        after_sync["model.embed_tokens.weight"],
        torch.full_like(after_sync["model.embed_tokens.weight"], 2.0),
        rtol=0,
        atol=0,
    )
    with torch.no_grad():
        model.model.embed_tokens.weight.fill_(4.0)
    for ema in emas:
        ema.maybe_update(6)
    after_update = _in_memory_merged_state(emas, model)
    torch.testing.assert_close(
        after_update["model.embed_tokens.weight"],
        torch.full_like(after_update["model.embed_tokens.weight"], 3.0),
        rtol=0,
        atol=0,
    )

    class TiedSmokeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(4, 3)
            self.head = torch.nn.Linear(3, 4, bias=False)
            self.head.weight = self.embed.weight

    tied_model = TiedSmokeModel().to(device)
    tied_layout, tied_emas = _make_virtual_emas(
        tied_model,
        world_size=2,
        decay=0.9,
        update_after_step=0,
        chunk_numel=4,
    )
    if tied_layout["canonical_for_name"]["embed.weight"] != tied_layout["canonical_for_name"]["head.weight"]:
        raise RuntimeError("Tied weights were not assigned to one canonical EMA tensor")
    for ema in tied_emas:
        for value in ema.shards.values():
            value.zero_()
    tied_model.embed.weight.fill_(10.0)
    for ema in tied_emas:
        ema.maybe_update(1)
    tied_merged = _in_memory_merged_state(tied_emas, tied_model)
    torch.testing.assert_close(
        tied_merged["embed.weight"],
        torch.ones_like(tied_merged["embed.weight"]),
        rtol=0,
        atol=0,
    )
    if tied_merged["embed.weight"].data_ptr() != tied_merged["head.weight"].data_ptr():
        raise RuntimeError("Merged tied weights do not share canonical storage")

    checkpoint_dir = output_dir / "ema-regression"
    for ema in reversed(emas):
        ema.save_checkpoint(
            checkpoint_dir,
            SmokeAccelerator(device, rank=ema.rank),
            global_step=6,
        )
    disk_merged = merge_sharded_ema_state_dict(checkpoint_dir)
    if list(disk_merged) != list(model.state_dict()):
        raise RuntimeError("Merged HF EMA state_dict is incomplete or out of order")
    for name, expected in after_update.items():
        torch.testing.assert_close(disk_merged[name], expected.cpu(), rtol=0, atol=0)

    reloaded = []
    for rank in range(2):
        ema = RankShardedEMA(
            layout,
            rank=rank,
            decay=0.5,
            update_after_step=5,
        )
        ema.bind(model)
        ema.load_checkpoint(
            checkpoint_dir,
            SmokeAccelerator(device, rank=rank),
            expected_global_step=6,
        )
        reloaded.append(ema)
    for old, new in zip(emas, reloaded):
        if old.started != new.started or old.global_step != new.global_step:
            raise RuntimeError("EMA resume metadata did not round-trip")
        for chunk_id in old.local_chunk_ids:
            torch.testing.assert_close(old.shards[chunk_id], new.shards[chunk_id], rtol=0, atol=0)

    mismatch = RankShardedEMA(
        build_sharded_ema_layout(model, world_size=1, chunk_numel=17),
        rank=0,
        decay=0.5,
        update_after_step=5,
    )
    mismatch.bind(model)
    try:
        mismatch.load_checkpoint(
            checkpoint_dir,
            SmokeAccelerator(device),
            expected_global_step=6,
        )
    except RuntimeError as error:
        if "same world size" not in str(error):
            raise
    else:
        raise RuntimeError("EMA resume accepted a different world size")

    return {
        "floating_dtype": "torch.float32",
        "world_size": 2,
        "chunk_count": len(layout["chunks"]),
        "rank_bytes": layout["rank_bytes"],
        "delay_sync_value": 2.0,
        "post_start_ema_value": 3.0,
        "tied_single_update_value": 1.0,
        "checkpoint_dir": str(checkpoint_dir),
    }, model, emas, checkpoint_dir


def save_and_reload(model, emas, checkpoint_dir, config, output_dir, batch, device):
    merged = merge_sharded_ema_state_dict(checkpoint_dir)
    hf_dir = output_dir / "hf_model-smoke-ema"
    model.save_pretrained(hf_dir, state_dict=dict(merged), safe_serialization=True)
    mark_hf_ema_config_fp32(hf_dir)
    loaded = Qwen3ForCausalLM.from_pretrained(hf_dir).to(device)
    loaded.eval()
    loaded_state = loaded.state_dict()
    if list(loaded_state) != list(merged):
        raise RuntimeError("HF EMA reload state_dict is incomplete")
    for name, expected in merged.items():
        torch.testing.assert_close(
            loaded_state[name].cpu(), expected.cpu(), rtol=0, atol=0
        )
    torch.manual_seed(991)
    loaded_metrics = run_validation(loaded, batch)

    _save_ema_image_flow_adapter(
        checkpoint_dir,
        config,
        SmokeAccelerator(device),
        "smoke",
    )
    adapter_path = output_dir / "image_flow_adapter-smoke.pt"
    adapter_loaded = Qwen3ForCausalLM(tiny_config()).to(device)
    _load_image_flow_adapter(adapter_loaded, adapter_path, config)
    adapter_state = torch.load(adapter_path, map_location="cpu", weights_only=True)
    for name, value in adapter_loaded.image_flow_head.state_dict().items():
        torch.testing.assert_close(value.cpu(), adapter_state["image_flow_head"][name], rtol=0, atol=0)
    for name, value in adapter_loaded.image_flow_condition_proj.state_dict().items():
        torch.testing.assert_close(
            value.cpu(), adapter_state["image_flow_condition_proj"][name], rtol=0, atol=0
        )

    ema_loaded = Qwen3ForCausalLM(tiny_config()).to(device)
    ema_loaded.load_state_dict(merged, strict=True)
    torch.manual_seed(991)
    ema_metrics = run_validation(ema_loaded, batch)
    return {
        "hf_model_dir": str(hf_dir),
        "adapter_path": str(adapter_path),
        "ema_manifest": str(checkpoint_dir / "ema_manifest.json"),
        "loaded_validation": loaded_metrics,
        "ema_validation": ema_metrics,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Tiny unified model smoke: train, validate, save, and reload.")
    parser.add_argument("--output_dir", default="output/unified_smoke")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--device",
        default="npu",
        help="Use npu for the production smoke, or cpu for a local fallback.",
    )
    parser.add_argument("--no_clean", action="store_true", help="Do not remove an existing output_dir first.")
    return parser.parse_args()


def main():
    args = parse_args()
    train_flow.logger = SmokeLogger()
    patch_cpu_runtime()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "npu":
        import torch_npu  # noqa: F401

        if not torch.npu.is_available():
            raise RuntimeError(
                "NPU was requested for unified smoke, but torch.npu.is_available() is false."
            )
        torch.npu.set_device(0 if device.index is None else device.index)
    elif device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested for unified smoke, but torch.cuda.is_available() is false."
        )

    output_dir = Path(args.output_dir)
    if output_dir.exists() and not args.no_clean:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = smoke_train_config(output_dir)
    ema_regression, model, emas, checkpoint_dir = verify_ema_precision_and_delay(
        device,
        output_dir,
    )
    batch = synthetic_batch(device)
    train_loss = run_train_step(model, batch)
    for ema in emas:
        ema.maybe_update(7)
    for ema in reversed(emas):
        ema.save_checkpoint(
            checkpoint_dir,
            SmokeAccelerator(device, rank=ema.rank),
            global_step=7,
        )

    merged = merge_sharded_ema_state_dict(checkpoint_dir)
    ema_model = Qwen3ForCausalLM(tiny_config()).to(device)
    ema_model.load_state_dict(merged, strict=True)
    validation = run_validation(ema_model, batch)
    reload_metrics = save_and_reload(
        model,
        emas,
        checkpoint_dir,
        config,
        output_dir,
        batch,
        device,
    )

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
