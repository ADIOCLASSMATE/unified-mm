#!/usr/bin/env python3
"""Reload a depth-scaling smoke EMA and exercise complete cached generation."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch_npu  # noqa: F401
from omegaconf import OmegaConf

from scripts.generate_unified_qualitative import build_t2i_item, noise_for, save_png
from utils.evaluation_model_source import configure_model_source, load_model_source_weights, resolve_evaluation_model_source
from utils.flow_head_scaling import HEAD_PARAMETERS, config_path
from utils.image_generation_io import decode_latents, load_vae
from utils.imagenet_flow_batching import collate_imagenet_flow_cache
from utils.utils import load_model_tokenizer


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depth", type=int, choices=(16, 30), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--weights", choices=("ema", "current"), default="ema")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-generation-file", type=Path)
    args = parser.parse_args()
    assert torch.npu.is_available() and torch.npu.device_count() == 16
    torch.npu.set_device(0)
    device = torch.device("npu", 0)
    config = OmegaConf.load(config_path(args.depth))
    source = None
    if args.weights == "ema":
        source = resolve_evaluation_model_source(args.checkpoint)
        configure_model_source(config, source)
    else:
        config.model.model_path = str(args.checkpoint.resolve())
    model, tokenizer = load_model_tokenizer(config, model_dtype=torch.bfloat16)
    weights = load_model_source_weights(model, source) if source else {
        "kind": "current", "path": str(args.checkpoint), "loaded_via": "from_pretrained"}
    model.to(device).eval()
    # Check every saved tensor's values against the reloaded model, after its
    # intended BF16 conversion. Read small slices; never hash large weights.
    from safetensors import safe_open
    reloaded_state = model.state_dict()
    with safe_open(str(args.checkpoint / "model.safetensors"), framework="pt", device="cpu") as saved:
        checked_keys = list(saved.keys())
        for key in checked_keys:
            actual = reloaded_state[key].detach().flatten()[:16].cpu()
            expected = saved.get_tensor(key).flatten()[:16].to(dtype=actual.dtype)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0, msg=key)
    del reloaded_state
    assert len(model.image_flow_head.net.blocks) == args.depth
    assert model.image_flow_head.net.grad_checkpointing is True
    assert model.config.image_flow_share_content is True
    assert sum(p.numel() for p in model.image_flow_head.parameters()) == HEAD_PARAMETERS[args.depth]
    assert model.config.flow_condition_contract == "backbone_xt_query_backbone_x0_content"
    prompt = "A golden retriever sitting on green grass beside a red ball."
    item = build_t2i_item(tokenizer, model, prompt, 0, 42, "random")
    batch = collate_imagenet_flow_cache([item], pad_to_length=512)
    torch.npu.reset_peak_memory_stats(device)
    started = time.monotonic()
    generation_kwargs = dict(
        input_ids=batch["input_ids"].to(device),
        token_types=batch["token_types"].to(device), sigma=batch["sigma"].to(device),
        spans=[(0, item["image_start"], item["image_start"] + 256)],
        image_latent_dim=16, initial_noise_bank=noise_for(0, 42).unsqueeze(0),
        flow_temperature=1.0, flow_cfg=3.5, flow_cfg_schedule="constant",
        flow_solver="heun", flow_num_steps=10, parallel_rate=1,
        order_strategy="spatial_halton", use_cache=True, return_trace=True,
    )
    latents, trace = model.generate("t2i", **generation_kwargs)
    torch.npu.synchronize()
    elapsed = time.monotonic() - started
    assert tuple(latents.shape) == (1, 16, 16, 16)
    assert bool(torch.isfinite(latents).all().item())
    assert trace.get("backbone_kv_cache_enabled") is True
    reference_report = None
    if args.reference_generation_file is not None:
        spec = importlib.util.spec_from_file_location(
            "models.modeling_model._refactor_generation_reference", args.reference_generation_file,
        )
        reference = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reference)
        reference_latents, reference_trace = reference.SelflessGenerationMixin.generate_image(
            model, **generation_kwargs,
        )
        torch.testing.assert_close(latents, reference_latents, rtol=0, atol=0)
        torch.testing.assert_close(trace["generation_order"], reference_trace["generation_order"], rtol=0, atol=0)
        reference_report = {"file": str(args.reference_generation_file), "latents_bitwise_equal": True,
                            "generation_order_equal": True, "same_model_weights_and_noise": True}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(latents.cpu(), args.output_dir / "latents.pt")
    config.experiment.validation_vae_module_root = "public/code/mar"
    config.experiment.validation_vae_path = "public/vae/mar-kl16/kl16.ckpt"
    config.experiment.validation_vae_scaling_factor = 0.2325
    vae = load_vae(config, device, "fp32")
    decoded = decode_latents(vae, latents.float(), 0.2325)
    assert bool(torch.isfinite(decoded).all().item())
    save_png(decoded[0], args.output_dir / "generated.png")
    report = {"schema": "flow_head_scaling_generation_smoke_v1", "passed": True,
              "depth": args.depth, "head_parameters": HEAD_PARAMETERS[args.depth],
              "checkpoint": str(args.checkpoint), "weights": weights,
              "export_tensor_samples_verified": len(checked_keys),
              "flow_checkpointing": True, "sampling_steps": 10, "solver": "heun",
              "cfg": 3.5, "order": "spatial_halton", "latent_shape": list(latents.shape),
              "latent_rms": float(latents.float().square().mean().sqrt().item()),
              "backbone_kv_cache_enabled": True, "generation_seconds": elapsed,
              "generation_refactor_parity": reference_report,
              "peak_allocated_bytes": torch.npu.max_memory_allocated(device),
              "peak_reserved_bytes": torch.npu.max_memory_reserved(device)}
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
