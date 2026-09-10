#!/usr/bin/env python3
"""Reload complete B + SigLIP raw/EMA exports and check NPU gradients and all generation tasks."""
import argparse
import gc
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import torch_npu
from omegaconf import OmegaConf
from safetensors import safe_open
from scripts.generate_unified_qualitative import build_t2i_item, noise_for, save_png
from pretrain.train_selfless_flow import _build_i2t_generation_prefix
from utils.evaluation_model_source import configure_model_source, resolve_evaluation_model_source
from utils.image_generation_io import load_vae, decode_latents
from utils.imagenet_flow_batching import collate_imagenet_flow_cache
from utils.selfless_flow_optimizer import optimizer_parameter_role
from utils.b_siglip_protocol import config_path
from utils.utils import load_model_tokenizer, get_selfless_mask


def check_gradients(model, batch):
    model.train()
    masks = dict(sigma=batch["sigma"], seq_len=batch["input_ids"].shape[1], device=batch["input_ids"].device,
                 input_ids=batch["input_ids"], token_types=batch["token_types"], boi_token_id=model.config.boi_token_id)
    result = model(attention_mask=get_selfless_mask(**masks),
        content_attention_mask=get_selfless_mask(**masks, include_diagonal=True), flow_sigma=batch["sigma"],
        X0_input_ids=batch["input_ids"], token_types=batch["token_types"],
        labels=batch["labels"], image_latents=batch["image_latents"],
        image_span_table=batch["image_span_table"], image_loss_mask=batch["image_loss_mask"],
        compute_text_loss=False, compute_image_loss=True)
    assert torch.isfinite(result.loss)
    result.loss.backward()
    norms = {}
    for name, p in model.named_parameters():
        if p.grad is None:
            raise AssertionError(f"Missing gradient: {name}")
        norm = p.grad.float().square().sum()
        if not torch.isfinite(norm):
            raise FloatingPointError(name)
        role = optimizer_parameter_role(name)
        norms[role] = norms.get(role, 0.) + float(norm)
    for role in ("backbone", "image_projector", "flow_head"):
        assert norms[role] > 0, (role, norms)
    assert norms["semantic_pretrained"] > 0
    model.zero_grad(set_to_none=True)
    model.eval()
    return {"loss": float(result.loss), "gradient_l2_by_role": {k: v ** .5 for k, v in norms.items()}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", choices=("b-siglip",), default="b-siglip")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    assert torch.npu.is_available() and torch.npu.device_count() == 16
    torch.npu.set_device(0); device = torch.device("npu", 0)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    for kind in ("current", "ema"):
        checkpoint = args.run_dir / ("hf_model-final-ema" if kind == "ema" else "hf_model-final")
        config = OmegaConf.load(config_path(args.variant))
        if kind == "ema":
            configure_model_source(config, resolve_evaluation_model_source(checkpoint))
        else:
            config.model.model_path = str(checkpoint.resolve())
        model, tokenizer = load_model_tokenizer(config, model_dtype=torch.bfloat16)
        model.to(device).eval()
        state = model.state_dict(); verified = 0
        with safe_open(str(checkpoint / "model.safetensors"), framework="pt", device="cpu") as saved:
            for key in saved.keys():
                actual = state[key].flatten()[:16].cpu()
                expected = saved.get_tensor(key).flatten()[:16].to(actual.dtype)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0, msg=key)
                verified += 1
        del state
        prompt = "A golden retriever sitting on green grass beside a red ball."
        item = build_t2i_item(tokenizer, model, prompt, 0, 42, "random")
        batch = collate_imagenet_flow_cache([item], pad_to_length=512)
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        cache = torch.load(config.dataset.params.image.validation.cache_path,
                           map_location="cpu", mmap=True, weights_only=True)
        stats = cache["posterior_stats"][0].float()
        observed = (stats[:, :16] + stats[:, 16:] * noise_for(1, 42)).to(device)
        start = item["image_start"]
        batch["image_latents"][:, start:start+256] = observed
        gradients = check_gradients(model, batch) if kind == "current" else None
        parity = None
        if kind == "current":
            kwargs = dict(input_ids=batch["input_ids"], token_types=batch["token_types"],
                sigma=batch["sigma"], spans=[(0, start, start+256)],
                initial_noise_bank=noise_for(0, 42).unsqueeze(0), flow_cfg=2.,
                flow_num_steps=10, flow_solver="heun", return_trace=True, debug_finite=True,
                _debug_max_generation_steps=8)
            cached, cached_trace = model.generate("t2i", **kwargs, use_cache=True)
            full, full_trace = model.generate("t2i", **kwargs, use_cache=False)
            error = (cached.float()-full.float()).abs()
            parity = {"generated_positions": 8, "atol": .002, "rtol": .002,
                      "max_abs_error": float(error.max()), "mean_abs_error": float(error.mean())}
            torch.testing.assert_close(cached, full, atol=.002, rtol=.002)
            del cached, full, cached_trace, full_trace
        torch.npu.reset_peak_memory_stats(device)
        began = time.monotonic()
        latent, trace = model.generate("t2i", input_ids=batch["input_ids"],
            token_types=batch["token_types"], sigma=batch["sigma"], spans=[(0, start, start+256)],
            initial_noise_bank=noise_for(0, 42).unsqueeze(0), flow_cfg=2.,
            flow_num_steps=10, flow_solver="heun", return_trace=True, debug_finite=True)
        torch.npu.synchronize()
        seconds = time.monotonic() - began
        assert latent.shape == (1, 16, 16, 16) and torch.isfinite(latent).all()
        assert trace["backbone_kv_cache_enabled"] is True
        assert trace["semantic_forward_calls"] > 0 and trace["semantic_cache_peak_bytes"] > 0
        config.experiment.validation_vae_module_root = "public/code/mar"
        config.experiment.validation_vae_path = "public/vae/mar-kl16/kl16.ckpt"
        vae = load_vae(config, device, "fp32")
        with torch.no_grad():
            decoded = decode_latents(vae, latent.float(), .2325)
        assert torch.isfinite(decoded).all()
        save_png(decoded[0], args.output_dir / f"{kind}-t2i.png")
        torch.save(latent.cpu(), args.output_dir / f"{kind}-t2i-latents.pt")
        ids = torch.tensor([tokenizer.encode("The purpose of this experiment is", add_special_tokens=False)], device=device)
        text, text_trace = model.generate("text", input_ids=ids, max_new_tokens=16, return_trace=True)
        image_ids, image_types, sigma, image_start = _build_i2t_generation_prefix(tokenizer,
            text_prefix=config.dataset.params.image.caption_i2t_prefix,
            boi_token_id=model.config.boi_token_id, eoi_token_id=model.config.eoi_token_id,
            image_mask_token_id=model.config.image_mask_token_id, image_tokens=256)
        image_latents = torch.zeros(1, image_ids.numel(), 16, device=device)
        image_latents[:, image_start:image_start+256] = observed
        caption, caption_trace = model.generate("i2t", input_ids=image_ids[None].to(device),
            token_types=image_types[None].to(device), sigma=sigma[None].to(device),
            image_latents=image_latents, max_new_tokens=16, return_trace=True)
        report = {"weights": kind, "checkpoint": str(checkpoint), "tensor_samples_verified": verified,
            "flow_gradients": gradients, "cache_full_parity": parity, "t2i": trace, "t2i_seconds": seconds,
            "peak_memory_allocated_bytes": torch.npu.max_memory_allocated(device),
            "text": {"trace": text_trace, "output": tokenizer.decode(text[0, ids.shape[1]:].tolist())},
            "i2t": {"trace": caption_trace, "output": tokenizer.decode(caption[0, image_ids.numel():].tolist())}}
        reports.append(report)
        (args.output_dir / f"{kind}-report.json").write_text(json.dumps(report, ensure_ascii=False,
            indent=2, default=lambda x: x.tolist() if torch.is_tensor(x) else str(x)))
        del model, vae, batch, cache, latent, decoded
        gc.collect(); torch.npu.empty_cache()
    result = {"schema": "b_siglip_generation_smoke_v1", "passed": True, "variant": args.variant, "reports": reports}
    (args.output_dir / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2,
        default=lambda x: x.tolist() if torch.is_tensor(x) else str(x)))
    print(json.dumps({"passed": True, "report": str(args.output_dir / "report.json")}), flush=True)


if __name__ == "__main__":
    main()
