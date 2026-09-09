#!/usr/bin/env python3
"""Bounded 16-NPU matrix smoke: native weights, paired full decodes and production batch capacity."""
from __future__ import annotations

import argparse
import gc
import glob
import os
from pathlib import Path
import sys
import time
import types

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.generate_unified_qualitative import (
    torch, OmegaConf, configure_model_source, resolve_evaluation_model_source,
    load_model_tokenizer, load_model_source_weights, check_loaded_values,
    build_t2i_item, collate_imagenet_flow_cache, load_vae, decode_latents, save_png,
)
from scripts.sweep_unified_t2i_sampling import read, write, require, now
from scripts.evaluate_single_stream_fid_is import build_canonical_initial_noise_bank
from utils.image_order_strategies import checkpoint_generation_contract


@torch.inference_mode()
def run(root):
    import torch_npu  # noqa: F401
    rank, world = int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"])
    require(world == torch.npu.device_count() == 16, "expected 16 real Ascend devices")
    torch.npu.set_device(rank)
    device = torch.device("npu", rank)
    protocol = read(root / "protocol.json")
    mid = list(protocol["models"])[rank % len(protocol["models"])]
    spec = protocol["models"][mid]
    out = root / "smoke" / f"rank-{rank:02d}"
    out.mkdir(parents=True, exist_ok=True)

    def progress(stage):
        write(out / "progress.json", {"rank": rank, "model": mid, "stage": stage, "at": now(), "pid": os.getpid()})

    progress("loading")
    cfg = OmegaConf.load(spec["config"])
    cfg.training.runtime_hashing_enabled = False
    source = resolve_evaluation_model_source(spec["model_source"])
    configure_model_source(cfg, source)
    torch.manual_seed(42)
    model, tokenizer = load_model_tokenizer(cfg, model_dtype=torch.bfloat16)
    loaded = load_model_source_weights(model, source)
    loaded["full_checkpoint_value_check"] = check_loaded_values(model, spec["model_source"])
    write(out / "load.json", loaded)
    contract = checkpoint_generation_contract(model.config)
    require(contract == spec["generation_contract"], f"loaded contract mismatch: {contract} != {spec['generation_contract']}")
    model.to(device).eval()
    records = [read(Path(protocol["baseline_reproduction"]) / "arms/halton/spatial_halton" / f"{i:08d}.json")
               for i in [0, 49999]]
    items = [build_t2i_item(tokenizer, model, r["prompt"], i, 42, spec["generation_contract"]["training_image_sigma_order"])
             for i, r in zip([0, 49999], records)]
    batch = collate_imagenet_flow_cache(items, pad_to_length=512)
    noise, _ = build_canonical_initial_noise_bank([0, 49999], evaluation_seed=42)
    payload = {"input_ids": batch["input_ids"].to(device), "token_types": batch["token_types"].to(device),
        "sigma": batch["sigma"].to(device),
        "spans": [(b, item["image_start"], item["image_start"] + 256) for b, item in enumerate(items)],
        "initial_noise_bank": noise, "flow_cfg": 2., "flow_cfg_schedule": "constant", "flow_solver": "heun",
        "flow_num_steps": 10, "flow_temperature": 1., "use_cache": True, "return_trace": True}
    generations, traces, times = {}, {}, {}
    strategies = ["spatial_halton", "confidence_stability", "confidence_halton"]
    if mid == "e_on_b":
        strategies.append("sequential")
    for strategy in strategies:
        progress(strategy)
        cpu_rng, npu_rng = torch.get_rng_state().clone(), torch.npu.get_rng_state().clone()
        start = time.monotonic()
        latents, trace = model.generate_image(**payload, order_strategy=strategy)
        torch.npu.synchronize()
        times[strategy] = time.monotonic() - start
        require(torch.equal(cpu_rng, torch.get_rng_state()) and torch.equal(npu_rng, torch.npu.get_rng_state()), "probe consumed RNG")
        require(bool(torch.isfinite(latents).all()), "nonfinite smoke generation")
        ranks = trace["generation_order"].flatten(1).cpu()
        require(torch.equal(ranks.sort(1).values, torch.arange(1, 257).repeat(2, 1)), "invalid permutation")
        if strategy.startswith("confidence_"):
            proxy = trace["order_confidence_proxy"].flatten(1).cpu()
            require(bool(torch.isfinite(proxy).all()), "nonfinite confidence score")
            base = traces["spatial_halton"]["generation_order"].flatten(1).cpu()
            require(torch.equal((ranks - 1) // 16, (base - 1) // 16), "changed Halton candidate blocks")
            if strategy == "confidence_halton":
                require(torch.equal(ranks, base), "probe control changed order")
            else:
                ordered_scores = proxy.gather(1, ranks.argsort(1)).reshape(2, 16, 16)
                require(bool((ordered_scores[..., 1:] >= ordered_scores[..., :-1]).all()), "score direction wrong")
        if mid == "d_on_b":
            require(trace["dynamic_xt_flow_content_condition_commits"] == 255, "D content was not committed exactly once")
            require(trace["dynamic_xt_conditional_velocity_evaluations"] == 5120 + (32 if strategy.startswith("confidence_") else 0), "D probe did not refresh XT")
        if mid == "f_on_b":
            require(trace["flow_content_cache_peak_bytes_per_sample"] == 0, "F gained a content stream")
        generations[strategy], traces[strategy] = latents.cpu(), trace
        write(out / f"{strategy}-trace.json", {k: v.detach().cpu().tolist() if isinstance(v, torch.Tensor) else v
            for k, v in trace.items()})
    progress("forced_order_reference")
    recorded = traces["confidence_stability"]["generation_order"].flatten(1).argsort(1).to(device)
    original = model._image_generation_orders
    model._image_generation_orders = types.MethodType(lambda self, **kw: ("spatial_halton", recorded.clone(), False), model)
    reference, _ = model.generate_image(**payload, order_strategy="spatial_halton")
    # Keep candidate-query GEMM shapes identical while fixing the observed
    # order. This isolates cache semantics from BF16 shape-dependent rounding.
    model._image_generation_orders = types.MethodType(lambda self, **kw: ("confidence_halton", recorded.clone(), False), model)
    matched, _ = model.generate_image(**payload, order_strategy="confidence_halton")
    model._image_generation_orders = original

    def relative_rmse(a, b):
        a, b = a.float().cpu(), b.float().cpu()
        return float((a - b).square().mean().sqrt() / b.square().mean().sqrt().clamp_min(1e-8))

    parity = relative_rmse(generations["confidence_stability"], reference)
    matched_parity = relative_rmse(generations["confidence_stability"], matched)
    control = relative_rmse(generations["confidence_halton"], generations["spatial_halton"])
    write(out / "parity.json", {"model": mid, "forced_order_relative_rmse": parity,
        "matched_probe_shape_relative_rmse": matched_parity, "halton_control_relative_rmse": control})
    require(matched_parity < .005, f"BF16 cache drift with matched probe shapes: {mid}: {matched_parity}")
    del reference, matched, trace, latents
    # Exercise the production per-rank batch, 512-token cache and a second
    # confidence block. The short decode is only a capacity gate, never FID.
    progress("production_batch_capacity")
    large = {**payload, **{k: payload[k].repeat(128, 1) for k in ["input_ids", "token_types", "sigma"]}}
    large["spans"] = [(i, items[i % 2]["image_start"], items[i % 2]["image_start"] + 256) for i in range(256)]
    large["initial_noise_bank"], _ = build_canonical_initial_noise_bank(list(range(256)), evaluation_seed=42)
    large_latents, large_trace = model.generate_image(**large, order_strategy="confidence_stability", _debug_max_generation_steps=17)
    require(bool(torch.isfinite(large_latents).all()) and int(large_trace["generation_order"].max()) == 17,
            "production batch smoke incomplete")
    peak_mib = torch.npu.max_memory_allocated() / 2**20
    del large_latents, large_trace, large, model
    gc.collect()
    torch.npu.empty_cache()
    progress("saving_images")
    vae = load_vae(cfg, device, "fp32")
    for strategy, latents in generations.items():
        pixels = decode_latents(vae, latents.float().to(device), float(cfg.experiment.validation_vae_scaling_factor))
        for i, pixel in enumerate(pixels):
            save_png(pixel, out / strategy / f"{i:02d}.png")
    write(out / "result.json", {"passed": True, "at": now(), "rank": rank, "model": mid,
        "generation_contract": contract, "checkpoint": source.report(), "strategies": strategies,
        "full_reveals": 256, "full_decode_batch": 2, "capacity_batch": 256, "capacity_reveals": 17,
        "padded_sequence_length": 512, "cfg": 2., "heun_steps": 10, "timings": times,
        "forced_order_relative_rmse": parity, "halton_control_relative_rmse": control,
        "matched_probe_shape_relative_rmse": matched_parity,
        "max_memory_allocated_mib": peak_mib, "device_count": torch.npu.device_count(),
        "device_name": torch.npu.get_device_name(), "driver_libraries": glob.glob('/usr/local/Ascend/driver/lib64/driver/libascend_hal.so*'),
        "cpu_and_npu_rng_unchanged": True, "runtime_hashing_enabled": False})
    progress("complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    run(parser.parse_args().output_dir.resolve())
