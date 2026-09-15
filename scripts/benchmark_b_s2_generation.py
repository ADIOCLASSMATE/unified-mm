#!/usr/bin/env python3
"""Compare unchanged B/S2 weights with explicit inference implementations.

Every timed trial runs in a fresh process on one NPU. All final timings use
full Heun-10 warmup, three complete generations, and device synchronization.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from statistics import mean, median
import subprocess
import sys
import time
import traceback

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.benchmark_generation_capacity import read, write, bounds, refine_batches

MODELS = ("b_x0", "s2_single")


def utc():
    return datetime.now(timezone.utc).isoformat()


class Inference:
    def __init__(self, args):
        import torch
        import torch_npu
        from omegaconf import OmegaConf
        from scripts.generate_unified_qualitative import BASE_CONFIG, build_t2i_item, noise_for
        from utils.evaluation_model_source import (configure_model_source,
            resolve_evaluation_model_source, load_model_source_weights)
        from utils.utils import load_model_tokenizer
        from utils.image_generation_io import load_vae, decode_latents
        self.torch = torch
        torch.set_num_threads(1)
        torch.npu.set_device(args.device)
        self.device = torch.device("npu", args.device)
        self.manifest = read(args.root / "manifest.json")
        spec = next(s for s in self.manifest["models"] if s["id"] == args.model)
        cfg = OmegaConf.load(BASE_CONFIG)
        cfg.training.runtime_hashing_enabled = False
        source = resolve_evaluation_model_source(spec["checkpoint"])
        configure_model_source(cfg, source)
        cfg.model.image_flow_num_sampling_steps = "10"
        self.model, tokenizer = load_model_tokenizer(cfg, model_dtype=torch.bfloat16)
        self.source = load_model_source_weights(self.model, source)
        self.model.to(self.device).eval()
        cfg.experiment.validation_vae_module_root = "public/code/mar"
        cfg.experiment.validation_vae_path = "public/vae/mar-kl16/kl16.ckpt"
        cfg.experiment.validation_vae_scaling_factor = .2325
        self.vae = load_vae(cfg, self.device, "fp32")
        with torch.inference_mode():
            decode_latents(self.vae, torch.zeros(4,16,16,16,device=self.device), .2325)
        torch.npu.synchronize()
        torch.npu.empty_cache()
        self.resident_bytes = torch.npu.memory_allocated()
        self.cfg = self.manifest["contract"]["model_cfg"][args.model]
        self.use_cache = args.model == "b_x0"
        self.items, noises = [], []
        self.rows = self.manifest["samples"]["t2i"]
        for j in range(args.start_index, args.start_index + args.batch_size):
            idx, seed = j % len(self.rows), 42 + j // len(self.rows)
            self.items.append(build_t2i_item(tokenizer, self.model,
                self.rows[idx]["prompt"], idx, seed, spec["image_order"]))
            noises.append(noise_for(idx, seed))
        self.noise = torch.stack(noises).to(self.device)
        self.spans = [(j, item["image_start"], item["image_start"]+256)
                      for j,item in enumerate(self.items)]
        self.batches = {}

    def prepare(self, implementation):
        from utils.imagenet_flow_batching import collate_imagenet_flow_cache
        if implementation not in self.batches:
            batch = collate_imagenet_flow_cache(self.items,
                pad_to_length=512 if implementation in ("legacy","clean") else None,
                pad_to_multiple_of=None if implementation in ("legacy","clean") else 32)
            self.batches[implementation] = {k:batch[k].to(self.device)
                for k in ("input_ids", "token_types", "sigma")}
        return self.batches[implementation]

    def generate(self, implementation, *, steps=10, trace=False, diagnostics=False):
        batch = self.prepare(implementation)
        extra = {}
        if self.use_cache:
            extra = dict(cache_diagnostics=diagnostics,
                compact_backbone_cache=implementation == "compact", cache_attention_block_size=32)
        with self.torch.inference_mode():
            return self.model.generate("t2i", **batch, spans=self.spans,
                image_latent_dim=16, initial_noise_bank=self.noise,
                flow_temperature=1., flow_cfg=self.cfg, flow_cfg_schedule="constant",
                flow_solver="heun", flow_num_steps=steps, parallel_rate=1,
                order_strategy="spatial_halton", use_cache=self.use_cache,
                return_trace=trace, **extra)

    def decode(self, latents):
        from utils.image_generation_io import decode_latents
        with self.torch.inference_mode():
            return decode_latents(self.vae, latents.float(), .2325)


def worker(args):
    result = dict(model=args.model, implementation=args.implementation,
        batch_size=args.batch_size, device=args.device, phase=args.phase,
        start_index=args.start_index, started_at=utc(), status="running",
        hostname=os.uname().nodename, steps=args.steps, warmup_batches=args.warmup,
        measured_repeats=args.repeats)
    write(args.result, result)
    try:
        run = Inference(args)
        torch = run.torch
        from scripts.generate_unified_qualitative import validate_generation_trace, save_png
        props = torch.npu.get_device_properties(run.device)
        batch = run.prepare(args.implementation)
        result.update(cfg=run.cfg, weight_source=run.source,
            model_dtype=str(next(run.model.parameters()).dtype),
            torch_version=torch.__version__, device_properties=str(props),
            device_total_memory_bytes=int(props.total_memory),
            resident_allocated_bytes=run.resident_bytes,
            sequence_length=batch["input_ids"].shape[1], generation_seconds=[], memory_peaks=[])
        write(args.result, result)
        for i in range(args.warmup + args.repeats):
            # Legacy intentionally reproduces the old timed diagnostic path.
            is_warmup = i < args.warmup
            trace_enabled = is_warmup or args.implementation == "legacy"
            torch.npu.synchronize()
            torch.npu.reset_peak_memory_stats()
            started = time.perf_counter()
            output = run.generate(args.implementation, steps=args.steps,
                trace=trace_enabled, diagnostics=args.implementation == "legacy")
            torch.npu.synchronize()
            elapsed = time.perf_counter() - started
            peak = dict(allocated_bytes=torch.npu.max_memory_allocated(),
                        reserved_bytes=torch.npu.max_memory_reserved())
            if trace_enabled:
                latents, trace = output
                validate_generation_trace(trace, use_cache=run.use_cache, task="t2i")
                result["trace"] = {k:v for k,v in trace.items() if v is None or isinstance(v,(str,int,float,bool))}
            else:
                latents = output
            if tuple(latents.shape) != (args.batch_size,16,16,16) or not bool(torch.isfinite(latents).all()):
                raise ValueError("invalid generated latents")
            if not is_warmup:
                result["generation_seconds"].append(elapsed)
                result["memory_peaks"].append(peak)
            if i == args.warmup + args.repeats - 1:
                preview = latents[:2].clone()
            del output, latents
            result["completed_calls"] = i+1
            write(args.result, result)
        images = run.decode(preview)
        for j,img in enumerate(images):
            save_png(img, args.result.with_name(args.result.stem+f"-sample-{j}.png"))
        times = result["generation_seconds"]
        result.update(status="ok", completed_at=utc(),
            mean_seconds_per_image=mean(times)/args.batch_size,
            median_seconds_per_image=median(times)/args.batch_size,
            median_images_per_second=args.batch_size/median(times),
            peak_allocated_bytes=max(x["allocated_bytes"] for x in result["memory_peaks"]),
            peak_reserved_bytes=max(x["reserved_bytes"] for x in result["memory_peaks"]))
    except Exception as error:
        oom = "out of memory" in str(error).lower() or type(error).__name__ == "OutOfMemoryError"
        result.update(status="oom" if oom else "error", completed_at=utc(),
            error=str(error)[-4000:], traceback=traceback.format_exc()[-6000:])
    write(args.result, result)
    if result["status"] not in ("ok", "oom"):
        raise RuntimeError(result["error"])


def equivalence(args):
    run = Inference(args)
    torch = run.torch
    from scripts.generate_unified_qualitative import save_png
    variants = ("legacy", "trimmed", "compact") if run.use_cache else ("legacy", "trimmed")
    reference = run.generate("legacy")
    ref_images = run.decode(reference)
    metrics = {}
    for variant in variants:
        latents = reference if variant == "legacy" else run.generate(variant)
        images = ref_images if variant == "legacy" else run.decode(latents)
        diff = latents.float()-reference.float()
        mse = (images.float()-ref_images.float()).square().flatten(1).mean(1)
        metrics[variant] = dict(latent_max_abs=diff.abs().max().item(),
            latent_rmse=diff.square().mean().sqrt().item(),
            image_mae=(images-ref_images).abs().mean().item(),
            image_psnr_min=(-10*torch.log10(mse.clamp_min(1e-12))).min().item(),
            image_psnr_mean=(-10*torch.log10(mse.clamp_min(1e-12))).mean().item(),
            finite=bool(torch.isfinite(latents).all()))
        for j, img in enumerate(images):
            save_png(img, args.root/"equivalence"/args.model/variant/f"{args.start_index+j:04d}.png")
    write(args.result, dict(status="ok", model=args.model, device=args.device,
        batch_size=args.batch_size, start_index=args.start_index,
        metrics=metrics, completed_at=utc()))


def control(args):
    run = Inference(args)
    torch = run.torch
    reference, _ = run.generate("legacy", trace=True, diagnostics=True)
    repeat = run.generate("legacy")
    clean = run.generate("clean")
    metrics = {"repeat_max_abs":(reference-repeat).abs().max().item(),
               "clean_max_abs":(reference-clean).abs().max().item()}
    if metrics["repeat_max_abs"] or metrics["clean_max_abs"]:
        raise RuntimeError(f"unchanged-path reproducibility failed: {metrics}")
    write(args.result,dict(status="ok",model=args.model,device=args.device,
        start_index=args.start_index,batch_size=args.batch_size,metrics=metrics,completed_at=utc()))


def controls(args):
    plans=[(mid,"clean",8,m*8+i,i*8) for m,mid in enumerate(MODELS) for i in range(8)]
    rows=wave(args,plans,"control",action="control")
    write(args.root/"control.json",dict(complete=True,rows=rows))


def wave(args, plans, phase, *, steps=10, warmup=1, repeats=3, action="worker"):
    # Plans are (model, implementation, batch, device, start_index).
    def launch(plan):
        mid, implementation, batch, device, start = plan
        path = args.root/phase/f"{mid}-{implementation}-b{batch:04d}-d{device:02d}-s{start:04d}.json"
        if path.exists() and read(path).get("status") in ("ok", "oom"):
            return read(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, str(Path(__file__).resolve()), action, "--root",str(args.root),
            "--model",mid,"--implementation",implementation,"--batch-size",str(batch),
            "--device",str(device),"--start-index",str(start),"--phase",phase,
            "--steps",str(steps),"--warmup",str(warmup),"--repeats",str(repeats),"--result",str(path)]
        env = {k:v for k,v in os.environ.items() if k not in ("RANK","WORLD_SIZE","LOCAL_RANK","LOCAL_WORLD_SIZE")}
        with path.with_suffix(".log").open("w") as log:
            proc = subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=14400)
        if not path.exists() or read(path).get("status") not in ("ok","oom"):
            raise RuntimeError(f"worker failure ({proc.returncode}): {path}")
        return read(path)
    write(args.root/"progress.json", dict(phase=phase, plans=plans, updated_at=utc()))
    with ThreadPoolExecutor(max_workers=16) as pool:
        return list(pool.map(launch, plans))


def smoke(args):
    plans = [(mid,"compact" if mid=="b_x0" else "trimmed",8,m*8+i,i*8)
             for m,mid in enumerate(MODELS) for i in range(8)]
    eq = wave(args, plans, "equivalence", action="equivalence")
    write(args.root/"equivalence.json", dict(complete=True, rows=eq))
    plans = [("b_x0",impl,batch,m*4+i,0)
        for m,batch in enumerate((8,256))
        for i,impl in enumerate(("legacy","trimmed","compact"))]
    plans += [("s2_single",impl,batch,8+m*2+i,0)
        for m,batch in enumerate((8,256)) for i,impl in enumerate(("legacy","trimmed"))]
    rows = wave(args, plans, "smoke-speed", repeats=1)
    write(args.root/"smoke.json", dict(complete=True, rows=rows, completed_at=utc()))


def run_benchmark(args):
    smoke_rows = read(args.root/"smoke.json")["rows"]
    equivalence_rows = read(args.root/"equivalence.json")["rows"]
    if not read(args.root/"control.json")["complete"]:
        raise RuntimeError("unchanged-path controls are incomplete")
    eligible = {}
    for mid in MODELS:
        candidates = ("trimmed","compact") if mid == "b_x0" else ("trimmed",)
        eligible[mid] = [impl for impl in candidates if all(
            row["metrics"][impl]["finite"]
            for row in equivalence_rows if row["model"] == mid)]
        if not eligible[mid]:
            raise RuntimeError(f"No finite implementation for {mid}")
    selected = {mid:max(eligible[mid], key=lambda impl: next(
        r["median_images_per_second"] for r in smoke_rows if r["model"]==mid
        and r["implementation"]==impl and r["batch_size"]==256)) for mid in MODELS}
    write(args.root/"implementation_selection.json", dict(implementations=selected,
        eligible=eligible, rule="Mathematical contract tested in CPU FP32; NPU BF16 image differences disclosed separately; select by batch-256 warmed smoke throughput",
        bitwise_image_equivalence=False, unchanged_path_control="legacy diagnostics vs clean: exact latent equality"))
    clean_plans=[(mid,"clean",batch,m*8+b*4+i,0) for m,mid in enumerate(MODELS)
                 for b,batch in enumerate((8,256)) for i in range(4)]
    clean_rows=wave(args,clean_plans,"clean-baseline")
    if any(r["status"]!="ok" for r in clean_rows):
        raise RuntimeError("clean baseline failed")
    write(args.root/"clean_baseline.json",dict(complete=True,rows=clean_rows))
    matched = {}
    for mid in MODELS:
        # Both models run identical batches on the exact same devices.
        plans = [(mid,selected[mid],batch,m*4+i,0)
                 for m,batch in enumerate((1,8,64,256)) for i in range(4)]
        matched[mid] = wave(args, plans, f"matched-{mid}")
        if any(r["status"] != "ok" for r in matched[mid]):
            raise RuntimeError(f"matched-batch measurement failed for {mid}")
    write(args.root/"matched.json", dict(complete=True,models=matched))
    trials = {mid:[] for mid in MODELS}
    candidates = {"b_x0":[256,384,512,640,768,896,1024,1280],
                  "s2_single":[1024,2048,3072,4096,5120,6144,7168,8192]}
    plans = [(mid,selected[mid],batch,m*8+i,0)
             for m,mid in enumerate(MODELS) for i,batch in enumerate(candidates[mid])]
    for row in wave(args,plans,"capacity-coarse",steps=1,warmup=0,repeats=1):
        trials[row["model"]].append(row)
    for iteration in range(8):
        plans=[]
        for m,mid in enumerate(MODELS):
            low,high=bounds(trials[mid])
            batches=([low*k for k in (2,3,4,6)] if high is None else
                     [8,32,64,128] if low == 0 else refine_batches(low,high))
            plans.extend((mid,selected[mid],batch,m*8+i,0) for i,batch in enumerate(batches))
        if not plans:
            break
        for row in wave(args,plans,f"capacity-refine-{iteration}",steps=1,warmup=0,repeats=1):
            trials[row["model"]].append(row)
    else:
        raise RuntimeError("capacity search did not converge")
    search={mid:dict(candidate_batch=bounds(rows)[0], first_oom_batch=bounds(rows)[1])
            for mid,rows in trials.items()}
    write(args.root/"capacity_search.json",dict(complete=True,models=search,grain=8,
        screening="complete 256-token generation, Heun 1; validated below with full Heun 10"))
    measured={}
    # Disjoint devices permit both long maximum-batch runs concurrently.
    # Matched comparisons above control for the identity of each device.
    current={mid:search[mid]["candidate_batch"] for mid in MODELS}
    for attempt in range(12):
        pending=[mid for mid in MODELS if mid not in measured]
        if not pending:
            break
        plans=[(mid,selected[mid],current[mid],MODELS.index(mid)*8+i,0)
               for mid in pending for i in range(8)]
        rows=wave(args,plans,f"maximum-{attempt}")
        for mid in pending:
            group=[r for r in rows if r["model"]==mid]
            if all(r["status"]=="ok" for r in group):
                measured[mid]=group
            else:
                current[mid]-=8
    else:
        raise RuntimeError("full-Heun maximum-batch validation did not converge")
    write(args.root/"measurements.json",dict(complete=True,completed_at=utc(),
        models=measured,matched=matched,search=search,implementations=selected,
        protocol=read(args.root/"manifest.json")["generation_benchmark"]))
    write(args.root/"progress.json",dict(phase="complete",complete=True,updated_at=utc()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["worker","equivalence","smoke","control","controls","run"])
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--model",choices=MODELS,default="b_x0")
    p.add_argument("--implementation",choices=["legacy","clean","trimmed","compact"],default="compact")
    p.add_argument("--batch-size",type=int,default=8)
    p.add_argument("--device",type=int,default=0)
    p.add_argument("--start-index",type=int,default=0)
    p.add_argument("--steps",type=int,default=10)
    p.add_argument("--warmup",type=int,default=1)
    p.add_argument("--repeats",type=int,default=3)
    p.add_argument("--phase",default="measure")
    p.add_argument("--result",type=Path)
    args = p.parse_args()
    args.root=args.root.resolve()
    {"worker":worker,"equivalence":equivalence,"smoke":smoke,"control":control,
     "controls":controls,"run":run_benchmark}[args.action](args)


if __name__ == "__main__":
    main()
