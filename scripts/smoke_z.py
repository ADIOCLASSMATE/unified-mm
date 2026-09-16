#!/usr/bin/env python3
"""16-NPU Z training/resume, raw/EMA generation, and training validation checks."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.joint_experiments import joint_experiment_protocol


def run(command, path):
    print(json.dumps({"command": command, "log": str(path)}), flush=True)
    with path.open("w") as stream:
        subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)


def generation(checkpoint, weights, output, experiment="z"):
    protocol = joint_experiment_protocol(experiment)
    import torch
    import torch_npu  # noqa: F401
    from omegaconf import OmegaConf
    from safetensors import safe_open
    from utils.image_generation_io import build_t2i_item, noise_for, save_png
    from utils.evaluation_model_source import configure_model_source, load_model_source_weights, resolve_evaluation_model_source
    from utils.imagenet_flow_batching import collate_imagenet_flow_cache
    from utils.image_generation_io import load_vae, decode_latents
    from utils.utils import load_model_tokenizer
    if not torch.npu.is_available() or torch.npu.device_count() != 16:
        raise RuntimeError("Run this smoke on the fixed 16-NPU development Notebook")
    torch.npu.set_device(0)
    device = torch.device("npu", 0)
    config = OmegaConf.load(ROOT / protocol.CONFIG)
    source = resolve_evaluation_model_source(checkpoint) if weights == "ema" else None
    if source:
        configure_model_source(config, source)
    else:
        config.model.model_path = str(checkpoint)
    model, tokenizer = load_model_tokenizer(config, model_dtype=torch.bfloat16)
    load_report = load_model_source_weights(model, source) if source else {"kind": "current"}
    model.to(device).eval()
    expected_head = "b_single_stream" if experiment == "z-b" else "s2"
    if getattr(model.config, "joint_dit_head_type", "s2") != expected_head:
        raise RuntimeError("Checkpoint restored the wrong flow-head implementation")
    count = 0
    state = model.state_dict()
    with safe_open(str(checkpoint / "model.safetensors"), framework="pt", device="cpu") as saved:
        for name in saved.keys():
            actual = state[name].detach().reshape(-1)[:16].cpu()
            expected = saved.get_tensor(name).reshape(-1)[:16].to(actual.dtype)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0, msg=name)
            count += 1
    del state
    item = build_t2i_item(tokenizer, model, "A golden retriever sitting beside a red ball on green grass.", 0, 42, "joint")
    batch = collate_imagenet_flow_cache([item], pad_to_length=512)
    calls = {"backbone": 0, "dit": 0}
    def hook(name):
        def increment(module, args):
            calls[name] += 1
        return increment
    handles = [model.model.register_forward_pre_hook(hook("backbone")),
               model.image_flow_head.net.register_forward_pre_hook(hook("dit"))]
    torch.npu.reset_peak_memory_stats(device)
    torch.npu.synchronize()
    started = time.monotonic()
    with torch.inference_mode():
        latents, trace = model.generate("t2i", input_ids=batch["input_ids"].to(device),
            token_types=batch["token_types"].to(device), sigma=batch["sigma"].to(device),
            spans=[(0, item["image_start"], item["image_start"] + 256)],
            initial_noise_bank=noise_for(0, 42)[None], flow_cfg=3.5, flow_solver="heun",
            flow_num_steps=10, order_strategy="joint", use_cache=False, return_trace=True,
            debug_finite=True)
    torch.npu.synchronize()
    elapsed = time.monotonic() - started
    for handle in handles:
        handle.remove()
    if calls != {"backbone": 1, "dit": 20} or latents.shape != (1, 16, 16, 16):
        raise RuntimeError(f"Unexpected generation call counts or shape: {calls}, {latents.shape}")
    if not torch.isfinite(latents).all():
        raise FloatingPointError("Nonfinite generated image")
    output.mkdir(parents=True, exist_ok=True)
    torch.save(latents.cpu(), output / "latents.pt")
    config.experiment.validation_vae_module_root = "external/mar" if (ROOT / "external/mar/models/vae.py").is_file() else "public/code/mar"
    config.experiment.validation_vae_path = "public/vae/mar-kl16/kl16.ckpt"
    config.experiment.validation_vae_scaling_factor = .2325
    vae = load_vae(config, device, "fp32")
    with torch.inference_mode():
        save_png(decode_latents(vae, latents.float(), .2325)[0], output / "generated.png")
        prefix = torch.tensor([tokenizer.encode("The purpose of scientific experiments is", add_special_tokens=False)], device=device)
        text, text_trace = model.generate("text", input_ids=prefix, max_new_tokens=16, return_trace=True)
        caption_ids = batch["input_ids"][:, :item["image_start"] + 257].to(device)
        caption_types = batch["token_types"][:, :caption_ids.shape[1]].to(device)
        clean = torch.zeros((*caption_ids.shape, 16), device=device, dtype=latents.dtype)
        clean[:, item["image_start"]:item["image_start"] + 256] = latents.flatten(2).transpose(1, 2)
        caption, caption_trace = model.generate("i2t", input_ids=caption_ids, token_types=caption_types,
            image_latents=clean, max_new_tokens=16, return_trace=True)
    report = dict(passed=True, checkpoint=str(checkpoint), weights=weights, load_report=load_report,
        export_tensor_samples_verified=count, call_counts=calls, trace=trace,
        generation_seconds=elapsed, peak_allocated_bytes=torch.npu.max_memory_allocated(device),
        text=tokenizer.decode(text[0, prefix.shape[1]:].tolist()), text_trace=text_trace,
        caption=tokenizer.decode(caption[0, caption_ids.shape[1]:].tolist()), caption_trace=caption_trace)
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


def validation_lifecycle(output, label, experiment="z"):
    """Use the production validation coordinator twice, then resume training."""
    from scripts.launch_z import launch_plan
    protocol = joint_experiment_protocol(experiment)
    from utils.evaluation_paths import training_validation_root
    from utils.training_checkpoint import _validate_checkpoint_complete

    plan = launch_plan(smoke=True, label=label, steps=5, environment={}, experiment=experiment)
    run_root = Path(plan["output_root"])
    if (run_root / "config.yaml").exists():
        raise FileExistsError(f"Validation smoke must start fresh: {run_root}")
    command = [part for part in plan["command"] if not part.startswith("experiment.val_every=")]
    command.append("experiment.val_every=2")
    (output / "launch-plan.json").write_text(json.dumps({**plan, "command": command}, indent=2) + "\n")
    run(plan["preflight"], output / "preflight.json")
    run(command, output / "train.log")
    runtime = json.loads((run_root / "training_runtime_metrics.json").read_text())
    if runtime["global_step"] != 5 or runtime["world_size"] != 16 or not math.isfinite(runtime["last_logged_loss"]):
        raise RuntimeError("Training around validation failed")
    reports, fixed_prompts = [], None
    for step in (2, 4):
        directory = training_validation_root(run_root)
        whole = json.loads((directory / f"validation_summary_step_{step}.json").read_text())
        images = json.loads((directory / f"validation_generation/step-{step}/summary.json").read_text())
        if not whole["complete"] or not images["complete"] or images["samples"] != 16:
            raise RuntimeError("Z loss, downstream, and image validation must all complete")
        if images["weight_source"] != "raw" or images["weight_step"] != step or "ema_step" in images:
            raise RuntimeError("Validation must generate from current raw weights")
        if whole["downstream"]["prepare_cache_hit"] != (step == 4):
            raise RuntimeError("Cold/warm validation cache contract failed")
        prompts = [(row["id"], row["prompt"], row["noise_seed"]) for row in images["images"]]
        fixed_prompts = prompts if fixed_prompts is None else fixed_prompts
        if prompts != fixed_prompts:
            raise RuntimeError("Validation prompts/noise changed between steps")
        for row in images["images"]:
            trace = row["trace"]
            if (trace["backbone_calls"], trace["flow_head_calls"]) != (1, 20) or trace["solver"] != "heun":
                raise RuntimeError("Unexpected Z validation generation call counts")
            if trace.get("flow_head_type", "s2") != ("b_single_stream" if experiment == "z-b" else "s2"):
                raise RuntimeError("Validation used the wrong flow-head implementation")
            if not (directory / f"validation_generation/step-{step}" / row["image"]).is_file():
                raise FileNotFoundError(row["image"])
        reports.append(whole)
    _validate_checkpoint_complete(run_root / "checkpoint-5", expected_global_step=5)
    run([sys.executable, "scripts/launch_z.py", "--experiment", experiment, "--smoke", "--label", label, "--steps", "6",
         "--resume-from-checkpoint", str(run_root / "checkpoint-5")], output / "resume.log")
    resumed = json.loads((run_root / "training_runtime_metrics.json").read_text())
    if resumed["run_start_global_step"] != 5 or resumed["global_step"] != 6 or not math.isfinite(resumed["last_logged_loss"]):
        raise RuntimeError("Checkpoint resume after validation failed")
    _validate_checkpoint_complete(run_root / "checkpoint-6", expected_global_step=6)
    for weights, directory in (("current", "hf_model-final"), ("ema", "hf_model-final-ema")):
        run([sys.executable, "scripts/smoke_z.py", "--experiment", experiment, "--checkpoint", str(run_root / directory),
             "--weights", weights, "--output-dir", str(output / weights)], output / f"{weights}.log")
    (output / "report.json").write_text(json.dumps(dict(passed=True, method=protocol.expected_config().experiment.identity.label, run=protocol.RUN,
        run_root=str(run_root), world_size=16, validation_steps=[2, 4],
        images_per_validation=16, fixed_prompts_and_noise=True, raw_and_ema_generation=True,
        solver="heun", steps=10, head_calls=20, image_input_noise_strength=0.01,
        fresh_runtime=runtime, resumed_runtime=resumed, validations=reports), indent=2) + "\n")
    (output / "SMOKE_PASSED").write_text("passed\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", default="r1")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--weights", choices=("current", "ema"), default="ema")
    parser.add_argument("--experiment", choices=("z", "z-b"), default="z")
    parser.add_argument("--validation", action="store_true", help="Train five steps with full validation at 2 and 4, then resume to 6")
    args = parser.parse_args()
    experiment = args.experiment
    protocol = joint_experiment_protocol(experiment)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.validation:
        if args.checkpoint:
            parser.error("--validation and --checkpoint cannot be combined")
        validation_lifecycle(output, args.label, experiment)
        return
    if args.checkpoint:
        generation(args.checkpoint.resolve(), args.weights, output, experiment)
        return
    from scripts.launch_z import launch_plan
    plan = launch_plan(smoke=True, label=args.label, steps=12, environment={}, experiment=experiment)
    run_root = Path(plan["output_root"])
    run([sys.executable, "scripts/launch_z.py", "--experiment", experiment, "--smoke", "--label", args.label, "--steps", "12"], output / "train.log")
    run([sys.executable, "scripts/launch_z.py", "--experiment", experiment, "--smoke", "--label", args.label,
         "--steps", "14", "--resume-from-checkpoint", str(run_root / "checkpoint-12")], output / "resume.log")
    runtime = json.loads((run_root / "training_runtime_metrics_step-12-to-14.json").read_text())
    if (runtime["global_step"] != 14 or runtime["run_start_global_step"] != 12 or runtime["world_size"] != 16
            or runtime["finite_loss_microbatches_checked"] != 8 or not math.isfinite(runtime["last_logged_loss"])):
        raise RuntimeError("Training resume/finite-loss checks failed")
    if not (run_root / "checkpoint-14/checkpoint_complete.json").is_file():
        raise FileNotFoundError("Missing complete resumed checkpoint")
    for weight, directory in (("current", "hf_model-final"), ("ema", "hf_model-final-ema")):
        run([sys.executable, "scripts/smoke_z.py", "--experiment", experiment, "--checkpoint", str(run_root / directory),
            "--weights", weight, "--output-dir", str(output / weight)], output / f"{weight}.log")
    (output / "report.json").write_text(json.dumps(dict(passed=True, run=protocol.RUN, run_root=str(run_root),
        fresh_steps=12, resumed_steps=14, runtime=runtime, raw_and_ema_generation=True), indent=2) + "\n")
    (output / "SMOKE_PASSED").write_text("passed\n")


if __name__ == "__main__":
    main()
