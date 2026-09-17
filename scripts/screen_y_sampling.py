#!/usr/bin/env python3
"""Bounded, paired Y K/CFG-schedule screen using the canonical FID evaluator."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = args.model_source.resolve()
    environment = dict(os.environ)
    for key in ("WORLD_SIZE", "RANK", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "GROUP_RANK",
                "GROUP_WORLD_SIZE", "ROLE_RANK", "ROLE_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        environment.pop(key, None)
    rows = []
    paired_records = None
    cases = [(8, "linear")] if args.smoke else [(k, s) for k in (8, 20, 32, 64) for s in ("constant", "linear")]
    for k, schedule in cases:
        case = output / f"k{k}-{schedule}"
        metrics = case / "metrics.json"
        command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=16",
            "scripts/evaluate_single_stream_fid_is.py", "--config", "configs/selfless/unified_y_33b_ascend64.yaml",
            "--model_source", str(source), "--output_dir", str(case), "--device", "npu",
            "--samples", "16" if args.smoke else "2000", "--batch_size", "16" if args.smoke else "256",
            "--is_splits", "1" if args.smoke else "2", "--caption_sequence_mode", "t2i",
            "--sampling_steps", "10", "--reveal_steps", str(k), "--temperature", "1.0", "--cfg", "3.5",
            "--cfg_schedule", "constant", "--reveal_cfg_schedule", schedule, "--flow_solver", "heun",
            "--strategies", "random", "--vae_dtype", "fp32", "--vae_decode_batch_size", "16",
            "--inception_weights_path", "public/models/torch-fidelity/weights-inception-2015-12-05-6726825d.pth",
            "--real_stats_path", "public/datasets/imagenet_full/fid_stats/inception_v3_2048_imagenet_val50000_256.pt",
            "--canonical_pairing", "--save_image_count", "16" if args.smoke else "64", "--no-resume_progress"]
        if not args.smoke:
            command += ["--samples_per_class", "2", "--screening_fid"]
        started = time.monotonic()
        if metrics.exists():
            raise FileExistsError(f"Use a fresh screening directory: {metrics}")
        (output / f"k{k}-{schedule}.command.json").write_text(json.dumps(command, indent=2) + "\n")
        with (output / f"k{k}-{schedule}.log").open("w") as log:
            subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
        result = json.loads(metrics.read_text())
        metric = result["strategies"]["random"]
        assert result["samples_evaluated"] == (16 if args.smoke else 2000)
        assert result["reveal_steps"] == k and result["reveal_cfg_schedule"] == schedule
        if not args.smoke:
            records = result["screening_sample_records"]
            if paired_records is None:
                paired_records = records
            assert records == paired_records
            assert result["metric_protocol"]["is_split_plan"]["class_count"] == 1000
        row = dict(K=k, S=10, cfg_max=3.5, temperature=1., reveal_cfg_schedule=schedule,
                   fid=metric["fid"], inception_score=metric["inception_score_mean"],
                   generation_seconds=metric["generation_wall_seconds"],
                   end_to_end_seconds=time.monotonic()-started,
                   samples_per_second=metric["generation_samples_per_second"],
                   cost=result["y_sampling_cost"], metrics=str(metrics))
        rows.append(row)
        report = dict(complete=len(rows)==len(cases), model_source=str(source), weights="EMA",
            checkpoint_step=2000, smoke=args.smoke, samples_per_case=16 if args.smoke else 2000,
            class_count=None if args.smoke else 1000, seed=42, small_sample_fid=True,
            interpretation="Paired screening only; not FID50k and not a final model ranking.", cases=rows)
        (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(row), flush=True)
    lines = ["# Y step-2000 sampling screen", "", "EMA; fixed prompts, initial noise and reveal orders. "
             "2000 generated samples vs ImageNet-val 50k reference; screening FID only.", "",
             "| K | Reveal CFG | FID | IS | Generation s | Head calls |", "| --- | --- | --- | --- | --- | --- |"]
    for row in rows:
        lines.append(f"| {row['K']} | {row['reveal_cfg_schedule']} | {row['fid']} | {row['inception_score']:.3f} | "
                     f"{row['generation_seconds']:.2f} | {row['cost']['head_calls']} |")
    (output / "summary.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
