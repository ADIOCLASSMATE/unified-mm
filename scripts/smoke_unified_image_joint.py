#!/usr/bin/env python3
"""Qualify joint T2I/I2T training, both data cursors, and raw/EMA generation."""

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.launch_unified_image_joint import launch_plan
from utils.image_joint_training import TOTAL_PARAMETERS


def run(command, output):
    environment = dict(os.environ)
    for key in ("WORLD_SIZE", "RANK", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "GROUP_RANK",
                "GROUP_WORLD_SIZE", "ROLE_RANK", "ROLE_WORLD_SIZE"):
        environment.pop(key, None)
    print(json.dumps({"event": "smoke_phase", "log": str(output), "command": command}), flush=True)
    with output.open("w") as stream:
        subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, env=environment, check=True)


def check_runtime(output, start, end):
    report = json.loads((output / f"training_runtime_metrics_step-{start}-to-{end}.json").read_text())
    assert report["run_start_global_step"] == start and report["global_step"] == end
    assert report["world_size"] == 16
    assert report["finite_loss_microbatches_checked"] == 2 * (end - start)
    assert math.isfinite(report["last_logged_loss"])
    assert report["trainability"]["frozen_numel"] == 0
    assert report["trainability"]["total_numel"] == TOTAL_PARAMETERS
    checkpoint = output / f"checkpoint-{end}"
    assert (checkpoint / "checkpoint_complete.json").is_file()
    for rank in range(16):
        state = torch.load(checkpoint / f"data_state_rank_{rank:05d}.pt",
                           map_location="cpu", weights_only=False)
        assert state["rank"] == rank and state["world_size"] == 16
        assert state["micro_step"] == 2 * end and state["global_step"] == end
        assert state["schedule"] == ["t2i", "i2t"] and state["schedule_position"] == 0
        assert "climbmix" not in state
        assert set(state["image_sources"]) == {"t2i", "i2t"}
        assert all(cursor == {"epoch": 0, "batches_consumed": end}
                   for cursor in state["image_sources"].values())
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", default="20260914-r1")
    parser.add_argument("--reuse-training", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    report_root = args.output_dir.resolve()
    report_root.mkdir(parents=True, exist_ok=True)
    plan = launch_plan(smoke=True, label=args.label, steps=12, environment={})
    output = ROOT / plan["output_root"]
    if not (args.reuse_training and (output / "training_runtime_metrics_step-0-to-12.json").is_file()):
        run([sys.executable, "scripts/launch_unified_image_joint.py", "--smoke",
             "--label", args.label, "--steps", "12"], report_root / "training.log")
    fresh_runtime = check_runtime(output, 0, 12)
    resume_command = []
    for argument in plan["command"]:
        if argument.startswith("experiment.resume_from_checkpoint="):
            argument = f"experiment.resume_from_checkpoint={output / 'checkpoint-12'}"
        elif argument.startswith("training.stop_after_steps="):
            argument = "training.stop_after_steps=14"
        elif argument.startswith("experiment.deepspeed_bf16_overflow_check_until_step="):
            argument = "experiment.deepspeed_bf16_overflow_check_until_step=14"
        resume_command.append(argument)
    if not (args.reuse_training and (output / "training_runtime_metrics_step-12-to-14.json").is_file()):
        run(resume_command, report_root / "resume.log")
    resumed_runtime = check_runtime(output, 12, 14)
    rows = [json.loads(line) for line in (output / "training_metrics.jsonl").read_text().splitlines()]
    for row in rows:
        metrics = row.get("metrics", row)
        assert metrics["source/t2i_microbatches"] == metrics["source/i2t_microbatches"] == 16
        assert not any("climbmix" in key for key in metrics)
        assert metrics["pack/seq_len"] == 512
        assert math.isfinite(metrics["train/loss_text"]) and math.isfinite(metrics["train/loss_image_flow"])
    for weight, directory in (("current", "hf_model-final"), ("ema", "hf_model-final-ema")):
        run([sys.executable, "scripts/smoke_unified_flow_head_generation.py",
             "--depth", "8", "--image-joint", "--weights", weight,
             "--checkpoint", str(output / directory),
             "--output-dir", str(report_root / f"{weight}-generation")],
            report_root / f"{weight}-generation.log")
    report = {"passed": True, "formal_world_size": 32, "smoke_world_size": 16,
              "same_formal_per_rank_batch": 32, "fresh_steps": 12, "resumed_to_step": 14,
              "all_rank_t2i_i2t_cursors_verified": True, "climbmix_loaded": False,
              "raw_and_ema_generation_passed": True, "fresh_runtime": fresh_runtime,
              "resumed_runtime": resumed_runtime, "output": str(output)}
    (report_root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (report_root / "SMOKE_PASSED").write_text("passed\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
