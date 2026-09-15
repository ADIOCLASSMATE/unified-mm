#!/usr/bin/env python3
"""Verify both F scales on real data, checkpoint resume, and raw/EMA generation."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.launch_unified_flow_head_scaling import launch_plan
from utils.positionwise_flow_head_scaling import HEAD_PARAMETERS, NON_HEAD_PARAMETERS


def run(command, output):
    environment = dict(os.environ)
    for key in ("WORLD_SIZE", "RANK", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "GROUP_RANK",
                "GROUP_WORLD_SIZE", "ROLE_RANK", "ROLE_WORLD_SIZE"):
        environment.pop(key, None)
    print(json.dumps({"event": "smoke_phase", "log": str(output), "command": command}), flush=True)
    with output.open("w") as stream:
        subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, env=environment, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", default="20260914-r1")
    parser.add_argument("--reuse-training", action="store_true",
                        help="Reuse completed 0-to-12 and 12-to-14 training reports when rerunning generation checks.")
    args = parser.parse_args()
    os.chdir(ROOT)
    report_root = args.output_dir.resolve()
    report_root.mkdir(parents=True, exist_ok=True)
    completed = []
    for depth in (30, 16):
        plan = launch_plan(depth, smoke=True, label=args.label, steps=12, environment={}, ablation="f")
        output = ROOT / plan["output_root"]
        fresh_report = output / "training_runtime_metrics_step-0-to-12.json"
        if not (args.reuse_training and fresh_report.is_file()):
            run([sys.executable, "scripts/launch_unified_flow_head_scaling.py", "--ablation", "f",
                 "--depth", str(depth), "--smoke", "--label", args.label, "--steps", "12"],
                report_root / f"depth{depth}-training.log")
        checkpoint = output / "checkpoint-12"
        if not (checkpoint / "checkpoint_complete.json").is_file():
            raise FileNotFoundError(checkpoint)
        fresh_runtime = json.loads(fresh_report.read_text())
        assert fresh_runtime["run_start_global_step"] == 0
        assert fresh_runtime["global_step"] == 12
        assert fresh_runtime["finite_loss_microbatches_checked"] == 48
        assert math.isfinite(fresh_runtime["last_logged_loss"])
        resume_command = []
        for argument in plan["command"]:
            if argument.startswith("experiment.resume_from_checkpoint="):
                argument = f"experiment.resume_from_checkpoint={checkpoint}"
            elif argument.startswith("training.stop_after_steps="):
                argument = "training.stop_after_steps=14"
            elif argument.startswith("experiment.deepspeed_bf16_overflow_check_until_step="):
                argument = "experiment.deepspeed_bf16_overflow_check_until_step=14"
            resume_command.append(argument)
        resume_report = output / "training_runtime_metrics_step-12-to-14.json"
        if not (args.reuse_training and resume_report.is_file()):
            run(resume_command, report_root / f"depth{depth}-resume.log")
        assert (output / "checkpoint-14/checkpoint_complete.json").is_file()
        runtime = json.loads(resume_report.read_text())
        assert runtime["global_step"] == 14
        assert runtime["run_start_global_step"] == 12
        assert runtime["world_size"] == 16
        assert runtime["finite_loss_microbatches_checked"] == 8
        assert runtime["cumulative_finite_loss_microbatches_checked"] == 56
        assert math.isfinite(runtime["last_logged_loss"])
        assert runtime["trainability"]["frozen_numel"] == 0
        assert runtime["trainability"]["total_numel"] == HEAD_PARAMETERS[depth] + NON_HEAD_PARAMETERS
        assert (output / "training_runtime_metrics_step-12-to-14.json").is_file()
        for weight, directory in (("current", "hf_model-final"), ("ema", "hf_model-final-ema")):
            run([sys.executable, "scripts/smoke_unified_flow_head_generation.py", "--ablation", "f",
                 "--depth", str(depth), "--weights", weight,
                 "--checkpoint", str(output / directory),
                 "--output-dir", str(report_root / f"depth{depth}-{weight}-generation")],
                report_root / f"depth{depth}-{weight}-generation.log")
        completed.append({"depth": depth, "head_parameters": HEAD_PARAMETERS[depth],
                          "output": str(output), "fresh_steps": 12, "resumed_to_step": 14,
                          "raw_and_ema_generation_passed": True,
                          "fresh_runtime": fresh_runtime, "runtime": runtime})
        (report_root / "report.json").write_text(json.dumps({"complete": len(completed) == 2,
            "arms": completed}, indent=2) + "\n")
    (report_root / "SMOKE_PASSED").write_text("passed\n")
    print(json.dumps({"complete": True, "depths": [30, 16], "report": str(report_root / 'report.json')}), flush=True)


if __name__ == "__main__":
    main()
