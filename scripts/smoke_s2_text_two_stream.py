#!/usr/bin/env python3
"""Train 12 updates, resume to 14, then validate raw/EMA on the fixed 16-NPU Notebook."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.launch_showo2_unified import launch_plan
from utils.training_checkpoint import _validate_checkpoint_complete

VARIANT = "single-text-two-stream"


def run(command, log):
    print(json.dumps({"command": command, "log": str(log)}), flush=True)
    with log.open("w") as stream:
        subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", default="r1")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "report.json").exists():
        raise FileExistsError(f"Smoke report already exists: {output}")
    plan = launch_plan(VARIANT, smoke=True, label=args.label, steps=12, environment={})
    run_root = Path(plan["output_root"]).resolve()
    command = [sys.executable, "scripts/launch_showo2_unified.py", "--variant", VARIANT,
               "--smoke", "--label", args.label]
    run(command + ["--steps", "12"], output / "train.log")
    _validate_checkpoint_complete(run_root / "checkpoint-12", expected_global_step=12)
    run(command + ["--steps", "14", "--resume-from-checkpoint", str(run_root / "checkpoint-12")],
        output / "resume.log")
    _validate_checkpoint_complete(run_root / "checkpoint-14", expected_global_step=14)
    runtime = json.loads((run_root / "training_runtime_metrics_step-12-to-14.json").read_text())
    if (runtime["global_step"] != 14 or runtime["run_start_global_step"] != 12 or runtime["world_size"] != 16
            or runtime["finite_loss_microbatches_checked"] != 8 or not math.isfinite(runtime["last_logged_loss"])):
        raise RuntimeError("Training resume or finite-loss validation failed")
    run([sys.executable, "scripts/smoke_showo2_generation.py", "--variant", VARIANT,
         "--run-dir", str(run_root), "--output-dir", str(output / "generation")], output / "generation.log")
    generation = json.loads((output / "generation/report.json").read_text())
    if not generation["passed"] or any(not x["text_two_stream"] for x in generation["reports"]):
        raise RuntimeError("Raw/EMA text two-stream verification failed")
    report = dict(passed=True, variant=VARIANT, run_root=str(run_root),
                  fresh_steps=12, resumed_steps=14, runtime=runtime, generation=generation)
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    (output / "SMOKE_PASSED").write_text("passed\n")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
