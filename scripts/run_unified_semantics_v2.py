"""Bounded sequential V2 extraction supervisor for the fixed development host."""

import argparse
import json
import subprocess
import time
from pathlib import Path


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output_dir.resolve()
    stages = [
        ("smoke-init42", "init42", "bare,native", 32),
        (
            "final-ema",
            "final_ema",
            "bare,native,neutral,bare_sigma1,bare_sigma2,native_sigma1,native_sigma2,bare_mean",
            0,
        ),
        ("final-raw", "final_raw", "bare,native", 0),
        ("init42", "init42", "bare,native", 0),
        ("init43", "init43", "bare,native", 0),
        ("init44", "init44", "bare,native", 0),
    ]
    plan = {
        "stages": stages,
        "timeout_seconds_per_stage": 2400,
        "training_updates": 0,
        "development_notebook": "dev-wjx-ascend",
    }
    plan_path = output / "execution-plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != json.loads(
        json.dumps(plan)
    ):
        raise ValueError("Existing execution plan differs")
    save(plan_path, plan)
    completed = []
    for name, state, profiles, limit in stages:
        command = [
            "bash",
            "script/selfless/probe_unified_semantics_v2_dev_ascend16.sh",
            str(output),
            state,
            profiles,
        ]
        if limit:
            command += ["--limit", str(limit)]
        started = time.monotonic()
        save(output / "driver-status.json", {"active": name, "completed": completed})
        print("Starting", name, flush=True)
        with (output / f"extract-{name}.log").open("a") as log:
            try:
                result = subprocess.run(
                    command,
                    cwd=root,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=2400,
                    check=False,
                )
                code = result.returncode
            except subprocess.TimeoutExpired:
                code = 124
        record = {
            "stage": name,
            "returncode": code,
            "seconds": time.monotonic() - started,
        }
        print(json.dumps(record), flush=True)
        if code:
            save(output / "extraction-failed.json", {**record, "completed": completed})
            raise SystemExit(code)
        completed.append(record)
    save(
        output / "extraction-complete.json",
        {"completed": completed, "training_updates": 0},
    )
    save(output / "driver-status.json", {"active": None, "completed": completed})


if __name__ == "__main__":
    main()
