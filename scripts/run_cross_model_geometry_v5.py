"""Persistent, resumable V5 stage supervision on the fixed development Notebook."""

import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path

from utils.research.geometry_v5_assets import ROOT, RUN, emit, write_json


def run_stage(root, name, command):
    marker = root / "supervisor-markers" / f"{name}.json"
    if marker.exists():
        value = json.loads(marker.read_text())
        if value["returncode"] == 0 and value["command"] == command:
            emit("stage_already_complete", stage=name)
            return
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    with (logs / f"{name}.log").open("a") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        write_json(
            root / "supervisor-status.json",
            {
                "driver_pid": os.getpid(),
                "active_stage": name,
                "child_pid": process.pid,
                "command": command,
                "log": str(logs / f"{name}.log"),
                "scope": "f_stages"
                if name.startswith("f-")
                else "flow_stages"
                if name.startswith(("janusflow-", "showo2-"))
                else "encoder_stages",
            },
        )
        emit("stage_started", stage=name, pid=process.pid)
        started = time.monotonic()
        try:
            while process.poll() is None:
                if time.monotonic() - started > 6 * 3600:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=60)
                    raise TimeoutError(f"V5 stage exceeded six-hour bound: {name}")
                time.sleep(5)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=60)
        value = {
            "stage": name,
            "returncode": process.returncode,
            "command": command,
            "seconds": time.monotonic() - started,
        }
        write_json(marker, value)
        emit("stage_finished", **value)
        if process.returncode:
            raise RuntimeError(f"Stage failed: {name}, inspect {log.name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=RUN)
    parser.add_argument("--stage-set", choices=["f", "encoders", "flow"], default="f")
    parser.add_argument("--models", default="qwen_text,dinov2,mae,siglip")
    args = parser.parse_args()
    if args.stage_set == "flow":
        for name in args.models.split(","):
            assert name in {"janusflow", "showo2"}
            base = [
                "bash",
                "script/selfless/extract_geometry_v5_flow_dev.sh",
                "--output-dir",
                str(args.output_dir),
                "--model",
                name,
            ]
            run_stage(
                args.output_dir,
                f"{name}-cal-smoke",
                base + ["--cal-smoke", "--batch-size", "2"],
            )
            run_stage(args.output_dir, f"{name}-full", base + ["--batch-size", "8"])
            run_stage(
                args.output_dir,
                f"{name}-robust",
                base
                + [
                    "--paths",
                    "generation",
                    "--profiles",
                    "seed1,seed2,image_midpoint",
                    "--batch-size",
                    "8",
                ],
            )
        write_json(
            args.output_dir / "supervisor-status.json",
            {
                "driver_pid": os.getpid(),
                "active_stage": None,
                "completed_stage_set": "flow",
                "models": args.models,
            },
        )
        return
    if args.stage_set == "encoders":
        for name in args.models.split(","):
            assert name in {"qwen_text", "dinov2", "mae", "siglip"}
            base = [
                "bash",
                "script/selfless/extract_cross_model_geometry_v5_dev_ascend16.sh",
                "scripts/extract_geometry_v5_encoders.py",
                "--output-dir",
                str(args.output_dir),
                "--model",
                name,
            ]
            run_stage(
                args.output_dir,
                f"{name}-cal-smoke",
                base + ["--cal-smoke", "--batch-size", "2"],
            )
            run_stage(
                args.output_dir,
                f"{name}-full",
                base + ["--batch-size", "8" if name == "siglip" else "32"],
            )
        write_json(
            args.output_dir / "supervisor-status.json",
            {
                "driver_pid": os.getpid(),
                "active_stage": None,
                "completed_stage_set": "encoders",
                "models": args.models,
            },
        )
        return
    base = [
        "bash",
        "script/selfless/extract_cross_model_geometry_v5_dev_ascend16.sh",
        "scripts/extract_geometry_v5_f.py",
    ]
    run_stage(
        args.output_dir,
        "f-cal-smoke",
        base
        + [
            "--output-dir",
            str(args.output_dir / "f-smoke"),
            "--profiles",
            "native",
            "--limit",
            "32",
            "--stage",
            "f-cal-smoke",
            "--batch-size",
            "2",
        ],
    )
    run_stage(
        args.output_dir,
        "f-full",
        base
        + [
            "--output-dir",
            str(args.output_dir / "f-v4"),
            "--profiles",
            "native,bare,neutral",
            "--stage",
            "f-full",
            "--batch-size",
            "16",
        ],
    )
    run_stage(
        args.output_dir,
        "f-robust",
        base
        + [
            "--output-dir",
            str(args.output_dir / "f-v4"),
            "--profiles",
            "native_sigma1,native_sigma2,native_mean",
            "--stage",
            "f-robust",
            "--batch-size",
            "16",
        ],
    )
    write_json(
        args.output_dir / "supervisor-status.json",
        {
            "driver_pid": os.getpid(),
            "active_stage": None,
            "completed_stage_set": args.stage_set,
        },
    )


if __name__ == "__main__":
    main()
