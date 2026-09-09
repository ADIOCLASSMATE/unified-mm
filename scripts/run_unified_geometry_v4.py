"""Bounded sequential development-Notebook supervisor for V4 extraction."""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.research.representation_protocol import emit, write_json


def run_stage(out, name, commands, completed, timeout=2400):
    processes, handles = [], []
    started = time.monotonic()
    try:
        for i, command in enumerate(commands):
            handle = (out / f"stage-{name}-{i:02d}.log").open("a")
            handles.append(handle)
            processes.append(
                subprocess.Popen(
                    command,
                    cwd=ROOT,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            )
        write_json(
            out / "driver-status.json",
            {
                "active": name,
                "driver_pid": os.getpid(),
                "child_pids": [p.pid for p in processes],
                "completed": completed,
            },
        )
        emit("geometry_stage_start", stage=name, pids=[p.pid for p in processes])
        codes = [
            p.wait(timeout=max(1, timeout - (time.monotonic() - started)))
            for p in processes
        ]
        if any(codes):
            raise RuntimeError(f"Stage {name} returned {codes}")
    finally:
        for process in processes:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
        for handle in handles:
            handle.close()
    record = {
        "stage": name,
        "seconds": time.monotonic() - started,
        "returncodes": codes,
    }
    completed.append(record)
    write_json(out / "driver-progress.json", completed)
    emit("geometry_stage_complete", **record)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    out = args.output_dir.resolve()
    progress = out / "driver-progress.json"
    completed = json.loads(progress.read_text()) if progress.exists() else []
    plan = {
        "notebook": "dev-wjx-ascend",
        "training_updates": 0,
        "timeout_per_stage": 2400,
        "download_wait_timeout": 1800,
        "states": ["final_ema", "final_raw", "init42", "init43", "init44"],
        "steps": [
            "existing-cache extraction",
            "wait for fixed COCO download",
            "COCO VAE encode",
            "COCO smoke",
            "COCO extraction",
            "ImageNet robustness",
        ],
    }
    if (out / "execution-plan.json").exists():
        assert json.loads((out / "execution-plan.json").read_text()) == plan
    else:
        write_json(out / "execution-plan.json", plan)

    def extraction(name, state, profiles, datasets, limit=0):
        command = [
            "bash",
            "script/selfless/extract_unified_geometry_v4_dev_ascend16.sh",
            "--output-dir",
            str(out),
            "--state",
            state,
            "--profiles",
            profiles,
            "--datasets",
            datasets,
            "--stage",
            name,
        ]
        if limit:
            command += ["--limit", str(limit)]
        if not any(record["stage"] == name for record in completed):
            run_stage(out, name, [command], completed)
        markers = list((out / "stage-markers" / name).glob("rank-*.json"))
        assert len(markers) == 16, f"Incomplete rank markers for {name}"
        ranks = set()
        for marker in markers:
            record = json.loads(marker.read_text())
            assert record["state"] == state and record["limit"] == limit
            assert record["profiles"] == profiles.split(",")
            assert record["datasets"] == datasets.split(",")
            assert record["world_size"] == 16 and record["arithmetic_checked"]
            assert all(Path(path).is_file() for path in record["files"])
            ranks.add(record["rank"])
        assert ranks == set(range(16))
        emit("geometry_stage_artifacts_verified", stage=name, rank_markers=16)

    try:
        for state in plan["states"]:
            profiles = "native,bare,neutral" if state == "final_ema" else "native"
            extraction(
                f"existing-{state}",
                state,
                profiles,
                "imagenet_images,imagenet_texts,aro_images,aro_texts",
            )
        started = time.monotonic()
        while not (out / "download-complete.json").exists():
            write_json(
                out / "driver-status.json",
                {
                    "active": "waiting-for-download",
                    "driver_pid": os.getpid(),
                    "wait_seconds": time.monotonic() - started,
                    "completed": completed,
                },
            )
            if time.monotonic() - started > 1800:
                raise TimeoutError("Fixed COCO image download did not complete")
            time.sleep(15)
        assert (
            json.loads((out / "download-complete.json").read_text())["images"] == 11776
        )
        commands = []
        for rank in range(16):
            commands.append(
                [
                    ".venv/bin/python",
                    "scripts/imagenet_encode_kl16_vae.py",
                    "--source_mode",
                    "manifest_jsonl",
                    "--source_manifest_jsonl",
                    str(out / "coco-image-manifest.jsonl"),
                    "--cache_shard_dir",
                    str(out / "coco-vae/shards"),
                    "--num_shards",
                    "16",
                    "--shard_index",
                    str(rank),
                    "--device",
                    f"npu:{rank}",
                    "--vae_dtype",
                    "fp16",
                    "--batch_size",
                    "32",
                    "--num_workers",
                    "2",
                    "--no_hash",
                ]
            )
        run_stage(out, "encode-coco", commands, completed)
        extraction(
            "smoke-coco",
            "final_ema",
            "native,bare,neutral",
            "coco_images,coco_texts",
            limit=64,
        )
        for state in plan["states"]:
            profiles = (
                "native,bare,neutral,native_sigma1,native_sigma2,native_mean"
                if state == "final_ema"
                else "native"
            )
            extraction(f"coco-{state}", state, profiles, "coco_images,coco_texts")
        extraction(
            "imagenet-robust",
            "final_ema",
            "native_sigma1,native_sigma2,native_mean",
            "imagenet_images,imagenet_texts",
        )
        write_json(
            out / "extraction-complete.json",
            {"completed": completed, "training_updates": 0},
        )
        write_json(
            out / "driver-status.json",
            {"active": None, "driver_pid": os.getpid(), "completed": completed},
        )
    except (AssertionError, OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
        write_json(
            out / "extraction-failed.json", {"error": str(exc), "completed": completed}
        )
        raise


if __name__ == "__main__":
    main()
