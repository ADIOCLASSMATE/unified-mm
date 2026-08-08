#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from caption_farm.io import atomic_write_json, load_json, utc_now
from caption_farm.queue import TaskStore
from caption_farm.worker import VllmOpenAIEngine, load_worker_tuning


def _events(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _worker_argv(
    args: argparse.Namespace,
    worker_id: str,
    gpu: int,
    port: int,
    *,
    post_claim_delay_seconds: float = 0,
    lease_seconds: float | None = None,
) -> tuple[list[str], dict[str, str]]:
    argv = [
        args.runtime_python,
        str(REPO_ROOT / "scripts" / "imagenet_qwen_caption_farm.py"),
        "worker",
        "--run-dir",
        str(args.run_dir),
        "--worker-id",
        worker_id,
        "--max-tasks",
        str(args.tasks_per_worker),
        "--request-concurrency",
        str(args.request_concurrency),
        "--claim-batch-size",
        str(args.claim_batch_size),
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--server-port",
        str(port),
    ]
    if post_claim_delay_seconds:
        argv.extend(["--post-claim-delay-seconds", str(post_claim_delay_seconds)])
    if lease_seconds is not None:
        argv.extend(["--lease-seconds", str(lease_seconds)])
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
        }
    )
    return argv, environment


def _start(
    args: argparse.Namespace,
    worker_id: str,
    gpu: int,
    port: int,
    **kwargs: Any,
) -> subprocess.Popen[str]:
    argv, environment = _worker_argv(args, worker_id, gpu, port, **kwargs)
    return subprocess.Popen(
        argv,
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def _terminate(processes: list[subprocess.Popen[str]]) -> None:
    for process in processes:
        if process.poll() is None:
            process.terminate()
    for process in processes:
        if process.poll() is None:
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)


def _worker_report(run_dir: Path, worker_id: str) -> dict[str, Any]:
    rows = _events(run_dir / "workers" / f"{worker_id}.jsonl")
    ready = next(row for row in rows if row["event"] == "model_ready")
    commits = [row for row in rows if row["event"] == "caption_committed"]
    batch_starts = [row for row in rows if row["event"] == "batch_started"]
    batch_ends = [row for row in rows if row["event"] == "batch_completed"]
    first = datetime.fromisoformat(batch_starts[0]["timestamp"])
    last = datetime.fromisoformat(batch_ends[-1]["timestamp"])
    generation_seconds = (last - first).total_seconds()
    return {
        "worker_id": worker_id,
        "load_seconds": ready["load_seconds"],
        "peak_gpu_memory_used_mib": max(
            int(row.get("peak_gpu_memory_used_mib") or 0) for row in rows
        ),
        "committed": len(commits),
        "generation_seconds": generation_seconds,
        "captions_per_second": len(commits) / generation_seconds,
        "task_keys": [row["task_key"] for row in commits],
    }


def _snapshot(run_dir: Path) -> dict[str, Any]:
    store = TaskStore(run_dir)
    try:
        return store.snapshot()
    finally:
        store.close()


def run_scale(args: argparse.Namespace, token: str) -> dict[str, Any]:
    if args.workers < 1 or args.workers > 4:
        raise ValueError("dev scale test supports one to four workers")
    before = _snapshot(args.run_dir)
    worker_ids = [f"dev-scale{args.workers}-{token}-gpu{gpu}" for gpu in range(args.workers)]
    processes = [
        _start(args, worker_id, gpu, args.base_port + gpu)
        for gpu, worker_id in enumerate(worker_ids)
    ]
    outputs: dict[str, str] = {}
    try:
        for worker_id, process in zip(worker_ids, processes, strict=True):
            output, _ = process.communicate(timeout=args.timeout_seconds)
            outputs[worker_id] = output[-4000:]
            if process.returncode != 0:
                raise RuntimeError(
                    f"worker {worker_id} exited {process.returncode}: {output[-4000:]}"
                )
    finally:
        _terminate(processes)
    workers = [_worker_report(args.run_dir, worker_id) for worker_id in worker_ids]
    keys = [key for worker in workers for key in worker["task_keys"]]
    if len(keys) != len(set(keys)):
        raise RuntimeError("multi-GPU workers produced duplicate visible task keys")
    after = _snapshot(args.run_dir)
    expected = args.workers * args.tasks_per_worker
    if int(after["COMPLETE"]) - int(before["COMPLETE"]) != expected:
        raise RuntimeError("queue completion delta did not match the scale-test task count")
    elapsed = max(worker["generation_seconds"] for worker in workers)
    return {
        "schema": "imagenet_caption_farm_dev_scale_v1",
        "timestamp": utc_now(),
        "mode": "scale",
        "worker_count": args.workers,
        "tasks_per_worker": args.tasks_per_worker,
        "request_concurrency": args.request_concurrency,
        "max_num_seqs": args.max_num_seqs,
        "claim_batch_size": args.claim_batch_size,
        "before": before,
        "after": after,
        "workers": workers,
        "duplicate_visible_keys": 0,
        "aggregate_generation_captions_per_second": expected / elapsed,
        "status": "passed",
    }


def run_kill_recovery(args: argparse.Namespace, token: str) -> dict[str, Any]:
    if args.workers < 2:
        raise ValueError("kill recovery needs at least two visible GPUs")
    before = _snapshot(args.run_dir)
    victim_id = f"dev-kill-{token}-gpu0"
    replacement_id = f"dev-recover-{token}-gpu1"
    victim = _start(
        args,
        victim_id,
        0,
        args.base_port,
        post_claim_delay_seconds=300,
        lease_seconds=args.kill_lease_seconds,
    )
    victim_log = args.run_dir / "workers" / f"{victim_id}.jsonl"
    deadline = time.monotonic() + args.timeout_seconds
    lease_id: str | None = None
    try:
        while time.monotonic() < deadline:
            rows = _events(victim_log)
            started = [row for row in rows if row["event"] == "batch_started"]
            if started:
                lease_id = str(started[-1]["lease_id"])
                break
            if victim.poll() is not None:
                output, _ = victim.communicate()
                raise RuntimeError(f"victim exited before claiming: {output[-4000:]}")
            time.sleep(0.5)
        if lease_id is None:
            raise TimeoutError("victim did not claim a batch before the dev-test timeout")
        lease = load_json(args.run_dir / "state" / "leases" / f"{lease_id}.json")
        victim_keys = {str(task["task_key"]) for task in lease["tasks"]}
        ready = next(row for row in _events(victim_log) if row["event"] == "model_ready")
        server_pid = int(ready["inference_server_pid"])
        victim.kill()
        victim.wait(timeout=30)
        try:
            os.killpg(server_pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        wait_seconds = max(0.0, float(lease["lease_expires_at"]) - time.time() + 1.0)
        time.sleep(wait_seconds)
        replacement = _start(args, replacement_id, 1, args.base_port + 1)
        output, _ = replacement.communicate(timeout=args.timeout_seconds)
        if replacement.returncode != 0:
            raise RuntimeError(f"replacement exited {replacement.returncode}: {output[-4000:]}")
        report = _worker_report(args.run_dir, replacement_id)
        recovered_keys = set(report["task_keys"])
        if recovered_keys != victim_keys:
            raise RuntimeError("replacement did not recover exactly the killed worker's lease")
        after = _snapshot(args.run_dir)
        if int(after["COMPLETE"]) - int(before["COMPLETE"]) != len(victim_keys):
            raise RuntimeError("kill-recovery completion delta is inconsistent")
        return {
            "schema": "imagenet_caption_farm_dev_kill_recovery_v1",
            "timestamp": utc_now(),
            "mode": "kill-recovery",
            "victim_worker": victim_id,
            "victim_exit_code": victim.returncode,
            "lease_id": lease_id,
            "lease_task_count": len(victim_keys),
            "lease_seconds": args.kill_lease_seconds,
            "replacement": report,
            "before": before,
            "after": after,
            "recovered_exactly": True,
            "status": "passed",
        }
    finally:
        _terminate([victim])


def run_task_regression(args: argparse.Namespace, token: str) -> dict[str, Any]:
    if not args.ordinals:
        raise ValueError("task-regression requires --ordinals")
    store = TaskStore(args.run_dir)
    try:
        run = deepcopy(store.run)
        tasks = [store._task_from_ordinal(ordinal) for ordinal in args.ordinals]
    finally:
        store.close()
    tuning = load_worker_tuning(args.run_dir, run)
    if tuning is not None:
        inference = run["model"]["inference"]
        for key in (
            "request_concurrency",
            "max_num_seqs",
            "max_image_side",
            "max_image_pixels",
        ):
            if key in tuning:
                inference[key] = int(tuning[key])
    inference = run["model"]["inference"]
    inference["port"] = args.base_port
    worker_id = f"dev-task-regression-{token}"
    engine = VllmOpenAIEngine(
        run,
        worker_id,
        args.run_dir / "workers" / f"{worker_id}.server.log",
    )
    results: list[dict[str, Any]] = []
    try:
        engine.start(tasks[0])
        with ThreadPoolExecutor(max_workers=min(len(tasks), args.request_concurrency)) as executor:
            futures = {executor.submit(engine.caption, task): task for task in tasks}
            for future in as_completed(futures):
                task = futures[future]
                result = future.result()
                results.append(
                    {
                        "ordinal": task["ordinal"],
                        "task_key": task["task_key"],
                        "image_id": task["image_id"],
                        "source_path": task["source_path"],
                        "caption_slot": task["caption_slot"],
                        **result,
                    }
                )
    finally:
        engine.close()
    results.sort(key=lambda item: int(item["ordinal"]))
    if len(results) != len(tasks) or any(not str(item.get("caption") or "").strip() for item in results):
        raise RuntimeError("task regression did not produce every requested non-empty caption")
    return {
        "schema": "imagenet_caption_farm_dev_task_regression_v1",
        "timestamp": utc_now(),
        "mode": "task-regression",
        "worker_id": worker_id,
        "model_load_seconds": engine.load_seconds,
        "request_concurrency": args.request_concurrency,
        "max_num_seqs": inference["max_num_seqs"],
        "max_image_side": inference.get("max_image_side"),
        "max_image_pixels": inference.get("max_image_pixels"),
        "results": results,
        "status": "passed",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Automated dev-wjx caption-farm GPU tests")
    parser.add_argument(
        "--mode",
        choices=("scale", "kill-recovery", "task-regression"),
        default="scale",
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--runtime-python", required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--tasks-per-worker", type=int, default=16)
    parser.add_argument("--request-concurrency", type=int, default=16)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--claim-batch-size", type=int, default=32)
    parser.add_argument("--base-port", type=int, default=18100)
    parser.add_argument("--kill-lease-seconds", type=float, default=30)
    parser.add_argument("--timeout-seconds", type=float, default=1500)
    parser.add_argument(
        "--ordinals",
        type=lambda value: [int(item) for item in value.split(",") if item.strip()],
    )
    args = parser.parse_args()
    args.run_dir = args.run_dir.resolve()
    token = datetime.now().strftime("%Y%m%dT%H%M%S")
    if args.mode == "scale":
        report = run_scale(args, token)
    elif args.mode == "kill-recovery":
        report = run_kill_recovery(args, token)
    else:
        report = run_task_regression(args, token)
    destination = args.run_dir / "dev_tests" / f"{args.mode}-{token}.json"
    atomic_write_json(destination, report)
    print(json.dumps({**report, "report_path": str(destination)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
