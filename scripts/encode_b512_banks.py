"""Encode frozen image banks on bounded CPU workers while text synthesis runs."""

import argparse
from collections import deque
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from scripts.prepare_b512_posterior_bank import prepare_bank
from data_synthesis.io import atomic_json
from data_synthesis.integrity import check_file_size, hashing_enabled


def run_plan(path):
    path = Path(path).resolve()
    plan = json.loads(path.read_text())
    compute_hashes = hashing_enabled() and hashing_enabled(plan)
    root = path.parent
    lock = (root / "encoding.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    workers, threads = int(plan.get("workers", 2)), int(plan.get("threads", 4))
    if workers < 1 or threads < 1:
        raise ValueError("positive worker/thread limits required")
    env = {key: value for key, value in os.environ.items() if key != "SII_API_KEY"}
    env.update(TORCH_DEVICE_BACKEND_AUTOLOAD="0", OMP_NUM_THREADS=str(threads),
               MKL_NUM_THREADS=str(threads), TOKENIZERS_PARALLELISM="false")
    banks = [{**bank, "pending": None, "done": 0, "merged": False} for bank in plan["banks"]]
    active, written = {}, {}
    started = datetime.now(timezone.utc).isoformat()

    def record(target, payload):
        if written.get(str(target)) != payload:
            atomic_json(target, {**payload, "updated_at": datetime.now(timezone.utc).isoformat()})
            written[str(target)] = payload

    def report(state="running", error=None):
        record(root / "job_status.json", {"controller_pid": os.getpid(), "started_at": started,
               "state": state, "error": error, "active_children": sorted(active), "compute_hashes": compute_hashes,
               "completed_banks": [b["name"] for b in banks if b["merged"]]})
        for bank in banks:
            directory = Path(bank["manifest_dir"])
            directory.mkdir(parents=True, exist_ok=True)
            ids = [pid for pid, value in active.items() if value["bank"] is bank]
            status = ("completed" if bank["merged"] else state if state != "running"
                      else "running" if ids else "queued")
            record(directory / "job_status.json", {"controller_pid": os.getpid(), "state": status,
                   "bank": bank["name"], "child_pids": ids, "completed_shards": bank["done"],
                   "total_shards": bank.get("shards"), "records": bank.get("records"),
                   "waiting_for": bank.get("wait_for") if bank["pending"] is None else None,
                   "posterior_index": str(Path(bank["cache_dir"]) / "posterior_index.json")})

    def launch(bank, argv, name, kind):
        log_path = Path(bank["manifest_dir"]) / (name + ".log")
        with log_path.open("a") as log:
            proc = subprocess.Popen(argv, cwd=plan["cwd"], env=env, stdin=subprocess.DEVNULL,
                                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        active[proc.pid] = {"process": proc, "bank": bank, "kind": kind, "log": str(log_path)}

    try:
        report()
        while not all(bank["merged"] for bank in banks):
            for pid, task in list(active.items()):
                code = task["process"].poll()
                if code is None:
                    continue
                del active[pid]
                if code:
                    raise RuntimeError(f"{task['bank']['name']} {task['kind']} failed ({code}): {task['log']}")
                if task["kind"] == "merge":
                    task["bank"]["merged"] = True
                else:
                    task["bank"]["done"] += 1
            for bank in banks:
                if bank["pending"] is not None:
                    continue
                if bank.get("wait_for"):
                    dependency = json.loads(Path(bank["wait_for"]).read_text())
                    if dependency["state"] != "completed":
                        if dependency["state"] not in {"running", "queued"}:
                            raise RuntimeError(f"source dependency failed: {bank['wait_for']}")
                        continue
                contract = prepare_bank(bank["run"], bank["manifest_dir"], compute_hashes=compute_hashes)
                bank["records"] = contract["records"]
                bank["shards"] = max(1, math.ceil(bank["records"] / int(plan.get("images_per_shard", 512))))
                bank["pending"] = deque(range(bank["shards"]))
                existing = Path(bank["cache_dir"]) / "posterior_index.json"
                if existing.exists():
                    value = json.loads(existing.read_text())
                    meta = value["metadata"]
                    if ((compute_hashes and (meta.get("source_manifest_sha256") != contract["manifest_sha256"]
                                             or not meta.get("source_view_hashes_verified")))
                            or (not compute_hashes and Path(meta["manifest_jsonl"]).resolve()
                                != Path(contract["manifest_jsonl"]).resolve())
                            or not all(Path(p).exists() for p in value["shards"])):
                        raise ValueError(f"existing bank index contract changed: {existing}")
                    check_file_size(contract["manifest_jsonl"], meta.get("manifest_bytes"))
                    bank.update(merged=True, done=bank["shards"], pending=deque())
            for bank in banks:
                if bank["merged"] or bank["pending"] is None:
                    continue
                manifest = str(Path(bank["manifest_dir"]) / "manifest.jsonl")
                while bank["pending"] and len(active) < workers:
                    shard = bank["pending"].popleft()
                    launch(bank, [sys.executable, "-u", "scripts/imagenet_encode_kl16_vae.py",
                           "--source_mode", "manifest_jsonl", "--source_manifest_jsonl", manifest,
                           "--cache_shard_dir", bank["cache_dir"], "--image_size", "512",
                           "--frozen_views", "--verify_view_hashes" if compute_hashes else "--no_hash", "--batch_size", "4",
                           "--num_workers", "1", "--device", "cpu", "--vae_dtype", "fp32",
                           "--num_shards", str(bank["shards"]), "--shard_index", str(shard)],
                           f"shard-{shard:05d}", "encode")
                own_tasks = [task for task in active.values() if task["bank"] is bank]
                if not bank["pending"] and not own_tasks and len(active) < workers:
                    launch(bank, [sys.executable, "pretrain/merge_flow_latent_shards.py",
                           "--shard_dir", bank["cache_dir"], "--manifest_jsonl", manifest,
                           "--index_only", "--output_path", str(Path(bank["cache_dir"]) / "posterior_index.json")]
                           + ([] if compute_hashes else ["--no_hash"]),
                           "merge", "merge")
            report()
            if active or not all(bank["merged"] for bank in banks):
                time.sleep(1)
        report("completed")
    except BaseException as exc:
        for task in active.values():
            try:
                os.killpg(task["process"].pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        for task in active.values():
            try:
                task["process"].wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(task["process"].pid, signal.SIGKILL)
                task["process"].wait()
        active.clear()
        report("interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", str(exc))
        raise
    finally:
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    args = parser.parse_args()
    def stop(_signum, _frame):
        raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM, stop)
    run_plan(args.plan)


if __name__ == "__main__":
    main()
