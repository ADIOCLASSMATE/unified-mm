"""Run a finite Ascend pass over missing shards after CPU writers are paused."""

import argparse
from collections import deque
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from data_synthesis.io import atomic_json, file_sha
from data_synthesis.config import load_config
from data_synthesis.integrity import check_file_size, hashing_enabled
from scripts.encode_b512_supply import freeze_bank


def prepare(args):
    compute_hashes = hashing_enabled()
    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    destination = root / "plan.json"
    if destination.exists():
        raise FileExistsError("acceleration plans are immutable; choose a new wave directory")
    base = json.loads(Path(args.base_plan).read_text())
    banks = []
    for item in base["banks"]:
        work = Path(item["manifest_dir"])
        bank = json.loads((work / "bank.json").read_text())
        manifest = work / "manifest.jsonl"
        if compute_hashes and file_sha(manifest) != bank["manifest_sha256"]:
            raise ValueError(f"frozen manifest changed: {manifest}")
        banks.append({"name": item["name"], "manifest": str(manifest),
                      "manifest_sha256": bank["manifest_sha256"] if compute_hashes else None,
                      "manifest_bytes": check_file_size(manifest, bank.get("manifest_bytes")), "records": bank["records"],
                      "shards": max(1, math.ceil(bank["records"] / base.get("images_per_shard", 512))),
                      "cache_dir": str(Path(item["cache_dir"]).resolve())})
    supply, cache = Path(args.supply_root).resolve(), Path(args.posterior_root).resolve()
    for marker in sorted((supply / "prepared_batches").glob("*/batch.json")):
        # A separate manifest directory avoids touching the CPU controller's
        # files. The canonical row order and manifest bytes remain identical.
        bank = freeze_bank(marker.parent, root / "banks" / marker.parent.name, compute_hashes=compute_hashes)
        if bank["records"]:
            banks.append({"name": marker.parent.name, "manifest": bank["manifest_jsonl"],
                          "manifest_sha256": bank["manifest_sha256"], "manifest_bytes": bank["manifest_bytes"], "records": bank["records"],
                          "shards": 1, "cache_dir": str(cache / "vae_supply" / marker.parent.name / "shards")})
    if getattr(args, "validation_contract", None):
        validation = json.loads(Path(args.validation_contract).read_text())
        manifest = Path(validation["manifest"])
        if validation["image_size"] != 512 or (compute_hashes and file_sha(manifest) != validation["manifest_sha256"]):
            raise ValueError("validation image contract changed")
        banks.append({"name": "validation_imagenet512", "manifest": str(manifest),
                      "manifest_sha256": validation["manifest_sha256"] if compute_hashes else None,
                      "manifest_bytes": check_file_size(manifest, validation.get("manifest_bytes")), "records": validation["records"],
                      "shards": math.ceil(validation["records"] / 512),
                      "cache_dir": str(cache / "validation" / validation["manifest_sha256"] / "shards"),
                      "frozen_views": False, "verify_view_hashes": False})
    atomic_json(destination, {"version": "b512_finite_npu_pass_v1", "created_at": time.time(),
        "cwd": str(Path.cwd()), "banks": banks, "records": sum(b["records"] for b in banks),
        "compute_hashes": compute_hashes,
        "encoder_sha256": file_sha("scripts/imagenet_encode_kl16_vae.py") if compute_hashes else None,
        "policy": "fp32; reuse completed shards; require a recorded handoff from paused CPU controllers"})
    print(json.dumps({"plan": str(destination), "banks": len(banks),
                      "records": sum(b["records"] for b in banks)}))


def run(args):
    root = Path(args.root).resolve()
    lock = (root / "controller.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = json.loads((root / "plan.json").read_text())
    compute_hashes = hashing_enabled() and hashing_enabled(plan)
    handoff_path = Path(args.cpu_handoff).resolve()
    handoff = json.loads(handoff_path.read_text())
    if (handoff.get("state") != "cpu_writers_paused" or not handoff.get("controllers")
            or any(p.get("confirmed_process_state") != "T"
                   for p in handoff["controllers"] + handoff["writers"])):
        raise ValueError("pause and record CPU controllers and writers before starting an NPU pass")
    handoff_sha256 = file_sha(handoff_path) if compute_hashes else None
    os.chdir(plan["cwd"])
    if compute_hashes and file_sha("scripts/imagenet_encode_kl16_vae.py") != plan["encoder_sha256"]:
        raise ValueError("encoder changed after the acceleration plan was frozen")
    import torch
    import torch_npu  # noqa: F401
    if not torch.npu.is_available() or torch.npu.device_count() < args.workers:
        raise RuntimeError(f"this pass requires {args.workers} visible Ascend devices")
    queue, records = deque(), []
    for bank in plan["banks"]:
        if compute_hashes and file_sha(bank["manifest"]) != bank["manifest_sha256"]:
            raise ValueError(f"input manifest changed: {bank['manifest']}")
        check_file_size(bank["manifest"], bank.get("manifest_bytes"))
        cache = Path(bank["cache_dir"])
        cache.mkdir(parents=True, exist_ok=True)
        for shard in range(bank["shards"]):
            path = cache / f"shard-{shard:05d}-of-{bank['shards']:05d}.pt"
            unit = {"bank": bank["name"], "shard": shard, "path": str(path)}
            if path.exists():
                records.append({**unit, "state": "already_present"})
            else:
                queue.append((bank, unit))
    active = {}
    free = deque(range(args.workers))
    env = {k: v for k, v in os.environ.items() if k != "SII_API_KEY"}
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(Path.cwd()), env.get("PYTHONPATH"))))
    env.update(OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1",
               TOKENIZERS_PARALLELISM="false")
    status = {"controller_pid": os.getpid(), "state": "running", "started_at": time.time(),
              "plan_sha256": file_sha(root / "plan.json") if compute_hashes else None, "workers": args.workers,
              "compute_hashes": compute_hashes,
              "cpu_handoff": str(handoff_path), "cpu_handoff_sha256": handoff_sha256}

    def report(state="running"):
        status.update(state=state, active_children=list(active), pending=len(queue),
                      finished_units=len(records), updated_at=time.time())
        atomic_json(root / "job_status.json", status)

    def stop(_signum, _frame):
        raise KeyboardInterrupt()

    signal.signal(signal.SIGTERM, stop)
    try:
        while queue or active:
            if ((compute_hashes and file_sha(handoff_path) != handoff_sha256)
                    or (not compute_hashes and json.loads(handoff_path.read_text()) != handoff)):
                raise ValueError("CPU handoff changed while the NPU pass was running")
            while queue and free:
                bank, unit = queue.popleft()
                device = free.popleft()
                argv = [sys.executable, "-u", "scripts/imagenet_encode_kl16_vae.py",
                    "--source_mode", "manifest_jsonl", "--source_manifest_jsonl", bank["manifest"],
                    "--cache_shard_dir", bank["cache_dir"], "--image_size", "512",
                    "--skip_locked", "--device", f"npu:{device}",
                    "--vae_dtype", "fp32", "--batch_size", str(args.batch_size), "--num_workers", "2",
                    "--prefetch_factor", "2", "--num_shards", str(bank["shards"]),
                    "--shard_index", str(unit["shard"])]
                if bank.get("frozen_views", True):
                    argv.append("--frozen_views")
                if compute_hashes and bank.get("verify_view_hashes", True):
                    argv.append("--verify_view_hashes")
                if not compute_hashes:
                    argv.append("--no_hash")
                log = root / "logs" / f"{unit['bank']}-{unit['shard']:05d}.log"
                log.parent.mkdir(exist_ok=True)
                with log.open("a") as handle:
                    process = subprocess.Popen(argv, env=env, stdin=subprocess.DEVNULL,
                        stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
                active[process.pid] = (process, device, unit, argv, log, time.time())
            for pid, (process, device, unit, argv, log, started) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                del active[pid]
                free.append(device)
                if code:
                    raise RuntimeError(f"encoder exited {code}: {log}")
                present = Path(unit["path"]).exists()
                if not present and "Skipped busy posterior shard:" not in log.read_text():
                    raise RuntimeError(f"successful encoder produced no shard: {log}")
                records.append({**unit, "state": "available" if present else "deferred_external_writer",
                    "command": argv, "log": str(log), "log_sha256": file_sha(log) if compute_hashes else None,
                    "seconds": time.time() - started})
                atomic_json(root / "results.json", {"units": records})
            report()
            if active:
                time.sleep(2)
        atomic_json(root / "results.json", {"units": records})
        status["note"] = "Finite helper pass only. Original bank controllers verify, merge and publish all shards."
        report("completed")
    except BaseException as exc:
        status["error"] = str(exc)
        for process, *_ in active.values():
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        for process, *_ in active.values():
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        report("interrupted" if isinstance(exc, KeyboardInterrupt) else "failed")
        raise
    finally:
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("plan")
    prepare_parser.add_argument("--root", required=True)
    prepare_parser.add_argument("--base-plan", required=True)
    prepare_parser.add_argument("--supply-root", required=True)
    prepare_parser.add_argument("--posterior-root", default=load_config()["posterior_root"])
    prepare_parser.add_argument("--validation-contract", help="Also fill the separate, frozen validation cache")
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--root", required=True)
    run_parser.add_argument("--cpu-handoff", required=True)
    run_parser.add_argument("--workers", type=int, default=16)
    run_parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.command == "run" and (args.workers < 1 or args.batch_size < 1):
        parser.error("workers and batch size must be positive")
    (prepare if args.command == "plan" else run)(args)


if __name__ == "__main__":
    main()
