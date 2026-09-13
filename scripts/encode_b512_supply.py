"""Encode immutable supplemental 512px batches independently of text synthesis."""

import argparse
from contextlib import redirect_stderr, redirect_stdout
import fcntl
import json
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import time

from data_synthesis.io import atomic_json, cohort_id, dumps, file_sha


def freeze_bank(source, output):
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    marker = output / "bank.json"
    if marker.exists():
        value = json.loads(marker.read_text())
        if file_sha(output / "manifest.jsonl") != value["manifest_sha256"]:
            raise ValueError("supplemental VAE manifest changed")
        return value
    prepared = json.loads((source / "batch.json").read_text())
    if file_sha(source / "state.sqlite3") != prepared["state_sha256"]:
        raise ValueError("prepared batch changed before VAE encoding")
    db = sqlite3.connect(f"file:{source / 'state.sqlite3'}?mode=ro", uri=True)
    count = 0
    temporary = output / "manifest.jsonl.tmp"
    try:
        with temporary.open("w") as handle:
            for key, row, view in db.execute("SELECT key,row,view FROM tasks WHERE status='prepared' ORDER BY key"):
                row, view = json.loads(row), json.loads(view)
                count += 1
                handle.write(dumps({**view, "img_id": count, "key": key, "source": row["source"],
                                    "source_id": row["source_id"], "split": "train"}) + "\n")
    finally:
        db.close()
    if count != prepared["records"]:
        raise ValueError("prepared/VAE manifest counts differ")
    temporary.replace(output / "manifest.jsonl")
    value = {"schema": "b512_frozen_image_bank_v1", "source_run": str(source), "records": count,
             "manifest_sha256": file_sha(output / "manifest.jsonl"), "manifest_jsonl": str(output / "manifest.jsonl"),
             "source_run_contract_sha256": file_sha(source / "run.json")}
    atomic_json(marker, value)
    return value


def finalize_cached_bank(bank, destination, work, vae_hashes):
    """Verify an NPU-completed bank and index it without two Python startups."""
    shard = destination / "shards/shard-00000-of-00001.pt"
    if not shard.exists():
        return False
    from scripts.imagenet_encode_kl16_vae import load_samples_from_manifest, validate_reusable_shard
    from pretrain.merge_flow_latent_shards import main as merge_index

    manifest = Path(bank["manifest_jsonl"])
    if file_sha(manifest) != bank["manifest_sha256"]:
        raise ValueError("cached bank manifest changed")
    samples = load_samples_from_manifest(manifest, -1, None)
    if len(samples) != bank["records"]:
        raise ValueError("cached bank sample count changed")
    validate_reusable_shard(shard, samples, num_shards=1, shard_index=0,
        scaling_factor=0.2325, vae_checkpoint_sha256=vae_hashes[0], vae_module_sha256=vae_hashes[1],
        source_manifest_sha256=bank["manifest_sha256"], source_image_root=None, image_size=512,
        frozen_views=True, verify_view_hashes=True)
    if not (destination / "posterior_index.json").exists():
        with (work / "index.log").open("a") as log, redirect_stdout(log), redirect_stderr(log):
            merge_index(["--shard_dir", str(destination / "shards"), "--output_path", str(destination / "posterior_index.json"),
                         "--manifest_jsonl", str(manifest), "--index_only",
                         "--row_index_path", str(destination / "posterior.rows.pt")])
    return True


def run(args):
    source, images = Path(args.supply_root).resolve(), Path(args.image_root).resolve()
    root = source / "posterior_encoding"
    root.mkdir(exist_ok=True)
    lock = (root / "controller.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = {"controller_pid": os.getpid(), "state": "running", "started_at": time.time(),
             "completed_batches": 0, "encoded_records": 0, "cpu_threads": args.cpu_threads}
    child = None
    env = os.environ.copy()
    env.update(TORCH_DEVICE_BACKEND_AUTOLOAD="0", OMP_NUM_THREADS=str(args.cpu_threads),
               MKL_NUM_THREADS=str(args.cpu_threads), OPENBLAS_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false")

    def stop(_signum, _frame):
        raise KeyboardInterrupt()

    signal.signal(signal.SIGTERM, stop)
    try:
        import torch
        torch.set_num_threads(args.cpu_threads)
        vae_hashes = (file_sha("public/vae/mar-kl16/kl16.ckpt"), file_sha("public/code/mar/models/vae.py"))
        while True:
            worked = False
            completed, records = 0, 0
            markers = sorted((source / "prepared_batches").glob("*/batch.json"))
            for marker in markers:
                batch_id = marker.parent.name
                work = source / "posterior_banks" / batch_id
                work.mkdir(parents=True, exist_ok=True)
                done = work / "complete.json"
                if done.exists():
                    value = json.loads(done.read_text())
                    completed += 1
                    records += value["records"]
                    continue
                worked = True
                bank = freeze_bank(marker.parent, work)
                if bank["records"]:
                    destination = images / "vae_supply" / cohort_id(source) / batch_id
                    destination.mkdir(parents=True, exist_ok=True)
                    commands = [
                        [sys.executable, "scripts/imagenet_encode_kl16_vae.py", "--source_mode", "manifest_jsonl",
                         "--source_manifest_jsonl", bank["manifest_jsonl"], "--cache_shard_dir", str(destination / "shards"),
                         "--image_size", "512", "--frozen_views", "--verify_view_hashes", "--device", "cpu",
                         "--vae_dtype", "fp32", "--batch_size", "4", "--num_workers", "1"],
                        [sys.executable, "pretrain/merge_flow_latent_shards.py", "--shard_dir", str(destination / "shards"),
                         "--output_path", str(destination / "posterior_index.json"), "--manifest_jsonl", bank["manifest_jsonl"],
                         "--index_only", "--row_index_path", str(destination / "posterior.rows.pt")],
                    ]
                    if finalize_cached_bank(bank, destination, work, vae_hashes):
                        commands = []
                    for stage, argv in zip(("encode", "index"), commands):
                        if stage == "index" and (destination / "posterior_index.json").exists():
                            continue
                        with (work / (stage + ".log")).open("a") as log:
                            child = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=log,
                                stderr=subprocess.STDOUT, env=env, start_new_session=True)
                        state.update(stage=stage, batch_id=batch_id, child_pid=child.pid, updated_at=time.time())
                        atomic_json(root / "job_status.json", state)
                        code = child.wait()
                        child = None
                        if code:
                            raise RuntimeError(f"{stage} {batch_id} exited with {code}")
                atomic_json(done, {"batch_id": batch_id, "records": bank["records"], "completed_at": time.time()})
                completed += 1
                records += bank["records"]
                state.update(completed_batches=completed, encoded_records=records, updated_at=time.time())
                state.pop("child_pid", None)
                atomic_json(root / "job_status.json", state)
            if (source.parent / "input_queue" / "closed.json").exists() and not worked:
                expected = json.loads((source.parent / "input_queue" / "closed.json").read_text())["batches"]
                if completed == expected:
                    state.update(state="completed", completed_at=time.time())
                    atomic_json(root / "job_status.json", state)
                    return
            if not worked:
                state.update(stage="waiting_prepared_batches", updated_at=time.time())
                atomic_json(root / "job_status.json", state)
                time.sleep(5)
    except BaseException as exc:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        state.update(state="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error=str(exc))
        atomic_json(root / "job_status.json", state)
        raise
    finally:
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--supply-root", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--cpu-threads", type=int, default=2)
    args = parser.parse_args()
    if args.cpu_threads < 1:
        parser.error("cpu-threads must be positive")
    run(args)


if __name__ == "__main__":
    main()
