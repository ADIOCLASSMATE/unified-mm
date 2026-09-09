#!/usr/bin/env python3
"""Paired ImageNet-val 50K CFG sweep, followed by Heun steps at both optima.

The CPU scheduler is shared by independent 16-NPU Job instances. Generation
uses a frozen, already validated evaluator; no model or default is modified.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import fcntl
import glob
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def require(condition, message):
    if not condition:
        raise ValueError(message)


@contextmanager
def locked(root):
    with (root / "scheduler.lock").open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def task(cfg, steps, phase):
    return {"id": f"cfg-{cfg:.1f}-heun-{steps:03d}", "cfg": cfg,
            "steps": steps, "phase": phase, "status": "pending"}


def arm_protocol(protocol, arm):
    if "model" not in arm:
        return protocol
    return {**protocol, **protocol["models"][arm["model"]]}


def validate_metrics(path, protocol, arm):
    protocol = arm_protocol(protocol, arm)
    data = read(path)
    expected = {"schema": "selfless_imagenet_val_t2i_fid_is_v2",
                "project_formal_protocol": True, "runtime_hashing_enabled": False,
                "samples_requested": 50000, "samples_evaluated": 50000,
                "split": "val", "seed": 42, "batch_size": 4096,
                "cfg": arm["cfg"], "flow_solver": "heun", "cfg_schedule": "constant",
                "temperature": 1.0, "parallel_rate": 1, "backbone_kv_cache": True}
    for key, value in expected.items():
        require(data.get(key) == value, f"{path}: {key} != {value!r}")
    require(str(data.get("sampling_steps")) == str(arm["steps"]), f"{path}: steps mismatch")
    source = data["evaluation_model_source"]
    require(source["kind"] == "hf_final_ema" and source["global_step"] == protocol["checkpoint_step"]
            and Path(source["path"]).resolve() == Path(protocol["model_source"]),
            f"{path}: checkpoint mismatch")
    require(Path(data["config"]).resolve() == Path(protocol["config"]), f"{path}: config mismatch")
    require(Path(data["real_stats_path"]).resolve() == Path(protocol["real_stats"]),
            f"{path}: real-reference mismatch")
    require(data["distributed"]["world_size"] == 16, f"{path}: rank count mismatch")
    precision = data["precision_protocol"]
    require(precision["model_dtype"] == "bf16" and precision["vae_dtype"] == "fp32"
            and precision["flow_integrator_dtype"] == "fp32", f"{path}: precision mismatch")
    contracts = data["implementation_contracts"]
    require(contracts["canonical_initial_noise_enabled"] is True
            and contracts["paired_sample_count"] == 50000
            and contracts["ordered_sample_count"] == 50000, f"{path}: pairing incomplete")
    require(contracts["backbone_attention"]["dual_stream_attention_contract"] == protocol.get("backbone_attention", "xlnet_content_diagonal")
            and contracts["flow_head_attention"]["flow_head_attention_contract"] == protocol.get("flow_attention", "xlnet_content_diagonal"),
            f"{path}: checkpoint attention mismatch")
    if protocol.get("generation_contract"):
        require(contracts.get("checkpoint_generation") == protocol["generation_contract"],
                f"{path}: checkpoint generation contract mismatch")
    metric = data["metric_protocol"]
    require(metric["protocol_name"] == "imagenet_val_fid50k_torch_fidelity_stratified_is"
            and metric["reference_distribution"] == "imagenet_val_50000"
            and metric["is_splits"] == 10, f"{path}: metric protocol mismatch")
    splits = metric["is_split_plan"]
    require(splits["assignment"] == "stratified_by_synset" and splits["source_dataset_split"] == "val"
            and splits["classes_per_split"] == [1000] * 10
            and splits["samples_per_split"] == [5000] * 10
            and splits["samples_per_class_per_split_min"] == 5
            and splits["samples_per_class_per_split_max"] == 5, f"{path}: IS split mismatch")
    require(data["mechanism_diagnostics"]["generated_latent_finite_rate"] == 1.0,
            f"{path}: nonfinite generation")
    strategy = arm.get("strategy", "spatial_halton")
    require(set(data["strategies"]) == {strategy}, f"{path}: strategy mismatch")
    result = data["strategies"][strategy]
    if protocol.get("order_policies"):
        require(data.get("order_strategy_protocols", {}).get(strategy) == protocol["order_policies"][strategy],
                f"{path}: order-policy mismatch")
    if protocol.get("save_image_count"):
        expected_indices = protocol["saved_image_indices"]
        require(data.get("saved_image_subset", {}).get("global_sample_indices") == expected_indices,
                f"{path}: saved image selection mismatch")
        image_dir = Path(path).parent / strategy
        require(sorted(int(p.stem) for p in image_dir.glob("*.png")) == expected_indices,
                f"{path}: saved images incomplete")
        for index in expected_indices:
            record = read(image_dir / f"{index:08d}.json")
            require(record["global_sample_index"] == index and record["canonical_noise_seed"] == 42 + index
                    and bool(record["prompt"]), f"{path}: saved sample identity mismatch")
    require(result["count"] == 50000, f"{path}: incomplete generation")
    for key in ("fid", "inception_score_mean", "inception_score_std"):
        require(math.isfinite(result[key]) and result[key] >= 0, f"{path}: invalid {key}")
    return {"fid": result["fid"], "is": result["inception_score_mean"],
            "is_std": result["inception_score_std"],
            "generation_seconds": result["generation_wall_seconds"],
            "metrics_path": str(Path(path).resolve())}


def winners(arms):
    require(bool(arms) and all(a["status"] == "done" for a in arms), "selection requires complete arms")
    return {"best_fid": min(arms, key=lambda a: (a["result"]["fid"], -a["result"]["is"], a["cfg"], a["steps"])),
            "best_is": min(arms, key=lambda a: (-a["result"]["is"], a["result"]["fid"], a["cfg"], a["steps"]))}


def advance(state, protocol):
    """Called under the scheduler lock; stage two cannot start early."""
    if state["status"] in {"complete", "failed"}:
        return
    if any(a["status"] != "done" for a in state["tasks"]):
        return
    if state["phase"] == "matrix":
        state.update(status="complete", completed_at=now())
        return
    if state["phase"] == "order":
        state["order_selection"] = {name: dict(arm) for name, arm in winners(state["tasks"]).items()}
        state.update(status="complete", completed_at=now())
        return
    if state["phase"] == "cfg":
        selected = winners(state["tasks"])
        low = min(a["cfg"] for a in state["tasks"])
        high = max(a["cfg"] for a in state["tasks"])
        extend = []
        if any(a["cfg"] == low for a in selected.values()) and low > 0:
            extend.append(low - 0.5)
        if any(a["cfg"] == high for a in selected.values()):
            if high >= protocol["cfg_limit"]:
                state.update(status="failed", error="CFG optimum is still on the upper guard boundary; extend cfg_limit before claiming an optimum.")
                return
            extend.append(high + 0.5)
        if extend:
            state["tasks"].extend(task(c, protocol["cfg_steps"], "cfg") for c in extend)
            state.setdefault("boundary_extensions", []).append({"at": now(), "cfg": extend})
            return
        state["cfg_selection"] = {name: dict(arm) for name, arm in selected.items()}
        state["phase"] = "heun"
        cfgs = sorted({arm["cfg"] for arm in selected.values()})
        existing = {arm["id"] for arm in state["tasks"]}
        # Dispatch the long arms first to reduce idle time at the end.
        for steps in sorted(protocol["heun_steps"], reverse=True):
            for cfg in cfgs:
                arm = task(cfg, steps, "heun")
                if arm["id"] not in existing:
                    state["tasks"].append(arm)
        return
    state["heun_selection"] = {}
    for name, selected in state["cfg_selection"].items():
        arms = [a for a in state["tasks"] if a["cfg"] == selected["cfg"]
                and a["steps"] in protocol["heun_steps"]]
        require({a["steps"] for a in arms} == set(protocol["heun_steps"]), "missing requested Heun steps")
        state["heun_selection"][name] = winners(arms)
    state.update(status="complete", completed_at=now())


def report(root, state, protocol):
    if state["phase"] == "matrix":
        rows = [{"model": a["model"], "strategy": a["strategy"], "cfg": a["cfg"],
                 "heun_steps": a["steps"], "status": a["status"], **a.get("result", {})}
                for a in state["tasks"]]
        write(root / "summary.json", {"status": state["status"], "phase": "matrix", "updated_at": now(),
              "protocol": protocol, "results": rows})
        with (root / "results.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["model", "strategy", "cfg", "heun_steps", "status",
                "fid", "is", "is_std", "generation_seconds", "metrics_path"])
            writer.writeheader()
            writer.writerows(rows)
        lines = ["# Ablation matrix: CFG 2.0 / Heun 10", "", f"Status: **{state['status']}**.", "",
            "Each arm uses its own final EMA and native architecture, paired ImageNet-val 50K, seed 42, 16 NPUs.",
            "All models compare " + ", ".join(protocol.get("matrix_strategies", ["spatial_halton", "confidence_stability"])) + "; E also retains a sequential control.", "",
            "| Model | Strategy | FID ↓ | IS ↑ | Generation minutes | Status |", "|---|---|---:|---:|---:|---|"]
        for row in rows:
            values = f"{row['fid']:.4f} | {row['is']:.3f} ± {row['is_std']:.3f} | {row['generation_seconds']/60:.1f}" if "fid" in row else "— | — | —"
            lines.append(f"| {row['model']} | {row['strategy']} | {values} | {row['status']} |")
        (root / "README.md").write_text("\n".join(lines) + "\n")
        return
    if state["phase"] == "order":
        rows = [{"strategy": a["strategy"], "cfg": a["cfg"], "heun_steps": a["steps"],
                 "status": a["status"], **a.get("result", {})} for a in state["tasks"]]
        write(root / "summary.json", {"status": state["status"], "phase": "order", "updated_at": now(),
              "protocol": protocol, "results": rows, "order_selection": state.get("order_selection")})
        with (root / "results.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["strategy", "cfg", "heun_steps", "status", "fid", "is",
                                                       "is_std", "generation_seconds", "metrics_path"])
            writer.writeheader()
            writer.writerows(rows)
        lines = ["# B-X0 reveal-order sweep", "", f"Status: **{state['status']}**.", "",
                 "Fixed CFG=2.0, Heun=10; final EMA step 95415; paired ImageNet-val 50K, seed 42; 16 NPUs per arm.",
                 "Selection uses minimum validation FID. Confidence scores are heuristic proxies, not calibrated probabilities.",
                 "The 16-position Halton candidate blocks are refreshed from generated context; proposals are discarded.", "",
                 "| Strategy | FID ↓ | IS ↑ | Generation minutes | Status |", "|---|---:|---:|---:|---|"]
        for row in rows:
            values = f"{row['fid']:.4f} | {row['is']:.3f} ± {row['is_std']:.3f} | {row['generation_seconds']/60:.1f}" if "fid" in row else "— | — | —"
            lines.append(f"| {row['strategy']} | {values} | {row['status']} |")
        if state["status"] == "complete":
            best = state["order_selection"]["best_fid"]
            lines.extend(["", f"Lowest measured FID: **{best['strategy']}**, FID {best['result']['fid']:.4f}."])
        (root / "README.md").write_text("\n".join(lines) + "\n")
        return
    rows = [{"phase": a["phase"], "cfg": a["cfg"], "heun_steps": a["steps"],
             "status": a["status"], **a.get("result", {})} for a in state["tasks"]]
    write(root / "summary.json", {"status": state["status"], "phase": state["phase"],
          "updated_at": now(), "protocol": protocol, "results": rows,
          "cfg_selection": state.get("cfg_selection"), "heun_selection": state.get("heun_selection"),
          "error": state.get("error")})
    with (root / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["phase", "cfg", "heun_steps", "status", "fid", "is",
                                                   "is_std", "generation_seconds", "metrics_path"])
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# B-X0 CFG / Heun sweep", "", f"Status: **{state['status']}**; phase: **{state['phase']}**.", "",
             "Final EMA, ImageNet-val 50K, seed 42, paired initial noise, BF16 model / FP32 VAE.",
             "IS uses ten class-stratified splits. FID is comparable within this val-reference protocol.",
             "This is validation-set hyperparameter selection; the selected scores are not an independent holdout estimate.", "",
             "CFG search uses 0.5 increments at 10 Heun steps and extends any winning boundary (lower bound 0).",
             "Lowest FID and highest IS are selected separately; both selected CFGs receive 5/10/20/50/100 steps.", "",
             "| Phase | CFG | Heun steps | Status | FID ↓ | IS ↑ |", "|---|---:|---:|---|---:|---:|"]
    for row in sorted(rows, key=lambda r: (r["phase"], r["cfg"], r["heun_steps"])):
        metric_text = f"{row['fid']:.4f} | {row['is']:.3f} ± {row['is_std']:.3f}" if "fid" in row else "— | —"
        lines.append(f"| {row['phase']} | {row['cfg']:.1f} | {row['heun_steps']} | {row['status']} | {metric_text} |")
    if state.get("cfg_selection"):
        lines.extend(["", "CFG selection:", ""])
        for name, arm in state["cfg_selection"].items():
            lines.append(f"- {name}: CFG {arm['cfg']:.1f}; FID {arm['result']['fid']:.4f}, IS {arm['result']['is']:.3f}.")
    if state["status"] == "complete":
        lines.extend(["", "Best measured combinations after the Heun sweep:", ""])
        for name, arm in winners(state["tasks"]).items():
            result = arm["result"]
            lines.append(f"- {name}: CFG {arm['cfg']:.1f}, Heun {arm['steps']}; "
                         f"FID {result['fid']:.4f}, IS {result['is']:.3f} ± {result['is_std']:.3f}; "
                         f"50K generation {result['generation_seconds'] / 60:.1f} minutes on 16 NPUs.")
    if state.get("error"):
        lines.extend(["", f"Error: {state['error']}"])
    (root / "README.md").write_text("\n".join(lines) + "\n")


def save(root, state, protocol):
    state["updated_at"] = now()
    write(root / "state.json", state)
    report(root, state, protocol)


def evaluator_command(protocol, arm, output):
    protocol = arm_protocol(protocol, arm)
    command = [protocol["python"], "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=16",
            "scripts/evaluate_single_stream_fid_is.py", "--config", protocol["config"],
            "--model_source", protocol["model_source"], "--output_dir", str(output),
            "--device", "npu", "--model_dtype", "bf16", "--seed", "42", "--samples", "50000",
            "--batch_size", "4096", "--caption_sequence_mode", "t2i", "--sampling_steps", str(arm["steps"]),
            "--temperature", "1.0", "--cfg", str(arm["cfg"]), "--cfg_schedule", "constant",
            "--flow_solver", "heun", "--parallel_rate", "1", "--strategies", arm.get("strategy", "spatial_halton"),
            "--vae_dtype", "fp32", "--vae_decode_batch_size", "16", "--fid_feature", "2048",
            "--is_splits", "10", "--inception_weights_path", protocol["inception_weights"],
            "--real_stats_path", protocol["real_stats"], "--canonical_pairing", "--require_formal_protocol",
            "--resume_progress", "--resume_checkpoint_interval_batches", "1"]
    if protocol.get("save_image_count"):
        command.extend(["--save_image_count", str(protocol["save_image_count"])])
    return command


def stop_process(process):
    if process is None or process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def worker(root, first_cfg=None, task_id=None):
    protocol = read(root / "protocol.json")
    worker_id = f"{socket.gethostname()}-{os.getpid()}"
    # A platform instance runs one scheduler and one local 16-rank evaluator.
    env = os.environ.copy()
    for key in ("WORLD_SIZE", "RANK", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "GROUP_RANK", "GROUP_WORLD_SIZE",
                "ROLE_RANK", "ROLE_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "CUDA_VISIBLE_DEVICES"):
        env.pop(key, None)
    process = None
    arm = None
    def interrupted(signum, frame):
        raise RuntimeError(f"worker received signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        while True:
            with locked(root):
                state = read(root / "state.json")
                advance(state, protocol)
                if state["status"] == "failed":
                    raise RuntimeError(state.get("error", "another worker failed"))
                if state["status"] == "complete":
                    save(root, state, protocol)
                    print("SWEEP_COMPLETE", flush=True)
                    return
                pending = [a for a in state["tasks"] if a["status"] == "pending"]
                if task_id is not None:
                    requested = next(a for a in state["tasks"] if a["id"] == task_id)
                    if requested["status"] == "done":
                        print(f"ALREADY_COMPLETE {task_id}", flush=True)
                        return
                    require(requested["status"] == "pending", f"task already owned: {task_id}")
                    pending = [requested]
                preferred = [a for a in pending if a["cfg"] == first_cfg] if first_cfg is not None else []
                arm = next(iter(preferred or pending), None)
                if arm:
                    arm.update(status="running", worker=worker_id, started_at=now())
                    save(root, state, protocol)
                elif state["phase"] == "heun":
                    # All remaining arms are owned. Release this independent
                    # single-node Job instead of idling through 100-step runs.
                    print("WORKER_FINISHED: remaining Heun arms have active workers", flush=True)
                    return
            if arm is None:
                time.sleep(10)
                continue
            output = root / "arms" / arm["id"]
            output.mkdir(parents=True, exist_ok=True)
            driver_libraries = []
            for pattern in ("/usr/local/Ascend/driver/**/libascend_hal.so*",
                            "/usr/local/Ascend/driver*/lib64/**/libascend_hal.so*",
                            "/usr/lib64/libascend_hal.so*", "/usr/lib/*-linux-gnu/libascend_hal.so*"):
                driver_libraries.extend(path for path in glob.glob(pattern, recursive=True)
                                        if Path(path).is_file() and "stub" not in path.lower())
            driver_libraries = sorted(set(driver_libraries))
            library_dirs = sorted({str(Path(path).parent) for path in driver_libraries})
            if library_dirs:
                env["LD_LIBRARY_PATH"] = ":".join([*library_dirs, env.get("LD_LIBRARY_PATH", "")])
            write(output / "environment.json", {
                "worker": worker_id, "machine": os.uname().machine, "driver_libraries": driver_libraries,
                "driver_directory_exists": Path("/usr/local/Ascend/driver").exists(),
                "devices": sorted(glob.glob("/dev/davinci*")),
                "LD_LIBRARY_PATH": env.get("LD_LIBRARY_PATH"),
                "ASCEND_HOME_PATH": env.get("ASCEND_HOME_PATH"),
            })
            command = evaluator_command(protocol, arm, output)
            write(output / "command.json", {"argv": command, "cwd": protocol["source_repo"], "worker": worker_id})
            print(f"START {arm['id']} worker={worker_id}", flush=True)
            start = time.monotonic()
            with (output / "run.log").open("a") as logfile:
                process = subprocess.Popen(command, cwd=protocol["source_repo"], env=env,
                                           stdout=logfile, stderr=subprocess.STDOUT, start_new_session=True)
                while process.poll() is None:
                    time.sleep(5)
                    if read(root / "state.json")["status"] == "failed":
                        raise RuntimeError("another worker failed")
                    if time.monotonic() - start > protocol["arm_timeout_hours"] * 3600:
                        raise TimeoutError(f"{arm['id']} exceeded arm timeout")
                require(process.returncode == 0, f"{arm['id']}: evaluator exited {process.returncode}; see {output / 'run.log'}")
            result = validate_metrics(output / "metrics.json", protocol, arm)
            with locked(root):
                state = read(root / "state.json")
                current = next(a for a in state["tasks"] if a["id"] == arm["id"])
                current.update(status="done", completed_at=now(), result=result,
                               elapsed_seconds=time.monotonic() - start)
                advance(state, protocol)
                save(root, state, protocol)
            print(f"DONE {arm['id']} FID={result['fid']:.6f} IS={result['is']:.6f}", flush=True)
            if task_id is not None:
                return
            process = None
            arm = None
            first_cfg = None
    except BaseException as error:
        stop_process(process)
        with locked(root):
            state = read(root / "state.json")
            if arm:
                current = next(a for a in state["tasks"] if a["id"] == arm["id"])
                if current["status"] != "done":
                    current.update(status="failed", error=str(error))
            state.setdefault("worker_errors", []).append({"at": now(), "worker": worker_id,
                                                           "task_id": arm["id"] if arm else task_id,
                                                           "error": str(error)})
            save(root, state, protocol)
        raise


def prepare(args):
    root = args.output_dir.resolve()
    require(not root.exists(), f"refusing to overwrite existing sweep: {root}")
    require(0 <= args.cfg_min < args.cfg_max < args.cfg_limit, "invalid CFG bounds")
    require(all(v * 2 == int(v * 2) for v in (args.cfg_min, args.cfg_max, args.cfg_limit)), "CFG bounds must be multiples of 0.5")
    model = args.model_source.resolve()
    config = args.config.resolve()
    source = args.source_repo.resolve()
    repo = Path(__file__).resolve().parents[1]
    require(not root.is_relative_to(source), "new output must be outside the frozen source snapshot")
    require(1 <= args.save_image_count <= 50000, "save-image-count must be between 1 and 50000")
    platform = read(args.platform_json)
    require(all(platform.get(key) for key in ("workspace", "project", "compute_group", "quota", "image", "requested_priority")),
            "platform JSON is missing explicit scheduling fields")
    require(platform["quota"].split(",")[0] == "16", "each sweep Job must request 16 NPUs")
    saved = read(model / "config.json")
    require(saved["flow_condition_contract"] == "backbone_xt_query_backbone_x0_content"
            and saved["flow_head_attention_contract"] == "xlnet_content_diagonal",
            "expected B-X0 checkpoint contracts")
    protocol = {"schema": "unified_t2i_cfg_heun_sweep_v1", "created_at": now(),
                "model_source": str(model), "config": str(config),
                "checkpoint_step": read(model / "ema_export_metadata.json")["source_global_step"],
                "source_repo": str(root / "launch/source"), "source_provenance": str(source),
                "python": str(repo / ".venv/bin/python"), "runtime_hashing_enabled": False,
                "cfg_initial": [i / 2 for i in range(int(args.cfg_min * 2), int(args.cfg_max * 2) + 1)],
                "cfg_increment": 0.5, "cfg_limit": args.cfg_limit, "cfg_steps": 10,
                "boundary_policy": "extend_by_0.5_until_both_winners_are_interior_or_lower_bound_zero",
                "selection": "minimum_fid_and_maximum_is_separately", "heun_steps": [5, 10, 20, 50, 100],
                "samples_per_arm": 50000, "seed": 42, "npu_per_worker": 16, "arm_timeout_hours": 12,
                "real_stats": str((repo / "public/datasets/imagenet_full/fid_stats/inception_v3_2048_imagenet_val50000_256.pt").resolve()),
                "inception_weights": str((repo / "public/models/torch-fidelity/weights-inception-2015-12-05-6726825d.pth").resolve())}
    inputs = (config, source / "scripts/evaluate_single_stream_fid_is.py", Path(protocol["python"]),
                 Path(protocol["real_stats"]), Path(protocol["inception_weights"]),
                 repo / "public/vae/mar-kl16/kl16.ckpt",
                 repo / "public/datasets/imagenet_full/vae_posterior_mar_kl16/posterior_stats_imagenet1k_val_fp16.pt",
                 repo / "public/datasets/imagenet1k_synthetic_v1/indexed/val/manifest.json")
    for path in inputs:
        require(path.is_file(), f"missing input: {path}")
    require("--save_image_count" in (source / "scripts/evaluate_single_stream_fid_is.py").read_text()
            and (source / "utils/evaluation_image_subset.py").is_file(),
            "source snapshot must support deterministic image-subset export")
    baseline = read(args.baseline_metrics)
    reused = task(baseline["cfg"], int(baseline["sampling_steps"]), "cfg")
    result = validate_metrics(args.baseline_metrics, protocol, reused)
    require(reused["cfg"] in protocol["cfg_initial"] and reused["steps"] == 10, "baseline outside initial grid")
    sys.path.insert(0, str(repo))
    from utils.evaluation_image_subset import evenly_spaced_image_indices
    protocol.update(save_image_count=args.save_image_count,
                    saved_image_indices=evenly_spaced_image_indices(50000, args.save_image_count),
                    cfg_extension_points=4, platform=platform)
    platform.update(nodes_per_job=1, initial_jobs=len(protocol["cfg_initial"]),
                    initial_gpus=16 * len(protocol["cfg_initial"]))
    root.mkdir(parents=True)
    shutil.copytree(source, root / "launch/source", symlinks=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".git"))
    shutil.copy2(__file__, root / "launch/sweep_unified_t2i_sampling.py")
    shutil.copy2(repo / "scripts/submit_unified_t2i_sweep.py", root / "launch/submit_unified_t2i_sweep.py")
    context = root / "launch/cli-context/.inspire"
    context.mkdir(parents=True)
    (context / "config.toml").write_text("[path_aliases]\nme = " + json.dumps(str((repo / "public").resolve()) + "/") + "\n")
    evidence = []
    for path in sorted({p.resolve() for p in (*inputs, *model.iterdir()) if p.is_file()}):
        stat = path.stat()
        evidence.append({"input": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    write(root / "launch/input-audit.json", {"files": evidence, "runtime_hashing_enabled": False})
    write(root / "protocol.json", protocol)
    state = {"status": "running", "phase": "cfg", "created_at": now(),
             "tasks": [task(cfg, 10, "cfg") for cfg in protocol["cfg_initial"]]}
    # Re-evaluate the baseline too, so all arms retain the same paired images.
    state["baseline_reference"] = {**reused, "status": "done", "result": result,
                                   "reused_from": str(args.baseline_metrics.resolve())}
    # Start around the known baseline while every point in the grid is retained.
    state["tasks"].sort(key=lambda a: (abs(a["cfg"] - reused["cfg"]), a["cfg"]))
    save(root, state, protocol)
    launch = ["#!/usr/bin/env bash", "set -euo pipefail", "set +u",
              'source /usr/local/Ascend/ascend-toolkit/set_env.sh', "set -u",
              'for sweep_driver_dir in /usr/local/Ascend/driver/lib64/driver /usr/local/Ascend/driver/lib64/common /usr/local/Ascend/driver/lib64; do',
              '  if [[ -d "$sweep_driver_dir" ]]; then export LD_LIBRARY_PATH="${sweep_driver_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"; fi',
              'done',
              f"export UNIFIED_MM_VENV={shlex.quote(str(repo / '.venv'))}",
              f"source {shlex.quote(str(root / 'launch/source/script/offline_env.sh'))}",
              "export WANDB_MODE=disabled PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false",
              "export HCCL_INTRA_ROCE_ENABLE=1 HCCL_CONNECT_TIMEOUT=600",
              "export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True TRANSFORMERS_VERBOSITY=error",
              "unset PYTORCH_CUDA_ALLOC_CONF CUDA_VISIBLE_DEVICES",
              f"exec {shlex.quote(protocol['python'])} {shlex.quote(str(root / 'launch/sweep_unified_t2i_sampling.py'))} worker --output-dir {shlex.quote(str(root))} \"$@\""]
    (root / "launch/run.sh").write_text("\n".join(launch) + "\n")
    print(json.dumps({"root": str(root), "cfg": protocol["cfg_initial"], "baseline": result}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--output-dir", type=Path, required=True)
    prepare_parser.add_argument("--model-source", type=Path, required=True)
    prepare_parser.add_argument("--config", type=Path, required=True)
    prepare_parser.add_argument("--source-repo", type=Path, required=True)
    prepare_parser.add_argument("--baseline-metrics", type=Path, required=True)
    prepare_parser.add_argument("--platform-json", type=Path, required=True,
                                help="Explicit workspace, project, compute_group, quota, image and requested_priority.")
    prepare_parser.add_argument("--save-image-count", type=int, default=64)
    prepare_parser.add_argument("--cfg-min", type=float, default=1.0)
    prepare_parser.add_argument("--cfg-max", type=float, default=6.0)
    prepare_parser.add_argument("--cfg-limit", type=float, default=12.0)
    worker_parser = sub.add_parser("worker")
    worker_parser.add_argument("--output-dir", type=Path, required=True)
    worker_parser.add_argument("--first-cfg", type=float)
    worker_parser.add_argument("--task-id", help="Run exactly this arm, then release the 16-NPU Job.")
    for name in ("report", "audit"):
        sub.add_parser(name).add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args)
    elif args.action == "worker":
        worker(args.output_dir.resolve(), args.first_cfg, args.task_id)
    else:
        root = args.output_dir.resolve()
        with locked(root):
            state, protocol = read(root / "state.json"), read(root / "protocol.json")
            for arm in state["tasks"]:
                if arm["status"] == "done":
                    require(validate_metrics(arm["result"]["metrics_path"], protocol, arm) == arm["result"], "retained metric changed")
            if args.action == "audit":
                require(state["status"] == "complete", "sweep not complete")
            report(root, state, protocol)
            print(json.dumps({"status": state["status"], "phase": state["phase"],
                  "done": sum(a["status"] == "done" for a in state["tasks"]), "total": len(state["tasks"])}, indent=2))


if __name__ == "__main__":
    main()
