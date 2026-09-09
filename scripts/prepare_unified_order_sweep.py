#!/usr/bin/env python3
"""Prepare a frozen B order study after auditing the final CFG neighborhood."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
import shlex
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.sweep_unified_t2i_sampling import read, write, require, now, validate_metrics, winners, save, evaluator_command
from utils.image_order_strategies import CONFIDENCE_STRATEGIES, order_policy


def prepare(previous, root):
    repo = Path(__file__).resolve().parents[1]
    previous, root = previous.resolve(), root.resolve()
    require(not root.exists(), f"refusing to overwrite {root}")
    old_protocol, old_state = read(previous / "protocol.json"), read(previous / "state.json")
    require(old_state["status"] == "complete", "finish CFG/Heun study before order selection")
    for arm in old_state["tasks"]:
        require(validate_metrics(arm["result"]["metrics_path"], old_protocol, arm) == arm["result"], "previous result changed")
    selected = winners(old_state["tasks"])["best_fid"]
    cfgs = [selected["cfg"] + offset for offset in (-1, -.5, 0, .5, 1) if selected["cfg"] + offset >= 0]
    neighborhood = [a for a in old_state["tasks"] if a["cfg"] in cfgs and a["steps"] == selected["steps"]]
    require({a["cfg"] for a in neighborhood} == set(cfgs), "missing final-step CFG neighbors: run them before preparing order sweep")
    best = winners(neighborhood)["best_fid"]
    require(min(cfgs) < best["cfg"] < max(cfgs), "CFG optimum at neighborhood boundary: extend before order sweep")
    require(best["cfg"] == 2 and best["steps"] == 10, "review order-study protocol after a changed CFG/Heun selection")
    root.mkdir(parents=True)
    write(root / "cfg-refinement.json", {
        "status": "complete", "audited_at": now(), "selected_heun_steps": best["steps"],
        "center_cfg": selected["cfg"], "offsets": [-1, -.5, 0, .5, 1], "cfg_values": cfgs,
        "evidence": "Reused exact completed 50K arms at the selected Heun step; no duplicate evaluation.",
        "arms": sorted(neighborhood, key=lambda a: a["cfg"]), "best_fid": best,
        "optimum_is_interior": True, "extension_required": False,
    })
    source = root / "launch/source"
    shutil.copytree(previous / "launch/source", source, symlinks=True,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".git"))
    changes = ["models/modeling_model/modeling_selfless_generation.py",
               "scripts/evaluate_single_stream_fid_is.py", "utils/image_order_strategies.py"]
    for name in changes:
        shutil.copy2(repo / name, source / name)
    for name in ("sweep_unified_t2i_sampling.py", "submit_unified_t2i_sweep.py", "prepare_unified_order_sweep.py"):
        shutil.copy2(repo / "scripts" / name, root / "launch" / name)
    shutil.copytree(previous / "launch/cli-context", root / "launch/cli-context")
    strategies = ["spatial_halton", "sequential", "spatial_uniform", "random", *CONFIDENCE_STRATEGIES]
    labels = ["halton", "raster", "center", "random", "cfg-agree", "cfg-reverse", "stability", "probe-control"]
    protocol = copy.deepcopy(old_protocol)
    protocol.update(schema="unified_t2i_order_sweep_v1", created_at=now(), source_repo=str(source),
                    source_provenance=str(previous / "launch/source"), previous_sweep=str(previous),
                    selection="minimum_fid", cfg_fixed=best["cfg"], heun_fixed=best["steps"],
                    strategies=strategies, order_policies={s: order_policy(s) for s in strategies},
                    source_changes=changes, require_smoke=True)
    protocol["platform"].update(initial_jobs=len(strategies), initial_gpus=16 * len(strategies))
    write(root / "protocol.json", protocol)
    tasks = [{"id": label, "job_label": label, "strategy": strategy, "cfg": best["cfg"],
              "steps": best["steps"], "phase": "order", "status": "pending"}
             for strategy, label in zip(strategies, labels)]
    save(root, {"status": "running", "phase": "order", "created_at": now(), "tasks": tasks}, protocol)
    audited = [p for p in Path(protocol["model_source"]).iterdir() if p.is_file()]
    audited += [Path(protocol[k]) for k in ("config", "real_stats", "inception_weights")]
    audited += [source / name for name in changes]
    write(root / "launch/input-audit.json", {"runtime_hashing_enabled": False, "files": [
        {"input": str(p), "size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns} for p in audited]})
    run = (previous / "launch/run.sh").read_text().replace(str(previous), str(root))
    (root / "launch/run.sh").write_text(run)
    smoke = root / "smoke"
    smoke.mkdir()
    command = evaluator_command(protocol, tasks[0], smoke)
    for flag, value in {"--samples": "32", "--batch_size": "32", "--is_splits": "8",
                        "--vae_decode_batch_size": "2", "--save_image_count": "8",
                        "--strategies": ",".join(strategies)}.items():
        command[command.index(flag) + 1] = value
    command.remove("--require_formal_protocol")
    command[command.index("--resume_progress")] = "--no-resume_progress"
    setup = run[:run.index("exec ")]
    setup += f"cd {shlex.quote(str(source))}\n"
    setup += shlex.join(command) + f" > {shlex.quote(str(smoke / 'run.log'))} 2>&1\n"
    setup += f"touch {shlex.quote(str(smoke / 'EVALUATOR_SUCCEEDED'))}\n"
    (root / "launch/smoke.sh").write_text(setup)
    print(str(root))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-sweep", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.previous_sweep, args.output_dir)
