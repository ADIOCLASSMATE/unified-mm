#!/usr/bin/env python3
"""Launch a fresh 31,800-update ablation on four 16-NPU instances."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.short_ablation_protocol import ARMS, asset_preflight, config_path, validate_short_config


def launch_plan(arm, *, environment, smoke=False, label="r1", steps=12, resume=None):
    config_file = config_path(arm)
    config = OmegaConf.load(ROOT / config_file)
    contract = validate_short_config(arm, config)
    if not re.fullmatch(r"[a-z0-9-]+", label) or not 2 <= steps <= 100:
        raise ValueError("Smoke requires a lowercase/digit/hyphen label and 2..100 steps")
    machines, world = (1, 16) if smoke else (4, 64)
    rank = 0 if smoke else int(environment.get("PET_NODE_RANK", -1))
    if not smoke:
        nodes = int(environment.get("PET_NNODES", 0))
        local_processes = int(environment.get("PET_NPROC_PER_NODE", 0))
        if not 0 <= rank < machines or nodes != machines or local_processes not in (0, 16):
            raise ValueError("Formal training requires PET_NODE_RANK=0..3, PET_NNODES=4, PET_NPROC_PER_NODE=0 or 16")
    run = f"{contract['run_project']}-smoke-{label}" if smoke else contract["run_project"]
    output = ROOT / config.experiment.output_dir / run
    audit = output / "prelaunch_audit" / f"node-{rank}"
    resume_step = 0
    if resume:
        from utils.training_checkpoint import _validate_checkpoint_complete
        checkpoint = Path(resume).resolve()
        if checkpoint.parent != output.resolve():
            raise ValueError("Resume checkpoint must belong to this exact short-budget run")
        metadata = json.loads((checkpoint / "metadata.json").read_text())
        resume_step = int(metadata["global_step"])
        target = steps if smoke else contract["optimizer_steps"]
        if int(metadata["world_size"]) != world or not 0 < resume_step < target:
            raise ValueError("Resume must retain world size and advance beyond the saved step")
        _validate_checkpoint_complete(checkpoint, expected_global_step=resume_step)
        resume = str(checkpoint)
        audit = output / "prelaunch_audit" / f"resume-{resume_step}-to-{target}" / f"node-{rank}"
    accelerate = f"accelerate_configs/{world}_npus_{machines}{'node' if machines == 1 else 'nodes'}_deepspeed_zero2.yaml"
    command = [sys.executable, "scripts/launch_accelerate_multinode.py", "launch",
               "--config_file", accelerate, "--num_machines", str(machines),
               "--num_processes", str(world), "--machine_rank", str(rank)]
    if not smoke:
        address = environment.get("PET_MASTER_ADDR") or environment.get("MASTER_ADDR")
        port = environment.get("PET_MASTER_PORT") or environment.get("MASTER_PORT")
        if not address or not port:
            raise ValueError("Formal training requires the platform master address and port")
        command += ["--main_process_ip", address, "--main_process_port", port,
                    "--rdzv_backend", "static", "--same_network"]
    command += ["pretrain/train_selfless_flow.py", f"config={config_file}",
                f"experiment.project={run}", f"experiment.name={run}",
                f"experiment.resume_from_checkpoint={resume or 'none'}"]
    if smoke:
        command += [f"training.stop_after_steps={steps}", "experiment.log_every=1",
                    "experiment.log_grad_norm_every=1", "experiment.flow_stats_every=1",
                    f"experiment.deepspeed_bf16_overflow_check_until_step={steps}",
                    "experiment.val_every=0", "experiment.save_every=1000000000",
                    "experiment.checkpoint_milestone_every=0", "experiment.save_ema_eval_every=0"]
    return dict(contract=contract, smoke=smoke, world_size=world, rank=rank,
                output_root=str(output), audit_root=str(audit), resume_step=resume_step,
                command=command, preflight=asset_preflight(arm, sys.executable, assets=rank == 0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=tuple(ARMS), required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--label", default="r1")
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    plan = launch_plan(args.arm, environment=dict(os.environ), smoke=args.smoke,
                       label=args.label, steps=args.steps, resume=args.resume_from_checkpoint)
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    output, audit = Path(plan["output_root"]), Path(plan["audit_root"])
    if plan["rank"] == 0 and not plan["resume_step"] and (output / "config.yaml").exists():
        raise FileExistsError(f"Fresh run would overwrite {output}")
    audit.mkdir(parents=True, exist_ok=True)
    (audit / "launch_plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n")
    (audit / "launch_command.sh").write_text(shlex.join(plan["command"]) + "\n")
    with (audit / "asset_preflight.json").open("w") as stream:
        subprocess.run(plan["preflight"], stdout=stream, check=True)
    environment = dict(os.environ)
    for key in ("WORLD_SIZE", "RANK", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "GROUP_RANK",
                "GROUP_WORLD_SIZE", "ROLE_RANK", "ROLE_WORLD_SIZE"):
        environment.pop(key, None)
    print(json.dumps({"event": "training_launch", "contract": plan["contract"]}, ensure_ascii=False), flush=True)
    with (audit / "training.log").open("w") as stream:
        process = subprocess.Popen(plan["command"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1, env=environment)
        assert process.stdout is not None
        for line in process.stdout:
            stream.write(line)
            stream.flush()
            print(line, end="", flush=True)
        code = process.wait()
    (audit / "exit_code.txt").write_text(f"{code}\n")
    raise SystemExit(code)


if __name__ == "__main__":
    main()
