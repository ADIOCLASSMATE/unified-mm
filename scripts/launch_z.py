#!/usr/bin/env python3
"""Launch experiment Z on 64 NPUs, or its bounded 16-NPU smoke."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from omegaconf import OmegaConf
from utils.joint_dit_protocol import CONFIG, RUN, validate_joint_dit_config


def launch_plan(*, smoke, label, steps, environment, resume=None):
    config = OmegaConf.load(ROOT / CONFIG)
    contract = validate_joint_dit_config(config)
    if not re.fullmatch(r"[a-z0-9-]+", label) or not 2 <= steps <= 100:
        raise ValueError("Smoke requires a lowercase/digit/hyphen label and 2..100 steps")
    machines, world = (1, 16) if smoke else (4, 64)
    rank = 0 if smoke else int(environment.get("PET_NODE_RANK", -1))
    if not smoke and (not 0 <= rank < 4 or int(environment.get("PET_NNODES", 0)) != 4
                      or int(environment.get("PET_NPROC_PER_NODE", 16)) != 16):
        raise ValueError("Formal training requires four platform instances with 16 NPUs each")
    project = f"{RUN}-smoke-{label}" if smoke else RUN
    output = ROOT / config.experiment.output_dir / project
    audit = output / "prelaunch_audit" / f"node-{rank}"
    resume_step = 0
    if resume:
        from utils.training_checkpoint import _validate_checkpoint_complete
        path = Path(resume).resolve()
        if path.parent != output.resolve():
            raise ValueError("Resume checkpoint must belong to this exact run")
        metadata = json.loads((path / "metadata.json").read_text())
        resume_step = int(metadata["global_step"])
        if metadata["world_size"] != world or not 0 < resume_step < (steps if smoke else 95415):
            raise ValueError("Resume must retain world size and advance beyond the saved step")
        _validate_checkpoint_complete(path, expected_global_step=resume_step)
        resume = str(path)
        audit = output / "prelaunch_audit" / f"resume-{resume_step}" / f"node-{rank}"
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
    command += ["pretrain/train_selfless_flow.py", f"config={CONFIG}",
                f"experiment.project={project}", f"experiment.name={project}",
                f"experiment.resume_from_checkpoint={resume or 'none'}"]
    if smoke:
        command += [f"training.stop_after_steps={steps}", "experiment.log_every=1",
            "experiment.log_grad_norm_every=1", "experiment.flow_stats_every=1",
            f"experiment.deepspeed_bf16_overflow_check_until_step={steps}",
            "experiment.val_every=0", "experiment.save_every=1000000000",
            "experiment.checkpoint_milestone_every=0", "experiment.save_ema_eval_every=0"]
    preflight = [sys.executable, "scripts/validate_z.py", "--require-npu-count", "16"]
    if rank == 0:
        preflight.append("--assets")
    return dict(contract=contract, smoke=smoke, world_size=world, rank=rank,
                output_root=str(output), audit_root=str(audit), resume_step=resume_step,
                command=command, preflight=preflight)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--label", default="r1")
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    plan = launch_plan(smoke=args.smoke, label=args.label, steps=args.steps,
                       environment=dict(os.environ), resume=args.resume_from_checkpoint)
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    output, audit = Path(plan["output_root"]), Path(plan["audit_root"])
    if plan["rank"] == 0 and not plan["resume_step"] and (output / "config.yaml").exists():
        raise FileExistsError(f"Fresh run would overwrite {output}")
    audit.mkdir(parents=True, exist_ok=True)
    (audit / "launch_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    (audit / "launch_command.sh").write_text(shlex.join(plan["command"]) + "\n")
    with (audit / "asset_preflight.json").open("w") as stream:
        subprocess.run(plan["preflight"], stdout=stream, check=True)
    environment = dict(os.environ)
    for key in ("WORLD_SIZE", "RANK", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "GROUP_RANK",
                "GROUP_WORLD_SIZE", "ROLE_RANK", "ROLE_WORLD_SIZE"):
        environment.pop(key, None)
    with (audit / "training.log").open("w") as stream:
        process = subprocess.Popen(plan["command"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1, env=environment)
        for line in process.stdout:
            stream.write(line); stream.flush()
            print(line, end="", flush=True)
        code = process.wait()
    (audit / "exit_code.txt").write_text(f"{code}\n")
    raise SystemExit(code)


if __name__ == "__main__":
    main()
