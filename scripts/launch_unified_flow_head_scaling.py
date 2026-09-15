#!/usr/bin/env python3
"""Launch a real-data 16-NPU memory smoke or a complete 64-NPU scaling arm."""

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

from utils.flow_head_scaling import config_path, validate_scaling_config
from utils.experiment_registry import experiment_identity


def launch_plan(depth: int, *, smoke: bool, label: str, steps: int, environment: dict,
                ablation: str = "b") -> dict:
    if ablation == "f":
        from utils.positionwise_flow_head_scaling import config_path as arm_config_path
        from utils.positionwise_flow_head_scaling import validate_scaling_config as validate_arm
    elif ablation == "b":
        arm_config_path, validate_arm = config_path, validate_scaling_config
    else:
        raise ValueError("flow-head scaling is defined only for B and F")
    config_file = arm_config_path(depth)
    config = OmegaConf.load(config_file)
    contract = validate_arm(config)
    if not re.fullmatch(r"[a-z0-9-]+", label):
        raise ValueError("smoke label must contain only lowercase letters, digits and hyphens")
    if not 2 <= steps <= 100:
        raise ValueError("smoke steps must be between 2 and 100")
    if smoke:
        world, machines, rank = 16, 1, 0
        project = f"{contract['run_project']}-smoke-{label}"
        accelerate_file = "accelerate_configs/16_npus_1node_deepspeed_zero2.yaml"
    else:
        world, machines = 64, 4
        rank = int(environment.get("PET_NODE_RANK", "-1"))
        if not 0 <= rank < machines or int(environment.get("PET_NNODES", "0")) != machines:
            raise ValueError("formal scaling requires PET_NODE_RANK=0..3 and PET_NNODES=4")
        if int(environment.get("PET_NPROC_PER_NODE", "0")) not in (0, 16):
            raise ValueError("formal scaling requires 16 NPUs per instance")
        project = contract["run_project"]
        accelerate_file = "accelerate_configs/64_npus_4nodes_deepspeed_zero2.yaml"
    output_root = Path(config.experiment.output_dir) / project
    audit_root = output_root / "prelaunch_audit" / f"node-{rank}"
    command = [
        sys.executable, "scripts/launch_accelerate_multinode.py", "launch",
        "--config_file", accelerate_file,
        "--num_machines", str(machines), "--num_processes", str(world),
        "--machine_rank", str(rank),
    ]
    if not smoke:
        address = environment.get("PET_MASTER_ADDR") or environment.get("MASTER_ADDR")
        port = environment.get("PET_MASTER_PORT") or environment.get("MASTER_PORT")
        if not address or not port:
            raise ValueError("formal scaling requires the platform master address and port")
        command.extend(["--main_process_ip", address, "--main_process_port", port,
                        "--rdzv_backend", "static", "--same_network"])
    command.extend([
        "pretrain/train_selfless_flow.py", f"config={config_file}",
        f"experiment.project={project}", f"experiment.name={project}",
        "experiment.resume_from_checkpoint=none",
    ])
    if smoke:
        command.extend([
            f"training.stop_after_steps={steps}",
            "experiment.log_every=1", "experiment.log_grad_norm_every=1",
            "experiment.flow_stats_every=2",
            f"experiment.deepspeed_bf16_overflow_check_until_step={steps}",
            "experiment.val_every=0", "experiment.save_every=1000000000",
            "experiment.checkpoint_milestone_every=0", "experiment.save_ema_eval_every=0",
            "experiment.save_final=true", "experiment.save_final_checkpoint=true",
        ])
    preflight = [
        sys.executable, "scripts/validate_unified_baseline.py",
        "--config", config_file, "--formal-world-size", "64",
        "--require-npu-count", "16", "--run-project", project,
        "--ablation", ablation, "--flow-head-scaling",
    ]
    if rank == 0:
        preflight.append("--tokenizer-probe")
    return {"contract": contract, "smoke": smoke, "world_size": world,
            "experiment_identity": experiment_identity(project, OmegaConf.to_container(config, resolve=True)),
            "per_rank_batches": {"climbmix": 4, "t2i": 16, "i2t": 16},
            "gradient_accumulation_steps": 4, "image_flow_batch_mul": 4,
            "rank": rank, "output_root": str(output_root), "audit_root": str(audit_root),
            "preflight": preflight, "command": command}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ablation", choices=("b", "f"), default="b")
    parser.add_argument("--depth", type=int, choices=(16, 30), required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--label", default="memory-r1")
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    plan = launch_plan(args.depth, smoke=args.smoke, label=args.label, steps=args.steps,
                       environment=dict(os.environ), ablation=args.ablation)
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    output_root, audit_root = Path(plan["output_root"]), Path(plan["audit_root"])
    if plan["rank"] == 0 and (output_root / "config.yaml").exists():
        raise FileExistsError(f"refusing a fresh run over {output_root}/config.yaml")
    audit_root.mkdir(parents=True, exist_ok=True)
    (audit_root / "launch_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    (audit_root / "launch_command.sh").write_text(shlex.join(plan["command"]) + "\n")
    with (audit_root / "asset_preflight.json").open("w") as stream:
        subprocess.run(plan["preflight"], stdout=stream, check=True)
    child_environment = dict(os.environ)
    for key in ("WORLD_SIZE", "RANK", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "GROUP_RANK",
                "GROUP_WORLD_SIZE", "ROLE_RANK", "ROLE_WORLD_SIZE"):
        child_environment.pop(key, None)
    print(json.dumps({"event": "training_launch", "output_root": str(output_root),
                      "head": plan["contract"]}), flush=True)
    with (audit_root / "training.log").open("w") as stream:
        process = subprocess.Popen(plan["command"], stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1,
                                   env=child_environment)
        assert process.stdout is not None
        for line in process.stdout:
            stream.write(line)
            stream.flush()
            print(line, end="", flush=True)
        code = process.wait()
    (audit_root / "exit_code.txt").write_text(f"{code}\n")
    raise SystemExit(code)


if __name__ == "__main__":
    main()
