#!/usr/bin/env python3
"""Freeze the selected final-EMA ablation matrix for paired CFG=2/Heun=10 evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.sweep_unified_t2i_sampling import read, write, require, now, save
from utils.evaluation_image_subset import evenly_spaced_image_indices
from utils.image_order_strategies import order_policy, checkpoint_generation_contract


def prepare(root, include_random=False):
    repo = Path(__file__).resolve().parents[1]
    root = root.resolve()
    require(not root.exists(), f"refusing to overwrite {root}")
    selection = read(repo / "configs/protocols/evaluation_report.json")
    gallery = read(repo / "output/evaluation" / selection["qualitative"] / "manifest.json")
    selected = [m for m in gallery["models"] if m["id"] in selection["models"]]
    require(len(selected) == len(selection["models"]) == 9, "review changed formal matrix inventory")
    source = root / "launch/source"
    source.mkdir(parents=True)
    for folder in ("models", "utils", "pretrain", "scripts", "script", "configs", "accelerate_configs"):
        shutil.copytree(repo / folder, source / folder, symlinks=True,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".git"))
    for name in ("public", ".venv"):
        (source / name).symlink_to((repo / name).resolve(), target_is_directory=True)
    models, evidence = {}, []
    for model in selected:
        path = Path(model["checkpoint"])
        config_path = root / "launch/configs" / f"{model['id']}.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path.parent / "config.yaml", config_path)
        export = read(path / "ema_export_metadata.json")
        require(export["source_global_step"] == model["source"]["global_step"] == 95415, "final EMA step mismatch")
        contract = checkpoint_generation_contract(read(path / "config.json"))
        require(contract["backbone_attention"] == model["backbone_attention"]
                and contract["flow_attention"] == model["flow_attention"], "checkpoint inventory drift")
        models[model["id"]] = {
            "model_source": str(path), "config": str(config_path), "checkpoint_step": 95415,
            "label": model["label"], "run": model["run"], "generation_contract": contract,
            "backbone_attention": contract["backbone_attention"], "flow_attention": contract["flow_attention"],
            "native_strategy": "sequential" if model["image_order"] == "sequential" else "spatial_halton",
            "previous_evaluation": selection["models"][model["id"]],
        }
        evidence += [p for p in path.iterdir() if p.is_file()] + [config_path]
    comparisons = ["spatial_halton", "confidence_stability"] + (["random"] if include_random else [])
    platform = {
        "workspace": "昇腾卡公共空间", "project": "随机序语言建模-统一自回归与掩码扩散的随机顺序生成框架",
        "compute_group": "910B资源", "quota": "16,128,1024",
        "image": "docker-t.sii.shaipower.online/inspire-studio/dev-wjx-ascend:v-1.3",
        "requested_priority": 6, "nodes_per_job": 1, "initial_jobs": len(models) * len(comparisons) + 1,
        "max_concurrent_jobs": 12, "max_observable_gpus": 224,
        "exclude_nodes": ["infra-gpu-npu-248.host.shzhisuan.com", "infra-gpu-npu-259.host.shzhisuan.com"],
    }
    protocol = {
        "schema": "unified_t2i_ablation_matrix_v1", "created_at": now(), "models": models,
        "source_repo": str(source), "source_provenance": str(repo),
        "python": str(repo / ".venv/bin/python"), "runtime_hashing_enabled": False,
        "cfg_fixed": 2.0, "heun_fixed": 10, "samples_per_arm": 50000, "seed": 42,
        "npu_per_worker": 16, "arm_timeout_hours": 12, "require_smoke": True,
        "save_image_count": 64, "saved_image_indices": evenly_spaced_image_indices(50000, 64),
        "matrix_strategies": comparisons,
        "strategies": [*comparisons, "sequential"],
        "order_policies": {s: order_policy(s) for s in [*comparisons, "sequential"]},
        "real_stats": str((repo / "public/datasets/imagenet_full/fid_stats/inception_v3_2048_imagenet_val50000_256.pt").resolve()),
        "inception_weights": str((repo / "public/models/torch-fidelity/weights-inception-2015-12-05-6726825d.pth").resolve()),
        "platform": platform,
        "scope": "Nine selected formal final EMAs; paired " + ", ".join(comparisons) + " plus E-native sequential control.",
        "baseline_reproduction": str(repo / "output/evaluation/unified-b-x0content-0p6b/sweeps/order-20260908-r1"),
    }
    tasks = []
    # D is substantially slower: dispatch both independent D arms first.
    ordered_models = sorted(models, key=lambda m: (m != "d_on_b", list(models).index(m)))
    for mid in ordered_models:
        for strategy in comparisons:
            label = {"spatial_halton": "halton", "confidence_stability": "stability", "random": "random"}[strategy]
            identity = f"{mid.replace('_', '-')}-{label}"
            tasks.append({"id": identity, "job_label": identity, "model": mid, "strategy": strategy,
                "cfg": 2.0, "steps": 10, "phase": "matrix", "status": "pending"})
    tasks.append({"id": "e-on-b-sequential", "job_label": "e-on-b-sequential", "model": "e_on_b",
        "strategy": "sequential", "cfg": 2.0, "steps": 10, "phase": "matrix", "status": "pending"})
    write(root / "protocol.json", protocol)
    save(root, {"status": "running", "phase": "matrix", "created_at": now(), "tasks": tasks}, protocol)
    write(root / "launch/selected-inventory.json", {"selection": selection, "models": selected})
    inputs = [Path(protocol[k]) for k in ["real_stats", "inception_weights", "python"]]
    inputs += [(repo / p).resolve() for p in [
        "public/vae/mar-kl16/kl16.ckpt", "public/datasets/imagenet_full/manifest_val.jsonl",
        "public/datasets/imagenet_full/vae_posterior_mar_kl16/posterior_stats_imagenet1k_val_fp16.pt",
        "public/datasets/imagenet1k_synthetic_v1/indexed/val/manifest.json"]]
    evidence += inputs + [p for folder in ["models", "scripts", "utils"] for p in (source / folder).rglob("*.py")]
    write(root / "launch/input-audit.json", {"runtime_hashing_enabled": False, "files": [
        {"input": str(p), "size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns} for p in evidence]})
    context = root / "launch/cli-context/.inspire"
    context.mkdir(parents=True)
    (context / "config.toml").write_text('[path_aliases]\nme = ' + json.dumps(str((repo / 'public').resolve()) + '/') + '\n')
    for name in ["sweep_unified_t2i_sampling.py", "submit_unified_t2i_sweep.py", "prepare_unified_matrix_sweep.py"]:
        shutil.copy2(repo / "scripts" / name, root / "launch" / name)
    setup = ["#!/usr/bin/env bash", "set -euo pipefail", "set +u",
        "source /usr/local/Ascend/ascend-toolkit/set_env.sh", "set -u",
        'for matrix_driver_dir in /usr/local/Ascend/driver/lib64/driver /usr/local/Ascend/driver/lib64/common /usr/local/Ascend/driver/lib64; do',
        '  if [[ -d "$matrix_driver_dir" ]]; then export LD_LIBRARY_PATH="${matrix_driver_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"; fi', "done",
        f"export UNIFIED_MM_VENV={shlex.quote(str(repo / '.venv'))}",
        f"source {shlex.quote(str(source / 'script/offline_env.sh'))}",
        "export WANDB_MODE=disabled PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false",
        "export HCCL_INTRA_ROCE_ENABLE=1 HCCL_CONNECT_TIMEOUT=600",
        "export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True TRANSFORMERS_VERBOSITY=error",
        "unset PYTORCH_CUDA_ALLOC_CONF CUDA_VISIBLE_DEVICES",
        f"cd {shlex.quote(str(source))}"]
    (root / "launch/run.sh").write_text("\n".join([*setup,
        f"exec {shlex.quote(protocol['python'])} {shlex.quote(str(root / 'launch/sweep_unified_t2i_sampling.py'))} worker --output-dir {shlex.quote(str(root))} \"$@\""]) + "\n")
    (root / "smoke").mkdir()
    (root / "launch/smoke.sh").write_text("\n".join([*setup,
        f"exec {shlex.quote(protocol['python'])} -m torch.distributed.run --standalone --nproc_per_node=16 scripts/smoke_unified_matrix_order.py --output-dir {shlex.quote(str(root))}"]) + "\n")
    print(json.dumps({"root": str(root), "models": list(models), "arms": len(tasks), "concurrent_gpus": 192}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--include-random", action="store_true", help="Add one random-order arm per model (28 total)")
    args = parser.parse_args()
    prepare(args.output_dir, include_random=args.include_random)
