#!/usr/bin/env python3
"""Exercise the real S2 validation image runner twice on the fixed 16-NPU Notebook."""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import random
import sys

import numpy as np
import torch
import torch.distributed as dist
import torch_npu  # noqa: F401
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.short_ablation_protocol import config_path, validate_short_config
from utils.training_image_generation import TrainingImageGenerator
from utils.utils import load_model_tokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    os.chdir(ROOT)
    rank = int(os.environ["LOCAL_RANK"])
    if torch.npu.device_count() != 16 or int(os.environ["WORLD_SIZE"]) != 16:
        raise RuntimeError("Use the fixed 16-NPU development Notebook")
    torch.npu.set_device(rank)
    device = torch.device("npu", rank)
    dist.init_process_group("hccl")
    reports = []
    try:
        for arm in ("s2-single", "s2-text2stream"):
            random.seed(42)
            np.random.seed(42)
            torch.manual_seed(42)
            torch.npu.manual_seed(42)
            config = OmegaConf.load(config_path(arm))
            validate_short_config(arm, config)
            model, tokenizer = load_model_tokenizer(config, model_dtype=torch.bfloat16)
            model.to(device).train()
            model.image_flow_head.layers[0].eval()
            modes = [m.training for m in model.modules()]
            samples = {name: value.detach().reshape(-1)[:16].clone() for name, value in model.state_dict().items()}
            rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.npu.get_rng_state(device))
            runner = TrainingImageGenerator(config)
            output = args.output_dir / arm
            rounds = []
            for step in (2, 4):
                result = runner.run(model, tokenizer, device=device, step=step, output_dir=output)
                assert result["complete"] and result["samples"] == 16
                assert result["weight_source"] == "raw" and result["cache_mode"] == "full_model_refresh"
                assert [m.training for m in model.modules()] == modes
                assert random.getstate() == rng[0]
                assert np.array_equal(np.random.get_state()[1], rng[1][1])
                assert torch.equal(torch.get_rng_state(), rng[2])
                assert torch.equal(torch.npu.get_rng_state(device), rng[3])
                for name, value in model.state_dict().items():
                    torch.testing.assert_close(value.detach().reshape(-1)[:16], samples[name], rtol=0, atol=0)
                for row in result["images"]:
                    assert row["trace"]["backbone_calls"] == row["trace"]["flow_head_calls"] == 40
                    assert row["trace"]["backbone_kv_cache_enabled"] is False
                rounds.append(dict(step=step, samples=16, wall_seconds=result["wall_seconds"]))
            for row in result["images"]:
                first = output / "validation_generation/step-2" / row["image"]
                second = output / "validation_generation/step-4" / row["image"]
                assert first.read_bytes() == second.read_bytes()
            reports.append(dict(arm=arm, passed=True, rounds=rounds, full_latent_tokens=256,
                                fixed_images_repeat=True, rng_and_modes_restored=True,
                                parameter_samples_unchanged=len(samples), weights="raw", cfg=3.5, solver="heun", steps=10))
            if rank == 0:
                print(json.dumps(reports[-1]), flush=True)
            del model, tokenizer, runner, samples
            gc.collect()
            torch.npu.empty_cache()
            dist.barrier()
        if rank == 0:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            (args.output_dir / "report.json").write_text(json.dumps(dict(passed=True, world_size=16, reports=reports), indent=2) + "\n")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
