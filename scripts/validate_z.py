#!/usr/bin/env python3
"""Audit experiment Z, its parameter budget, validation images, and data assets."""
import argparse
import glob
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from omegaconf import OmegaConf
from utils.joint_experiments import joint_experiment_protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", action="store_true")
    parser.add_argument("--require-npu-count", type=int, default=0)
    parser.add_argument("--experiment", choices=("z", "z-b"), default="z")
    args = parser.parse_args()
    protocol = joint_experiment_protocol(args.experiment)
    config = OmegaConf.load(ROOT / protocol.CONFIG)
    report = dict(contract=protocol.validate_joint_dit_config(config), parameters=protocol.parameter_report(config),
                  runtime_hashing_enabled=False)
    if args.assets:
        from scripts.validate_unified_baseline import (
            _require_file, _load_image_cache, _audit_imagenet_manifest, _validate_text_index)
        image = config.dataset.params.image
        val = image.validation
        val_manifest = _audit_imagenet_manifest(Path(val.manifest_jsonl), expected_split="val",
                                               expected_records=50000, retain_identities=True)
        train_manifest = _audit_imagenet_manifest(Path(image.manifest_jsonl), expected_split="train",
            expected_records=1281167, forbidden_identities=val_manifest["identities"])
        for split, asset in (("train", image), ("val", val)):
            for field in ("cache_path", "manifest_jsonl", "caption_jsonl", "synthetic_text_index_manifest"):
                _require_file(asset[field], f"{split} {field}")
            _load_image_cache(Path(asset.cache_path), expected_records=asset.expected_records)
            _validate_text_index(Path(asset.synthetic_text_index_manifest), split=split, records=asset.expected_records)
        shards = glob.glob(config.dataset.params.sources.climbmix.shard_glob)
        if len(shards) != 100:
            raise ValueError(f"Expected 100 ClimbMix shards, got {len(shards)}")
        for name in ("config.json", "model.safetensors", "tokenizer.json"):
            _require_file(str(Path(config.model.model_path) / name), f"Qwen {name}")
        generation = config.experiment.validation_generation
        _require_file(generation.prompt_file, "validation prompts")
        _require_file(str(Path(generation.vae_module_root) / "models/vae.py"), "validation VAE module")
        _require_file(generation.vae_path, "validation VAE checkpoint")
        from utils.training_image_generation import TrainingImageGenerator
        prompts = TrainingImageGenerator(config).prompts
        report["assets"] = dict(train_records=train_manifest["records"], val_records=val_manifest["records"],
                                image_identity_overlap=0, climbmix_shards=len(shards),
                                validation_image_prompts=len(prompts))
    if args.require_npu_count:
        import torch
        import torch_npu  # noqa: F401
        if not torch.npu.is_available() or torch.npu.device_count() != args.require_npu_count:
            raise RuntimeError(f"Expected {args.require_npu_count} available NPUs")
        report["npu_count"] = torch.npu.device_count()
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
