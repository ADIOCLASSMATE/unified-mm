#!/usr/bin/env python3
"""Validate the B + SigLIP training contract and B's shared assets without content hashing."""
import argparse
import glob
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from omegaconf import OmegaConf
from utils.b_siglip_protocol import validate_b_siglip_config, parameter_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--assets", action="store_true")
    parser.add_argument("--require-npu-count", type=int, default=0)
    args = parser.parse_args()
    os.chdir(ROOT)
    config = OmegaConf.load(args.config)
    report = {"schema": "unified_b_siglip_preflight_v1", "contract": validate_b_siglip_config(config),
              "parameters": parameter_report(config), "runtime_hashing_enabled": False}
    if args.assets:
        from scripts.validate_unified_baseline import (
            _require_file, _load_image_cache, _audit_imagenet_manifest, _validate_text_index)
        image = config.dataset.params.image
        files = []
        val = image.validation
        val_manifest = _audit_imagenet_manifest(Path(val.manifest_jsonl), expected_split="val",
                                               expected_records=50000, retain_identities=True)
        train_manifest = _audit_imagenet_manifest(Path(image.manifest_jsonl), expected_split="train",
            expected_records=1281167, forbidden_identities=val_manifest["identities"])
        for split, asset in [("train", image), ("val", val)]:
            for field in ("cache_path", "manifest_jsonl", "caption_jsonl", "synthetic_text_index_manifest"):
                p = _require_file(asset[field], f"{split} {field}")
                files.append({"path": str(p), "bytes": p.stat().st_size})
            _load_image_cache(Path(asset.cache_path), expected_records=asset.expected_records)
            _validate_text_index(Path(asset.synthetic_text_index_manifest), split=split, records=asset.expected_records)
        shards = sorted(glob.glob(config.dataset.params.sources.climbmix.shard_glob))
        if len(shards) != 100:
            raise ValueError(f"Expected 100 ClimbMix shards, got {len(shards)}")
        files.extend({"path": p, "bytes": Path(p).stat().st_size} for p in shards)
        for name in ("config.json", "model.safetensors", "tokenizer.json"):
            p = _require_file(str(Path(config.model.model_path) / name), f"Qwen {name}")
            files.append({"path": str(p), "bytes": p.stat().st_size})
        if config.model.architecture_variant == "selfless_siglip":
            from safetensors import safe_open
            p = _require_file(str(Path(config.model.b_siglip_path) / "model.safetensors"), "SigLIP")
            if p.stat().st_size != 3511950624:
                raise ValueError("SigLIP file length differs from the verified official checkpoint")
            with safe_open(str(p), framework="pt", device="cpu") as f:
                if f.get_slice("vision_model.embeddings.position_embedding.weight").get_shape() != [729, 1152]:
                    raise ValueError("SigLIP position shape mismatch")
                for index in range(26):
                    if f.get_slice(f"vision_model.encoder.layers.{index}.self_attn.q_proj.weight").get_shape() != [1152,1152]:
                        raise ValueError(f"SigLIP vision block {index} is incompatible")
            files.append({"path": str(p), "bytes": p.stat().st_size})
        report["assets"] = {"files": files, "climbmix_shards": len(shards),
            "train_records": train_manifest["records"], "val_records": val_manifest["records"],
            "image_identity_overlap": 0}
    if args.require_npu_count:
        import torch
        import torch_npu
        count = torch.npu.device_count()
        if not torch.npu.is_available() or count != args.require_npu_count:
            raise RuntimeError(f"Expected {args.require_npu_count} NPUs, got {count}")
        report["npu_count"] = count
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
