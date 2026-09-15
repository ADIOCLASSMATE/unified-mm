#!/usr/bin/env python3
"""Audit the B-matched joint-image recipe and its local assets without hashing."""

import argparse
import json
import os
from pathlib import Path
import sys

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.image_joint_training import CONFIG, validate_config
from scripts.validate_unified_single_source import (
    _audit_imagenet_assets, _audit_model_assets, _tokenizer_probe,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-npu-count", type=int, choices=(0, 16), default=0)
    parser.add_argument("--tokenizer-probe", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    config = OmegaConf.load(CONFIG)
    report = {"contract": validate_config(config), "runtime_hashing_enabled": False}
    report["model"] = _audit_model_assets(config)
    report["images"] = _audit_imagenet_assets(config)
    if args.tokenizer_probe:
        report["tokenizer"] = [_tokenizer_probe(config, source=source, world_size=32)
                               for source in ("t2i", "i2t")]
        if any(row["max_serialized_length"] > 512 for row in report["tokenizer"]):
            raise ValueError("tokenizer probe exceeds the padded sequence length")
    if args.require_npu_count:
        import torch
        import torch_npu  # noqa: F401
        if not torch.npu.is_available() or torch.npu.device_count() != args.require_npu_count:
            raise RuntimeError("expected exactly 16 available Ascend devices on this node")
        report["local_npu_count"] = torch.npu.device_count()
    print(json.dumps({"passed": True, **report}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
