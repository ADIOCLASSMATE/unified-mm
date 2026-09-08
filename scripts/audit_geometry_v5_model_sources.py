"""Persist actual loaded parameter scopes, source revisions and B/F training differences."""

import argparse
import json
import math
import tarfile
from collections import defaultdict
from pathlib import Path

import torch
import yaml
from safetensors import safe_open

from scripts.prepare_cross_model_geometry_v5 import BASELINE, F_RUN
from scripts.prepare_geometry_v5_assets import MODELS, RUN, emit, write_json


def read(path):
    return json.loads(Path(path).read_text())


def shape_counts(directory):
    total, tensors, groups = 0, 0, defaultdict(int)
    for file in sorted(directory.glob("*.safetensors")):
        with safe_open(file, framework="pt") as value:
            for name in value.keys():  # noqa: SIM118 (safetensors reader, not a dict)
                count = math.prod(value.get_slice(name).get_shape())
                total += count
                tensors += 1
                prefix = ".".join(name.split(".")[:2])
                groups[prefix] += count
    assert total > 0
    return {
        "stored_elements": total,
        "stored_tensors": tensors,
        "components": dict(groups),
    }


def differences(a, b, prefix=""):
    if isinstance(a, dict) and isinstance(b, dict):
        return [
            r
            for key in sorted(a.keys() | b.keys())
            for r in differences(a.get(key), b.get(key), f"{prefix}.{key}".lstrip("."))
        ]
    return [] if a == b else [{"field": prefix, "b": a, "f": b}]


def main(root):
    assets = read(root / "asset-manifest.json")
    source_archives = []
    for file in assets["files"]:
        if file["bytes"] is not None:
            assert Path(file["path"]).stat().st_size == file["bytes"], file["path"]
        else:
            assert file["name"] == "source.tar.gz"
            count = 0
            with tarfile.open(file["path"]) as archive:
                for member in archive.getmembers():
                    if not member.isfile():
                        continue
                    relative = Path(*Path(member.name).parts[1:])
                    assert not relative.is_absolute() and ".." not in relative.parts
                    actual = Path(file["extract_to"]) / relative
                    with archive.extractfile(member) as content:
                        assert actual.read_bytes() == content.read(), actual
                    count += 1
            source_archives.append(
                {
                    "path": file["path"],
                    "bytes": Path(file["path"]).stat().st_size,
                    "original_files_exact": count,
                    "published_size": None,
                }
            )
    records = {}
    configs = {}
    for model, base in (("b", BASELINE), ("f", F_RUN)):
        path = base / "hf_model-final-ema"
        metadata = read(path / "ema_export_metadata.json")
        count = shape_counts(path)
        assert count["stored_tensors"] == metadata["stored_weight_key_count"]
        assert (
            metadata["source_global_step"] == 95415
            and metadata["source_world_size"] == 64
        )
        configs[model] = yaml.safe_load((base / "config.yaml").read_text())
        config = read(path / "config.json")
        records[model] = {
            "source": str(path),
            "configuration": config,
            "ema": metadata,
            **count,
        }
    assert (
        records["f"]["configuration"]["architecture_variant"]
        == "positionwise_flow_head_on_b"
    )
    assert (
        records["b"]["configuration"]["hidden_size"]
        == records["f"]["configuration"]["hidden_size"]
        == 1024
    )
    assert (
        records["b"]["configuration"]["mask_token_id"]
        != records["b"]["configuration"]["image_mask_token_id"]
    )
    # Compare all dataset/optimizer/scheduler/training fields, not just model names.
    training_differences = {
        key: differences(configs["b"][key], configs["f"][key], key)
        for key in ("model", "dataset", "optimizer", "lr_scheduler", "training")
    }
    for name in ("qwen_text", "dinov2", "mae", "siglip", "janusflow", "showo2"):
        verification = read(root / "model-verification" / f"{name}.json")
        contract = read(root / "adapter-contracts" / f"{name}.json")
        assert (
            verification["contract"] == contract
            and verification["parameters_verified"] > 0
        )
        records[name] = {
            "source": contract["source_path"],
            "loaded_parameter_elements_named": verification["parameter_elements"],
            "parameters_verified": verification["parameters_verified"],
            "precision": contract["precision"],
            "storage_dtype": contract.get("storage_dtype", "bfloat16"),
            "contract": str(root / "adapter-contracts" / f"{name}.json"),
            "verification": str(root / "model-verification" / f"{name}.json"),
        }
    source = torch.load(
        MODELS / "showlab--show-o2-1.5B/pytorch_model.bin",
        weights_only=True,
        mmap=True,
        map_location="cpu",
    )
    storage, unique, named = {}, 0, 0
    for name, value in source.items():
        assert isinstance(value, torch.Tensor)
        # Pointer identities are used locally only; no persistent addresses/hashes.
        identity = (
            value.untyped_storage().data_ptr(),
            value.storage_offset(),
            tuple(value.shape),
            tuple(value.stride()),
        )
        if identity not in storage:
            unique += value.numel()
        storage.setdefault(identity, []).append(name)
        named += value.numel()
    aliases = [names for names in storage.values() if len(names) > 1]
    assert aliases == [["showo.model.embed_tokens.weight", "showo.lm_head.weight"]]
    assert named == records["showo2"]["loaded_parameter_elements_named"]
    records["showo2"].update(
        unique_source_tensor_view_elements=unique, shared_weight_names=aliases
    )
    for name in ("janusflow", "showo2"):
        value = read(root / "model-verification" / f"{name}-vae.json")
        assert value["parameters_verified"] > 0
        records[name]["separately_loaded_vae"] = value
    # DINO/MAE are encoders only; classifier/MAE decoder metadata is not their probe size.
    records["mae"]["parameter_scope"] = (
        "loaded unfinetuned MAE encoder only; all patches kept, decoder not used"
    )
    records["siglip"]["parameter_scope"] = (
        "both towers plus trained poolers/projections, not just the so400m visual name"
    )
    records["qwen_text"]["parameter_scope"] = (
        "original Qwen3-0.6B-Base; not B init reference or B-trained text weights"
    )
    result = {
        "schema": "geometry_v5_model_source_ledger_1",
        "status": "passed",
        "models": records,
        "asset_revisions": assets["models"],
        "official_code_revisions": assets["sources"],
        "download_files_checked_by_bytes": sum(
            f["bytes"] is not None for f in assets["files"]
        ),
        "source_archives_exact_content_verification": source_archives,
        "proxy_policy": assets["proxy_policy"],
        "hashing": "none for weights or data",
        "new_training_updates": 0,
        "b_f_training_differences": training_differences,
        "b_f_parameter_matched_head_relative_difference": abs(
            records["b"]["components"]["image_flow_head.net"]
            - records["f"]["components"]["image_flow_head.net"]
        )
        / records["b"]["components"]["image_flow_head.net"],
        "semantic_sources": {
            "b_f": {
                "initialization": "Qwen3 Base and MAR VAE",
                "training": "paired conditional ImageNet generation and ClimbMix; no claim of from-scratch or unpaired pure compression",
                "source_configs": [
                    str(BASELINE / "config.yaml"),
                    str(F_RUN / "config.yaml"),
                ],
            },
            "janusflow": {
                "teacher": "pretrained SigLIP understanding encoder; REPA on generation intermediate representations",
                "objectives": "autoregressive + rectified flow + representation alignment, followed by SFT",
                "source": "https://arxiv.org/html/2411.07975v1",
            },
            "showo2": {
                "teacher": "SigLIP-distilled VAE semantic branch, fused with a low-level branch",
                "initialization": "Qwen2.5-Instruct and Wan VAE",
                "objectives": "AR and flow matching with staged multimodal/instruction training",
                "source": "https://arxiv.org/html/2506.15564v2",
            },
            "siglip": {
                "objectives": "explicit paired image-text sigmoid alignment",
                "source": "https://arxiv.org/abs/2303.15343",
            },
            "dinov2": {
                "objectives": "self-supervised visual learning; smaller models distilled from a larger teacher",
                "source": "https://arxiv.org/abs/2304.07193",
            },
            "mae": {
                "objectives": "masked pixel reconstruction",
                "source": "https://arxiv.org/abs/2111.06377",
            },
            "qwen_text": {
                "objectives": "original causal language pretraining, no new visual connection",
                "source": "https://huggingface.co/Qwen/Qwen3-0.6B-Base",
            },
        },
        "interpretation": "External parameter/data/initialization/teacher/SFT budgets are unmatched; rankings cannot isolate architecture. F shares B architecture, not weights, and removes head cross-token interaction, not all head semantic computation.",
    }
    write_json(root / "audits/model-sources-and-training.json", result)
    emit(
        "v5_model_source_ledger_passed",
        models=len(records),
        showo_unique_elements=unique,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=RUN)
    main(parser.parse_args().output_dir)
