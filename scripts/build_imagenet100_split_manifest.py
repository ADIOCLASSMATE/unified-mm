#!/usr/bin/env python3
"""Freeze the existing flow-cache ImageNet-100 split as image-id assignments.

The original flow loader shuffles per-class *cache indices*.  The packed flow
cache order is not the same as the sorted subset manifest order, so applying
the same random seed to another cache can silently select different validation
images.  This script records the authoritative membership once and lets every
architecture, real-stat cache, and evaluator join it by ``img_id``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch

from utils.dataset_imagenet_flow_cache import _build_split_indices


class SplitReferenceDataset:
    def __init__(self, image_ids: torch.Tensor, synsets: dict[int, str]):
        self.img_ids = image_ids.long().contiguous()
        self.synsets = synsets

    def __len__(self) -> int:
        return int(self.img_ids.numel())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path):
    rows = {}
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            image_id = int(row["img_id"])
            if image_id in rows:
                raise ValueError(f"duplicate img_id={image_id} in {path}")
            rows[image_id] = row
    return rows


def atomic_write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reference_cache",
        default=(
            "public/datasets/imagenet_ablation_100c_balanced/"
            "vae_latents_mar_kl16/flow_latents_100c_1250pc_fp16.pt"
        ),
    )
    parser.add_argument(
        "--manifest",
        default="public/datasets/imagenet_ablation_100c_balanced/manifest.jsonl",
    )
    parser.add_argument(
        "--output",
        default=(
            "public/datasets/imagenet_ablation_100c_balanced/"
            "split_seed42_val100.jsonl"
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_samples_per_class", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    reference_cache = Path(args.reference_cache)
    manifest_path = Path(args.manifest)
    output_path = Path(args.output)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"{output_path} exists; pass --overwrite to replace it")

    payload = torch.load(reference_cache, map_location="cpu", mmap=True)
    image_ids = payload.get("img_ids", payload.get("image_ids"))
    if image_ids is None:
        raise ValueError(f"{reference_cache} has no img_ids/image_ids tensor")
    image_ids = torch.as_tensor(image_ids).long().contiguous()
    manifest = load_manifest(manifest_path)
    if image_ids.numel() != len(manifest):
        raise ValueError(
            f"cache/manifest size mismatch: {image_ids.numel()} vs {len(manifest)}"
        )
    if set(image_ids.tolist()) != set(manifest):
        raise ValueError("cache and manifest contain different image-id sets")

    synsets = {image_id: str(row["synset"]) for image_id, row in manifest.items()}
    reference = SplitReferenceDataset(image_ids, synsets)
    train_indices, val_indices = _build_split_indices(
        dataset=reference,
        val_ratio=0.08,
        seed=int(args.seed),
        strategy="stratified",
        val_samples_per_class=int(args.val_samples_per_class),
    )
    assignments = []
    for split, indices in (("train", train_indices), ("validation", val_indices)):
        for split_index, cache_index in enumerate(indices):
            image_id = int(image_ids[int(cache_index)].item())
            assignments.append(
                {
                    "img_id": image_id,
                    "synset": synsets[image_id],
                    "split": split,
                    "split_index": int(split_index),
                }
            )
    atomic_write_jsonl(output_path, assignments)
    metadata = {
        "format": "imagenet100-explicit-split-v1",
        "reference_cache": str(reference_cache),
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": sha256_file(manifest_path),
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
        "seed": int(args.seed),
        "val_samples_per_class": int(args.val_samples_per_class),
        "train_samples": len(train_indices),
        "validation_samples": len(val_indices),
        "classes": len({synsets[int(image_ids[index])] for index in val_indices}),
    }
    metadata_path = output_path.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
