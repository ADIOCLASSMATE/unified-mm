#!/usr/bin/env python3
"""Build a small class subset from the full ImageNet flow-latent cache."""

import argparse
import json
from collections import OrderedDict, defaultdict
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source_cache",
        default="public/datasets/imagenet_full/vae_latents_mar_kl16/flow_latents_all_fp16.pt",
        help="Full cache produced by scripts/imagenet_encode_mar_kl16.py.",
    )
    parser.add_argument(
        "--source_manifest",
        default="public/datasets/imagenet_full/manifest.jsonl",
        help="JSONL manifest with img_id and synset fields.",
    )
    parser.add_argument(
        "--output_dir",
        default="public/datasets/imagenet_stage0_10c/vae_latents_mar_kl16",
        help="Directory for the subset cache and manifest.",
    )
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument(
        "--synsets",
        default="",
        help="Comma-separated synsets to keep. Overrides --num_classes when non-empty.",
    )
    parser.add_argument(
        "--synsets_file",
        default="",
        help="Optional file containing one synset per line. Overrides --num_classes when non-empty.",
    )
    parser.add_argument(
        "--max_samples_per_class",
        type=int,
        default=-1,
        help="Limit samples per selected class after optional class selection.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--shuffle_within_class",
        action="store_true",
        help="Shuffle examples inside each class before applying --max_samples_per_class.",
    )
    parser.add_argument(
        "--output_name",
        default="",
        help="Output cache filename. Defaults to flow_latents_{num_classes}c_fp16.pt.",
    )
    return parser.parse_args()


def read_manifest(path: Path):
    rows = []
    by_synset = OrderedDict()
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            img_id = int(row["img_id"])
            synset = str(row["synset"])
            rows.append(row)
            by_synset.setdefault(synset, []).append(img_id)
    return rows, by_synset


def read_synset_list(args, by_synset):
    if args.synsets_file:
        synsets = [
            line.strip()
            for line in Path(args.synsets_file).read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
    elif args.synsets:
        synsets = [s.strip() for s in args.synsets.split(",") if s.strip()]
    else:
        synsets = list(by_synset.keys())[: int(args.num_classes)]

    missing = [synset for synset in synsets if synset not in by_synset]
    if missing:
        raise ValueError(f"Requested synsets not found in manifest: {missing}")
    return synsets


def main():
    args = parse_args()
    source_cache = Path(args.source_cache)
    source_manifest = Path(args.source_manifest)
    output_dir = Path(args.output_dir)
    if not source_cache.exists():
        raise FileNotFoundError(source_cache)
    if not source_manifest.exists():
        raise FileNotFoundError(source_manifest)

    rows, by_synset = read_manifest(source_manifest)
    selected_synsets = read_synset_list(args, by_synset)
    selected_set = set(selected_synsets)

    rng = torch.Generator().manual_seed(int(args.seed))
    selected_img_ids = []
    for synset in selected_synsets:
        img_ids = torch.tensor(by_synset[synset], dtype=torch.long)
        if args.shuffle_within_class:
            img_ids = img_ids[torch.randperm(img_ids.numel(), generator=rng)]
        if args.max_samples_per_class and args.max_samples_per_class > 0:
            img_ids = img_ids[: int(args.max_samples_per_class)]
        selected_img_ids.extend(int(x) for x in img_ids.tolist())
    selected_img_id_set = set(selected_img_ids)

    obj = torch.load(source_cache, map_location="cpu")
    latents = obj["latents"]
    img_ids = obj.get("img_ids", torch.arange(latents.shape[0]))
    img_id_to_index = {int(img_id): idx for idx, img_id in enumerate(img_ids.tolist())}
    missing_img_ids = [img_id for img_id in selected_img_ids if img_id not in img_id_to_index]
    if missing_img_ids:
        raise ValueError(f"Selected img_ids missing from cache: {missing_img_ids[:10]}")

    indices = torch.tensor([img_id_to_index[img_id] for img_id in selected_img_ids], dtype=torch.long)
    subset = {}
    for key, value in obj.items():
        if torch.is_tensor(value) and value.shape[:1] == latents.shape[:1]:
            subset[key] = value.index_select(0, indices).contiguous()
        else:
            subset[key] = value
    subset["img_ids"] = torch.tensor(selected_img_ids, dtype=img_ids.dtype)
    subset["selected_synsets"] = list(selected_synsets)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_name = args.output_name or f"flow_latents_{len(selected_synsets)}c_fp16.pt"
    output_cache = output_dir / output_name
    torch.save(subset, output_cache)

    selected_counts = defaultdict(int)
    output_manifest = output_dir.parent / "manifest.jsonl"
    with output_manifest.open("w") as f:
        for row in rows:
            img_id = int(row["img_id"])
            synset = str(row["synset"])
            if synset not in selected_set or img_id not in selected_img_id_set:
                continue
            f.write(json.dumps(row) + "\n")
            selected_counts[synset] += 1

    classes_path = output_dir.parent / "classes.txt"
    with classes_path.open("w") as f:
        for synset in selected_synsets:
            f.write(f"{synset}\t{selected_counts[synset]}\n")

    total = int(subset["latents"].shape[0])
    print(f"Wrote {total} samples from {len(selected_synsets)} classes")
    print(f"Cache: {output_cache}")
    print(f"Manifest: {output_manifest}")
    print(f"Classes: {classes_path}")


if __name__ == "__main__":
    main()
