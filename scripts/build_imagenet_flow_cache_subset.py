#!/usr/bin/env python3
"""Build a balanced class subset from the full ImageNet flow-latent cache."""

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
        help="Full cache produced by scripts/imagenet_encode_kl16_vae.py.",
    )
    parser.add_argument(
        "--source_manifest",
        default="public/datasets/imagenet_full/manifest.jsonl",
        help="JSONL manifest with img_id and synset fields.",
    )
    parser.add_argument(
        "--output_dir",
        default="public/datasets/imagenet_subset/vae_latents_mar_kl16",
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
        "--class_selection",
        choices=["first", "random", "stratified"],
        default="first",
        help=(
            "How to select classes when --synsets/--synsets_file are not provided. "
            "'stratified' samples one class from each evenly spaced range of the manifest class order."
        ),
    )
    parser.add_argument(
        "--min_samples_per_class",
        type=int,
        default=1,
        help="Only automatically select classes with at least this many source samples.",
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
    parser.add_argument(
        "--mmap",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Memory-map the source torch cache instead of materializing the entire full-ImageNet tensor.",
    )
    return parser.parse_args()


def read_manifest(path: Path):
    by_synset = OrderedDict()
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            img_id = int(row["img_id"])
            synset = str(row["synset"])
            by_synset.setdefault(synset, []).append(img_id)
    return by_synset


def select_synsets(args, by_synset):
    if args.synsets_file:
        synsets = [
            line.strip()
            for line in Path(args.synsets_file).read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
    elif args.synsets:
        synsets = [s.strip() for s in args.synsets.split(",") if s.strip()]
    else:
        min_samples = max(1, int(args.min_samples_per_class))
        eligible = [
            synset
            for synset, img_ids in by_synset.items()
            if len(img_ids) >= min_samples
        ]
        num_classes = int(args.num_classes)
        if num_classes <= 0:
            raise ValueError(f"--num_classes must be positive, got {num_classes}")
        if num_classes > len(eligible):
            raise ValueError(
                f"Requested {num_classes} classes but only {len(eligible)} have at least "
                f"{min_samples} samples."
            )

        generator = torch.Generator().manual_seed(int(args.seed))
        if args.class_selection == "first":
            synsets = eligible[:num_classes]
        elif args.class_selection == "random":
            order = torch.randperm(len(eligible), generator=generator)[:num_classes]
            synsets = [eligible[int(idx)] for idx in order.tolist()]
        else:
            synsets = []
            for bin_idx in range(num_classes):
                start = (bin_idx * len(eligible)) // num_classes
                end = ((bin_idx + 1) * len(eligible)) // num_classes
                offset = int(torch.randint(end - start, (1,), generator=generator).item())
                synsets.append(eligible[start + offset])

    missing = [synset for synset in synsets if synset not in by_synset]
    if missing:
        raise ValueError(f"Requested synsets not found in manifest: {missing}")
    duplicates = [synset for synset in OrderedDict.fromkeys(synsets) if synsets.count(synset) > 1]
    if duplicates:
        raise ValueError(f"Requested synsets contain duplicates: {duplicates}")
    undersized = [
        (synset, len(by_synset[synset]))
        for synset in synsets
        if len(by_synset[synset]) < max(1, int(args.min_samples_per_class))
    ]
    if undersized:
        raise ValueError(
            "Requested synsets do not satisfy --min_samples_per_class: "
            + ", ".join(f"{synset}={count}" for synset, count in undersized)
        )
    return synsets


def write_selected_manifest(
    source_manifest: Path,
    output_manifest: Path,
    selected_synsets,
    selected_img_ids,
):
    selected_set = set(selected_synsets)
    selected_img_id_set = set(selected_img_ids)
    selected_counts = defaultdict(int)
    with source_manifest.open() as source, output_manifest.open("w") as output:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            img_id = int(row["img_id"])
            synset = str(row["synset"])
            if synset not in selected_set or img_id not in selected_img_id_set:
                continue
            output.write(line if line.endswith("\n") else line + "\n")
            selected_counts[synset] += 1
    return selected_counts


def main():
    args = parse_args()
    source_cache = Path(args.source_cache)
    source_manifest = Path(args.source_manifest)
    output_dir = Path(args.output_dir)
    if not source_cache.exists():
        raise FileNotFoundError(source_cache)
    if not source_manifest.exists():
        raise FileNotFoundError(source_manifest)

    by_synset = read_manifest(source_manifest)
    selected_synsets = select_synsets(args, by_synset)

    rng = torch.Generator().manual_seed(int(args.seed) + 1)
    selected_img_ids = []
    for synset in selected_synsets:
        img_ids = torch.tensor(by_synset[synset], dtype=torch.long)
        if args.shuffle_within_class:
            img_ids = img_ids[torch.randperm(img_ids.numel(), generator=rng)]
        if args.max_samples_per_class and args.max_samples_per_class > 0:
            img_ids = img_ids[: int(args.max_samples_per_class)]
        selected_img_ids.extend(int(x) for x in img_ids.tolist())

    obj = torch.load(source_cache, map_location="cpu", mmap=bool(args.mmap))
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
    source_metadata = obj.get("metadata")
    subset_metadata = dict(source_metadata) if isinstance(source_metadata, dict) else {}
    subset_metadata.update(
        {
            "num_images": len(selected_img_ids),
            "subset_num_classes": len(selected_synsets),
            "subset_samples_per_class": (
                int(args.max_samples_per_class)
                if args.max_samples_per_class and args.max_samples_per_class > 0
                else None
            ),
            "subset_class_selection": str(args.class_selection),
            "subset_seed": int(args.seed),
        }
    )
    subset["metadata"] = subset_metadata

    output_dir.mkdir(parents=True, exist_ok=True)
    output_name = args.output_name or f"flow_latents_{len(selected_synsets)}c_fp16.pt"
    output_cache = output_dir / output_name
    torch.save(subset, output_cache)

    output_manifest = output_dir.parent / "manifest.jsonl"
    selected_counts = write_selected_manifest(
        source_manifest=source_manifest,
        output_manifest=output_manifest,
        selected_synsets=selected_synsets,
        selected_img_ids=selected_img_ids,
    )

    classes_path = output_dir.parent / "classes.txt"
    with classes_path.open("w") as f:
        for synset in selected_synsets:
            f.write(f"{synset}\t{selected_counts[synset]}\n")

    total = int(subset["latents"].shape[0])
    if total != sum(selected_counts.values()):
        raise RuntimeError(
            f"Cache/manifest size mismatch: cache has {total}, manifest has {sum(selected_counts.values())}"
        )

    metadata_path = output_dir.parent / "subset_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "source_cache": str(source_cache),
                "source_manifest": str(source_manifest),
                "output_cache": str(output_cache),
                "output_manifest": str(output_manifest),
                "num_classes": len(selected_synsets),
                "num_samples": total,
                "class_selection": str(args.class_selection),
                "min_samples_per_class": int(args.min_samples_per_class),
                "max_samples_per_class": int(args.max_samples_per_class),
                "shuffle_within_class": bool(args.shuffle_within_class),
                "seed": int(args.seed),
                "mmap": bool(args.mmap),
                "latent_shape": list(subset["latents"].shape),
                "latent_dtype": str(subset["latents"].dtype),
                "classes": [
                    {
                        "synset": synset,
                        "available_samples": len(by_synset[synset]),
                        "selected_samples": selected_counts[synset],
                    }
                    for synset in selected_synsets
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Wrote {total} samples from {len(selected_synsets)} classes")
    print(f"Cache: {output_cache}")
    print(f"Manifest: {output_manifest}")
    print(f"Classes: {classes_path}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
