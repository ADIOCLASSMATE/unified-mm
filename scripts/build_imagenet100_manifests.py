#!/usr/bin/env python3
"""Build the canonical ImageNet and ImageNet-100 manifests without VAE encoding.

The ImageNet-100 membership is the historical Selfless-Flow selection: 100
stratified classes selected with seed 42, followed by a single seed-43 torch
generator used to shuffle each selected class and keep 1,250 images.  The
train/validation assignment is reconstructed from that selected-image order;
sorting the image ids before applying the split RNG would change membership.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import Counter, OrderedDict
from collections.abc import Iterable
from pathlib import Path

import torch

CANONICAL_FULL_SHA256 = (
    "9d165263e8cf4ba6d537d084a8cc3b87af2eaf5ef9a5b59e1360a6228c840759"
)
CANONICAL_SUBSET_SHA256 = (
    "6c6fc84e6ec9cb8b92421659d44abbb24d1cc34558213d9fb9db7e4ec44f1c3a"
)
CANONICAL_SPLIT_SHA256 = (
    "02e5c67c058f95bcca46c82f3c1fc81086f61dcec62ce25049843f09d930a5ba"
)
CANONICAL_SOURCE_ROOT = Path("/inspire/dataset/imagenet/v1/ILSVRC/Data/CLS-LOC/train")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_jsonl(path: Path, rows: Iterable[dict], *, sort_keys: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=sort_keys) + "\n")
    os.replace(temporary, path)


def scan_imagenet(
    train_dir: Path,
    canonical_source_root: Path,
) -> tuple[list[dict], OrderedDict[str, list[int]]]:
    if not train_dir.is_dir():
        raise FileNotFoundError(train_dir)
    rows: list[dict] = []
    by_synset: OrderedDict[str, list[int]] = OrderedDict()
    image_id = 0
    for class_dir in sorted(path for path in train_dir.iterdir() if path.is_dir()):
        synset = class_dir.name
        image_ids: list[int] = []
        for image_path in sorted(
            path for path in class_dir.iterdir() if path.is_file()
        ):
            image_id += 1
            image_ids.append(image_id)
            rows.append(
                {
                    "img_id": image_id,
                    # This logical path is part of the frozen manifest contract.
                    # Consumers resolve the final synset/filename against the
                    # current ImageNet root and do not require this prefix to exist.
                    "source_path": str(
                        canonical_source_root / synset / image_path.name
                    ),
                    "synset": synset,
                }
            )
        by_synset[synset] = image_ids
    return rows, by_synset


def select_membership(
    by_synset: OrderedDict[str, list[int]],
    *,
    num_classes: int,
    samples_per_class: int,
    class_seed: int,
    member_seed: int,
) -> tuple[list[str], list[int]]:
    eligible = [
        synset
        for synset, image_ids in by_synset.items()
        if len(image_ids) >= samples_per_class
    ]
    if num_classes > len(eligible):
        raise ValueError(
            f"requested {num_classes} classes, but only {len(eligible)} contain "
            f"at least {samples_per_class} images"
        )

    class_generator = torch.Generator().manual_seed(class_seed)
    selected_synsets: list[str] = []
    for bin_index in range(num_classes):
        start = (bin_index * len(eligible)) // num_classes
        end = ((bin_index + 1) * len(eligible)) // num_classes
        offset = int(
            torch.randint(
                end - start,
                (1,),
                generator=class_generator,
            ).item()
        )
        selected_synsets.append(eligible[start + offset])

    member_generator = torch.Generator().manual_seed(member_seed)
    selected_image_ids: list[int] = []
    for synset in selected_synsets:
        image_ids = torch.tensor(by_synset[synset], dtype=torch.long)
        order = torch.randperm(image_ids.numel(), generator=member_generator)
        selected_image_ids.extend(
            int(value) for value in image_ids[order][:samples_per_class].tolist()
        )
    return selected_synsets, selected_image_ids


def build_split_rows(
    selected_image_ids: list[int],
    synset_by_image_id: dict[int, str],
    *,
    seed: int,
    validation_per_class: int,
) -> list[dict]:
    groups: dict[str, list[int]] = {}
    for cache_index, image_id in enumerate(selected_image_ids):
        groups.setdefault(synset_by_image_id[image_id], []).append(cache_index)

    rng = random.Random(seed)
    train_indices: list[int] = []
    validation_indices: list[int] = []
    for synset in sorted(groups):
        group_indices = list(groups[synset])
        rng.shuffle(group_indices)
        validation_indices.extend(group_indices[:validation_per_class])
        train_indices.extend(group_indices[validation_per_class:])
    rng.shuffle(train_indices)
    rng.shuffle(validation_indices)

    rows: list[dict] = []
    for split, indices in (
        ("train", train_indices),
        ("validation", validation_indices),
    ):
        for split_index, cache_index in enumerate(indices):
            image_id = selected_image_ids[cache_index]
            rows.append(
                {
                    "img_id": image_id,
                    "synset": synset_by_image_id[image_id],
                    "split": split,
                    "split_index": split_index,
                }
            )
    return rows


def require_digest(path: Path, expected: str, label: str) -> str:
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(
            f"{label} SHA256 mismatch: expected={expected}, actual={actual}, path={path}"
        )
    return actual


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--imagenet_train_dir",
        default="public/dataset/imagenet/v1/ILSVRC/Data/CLS-LOC/train",
    )
    parser.add_argument(
        "--canonical_source_root",
        default=str(CANONICAL_SOURCE_ROOT),
        help="Frozen logical source_path prefix used by the canonical manifest.",
    )
    parser.add_argument(
        "--full_manifest",
        default="public/datasets/imagenet_full/manifest.jsonl",
    )
    parser.add_argument(
        "--output_dir",
        default="public/datasets/imagenet_ablation_100c_balanced",
    )
    parser.add_argument("--num_classes", type=int, default=100)
    parser.add_argument("--samples_per_class", type=int, default=1250)
    parser.add_argument("--validation_per_class", type=int, default=100)
    parser.add_argument("--class_seed", type=int, default=42)
    parser.add_argument("--member_seed", type=int, default=43)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--no_require_canonical_hashes",
        action="store_true",
        help="Allow non-canonical inputs for unit tests or diagnostics.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_dir = Path(args.imagenet_train_dir)
    full_manifest = Path(args.full_manifest)
    output_dir = Path(args.output_dir)
    subset_manifest = output_dir / "manifest.jsonl"
    split_manifest = output_dir / "split_seed42_val100.jsonl"
    metadata_path = output_dir / "membership_metadata.json"
    classes_path = output_dir / "classes.txt"
    outputs = (
        full_manifest,
        subset_manifest,
        split_manifest,
        metadata_path,
        classes_path,
    )
    existing = [path for path in outputs if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "outputs already exist; pass --overwrite after verifying the targets: "
            + ", ".join(str(path) for path in existing)
        )

    rows, by_synset = scan_imagenet(
        train_dir,
        Path(args.canonical_source_root),
    )
    if len(by_synset) != 1000 or len(rows) != 1_281_167:
        raise RuntimeError(
            "ImageNet source contract failed: "
            f"classes={len(by_synset)}, images={len(rows)}"
        )

    selected_synsets, selected_image_ids = select_membership(
        by_synset,
        num_classes=args.num_classes,
        samples_per_class=args.samples_per_class,
        class_seed=args.class_seed,
        member_seed=args.member_seed,
    )
    selected_set = set(selected_image_ids)
    subset_rows = [row for row in rows if int(row["img_id"]) in selected_set]
    synset_by_image_id = {int(row["img_id"]): str(row["synset"]) for row in subset_rows}
    split_rows = build_split_rows(
        selected_image_ids,
        synset_by_image_id,
        seed=args.split_seed,
        validation_per_class=args.validation_per_class,
    )

    expected_subset_size = args.num_classes * args.samples_per_class
    if (
        len(subset_rows) != expected_subset_size
        or len(selected_set) != expected_subset_size
    ):
        raise RuntimeError(
            f"membership size mismatch: rows={len(subset_rows)}, unique={len(selected_set)}"
        )
    membership_counts = Counter(row["synset"] for row in subset_rows)
    if set(membership_counts.values()) != {args.samples_per_class}:
        raise RuntimeError(f"unbalanced membership counts: {membership_counts}")
    split_counts = Counter(row["split"] for row in split_rows)
    expected_validation = args.num_classes * args.validation_per_class
    if split_counts != Counter(
        {
            "train": expected_subset_size - expected_validation,
            "validation": expected_validation,
        }
    ):
        raise RuntimeError(f"split size mismatch: {dict(split_counts)}")

    atomic_write_jsonl(full_manifest, rows, sort_keys=False)
    atomic_write_jsonl(subset_manifest, subset_rows, sort_keys=False)
    atomic_write_jsonl(split_manifest, split_rows, sort_keys=True)

    digests = {
        "full_manifest_sha256": sha256_file(full_manifest),
        "subset_manifest_sha256": sha256_file(subset_manifest),
        "split_manifest_sha256": sha256_file(split_manifest),
    }
    canonical = not args.no_require_canonical_hashes
    if canonical:
        require_digest(full_manifest, CANONICAL_FULL_SHA256, "full manifest")
        require_digest(subset_manifest, CANONICAL_SUBSET_SHA256, "subset manifest")
        require_digest(split_manifest, CANONICAL_SPLIT_SHA256, "split manifest")

    output_dir.mkdir(parents=True, exist_ok=True)
    classes_temporary = classes_path.with_suffix(classes_path.suffix + ".tmp")
    with classes_temporary.open("w", encoding="utf-8") as handle:
        for synset in selected_synsets:
            handle.write(f"{synset}\t{membership_counts[synset]}\n")
    os.replace(classes_temporary, classes_path)

    metadata = {
        "format": "imagenet100-canonical-membership-v1",
        "imagenet_train_dir": str(train_dir),
        "canonical_source_root": str(args.canonical_source_root),
        "full_manifest": str(full_manifest),
        "subset_manifest": str(subset_manifest),
        "split_manifest": str(split_manifest),
        "full_images": len(rows),
        "eligible_classes": sum(
            len(image_ids) >= args.samples_per_class for image_ids in by_synset.values()
        ),
        "selected_classes": selected_synsets,
        "selected_images": len(subset_rows),
        "train_images": split_counts["train"],
        "validation_images": split_counts["validation"],
        "class_seed": args.class_seed,
        "member_seed": args.member_seed,
        "split_seed": args.split_seed,
        "samples_per_class": args.samples_per_class,
        "validation_per_class": args.validation_per_class,
        "canonical_hashes_required": canonical,
        **digests,
    }
    temporary = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, metadata_path)
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
