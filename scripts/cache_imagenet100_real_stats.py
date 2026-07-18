#!/usr/bin/env python3
"""Cache original-ImageNet Inception moments for the shared 100C protocol."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_qwen_showo_fid_is import (  # noqa: E402
    DEFAULT_INCEPTION_WEIGHTS,
    DEFAULT_MANIFEST,
    DEFAULT_REAL_STATS,
    DEFAULT_SPLIT_MANIFEST,
    DEFAULT_SYNSET_MAPPING,
    FeatureMoments,
    build_expected_real_metadata,
    build_inception_extractor,
    build_metric_transform,
    distributed_barrier,
    extract_inception_features,
    feature_metadata,
    init_distributed,
    load_manifest,
    load_fixed_val_records,
    load_synset_names,
    metric_transform_metadata,
    resolve_original_image_path,
    validate_protocol_settings,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cache FID real statistics from the original ImageNet images selected "
            "by the fixed balanced ImageNet-100 validation split."
        )
    )
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--split_manifest", default=str(DEFAULT_SPLIT_MANIFEST))
    parser.add_argument("--synset_mapping", default=str(DEFAULT_SYNSET_MAPPING))
    parser.add_argument(
        "--imagenet_train_dir",
        default="/inspire/dataset/imagenet/v1/ILSVRC/Data/CLS-LOC/train",
    )
    parser.add_argument("--output", default=str(DEFAULT_REAL_STATS))
    parser.add_argument(
        "--inception_weights_path",
        default=str(DEFAULT_INCEPTION_WEIGHTS) if DEFAULT_INCEPTION_WEIGHTS.exists() else "",
        required=not DEFAULT_INCEPTION_WEIGHTS.exists(),
        help=(
            "Local torch-fidelity Inception weights. The file content hash is "
            "recorded and enforced by every evaluator."
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--fid_feature", type=int, default=2048)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--val_samples_per_class", type=int, default=100)
    parser.add_argument("--no_progress", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


class OriginalImageDataset:
    def __init__(self, records, imagenet_train_dir, transform):
        self.records = list(records)
        self.imagenet_train_dir = Path(imagenet_train_dir)
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        from PIL import Image

        record = self.records[index]
        path = resolve_original_image_path(record, self.imagenet_train_dir)
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, int(record["evaluation_index"]), str(path)


def atomic_torch_save(payload, output: Path) -> None:
    import torch

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(output)


def main() -> None:
    import torch
    import torch.distributed as dist
    from torch.utils.data import DataLoader, Subset
    from tqdm.auto import tqdm

    args = parse_args()
    validate_protocol_settings(
        split_seed=args.split_seed,
        val_samples_per_class=args.val_samples_per_class,
        image_size=args.image_size,
        fid_feature=args.fid_feature,
    )
    distributed, rank, world_size, local_rank, device = init_distributed(args.device)
    is_main = rank == 0
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"{output} already exists; pass --overwrite to replace it")

    manifest_records = load_manifest(args.manifest)
    selected_records = load_fixed_val_records(
        manifest_records,
        args.split_manifest,
        load_synset_names(args.synset_mapping),
        expected_samples_per_class=args.val_samples_per_class,
    )
    if len(selected_records) != 10_000:
        raise ValueError(
            f"shared ImageNet-100 protocol requires 10,000 real images, "
            f"found {len(selected_records)}"
        )
    metadata = build_expected_real_metadata(
        manifest_path=args.manifest,
        split_manifest_path=args.split_manifest,
        selected_records=selected_records,
        transform=metric_transform_metadata(args.image_size),
        feature=feature_metadata(args.fid_feature, args.inception_weights_path),
        val_samples_per_class=args.val_samples_per_class,
        split_seed=args.split_seed,
    )

    dataset = OriginalImageDataset(
        selected_records,
        args.imagenet_train_dir,
        build_metric_transform(args.image_size),
    )
    local_indices = [
        index
        for index, record in enumerate(selected_records)
        if int(record["evaluation_index"]) % world_size == rank
    ]
    loader = DataLoader(
        Subset(dataset, local_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=args.num_workers > 0,
    )
    inception = build_inception_extractor(
        args.fid_feature, args.inception_weights_path, device
    )
    moments = FeatureMoments.zeros(args.fid_feature, device)
    iterator = tqdm(
        loader,
        total=math.ceil(len(local_indices) / args.batch_size),
        disable=args.no_progress or not is_main,
        desc="original ImageNet real stats",
        dynamic_ncols=True,
    )
    with torch.inference_mode():
        for images, _, _ in iterator:
            features, _ = extract_inception_features(
                inception, images.to(device, non_blocking=True)
            )
            moments.update(features)
    moments.all_reduce_()
    distributed_barrier(distributed, device)

    if is_main:
        count = int(moments.count.item())
        if count != len(selected_records):
            raise RuntimeError(
                f"distributed feature count={count}; expected={len(selected_records)}"
            )
        mean, covariance = moments.mean_cov()
        payload = {
            "metadata": metadata,
            "stats": {
                "count": count,
                "sum": moments.sum.cpu(),
                "outer_sum": moments.outer_sum.cpu(),
                # Convenience fields for other architecture evaluators.
                "mean": mean.cpu(),
                "covariance": covariance.cpu(),
            },
        }
        atomic_torch_save(payload, output)
        sidecar = output.with_suffix(output.suffix + ".metadata.json")
        write_json(sidecar, metadata)
        print(
            json.dumps(
                {
                    "output": str(output.resolve()),
                    "metadata": metadata,
                    "count": count,
                },
                indent=2,
                sort_keys=True,
            )
        )

    distributed_barrier(distributed, device)
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
