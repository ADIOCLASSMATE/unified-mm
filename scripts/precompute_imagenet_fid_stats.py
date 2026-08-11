#!/usr/bin/env python3
"""Precompute deterministic ImageNet-subset Inception moments."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

import torch
import torch.distributed as dist
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_single_stream_fid_is import (
    build_real_image_transform,
    file_sha256,
    init_distributed,
)
from scripts.image_evaluation_metrics import (
    FeatureMoments,
    build_inception_extractor,
    extract_inception_features,
)


class RankStrideSampler(Sampler[int]):
    def __init__(self, size: int, rank: int, world_size: int):
        self.size = int(size)
        self.rank = int(rank)
        self.world_size = int(world_size)

    def __iter__(self):
        return iter(range(self.rank, self.size, self.world_size))

    def __len__(self) -> int:
        if self.rank >= self.size:
            return 0
        return (self.size - 1 - self.rank) // self.world_size + 1


class ImageDataset(Dataset):
    def __init__(self, paths: list[Path], image_size: int):
        self.paths = paths
        self.transform = build_real_image_transform(image_size)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        with Image.open(self.paths[int(index)]) as image:
            return self.transform(image.convert("RGB"))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--split_manifest", default=None)
    parser.add_argument("--imagenet_train_dir", default=None)
    parser.add_argument(
        "--class_image_root",
        default=None,
        help=(
            "Class-folder image root used directly as the real distribution. "
            "This is mutually exclusive with --manifest/--split_manifest."
        ),
    )
    parser.add_argument("--inception_weights_path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="npu")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--expected_samples", type=int, default=10000)
    parser.add_argument("--expected_classes", type=int, default=100)
    parser.add_argument("--expected_samples_per_class", type=int, default=100)
    parser.add_argument("--feature", type=int, default=2048)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size_per_rank", type=int, default=16)
    parser.add_argument("--dataloader_workers", type=int, default=0)
    return parser.parse_args()


def load_selected_paths(args) -> tuple[list[Path], list[dict[str, object]]]:
    if args.class_image_root:
        if args.manifest or args.split_manifest or args.imagenet_train_dir:
            raise ValueError(
                "--class_image_root is mutually exclusive with "
                "--manifest/--split_manifest/--imagenet_train_dir"
            )
        root = Path(args.class_image_root)
        if not root.is_dir():
            raise FileNotFoundError(root)
        paths: list[Path] = []
        selected: list[dict[str, object]] = []
        for class_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            class_paths = sorted(path for path in class_dir.iterdir() if path.is_file())
            if len(class_paths) != int(args.expected_samples_per_class):
                raise ValueError(
                    f"{class_dir} has {len(class_paths)} images; expected "
                    f"{args.expected_samples_per_class}"
                )
            for path in class_paths:
                selected.append(
                    {
                        "synset": class_dir.name,
                        # Keep the selected-set identity independent of the
                        # platform mount point while retaining the absolute
                        # root separately in metadata.
                        "source_path": path.relative_to(root).as_posix(),
                        "split": str(args.split),
                        "split_index": len(selected),
                    }
                )
                paths.append(path)
        if len(paths) != int(args.expected_samples):
            raise ValueError(
                f"{root} has {len(paths)} class-folder images; expected "
                f"{args.expected_samples}"
            )
        class_count = len({str(row["synset"]) for row in selected})
        if class_count != int(args.expected_classes):
            raise ValueError(
                f"{root} has {class_count} classes; expected {args.expected_classes}"
            )
        return paths, selected

    if not args.manifest or not args.split_manifest or not args.imagenet_train_dir:
        raise ValueError(
            "provide either --class_image_root or all of --manifest, "
            "--split_manifest and --imagenet_train_dir"
        )
    manifest_path = Path(args.manifest)
    split_path = Path(args.split_manifest)
    image_root = Path(args.imagenet_train_dir)
    sources: dict[int, tuple[str, Path]] = {}
    with manifest_path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            sources[int(record["img_id"])] = (
                str(record["synset"]),
                Path(record["source_path"]),
            )

    selected = []
    with split_path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if str(record["split"]) == str(args.split):
                selected.append(record)
    selected.sort(key=lambda record: int(record["split_index"]))
    if len(selected) != int(args.expected_samples):
        raise ValueError(
            f"split={args.split!r} has {len(selected)} rows; "
            f"expected {args.expected_samples}"
        )
    per_class = Counter(str(record["synset"]) for record in selected)
    if len(per_class) != int(args.expected_classes):
        raise ValueError(
            f"selected split has {len(per_class)} classes; "
            f"expected {args.expected_classes}"
        )
    unexpected_counts = {
        synset: count
        for synset, count in per_class.items()
        if count != int(args.expected_samples_per_class)
    }
    if unexpected_counts:
        raise ValueError(
            "selected split is not class balanced: "
            f"{unexpected_counts}"
        )

    paths = []
    for record in selected:
        img_id = int(record["img_id"])
        if img_id not in sources:
            raise KeyError(f"img_id={img_id} is absent from {manifest_path}")
        source_synset, source_path = sources[img_id]
        if source_synset != str(record["synset"]):
            raise ValueError(
                f"img_id={img_id} synset mismatch: "
                f"manifest={source_synset}, split={record['synset']}"
            )
        path = source_path
        if not path.is_file():
            path = image_root / source_synset / source_path.name
        if not path.is_file():
            raise FileNotFoundError(path)
        paths.append(path)
    return paths, selected


def selected_records_sha256(records: list[dict[str, object]]) -> str:
    payload = "".join(
        json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
        for record in records
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def save_atomic(payload: dict[str, object], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{output.name}.",
        suffix=".tmp",
        dir=output.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(payload, temporary)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    distributed, rank, world_size, _, device = init_distributed(args.device)
    if device.type == "npu":
        # Inception uses operators provided by the installed torch_npu binary.
        # Avoid compiling the same inference graph independently on every rank.
        torch.npu.set_compile_mode(jit_compile=False)
    paths, selected = load_selected_paths(args)
    sampler = RankStrideSampler(len(paths), rank, world_size)
    loader = DataLoader(
        ImageDataset(paths, int(args.image_size)),
        batch_size=int(args.batch_size_per_rank),
        sampler=sampler,
        num_workers=int(args.dataloader_workers),
        pin_memory=False,
        drop_last=False,
    )
    extractor = build_inception_extractor(
        int(args.feature),
        args.inception_weights_path,
        device,
    )
    moments = FeatureMoments.zeros(int(args.feature), device)
    for batch_idx, images in enumerate(loader):
        images = images.to(device=device, dtype=torch.float32)
        features, _ = extract_inception_features(extractor, images)
        if not bool(torch.isfinite(features).all()):
            raise FloatingPointError(
                f"non-finite Inception features at rank={rank}, batch={batch_idx}"
            )
        moments.update(features)
        if rank == 0 and (batch_idx + 1) % 10 == 0:
            print(
                f"real-stat progress: local_batch={batch_idx + 1}/{len(loader)}",
                flush=True,
            )
    moments.all_reduce_()
    if int(moments.count.item()) != int(args.expected_samples):
        raise RuntimeError(
            f"distributed feature count={int(moments.count.item())}; "
            f"expected={args.expected_samples}"
        )

    if rank == 0:
        payload = {
            "schema": "imagenet_inception_feature_moments_v1",
            "stats": {
                "count": int(moments.count.item()),
                "sum": moments.sum.detach().cpu(),
                "outer_sum": moments.outer_sum.detach().cpu(),
            },
            "metadata": {
                "source": {
                    "manifest": (
                        str(Path(args.manifest).resolve())
                        if args.manifest
                        else None
                    ),
                    "manifest_sha256": (
                        file_sha256(args.manifest) if args.manifest else None
                    ),
                    "split_manifest": (
                        str(Path(args.split_manifest).resolve())
                        if args.split_manifest
                        else None
                    ),
                    "split_manifest_sha256": (
                        file_sha256(args.split_manifest)
                        if args.split_manifest
                        else None
                    ),
                    "class_image_root": (
                        str(Path(args.class_image_root).resolve())
                        if args.class_image_root
                        else None
                    ),
                    "selected_records_sha256": selected_records_sha256(selected),
                    "split": str(args.split),
                    "classes": int(args.expected_classes),
                    "samples_per_class": int(args.expected_samples_per_class),
                },
                "feature": {
                    "extractor": "torch-fidelity-inception-v3-compat",
                    "feature": int(args.feature),
                    "weights_sha256": file_sha256(args.inception_weights_path),
                    "accumulation_dtype": str(moments.sum.dtype),
                },
                "image_transform": {
                    "resize": int(args.image_size),
                    "interpolation": "bicubic",
                    "center_crop": int(args.image_size),
                    "color_mode": "RGB",
                },
                "distributed": {
                    "world_size": int(world_size),
                    "backend": (
                        dist.get_backend()
                        if distributed and dist.is_initialized()
                        else None
                    ),
                    "device_type": str(device.type),
                    "batch_size_per_rank": int(args.batch_size_per_rank),
                    "dataloader_workers": int(args.dataloader_workers),
                },
            },
        }
        output = Path(args.output)
        save_atomic(payload, output)
        print(
            f"PASS saved {int(moments.count.item())} real-image moments to {output}",
            flush=True,
        )
    if distributed and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
