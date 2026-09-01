#!/usr/bin/env python3
"""Precompute Inception moments for the original 50K ImageNet-val images."""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_single_stream_fid_is import init_distributed
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
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    int(image_size),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.CenterCrop(int(image_size)),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        with Image.open(self.paths[int(index)]) as image:
            return self.transform(image.convert("RGB"))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--imagenet_val_dir",
        required=True,
        help="Official ImageNet-val class-folder root (1,000 classes, 50 images each).",
    )
    parser.add_argument("--inception_weights_path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="npu")
    parser.add_argument("--expected_samples", type=int, default=10000)
    parser.add_argument("--expected_classes", type=int, default=100)
    parser.add_argument("--expected_samples_per_class", type=int, default=100)
    parser.add_argument("--feature", type=int, default=2048)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size_per_rank", type=int, default=16)
    parser.add_argument("--dataloader_workers", type=int, default=0)
    return parser.parse_args()


def load_selected_paths(args) -> tuple[list[Path], list[dict[str, object]]]:
    root = Path(args.imagenet_val_dir)
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
                    "source_path": path.relative_to(root).as_posix(),
                    "split": "val",
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
            "schema": "imagenet_val_inception_feature_moments_v2",
            "stats": {
                "count": int(moments.count.item()),
                "sum": moments.sum.detach().cpu(),
                "outer_sum": moments.outer_sum.detach().cpu(),
            },
            "metadata": {
                "source": {
                    "dataset": "ImageNet-1K",
                    "imagenet_val_dir": str(Path(args.imagenet_val_dir).resolve()),
                    "split": "val",
                    "records": len(selected),
                    "classes": int(args.expected_classes),
                    "samples_per_class": int(args.expected_samples_per_class),
                },
                "feature": {
                    "extractor": "torch-fidelity-inception-v3-compat",
                    "feature": int(args.feature),
                    "weights_path": str(Path(args.inception_weights_path).resolve()),
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
