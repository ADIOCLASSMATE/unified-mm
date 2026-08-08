"""Cache scaled KL16 VAE posterior mean/std for ImageNet.

Each shard contains one FP16 tensor with shape ``[N, 256, 32]``.  The last
dimension is ``concat(scaled_mean, scaled_std)``.  Training can therefore draw
fresh posterior samples without running the VAE or evaluating exp(logvar).
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm


POSTERIOR_CACHE_FORMAT = "imagenet_kl16_scaled_posterior_v1"
POSTERIOR_STATS_LAYOUT = "scaled_mean_then_scaled_std"
VAE_MODULE_ROOT = Path("/inspire/hdd/global_user/wanjiaxin-253108030048/code/mar")
if str(VAE_MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(VAE_MODULE_ROOT))

from models.vae import AutoencoderKL  # noqa: E402


class ImagePathDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[Tuple[int, Path, Optional[str]]],
        image_size: int,
    ):
        self.samples = list(samples)
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    image_size,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
                ),
            ]
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_id, path, synset = self.samples[idx]
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        return img_id, tensor, str(path), synset or ""


def load_img_ids(docs_jsonl: Path) -> List[int]:
    seen = set()
    img_ids: List[int] = []
    with docs_jsonl.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            for img_id in record["img_ids"]:
                img_id = int(img_id)
                if img_id not in seen:
                    seen.add(img_id)
                    img_ids.append(img_id)
    return img_ids


def load_samples_from_docs(
    docs_jsonl: Path,
    image_dir: Path,
    image_extension: str,
) -> List[Tuple[int, Path, Optional[str]]]:
    return [
        (img_id, image_dir / f"{img_id:012d}.{image_extension}", None)
        for img_id in load_img_ids(docs_jsonl)
    ]


def load_samples_from_imagenet_train(
    train_dir: Path,
    start_img_id: int,
    max_images: int,
) -> List[Tuple[int, Path, Optional[str]]]:
    samples: List[Tuple[int, Path, Optional[str]]] = []
    used = 0
    for class_dir in sorted(path for path in train_dir.iterdir() if path.is_dir()):
        synset = class_dir.name
        for path in sorted(class_dir.iterdir()):
            if not path.is_file():
                continue
            if max_images > 0 and used >= max_images:
                return samples
            samples.append((start_img_id + used, path, synset))
            used += 1
    return samples


def load_samples_from_manifest(
    manifest_jsonl: Path,
    max_images: int,
) -> List[Tuple[int, Path, Optional[str]]]:
    samples: List[Tuple[int, Path, Optional[str]]] = []
    with manifest_jsonl.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            samples.append(
                (
                    int(row["img_id"]),
                    Path(row["source_path"]),
                    row.get("synset"),
                )
            )
            if max_images > 0 and len(samples) >= max_images:
                break
    return samples


def save_manifest(
    path: Path,
    samples: Sequence[Tuple[int, Path, Optional[str]]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for img_id, source_path, synset in samples:
            handle.write(
                json.dumps(
                    {
                        "img_id": int(img_id),
                        "source_path": str(source_path),
                        "synset": synset,
                    },
                    ensure_ascii=True,
                )
                + "\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source_mode",
        choices=["docs_jsonl", "imagenet_train", "manifest_jsonl"],
        default="docs_jsonl",
    )
    parser.add_argument(
        "--docs_jsonl",
        default="public/datasets/imagenet_prompt_500c_all/docs/train.jsonl",
    )
    parser.add_argument(
        "--image_dir", default="public/datasets/imagenet_prompt_500c_all/images"
    )
    parser.add_argument(
        "--imagenet_train_dir",
        default="/inspire/dataset/imagenet/v1/ILSVRC/Data/CLS-LOC/train",
    )
    parser.add_argument(
        "--source_manifest_jsonl",
        default="public/datasets/imagenet_full/manifest.jsonl",
    )
    parser.add_argument(
        "--vae_path", default="public/vae/mar-kl16/kl16.ckpt"
    )
    parser.add_argument("--cache_shard_dir", required=True)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--image_extension", default="jpg")
    parser.add_argument("--scaling_factor", type=float, default=0.2325)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max_images", type=int, default=-1)
    parser.add_argument("--start_img_id", type=int, default=1)
    parser.add_argument("--manifest_jsonl", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.num_shards < 1:
        raise ValueError("--num_shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard_index must be in [0, num_shards)")

    vae_path = Path(args.vae_path)
    if not vae_path.exists():
        raise FileNotFoundError(f"Missing KL16 checkpoint: {vae_path}")

    if args.source_mode == "docs_jsonl":
        samples = load_samples_from_docs(
            Path(args.docs_jsonl),
            Path(args.image_dir),
            args.image_extension,
        )
        if args.max_images > 0:
            samples = samples[: args.max_images]
    elif args.source_mode == "imagenet_train":
        samples = load_samples_from_imagenet_train(
            Path(args.imagenet_train_dir),
            args.start_img_id,
            args.max_images,
        )
    else:
        samples = load_samples_from_manifest(
            Path(args.source_manifest_jsonl), args.max_images
        )

    if args.manifest_jsonl and args.shard_index == 0:
        save_manifest(Path(args.manifest_jsonl), samples)

    samples = [
        sample
        for index, sample in enumerate(samples)
        if index % args.num_shards == args.shard_index
    ]
    cache_shard_dir = Path(args.cache_shard_dir)
    cache_shard_dir.mkdir(parents=True, exist_ok=True)
    shard_path = cache_shard_dir / (
        f"shard-{args.shard_index:05d}-of-{args.num_shards:05d}.pt"
    )
    if shard_path.exists() and not args.overwrite:
        print(f"Posterior cache shard already exists, skipping: {shard_path}")
        return

    device = torch.device(args.device)
    torch.manual_seed(args.seed + args.shard_index)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    vae_dtype = torch.float16 if device.type == "cuda" else torch.float32
    vae = AutoencoderKL(
        embed_dim=16,
        ch_mult=(1, 1, 2, 2, 4),
        ckpt_path=str(vae_path),
    )
    vae = vae.to(device=device, dtype=vae_dtype).eval()
    for parameter in vae.parameters():
        parameter.requires_grad_(False)

    dataset = ImagePathDataset(samples, args.image_size)
    loader_kwargs = {}
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        **loader_kwargs,
    )

    # One large tensor is much faster to merge and mmap than per-image files.
    posterior_stats = torch.empty((len(samples), 256, 32), dtype=torch.float16)
    cache_img_ids = torch.empty((len(samples),), dtype=torch.long)
    encoded = 0
    progress = tqdm(total=len(samples), desc="Caching KL16 posterior", unit="img")
    with torch.inference_mode():
        for batch_img_ids, images, _, _ in loader:
            images = images.to(
                device=device,
                dtype=vae_dtype,
                non_blocking=True,
            )
            posterior = vae.encode(images)
            batch_stats = torch.cat(
                (posterior.mean, posterior.std), dim=1
            ).mul_(args.scaling_factor)
            if tuple(batch_stats.shape[1:]) != (32, 16, 16):
                raise ValueError(
                    "KL16 posterior must have shape [B, 32, 16, 16], got "
                    f"{tuple(batch_stats.shape)}"
                )
            batch_stats = (
                batch_stats.permute(0, 2, 3, 1)
                .reshape(-1, 256, 32)
                .to(device="cpu", dtype=torch.float16)
            )
            batch_size = int(batch_stats.shape[0])
            posterior_stats[encoded : encoded + batch_size].copy_(batch_stats)
            cache_img_ids[encoded : encoded + batch_size].copy_(batch_img_ids)
            encoded += batch_size
            progress.update(batch_size)
    progress.close()

    if encoded != len(samples):
        raise RuntimeError(f"Encoded {encoded} images, expected {len(samples)}")
    metadata = {
        "format": POSTERIOR_CACHE_FORMAT,
        "stats_layout": POSTERIOR_STATS_LAYOUT,
        "stats_are_scaled": True,
        "num_images": encoded,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "source_mode": args.source_mode,
        "source_manifest_jsonl": (
            args.source_manifest_jsonl
            if args.source_mode == "manifest_jsonl"
            else None
        ),
        "imagenet_train_dir": (
            args.imagenet_train_dir
            if args.source_mode == "imagenet_train"
            else None
        ),
        "vae": "mar-kl16",
        "vae_checkpoint": str(vae_path),
        "scaling_factor": args.scaling_factor,
        "image_size": args.image_size,
        "posterior_shape": [16, 16, 32],
        "token_shape": [256, 32],
        "storage_dtype": "float16",
    }
    temporary_path = shard_path.with_suffix(shard_path.suffix + ".tmp")
    torch.save(
        {
            "posterior_stats": posterior_stats,
            "img_ids": cache_img_ids,
            "metadata": metadata,
        },
        temporary_path,
    )
    temporary_path.replace(shard_path)
    print(f"Saved {encoded} posterior rows to {shard_path}")


if __name__ == "__main__":
    main()
