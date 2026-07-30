"""
Encode ImageNet prompt-dataset images with the KL16 VAE.

This encoder can write per-image .pt files containing:
    {"latent": Tensor[16,16,16], "scaling_factor": 0.2325}

For high-throughput full-ImageNet flow warmup, prefer --cache_shard_dir and
--skip_per_image. That writes one Tensor[N,256,16] shard per GPU instead of
millions of tiny files.
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


VAE_MODULE_ROOT = Path("/inspire/hdd/global_user/wanjiaxin-253108030048/code/mar")
if str(VAE_MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(VAE_MODULE_ROOT))

from models.vae import AutoencoderKL  # noqa: E402


class ImagePathDataset(Dataset):
    def __init__(self, samples: Sequence[Tuple[int, Path, Optional[str]]], image_size: int):
        self.samples = list(samples)
        self.transform = transforms.Compose(
            [
                transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_id, path, synset = self.samples[idx]
        with Image.open(path) as image:
            image = image.convert("RGB")
            tensor = self.transform(image)
        return img_id, tensor, str(path), synset or ""


def load_img_ids(docs_jsonl: Path) -> List[int]:
    seen = set()
    img_ids: List[int] = []
    with docs_jsonl.open() as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            for img_id in record["img_ids"]:
                img_id = int(img_id)
                if img_id not in seen:
                    seen.add(img_id)
                    img_ids.append(img_id)
    return img_ids


def load_samples_from_docs(docs_jsonl: Path, image_dir: Path, image_extension: str) -> List[Tuple[int, Path, Optional[str]]]:
    return [
        (img_id, image_dir / f"{img_id:012d}.{image_extension}", None)
        for img_id in load_img_ids(docs_jsonl)
    ]


def load_samples_from_imagenet_train(train_dir: Path, start_img_id: int, max_images: int) -> List[Tuple[int, Path, Optional[str]]]:
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


def save_manifest(path: Path, samples: Sequence[Tuple[int, Path, Optional[str]]]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for img_id, source_path, synset in samples:
            f.write(
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


def save_latent(path: Path, latent: torch.Tensor, scaling_factor: float, save_dtype: torch.dtype) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "latent": latent.detach().to(save_dtype).cpu(),
            "scaling_factor": scaling_factor,
            "vae": "mar-kl16",
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_mode", choices=["docs_jsonl", "imagenet_train"], default="docs_jsonl")
    parser.add_argument("--docs_jsonl", default="public/datasets/imagenet_prompt_500c_all/docs/train.jsonl")
    parser.add_argument("--image_dir", default="public/datasets/imagenet_prompt_500c_all/images")
    parser.add_argument("--imagenet_train_dir", default="/inspire/dataset/imagenet/v1/ILSVRC/Data/CLS-LOC/train")
    parser.add_argument("--output_dir", default="public/datasets/imagenet_prompt_500c_all/vae_latents_mar_kl16")
    parser.add_argument("--vae_path", default="public/vae/mar-kl16/kl16.ckpt")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--image_extension", default="jpg")
    parser.add_argument("--scaling_factor", type=float, default=0.2325)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sample_posterior", action="store_true")
    parser.add_argument("--save_dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max_images", type=int, default=-1)
    parser.add_argument("--start_img_id", type=int, default=1)
    parser.add_argument("--manifest_jsonl", default=None)
    parser.add_argument("--cache_path", default=None,
                        help="Optional latent cache Tensor[N,256,16] written after encoding. Use only with one shard.")
    parser.add_argument("--cache_shard_dir", default=None,
                        help="Optional directory for per-shard latent cache files.")
    parser.add_argument("--skip_per_image", action="store_true",
                        help="Do not save one .pt file per image; intended for fast cache-shard encoding.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.num_shards < 1:
        raise ValueError("--num_shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard_index must be in [0, num_shards)")

    output_dir = Path(args.output_dir)
    vae_path = Path(args.vae_path)
    if not vae_path.exists():
        raise FileNotFoundError(f"Missing KL16 checkpoint: {vae_path}")

    if args.source_mode == "docs_jsonl":
        samples = load_samples_from_docs(Path(args.docs_jsonl), Path(args.image_dir), args.image_extension)
    else:
        samples = load_samples_from_imagenet_train(Path(args.imagenet_train_dir), args.start_img_id, args.max_images)

    if args.manifest_jsonl and args.shard_index == 0:
        save_manifest(Path(args.manifest_jsonl), samples)

    samples = [sample for i, sample in enumerate(samples) if i % args.num_shards == args.shard_index]
    if not args.overwrite and not args.skip_per_image:
        samples = [
            sample for sample in samples
            if not (output_dir / f"{int(sample[0]):012d}.pt").exists()
        ]

    if args.cache_path and args.num_shards != 1:
        raise ValueError("--cache_path is only supported when --num_shards=1")

    save_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.save_dtype]

    device = torch.device(args.device)
    torch.manual_seed(args.seed + args.shard_index)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    vae = AutoencoderKL(embed_dim=16, ch_mult=(1, 1, 2, 2, 4), ckpt_path=str(vae_path))
    vae = vae.to(device=device, dtype=torch.float16 if device.type == "cuda" else torch.float32).eval()
    for param in vae.parameters():
        param.requires_grad_(False)

    dataset = ImagePathDataset(samples, args.image_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        persistent_workers=args.num_workers > 0,
    )

    cache_tensor = None
    if args.cache_path or args.cache_shard_dir:
        cache_tensor = torch.empty((len(samples), 256, 16), dtype=save_dtype)
        cache_img_ids = torch.empty((len(samples),), dtype=torch.long)

    errors = 0
    encoded = 0
    progress = tqdm(total=len(samples), desc="Encoding KL16 latents", unit="img")
    with torch.inference_mode():
        for batch_img_ids, images, source_paths, synsets in loader:
            try:
                images = images.to(device=device, dtype=next(vae.parameters()).dtype, non_blocking=True)
                posterior = vae.encode(images)
                latents = posterior.sample() if args.sample_posterior else posterior.mode()
                latents = latents * args.scaling_factor
                for i, img_id in enumerate(batch_img_ids.tolist()):
                    if not args.skip_per_image:
                        save_latent(output_dir / f"{int(img_id):012d}.pt", latents[i], args.scaling_factor, save_dtype)
                    if cache_tensor is not None:
                        cache_tensor[encoded] = latents[i].detach().to(save_dtype).cpu().permute(1, 2, 0).reshape(256, 16)
                        cache_img_ids[encoded] = int(img_id)
                    encoded += 1
            except Exception as exc:
                errors += len(batch_img_ids)
                tqdm.write(f"ERROR batch starting img_id={int(batch_img_ids[0])}: {exc}")
            progress.update(len(batch_img_ids))
    progress.close()

    if cache_tensor is not None and args.cache_path:
        cache_path = Path(args.cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "latents": cache_tensor[:encoded].contiguous(),
                "img_ids": cache_img_ids[:encoded].contiguous(),
                "metadata": {
                    "num_images": encoded,
                    "source_mode": args.source_mode,
                    "imagenet_train_dir": args.imagenet_train_dir if args.source_mode == "imagenet_train" else None,
                    "vae": "mar-kl16",
                    "scaling_factor": args.scaling_factor,
                    "image_size": args.image_size,
                    "latent_shape": [16, 16, 16],
                    "token_shape": [256, 16],
                },
            },
            cache_path,
        )
        print(f"Saved latent cache to {cache_path}")
    if cache_tensor is not None and args.cache_shard_dir:
        cache_shard_dir = Path(args.cache_shard_dir)
        cache_shard_dir.mkdir(parents=True, exist_ok=True)
        shard_path = cache_shard_dir / f"shard-{args.shard_index:05d}-of-{args.num_shards:05d}.pt"
        torch.save(
            {
                "latents": cache_tensor[:encoded].contiguous(),
                "img_ids": cache_img_ids[:encoded].contiguous(),
                "metadata": {
                    "num_images": encoded,
                    "num_shards": args.num_shards,
                    "shard_index": args.shard_index,
                    "source_mode": args.source_mode,
                    "imagenet_train_dir": args.imagenet_train_dir if args.source_mode == "imagenet_train" else None,
                    "vae": "mar-kl16",
                    "scaling_factor": args.scaling_factor,
                    "image_size": args.image_size,
                    "latent_shape": [16, 16, 16],
                    "token_shape": [256, 16],
                },
            },
            shard_path,
        )
        print(f"Saved latent cache shard to {shard_path}")
    print(f"Done: {encoded} encoded, {errors} errors, saved to {output_dir}")


if __name__ == "__main__":
    main()
