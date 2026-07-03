"""
Encode ImageNet prompt-dataset images into continuous VAE latents.

This script reads img_ids from an OmniCorpus-compatible docs JSONL, loads images
from a numeric image directory, and writes one .pt file per image id.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Iterable, List, Tuple

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("DIFFUSERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def iter_image_ids(jsonl_path: Path) -> Iterable[int]:
    seen = set()
    with jsonl_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            for img_id in record.get("img_ids", []):
                img_id = int(img_id)
                if img_id not in seen:
                    seen.add(img_id)
                    yield img_id


def center_crop_resize(image: Image.Image, image_size: int) -> Image.Image:
    width, height = image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    image = image.crop((left, top, left + side, top + side))
    return image.resize((image_size, image_size), Image.Resampling.BICUBIC)


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    data = torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes()))
    data = data.view(image.size[1], image.size[0], 3)
    tensor = data.permute(2, 0, 1).float().div_(127.5).sub_(1.0)
    return tensor


class ImageLatentDataset(Dataset):
    def __init__(self, image_dir: Path, img_ids: List[int], image_size: int, extension: str):
        self.image_dir = image_dir
        self.img_ids = img_ids
        self.image_size = image_size
        self.extension = extension

    def __len__(self) -> int:
        return len(self.img_ids)

    def __getitem__(self, idx: int):
        img_id = int(self.img_ids[idx])
        image_path = self.image_dir / f"{img_id:012d}.{self.extension}"
        try:
            with Image.open(image_path) as image:
                image = center_crop_resize(image.convert("RGB"), self.image_size)
                tensor = image_to_tensor(image)
            return img_id, tensor, ""
        except Exception as exc:
            return img_id, torch.empty(0), str(exc)


def collate_image_batch(items):
    img_ids = []
    images = []
    errors = []
    for img_id, image_tensor, error in items:
        if error:
            errors.append((img_id, error))
        else:
            img_ids.append(img_id)
            images.append(image_tensor)
    if images:
        return img_ids, torch.stack(images, dim=0), errors
    return img_ids, None, errors


def torch_dtype(name: str, device: str):
    if name == "auto":
        return torch.float16 if device.startswith("cuda") else torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"unknown dtype: {name}")


def load_vae(vae_path: Path, device: str, dtype: torch.dtype):
    from diffusers import AutoencoderKL

    vae = AutoencoderKL.from_pretrained(str(vae_path), torch_dtype=dtype)
    vae = vae.to(device).eval()
    for param in vae.parameters():
        param.requires_grad_(False)
    return vae


def encode_distribution(vae, images: torch.Tensor, sample_posterior: bool, generator):
    posterior = vae.encode(images).latent_dist
    if sample_posterior:
        latents = posterior.sample(generator=generator)
    else:
        latents = posterior.mean
    return latents, posterior


def save_latents(
    output_path: Path,
    latent: torch.Tensor,
    scaling_factor: float,
    save_dtype: torch.dtype,
    mean: torch.Tensor = None,
    logvar: torch.Tensor = None,
    latent_flip: torch.Tensor = None,
    mean_flip: torch.Tensor = None,
    logvar_flip: torch.Tensor = None,
) -> None:
    record = {
        "latent": latent.detach().to(save_dtype).cpu(),
        "scaling_factor": float(scaling_factor),
    }
    if latent_flip is not None:
        record["latent_flip"] = latent_flip.detach().to(save_dtype).cpu()
    if mean is not None and logvar is not None:
        record["mean"] = mean.detach().to(save_dtype).cpu()
        record["logvar"] = logvar.detach().to(save_dtype).cpu()
    if mean_flip is not None and logvar_flip is not None:
        record["mean_flip"] = mean_flip.detach().to(save_dtype).cpu()
        record["logvar_flip"] = logvar_flip.detach().to(save_dtype).cpu()
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(record, tmp_path)
    tmp_path.replace(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs_jsonl", default="public/datasets/imagenet_prompt/docs/train.jsonl")
    parser.add_argument("--image_dir", default="public/datasets/imagenet_prompt/images")
    parser.add_argument("--output_dir", default="public/datasets/imagenet_prompt/vae_latents_sd_ft_mse")
    parser.add_argument("--vae_path", default="public/vae/stabilityai-sd-vae-ft-mse")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--image_extension", default="jpg")
    parser.add_argument("--scaling_factor", type=float, default=0.18215,
                        help="Applied as latent = latent * scaling_factor. SD-VAE convention is 0.18215.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--save_dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--sample_posterior", action="store_true",
                        help="Sample from posterior instead of saving posterior mean.")
    parser.add_argument("--save_moments", action="store_true",
                        help="Also save posterior mean/logvar for each image.")
    parser.add_argument("--save_flip", action="store_true",
                        help="Also encode the horizontally flipped image.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_images", type=int, default=-1)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no_pin_memory", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch_size must be >= 1")
    if args.num_workers < 0:
        raise ValueError("--num_workers must be >= 0")
    if args.prefetch_factor < 1:
        raise ValueError("--prefetch_factor must be >= 1")
    if args.num_shards < 1:
        raise ValueError("--num_shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard_index must be in [0, num_shards)")

    docs_jsonl = Path(args.docs_jsonl)
    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)
    vae_path = Path(args.vae_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    img_ids = list(iter_image_ids(docs_jsonl))
    if args.max_images > 0:
        img_ids = img_ids[:args.max_images]
    if args.num_shards > 1:
        img_ids = [img_id for i, img_id in enumerate(img_ids) if i % args.num_shards == args.shard_index]

    pending = []
    existing = 0
    for img_id in tqdm(img_ids, desc="Checking existing latents"):
        output_path = output_dir / f"{img_id:012d}.pt"
        if output_path.exists() and not args.overwrite:
            existing += 1
        else:
            pending.append(img_id)

    print(
        f"Encoding {len(pending)} pending images ({existing} existing) "
        f"on shard {args.shard_index}/{args.num_shards}"
    )
    if not pending:
        return

    dtype = torch_dtype(args.dtype, args.device)
    save_dtype = torch_dtype(args.save_dtype, "cpu")
    generator = None
    if args.sample_posterior:
        generator = torch.Generator(device=args.device)
        generator.manual_seed(args.seed + args.shard_index)

    dataset = ImageLatentDataset(image_dir, pending, args.image_size, args.image_extension)
    loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "collate_fn": collate_image_batch,
        "pin_memory": not args.no_pin_memory,
    }
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
        loader_kwargs["persistent_workers"] = True
    dataloader = DataLoader(dataset, **loader_kwargs)

    print(f"Loading VAE from {vae_path}...")
    vae = load_vae(vae_path, args.device, dtype)

    ok = 0
    errors = 0
    progress = tqdm(total=len(pending), desc="Encoding VAE latents")
    with torch.no_grad():
        for batch_ids, images, load_errors in dataloader:
            if load_errors:
                for img_id, error in load_errors:
                    print(f"Failed img_id={img_id}: {error}")
                errors += len(load_errors)
                progress.update(len(load_errors))
            if images is None or not batch_ids:
                continue

            images = images.to(device=args.device, dtype=dtype, non_blocking=True)
            try:
                latents, posterior = encode_distribution(vae, images, args.sample_posterior, generator)
                latents = latents * args.scaling_factor
                latents_flip = None
                posterior_flip = None
                if args.save_flip:
                    flipped = images.flip(dims=[3])
                    latents_flip, posterior_flip = encode_distribution(
                        vae, flipped, args.sample_posterior, generator
                    )
                    latents_flip = latents_flip * args.scaling_factor

                for i, img_id in enumerate(batch_ids):
                    save_latents(
                        output_dir / f"{int(img_id):012d}.pt",
                        latent=latents[i],
                        scaling_factor=args.scaling_factor,
                        save_dtype=save_dtype,
                        mean=posterior.mean[i] if args.save_moments else None,
                        logvar=posterior.logvar[i] if args.save_moments else None,
                        latent_flip=latents_flip[i] if latents_flip is not None else None,
                        mean_flip=(
                            posterior_flip.mean[i]
                            if args.save_moments and posterior_flip is not None
                            else None
                        ),
                        logvar_flip=(
                            posterior_flip.logvar[i]
                            if args.save_moments and posterior_flip is not None
                            else None
                        ),
                    )
                    ok += 1
            except Exception as exc:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print(f"Batch failed with {len(batch_ids)} images: {exc}")
                errors += len(batch_ids)
            progress.update(len(batch_ids))
    progress.close()
    print(f"Done: {ok} encoded, {errors} errors, saved to {output_dir}")


if __name__ == "__main__":
    main()
