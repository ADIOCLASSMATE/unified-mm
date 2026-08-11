"""Cache scaled KL16 VAE posterior mean/std for ImageNet.

Each shard contains one FP16 tensor with shape ``[N, 256, 32]``.  The last
dimension is ``concat(scaled_mean, scaled_std)``.  Training can therefore draw
fresh posterior samples without running the VAE or evaluating exp(logvar).
"""

import argparse
import hashlib
import importlib.util
import json
import os
from collections.abc import Sequence
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

POSTERIOR_CACHE_FORMAT = "imagenet_kl16_scaled_posterior_v1"
POSTERIOR_STATS_LAYOUT = "scaled_mean_then_scaled_std"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_vae_class(module_root: Path):
    module_path = module_root / "models" / "vae.py"
    if not module_path.is_file():
        raise FileNotFoundError(f"Missing MAR KL16 VAE module: {module_path}")
    spec = importlib.util.spec_from_file_location("mar_kl16_vae", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load MAR KL16 VAE module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.AutoencoderKL


def resolve_device(value: str) -> torch.device:
    requested = str(value).strip().lower()
    if requested == "auto":
        try:
            import tbe
            import torch_npu

            if torch.npu.is_available():
                requested = "npu:0"
            elif torch.cuda.is_available():
                requested = "cuda:0"
            else:
                requested = "cpu"
        except ImportError:
            requested = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "npu":
        import tbe  # noqa: F401
        import torch_npu  # noqa: F401

        if not torch.npu.is_available():
            raise RuntimeError(
                "NPU was requested but torch.npu.is_available() is false"
            )
        index = 0 if device.index is None else int(device.index)
        if index >= torch.npu.device_count():
            raise RuntimeError(
                f"NPU index {index} is out of range for {torch.npu.device_count()} devices"
            )
        # This encoder only uses standard operators covered by the installed
        # torch_npu binaries.  Avoid per-process JIT kernel compilation when
        # many independent cache shards are encoded on one Ascend node.
        torch.npu.set_compile_mode(jit_compile=False)
        torch.npu.set_device(index)
        device = torch.device("npu", index)
    elif device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but torch.cuda.is_available() is false"
            )
        index = 0 if device.index is None else int(device.index)
        torch.cuda.set_device(index)
        device = torch.device("cuda", index)
    return device


def resolve_vae_dtype(value: str, device: torch.device) -> torch.dtype:
    value = str(value).strip().lower()
    if value == "auto":
        value = "fp16" if device.type in {"cuda", "npu"} else "fp32"
    mapping = {
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if value not in mapping:
        raise ValueError(f"Unsupported --vae_dtype={value!r}")
    if device.type == "cpu" and mapping[value] != torch.float32:
        raise ValueError("CPU VAE encoding requires --vae_dtype fp32")
    return mapping[value]


def validate_reusable_shard(
    path: Path,
    samples: Sequence[tuple[int, Path, str | None]],
    *,
    num_shards: int,
    shard_index: int,
    scaling_factor: float,
    vae_checkpoint_sha256: str,
    vae_module_sha256: str,
    source_manifest_sha256: str | None,
    source_image_root: str | None,
) -> None:
    payload = torch.load(
        str(path),
        map_location="cpu",
        mmap=True,
        weights_only=True,
    )
    stats = payload.get("posterior_stats")
    image_ids = payload.get("img_ids")
    metadata = payload.get("metadata", {})
    expected_ids = torch.tensor([sample[0] for sample in samples], dtype=torch.int64)
    expected_shape = (len(samples), 256, 32)
    if not torch.is_tensor(stats) or tuple(stats.shape) != expected_shape:
        raise RuntimeError(
            f"existing shard cannot be reused; shape={getattr(stats, 'shape', None)}, "
            f"expected={expected_shape}: {path}"
        )
    if stats.dtype != torch.float16 or not torch.equal(image_ids, expected_ids):
        raise RuntimeError(f"existing shard tensor contract mismatch: {path}")
    expected_metadata = {
        "format": POSTERIOR_CACHE_FORMAT,
        "stats_layout": POSTERIOR_STATS_LAYOUT,
        "num_shards": num_shards,
        "shard_index": shard_index,
        "scaling_factor": scaling_factor,
        "vae_checkpoint_sha256": vae_checkpoint_sha256,
        "vae_module_sha256": vae_module_sha256,
        "source_manifest_sha256": source_manifest_sha256,
        "source_image_root": source_image_root,
    }
    for field, expected in expected_metadata.items():
        if metadata.get(field) != expected:
            raise RuntimeError(
                f"existing shard metadata mismatch for {field}: "
                f"{metadata.get(field)!r} != {expected!r}: {path}"
            )
    for start in range(0, stats.shape[0], 512):
        chunk = stats[start : start + 512]
        if not bool(torch.isfinite(chunk).all()) or bool((chunk[..., 16:] < 0).any()):
            raise RuntimeError(
                f"existing shard contains invalid posterior stats: {path}"
            )


class ImagePathDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[tuple[int, Path, str | None]],
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
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_id, path, synset = self.samples[idx]
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        return img_id, tensor, str(path), synset or ""


def load_img_ids(docs_jsonl: Path) -> list[int]:
    seen = set()
    img_ids: list[int] = []
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
) -> list[tuple[int, Path, str | None]]:
    return [
        (img_id, image_dir / f"{img_id:012d}.{image_extension}", None)
        for img_id in load_img_ids(docs_jsonl)
    ]


def load_samples_from_imagenet_train(
    train_dir: Path,
    start_img_id: int,
    max_images: int,
) -> list[tuple[int, Path, str | None]]:
    samples: list[tuple[int, Path, str | None]] = []
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
    source_image_root: Path | None = None,
) -> list[tuple[int, Path, str | None]]:
    samples: list[tuple[int, Path, str | None]] = []
    with manifest_jsonl.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            synset = row.get("synset")
            source_path = Path(row["source_path"])
            if source_image_root is not None:
                if not synset:
                    raise ValueError(
                        "--source_image_root requires every manifest row to contain synset"
                    )
                source_path = source_image_root / str(synset) / source_path.name
            samples.append(
                (
                    int(row["img_id"]),
                    source_path,
                    synset,
                )
            )
            if max_images > 0 and len(samples) >= max_images:
                break
    return samples


def save_manifest(
    path: Path,
    samples: Sequence[tuple[int, Path, str | None]],
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
        "--source_image_root",
        default=None,
        help="Resolve manifest synset/filename under this current filesystem root.",
    )
    parser.add_argument("--vae_path", default="public/vae/mar-kl16/kl16.ckpt")
    parser.add_argument("--vae_module_root", default="public/code/mar")
    parser.add_argument("--cache_shard_dir", required=True)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--image_extension", default="jpg")
    parser.add_argument("--scaling_factor", type=float, default=0.2325)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--vae_dtype",
        choices=["auto", "fp16", "bf16", "fp32"],
        default="auto",
    )
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
    vae_module_root = Path(args.vae_module_root)
    AutoencoderKL = load_vae_class(vae_module_root)
    vae_checkpoint_sha256 = sha256_file(vae_path)
    vae_module_sha256 = sha256_file(vae_module_root / "models" / "vae.py")

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
            Path(args.source_manifest_jsonl),
            args.max_images,
            (
                Path(args.source_image_root)
                if args.source_image_root is not None
                else None
            ),
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
        source_manifest_sha256 = (
            sha256_file(Path(args.source_manifest_jsonl))
            if args.source_mode == "manifest_jsonl"
            else None
        )
        validate_reusable_shard(
            shard_path,
            samples,
            num_shards=args.num_shards,
            shard_index=args.shard_index,
            scaling_factor=args.scaling_factor,
            vae_checkpoint_sha256=vae_checkpoint_sha256,
            vae_module_sha256=vae_module_sha256,
            source_manifest_sha256=source_manifest_sha256,
            source_image_root=args.source_image_root,
        )
        print(f"Validated and reused posterior cache shard: {shard_path}")
        return

    device = resolve_device(args.device)
    vae_dtype = resolve_vae_dtype(args.vae_dtype, device)
    torch.manual_seed(args.seed + args.shard_index)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

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
        # torch_npu does not use CUDA's pinned-memory allocator.  Asynchronous
        # host-to-NPU copies are still requested below and dispatch correctly.
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
            batch_stats = torch.cat((posterior.mean, posterior.std), dim=1).mul_(
                args.scaling_factor
            )
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
            args.source_manifest_jsonl if args.source_mode == "manifest_jsonl" else None
        ),
        "source_manifest_sha256": (
            sha256_file(Path(args.source_manifest_jsonl))
            if args.source_mode == "manifest_jsonl"
            else None
        ),
        "source_image_root": args.source_image_root,
        "imagenet_train_dir": (
            args.imagenet_train_dir if args.source_mode == "imagenet_train" else None
        ),
        "vae": "mar-kl16",
        "vae_module_root": str(vae_module_root),
        "vae_module_sha256": vae_module_sha256,
        "vae_checkpoint": str(vae_path),
        "vae_checkpoint_sha256": vae_checkpoint_sha256,
        "device_type": device.type,
        "vae_dtype": str(vae_dtype).removeprefix("torch."),
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
    os.replace(temporary_path, shard_path)
    print(f"Saved {encoded} posterior rows to {shard_path}")


if __name__ == "__main__":
    main()
