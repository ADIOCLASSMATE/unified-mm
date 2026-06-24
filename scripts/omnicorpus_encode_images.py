"""
Encode downloaded OmniCorpus images into Open-MAGVIT2 discrete tokens.
"""

import argparse
import json
import os
import sys
from pathlib import Path

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from tqdm import tqdm

from scripts.magvit2_wrapper import MAGVIT2Wrapper


def iter_image_ids(jsonl_path: Path):
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


class ImageTokenizeDataset(Dataset):
    def __init__(self, image_dir: Path, img_ids, preprocess):
        self.image_dir = image_dir
        self.img_ids = list(img_ids)
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.img_ids)

    def __getitem__(self, idx: int):
        img_id = int(self.img_ids[idx])
        image_path = self.image_dir / f"{img_id:012d}.jpg"
        try:
            with Image.open(image_path) as image:
                image_tensor = self.preprocess(image.convert("RGB"))
            return img_id, image_tensor, ""
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


def build_image_preprocess(image_size: int = 256):
    return T.Compose([
        T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs_jsonl", default="public/datasets/omnicorpus/docs/train.jsonl")
    parser.add_argument("--image_dir", default="public/datasets/omnicorpus/images")
    parser.add_argument("--output_dir", default="public/datasets/omnicorpus/image_tokens_magvit2")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_images", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Number of images to encode per Open-MAGVIT2 forward pass.")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="CPU worker processes per encoder shard for image loading/preprocessing.")
    parser.add_argument("--prefetch_factor", type=int, default=2,
                        help="DataLoader prefetch factor when num_workers > 0.")
    parser.add_argument("--no_pin_memory", action="store_true",
                        help="Disable pinned-memory batches before GPU transfer.")
    parser.add_argument("--num_shards", type=int, default=1,
                        help="Split image ids into this many deterministic shards for parallel encoding.")
    parser.add_argument("--shard_index", type=int, default=0,
                        help="Encode only ids whose ordinal modulo num_shards equals this index.")
    args = parser.parse_args()

    if args.num_shards < 1:
        raise ValueError("--num_shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard_index must be in [0, num_shards)")
    if args.batch_size < 1:
        raise ValueError("--batch_size must be >= 1")
    if args.num_workers < 0:
        raise ValueError("--num_workers must be >= 0")
    if args.prefetch_factor < 1:
        raise ValueError("--prefetch_factor must be >= 1")

    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    img_ids = list(iter_image_ids(Path(args.docs_jsonl)))
    if args.max_images > 0:
        img_ids = img_ids[:args.max_images]
    if args.num_shards > 1:
        img_ids = [img_id for i, img_id in enumerate(img_ids) if i % args.num_shards == args.shard_index]

    pending_img_ids = []
    ok = 0
    for img_id in tqdm(img_ids, desc="Checking existing tokens"):
        output_path = output_dir / f"{img_id:012d}.pt"
        if output_path.exists():
            ok += 1
        else:
            pending_img_ids.append(img_id)

    print(
        f"Encoding {len(pending_img_ids)} pending images "
        f"({ok} already encoded) on shard {args.shard_index}/{args.num_shards} "
        f"with batch_size={args.batch_size}, num_workers={args.num_workers}, "
        f"prefetch_factor={args.prefetch_factor}..."
    )

    errors = 0

    dataset = ImageTokenizeDataset(image_dir, pending_img_ids, build_image_preprocess())
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
    dataloader_iter = iter(dataloader)

    print("Loading Open-MAGVIT2 model...")
    wrapper = MAGVIT2Wrapper(device=args.device)

    def save_tokens(img_id: int, tokens: torch.Tensor) -> None:
        tokens = tokens.long().view(-1).cpu()
        if tokens.numel() != 256:
            raise ValueError(f"expected 256 tokens, got {tokens.numel()}")
        torch.save(tokens, output_dir / f"{img_id:012d}.pt")

    def encode_individual(img_id: int, image_tensor: torch.Tensor) -> bool:
        try:
            tokens = wrapper.encode(image_tensor)
            save_tokens(img_id, tokens)
            return True
        except Exception as exc:
            print(f"Failed img_id={img_id}: {exc}")
            return False

    def encode_batch(batch_ids, images) -> None:
        nonlocal ok, errors
        try:
            batch_tokens = wrapper.encode(images).long()
            batch_tokens = batch_tokens.view(len(batch_ids), -1)
            if batch_tokens.shape[1] != 256:
                raise ValueError(f"expected [B, 256] tokens, got {tuple(batch_tokens.shape)}")
            for img_id, tokens in zip(batch_ids, batch_tokens):
                save_tokens(img_id, tokens)
                ok += 1
        except Exception as exc:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"Batch failed with {len(batch_ids)} images: {exc}. Retrying individually.")
            for img_id, image_tensor in zip(batch_ids, images):
                if encode_individual(img_id, image_tensor):
                    ok += 1
                else:
                    errors += 1

    progress = tqdm(total=len(pending_img_ids), desc="Encoding")
    for batch_ids, images, load_errors in dataloader_iter:
        if load_errors:
            for img_id, error in load_errors:
                print(f"Failed img_id={img_id}: {error}")
            errors += len(load_errors)
            progress.update(len(load_errors))
        if images is None or not batch_ids:
            continue
        encode_batch(batch_ids, images)
        progress.update(len(batch_ids))
    progress.close()

    print(f"Done: {ok} encoded, {errors} errors, saved to {output_dir}")


if __name__ == "__main__":
    main()
