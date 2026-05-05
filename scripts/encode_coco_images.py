"""
Batch encode COCO images to XQ-GAN discrete tokens.

Usage:
    uv run python scripts/encode_coco_images.py --max_images 100 --device cuda
    uv run python scripts/encode_coco_images.py --max_images 30000 --device cuda --batch_size 8
"""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from scripts.xqgan_wrapper import XQGANWrapper


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--captions_jsonl", default="public/datasets/coco/captions.jsonl")
    parser.add_argument("--output_dir", default="public/datasets/coco/image_tokens")
    parser.add_argument("--max_images", type=int, default=100,
                        help="Max images to encode (default: 100 for testing)")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size (increase for GPU efficiency)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--decode_back", action="store_true",
                        help="Decode a sample back to verify reconstruction")
    parser.add_argument("--samples", type=int, default=3,
                        help="Number of samples to decode back for inspection")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load wrapper
    print("Loading XQ-GAN model...")
    wrapper = XQGANWrapper(device=args.device)
    print(f"Model loaded: {wrapper.total_tokens} tokens/image")

    # Read captions JSONL to get image list
    captions_path = Path(args.captions_jsonl)
    img_entries = []
    with open(captions_path) as f:
        for line in f:
            entry = json.loads(line)
            img_entries.append(entry)

    if len(img_entries) > args.max_images:
        img_entries = img_entries[:args.max_images]

    print(f"Encoding {len(img_entries)} images...")

    success = 0
    errors = 0
    for entry in tqdm(img_entries, desc="Encoding"):
        img_id = entry["img_id"]
        img_path = entry["image_path"]
        out_path = output_dir / f"{img_id:012d}.pt"

        if out_path.exists():
            success += 1
            continue

        try:
            img = Image.open(img_path).convert("RGB")
            tokens = wrapper.encode(img)
            torch.save(tokens.cpu(), out_path)
            success += 1
        except Exception as e:
            print(f"Error encoding img_id={img_id}: {e}")
            errors += 1

    print(f"Done: {success} encoded, {errors} errors, saved to {output_dir}")

    # Decode back samples for inspection
    if args.decode_back:
        print(f"\nDecoding {args.samples} samples for visual inspection...")
        sample_dir = Path("public/datasets/coco/recon_samples")
        sample_dir.mkdir(parents=True, exist_ok=True)

        for entry in img_entries[:args.samples]:
            img_id = entry["img_id"]
            token_path = output_dir / f"{img_id:012d}.pt"
            if not token_path.exists():
                continue

            tokens = torch.load(token_path)
            recon = wrapper.decode(tokens)
            recon.save(sample_dir / f"{img_id:012d}_recon.png")

            # Also save original for comparison
            orig = Image.open(entry["image_path"]).convert("RGB").resize((256, 256))
            orig.save(sample_dir / f"{img_id:012d}_orig.png")

        print(f"Reconstruction samples saved to {sample_dir}")


if __name__ == "__main__":
    main()
