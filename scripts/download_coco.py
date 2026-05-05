"""
Download COCO 2017 train/val images and captions.

Uses HuggingFace datasets: lmms-lab/COCO-Caption2017
Saves images to public/datasets/coco/images/ and captions to JSONL.

Usage:
    uv run python scripts/download_coco.py --max_images 30000
    uv run python scripts/download_coco.py --max_images 100  # quick test
"""

import argparse
import json
import os
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Download COCO 2017 images + captions")
    parser.add_argument("--max_images", type=int, default=30000,
                        help="Maximum number of images to download (default: 30000 for Phase 1)")
    parser.add_argument("--output_dir", type=str,
                        default="public/datasets/coco",
                        help="Output directory for images and captions")
    parser.add_argument("--split", type=str, default="train",
                        help="Dataset split: train or validation")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    image_dir = output_dir / "images"
    caption_file = output_dir / "captions.jsonl"
    image_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading COCO-Caption2017 ({args.split} split)...")
    ds = load_dataset("lmms-lab/COCO-Caption2017", split=args.split, streaming=True)

    img_to_captions = {}
    count = 0

    for item in tqdm(ds, desc="Processing captions", total=args.max_images * 5):
        img_id = item["image_id"]
        caption = item["caption"]

        if img_id not in img_to_captions:
            if len(img_to_captions) >= args.max_images:
                break
            img_to_captions[img_id] = {"image": item["image"], "captions": []}

        img_to_captions[img_id]["captions"].append(caption)

    print(f"Collected {len(img_to_captions)} unique images with captions")

    # Save images and captions
    with open(caption_file, "w") as f:
        for img_id, data in tqdm(img_to_captions.items(), desc="Saving images + captions"):
            # Save image
            img_path = image_dir / f"{img_id:012d}.jpg"
            if data["image"].mode == "RGBA":
                data["image"] = data["image"].convert("RGB")
            data["image"].save(img_path)

            # Write captions JSONL
            f.write(json.dumps({
                "img_id": img_id,
                "captions": data["captions"],
                "image_path": str(img_path),
            }) + "\n")

    print(f"Done. {len(img_to_captions)} images saved to {image_dir}")
    print(f"Captions saved to {caption_file}")


if __name__ == "__main__":
    main()
