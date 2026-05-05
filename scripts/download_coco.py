"""
Download COCO images and captions for multimodal training.

Uses HuggingFace datasets: jxie/coco_captions (COCO 2014 train, ~414K captions)
Saves images to public/datasets/coco/images/ and captions to JSONL.

Usage:
    uv run python scripts/download_coco.py --max_images 100   # quick test
    uv run python scripts/download_coco.py --max_images 30000 # Phase 1
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Download COCO images + captions")
    parser.add_argument("--max_images", type=int, default=30000,
                        help="Maximum number of unique images (default: 30000)")
    parser.add_argument("--output_dir", type=str,
                        default="public/datasets/coco",
                        help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    image_dir = output_dir / "images"
    caption_file = output_dir / "captions.jsonl"
    image_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading jxie/coco_captions (train split) via streaming...")
    ds = load_dataset("jxie/coco_captions", split="train", streaming=True)

    # Group captions by image while streaming
    # We store captions but NOT images in memory (save images immediately)
    # This avoids memory blowup from PIL images
    img_info = {}  # img_id -> {"captions": [...], "saved": bool}
    save_count = 0

    for item in tqdm(ds, desc="Processing", total=args.max_images * 5):
        img_id = item["cocoid"]
        caption = item["caption"]

        if img_id not in img_info:
            if len(img_info) >= args.max_images:
                # Check if all images saved, if so, break
                if save_count >= args.max_images:
                    break
                # Otherwise continue adding captions for images we've seen
            else:
                img_info[img_id] = {"captions": [], "image": item["image"],
                                     "saved": False}

        if img_id in img_info:
            img_info[img_id]["captions"].append(caption)

            # Save image if new and we have enough captions
            if not img_info[img_id]["saved"] and len(img_info[img_id]["captions"]) >= 5:
                data = img_info[img_id]
                img_path = image_dir / f"{img_id:012d}.jpg"

                try:
                    img = data["image"]
                    if img.mode == "RGBA":
                        img = img.convert("RGB")
                    img.save(img_path)

                    with open(caption_file, "a") as f:
                        f.write(json.dumps({
                            "img_id": img_id,
                            "captions": data["captions"],
                            "image_path": str(img_path.absolute()),
                        }) + "\n")

                    data["saved"] = True
                    save_count += 1

                    # Free image from memory
                    del data["image"]
                except Exception as e:
                    print(f"\n  Error saving {img_id}: {e}")

    print(f"\nDone! {save_count} images saved to {image_dir}")
    print(f"Captions saved to {caption_file}")

    if save_count < args.max_images:
        print(f"Warning: Only found {save_count} images (requested {args.max_images})")


if __name__ == "__main__":
    main()
