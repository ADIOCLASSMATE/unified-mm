"""
Prepare an OmniCorpus snapshot for training.

This script intentionally keeps the Arrow format head-agnostic:
    - text/special tokens use tokenizer ids
    - image tokens are raw Open-MAGVIT2 codebook ids in [0, image_vocab_size)
    - token_types identify text/image/special/padding

That single Arrow output works for both unified lm_head and separate image
lm_head. The model/dataloader applies image_offset only at runtime when
unified_head=true.

Example:
    uv run python scripts/omnicorpus_preprocess_training.py \
        --docs_jsonl public/datasets/omnicorpus/docs/snapshots/train_20260623_064740_2946703docs.jsonl \
        --max_docs 2000 \
        --work_dir public/datasets/omnicorpus/pretrain_smoke
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Set


def copy_subset(docs_jsonl: Path, subset_jsonl: Path, image_dir: Path, max_docs: int) -> Dict[str, int]:
    subset_jsonl.parent.mkdir(parents=True, exist_ok=True)

    written_docs = 0
    scanned_docs = 0
    skipped_missing_images = 0
    image_ids: Set[int] = set()

    with docs_jsonl.open() as src, subset_jsonl.open("w") as dst:
        for line in src:
            if max_docs > 0 and written_docs >= max_docs:
                break
            if not line.strip():
                continue
            scanned_docs += 1
            record = json.loads(line)
            ids = [int(x) for x in record.get("img_ids", [])]
            if not ids:
                continue
            missing = [img_id for img_id in ids if not (image_dir / f"{img_id:012d}.jpg").exists()]
            if missing:
                skipped_missing_images += 1
                continue
            dst.write(json.dumps(record, ensure_ascii=False) + "\n")
            written_docs += 1
            image_ids.update(ids)

    if written_docs == 0:
        raise RuntimeError(f"No usable docs were written from {docs_jsonl}")

    return {
        "scanned_docs": scanned_docs,
        "written_docs": written_docs,
        "unique_images": len(image_ids),
        "skipped_missing_images": skipped_missing_images,
    }


def run(cmd, env=None) -> None:
    print("+ " + " ".join(str(x) for x in cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/selfless/omnicorpus.yaml")
    parser.add_argument("--docs_jsonl", default="public/datasets/omnicorpus/docs/train.jsonl")
    parser.add_argument("--image_dir", default="public/datasets/omnicorpus/images")
    parser.add_argument("--work_dir", default="public/datasets/omnicorpus/pretrain_smoke")
    parser.add_argument("--subset_jsonl", default=None)
    parser.add_argument("--image_token_dir", default=None)
    parser.add_argument("--arrow_dir", default=None)
    parser.add_argument("--max_docs", type=int, default=2000,
                        help="Number of usable docs to include. Use -1 for all docs.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard_size", type=int, default=10000)
    parser.add_argument("--skip_encode", action="store_true")
    parser.add_argument("--skip_arrow", action="store_true")
    parser.add_argument("--reuse_subset", action="store_true")
    args = parser.parse_args()

    docs_jsonl = Path(args.docs_jsonl)
    image_dir = Path(args.image_dir)
    work_dir = Path(args.work_dir)
    subset_jsonl = Path(args.subset_jsonl) if args.subset_jsonl else work_dir / "docs.jsonl"
    image_token_dir = Path(args.image_token_dir) if args.image_token_dir else work_dir / "image_tokens_magvit2"
    arrow_dir = Path(args.arrow_dir) if args.arrow_dir else work_dir / "arrow"
    metadata_path = work_dir / "preprocess_metadata.json"

    work_dir.mkdir(parents=True, exist_ok=True)

    if args.reuse_subset and subset_jsonl.exists():
        subset_stats = {"reused_subset": 1}
        print(f"Reusing subset: {subset_jsonl}")
    else:
        subset_stats = copy_subset(docs_jsonl, subset_jsonl, image_dir, args.max_docs)
        print(f"Subset written: {subset_stats}")

    if not args.skip_encode:
        run([
            sys.executable,
            "scripts/omnicorpus_encode_images.py",
            "--docs_jsonl", str(subset_jsonl),
            "--image_dir", str(image_dir),
            "--output_dir", str(image_token_dir),
            "--device", args.device,
        ])

    if not args.skip_arrow:
        run([
            sys.executable,
            "scripts/omnicorpus_build_arrow.py",
            "--config", args.config,
            "--docs_jsonl", str(subset_jsonl),
            "--image_token_dir", str(image_token_dir),
            "--output_dir", str(arrow_dir),
            "--shard_size", str(args.shard_size),
        ])

    metadata = {
        "source_docs_jsonl": str(docs_jsonl),
        "subset_jsonl": str(subset_jsonl),
        "image_dir": str(image_dir),
        "image_token_dir": str(image_token_dir),
        "arrow_dir": str(arrow_dir),
        "max_docs": args.max_docs,
        "subset_stats": subset_stats,
        "format": {
            "input_ids": "text/special tokenizer ids plus raw Open-MAGVIT2 image code ids",
            "token_types": {"0": "text", "1": "image", "2": "special", "3": "padding"},
            "image_token_format": "raw_codebook_ids_no_offset",
            "compatible_lm_heads": ["unified_head", "separate_text_image_heads"],
            "unified_head_image_offset_applied_at": "model forward/loss, not preprocessing",
        },
    }
    with metadata_path.open("w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata written: {metadata_path}")
    print(f"Ready Arrow dir: {arrow_dir}")


if __name__ == "__main__":
    main()
