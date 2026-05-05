"""
Preprocess FineWeb-Edu raw parquet data for Level 3 (text-only) training.

Filters by quality score, chunks long documents, outputs JSONL.

Usage:
    uv run python scripts/preprocess_fineweb.py --max_docs 1000 --output_dir /tmp/test_synth/
    uv run python scripts/preprocess_fineweb.py  # Full Phase 1
"""

import argparse
import json
import os
import random
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        default="/inspire/dataset/fineweb-edu/v1/sample/10BT",
        help="Directory containing FineWeb-Edu parquet files",
    )
    parser.add_argument(
        "--output_dir",
        default="public/datasets/fineweb/phase1",
    )
    parser.add_argument("--max_docs", type=int, default=0,
                        help="Max documents to process (0 = unlimited)")
    parser.add_argument("--min_score", type=float, default=2.5,
                        help="Minimum quality score filter")
    parser.add_argument("--chunk_min_chars", type=int, default=500,
                        help="Minimum characters per chunk")
    parser.add_argument("--chunk_max_chars", type=int, default=4096,
                        help="Approximate maximum characters per chunk")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find parquet files
    parquet_files = sorted(data_dir.glob("*.parquet"))
    if not parquet_files:
        # Try subdirectories
        parquet_files = sorted(data_dir.rglob("*.parquet"))

    print(f"Found {len(parquet_files)} parquet files in {data_dir}")

    output_path = output_dir / "train.jsonl"
    total_chunks = 0
    total_docs = 0

    with open(output_path, "w") as out_f:
        for pq_file in tqdm(parquet_files, desc="Processing parquet files"):
            df = pd.read_parquet(pq_file)

            # Filter
            df = df[df["score"] >= args.min_score]
            if "language" in df.columns:
                df = df[df["language"] == "en"]

            for _, row in df.iterrows():
                if args.max_docs > 0 and total_docs >= args.max_docs:
                    break

                text = row["text"]
                total_docs += 1

                # Split long documents into chunks
                chunks = split_into_chunks(
                    text,
                    min_chars=args.chunk_min_chars,
                    max_chars=args.chunk_max_chars,
                )

                for chunk in chunks:
                    out_f.write(json.dumps({
                        "task_mode": "text_only",
                        "text": chunk,
                    }) + "\n")
                    total_chunks += 1

            if args.max_docs > 0 and total_docs >= args.max_docs:
                break

    print(f"Done: {total_docs} documents -> {total_chunks} chunks")
    print(f"Saved to {output_path}")


def split_into_chunks(
    text: str,
    min_chars: int = 500,
    max_chars: int = 4096,
) -> list:
    """Split long text into reasonably-sized chunks at paragraph boundaries."""
    if len(text) <= max_chars:
        return [text] if len(text) >= min_chars else []

    chunks = []
    paragraphs = text.split("\n\n")
    current = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(current) + len(para) + 2 <= max_chars:
            current = (current + "\n\n" + para) if current else para
        else:
            if len(current) >= min_chars:
                chunks.append(current)
            current = para

    if len(current) >= min_chars:
        chunks.append(current)

    return chunks


if __name__ == "__main__":
    main()
