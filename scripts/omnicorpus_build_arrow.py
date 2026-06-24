"""
Build pre-tokenized Arrow shards for OmniCorpus-only multimodal pretraining.

Each output row is one document:
    text [BOI] image_tokens [EOI] text ... [EOS]

Image spans are never sliced. If a document is too long, text is trimmed first;
then complete image blocks are dropped from the end until the sequence fits.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Tuple

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import torch
from datasets import Dataset as HFDataset
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import AutoTokenizer


def load_tokenizer(config_path: str):
    config = OmegaConf.load(config_path)
    tokenizer = AutoTokenizer.from_pretrained(config.model.model_path, fix_mistral_regex=True)
    if "<|mdm_mask|>" not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"mask_token": "<|mdm_mask|>"})
    if "<|boi|>" not in tokenizer.get_vocab():
        tokenizer.add_tokens(["<|boi|>", "<|eoi|>"], special_tokens=True)
    return config, tokenizer


def validate_spans(ids: List[int], types: List[int], boi_id: int, eoi_id: int, image_tokens_per_img: int):
    i = 0
    while i < len(types):
        if types[i] != 1:
            i += 1
            continue
        start = i
        while i < len(types) and types[i] == 1:
            i += 1
        end = i
        if end - start != image_tokens_per_img:
            raise ValueError(f"bad image span length {end - start}")
        if start == 0 or ids[start - 1] != boi_id:
            raise ValueError("image span missing BOI")
        if end >= len(ids) or ids[end] != eoi_id:
            raise ValueError("image span missing EOI")


def truncate_preserving_images(
    ids: List[int],
    types: List[int],
    max_seq_length: int,
    boi_id: int,
    eoi_id: int,
    image_tokens_per_img: int,
) -> Tuple[List[int], List[int]]:
    while len(ids) > max_seq_length:
        longest = None
        i = 0
        while i < len(types):
            if types[i] == 0:
                start = i
                while i < len(types) and types[i] == 0:
                    i += 1
                length = i - start
                if longest is None or length > longest[2]:
                    longest = (start, i, length)
            else:
                i += 1
        if longest is None or longest[2] <= 16:
            break
        start, _, length = longest
        remove = min(length - 16, len(ids) - max_seq_length)
        ids = ids[:start] + ids[start + remove:]
        types = types[:start] + types[start + remove:]

    while len(ids) > max_seq_length:
        blocks = []
        i = 0
        while i < len(types):
            if types[i] == 1:
                img_start = i
                while i < len(types) and types[i] == 1:
                    i += 1
                img_end = i
                block_start = img_start - 1 if img_start > 0 and ids[img_start - 1] == boi_id else img_start
                block_end = img_end + 1 if img_end < len(ids) and ids[img_end] == eoi_id else img_end
                blocks.append((block_start, block_end))
            else:
                i += 1
        if not blocks:
            ids = ids[:max_seq_length]
            types = types[:max_seq_length]
            break
        block_start, block_end = blocks[-1]
        ids = ids[:block_start] + ids[block_end:]
        types = types[:block_start] + types[block_end:]

    validate_spans(ids, types, boi_id, eoi_id, image_tokens_per_img)
    return ids, types


def ensure_terminal_eos(
    ids: List[int],
    types: List[int],
    eos_id: int,
    boi_id: int,
    eoi_id: int,
    image_tokens_per_img: int,
    max_seq_length: int,
) -> Tuple[List[int], List[int]]:
    if ids and ids[-1] == eos_id:
        return ids, types
    if len(ids) < max_seq_length:
        ids.append(eos_id)
        types.append(2)
        return ids, types

    for i in range(len(types) - 1, -1, -1):
        if types[i] == 0:
            ids = ids[:i] + ids[i + 1:] + [eos_id]
            types = types[:i] + types[i + 1:] + [2]
            validate_spans(ids, types, boi_id, eoi_id, image_tokens_per_img)
            return ids, types

    image_blocks = []
    i = 0
    while i < len(types):
        if types[i] == 1:
            img_start = i
            while i < len(types) and types[i] == 1:
                i += 1
            img_end = i
            block_start = img_start - 1 if img_start > 0 and ids[img_start - 1] == boi_id else img_start
            block_end = img_end + 1 if img_end < len(ids) and ids[img_end] == eoi_id else img_end
            image_blocks.append((block_start, block_end))
        else:
            i += 1
    if image_blocks:
        block_start, block_end = image_blocks[-1]
        ids = ids[:block_start] + ids[block_end:]
        types = types[:block_start] + types[block_end:]
        ids.append(eos_id)
        types.append(2)
        validate_spans(ids, types, boi_id, eoi_id, image_tokens_per_img)
        return ids, types

    ids[-1] = eos_id
    types[-1] = 2
    return ids, types


def build_record(record, tokenizer, image_token_dir: Path, cfg):
    eos_id = tokenizer.eos_token_id
    boi_id = tokenizer.convert_tokens_to_ids("<|boi|>")
    eoi_id = tokenizer.convert_tokens_to_ids("<|eoi|>")
    image_tokens_per_img = cfg.model.get("image_tokens_per_img", 256)
    max_seq_length = cfg.dataset.preprocessing.max_seq_length

    ids = []
    types = []
    img_ids = record["img_ids"]

    for seg in record["segments"]:
        if seg["type"] == "text":
            text_ids = tokenizer.encode(seg["content"], add_special_tokens=False)
            ids.extend(text_ids)
            types.extend([0] * len(text_ids))
        elif seg["type"] == "image":
            img_id = int(img_ids[int(seg["img_idx"])])
            token_path = image_token_dir / f"{img_id:012d}.pt"
            if not token_path.exists():
                raise FileNotFoundError(token_path)
            image_tokens = torch.load(token_path, map_location="cpu").long().view(-1)
            if image_tokens.numel() != image_tokens_per_img:
                raise ValueError(f"{token_path}: expected {image_tokens_per_img}, got {image_tokens.numel()}")
            ids.extend([boi_id] + image_tokens.tolist() + [eoi_id])
            types.extend([2] + [1] * image_tokens_per_img + [2])

    ids.append(eos_id)
    types.append(2)
    if len(ids) > max_seq_length:
        ids, types = truncate_preserving_images(
            ids, types, max_seq_length, boi_id, eoi_id, image_tokens_per_img
        )
    ids, types = ensure_terminal_eos(
        ids, types, eos_id, boi_id, eoi_id, image_tokens_per_img, max_seq_length
    )

    validate_spans(ids, types, boi_id, eoi_id, image_tokens_per_img)
    return {"input_ids": ids, "token_types": types}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/selfless/omnicorpus.yaml")
    parser.add_argument("--docs_jsonl", default="public/datasets/omnicorpus/docs/train.jsonl")
    parser.add_argument("--image_token_dir", default="public/datasets/omnicorpus/image_tokens_magvit2")
    parser.add_argument("--output_dir", default="public/datasets/omnicorpus/arrow")
    parser.add_argument("--shard_size", type=int, default=100000)
    parser.add_argument("--max_docs", type=int, default=-1)
    parser.add_argument("--num_shards", type=int, default=1,
                        help="Split source docs into this many deterministic build workers.")
    parser.add_argument("--shard_index", type=int, default=0,
                        help="Build only docs whose source ordinal modulo num_shards equals this index.")
    parser.add_argument("--output_prefix", default=None,
                        help="Output shard directory prefix. Defaults to shard- for single worker and shard-wNNN- for parallel workers.")
    args = parser.parse_args()

    if args.num_shards < 1:
        raise ValueError("--num_shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard_index must be in [0, num_shards)")

    cfg, tokenizer = load_tokenizer(args.config)
    docs_jsonl = Path(args.docs_jsonl)
    image_token_dir = Path(args.image_token_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    shard_records = []
    shard_idx = 0
    total_written = 0
    total_seen = 0
    output_prefix = args.output_prefix
    if output_prefix is None:
        output_prefix = "shard-" if args.num_shards == 1 else f"shard-w{args.shard_index:03d}-"

    def flush():
        nonlocal shard_records, shard_idx
        if not shard_records:
            return
        shard_path = output_dir / f"{output_prefix}{shard_idx:05d}"
        if shard_path.exists():
            raise FileExistsError(shard_path)
        ds = HFDataset.from_list(shard_records)
        ds.save_to_disk(str(shard_path))
        print(f"Saved {len(ds)} docs to {shard_path}")
        shard_records = []
        shard_idx += 1

    with docs_jsonl.open() as f:
        for line in tqdm(f, desc="Building Arrow"):
            if args.max_docs > 0 and total_seen >= args.max_docs:
                break
            if not line.strip():
                continue
            source_ordinal = total_seen
            total_seen += 1
            if source_ordinal % args.num_shards != args.shard_index:
                continue
            record = json.loads(line)
            try:
                shard_records.append(build_record(record, tokenizer, image_token_dir, cfg))
            except Exception as exc:
                print(f"Skipping doc {record.get('doc_id', '?')}: {exc}")
                continue
            total_written += 1
            if len(shard_records) >= args.shard_size:
                flush()
    flush()
    print(
        f"Done: {total_written} docs -> {output_dir} "
        f"(worker {args.shard_index}/{args.num_shards}, seen_source_docs={total_seen})"
    )


if __name__ == "__main__":
    main()
