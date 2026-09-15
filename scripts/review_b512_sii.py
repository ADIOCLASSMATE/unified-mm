"""Review every selected image via SII, with an independently qualified pilot."""
import argparse
import asyncio
import json
from pathlib import Path

from data_synthesis.config import load_config
from data_synthesis.review_farm import prepare_pilot, run_farm


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["sample", "pilot", "bulk"])
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--selection", type=Path, default=Path("configs/protocols/unified_b_prepared_frozen_20260914.json"))
    p.add_argument("--qualification", type=Path)
    args = p.parse_args()
    config = load_config(args.config)
    if args.mode == "sample":
        print(prepare_pilot(args.selection, args.root))
        return 0
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer"], local_files_only=True)
    result = asyncio.run(run_farm(args, config, tokenizer))
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0 if result["state"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
