#!/usr/bin/env python3
"""Reserve fixed JSONL records by byte position without scanning the corpus."""
from __future__ import annotations

import argparse
from datetime import UTC, datetime
import glob
import json
import os
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.climbmix_validation_data import DEFAULT_MANIFEST, SCHEMA, load_validation_manifest


def prepare(shard_glob, output, *, per_shard=4, seed=424242):
    output = Path(output).resolve()
    if output.exists():
        data, _, _, _ = load_validation_manifest(output)
        if data["seed"] != seed or data["records_per_shard"] != per_shard:
            raise ValueError("existing subset uses different parameters; select a new output path")
        if sorted(str(p.resolve()) for p in map(Path, glob.glob(shard_glob))) != sorted(
                str((output.parent / s["path"]).resolve()) for s in data["shards"]):
            raise ValueError("existing subset uses different source shards")
        return data
    paths = sorted(Path(p).resolve() for p in glob.glob(shard_glob))
    if not paths or per_shard <= 0:
        raise ValueError("require source shards and a positive record count")
    rng = random.Random(seed)
    sources, records = [], []
    for index, path in enumerate(paths):
        stat = path.stat()
        sources.append({"id": str(index), "path": os.path.relpath(path, output.parent),
                        "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns})
        offsets = set()
        with path.open("rb") as handle:
            for _ in range(per_shard * 100):
                if len(offsets) == per_shard:
                    break
                probe = rng.randrange(stat.st_size)
                handle.seek(probe)
                if probe:
                    handle.readline()
                offset = handle.tell()
                raw = handle.readline()
                if not raw or offset in offsets:
                    continue
                row = json.loads(raw)
                if not isinstance(row.get("text"), str) or not row["text"].strip():
                    continue
                offsets.add(offset)
        if len(offsets) != per_shard:
            raise ValueError(f"not enough distinct records in {path}")
        records.extend({"shard": str(index), "byte_offset": offset} for offset in sorted(offsets))
    data = {"schema": SCHEMA, "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "source": "OptimalScale/ClimbMix", "seed": seed, "records_per_shard": per_shard,
            "sampling": "Uniform byte positions per shard; select the following complete nonempty JSONL record. This is a fixed probe, not document-uniform sampling.",
            "independence": "Held out only for training that explicitly excludes these rows from the start; earlier checkpoints may have seen them. Duplicate text at other offsets is not removed.",
            "shards": sources, "records": records}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(output)
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-glob", default="public/ClimbMix/*.jsonl")
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_MANIFEST))
    parser.add_argument("--records-per-shard", type=int, default=4)
    parser.add_argument("--seed", type=int, default=424242)
    args = parser.parse_args()
    result = prepare(args.shard_glob, args.output, per_shard=args.records_per_shard, seed=args.seed)
    print(json.dumps({"manifest": str(args.output), "records": len(result["records"]), "shards": len(result["shards"])}))
