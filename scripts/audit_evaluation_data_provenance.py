#!/usr/bin/env python3
"""Count the actual synthetic-data provenance once, without hashing or inference."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import UTC, datetime
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def audit(repo: Path, destination: Path):
    base = repo / "public/datasets/imagenet1k_synthetic_v1"
    result = {"schema": "evaluation_data_provenance_audit_v1",
              "audited_at": datetime.now(UTC).isoformat(timespec="seconds"),
              "runtime_hashing_enabled": False, "files": [], "t2i": {}, "captions": {}}

    def identity(path):
        stat = path.stat()
        return {"path": str(path.relative_to(repo)), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}

    for split in ("train", "val"):
        index = base / "indexed" / split / "manifest.json"
        manifest = json.loads(index.read_text())
        result["files"].append(identity(index))
        counts, examples, records, bad_styles = Counter(), [], 0, 0
        styles = json.loads((base / "t2i/manifest.json").read_text())["styles"]
        for shard in manifest["t2i"]["shards"]:
            path = index.parent / shard["path"]
            before = identity(path)
            n = 0
            with path.open() as handle:
                for line in handle:
                    row = json.loads(line)
                    generation = row["generation"]
                    counts[(generation.get("model"), generation.get("reasoning_effort"), generation.get("prompt_version"))] += 1
                    bad_styles += [p["style"] for p in row["model_result"]["prompts"]] != styles
                    if row["split"] != split:
                        raise ValueError(f"split mismatch: {path}")
                    if not examples:
                        examples.append(row)
                    n += 1
            if n != shard["records"] or before != identity(path):
                raise ValueError(f"count mismatch or changed source: {path}")
            result["files"].append(before)
            records += n
        result["t2i"][split] = {"records": records, "models": [
            {"model": k[0], "reasoning_effort": k[1], "prompt_version": k[2], "records": v}
            for k, v in counts.items()], "style_order_mismatches": bad_styles, "examples": examples}
        if records != manifest["records"] or bad_styles:
            raise ValueError(f"invalid T2I {split} coverage/styles")
        print(f"T2I {split}: {records:,} records, {dict(counts)}", flush=True)

        path = (index.parent / manifest["caption"]["path"]).absolute()
        before = identity(path)
        counts, records, examples, caption_counts = Counter(), 0, [], Counter()
        with path.open() as handle:
            for line in handle:
                row = json.loads(line)
                for caption in row["captions"]:
                    counts[(caption.get("source"), caption.get("model"), caption.get("prompt_version"))] += 1
                caption_counts[len(row["captions"])] += 1
                if not examples:
                    examples.append(row)
                records += 1
        if records != manifest["records"] or before != identity(path):
            raise ValueError(f"caption coverage mismatch or changed source: {path}")
        result["files"].append(before)
        result["captions"][split] = {"records": records, "captions_per_image": dict(caption_counts),
            "sources": [{"source": k[0], "model": k[1], "prompt_version": k[2], "captions": v}
                        for k, v in counts.items()], "examples": examples}
        print(f"Caption {split}: {records:,} records, {dict(counts)}", flush=True)
    result["complete"] = True
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(destination)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPO / "output/evaluation/data-provenance/audit.json")
    args = parser.parse_args()
    audit(REPO, args.output)
