"""A fixed text probe and explicit, resumable training-row exclusions.

Source identities are shard paths and JSONL byte offsets. No corpus hashing or
network access is performed. A newly reserved subset is not retroactively held
out from checkpoints trained before exclusions were enabled.
"""
from __future__ import annotations

import json
from pathlib import Path


DEFAULT_MANIFEST = "public/datasets/climbmix_validation_v1/manifest.json"
SCHEMA = "climbmix_validation_subset_v1"


def load_validation_manifest(path):
    path = Path(path).resolve()
    data = json.loads(path.read_text())
    if data.get("schema") != SCHEMA or not data.get("records"):
        raise ValueError(f"invalid ClimbMix validation manifest: {path}")
    records = data["records"]
    identities = [(r["shard"], r["byte_offset"]) for r in records]
    if len(set(identities)) != len(records):
        raise ValueError("duplicate ClimbMix validation records")
    shards = {}
    for source in data["shards"]:
        shard = (path.parent / source["path"]).resolve()
        stat = shard.stat()
        if stat.st_size != source["bytes"] or stat.st_mtime_ns != source["mtime_ns"]:
            raise ValueError(f"ClimbMix validation source changed: {shard}")
        shards[source["id"]] = shard
    excluded = {str(p): set() for p in shards.values()}
    for row in records:
        offset = row["byte_offset"]
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ValueError("invalid validation byte offset")
        shard = shards[row["shard"]]
        if offset >= shard.stat().st_size:
            raise ValueError("validation byte offset exceeds source")
        excluded[str(shard)].add(offset)
    stat = path.stat()
    contract = {"manifest": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    return data, shards, excluded, contract


def validation_documents(manifest_path):
    data, shards, _, contract = load_validation_manifest(manifest_path)
    documents = []
    handles = {}
    try:
        for row in data["records"]:
            name = row["shard"]
            handle = handles.setdefault(name, None)
            if handle is None:
                handle = handles[name] = shards[name].open("rb")
            handle.seek(row["byte_offset"])
            record = json.loads(handle.readline())
            text = record.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError("validation document has no nonempty text")
            documents.append({"id": f"{name}:{row['byte_offset']}", "text": text})
    finally:
        for handle in handles.values():
            if handle is not None:
                handle.close()
    return documents, contract
