#!/usr/bin/env python3
"""Bounded real-image probes through the same shell-configured SII client as production."""
import argparse
import asyncio
import copy
import hashlib
import io
import json
from pathlib import Path
import re
import time

from PIL import Image
from data_synthesis.clients import SIIClient
from data_synthesis.config import DEFAULT_CONFIG, load_config, load_sii_settings, fingerprint
from data_synthesis.contract import CONTRACT_HASH, parse_pair
from data_synthesis.io import atomic_json
from utils.image_shard_io import read_image_bytes

REPO = Path(__file__).resolve().parents[1]
PUBLIC = (REPO / "public").resolve()
ROOT = PUBLIC / "data_preparation/unified_b_corners_api_v3/model_selection_20260913"
IMAGES = PUBLIC / "datasets/unified_image_pool_512_v3/api_model_selection"
MODELS = ("qwen3.8-max", "deepseek-v4-pro-0813")


def prepare(root, per_group=4):
    destination = root / "samples.json"
    if destination.exists():
        return json.loads(destination.read_text())
    groups = {name: [] for name in ("relations", "count_2_4", "count_5_plus", "english_text",
                                   "chinese_text", "painting", "graphic_style", "uncertain_scene")}
    manifest = PUBLIC / "datasets/unified_image_text_512_sol_v1/releases/sol_100k_plus_corners_v1/manifest.jsonl"
    for line in manifest.open():
        row = json.loads(line)
        if row["source"] == "imagenet":
            continue
        obs = row.get("observations", {})
        counts = [c["count"] for c in obs.get("counts", []) if type(c.get("count")) is int]
        text = " ".join(obs.get("visible_text", []))
        cjk = bool(re.search(r"[\u4e00-\u9fff]", text))
        eligible = []
        if row["source"] == "wikiart":
            eligible.append("painting")
        elif row["selection_bucket"] == "style":
            eligible.append("graphic_style")
        elif cjk:
            eligible.append("chinese_text")
        elif row["selection_bucket"] == "textocr_readable" and len(text) >= 12:
            eligible.append("english_text")
        elif row["source"] == "openimages" and len(obs.get("relations", [])) >= 2:
            eligible.append("relations")
        elif any(c >= 5 for c in counts):
            eligible.append("count_5_plus")
        elif any(2 <= c <= 4 for c in counts):
            eligible.append("count_2_4")
        elif row.get("uncertainties"):
            eligible.append("uncertain_scene")
        for group in eligible:
            score = hashlib.sha256(("20260913-v3:" + row["key"]).encode()).hexdigest()
            groups[group].append((score, row))
            groups[group].sort(key=lambda x: x[0])
            del groups[group][per_group:]
    items = []
    for group, selected in groups.items():
        if len(selected) != per_group:
            raise ValueError(f"insufficient distinct examples for {group}: {len(selected)}")
        for _, row in selected:
            data = read_image_bytes(row["source_path"])
            with Image.open(io.BytesIO(data)) as image:
                image.load()
                assert image.size == (512, 512) and image.mode == "RGB"
            assert hashlib.sha256(data).hexdigest() == row["view_sha256"]
            item_id = f"sample-{len(items):03d}"
            path = IMAGES / (item_id + "." + row["extension"])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            items.append({"id": item_id, "group": group, "source": row["source"], "key": row["key"],
                          "image_path": str(path), "view_sha256": row["view_sha256"],
                          "selection_note": "Stratified by old metadata; old text/observations never sent to API or used as ground truth."})
    atomic_json(destination, items)
    return items


async def run(args):
    from transformers import AutoTokenizer
    config = load_config(args.config)
    settings = load_sii_settings()
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer"], local_files_only=True)
    args.root.mkdir(parents=True, exist_ok=True)
    samples = prepare(args.root)
    if args.mode == "smoke":
        samples = [next(s for s in samples if s["group"] == group) for group in ("english_text", "painting")]
    policy = {"contract_hash": CONTRACT_HASH, "runtime_sha256": fingerprint(config), "image_size": 512,
              "credential_source": "SII_API_KEY/SII_BASE_URL", "proxy": False, "models": args.models,
              "production_allowed": False, "visual_review_completed": False}
    atomic_json(args.root / "protocol.json", policy)
    semaphore = asyncio.Semaphore(args.concurrency)
    clients = {}
    for model in args.models:
        api = copy.deepcopy(config["sii"])
        api["vision_models"] = [model]  # Explicit diagnostic, not a production qualification.
        clients[model] = SIIClient(settings, api)
    async def one(model, sample):
        path = args.root / "responses" / model / (sample["id"] + ".json")
        if path.exists():
            old = json.loads(path.read_text())
            if old.get("policy") == policy and old.get("schema_passed") and old.get("view_sha256") == sample["view_sha256"]:
                return
            target = path.parent / "attempts" / (sample["id"] + f"-{time.time_ns()}.json")
            target.parent.mkdir(parents=True, exist_ok=True)
            path.replace(target)
        item = {"key": sample["id"], "row": {}, "view": {"source_path": sample["image_path"], "view_sha256": sample["view_sha256"]}}
        async with semaphore:
            raw = await clients[model].generate(item, 1)
        output = {"policy": policy, "requested_model": model, "sample_id": sample["id"], "group": sample["group"],
                  "view_sha256": sample["view_sha256"], "raw": raw, "request_succeeded": raw["status"] == "completed"}
        atomic_json(path, output)  # Durable raw response before parsing / acceptance.
        try:
            output.update(parsed=parse_pair(raw, sample["id"], tokenizer), schema_passed=True)
        except Exception as exc:
            output.update(schema_passed=False, error=settings.redact(str(exc))[:500])
        atomic_json(path, output)
        print(json.dumps({k:output[k] for k in ("requested_model","sample_id","schema_passed")}), flush=True)
    try:
        await asyncio.gather(*(one(m,s) for m in args.models for s in samples))
    finally:
        await asyncio.gather(*(client.close() for client in clients.values()))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("smoke","pilot"))
    parser.add_argument("--root",type=Path,default=ROOT)
    parser.add_argument("--config",type=Path,default=DEFAULT_CONFIG)
    parser.add_argument("--models",nargs="+",default=["qwen3.8-max"])
    parser.add_argument("--concurrency",type=int,default=4)
    args=parser.parse_args()
    if args.concurrency<1:parser.error("concurrency must be positive")
    asyncio.run(run(args))
