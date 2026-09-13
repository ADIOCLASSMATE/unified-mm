#!/usr/bin/env python3
"""Paired, resumable image-grounded probes of the user's SII model routes."""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import replace
import hashlib
import io
import json
import os
from pathlib import Path
import re
import time

from anthropic import AsyncAnthropic
import httpx
from PIL import Image

from utils.direct_network import direct_ssl_context
from utils.image_shard_io import read_image_bytes
from utils.image_text_teacher import QwenGenerator, SCHEMA, load_qwen_settings

REPO = Path(__file__).resolve().parents[1]
PUBLIC = (REPO / "public").resolve()
ROOT = PUBLIC / "data_preparation/unified_b_corners_api_v3/model_selection_20260913"
IMAGES = PUBLIC / "datasets/unified_image_pool_512_v1/api_model_selection_v3"
MODELS = ("qwen3.8-max", "deepseek-v4-pro-0813")


def atomic_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


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


def parsed_result(raw, image_id):
    import jsonschema
    text = raw["output_text"].strip()
    if text.startswith("```json\n") and text.endswith("\n```"):
        text = text[8:-4]
    elif text.startswith("```\n") and text.endswith("\n```"):
        text = text[4:-4]
    value = json.loads(text)
    jsonschema.validate(value, SCHEMA)
    assert value["image_id"] == image_id
    assert raw["status"] == "completed"
    return value


async def run(args):
    # These environment edits only affect this child process. Keep mihomo and
    # the parent Codex connection untouched; transport also bypasses proxy env.
    for key in list(os.environ):
        if key.lower() in {"http_proxy", "https_proxy", "all_proxy", "ftp_proxy", "socks_proxy", "no_proxy"}:
            os.environ.pop(key)
    os.environ.update(NO_PROXY="*", no_proxy="*")
    root = args.root.resolve()
    samples = prepare(root)
    if args.mode == "smoke":
        samples = [next(row for row in samples if row["group"] == group)
                   for group in ("english_text", "painting")]
    settings = load_qwen_settings(REPO / "test_api.py")
    if args.thinking == "disabled":
        settings = replace(settings, thinking={"type": "disabled"})
    settings = replace(settings, max_tokens=args.max_tokens)
    request_contract = {"transport": args.transport, "thinking": settings.thinking,
                        "max_tokens": settings.max_tokens, "prompt_version": "b512-paired-pilot-v3"}
    atomic_json(root / "protocol.json", {
        "models": list(MODELS), "api_example": str(REPO / "test_api.py"),
        "credential_source": "AST-read original SII key/URL; key never persisted", "proxy": "disabled_per_process",
        "image_size": 512, "image_tokens_training": 1024, "requested_thinking": settings.thinking,
        "max_tokens": settings.max_tokens, "quality_reviewer": "GPT-6 in the current Codex session, real image inspection",
        "audit_mode": "strict transport, schema and provenance; one-time visual selection review",
        "full_corpus_per_item_sol_review": False, "pilot_images": 32,
        "production_allowed": False,
        "request_contract": request_contract,
    })
    lock = asyncio.Semaphore(args.concurrency)
    generators = {model: QwenGenerator(replace(settings, model=model), rpm=6000, tpm=20000000,
                                       connections=args.concurrency) for model in args.models}
    if args.transport != "native-curl":
        for generator in generators.values():
            await generator.close()
            transport = httpx.AsyncClient(
                trust_env=False, proxy=None, verify=direct_ssl_context(),
                http2=args.transport == "pooled-http2", follow_redirects=False,
                limits=httpx.Limits(max_connections=args.concurrency, max_keepalive_connections=args.concurrency),
                timeout=httpx.Timeout(300, connect=15),
            )
            generator.client = AsyncAnthropic(base_url=settings.base_url, api_key=settings.api_key,
                                               max_retries=0, http_client=transport)

    async def one(model, sample):
        path = root / "responses" / model / (sample["id"] + ".json")
        if path.exists():
            prior = json.loads(path.read_text())
            if prior.get("request_succeeded") and prior.get("request_contract") == request_contract:
                return
            archive = path.parent / "attempts" / (sample["id"] + f"-{time.time_ns()}.json")
            archive.parent.mkdir(exist_ok=True)
            path.replace(archive)
        async with lock:
            started = time.time()
            output = {"requested_model": model, "sample_id": sample["id"], "group": sample["group"],
                      "view_sha256": sample["view_sha256"], "started_at": started,
                      "request_contract": request_contract}
            try:
                image_path = Path(sample["image_path"])
                raw = await generators[model].generate(sample["id"], image_path.read_bytes(), image_path.suffix[1:])
                raw["transport"] = args.transport
                output.update(raw=raw, request_succeeded=True)
                try:
                    output.update(parsed=parsed_result(raw, sample["id"]), schema_passed=True)
                except Exception as exc:
                    output.update(schema_passed=False, schema_error=f"{type(exc).__name__}: {str(exc)[:500]}")
            except Exception as exc:
                message = str(exc).replace(settings.api_key, "[REDACTED]")
                output.update(request_succeeded=False, error_type=type(exc).__name__, error=message[:1600])
                causes = []
                cause = exc.__cause__
                while cause is not None and len(causes) < 4:
                    causes.append(type(cause).__name__ + ": " + str(cause).replace(settings.api_key, "[REDACTED]")[:1000])
                    cause = cause.__cause__
                output["error_causes"] = causes
            output["seconds"] = round(time.time() - started, 3)
            atomic_json(path, output)
            print(json.dumps({k: output[k] for k in ("requested_model", "sample_id", "group", "request_succeeded", "seconds")}
                             | {"schema_passed": output.get("schema_passed"), "error": output.get("error")}, ensure_ascii=False), flush=True)
    try:
        await asyncio.gather(*(one(model, sample) for sample in samples for model in args.models))
    finally:
        await asyncio.gather(*(generator.close() for generator in generators.values()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("smoke", "pilot"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--transport", choices=("native-curl", "pooled-http1", "pooled-http2"), default="pooled-http1")
    parser.add_argument("--thinking", choices=("example", "disabled"), default="example")
    parser.add_argument("--max-tokens", type=int, default=3200)
    args = parser.parse_args()
    asyncio.run(run(args))
