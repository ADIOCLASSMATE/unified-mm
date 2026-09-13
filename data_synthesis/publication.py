"""Immutable, loader-compatible text publication and complete provenance audit."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import shutil
import sqlite3
import struct
import tempfile
import time

from data_synthesis.clients import frozen_pixels
from data_synthesis.contract import CONTRACT_HASH, parse_pair
from data_synthesis.io import atomic_json, dumps, file_sha, sha, training_image_id
from data_synthesis.pipeline import PIPELINE_VERSION
from data_synthesis.reuse import choose_reuse
from data_synthesis.state import State
from utils.image_shard_io import read_image_bytes
from utils.imagenet_synthetic_text_index import INDEX_SCHEMA, ImageNetSyntheticTextIndex


def audit(dataset, *, tokenizer, require_posterior=False, image_root=None):
    root = Path(dataset).resolve()
    publication = json.loads((root / "publication.json").read_text())
    if publication.get("pipeline") != PIPELINE_VERSION:
        raise ValueError("not a reuse/SII/fallback publication")
    for name, digest in publication["file_sha256"].items():
        if file_sha(root / name) != digest:
            raise ValueError(f"published file checksum mismatch: {name}")
    index = ImageNetSyntheticTextIndex(root / "text_index.json")
    counts, authors, sources = Counter(), Counter(), Counter()
    verified = 0
    with State(publication["state_root"], readonly=True) as state:
        state.db.execute("CREATE TEMP TABLE published_keys(key TEXT PRIMARY KEY)")
        config = state.meta("config")
        if state.meta("contract")["pair_contract"] != CONTRACT_HASH:
            raise ValueError("publication references a different pair contract")
        try:
            with (root / "manifest.jsonl").open() as handle:
                for offset, line in enumerate(handle):
                    view = json.loads(line)
                    try:
                        state.db.execute("INSERT INTO published_keys VALUES (?)", (view["key"],))
                    except sqlite3.IntegrityError as exc:
                        raise ValueError("duplicate image in publication") from exc
                    item = state.item(view["key"])
                    if item["status"] != "ready" or view["img_id"] != offset + 1 or view["split"] != "train":
                        raise ValueError("unready, misnumbered or non-train image")
                    if view["view_sha256"] != item["view_sha256"] or view["source_sha256"] != item["source_sha256"]:
                        raise ValueError("publication changed image identity")
                    if view["source_path"] != item["view"]["source_path"]:
                        raise ValueError("publication changed frozen image reference")
                    frozen_pixels(item)
                    if sha(read_image_bytes(view["original_ref"])) != view["source_sha256"]:
                        raise ValueError("original image checksum mismatch")
                    for ref in (view["original_ref"], view["source_path"]):
                        path = Path(ref[4:].rsplit("::", 1)[0] if ref.startswith("tar:") else ref).resolve()
                        if path.is_relative_to(root) or (image_root and not path.is_relative_to(Path(image_root).resolve())):
                            raise ValueError("image binaries must remain in the declared image pool")
                    pair = json.loads(item["result_json"])
                    evidence = json.loads(item["evidence_json"])
                    parse_pair({"status": "completed", "output_text": dumps(pair)}, item["key"], tokenizer)
                    state.check_test_prompts(pair)
                    route = evidence["route"]
                    if route in {"reuse", "normalize", "annotation"}:
                        replay, replay_evidence, _ = choose_reuse(item, tokenizer, config)
                        if replay != pair or replay_evidence != {k: v for k, v in evidence.items() if k not in {"pipeline", "view_sha256", "contract_hash"}}:
                            raise ValueError("source caption/annotation no longer reproduces the stored result")
                        if item["api_attempts"] or item["codex_attempts"]:
                            raise ValueError("reused text unexpectedly consumed inference attempts")
                    elif route in {"sii", "codex_fallback"}:
                        attempt = state.db.execute("SELECT * FROM attempts WHERE id=?", (evidence["attempt_id"],)).fetchone()
                        raw = state.raw(evidence["attempt_id"])
                        if not attempt or attempt["status"] != "succeeded" or attempt["item_key"] != item["key"]:
                            raise ValueError("missing successful source attempt")
                        if (raw.get("backend") != route or raw.get("view_sha256") != item["view_sha256"]
                                or raw.get("contract_hash") != CONTRACT_HASH or raw.get("image_attached") is not True
                                or raw.get("prompt_sha256") != sha(raw["prompt"].encode())):
                            raise ValueError("request/attachment provenance mismatch")
                        if parse_pair(raw, item["key"], tokenizer) != pair or attempt["result_sha256"] != sha(dumps(pair).encode()):
                            raise ValueError("raw provider response differs from published text")
                        if route == "sii":
                            if raw["requested_model"] not in config["sii"]["vision_models"] or raw.get("proxy") is not False:
                                raise ValueError("SII model/network contract mismatch")
                        else:
                            from scripts.legacy.distill_b512_codex import inspect_events
                            inspect_events(raw["events"], raw["output_text"])
                            if item["api_attempts"] < config["sii"]["max_attempts"]:
                                raise ValueError("Codex fallback preceded SII retry exhaustion")
                            failed = state.db.execute("SELECT count(*) FROM attempts WHERE item_key=? AND backend='sii' AND status IN ('failed','interrupted')", (item["key"],)).fetchone()[0]
                            if failed < config["sii"]["max_attempts"]:
                                raise ValueError("Codex fallback lacks complete SII failure history")
                            command = raw["command"]
                            if (raw["requested_model"] != "gpt-5.6-sol" or raw["reasoning_effort"] != "low"
                                    or '--ephemeral' not in command or '--image' not in command
                                    or 'model_reasoning_effort="low"' not in command):
                                raise ValueError("Codex fallback contract mismatch")
                    else:
                        raise ValueError("unknown publication route")
                    caption, prompt = index.read_caption(offset), index.read_t2i(offset)
                    provenance = {"pipeline": PIPELINE_VERSION, **evidence, "state_root": str(state.root),
                                  "generator_model": evidence["generator_models"]["i2t"],
                                  "reused_unchanged": route == "reuse"}
                    if (caption["manifest_index"] != offset or caption["img_id"] != view["img_id"]
                            or prompt["image_id"] != training_image_id(view)
                            or caption["provenance"] != prompt["provenance"] or caption["provenance"] != provenance
                            or caption["captions"] != [{"source": provenance["generator_model"], "text": pair["i2t"]}]
                            or prompt["model_result"]["prompts"] != [{"prompt": pair["t2i"]}]):
                        raise ValueError("loader text rows differ from durable results")
                    counts[route] += 1
                    authors[provenance["generator_model"]] += 1
                    sources[view["source"]] += 1
                    verified += 1
            if verified != index.row_count or verified != publication["records"]:
                raise ValueError("publication coverage mismatch")
            if verified != state.db.execute("SELECT count(*) FROM items WHERE status='ready'").fetchone()[0]:
                raise ValueError("publication omitted ready items")
        finally:
            index.close()
    posterior = None
    if require_posterior or (root / "posterior_index.json").exists():
        from scripts.audit_image_text_publication import audit_posterior
        posterior = audit_posterior(root, verified, file_sha(root / "manifest.jsonl"))
    return {"status": "passed", "audit_mode": "strict", "pipeline": PIPELINE_VERSION,
            "dataset": str(root), "verified_images": verified, "records": verified,
            "manifest_sha256": file_sha(root / "manifest.jsonl"), "posterior": posterior,
            "all_rgb_512": True, "all_source_and_view_sha256_verified": True,
            "all_seek_rows_verified": True, "all_reuse_text_sha_verified": True,
            "all_raw_responses_verified": True, "all_codex_calls_follow_sii_exhaustion": True,
            "token_budget_checked": True, "routes": dict(counts), "sources": dict(sources),
            "i2t_generators": dict(authors), "semantic_accuracy_independently_verified": False,
            "audited_at": time.time()}


def export(root, destination, *, tokenizer, shard_records=100000):
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError("publications are immutable; choose a new release directory")
    if shard_records < 1:
        raise ValueError("shard_records must be positive")
    temporary = None
    with State(root) as state:
        frozen = state.meta("frozen")
        counts = state.counts()
        if not frozen or any(n for status, n in counts.items() if status not in {"ready", "quarantined"}):
            raise ValueError("cannot publish pending, failed or partially completed source scope")
        total = counts.get("ready", 0)
        if not total:
            raise ValueError("empty publication")
        accepted_non_imagenet = state.db.execute("SELECT count(*) FROM items WHERE status='ready' AND json_extract(row_json,'$.source') != 'imagenet'").fetchone()[0]
        if not frozen["pilot"] and accepted_non_imagenet < frozen["minimum_images"]:
            raise ValueError("accepted non-ImageNet images fell below the production minimum after synthesis")
        if (total + shard_records - 1) // shard_records > 256:
            raise ValueError("increase shard_records to fit the existing uint8 T2I shard index")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=destination.name + ".tmp-", dir=destination.parent))
        try:
            shards, prompt, offsets = [], None, None
            with (temporary / "manifest.jsonl").open("w") as manifest, \
                 (temporary / "captions.jsonl").open("wb") as captions, \
                 (temporary / "captions.offsets.u64").open("wb") as caption_offsets, \
                 (temporary / "t2i_mapping.bi").open("wb") as mapping:
                caption_offsets.write(struct.pack("<Q", 0))
                try:
                    for offset, record in enumerate(state.db.execute("SELECT key,row_json,view_json,result_json,evidence_json FROM items WHERE status='ready' ORDER BY key")):
                        key, row, view, pair, evidence = record
                        row, view, pair, evidence = map(json.loads, (row, view, pair, evidence))
                        if offset % shard_records == 0:
                            if prompt:
                                prompt.close()
                                offsets.close()
                            number = len(shards)
                            name = f"t2i-{number:03d}.jsonl"
                            prompt = (temporary / name).open("wb")
                            offsets = (temporary / (name + ".offsets.u64")).open("wb")
                            offsets.write(struct.pack("<Q", 0))
                            shards.append({"shard_index": number, "records": 0, "path": name, "offsets_path": name + ".offsets.u64"})
                        manifest.write(dumps({**view, "img_id": offset + 1, "key": key, "split": "train",
                            "source": row["source"], "source_id": row["source_id"],
                            "selection_bucket": row.get("selection_bucket", row["source"]),
                            "capabilities": pair["capabilities"], "observations": pair["observations"],
                            "uncertainties": pair["uncertainties"]}) + "\n")
                        provenance = {"pipeline": PIPELINE_VERSION, **evidence, "state_root": str(state.root),
                                      "generator_model": evidence["generator_models"]["i2t"],
                                      "reused_unchanged": evidence["route"] == "reuse"}
                        caption = {"manifest_index": offset, "img_id": offset + 1, "provenance": provenance,
                                   "captions": [{"source": provenance["generator_model"], "text": pair["i2t"]}]}
                        generation = {"image_id": training_image_id(view), "provenance": provenance,
                                      "model_result": {"prompts": [{"prompt": pair["t2i"]}]}}
                        captions.write((dumps(caption) + "\n").encode())
                        caption_offsets.write(struct.pack("<Q", captions.tell()))
                        prompt.write((dumps(generation) + "\n").encode())
                        offsets.write(struct.pack("<Q", prompt.tell()))
                        mapping.write(struct.pack("<BI", len(shards) - 1, shards[-1]["records"]))
                        shards[-1]["records"] += 1
                finally:
                    if prompt:
                        prompt.close()
                        offsets.close()
            atomic_json(temporary / "text_index.json", {"schema": INDEX_SCHEMA, "split": "train", "records": total,
                "caption": {"path": "captions.jsonl", "offsets_path": "captions.offsets.u64"},
                "t2i": {"shards": shards}, "mapping": {"path": "t2i_mapping.bi"}})
            atomic_json(temporary / "publication.json", {"pipeline": PIPELINE_VERSION, "records": total,
                "image_size": 512, "image_tokens": 1024, "state_root": str(state.root), "pilot": frozen["pilot"],
                "quarantined_images": counts.get("quarantined", 0), "accepted_non_imagenet_images": accepted_non_imagenet,
                "frozen_pool": frozen, "contract": state.meta("contract"),
                "file_sha256": {p.name: file_sha(p) for p in temporary.iterdir() if p.is_file()}})
            # Audit using a separate read-only connection while holding the writer lock.
            report = audit(temporary, tokenizer=tokenizer)
            report["dataset"] = str(destination)
            atomic_json(temporary / "audit.json", report)
            temporary.replace(destination)
            return report
        finally:
            if temporary and temporary.exists():
                shutil.rmtree(temporary)
