"""Check every published image/text identity and final-teacher provenance offline."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path

from PIL import Image

from utils.image_shard_io import read_image_bytes
from utils.image_text_teacher import EFFORT, FINAL_MODEL, PRIMARY_MODEL
from utils.imagenet_synthetic_text_index import ImageNetSyntheticTextIndex


def require(condition, message):
    if not condition:
        raise ValueError(message)


def archive_path(reference):
    value = str(reference)
    return Path(value[4:].rsplit("::", 1)[0] if value.startswith("tar:") else value).resolve()


def audit_publication(dataset, *, image_root=None, tokenizer=None, require_posterior=False):
    root = Path(dataset).resolve()
    image_root = Path(image_root).resolve() if image_root else None
    publication = json.loads((root / "publication.json").read_text())
    if publication.get("pipeline") == "b512-reuse-sii-fallback-v1":
        if tokenizer is None:
            raise ValueError("current publications require tokenizer validation")
        from data_synthesis.publication import audit
        return audit(root, tokenizer=tokenizer, require_posterior=require_posterior, image_root=image_root)
    allowed_runs = {r["path"] for r in publication["source_runs"]}
    for path in root.rglob("*"):
        require(path.suffix.lower() not in {
            ".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff",
            ".tar", ".zip", ".pt", ".pth", ".safetensors",
        }, f"image/latent binary inside text publication: {path}")
    sources, buckets, generators, decisions = Counter(), Counter(), Counter(), Counter()
    identities = {field: set() for field in ("key", "source_sha256", "view_sha256")}
    count = 0
    index = ImageNetSyntheticTextIndex(root / "text_index.json")
    try:
        require(index.split == "train", "expected train text index")
        with (root / "manifest.jsonl").open() as handle:
            for offset, line in enumerate(handle):
                row = json.loads(line)
                key = row["key"]
                require(row["img_id"] == offset + 1 and row["split"] == "train", f"noncontiguous ID: {key}")
                for field, seen in identities.items():
                    require(row[field] not in seen, f"duplicate {field}: {key}")
                    seen.add(row[field])
                for field, digest_field in (("source_path", "view_sha256"), ("original_ref", "source_sha256")):
                    path = archive_path(row[field])
                    require(not path.is_relative_to(root), f"image stored in text directory: {key}")
                    if image_root:
                        require(path.is_relative_to(image_root), f"image outside expected pool: {key}")
                    data = read_image_bytes(row[field])
                    require(hashlib.sha256(data).hexdigest() == row[digest_field], f"{digest_field} mismatch: {key}")
                    if field == "source_path":
                        with Image.open(io.BytesIO(data)) as image:
                            image.load()
                            require(image.mode == "RGB" and image.size == (512, 512), f"invalid frozen view: {key}")
                caption, prompt = index.read_caption(offset), index.read_t2i(offset)
                require(caption["manifest_index"] == offset and caption["img_id"] == row["img_id"], f"caption ID mismatch: {key}")
                require(prompt["image_id"] == "train/" + key, f"T2I ID mismatch: {key}")
                provenance = caption["provenance"]
                require(provenance == prompt["provenance"], f"I2T/T2I provenance mismatch: {key}")
                require(provenance["source_run"] in allowed_runs, f"undeclared source run: {key}")
                codex_only = provenance.get("pipeline") == "b512-codex-only-v1"
                role = "finalizer" if codex_only else "judge"
                require(provenance[f"{role}_model"] == FINAL_MODEL and provenance[f"{role}_effort"] == EFFORT
                        and provenance[f"{role}_backend"] == "codex_cli", f"wrong final teacher: {key}")
                decision = provenance["decision"]
                require(decision in ({"generate", "reuse"} if codex_only else {"accept", "replace"}), f"unapproved pair: {key}")
                require(len(caption["captions"]) == len(prompt["model_result"]["prompts"]) == 1,
                        f"expected one synthetic pair: {key}")
                texts = {"i2t": caption["captions"][0]["text"], "t2i": prompt["model_result"]["prompts"][0]["prompt"]}
                reuse = provenance.get("reuse_candidate")
                for task, text in texts.items():
                    require(isinstance(text, str) and bool(text.strip()), f"empty {task}: {key}")
                    if tokenizer is not None:
                        require(len(tokenizer.encode(text, add_special_tokens=False)) <= 960, f"{task} exceeds token budget: {key}")
                    if provenance["reused_unchanged"]:
                        require(decision == ("reuse" if codex_only else "accept") and bool(reuse), f"invalid reuse: {key}")
                        require(hashlib.sha256(text.encode()).hexdigest() == reuse[f"{task}_text_sha256"],
                                f"reused {task} text changed: {key}")
                        expected = reuse[f"{task}_model"]
                    else:
                        expected = PRIMARY_MODEL if not codex_only and decision == "accept" else FINAL_MODEL
                    require(provenance["generator_models"][task] == expected, f"wrong {task} generator: {key}")
                require(caption["captions"][0]["source"] == provenance["generator_models"]["i2t"]
                        == provenance["generator_model"], f"caption source mismatch: {key}")
                count += 1
                sources[row["source"]] += 1
                buckets[row["selection_bucket"]] += 1
                generators[provenance["generator_model"]] += 1
                origin = "legacy_candidate" if reuse else "codex_direct" if codex_only else "qwen_candidate"
                decisions[f"{row['source']}:{origin}:{decision}"] += 1
        require(count == index.row_count == publication["records"] and count > 0, "publication row count mismatch")
    finally:
        index.close()
    manifest_digest = hashlib.sha256((root / "manifest.jsonl").read_bytes()).hexdigest()
    posterior = None
    posterior_index = root / "posterior_index.json"
    if require_posterior or posterior_index.exists():
        posterior = audit_posterior(root, count, manifest_digest)
    return {
        "audited_at": datetime.now(timezone.utc).isoformat(), "dataset": str(root),
        "verified_images": count, "all_rgb_512": True, "all_source_and_view_sha256_verified": True,
        "all_seek_rows_verified": True, "all_sol_low_approved": True,
        "all_reuse_text_sha_verified": True, "image_binaries_in_text_directory": 0,
        "token_budget_checked": tokenizer is not None,
        "manifest_sha256": manifest_digest, "posterior": posterior,
        "sources": dict(sources), "selection_buckets": dict(buckets),
        "i2t_generators": dict(generators), "decisions_by_source": dict(decisions),
        "note": "Checks identity and publication contracts; teacher approval and selection buckets are not independent factual-accuracy measurements.",
    }


def audit_posterior(root, count, manifest_digest, *, compute_hashes=True):
    posterior_index = Path(root) / "posterior_index.json"
    import torch
    from pretrain.merge_flow_latent_shards import POSTERIOR_CACHE_FORMAT, POSTERIOR_STATS_LAYOUT
    from utils.sharded_posterior import load_sharded_posterior
    cached = load_sharded_posterior(posterior_index)
    stats, ids, meta = cached["posterior_stats"], cached["img_ids"], cached["metadata"]
    require(meta["format"] == POSTERIOR_CACHE_FORMAT and meta["stats_layout"] == POSTERIOR_STATS_LAYOUT
            and meta["stats_are_scaled"] is True, "unsupported posterior cache contract")
    if compute_hashes:
        require(meta["manifest_sha256"] == manifest_digest, "posterior index refers to a different publication")
    else:
        from data_synthesis.integrity import check_file_size
        require(Path(meta["manifest_jsonl"]).resolve() == (Path(root) / "manifest.jsonl").resolve(),
                "posterior index refers to a different publication")
        check_file_size(Path(root) / "manifest.jsonl", meta.get("manifest_bytes"))
        require(meta.get("identity_mapping") == "frozen_image_reference"
                and meta.get("source_image_references_verified") is True,
                "posterior bank references have not been joined to publication images")
    require(torch.equal(ids, torch.arange(1, count + 1)), "posterior/publication IDs differ")
    require(stats.shape == (count, 1024, 32) and meta["frozen_views"], "posterior is not a frozen 512px cache")
    for i in torch.argsort(stats.shard_rows[:, 0], stable=True).tolist():
        value = stats[i]
        require(bool(torch.isfinite(value).all()) and not bool((value[..., 16:] < 0).any()),
                f"invalid posterior for image {i + 1}")
    posterior = {"index": str(posterior_index), "shape": list(stats.shape), "all_rows_verified": True,
                 "compute_hashes": compute_hashes,
                 "source_image_references_verified": bool(meta.get("source_image_references_verified")),
                 "source_view_hashes_verified_at_encoding": bool(meta.get("source_view_hashes_verified"))}
    return posterior


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tokenizer", default="public/models/Qwen--Qwen3-0.6B-Base")
    parser.add_argument("--require-posterior", action="store_true")
    args = parser.parse_args()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    report = audit_publication(args.dataset, image_root=args.image_root, tokenizer=tokenizer,
                               require_posterior=args.require_posterior)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(output)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
