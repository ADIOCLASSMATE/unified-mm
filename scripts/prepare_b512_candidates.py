#!/usr/bin/env python3
"""Prepare auditable candidates from local ImageNet and frozen PixMo metadata."""

import argparse
import csv
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import math
from pathlib import Path
import random
import re
from urllib.parse import urlsplit

from PIL import Image

from utils.imagenet_synthetic_text_index import ImageNetSyntheticTextIndex


def imagenet_candidates(public, count, seed, fresh_every=0, excluded_ids=frozenset()):
    dataset = public / "datasets/imagenet1k_synthetic_v1"
    root = public / "dataset/imagenet/v1/ILSVRC/Data/CLS-LOC/train"
    index = ImageNetSyntheticTextIndex(dataset / "indexed/train/manifest.json")
    total = index.manifest["records"]
    step = 104729
    while math.gcd(step, total) != 1:
        step += 2
    accepted = 0
    try:
        for ordinal in range(total):
            idx = (seed + ordinal * step) % total
            caption = index.read_caption(idx)
            if "imagenet:" + caption["id"] in excluded_ids:
                continue
            path = root / caption["path"]
            with Image.open(path) as im:
                width, height = im.size
            if min(width, height) < 224 or max(width, height) / min(width, height) > 1.8:
                continue
            t2i = index.read_t2i(idx)
            if t2i["image_id"] != "train/" + caption["id"]:
                raise ValueError("legacy I2T/T2I identities differ")
            prompts = [r for r in t2i["model_result"]["prompts"] if r.get("style") == "faithful_photo"]
            captions = [r for r in caption["captions"] if r.get("source") != "original"
                        and r.get("model") and 20 <= len(r.get("text", "").split()) <= 120]
            if not prompts or not captions:
                continue
            # Select a concise candidate; sol still checks every visible fact.
            previous = min(captions, key=lambda r: (
                len(re.findall(r"\b(likely|suggesting|probably|proudly)\b", r["text"], re.I)),
                abs(len(r["text"].split()) - 50)))
            row = {"source": "imagenet", "source_id": caption["id"], "split": "train",
                   "local_path": str(path.resolve()), "synset": caption["synset"],
                   "expected_source_sha256": caption["source_image_sha256"],
                   "min_short_side": 224, "capabilities": ["imagenet_long_tail"]}
            if not fresh_every or accepted % fresh_every:
                row["reuse_pair"] = {
                    "i2t": previous["text"], "t2i": prompts[0]["prompt"],
                    "i2t_model": previous["model"], "t2i_model": t2i["generation"]["model"],
                    "i2t_source": previous["source"], "t2i_style": "faithful_photo",
                    "source_dataset": str(dataset.resolve()), "source_row": idx,
                    "i2t_text_sha256": hashlib.sha256(previous["text"].encode()).hexdigest(),
                    "t2i_text_sha256": hashlib.sha256(prompts[0]["prompt"].encode()).hexdigest(),
                }
            yield row
            accepted += 1
            if accepted == count:
                return
        raise ValueError(f"only {accepted} ImageNet candidates available, requested {count}")
    finally:
        index.close()


def pixmo_candidates(parquet, count, revision, seed, allowed_hosts=frozenset(), excluded_ids=frozenset()):
    import pyarrow.parquet as pq
    rng = random.Random(seed)
    seen = set()
    buckets, visits = {}, Counter()
    patterns = [("style", r"\b(watercolor|illustration|painting|cartoon|3d render|pixel art)\b"),
                ("ocr", r"\b(text reads|words? (read|say|written)|lettering|sign (reads|says)|label (reads|says))\b"),
                ("counting", r"\b(two|three|four|five|six|seven|eight|nine|ten)\b"),
                ("attribute_binding", r"\b(red|blue|yellow|green|purple|orange)\b.*\b(red|blue|yellow|green|purple|orange)\b"),
                ("relation", r"\b(left of|right of|in front of|behind|underneath|above|below)\b"),
                ("action", r"\b(holding|riding|playing|carrying|climbing|cooking)\b")]
    patterns = [(name, re.compile(pattern, re.I)) for name, pattern in patterns]
    for batch in pq.ParquetFile(parquet).iter_batches(batch_size=512):
        for item in batch.to_pylist():
            url = item.get("image_url") or item.get("url")
            if not isinstance(url, str) or not url.startswith(("http://", "https://")) or url in seen:
                continue
            if allowed_hosts and urlsplit(url).hostname not in allowed_hosts:
                continue
            seen.add(url)
            bucket = next((name for name, pattern in patterns if pattern.search(item.get("caption", ""))), "general")
            identity = item.get("image_sha256") or hashlib.sha256(url.encode()).hexdigest()
            if "pixmo_cap:" + identity in excluded_ids:
                continue
            row = {"source": "pixmo_cap", "source_id": identity, "url": url, "split": "train",
                   "metadata_revision": revision, "min_short_side": 512,
                   "capabilities": [bucket], "selection_bucket": bucket}
            if item.get("caption"):
                from data_synthesis.sources import caption_candidate
                row["caption_candidates"] = [caption_candidate(
                    row, item["caption"], author="allenai/pixmo-cap", kind="human_caption",
                    provenance={"dataset": "allenai/pixmo-cap", "revision": revision,
                                "field": "caption", "shard": str(parquet)})]
            row["view_policy"] = "fit_pad"
            if item.get("image_sha256"):
                row["expected_source_sha256"] = item["image_sha256"]
            visits[bucket] += 1
            reservoir = buckets.setdefault(bucket, [])
            if len(reservoir) < count:
                reservoir.append(row)
            else:
                offset = rng.randrange(visits[bucket])
                if offset < count:
                    reservoir[offset] = row
    # Round-robin capability buckets after a full streaming reservoir pass;
    # never take the first rows of metadata sorted into topics such as birds.
    accepted = 0
    while accepted < count:
        before = accepted
        for bucket in [name for name, _ in patterns] + ["general"]:
            if buckets.get(bucket):
                yield buckets[bucket].pop()
                accepted += 1
                if accepted == count:
                    return
        if before == accepted:
            raise ValueError(f"only {accepted} PixMo candidates available, requested {count}")


def prepare_exclusions(public, output):
    identities, paths = set(), set()
    benchmarks = public / "benchmarks"
    for manifest in sorted(benchmarks.glob("*/image_manifest.jsonl")):
        for line in manifest.open():
            row = json.loads(line)
            path = Path(row["source_path"])
            if str(path).startswith("/inspire/dataset/"):
                path = public / "dataset" / path.relative_to("/inspire/dataset")
            if path.is_file():
                paths.add(path)
            if "COCO_" in path.name:
                identities.add("coco:" + str(int(path.stem.rsplit("_", 1)[-1])))
            elif "flickr" in str(manifest):
                identities.add("flickr:" + path.stem)
            elif "imagenet" in str(manifest):
                identities.add("imagenet:" + path.stem)
    for source, glob in [("coco", "sugarcrepe*.jsonl"), ("vg", "aro_vg*.jsonl")]:
        for path in (benchmarks / "selfless_multimodal_likelihood_v1/tasks").glob(glob):
            for line in path.open():
                meta = json.loads(line).get("metadata", {})
                name = meta.get("source_image") or meta.get("filename")
                if name and Path(name).stem.isdigit():
                    identities.add(source + ":" + str(int(Path(name).stem)))
    val = public / "datasets/imagenet_full/manifest_val.jsonl"
    for line in val.open():
        identities.add("imagenet:" + Path(json.loads(line)["source_path"]).stem)

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    with ThreadPoolExecutor(max_workers=8) as pool:
        identities.update(pool.map(digest, sorted(paths)))
    output.write_text("\n".join(sorted(identities)) + "\n")
    return {"exclusion_entries": len(identities), "hashed_benchmark_images": len(paths)}


def openimages_candidates(csv_path, count, seed, excluded_ids=frozenset()):
    """Balance relationship types; retain their two boxes for safe square crops."""
    rng, seen, visits, buckets = random.Random(seed), set(), Counter(), {}
    capacity = min(count, max(256, math.ceil(count / 4)))
    with Path(csv_path).open(newline="") as handle:
        for item in csv.DictReader(handle):
            identity = item["ImageID"]
            relation = item.get("RelationshipLabel") or item["RelationLabel"]
            if not re.fullmatch(r"[0-9a-f]{16}", identity):
                raise ValueError("invalid Open Images ID")
            if (identity, relation) in seen or "openimages:" + identity in excluded_ids:
                continue
            seen.add((identity, relation))
            boxes = [[float(item[f"{axis}{bound}{i}"]) for axis, bound in
                      (("X", "Min"), ("Y", "Min"), ("X", "Max"), ("Y", "Max"))] for i in (1, 2)]
            row = {"source": "openimages", "source_id": identity, "split": "train",
                   "url": f"https://open-images-dataset.s3.amazonaws.com/train/{identity}.jpg",
                   "min_short_side": 512, "required_boxes_normalized": boxes,
                   "capabilities": ["attribute_binding" if relation == "is" else "relation"],
                   "selection_bucket": "openimages_" + relation,
                   "selection_annotation": {"relation": relation, "labels": [item["LabelName1"], item["LabelName2"]]},
                   "metadata_generation": "1611743564997636"}
            visits[relation] += 1
            reservoir = buckets.setdefault(relation, [])
            if len(reservoir) < capacity:
                reservoir.append(row)
            else:
                offset = rng.randrange(visits[relation])
                if offset < capacity:
                    reservoir[offset] = row
    used = set()
    while len(used) < count:
        before = len(used)
        for relation in sorted(buckets):
            while buckets[relation]:
                row = buckets[relation].pop()
                if row["source_id"] not in used:
                    used.add(row["source_id"])
                    yield row
                    break
            if len(used) == count:
                return
        if before == len(used):
            raise ValueError(f"only {len(used)} Open Images relationship candidates available")


def wikiart_candidates(parquet, source_archive, count, revision, seed, excluded_ids=frozenset()):
    """Extract embedded images once, then select across upstream numeric style IDs."""
    import pyarrow.parquet as pq
    from data_synthesis.io import ImageArchives, digest_file
    if not source_archive:
        raise ValueError("WikiArt extraction requires a separate image archive directory")
    archive_root = Path(source_archive).resolve()
    if list(archive_root.glob("*.tar")):
        raise FileExistsError("source image extraction is immutable; use another archive directory")
    source_digest = digest_file(parquet)
    parquet_file = pq.ParquetFile(parquet)
    archives = ImageArchives(archive_root)
    buckets, seen = {}, set()
    try:
        for batch in parquet_file.iter_batches(batch_size=16):
            for item in batch.to_pylist():
                data = item["image"]["bytes"]
                if not data or len(data) > 20 << 20:
                    continue
                with Image.open(io.BytesIO(data)) as image:
                    width, height = image.size
                if min(width, height) < 512 or max(width, height) / min(width, height) > 1.8:
                    continue
                identity = hashlib.sha256(data).hexdigest()
                if identity in seen or "wikiart:" + identity in excluded_ids:
                    continue
                seen.add(identity)
                style = int(item["style"])
                row = {"source": "wikiart", "source_id": identity, "split": "train",
                       "local_path": archives.add(identity + ".original", data),
                       "expected_source_sha256": identity, "min_short_side": 512,
                       "capabilities": ["style"], "selection_bucket": f"wikiart_style_{style}",
                       "metadata_repository": "huggan/wikiart", "metadata_revision": revision,
                       "metadata_parquet_sha256": source_digest,
                       "metadata_license": "unknown; data files copyright original authors",
                       "selection_annotation": {"style_id": style, "artist_id": item["artist"],
                                                "genre_id": item["genre"], "upstream_path": item["image"]["path"]}}
                buckets.setdefault(style, []).append(row)
    finally:
        archives.close()
    rng = random.Random(seed)
    for rows in buckets.values():
        rng.shuffle(rows)
    selected = 0
    while selected < count and any(buckets.values()):
        for style in sorted(buckets):
            if buckets[style]:
                yield buckets[style].pop()
                selected += 1
                if selected == count:
                    return


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public", default="public")
    parser.add_argument("--output", required=True)
    parser.add_argument("--imagenet-count", type=int, default=0)
    parser.add_argument("--imagenet-fresh-every", type=int, default=0)
    parser.add_argument("--pixmo-count", type=int, default=0)
    parser.add_argument("--pixmo-parquet")
    parser.add_argument("--openimages-count", type=int, default=0)
    parser.add_argument("--openimages-relationships")
    parser.add_argument("--wikiart-count", type=int, default=0)
    parser.add_argument("--wikiart-parquet")
    parser.add_argument("--wikiart-source-archive")
    parser.add_argument("--wikiart-revision", default="d559852d2b232e0fcf195e775866964f0564f2b5")
    parser.add_argument("--pixmo-revision", default="edce6390d9d5be6c8db0d863fbe62718c88988a4")
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--exclude-candidates", action="append", default=[], help="Previous immutable manifests to avoid repeated source images.")
    parser.add_argument("--pixmo-host", action="append", default=[], help="Restrict to a verified direct source host; repeat for multiple hosts.")
    parser.add_argument("--write-exclusions", action="store_true")
    args = parser.parse_args()
    public, output = Path(args.public).resolve(), Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    path = output / "candidates.jsonl"
    if path.exists():
        raise FileExistsError("candidate manifests are immutable; use another directory")
    rows = []
    excluded_ids = set()
    for manifest in args.exclude_candidates:
        with Path(manifest).open() as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    excluded_ids.add(row["source"] + ":" + row["source_id"])
    if args.imagenet_count:
        rows.extend(imagenet_candidates(public, args.imagenet_count, args.seed, args.imagenet_fresh_every, excluded_ids))
    if args.pixmo_count:
        rows.extend(pixmo_candidates(args.pixmo_parquet, args.pixmo_count, args.pixmo_revision, args.seed,
                                    set(args.pixmo_host), excluded_ids))
    if args.openimages_count:
        rows.extend(openimages_candidates(args.openimages_relationships, args.openimages_count, args.seed, excluded_ids))
    if args.wikiart_count:
        rows.extend(wikiart_candidates(args.wikiart_parquet, args.wikiart_source_archive,
                                       args.wikiart_count, args.wikiart_revision, args.seed, excluded_ids))
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    report = {"records": len(rows), "sources": dict(Counter(r["source"] for r in rows)),
              "selection_buckets": dict(Counter(r.get("selection_bucket", "imagenet") for r in rows)),
              "reuse_candidates": sum("reuse_pair" in r for r in rows), "seed": args.seed,
              "pixmo_allowed_hosts": args.pixmo_host, "excluded_source_ids": len(excluded_ids),
              "public": str(public), "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    if args.write_exclusions:
        report.update(prepare_exclusions(public, output / "benchmark_exclusions.txt"))
    (output / "preparation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
