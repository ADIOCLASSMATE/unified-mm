"""Normalize pinned PixMo-Points train shards without joining image versions."""
from collections import Counter
from pathlib import Path
import re

import pyarrow.parquet as pq

from data_synthesis.io import atomic_json, dumps, file_sha, sha


def point_rows(inputs, excluded, audit_root, compute_hashes=True):
    """Quarantine every annotation of an ambiguous URL, including earlier rows.

    Scan only URL/hash columns first, across every shard. Annotation payloads
    then stream in a second pass; memory scales with images, not point count.
    """
    audit_root = Path(audit_root)
    audit_root.mkdir(parents=True, exist_ok=True)
    versions, conflicts, counts, provenances = {}, {}, Counter(), []
    for obj in inputs:
        if not obj["upstream_path"].startswith("data/train-") or not obj.get("revision"):
            raise ValueError("PixMo-Points requires pinned original training shards")
        metadata_sha = file_sha(obj["path"]) if compute_hashes else None
        if compute_hashes and metadata_sha != obj["sha256"]:
            raise ValueError("PixMo-Points metadata SHA256 mismatch")
        provenances.append({"dataset": "allenai/pixmo-points", "revision": obj["revision"],
                            "field": "points/count/label/collection_method",
                            "metadata_path": obj["path"], "metadata_sha256": metadata_sha})
        for batch in pq.ParquetFile(obj["path"]).iter_batches(
                batch_size=8192, columns=["image_url", "image_sha256"]):
            for item in batch.to_pylist():
                url, image_sha = item["image_url"], item["image_sha256"]
                if not isinstance(url, str) or not url or not re.fullmatch("[0-9a-f]{64}", image_sha or ""):
                    raise ValueError("PixMo-Points image URL/hash is invalid")
                first = versions.setdefault(url, image_sha)
                counts[url] += 1
                if first != image_sha:
                    conflicts.setdefault(url, {first}).add(image_sha)
    report = {"schema": "pixmo_points_normalization_v2", "inputs": provenances,
              "input_annotation_rows": sum(counts.values()), "input_unique_urls": len(versions),
              "quarantined_urls": len(conflicts),
              "conflicts": [{"url": url, "source_id": sha(url.encode()),
                             "image_sha256_versions": sorted(hashes), "annotation_rows": counts[url]}
                            for url, hashes in sorted(conflicts.items())],
              "coordinate_system": "upstream_raw", "coordinate_system_verified": False}
    atomic_json(audit_root / "hash_conflicts.json", report)
    del versions, counts
    quarantine = audit_root / "quarantined_annotations.jsonl"
    temporary = quarantine.with_suffix(".tmp")
    kept, excluded_rows, quarantined, accepted_ids = 0, 0, 0, set()
    with temporary.open("w") as rejected:
        for obj, provenance in zip(inputs, provenances, strict=True):
            for batch in pq.ParquetFile(obj["path"]).iter_batches(batch_size=1024):
                for item in batch.to_pylist():
                    url = item["image_url"]
                    if url in conflicts:
                        rejected.write(dumps({"reason": "conflicting_image_sha256_for_url",
                                              "provenance": provenance, "annotation": item}) + "\n")
                        quarantined += 1
                        continue
                    identity = sha(url.encode())
                    if "pixmo_points:" + identity in excluded:
                        excluded_rows += 1
                        continue
                    accepted_ids.add(identity)
                    kept += 1
                    yield {"source": "pixmo_points", "source_id": identity, "split": "train",
                           "url": url, "view_policy": "fit_pad", "min_short_side": 512,
                           "capabilities": ["counting", "pointing"],
                           "identity_aliases": ["urlsha256:" + identity, "pixmo_cap:" + identity],
                           "expected_source_sha256": item["image_sha256"],
                           "annotations": [{"type": "pixmo_points", "provenance": provenance,
                                            "verified": False,
                                            "coordinate_system": "upstream_raw",
                                            "coordinate_system_verified": False,
                                            **{k: item[k] for k in ("points", "count", "label", "collection_method")}}]}
    temporary.replace(quarantine)
    atomic_json(audit_root / "normalization_summary.json", {
        **report, "state": "completed", "kept_annotation_rows": kept,
        "candidate_images": len(accepted_ids), "excluded_annotation_rows": excluded_rows,
        "quarantined_annotation_rows": quarantined,
        "compute_hashes": compute_hashes,
        "quarantine_path": str(quarantine), "quarantine_sha256": file_sha(quarantine) if compute_hashes else None})
