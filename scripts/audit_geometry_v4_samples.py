"""Audit semantic split isolation and known COCO/Visual Genome image overlap."""

import argparse
import io
import json
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.research.representation_protocol import emit, write_json

VG_SOURCE = "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/image_data.json.zip"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    root = a.output_dir
    samples = json.loads((root / "samples.json").read_text())
    metadata_path = root / "vg-image-metadata.json"
    if not metadata_path.exists():
        response = requests.get(VG_SOURCE, timeout=(20, 90))
        response.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            metadata = json.loads(archive.read("image_data.json"))
        write_json(metadata_path, metadata)
    metadata = json.loads(metadata_path.read_text())
    metadata = {int(r["image_id"]): r for r in metadata}
    old = [
        json.loads((root.parent / name / "samples.json").read_text())
        for name in ("step-95415-ema-20260907", "semantic-v2-20260907")
    ]
    old_imagenet = {r["image_id"] for s in old for r in s["imagenet_images"]}
    images = samples["imagenet_images"]
    assert len(images) == len({r["image_id"] for r in images}) == 32000
    assert not old_imagenet.intersection(r["image_id"] for r in images)
    by_class = defaultdict(list)
    for row in images:
        by_class[row["group"]].append(row)
    assert len(by_class) == 1000
    assignment = {}
    for group, rows in by_class.items():
        assert Counter(r["split"] for r in rows) == {"cal": 2, "a": 15, "b": 15}
        assert len({r["mapping_split"] for r in rows}) == 1
        assignment[group] = rows[0]["mapping_split"]
        for view in ("a", "b"):
            assert sorted(r["view_index"] for r in rows if r["split"] == view) == list(
                range(15)
            )
    assert Counter(assignment.values()) == {"fit": 600, "dev": 200, "test": 200}
    for row in samples["imagenet_texts"]:
        assert row["mapping_split"] == assignment[row["group"]]
    assert Counter(
        (r["group"], r["split"]) for r in samples["imagenet_texts"]
    ) == Counter({(g, s): 4 for g in range(1000) for s in ("cal", "a", "b")})
    coco = {r["source_image_id"]: r for r in samples["coco_images"]}
    assert len(coco) == 11776
    assert Counter(r["split"] for r in coco.values()) == {
        "cal": 512,
        "fit": 8192,
        "dev": 1024,
        "test": 2048,
    }
    texts = defaultdict(list)
    for row in samples["coco_texts"]:
        assert row["split"] == coco[row["source_image_id"]]["split"]
        assert row["group"] == row["source_image_id"]
        assert row["text"].strip()
        texts[row["group"]].append(row)
    assert set(texts) == set(coco)
    assert all(
        len(rs) >= 5 and sorted(r["caption_index"] for r in rs) == list(range(len(rs)))
        for rs in texts.values()
    )
    old_coco = {int(r["source_image_id"]) for r in old[1]["coco_images"]}
    retrieval = ROOT / "public/benchmarks/mscoco_karpathy_retrieval_v1/retrieval.jsonl"
    old_coco.update(
        int(json.loads(line)["source_image_id"])
        for line in retrieval.read_text().splitlines()
    )
    old_hard = {r["image_id"] for r in old[1]["hard_images"]}
    sugar = (
        ROOT
        / "public/benchmarks/selfless_multimodal_likelihood_v1/tasks/sugarcrepe.jsonl"
    )
    for line in sugar.read_text().splitlines():
        row = json.loads(line)
        if row["image_id"] in old_hard:
            old_coco.add(int(Path(row["metadata"]["filename"]).stem))
    assert not set(coco).intersection(old_coco)
    aro = samples["aro_images"]
    assert len({r["source_image"] for r in aro}) == len(aro) == 2000
    assert Counter(r["task"] for r in aro) == {
        "aro_vg_relation": 1000,
        "aro_vg_attribution": 1000,
    }
    details, categories = [], Counter()
    for i, row in enumerate(aro):
        vg_id = int(Path(row["source_image"]).stem)
        assert vg_id in metadata, f"Missing VG metadata: {vg_id}"
        coco_id = metadata[vg_id].get("coco_id")
        coco_id = int(coco_id) if coco_id else None
        category = (
            "known_overlap"
            if coco_id in coco
            else "verified_coco_disjoint"
            if coco_id is not None
            else "unknown_coco_link"
        )
        categories[category] += 1
        details.append(
            {
                "index": i,
                "vg_id": vg_id,
                "coco_id": coco_id,
                "task": row["task"],
                "category": category,
                "overlap_split": coco[coco_id]["split"] if coco_id in coco else None,
            }
        )
    for group in (r["group"] for r in aro):
        pair = [r for r in samples["aro_texts"] if r["group"] == group]
        assert len(pair) == 2 and sum(r["positive"] for r in pair) == 1
    write_json(
        root / "sample-audit-v4.json",
        {
            "status": "passed_with_explicit_cross_dataset_overlap_check",
            "imagenet_new_images": 32000,
            "imagenet_excluded_prior_images": len(old_imagenet),
            "coco_disjoint_scenes": 11776,
            "coco_captions": len(samples["coco_texts"]),
            "coco_excluded_prior_identities": len(old_coco),
            "aro_identity_categories": dict(categories),
            "aro_identity_details": details,
            "vg_metadata_source": VG_SOURCE,
            "note": "No hashes or image-content decontamination. Known COCO-linked disjoint ARO subset is the strict identity-transfer control; unknown links are reported separately. Original 2000-example results remain archived and are not silently replaced.",
        },
    )
    emit("sample_audit_done_v4", aro_categories=dict(categories))


if __name__ == "__main__":
    main()
