"""Offline V4 identity reuse, split, crop and truncation audit; no downloads or hashes."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from scripts.geometry_v5_encoders import PixelLoader
from scripts.prepare_cross_model_geometry_v5 import ROOT, V4
from utils.research.geometry_v5_assets import RUN, emit, write_json


def audit_samples(root):
    samples = json.loads((root / "samples.json").read_text())
    old = json.loads((V4 / "samples.json").read_text())
    f_samples = json.loads((root / "f-v4/samples.json").read_text())
    assert f_samples == old
    expected_counts = {
        "imagenet_images": 32000,
        "imagenet_texts": 12000,
        "coco_images": 11776,
        "coco_texts": 58909,
        "aro_images": 868,
        "aro_texts": 1736,
    }
    assert {key: len(value) for key, value in samples.items()} == expected_counts
    for family in ("imagenet", "coco"):
        for modality in ("images", "texts"):
            key = f"{family}_{modality}"
            assert samples[key] == old[key], (
                f"V4 semantic identities or order changed: {key}"
            )
    images = samples["imagenet_images"]
    assert len({row["image_id"] for row in images}) == 32000
    classes = defaultdict(list)
    for row in images:
        classes[row["group"]].append(row)
    assert len(classes) == 1000
    assignments = {}
    for group, rows in classes.items():
        assert Counter(row["split"] for row in rows) == {"cal": 2, "a": 15, "b": 15}
        assert len({row["mapping_split"] for row in rows}) == 1
        assignments[group] = rows[0]["mapping_split"]
        for view in ("a", "b"):
            assert sorted(
                row["view_index"] for row in rows if row["split"] == view
            ) == list(range(15))
    assert Counter(assignments.values()) == {"fit": 600, "dev": 200, "test": 200}
    assert Counter(
        (row["group"], row["split"]) for row in samples["imagenet_texts"]
    ) == Counter(
        {(group, split): 4 for group in classes for split in ("cal", "a", "b")}
    )
    assert all(
        row["mapping_split"] == assignments[row["group"]]
        for row in samples["imagenet_texts"]
    )
    coco = {row["source_image_id"]: row for row in samples["coco_images"]}
    assert len(coco) == 11776 and Counter(row["split"] for row in coco.values()) == {
        "cal": 512,
        "fit": 8192,
        "dev": 1024,
        "test": 2048,
    }
    captions = defaultdict(list)
    for row in samples["coco_texts"]:
        assert row["group"] == row["source_image_id"]
        assert row["split"] == coco[row["group"]]["split"] and row["text"].strip()
        captions[row["group"]].append(row)
    assert captions.keys() == coco.keys()
    assert all(
        len(rows) >= 5
        and sorted(row["caption_index"] for row in rows) == list(range(len(rows)))
        for rows in captions.values()
    )
    metadata_path = V4 / "vg-image-metadata.json"
    metadata = {
        int(row["image_id"]): row for row in json.loads(metadata_path.read_text())
    }
    strict = []
    for index, row in enumerate(old["aro_images"]):
        vg_id = int(Path(row["source_image"]).stem)
        coco_id = metadata[vg_id].get("coco_id")
        if coco_id is not None and int(coco_id) != 0 and int(coco_id) not in coco:
            strict.append(index)
    assert (
        len(strict) == 868
        and [row["v4_index"] for row in samples["aro_images"]] == strict
    )
    registry = {
        int(row["img_id"]): row["source_path"]
        for row in map(
            json.loads,
            (
                ROOT
                / "public/benchmarks/selfless_multimodal_likelihood_v1/image_manifest.jsonl"
            )
            .read_text()
            .splitlines(),
        )
    }
    for row in samples["aro_images"]:
        original = old["aro_images"][row["v4_index"]]
        assert {
            key: value
            for key, value in row.items()
            if key not in {"v4_index", "source_path"}
        } == original
        assert row["source_path"] == registry[row["image_id"]]
    assert len({row["source_image"] for row in samples["aro_images"]}) == 868
    aro_groups = {row["group"] for row in samples["aro_images"]}
    assert samples["aro_texts"] == [
        row for row in old["aro_texts"] if row["group"] in aro_groups
    ]
    pairs = defaultdict(list)
    for row in samples["aro_texts"]:
        pairs[row["group"]].append(row)
    assert pairs.keys() == aro_groups and all(
        len(rows) == 2 and sum(row["positive"] for row in rows) == 1
        for rows in pairs.values()
    )
    for family in ("imagenet", "coco", "aro"):
        assert all(
            Path(row["source_path"]).is_file() for row in samples[f"{family}_images"]
        )
    robust = {}
    for family in ("imagenet", "coco"):
        groups = list(
            dict.fromkeys(
                row["group"]
                for row in samples[f"{family}_images"]
                if row["robust"]
                and (
                    row["mapping_split"] == "test"
                    if family == "imagenet"
                    else row["split"] == "test"
                )
            )
        )
        assert len(groups) == (32 if family == "imagenet" else 512)
        robust[family] = groups
    return samples, {
        "status": "passed",
        "counts": expected_counts,
        "aro_task_counts": dict(Counter(row["task"] for row in samples["aro_images"])),
        "aro_rederived_from_metadata": str(metadata_path),
        "robust_test_groups": robust,
        "scope": "Readable image identity and author VG-to-COCO links; no pixel-hash duplicate or full-pretraining decontamination claim",
        "f_original_v4_sample_manifest_exact": True,
        "non_aro_v4_rows_and_order_exact": True,
        "aro_original_crop_registry_exact": True,
    }


def audit_pixels(samples):
    loader = PixelLoader()
    reference = transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )
    after_crop = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3)]
    )
    result = []
    for family in ("imagenet", "coco", "aro"):
        rows = samples[f"{family}_images"][:32]
        for row in rows:
            with Image.open(row["source_path"]) as raw:
                expected = reference(raw.convert("RGB"))
            actual = after_crop(loader.one(row["source_path"]))
            assert np.array_equal(actual.numpy(), expected.numpy())
        result.append({"family": family, "pixel_checks": len(rows), "exact": True})
    loader.pool.shutdown()
    return {
        "checks": result,
        "source_reference": "scripts/imagenet_encode_kl16_vae.py:ImagePathDataset",
        "common_policy": "RGB -> shorter edge 256 bicubic -> center crop 256; subsequently resize to each adapter native resolution",
        "limitation": "Same 256px content view, not each external model's optimal raw high-resolution preprocessing",
    }


def audit_truncation(root):
    result = []
    for model in ("qwen_text", "siglip"):
        for dataset in ("imagenet_texts", "coco_texts", "aro_texts"):
            count, truncated = 0, 0
            for rank in range(16):
                path = root / "features" / model / f"{dataset}-rank-{rank:02d}-of-16.pt"
                shard = torch.load(
                    path, map_location="cpu", weights_only=True, mmap=True
                )
                count += len(shard["indices"])
                truncated += shard["truncated_texts"]
            result.append(
                {
                    "model": model,
                    "dataset": dataset,
                    "texts": count,
                    "truncated_texts": truncated,
                    "fraction": truncated / count,
                }
            )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=RUN)
    args = parser.parse_args()
    torch.set_num_threads(1)
    samples, result = audit_samples(args.output_dir)
    result["pixel_policy"] = audit_pixels(samples)
    result["runtime_text_truncations"] = audit_truncation(args.output_dir)
    result["flow_text_truncation"] = (
        "No truncation operation; full observed text with fixed scaffold is passed by the frozen adapters"
    )
    write_json(args.output_dir / "audits" / "samples-and-preprocessing.json", result)
    emit(
        "v5_sample_preprocessing_audit_passed",
        counts=result["counts"],
        truncations=result["runtime_text_truncations"],
    )


if __name__ == "__main__":
    main()
