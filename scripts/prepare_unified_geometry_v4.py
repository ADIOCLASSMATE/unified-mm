"""Freeze expanded B geometry samples and fetch public COCO images on CPU.

No hashes, no training, no accelerator forwards. Download only the fixed manifest.
"""

import argparse
import concurrent.futures
import io
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import requests
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.probe_unified_representations import emit, write_json
from scripts.probe_unified_semantics_v2 import LAYERS, RESEARCH, RUN

SEED = 20260909
TEMPLATES = (
    "a photo of a {name}.",
    "a photograph showing a {name}.",
    "this is a {name}.",
    "the subject is a {name}.",
    "an image of a {name}.",
    "there is a {name} in the picture.",
    "the picture shows a {name}.",
    "a {name} is shown here.",
    "a picture of a {name}.",
    "a photograph of the {name}.",
    "the object shown is a {name}.",
    "an example of a {name}.",
)


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def prepare(out):
    if (out / "protocol.json").exists():
        raise FileExistsError(
            "V4 samples are immutable; use existing protocol or a new version"
        )
    from scripts.evaluate_imagenet_pretraining_native import (
        DEFAULT_CLASSES,
        DEFAULT_CLASSNAMES,
        load_imagenet_records,
        load_openai_clip_class_names,
    )
    from utils.evaluation_model_source import resolve_evaluation_model_source

    rng = random.Random(SEED)
    diagnostic = RESEARCH / "representation-diagnostic"
    v1 = json.loads((diagnostic / "step-95415-ema-20260907/samples.json").read_text())
    v2 = json.loads((diagnostic / "semantic-v2-20260907/samples.json").read_text())
    excluded_imagenet = {r["image_id"] for s in (v1, v2) for r in s["imagenet_images"]}
    old_coco = {int(r["source_image_id"]) for r in v2["coco_images"]}
    # Exclude all previous COCO benchmark test images, and all prior V2 hard images.
    old_coco |= {
        int(r["source_image_id"])
        for r in jsonl(
            ROOT / "public/benchmarks/mscoco_karpathy_retrieval_v1/retrieval.jsonl"
        )
    }
    hard_ids = {r["image_id"] for r in v2["hard_images"]}
    for r in jsonl(
        ROOT
        / "public/benchmarks/selfless_multimodal_likelihood_v1/tasks/sugarcrepe.jsonl"
    ):
        if r["image_id"] in hard_ids:
            old_coco.add(int(Path(r["metadata"]["filename"]).stem))
    samples = {
        f"{family}_{modality}": []
        for family in ("imagenet", "coco", "aro")
        for modality in ("images", "texts")
    }
    imagenet_root = ROOT / "public/datasets/imagenet_full"
    records = load_imagenet_records(
        imagenet_root / "manifest_val.jsonl", ROOT / DEFAULT_CLASSES
    )
    names, _ = load_openai_clip_class_names(ROOT / DEFAULT_CLASSNAMES)
    available = defaultdict(list)
    for row in records:
        if row.img_id not in excluded_imagenet:
            available[row.class_index].append(row)
    class_order = list(range(1000))
    rng.shuffle(class_order)
    assignments = {
        cls: "fit" if i < 600 else "dev" if i < 800 else "test"
        for i, cls in enumerate(class_order)
    }
    for cls in range(1000):
        rows = available[cls]
        rng.shuffle(rows)
        assert len(rows) >= 32, (cls, len(rows))
        for j, row in enumerate(rows[:32]):
            view = "cal" if j < 2 else "a" if j < 17 else "b"
            index = j if j < 2 else j - 2 if j < 17 else j - 17
            samples["imagenet_images"].append(
                {
                    "group": cls,
                    "image_id": row.img_id,
                    "source_path": row.source_path,
                    "split": view,
                    "view_index": index,
                    "mapping_split": assignments[cls],
                    "robust": cls in class_order[800:832],
                }
            )
        for j, template in enumerate(TEMPLATES):
            view = "cal" if j < 4 else "a" if j < 8 else "b"
            samples["imagenet_texts"].append(
                {
                    "group": cls,
                    "image_id": cls,
                    "text": template.format(name=names[cls]),
                    "split": view,
                    "template_index": j,
                    "mapping_split": assignments[cls],
                    "robust": cls in class_order[800:832],
                }
            )
    coco_rows = json.loads(
        (
            ROOT
            / "public/benchmarks/cross_dataset_retrieval_sources/karpathy/dataset_coco.json"
        ).read_text()
    )["images"]
    coco_rows = [
        r
        for r in coco_rows
        if r["split"] == "train" and int(r["cocoid"]) not in old_coco
    ]
    rng.shuffle(coco_rows)
    image_manifest = []
    for i, row in enumerate(coco_rows[:11776]):
        split = (
            "cal" if i < 512 else "fit" if i < 8704 else "dev" if i < 9728 else "test"
        )
        local = out / "coco-images" / row["filename"]
        image_id = 8_200_000_000 + int(row["cocoid"])
        base = {
            "group": int(row["cocoid"]),
            "image_id": image_id,
            "source_image_id": int(row["cocoid"]),
            "source_split": "karpathy_train",
            "split": split,
            "robust": split == "cal" or 9728 <= i < 10240,
        }
        url = f"https://s3.amazonaws.com/images.cocodataset.org/{row['filepath']}/{row['filename']}"
        samples["coco_images"].append(
            {**base, "source_path": str(local.resolve()), "download_url": url}
        )
        image_manifest.append({"img_id": image_id, "source_path": str(local.resolve())})
        captions = [s["raw"].strip() for s in row["sentences"]]
        assert len(captions) >= 5 and all(captions)
        for j, caption in enumerate(captions):
            samples["coco_texts"].append({**base, "text": caption, "caption_index": j})
    assert len(samples["coco_images"]) == 11776
    used_vg = set()
    for task in ("aro_vg_relation", "aro_vg_attribution"):
        rows = jsonl(
            ROOT
            / f"public/benchmarks/selfless_multimodal_likelihood_v1/tasks/{task}.jsonl"
        )
        rng.shuffle(rows)
        chosen = []
        for row in rows:
            identity = row["metadata"]["source_image"]
            if identity not in used_vg:
                used_vg.add(identity)
                chosen.append(row)
            if len(chosen) == 1000:
                break
        assert len(chosen) == 1000, (task, len(chosen))
        for row in chosen:
            base = {
                "group": row["image_id"],
                "image_id": row["image_id"],
                "split": "test",
                "robust": False,
                "task": task,
                "category": row["category"],
                "source_image": row["metadata"]["source_image"],
            }
            samples["aro_images"].append(base)
            for j, caption in enumerate(row["candidates"]):
                samples["aro_texts"].append(
                    {**base, "text": caption, "positive": j == row["label"]}
                )
    profiles = {
        name: {"prompt": name, "sigma": 0, "posterior": "sample", "subset": "all"}
        for name in ("native", "bare", "neutral")
    }
    for name, sigma, posterior in (
        ("native_sigma1", 1, "sample"),
        ("native_sigma2", 2, "sample"),
        ("native_mean", 0, "mean"),
    ):
        profiles[name] = {
            "prompt": "native",
            "sigma": sigma,
            "posterior": posterior,
            "subset": "robust",
        }
    v2protocol = json.loads(
        (diagnostic / "semantic-v2-20260907/protocol.json").read_text()
    )
    source = resolve_evaluation_model_source(RUN / "hf_model-final-ema")
    protocol = {
        "schema": "unified_geometry_v4_frozen_1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": SEED,
        "model": source.report(),
        "states": v2protocol["states"],
        "model_weights_frozen": True,
        "training_updates": 0,
        "runtime_hashing_enabled": False,
        "status": "expanded prospective measurement informed by completed V3; not preregistered external study",
        "development_notebook": "dev-wjx-ascend",
        "layers": LAYERS,
        "pools": ["content_mean", "content_last_sigma", "query_native"],
        "profiles": profiles,
        "states_profiles": {
            "final_ema": list(profiles),
            **{s: ["native"] for s in ("final_raw", "init42", "init43", "init44")},
        },
        "counts": {k: len(v) for k, v in samples.items()},
        "imagenet": {
            "fit_classes": 600,
            "dev_classes": 200,
            "test_classes": 200,
            "images_per_class": 32,
            "views": "2 cal + 15 a + 15 b, all disjoint from V1/V2 images",
            "prototype_sizes": [1, 3, 5, 10, 15],
            "text_templates": "4 cal + 4 a + 4 b; primary uses a in all class splits, b is wording-shift control",
            "calibration": "only mapping-fit classes; unseen means unseen by map, not B training",
        },
        "coco": {
            "cal": 512,
            "fit": 8192,
            "dev": 1024,
            "test": 2048,
            "fit_sizes": [512, 2048, 8192],
            "captions": "all captions grouped by image; 1/3/5-caption sensitivity",
            "source": "Karpathy train; excludes entire old retrieval test pool and V2 hard images",
            "claim": "OOD relative to this B's ImageNet image training; not full LM/VAE decontamination",
        },
        "aro": {
            "relation_images": 1000,
            "attribute_images": 1000,
            "unique_source_images": 2000,
            "fit": "none; transfer the frozen COCO maps; compare positive against controlled negative",
            "claim": "new geometry diagnostic on existing benchmark assets; no pristine benchmark claim",
        },
        "analysis": {
            "modes": ["centered_euclidean", "centered_unit_sphere"],
            "dimensions": ["full", 32, 128, 512],
            "mapping": "orthogonal plus global-scale supplement; fit-only PCA/centering/RMS; shuffled-fit controls",
            "geometry": "RSA, linear CKA, kNN 5/10/20, identity permutations and within-modality repeated views",
            "selection": "dev only; fixed final_norm plus all-layer curves; no test-max tuning",
            "uncertainty": "held-out group bootstrap and repeat fit subsets; record rank/conditioning and variance retention",
            "head_claim": "functional negative-caption tests do not prove the head lacks semantic computation",
        },
        "cache_roots": {
            "imagenet": str(imagenet_root / "vae_posterior_mar_kl16/val_shards"),
            "coco": str(out.resolve() / "coco-vae/shards"),
            "aro": str(
                ROOT
                / "public/benchmarks/selfless_multimodal_likelihood_v1/vae_posterior_mar_kl16_v2/shards"
            ),
        },
        "excluded_id_counts": {
            "imagenet": len(excluded_imagenet),
            "coco": len(old_coco),
        },
    }
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "samples.json", samples)
    with (out / "coco-image-manifest.jsonl").open("w") as handle:
        for row in image_manifest:
            handle.write(json.dumps(row) + "\n")
    write_json(out / "protocol.json", protocol)
    emit("geometry_v4_prepared", **protocol["counts"])


def download(out, workers):
    rows = json.loads((out / "samples.json").read_text())["coco_images"]

    def one(row):
        path = Path(row["source_path"])
        if path.exists():
            with Image.open(path) as im:
                im.verify()
            return {
                "source_image_id": row["source_image_id"],
                "bytes": path.stat().st_size,
                "reused": True,
            }
        last = None
        for attempt in range(4):
            try:
                response = requests.get(row["download_url"], timeout=(15, 60))
                response.raise_for_status()
                with Image.open(io.BytesIO(response.content)) as im:
                    im.verify()
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix(".download")
                temporary.write_bytes(response.content)
                temporary.replace(path)
                return {
                    "source_image_id": row["source_image_id"],
                    "bytes": len(response.content),
                    "reused": False,
                }
            except (requests.RequestException, OSError, ValueError) as exc:
                last = str(exc)
                time.sleep(min(1 + attempt, 4))
        raise RuntimeError(f"Download failed for {row['source_image_id']}: {last}")

    started, records = time.monotonic(), []
    with concurrent.futures.ThreadPoolExecutor(workers) as pool:
        for i, result in enumerate(pool.map(one, rows), 1):
            records.append(result)
            if i % 250 == 0 or i == len(rows):
                emit(
                    "coco_download_progress_v4",
                    completed=i,
                    total=len(rows),
                    seconds=round(time.monotonic() - started, 1),
                )
    write_json(
        out / "download-complete.json",
        {
            "images": len(records),
            "bytes": sum(r["bytes"] for r in records),
            "records": records,
            "runtime_hashing_enabled": False,
            "tls_verification": True,
        },
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("prepare", "download"))
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--workers", type=int, default=24)
    a = p.parse_args()
    if a.action == "prepare":
        prepare(a.output_dir)
    else:
        download(a.output_dir, a.workers)


if __name__ == "__main__":
    main()
