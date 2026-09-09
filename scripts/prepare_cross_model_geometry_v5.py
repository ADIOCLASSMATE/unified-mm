"""Freeze V4-identical samples and initialize the cross-model V5 experiment."""

import argparse
import json
from pathlib import Path

from utils.research.geometry_v5_assets import ROOT, RUN, write_json

BASELINE = ROOT / "output/unified-b-x0content-0p6b-100b-imagenet-split-s42-r1"
V4 = ROOT / "output/evaluation/research" / BASELINE.name / "representation-diagnostic/geometry-v4-20260907"
F_RUN = ROOT / "output/unified-f-on-b-0p6b-100b-imagenet-split-s42-r1"


def freeze(out):
    existing = out / "protocol.json"
    if existing.exists():
        print("V5 protocol already frozen", flush=True)
        return
    old_samples = json.loads((V4 / "samples.json").read_text())
    old_protocol = json.loads((V4 / "protocol.json").read_text())
    audit = json.loads((V4 / "sample-audit-v4.json").read_text())
    assert len(old_samples["imagenet_images"]) == 32000
    assert len(old_samples["coco_images"]) == 11776
    strict = {
        r["index"]
        for r in audit["aro_identity_details"]
        if r["category"] == "verified_coco_disjoint"
    }
    assert len(strict) == 868
    sample_copy = json.loads(json.dumps(old_samples))
    image_paths = {
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
    for index, row in enumerate(sample_copy["aro_images"]):
        row["v4_index"] = index
        row["source_path"] = image_paths[row["image_id"]]
    sample_copy["aro_images"] = [
        row for i, row in enumerate(sample_copy["aro_images"]) if i in strict
    ]
    strict_groups = {r["group"] for r in sample_copy["aro_images"]}
    sample_copy["aro_texts"] = [
        row for row in sample_copy["aro_texts"] if row["group"] in strict_groups
    ]
    assert len(sample_copy["aro_texts"]) == 1736
    protocol = {
        "schema": "cross_model_geometry_v5_1",
        "seed": old_protocol["seed"],
        "baseline_v4": str(V4),
        "protocol_document": "docs/CROSS_MODEL_GEOMETRY_PROTOCOL_V5.md",
        "training_updates": 0,
        "runtime_hashing_enabled": False,
        "models": [
            "b",
            "f",
            "janusflow",
            "showo2",
            "siglip",
            "dinov2_qwen",
            "mae_qwen",
        ],
        "analysis": {
            "modes": ["centered_euclidean", "unit_sphere"],
            "pca_dimensions": [32, 128, 512],
            "full_dimension": "diagnostic_only_if_equal_dimensions",
            "mapping_fit_sizes": {"imagenet": [600], "coco": [512, 2048, 8192]},
            "permutations": 199,
            "bootstrap": 2000,
            "layer_selection": "source_dev_only",
            "target_refitting": False,
            "knn_k": [5, 10, 20],
            "geometry_test_units": {"imagenet": 200, "coco": 512},
        },
        "sample_counts": {k: len(v) for k, v in sample_copy.items()},
        "aro_identity": "868 verified COCO-pool-disjoint images; V4 crops and paired negatives",
        "notebook": {
            "name": "dev-wjx-ascend",
            "workspace": "昇腾卡公共空间",
            "account": "wjx-ascend",
            "started_by_this_experiment": True,
            "initial_state": "STOPPED",
            "start_accepted": True,
            "image": "dev-wjx-ascend:v-1.4",
            "accelerators": 16,
        },
    }
    write_json(out / "samples.json", sample_copy)
    write_json(existing, protocol)
    f_root = out / "f-v4"
    f_protocol = json.loads(json.dumps(old_protocol))
    f_protocol["model"] = {
        "run": str(F_RUN),
        "architecture_variant": "positionwise_flow_head_on_b",
    }
    f_protocol["states"] = {
        "final_ema": {"path": str(F_RUN / "hf_model-final-ema"), "init_seed": None}
    }
    f_protocol["states_profiles"] = {
        "final_ema": [
            "native",
            "bare",
            "neutral",
            "native_sigma1",
            "native_sigma2",
            "native_mean",
        ]
    }
    f_protocol["parent_cross_model_v5"] = str(out)
    write_json(f_root / "protocol.json", f_protocol)
    write_json(f_root / "samples.json", old_samples)
    print(
        json.dumps(
            {"event": "v5_protocol_frozen", "samples": protocol["sample_counts"]}
        ),
        flush=True,
    )


def ensure_f_smoke(out):
    target = out / "f-smoke"
    source = out / "f-v4"
    rows = json.loads((source / "samples.json").read_text())
    filtered = {
        name: [row for row in values if row["split"] == "cal"][:32]
        for name, values in rows.items()
        if not name.startswith("aro")
    }
    assert all(len(values) == 32 for values in filtered.values())
    if not (target / "samples.json").exists():
        write_json(target / "samples.json", filtered)
        write_json(
            target / "protocol.json", json.loads((source / "protocol.json").read_text())
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=RUN)
    output_dir = parser.parse_args().output_dir
    freeze(output_dir)
    ensure_f_smoke(output_dir)
