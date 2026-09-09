"""Read-only complete feature/cache audit for the expanded geometry run."""

import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path

import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.research.representation_protocol import emit, write_json


def audit_feature(task):
    path, indices, state, profile, dataset, layers, pools = task
    torch.set_num_threads(1)
    value = torch.load(path, weights_only=True, mmap=True, map_location="cpu")
    assert value["indices"].tolist() == indices, path
    assert value["state_checks"]["state"] == state
    assert value["profile"] == profile and value["dataset"] == dataset
    assert value["layers"] == layers and value["pools"] == pools
    assert value["schema"] == "unified_geometry_v4_features_1"
    assert value["hidden_target_invariance"]
    assert value["features"].shape == (len(indices), 30, 3, 1024)
    assert value["features"].dtype == torch.bfloat16
    for start in range(0, len(indices), 128):
        assert bool(torch.isfinite(value["features"][start : start + 128]).all()), path
    return {
        "path": str(path),
        "samples": len(indices),
        "bytes": path.stat().st_size,
        "state_checks": value["state_checks"],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    root = a.output_dir
    torch.set_num_threads(1)
    protocol = json.loads((root / "protocol.json").read_text())
    samples = json.loads((root / "samples.json").read_text())
    completion = json.loads((root / "extraction-complete.json").read_text())
    assert all(not any(s["returncodes"]) for s in completion["completed"])
    expected, tasks = set(), []
    for state, profiles in protocol["states_profiles"].items():
        for name in profiles:
            profile = protocol["profiles"][name]
            for dataset, rows in samples.items():
                eligible = [
                    i
                    for i, r in enumerate(rows)
                    if profile["subset"] == "all" or r["robust"]
                ]
                if not eligible:
                    continue
                for rank in range(16):
                    path = root / state / name / f"{dataset}-rank-{rank:02d}-of-16.pt"
                    expected.add(path)
                    tasks.append(
                        (
                            path,
                            eligible[rank::16],
                            state,
                            profile,
                            dataset,
                            protocol["layers"],
                            protocol["pools"],
                        )
                    )
    actual = {
        p
        for state in protocol["states_profiles"]
        for p in (root / state).glob("*/*-rank-*-of-16.pt")
        if not p.name.startswith("smoke-")
    }
    assert expected == actual, (
        f"Missing {len(expected - actual)}, unexpected {len(actual - expected)}"
    )
    records = []
    with mp.get_context("fork").Pool(a.workers) as pool:
        for i, record in enumerate(pool.imap_unordered(audit_feature, tasks), 1):
            records.append(record)
            if i % 100 == 0:
                emit("feature_audit_progress_v4", completed=i, total=len(tasks))
    for state in protocol["states"]:
        reference = json.loads(
            (root / "model-verification" / f"{state}.json").read_text()
        )
        if state.startswith("final"):
            stored = 0
            for source in Path(reference["path"]).glob("*.safetensors"):
                with safe_open(source, framework="pt", device="cpu") as handle:
                    stored += len(handle.keys())
            assert stored > 0 and reference["stored_tensors_exactly_loaded"] == stored
        else:
            assert (
                reference["pretrained_backbone_equality_checked"]
                and reference["mask_equal"]
            )
        for record in records:
            if record["state_checks"]["state"] == state:
                assert all(reference[k] == v for k, v in record["state_checks"].items())
    expected_ids = [r["image_id"] for r in samples["coco_images"]]
    shard_paths = list((root / "coco-vae/shards").glob("shard-*-of-00016.pt"))
    assert len(shard_paths) == 16
    found = []
    for path in shard_paths:
        shard = torch.load(path, weights_only=True, mmap=True, map_location="cpu")
        found.extend(shard["img_ids"].tolist())
        m, x = shard["metadata"], shard["posterior_stats"]
        assert m["runtime_hashing_enabled"] is False and m["device_type"] == "npu"
        assert m["scaling_factor"] == 0.2325 and x.shape[1:] == (256, 32)
        assert x.dtype == torch.float16 and bool(torch.isfinite(x).all())
        assert bool(x[..., 16:].ge(0).all())
    assert sorted(found) == sorted(expected_ids) and len(set(found)) == 11776
    write_json(
        root / "feature-audit-v4.json",
        {
            "status": "passed",
            "feature_shards": len(records),
            "feature_rows": sum(r["samples"] for r in records),
            "feature_bytes": sum(r["bytes"] for r in records),
            "coco_posterior_images": len(found),
            "model_states_verified": list(protocol["states"]),
            "checks": [
                "exact sample/rank coverage",
                "finite BF16 30x3x1024 features",
                "matching provenance/profiles",
                "hidden-target checks recorded",
                "loaded weights verified",
                "new VAE cache identity and numerical validation",
            ],
        },
    )
    emit(
        "feature_audit_passed_v4",
        shards=len(records),
        rows=sum(r["samples"] for r in records),
    )


if __name__ == "__main__":
    main()
