"""Read-only exact coverage, provenance, leakage and finite-value audit for V5."""

import argparse
import json
import multiprocessing as mp
from pathlib import Path

import torch
from safetensors import safe_open

from scripts.prepare_cross_model_geometry_v5 import F_RUN, V4
from utils.research.geometry_v5_assets import RUN, emit, write_json


def tasks_for(root, name):
    if name in {"b", "f"}:
        base = V4 if name == "b" else root / "f-v4"
        protocol = json.loads((base / "protocol.json").read_text())
        samples = json.loads((base / "samples.json").read_text())
        reference = json.loads((base / "model-verification/final_ema.json").read_text())
        stored = 0
        for file in Path(reference["path"]).glob("*.safetensors"):
            with safe_open(file, framework="pt") as source:
                stored += len(source.keys())
        assert stored == reference["stored_tensors_exactly_loaded"] and stored > 0
        if name == "f":
            assert reference["path"] == str(F_RUN / "hf_model-final-ema")
            assert (
                protocol["model"]["architecture_variant"]
                == "positionwise_flow_head_on_b"
            )
            for stage in ("f-cal-smoke", "f-full", "f-robust"):
                assert (
                    json.loads(
                        (root / "supervisor-markers" / f"{stage}.json").read_text()
                    )["returncode"]
                    == 0
                )
        tasks = []
        for profile_name in (
            "native",
            "bare",
            "neutral",
            "native_sigma1",
            "native_sigma2",
            "native_mean",
        ):
            profile = protocol["profiles"][profile_name]
            for dataset, rows in samples.items():
                indices = [
                    i
                    for i, row in enumerate(rows)
                    if profile["subset"] == "all" or row["robust"]
                ]
                if not indices:
                    continue
                for rank in range(16):
                    path = (
                        base
                        / "final_ema"
                        / profile_name
                        / f"{dataset}-rank-{rank:02d}-of-16.pt"
                    )
                    tasks.append(
                        {
                            "path": path,
                            "indices": indices[rank::16],
                            "name": name,
                            "dataset": dataset,
                            "profile": profile,
                            "reference": reference,
                            "layers": protocol["layers"],
                            "pools": protocol["pools"],
                        }
                    )
        return tasks, reference
    samples = json.loads((root / "samples.json").read_text())
    contract = json.loads((root / "adapter-contracts" / f"{name}.json").read_text())
    verification = json.loads(
        (root / "model-verification" / f"{name}.json").read_text()
    )
    assert (
        verification["contract"] == contract and verification["parameters_verified"] > 0
    )
    tasks = []
    flow = name in {"janusflow", "showo2"}
    if flow:
        vae = json.loads((root / "model-verification" / f"{name}-vae.json").read_text())
        assert vae["parameters_verified"] > 0
    for route in ("understanding", "generation") if flow else (None,):
        profiles = (
            ("main", "seed1", "seed2", "image_midpoint")
            if route == "generation"
            else ("main",)
        )
        for profile in profiles:
            for dataset, rows in samples.items():
                modality = "image" if dataset.endswith("images") else "text"
                if (
                    name == "qwen_text"
                    and modality != "text"
                    or name in {"dinov2", "mae"}
                    and modality != "image"
                ):
                    continue
                if profile == "image_midpoint" and modality != "image":
                    continue
                indices = [
                    i
                    for i, row in enumerate(rows)
                    if profile == "main" or row["robust"]
                ]
                if not indices:
                    continue
                directory = root / "features" / name
                if flow:
                    directory = directory / route / profile
                for rank in range(16):
                    tasks.append(
                        {
                            "path": directory / f"{dataset}-rank-{rank:02d}-of-16.pt",
                            "indices": indices[rank::16],
                            "name": name,
                            "dataset": dataset,
                            "profile": profile,
                            "route": route,
                            "contract": contract,
                            "layers": contract["layers"]
                            if flow
                            else contract["layers"][modality],
                            "pools": contract["pools"][route]
                            if flow
                            else contract["pools"][modality],
                        }
                    )
    actual = set((root / "features" / name).rglob("*-rank-*-of-16.pt"))
    assert actual == {t["path"] for t in tasks}, (
        name,
        "missing/extra feature shards",
        len(actual),
        len(tasks),
    )
    return tasks, verification


def audit_one(task):
    torch.set_num_threads(1)
    value = torch.load(task["path"], map_location="cpu", weights_only=True, mmap=True)
    assert value["indices"].tolist() == task["indices"]
    assert value["layers"] == task["layers"] and value["pools"] == task["pools"]
    assert value["dataset"] == task["dataset"]
    features = value["features"]
    assert features.shape[:3] == (
        len(task["indices"]),
        len(task["layers"]),
        len(task["pools"]),
    )
    expected_dtype = getattr(
        torch, task.get("contract", {}).get("storage_dtype", "bfloat16")
    )
    assert features.dtype == expected_dtype
    extras, upstream_shapes = [], {}
    if task["name"] in {"b", "f"}:
        assert value["schema"] == "unified_geometry_v4_features_1"
        assert value["profile"] == task["profile"] and value["hidden_target_invariance"]
        assert features.shape[-1] == 1024
        assert all(
            task["reference"][key] == val for key, val in value["state_checks"].items()
        )
        audit = {"hidden_target_invariance": True}
    else:
        assert value["schema"] == "cross_model_geometry_v5_features_1"
        assert value["contract"] == task["contract"] and value["training_updates"] == 0
        assert value["world_size"] == 16 and value["hostname"].startswith(
            "dev-wjx-ascend--"
        )
        audit = value["audit"]
        assert (
            audit["opposite_modality_replacement_exact"]
            and audit["batch_single_cosine"] > 0.995
        )
        if task["route"]:
            assert (
                value["path"] == task["route"] and value["profile"] == task["profile"]
            )
            upstream = value["upstream_features"]
            if task["route"] == "understanding" and task["dataset"].endswith("texts"):
                expected_upstream = {}
            elif task["name"] == "janusflow":
                expected_upstream = (
                    {"understanding_encoder": 1024}
                    if task["route"] == "understanding"
                    else {"generation_encoder": 768}
                )
            else:
                expected_upstream = {
                    "semantic_teacher_path": 1152,
                    "low_level_path": 1536,
                    "fused_input": 1536,
                }
            assert set(upstream) == set(expected_upstream)
            for key, width in expected_upstream.items():
                assert upstream[key].shape == (len(task["indices"]), width)
                upstream_shapes[key] = list(upstream[key].shape)
            extras.extend(upstream.values())
        else:
            extras.append(value["native_endpoint"])
    for tensor in [features, *extras]:
        assert len(tensor) == len(task["indices"])
        for start in range(0, len(tensor), 128):
            assert bool(torch.isfinite(tensor[start : start + 128]).all()), task["path"]
    return {
        "path": str(task["path"]),
        "bytes": task["path"].stat().st_size,
        "shape": list(features.shape),
        "audit": audit,
        "upstream_shapes": upstream_shapes,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=RUN)
    parser.add_argument(
        "--models", default="b,f,qwen_text,dinov2,mae,siglip,janusflow,showo2"
    )
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    for name in args.models.split(","):
        tasks, verification = tasks_for(args.output_dir, name)
        records = []
        with mp.get_context("spawn").Pool(args.workers) as pool:
            for i, row in enumerate(pool.imap_unordered(audit_one, tasks), 1):
                records.append(row)
                if i % 48 == 0:
                    emit(
                        "v5_feature_audit_progress",
                        model=name,
                        completed=i,
                        total=len(tasks),
                    )
        write_json(
            args.output_dir / "audits" / f"features-{name}.json",
            {
                "status": "passed",
                "model": name,
                "shards": len(records),
                "rows": sum(row["shape"][0] for row in records),
                "records": sorted(records, key=lambda row: row["path"]),
                "model_verification": verification,
                "scope": "all required formal and robust shards; for B/F audit retained 2000 ARO rows then analysis strictly filters verified868",
            },
        )
        emit("v5_feature_audit_passed", model=name, shards=len(records))


if __name__ == "__main__":
    main()
