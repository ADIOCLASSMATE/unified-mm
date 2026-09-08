"""Identity-paired V5 model contrasts at fixed and independently dev-selected endpoints."""

import argparse
import json
import multiprocessing as mp
from functools import lru_cache
from pathlib import Path

import torch

from scripts.bootstrap_geometry_v5 import SEED, freeze_statistics
from scripts.prepare_geometry_v5_assets import RUN, emit, write_json


@lru_cache(maxsize=8)
def draws_for(n, repeats=2000):
    return torch.randint(n, (repeats, n), generator=torch.Generator().manual_seed(SEED))


def interval(values):
    return values.quantile(torch.tensor([0.025, 0.975], dtype=values.dtype)).tolist()


def paired_r2(a, b, repeats=2000):
    assert torch.equal(a["groups"], b["groups"]), (
        "Pair by semantic identity, not array length"
    )
    assert a["target_family"] == b["target_family"]
    draw = draws_for(len(a["groups"]), repeats)
    estimates, advantages, points, point_advantages = [], [], [], []
    for value in (a, b):
        errors, baseline = value["errors"].double(), value["baseline"].double()
        benefit = value["shuffled_errors"].double() - errors
        denominator = baseline[draw].sum(1).clamp_min(1e-20)
        estimates.append(1 - errors[draw].sum(1) / denominator)
        advantages.append(benefit[draw].sum(1) / denominator)
        points.append(float(1 - errors.sum() / baseline.sum().clamp_min(1e-20)))
        point_advantages.append(float(benefit.sum() / baseline.sum().clamp_min(1e-20)))
    return {
        "points": len(a["groups"]),
        "target_family": a["target_family"],
        "a_r2": points[0],
        "b_r2": points[1],
        "a_minus_b_r2": points[0] - points[1],
        "a_minus_b_r2_95": interval(estimates[0] - estimates[1]),
        "a_minus_b_advantage_over_shuffled_fit": point_advantages[0]
        - point_advantages[1],
        "a_minus_b_advantage_over_shuffled_fit_95": interval(
            advantages[0] - advantages[1]
        ),
    }


def paired_mean(a, b, repeats=2000):
    assert a.shape == b.shape and a.ndim == 1
    difference = a.double() - b.double()
    return {
        "a": float(a.double().mean()),
        "b": float(b.double().mean()),
        "a_minus_b": float(difference.mean()),
        "a_minus_b_95": interval(difference[draws_for(len(a), repeats)].mean(1)),
        "points": len(a),
    }


@lru_cache(maxsize=64)
def load_arrays(path):
    return torch.load(path, map_location="cpu", weights_only=True)["arrays"]


def key_for(row, readout_override=None):
    return (
        row["endpoint"],
        readout_override or row["readout"],
        row["family"],
        row["mode"],
        row["fit_points"],
        row["requested_dimension"] if row["endpoint"] == "fixed" else "selected",
    )


def compare_endpoints(a, b, endpoint_paths):
    common = a["readout"] == b["readout"] and a["readout"] in {
        "content_mean",
        "content_last",
    }
    primary = (
        a["setting"] == "b_native"
        and b["setting"] not in {"b_bare", "b_neutral"}
        and a["endpoint"] == "fixed"
        and a["readout"] == b["readout"] == "content_mean"
        and a["mode"] == "centered_euclidean"
        and a["requested_dimension"] == b["requested_dimension"] == 32
        and a["fit_points"] == (600 if a["family"] == "imagenet" else 8192)
    )
    result = {
        "setting_a": a["setting"],
        "setting_b": b["setting"],
        "endpoint": a["endpoint"],
        "readout_a": a["readout"],
        "readout_b": b["readout"],
        "family": a["family"],
        "mode": a["mode"],
        "fit_points": a["fit_points"],
        "dimension_a": a["dimension"],
        "dimension_b": b["dimension"],
        "pair_a": a["pair"],
        "pair_b": b["pair"],
        "primary_contrast": primary,
        "common_content_readout": common,
        "endpoint_files": endpoint_paths,
        "scope": "same identity bootstrap, each model retains its own representation-space denominator; not equal pretrained data/parameter budgets",
    }
    va, vb = load_arrays(a["sample_statistics"]), load_arrays(b["sample_statistics"])
    for split in ("test", "transfer_test"):
        result[split] = paired_r2(va[split], vb[split])
    if "geometry" in va and "geometry" in vb:
        ga, gb = va["geometry"], vb["geometry"]
        assert torch.equal(ga["groups"], gb["groups"])
        result["geometry"] = {
            key: paired_mean(value, gb["anchors"][key])
            for key, value in ga["anchors"].items()
        }
    if "aro" in va and "aro" in vb:
        aa, ab = va["aro"], vb["aro"]
        assert torch.equal(aa["groups"], ab["groups"]) and aa["tasks"] == ab["tasks"]
        result["aro"] = {}
        for task in sorted(set(aa["tasks"])):
            ix = torch.tensor(
                [i for i, value in enumerate(aa["tasks"]) if value == task]
            )
            result["aro"][task] = {
                control: {
                    metric: paired_mean(
                        aa["scores"][control][metric][ix],
                        ab["scores"][control][metric][ix],
                    )
                    for metric in ("cosine", "distance")
                }
                for control in ("paired_fit", "shuffled_fit", "shuffled_image")
            }
    return result


def compare_task(task):
    return compare_endpoints(*task)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=RUN)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    assert args.workers > 0
    root = args.output_dir
    torch.set_num_threads(1)
    freeze_statistics(root)
    contract = json.loads((root / "comparison-contract.json").read_text())
    endpoints, locations = {}, {}
    for setting in contract["settings"]:
        audit = json.loads((root / "audits" / f"endpoints-{setting}.json").read_text())
        paths = sorted((root / "endpoints" / setting).glob("*.json"))
        assert {str(p) for p in paths} == set(audit["paths"]) and len(paths) == audit[
            "expected"
        ]
        endpoints[setting] = {}
        for path in paths:
            row = json.loads(path.read_text())
            if row["valid"] and row["requested_dimension"] != "full":
                key = key_for(row)
                assert key not in endpoints[setting]
                endpoints[setting][key] = row
                locations[setting, key] = str(path)
    contrasts = [
        ("b_native", other)
        for other in (
            "f_native",
            "dinov2_qwen",
            "mae_qwen",
            "siglip",
            "janusflow_understanding",
            "janusflow_generation",
            "showo2_understanding",
            "showo2_generation",
        )
    ]
    contrasts += [
        ("b_bare", "f_bare"),
        ("b_neutral", "f_neutral"),
        ("b_native", "b_bare"),
        ("b_native", "b_neutral"),
        ("f_native", "f_bare"),
        ("f_native", "f_neutral"),
    ]
    tasks, missing = [], []
    for a, b in contrasts:
        shared = sorted(endpoints[a].keys() & endpoints[b].keys(), key=str)
        missing.append(
            {
                "a": a,
                "b": b,
                "matched_valid": len(shared),
                "valid_only_a": len(endpoints[a].keys() - endpoints[b].keys()),
                "valid_only_b": len(endpoints[b].keys() - endpoints[a].keys()),
            }
        )
        for key in shared:
            tasks.append(
                (
                    endpoints[a][key],
                    endpoints[b][key],
                    [locations[a, key], locations[b, key]],
                )
            )
        emit("v5_model_contrast_planned", a=a, b=b, endpoints=len(shared))
    # The trained SigLIP pooler is a useful reference, but not common backbone pooling.
    for key, b in endpoints["siglip_native"].items():
        akey = key_for(b, "content_mean")
        if akey in endpoints["b_native"]:
            a = endpoints["b_native"][akey]
            tasks.append(
                (a, b, [locations["b_native", akey], locations["siglip_native", key]])
            )
    if args.workers == 1:
        rows = list(map(compare_task, tasks))
    else:
        rows = []
        with mp.get_context("fork").Pool(
            args.workers, initializer=torch.set_num_threads, initargs=(1,)
        ) as pool:
            for row in pool.imap(compare_task, tasks, chunksize=4):
                rows.append(row)
                if len(rows) % 128 == 0:
                    emit(
                        "v5_paired_comparison_progress",
                        completed=len(rows),
                        total=len(tasks),
                    )
    write_json(
        root / "paired-model-differences-v5.json",
        {
            "schema": "geometry_v5_paired_model_differences_1",
            "bootstrap": 2000,
            "seed": SEED,
            "rows": rows,
            "coverage": missing,
            "note": "Pointwise conditional intervals, not simultaneous multiple-comparison or new-training-run uncertainty. A and B in each row are comparison labels, not architectural variants.",
        },
    )
    emit("v5_paired_comparison_complete", contrasts=len(rows))


if __name__ == "__main__":
    main()
