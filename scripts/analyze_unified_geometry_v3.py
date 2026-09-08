"""CPU-only geometry exploration of frozen B features; never modifies B weights.

Compare point-cloud shape, semantic neighborhoods, and held-out orthogonal maps.
This is a post-V2 exploratory analysis, not a new untouched confirmatory test.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.analyze_unified_semantics_v2 import load_dataset
from scripts.probe_unified_representations import emit, write_json
from scripts.probe_unified_semantics_v2 import LAYERS

SEED = 20260908
READOUTS = {"content_mean": 0, "query_native": 2}
PCA_DIMS = (32, 128)
MODES = ("centered_euclidean", "centered_unit_sphere")
PAIRS = [
    (state, profile)
    for state in ("final_ema", "final_raw", "init42", "init43", "init44")
    for profile in (
        ("bare", "native", "neutral") if state == "final_ema" else ("bare", "native")
    )
]
DATA = {}
CONTEXT = {}


def prepare(root, source):
    source_protocol = json.loads((source / "protocol.json").read_text())
    protocol = {
        "schema": "unified_geometry_v3_1",
        "analysis_status": "exploratory: V2 data and results were already inspected",
        "source_features": str(source.resolve()),
        "source_model": source_protocol["model"],
        "states_and_profiles": PAIRS,
        "layers": LAYERS,
        "readouts": READOUTS,
        "preprocessing": MODES,
        "pca_dimensions": PCA_DIMS,
        "seed": SEED,
        "geometry_permutations": 199,
        "neighbor_k": [5, 10, 20],
        "bootstrap_repeats_final_norm": 2000,
        "training_updates": 0,
        "npu_forwards": 0,
        "runtime_hashing_enabled": False,
        "imagenet": {
            "unit": "class prototype: mean 3 image features / mean 2 text-template features BEFORE preprocessing",
            "fit_classes": 600,
            "dev_classes": 200,
            "test_classes": 200,
            "centering": "calibration images/texts ONLY from the 600 mapping-fit classes",
            "test": "test-class test images/templates; repeated view uses disjoint fit images/templates of the same test classes",
            "geometry_points": 200,
            "unseen_claim": "classes unseen by the fitted alignment map, NOT unseen by B training",
        },
        "coco": {
            "unit": "one image / mean of all its caption features BEFORE preprocessing",
            "calibration_scenes": 500,
            "fit_scenes": 500,
            "dev_scenes": 500,
            "test_scenes": 1000,
            "geometry_points": 500,
            "geometry_subset": "fixed permutation of the V2 test-scene order, seed 20260908",
            "text_repeat": "first two captions vs remaining captions; paired by image identity",
            "image_repeat": "not available: no independent image views collected in the main V2 cache",
            "domain_claim": "cross-dataset relative to B's ImageNet image training; no full pretraining decontamination claim",
        },
        "mapping": {
            "primary": "orthogonal Q, with independently fitted global RMS units; no per-coordinate whitening",
            "supplement": "also report one paired-fit global scale, i.e. similarity Procrustes",
            "fit": "fit-split centering, PCA, RMS units, Q; all frozen on dev/test and cross-dataset transfer",
            "full_dimension_warning": "full 1024-D rotation is underidentified by 600 class or 500 scene prototypes",
            "pca_warning": "independent per-modality PCA is fit only on alignment-fit examples; successful PCA maps concern subspaces, not the full representation",
            "transfer": "bidirectional ImageNet-to-COCO / COCO-to-ImageNet evaluation; no target recentering, tuning, or paired target fitting",
            "null": "one fixed training-pair shuffle with the same preprocessing/capacity, evaluated against correct held-out identities",
            "error": "NRMSE = sqrt(total error / total squared target distance from fit mean); R2 = 1 - NRMSE^2; negatives are retained",
        },
        "statistics": {
            "geometry": "same row permutation across all layers; excludes self-neighbors and diagonal RDM entries",
            "null_calibration": "record empirical null and max-over-layers null for each fixed curve; not whole-experiment familywise control",
            "bootstrap": "resample class/scene units at final_norm, conditional on fixed fit/PCA/candidate pool",
            "selection": "dev only: minimize orthogonal NRMSE jointly over layers and full/PCA dimensions, separately per source/profile/readout/mode",
            "degeneracy": "constant representations marked invalid, never treated as perfect geometric agreement",
            "reliability": "within-modality repeated-view references, not asserted to be strict upper bounds",
        },
    }
    path = root / "protocol.json"
    canonical = json.loads(json.dumps(protocol))
    if path.exists():
        if json.loads(path.read_text()) != canonical:
            raise ValueError("V3 protocol differs; use a new analysis version/path")
    else:
        write_json(path, protocol)
    return protocol


def average_groups(features, rows, groups, predicate):
    """Group before any L2 normalization; fixed group order preserves semantics."""
    lookup = {g: i for i, g in enumerate(groups)}
    chosen = [
        i for i, row in enumerate(rows) if predicate(row) and row["group"] in lookup
    ]
    dest = torch.tensor([lookup[rows[i]["group"]] for i in chosen])
    counts = torch.bincount(dest, minlength=len(groups))
    if not bool(counts.gt(0).all()):
        raise ValueError("A semantic group has no view")
    result = torch.zeros(len(groups), len(LAYERS), len(READOUTS), features.shape[-1])
    # Limit the FP32 temporary instead of expanding all raw examples together.
    for start in range(0, len(chosen), 256):
        block = chosen[start : start + 256]
        result.index_add_(
            0,
            dest[start : start + 256],
            features[block][:, :, list(READOUTS.values())].float(),
        )
    return result / counts[:, None, None, None]


def build_views(source, state, profile_name):
    source_protocol = json.loads((source / "protocol.json").read_text())
    samples = json.loads((source / "samples.json").read_text())
    profile = source_protocol["profiles"][profile_name]
    result = {}
    for family in ("imagenet", "coco"):
        bundle = {}
        if family == "imagenet":
            assignments = {
                row["group"]: row["mapping_split"] for row in samples["imagenet_images"]
            }
            groups = {
                split: sorted(g for g, s in assignments.items() if s == split)
                for split in ("fit", "dev", "test")
            }
            assert [len(groups[s]) for s in ("fit", "dev", "test")] == [600, 200, 200]
        else:
            groups = {
                split: [
                    r["group"] for r in samples["coco_images"] if r["split"] == split
                ]
                for split in ("cal", "fit", "dev", "test")
            }
        for modality in ("images", "texts"):
            features, rows, checks = load_dataset(
                source / state / profile_name,
                f"{family}_{modality}",
                samples[f"{family}_{modality}"],
                profile,
                state,
            )
            if features is None:
                raise ValueError(
                    f"Missing source {state}/{profile_name}/{family}/{modality}"
                )
            key = "x" if modality == "images" else "y"
            if family == "imagenet":
                calibration = average_groups(
                    features, rows, groups["fit"], lambda r: r["split"] == "cal"
                )
                bundle[f"cal_{key}"] = calibration.mean(0)
                for split in ("fit", "dev", "test"):
                    sample_split = "fit" if split in ("fit", "dev") else "test"
                    bundle[f"{split}_{key}"] = average_groups(
                        features,
                        rows,
                        groups[split],
                        lambda r, s=sample_split: r["split"] == s,
                    )
                bundle[f"repeat_{key}"] = average_groups(
                    features, rows, groups["test"], lambda r: r["split"] == "fit"
                )
            else:
                calibration = average_groups(
                    features, rows, groups["cal"], lambda r: r["split"] == "cal"
                )
                bundle[f"cal_{key}"] = calibration.mean(0)
                for split in ("fit", "dev", "test"):
                    bundle[f"{split}_{key}"] = average_groups(
                        features,
                        rows,
                        groups[split],
                        lambda r, s=split: r["split"] == s,
                    )
                if modality == "texts":
                    bundle["repeat_y_a"] = average_groups(
                        features,
                        rows,
                        groups["test"],
                        lambda r: r["split"] == "test" and r["caption_index"] < 2,
                    )
                    bundle["repeat_y_b"] = average_groups(
                        features,
                        rows,
                        groups["test"],
                        lambda r: r["split"] == "test" and r["caption_index"] >= 2,
                    )
            del features
            bundle["state_checks"] = checks
        bundle["test_groups"] = groups["test"]
        if family == "coco":
            bundle["geometry_indices"] = torch.randperm(
                len(groups["test"]), generator=torch.Generator().manual_seed(SEED)
            )[:500]
        else:
            bundle["geometry_indices"] = torch.arange(len(groups["test"]))
        result[family] = bundle
        emit("geometry_views_ready", state=state, profile=profile_name, family=family)
    return result


def preprocess(x, calibration, mode):
    centered = x.double() - calibration.double()
    return (
        F.normalize(centered, dim=-1, eps=1e-12)
        if mode == "centered_unit_sphere"
        else centered
    )


def normalized_vector(x):
    value = x - x.mean()
    return value / value.norm().clamp_min(1e-14)


def ranks_with_ties(values):
    ordered, indices = values.sort()
    _, inverse, counts = ordered.unique_consecutive(
        return_inverse=True, return_counts=True
    )
    end = counts.cumsum(0).to(values.dtype)
    average_rank = end - (counts.to(values.dtype) - 1) / 2
    result = torch.empty_like(values)
    result[indices] = average_rank[inverse]
    return result


def geometry_statistics(x, y, permutations=199, ks=(5, 10, 20)):
    n = len(x)
    if n != len(y) or n <= max(ks):
        raise ValueError("Geometry requires paired points and more points than k")
    x, y = x.double(), y.double()
    xc, yc = x - x.mean(0), y - y.mean(0)
    if min(float(xc.norm()), float(yc.norm())) <= 1e-10:
        return {"valid": False, "reason": "constant representation", "points": n}
    gx, gy = xc @ xc.T, yc @ yc.T
    gx, gy = gx / gx.norm(), gy / gy.norm()
    upper = torch.triu_indices(n, n, 1)
    dx, dy = torch.cdist(x, x), torch.cdist(y, y)
    vx, vy = dx[upper[0], upper[1]], dy[upper[0], upper[1]]
    px, py = normalized_vector(vx), normalized_vector(vy)
    rx, ry = (
        normalized_vector(ranks_with_ties(vx)),
        normalized_vector(ranks_with_ties(vy)),
    )
    pearson_y = torch.zeros(n, n, dtype=torch.float64)
    ranks_y = torch.zeros_like(pearson_y)
    pearson_y[upper[0], upper[1]] = py
    pearson_y[upper[1], upper[0]] = py
    ranks_y[upper[0], upper[1]] = ry
    ranks_y[upper[1], upper[0]] = ry
    dx.fill_diagonal_(float("inf"))
    dy.fill_diagonal_(float("inf"))
    ix = dx.argsort(dim=1, stable=True)[:, : max(ks)]
    iy = dy.argsort(dim=1, stable=True)[:, : max(ks)]
    adjacency_x, adjacency_y = {}, {}
    for k in ks:
        ax, ay = (
            torch.zeros(n, n, dtype=torch.bool),
            torch.zeros(n, n, dtype=torch.bool),
        )
        ax.scatter_(1, ix[:, :k], True)
        ay.scatter_(1, iy[:, :k], True)
        adjacency_x[k], adjacency_y[k] = ax, ay
    observed = {
        "linear_cka": float((gx * gy).sum()),
        "distance_pearson": float(px @ py),
        "rsa_spearman": float(rx @ ry),
        **{
            f"knn_{k}": float((adjacency_x[k] & adjacency_y[k]).sum()) / (n * k)
            for k in ks
        },
    }
    null = {key: [] for key in observed}
    gen = torch.Generator().manual_seed(SEED)
    for _ in range(permutations):
        p = torch.randperm(n, generator=gen)
        rr, cc = p[upper[0]], p[upper[1]]
        null["linear_cka"].append(float((gx * gy[p][:, p]).sum()))
        null["distance_pearson"].append(float(px @ pearson_y[rr, cc]))
        null["rsa_spearman"].append(float(rx @ ranks_y[rr, cc]))
        for k in ks:
            null[f"knn_{k}"].append(
                float((adjacency_x[k] & adjacency_y[k][p][:, p]).sum()) / (n * k)
            )
    output = {"valid": True, "points": n, "scores": observed, "null_samples": null}
    output["null_summary"] = {
        key: {
            "mean": sum(values) / len(values),
            "q95": float(torch.tensor(values).quantile(0.95)),
            "tail_p": (1 + sum(v >= observed[key] for v in values)) / (len(values) + 1),
        }
        for key, values in null.items()
        if values
    }
    output["knn_chance_adjusted"] = {
        str(k): (observed[f"knn_{k}"] - k / (n - 1)) / (1 - k / (n - 1)) for k in ks
    }
    return output


def fitting_frame(x, dimension):
    mean = x.mean(0)
    xc = x - mean
    total = xc.square().sum()
    if float(total) <= 1e-18:
        return None
    _, singular, vh = torch.linalg.svd(xc, full_matrices=False)
    rank = int((singular > singular[0] * 1e-6).sum())
    participation = float(singular.square().sum().square() / singular.pow(4).sum())
    if dimension == "full":
        projection = None
        z = xc
        retained = 1.0
    else:
        dim = int(dimension)
        if rank < dim:
            return {"invalid": True, "numerical_rank": rank, "requested_dimension": dim}
        projection = vh[:dim].T
        z = xc @ projection
        retained = float(z.square().sum() / total)
    radius = z.square().sum(1).mean().sqrt()
    return {
        "mean": mean,
        "projection": projection,
        "radius": radius,
        "fit": z / radius,
        "retained_variance_fit": retained,
        "numerical_rank": rank,
        "participation_ratio": participation,
        "dimension": x.shape[1] if dimension == "full" else int(dimension),
    }


def apply_frame(frame, values):
    centered = values - frame["mean"]
    projected = (
        centered if frame["projection"] is None else centered @ frame["projection"]
    )
    retained = float(
        projected.square().sum() / centered.square().sum().clamp_min(1e-20)
    )
    return projected / frame["radius"], retained


def solve_rotation(x, y):
    u, singular, vh = torch.linalg.svd(x.T @ y, full_matrices=False)
    q = u @ vh
    scale = singular.sum() / x.square().sum().clamp_min(1e-20)
    return q, scale


def error_metrics(predicted, y, scale=1.0):
    errors = (predicted - y).square().sum(1)
    baseline = y.square().sum(1)
    nmse = float(errors.sum() / baseline.sum().clamp_min(1e-20))
    scaled_nmse = float(
        (predicted * scale - y).square().sum() / baseline.sum().clamp_min(1e-20)
    )
    return (
        {
            "nrmse": nmse**0.5,
            "r2": 1 - nmse,
            "similarity_nrmse": scaled_nmse**0.5,
            "similarity_r2": 1 - scaled_nmse,
            "paired_cosine": float(
                F.cosine_similarity(predicted, y, dim=1, eps=1e-12).mean()
            ),
        },
        errors,
        baseline,
    )


def error_bootstrap(errors, null_errors, baseline, repeats=2000):
    draw = torch.randint(
        len(errors),
        (repeats, len(errors)),
        generator=torch.Generator().manual_seed(SEED),
    )
    denominator = baseline[draw].sum(1).clamp_min(1e-20)
    r2 = 1 - errors[draw].sum(1) / denominator
    advantage = (null_errors - errors)[draw].sum(1) / denominator
    quantiles = torch.tensor([0.025, 0.975], dtype=r2.dtype)
    return {
        "r2_95_interval": r2.quantile(quantiles).tolist(),
        "r2_advantage_over_shuffled_fit_95_interval": advantage.quantile(
            quantiles
        ).tolist(),
    }


def alignment_experiment(source, target, mode, dimension, bootstrap=False):
    xfit = preprocess(source["fit_x"], source["cal_x"], mode)
    yfit = preprocess(source["fit_y"], source["cal_y"], mode)
    fx, fy = fitting_frame(xfit, dimension), fitting_frame(yfit, dimension)
    if fx is None or fy is None or fx.get("invalid") or fy.get("invalid"):
        return {"valid": False, "reason": "constant or rank-deficient fitting subspace"}
    q, scale = solve_rotation(fx["fit"], fy["fit"])
    p = torch.randperm(len(xfit), generator=torch.Generator().manual_seed(SEED))
    null_q, null_scale = solve_rotation(fx["fit"], fy["fit"][p])
    output = {
        "valid": True,
        "fit_points": len(xfit),
        "dimension": fx["dimension"],
        "rank_x": fx["numerical_rank"],
        "rank_y": fy["numerical_rank"],
        "effective_dimension_x": fx["participation_ratio"],
        "effective_dimension_y": fy["participation_ratio"],
        "fit_variance_retained_x": fx["retained_variance_fit"],
        "fit_variance_retained_y": fy["retained_variance_fit"],
        "full_rotation_identified": min(fx["numerical_rank"], fy["numerical_rank"])
        >= fx["dimension"],
        "paired_fitted_global_scale": float(scale),
        "shuffled_fitted_global_scale": float(null_scale),
    }
    for split in ("fit", "dev", "test", "transfer_test"):
        values = target if split == "transfer_test" else source
        key = "test" if split == "transfer_test" else split
        # Freeze SOURCE preprocessing even on a different target dataset.
        x = preprocess(values[f"{key}_x"], source["cal_x"], mode)
        y = preprocess(values[f"{key}_y"], source["cal_y"], mode)
        x, retained_x = apply_frame(fx, x)
        y, retained_y = apply_frame(fy, y)
        paired, errors, baseline = error_metrics(x @ q, y, scale)
        shuffled, null_errors, _ = error_metrics(x @ null_q, y, null_scale)
        record = {
            "points": len(x),
            "paired": paired,
            "shuffled_fit": shuffled,
            "variance_retained_x": retained_x,
            "variance_retained_y": retained_y,
        }
        if bootstrap and split in ("test", "transfer_test"):
            record["bootstrap"] = error_bootstrap(errors, null_errors, baseline)
        output[split] = record
    return output


def layer_bundle(values, layer, pool):
    return {
        key: value[:, layer, pool].double()
        if key not in ("cal_x", "cal_y")
        else value[layer, pool].double()
        for key, value in values.items()
        if torch.is_tensor(value) and key != "geometry_indices"
    }


def analyze_one(task):
    layer_index, readout = task
    torch.set_num_threads(CONTEXT["threads"])
    layer = LAYERS[layer_index]
    path = Path(CONTEXT["out"]) / f"layer-{layer}-{readout}.json"
    if path.exists():
        return str(path)
    started = time.monotonic()
    pool = list(READOUTS).index(readout)
    bundles = {
        family: layer_bundle(values, layer_index, pool)
        for family, values in DATA.items()
    }
    row = {
        "state": CONTEXT["state"],
        "profile": CONTEXT["profile"],
        "layer": layer,
        "readout": readout,
        "families": {},
    }
    for family, source in bundles.items():
        other = "coco" if family == "imagenet" else "imagenet"
        indices = DATA[family]["geometry_indices"]
        modes = {}
        for mode in MODES:
            x = preprocess(source["test_x"], source["cal_x"], mode)[indices]
            y = preprocess(source["test_y"], source["cal_y"], mode)[indices]
            geometry = geometry_statistics(x, y, permutations=CONTEXT["permutations"])
            reliability = {}
            if family == "imagenet":
                for key, value in (("x", x), ("y", y)):
                    alternate = preprocess(
                        source[f"repeat_{key}"], source[f"cal_{key}"], mode
                    )[indices]
                    reliability[key] = geometry_statistics(
                        value, alternate, permutations=0
                    )
            else:
                ya = preprocess(source["repeat_y_a"], source["cal_y"], mode)[indices]
                yb = preprocess(source["repeat_y_b"], source["cal_y"], mode)[indices]
                reliability["y"] = geometry_statistics(ya, yb, permutations=0)
            mappings = {
                str(dim): alignment_experiment(
                    source, bundles[other], mode, dim, bootstrap=layer == "final_norm"
                )
                for dim in ("full", *PCA_DIMS)
            }
            modes[mode] = {
                "geometry": geometry,
                "within_modality_reference": reliability,
                "mappings": mappings,
            }
        row["families"][family] = modes
    row["seconds"] = time.monotonic() - started
    write_json(path, row)
    return str(path)


def summarize(root):
    rows = [
        json.loads(p.read_text())
        for p in sorted((root / "analysis").glob("*/*/layer-*.json"))
    ]
    flattened, candidates, curves = [], defaultdict(list), defaultdict(list)
    for row in rows:
        identity = {k: row[k] for k in ("state", "profile", "layer", "readout")}
        for family, modes in row["families"].items():
            for mode, values in modes.items():
                g = values["geometry"]
                base = {
                    **identity,
                    "family": family,
                    "mode": mode,
                    "geometry_valid": g["valid"],
                    **g.get("scores", {}),
                }
                if g["valid"]:
                    curves[
                        (row["state"], row["profile"], row["readout"], family, mode)
                    ].append((row["layer"], g))
                for dimension, mapping in values["mappings"].items():
                    item = {
                        **base,
                        "dimension": dimension,
                        "mapping_valid": mapping["valid"],
                    }
                    if mapping["valid"]:
                        for split in ("fit", "dev", "test", "transfer_test"):
                            item[f"{split}_r2"] = mapping[split]["paired"]["r2"]
                            item[f"{split}_nrmse"] = mapping[split]["paired"]["nrmse"]
                            item[f"{split}_shuffled_r2"] = mapping[split][
                                "shuffled_fit"
                            ]["r2"]
                        item["test_retained_x"] = mapping["test"]["variance_retained_x"]
                        item["test_retained_y"] = mapping["test"]["variance_retained_y"]
                        candidates[
                            (row["state"], row["profile"], row["readout"], family, mode)
                        ].append(item)
                    flattened.append(item)
    if flattened:
        columns = list(dict.fromkeys(key for item in flattened for key in item))
        with (root / "geometry-v3.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(flattened)
    selected = []
    for key, options in sorted(candidates.items()):
        selected.append(
            min(
                options,
                key=lambda r: (
                    r["dev_nrmse"],
                    LAYERS.index(r["layer"]),
                    1024 if r["dimension"] == "full" else int(r["dimension"]),
                ),
            )
        )
    maxima = []
    for key, curve in sorted(curves.items()):
        for metric in curve[0][1]["scores"]:
            valid = [(layer, g) for layer, g in curve if g["null_samples"].get(metric)]
            if not valid:
                continue
            obs = max(g["scores"][metric] for _, g in valid)
            null = (
                torch.tensor([g["null_samples"][metric] for _, g in valid])
                .max(0)
                .values
            )
            maxima.append(
                {
                    "state": key[0],
                    "profile": key[1],
                    "readout": key[2],
                    "family": key[3],
                    "mode": key[4],
                    "metric": metric,
                    "observed_max": obs,
                    "null_max_mean": float(null.mean()),
                    "null_max_q95": float(null.quantile(0.95)),
                    "max_over_layers_tail_p": (1 + int(null.ge(obs).sum()))
                    / (1 + len(null)),
                    "valid_layers": len(valid),
                }
            )
    write_json(root / "dev-selected-geometry-v3.json", selected)
    write_json(root / "layer-search-null-v3.json", maxima)
    write_json(
        root / "results-geometry-v3.json",
        {"schema": "unified_geometry_v3_results_1", "rows": rows},
    )
    return rows


def audit(root):
    expected = {
        (state, profile, layer, readout)
        for state, profile in PAIRS
        for layer in LAYERS
        for readout in READOUTS
    }
    paths = list((root / "analysis").glob("*/*/layer-*.json"))
    actual, invalid_geometry, invalid_mapping = set(), [], []
    geometry_count, mapping_count = 0, 0
    for path in paths:
        row = json.loads(path.read_text())
        key = tuple(row[k] for k in ("state", "profile", "layer", "readout"))
        assert key not in actual, f"Duplicate layer record: {key}"
        actual.add(key)
        assert set(row["families"]) == {"imagenet", "coco"}
        for family, modes in row["families"].items():
            assert set(modes) == set(MODES)
            for mode, values in modes.items():
                geometry_count += 1
                g = values["geometry"]
                assert g["points"] == (200 if family == "imagenet" else 500)
                if not g["valid"]:
                    invalid_geometry.append([*key, family, mode, g["reason"]])
                else:
                    assert all(math.isfinite(v) for v in g["scores"].values())
                    assert set(g["scores"]) == set(g["null_samples"])
                    for metric, null in g["null_samples"].items():
                        assert len(null) == 199 and all(math.isfinite(v) for v in null)
                        assert 0.005 <= g["null_summary"][metric]["tail_p"] <= 1
                    for k in (5, 10, 20):
                        assert 0 <= g["scores"][f"knn_{k}"] <= 1
                assert set(values["mappings"]) == {"full", "32", "128"}
                for dimension, mapping in values["mappings"].items():
                    mapping_count += 1
                    if not mapping["valid"]:
                        invalid_mapping.append([*key, family, mode, dimension])
                        continue
                    assert mapping["fit_points"] == (
                        600 if family == "imagenet" else 500
                    )
                    assert mapping["dimension"] == (
                        1024 if dimension == "full" else int(dimension)
                    )
                    if dimension == "full":
                        assert not mapping["full_rotation_identified"]
                    else:
                        assert mapping["full_rotation_identified"]
                    for split in ("fit", "dev", "test", "transfer_test"):
                        record = mapping[split]
                        expected_points = (
                            {"fit": 600, "dev": 200, "test": 200, "transfer_test": 1000}
                            if family == "imagenet"
                            else {
                                "fit": 500,
                                "dev": 500,
                                "test": 1000,
                                "transfer_test": 200,
                            }
                        )
                        assert record["points"] == expected_points[split]
                        for label in ("paired", "shuffled_fit"):
                            metrics = record[label]
                            assert all(math.isfinite(v) for v in metrics.values())
                            assert (
                                abs(metrics["r2"] - (1 - metrics["nrmse"] ** 2)) < 1e-8
                            )
                        for axis in ("x", "y"):
                            assert (
                                -1e-8 <= record[f"variance_retained_{axis}"] <= 1 + 1e-8
                            )
                        if key[2] == "final_norm" and split in (
                            "test",
                            "transfer_test",
                        ):
                            assert "bootstrap" in record
    assert actual == expected, (
        f"Missing {len(expected - actual)}, unexpected {len(actual - expected)} records"
    )
    result = {
        "status": "passed",
        "complete_layer_records": len(actual),
        "geometry_records": geometry_count,
        "mapping_records": mapping_count,
        "invalid_geometry_records": invalid_geometry,
        "invalid_mapping_records": invalid_mapping,
        "checks": [
            "exact state/profile/layer/readout coverage without duplicates",
            "dataset sizes and fixed dimensions",
            "finite scores, 199 permutations, valid probabilities",
            "full-dimensional underidentification reported",
            "PCA numerical identifiability and retained variance bounds",
            "NRMSE/R2 identity including negative R2",
            "final_norm conditional bootstrap present",
        ],
    }
    write_json(root / "audit-geometry-v3.json", result)
    return result


def neighbor_examples(root, source):
    """Export ALL 200 class neighborhoods, not only favorable examples."""
    bundle = build_views(source, "final_ema", "native")["imagenet"]
    samples = json.loads((source / "samples.json").read_text())
    labels = {
        row["group"]: row["text"].removeprefix("a photo of a ").removesuffix(".")
        for row in samples["imagenet_texts"]
        if row["template_index"] == 0
    }
    groups = bundle["test_groups"]
    records = []
    for layer in ("13", "final_norm"):
        for pool, readout in enumerate(READOUTS):
            values = layer_bundle(bundle, LAYERS.index(layer), pool)
            for mode in MODES:
                nearest = {}
                for key in ("x", "y"):
                    features = preprocess(
                        values[f"test_{key}"], values[f"cal_{key}"], mode
                    )
                    distances = torch.cdist(features, features)
                    distances.fill_diagonal_(float("inf"))
                    nearest[key] = distances.argsort(dim=1, stable=True)[
                        :, :10
                    ].tolist()
                for i, group in enumerate(groups):
                    shared = sorted(set(nearest["x"][i]) & set(nearest["y"][i]))
                    records.append(
                        {
                            "layer": layer,
                            "readout": readout,
                            "mode": mode,
                            "class_id": group,
                            "class_name": labels[group],
                            "image_neighbors": [
                                labels[groups[j]] for j in nearest["x"][i]
                            ],
                            "text_neighbors": [
                                labels[groups[j]] for j in nearest["y"][i]
                            ],
                            "shared_neighbor_ids": [groups[j] for j in shared],
                            "shared_neighbor_names": [
                                labels[groups[j]] for j in shared
                            ],
                        }
                    )
    write_json(
        root / "imagenet-neighbor-examples-v3.json",
        {
            "status": "qualitative illustration; all 200 test classes exported",
            "selection": "fixed final_norm and native query dev-selected layer 13; no layer selection from neighborhood examples",
            "candidate_pool": 200,
            "records": records,
        },
    )
    return len(records)


def main():
    global DATA, CONTEXT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--state", default="final_ema")
    parser.add_argument("--profiles", default="bare,native,neutral")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--layers", default=",".join(LAYERS))
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--neighbor-examples-only", action="store_true")
    parser.add_argument("--skip-summary", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    protocol = prepare(args.output_dir, args.source_dir)
    if args.prepare_only:
        emit("geometry_protocol_frozen", output_dir=str(args.output_dir))
        return
    if args.audit_only:
        result = audit(args.output_dir)
        emit(
            "geometry_audit_complete",
            status=result["status"],
            rows=result["complete_layer_records"],
        )
        return
    if args.neighbor_examples_only:
        count = neighbor_examples(args.output_dir, args.source_dir)
        emit("geometry_neighbor_examples_complete", rows=count)
        return
    if not args.summarize_only:
        for profile in args.profiles.split(","):
            if [args.state, profile] not in json.loads(json.dumps(PAIRS)):
                raise ValueError("State/profile is outside the frozen protocol")
            DATA = build_views(args.source_dir, args.state, profile)
            CONTEXT = {
                "out": str(args.output_dir / "analysis" / args.state / profile),
                "state": args.state,
                "profile": profile,
                "threads": args.threads,
                "permutations": protocol["geometry_permutations"],
            }
            tasks = [
                (LAYERS.index(layer), readout)
                for layer in args.layers.split(",")
                for readout in READOUTS
            ]
            emit(
                "geometry_analysis_start",
                state=args.state,
                profile=profile,
                tasks=len(tasks),
            )
            with mp.get_context("fork").Pool(args.workers) as workers:
                for index, _ in enumerate(
                    workers.imap_unordered(analyze_one, tasks), 1
                ):
                    if index % 5 == 0 or index == len(tasks):
                        emit(
                            "geometry_analysis_progress",
                            state=args.state,
                            profile=profile,
                            completed=index,
                            total=len(tasks),
                        )
    if not args.skip_summary:
        rows = summarize(args.output_dir)
        emit("geometry_analysis_complete", rows=len(rows))


if __name__ == "__main__":
    main()
