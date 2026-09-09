"""Expanded frozen-model geometry analysis: large fit sets and semantic controls."""

import argparse
import csv
import json
import multiprocessing as mp
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_unified_geometry_v3 import (
    apply_frame,
    error_bootstrap,
    geometry_statistics,
    preprocess,
)
from scripts.geometry_v4_math import describe_fit, evaluate_pair, fit_pair, spectral_fit
from utils.research.representation_protocol import emit, write_json
from scripts.probe_unified_semantics_v2 import LAYERS

SEED = 20260909
READOUTS = {"content_mean": 0, "content_last_sigma": 1, "query_native": 2}
MODES = ("centered_euclidean", "centered_unit_sphere")
DIMS = ("full", 32, 128, 512)
PROTOTYPES = (1, 3, 5, 10, 15)
DATA, CONTEXT = {}, {}


def freeze_analysis(root):
    protocol = {
        "schema": "geometry_v4_analysis_1",
        "seed": SEED,
        "layers": LAYERS,
        "readouts": READOUTS,
        "modes": MODES,
        "dimensions": DIMS,
        "coco_fit_sizes": [512, 2048, 8192],
        "imagenet_fit_classes": 600,
        "geometry_points": {"imagenet": 200, "coco": 512},
        "coco_geometry_subset": "first 512 test scenes in the prospectively shuffled manifest; matches robustness subset",
        "geometry_permutations": 199,
        "geometry_permutation_seed": 20260908,
        "bootstrap_final_norm": 2000,
        "bootstrap_seed": 20260908,
        "aro_bootstrap_seed": SEED,
        "imagenet_prototype_sizes": PROTOTYPES,
        "prototype_map_control": "fit at 15 images/class; evaluate frozen map at 1/3/5/10/15 test images, plus independently measured geometry/reliability",
        "caption_map_control": "fit on all captions; evaluate frozen map using first 1/3/5 test captions",
        "wording_control": "fit on a templates; test b templates without any recalibration",
        "aro": "frozen COCO maps at fit8192, and original centered-coordinate reference; separate cosine and negative-distance pair rankings, shuffled-fit and within-task shuffled-image controls",
        "selection": "dev only, jointly layer/dimension; separate sample-size curves and fixed final_norm",
        "repeated_fits": {
            "layers": ["13", "23", "final_norm"],
            "profiles": ["native"],
            "readouts": ["content_mean", "query_native"],
            "dimensions": ["full", 32],
            "seeds": [20260910, 20260911],
            "scheme": "512/2048 new without-replacement fit subsets; 8192 paired bootstrap-with-replacement; test fixed",
        },
        "arithmetic": "FP64 Gram eigensystem shared across dimensions; no per-axis whitening; singular-value relative rank threshold 1e-6",
        "multiplicity": "same geometry permutations across a fixed layer curve; max-over-layer calibration, not experiment-wide familywise control",
    }
    path = root / "analysis-protocol.json"
    canonical = json.loads(json.dumps(protocol))
    if path.exists():
        assert json.loads(path.read_text()) == canonical, "Analysis protocol changed"
    else:
        write_json(path, protocol)


def layer_values(family, layer, pool):
    data = DATA[family]
    result = {}
    for key, value in data.items():
        if torch.is_tensor(value) and value.ndim in (3, 4):
            result[key] = (
                value[:, layer, pool] if value.ndim == 4 else value[layer, pool]
            ).double()
    if family == "imagenet":
        for split in ("fit", "dev", "test"):
            ix = data[f"{split}_indices"]
            result[f"{split}_x"] = result["a_15_x"][ix]
            result[f"{split}_y"] = result["a_y"][ix]
        ix = data["test_indices"]
        result["repeat_x"] = result["b_15_x"][ix]
        result["repeat_y"] = result["b_y"][ix]
        for n in PROTOTYPES:
            result[f"test_prototype{n}_x"] = result[f"a_{n}_x"][ix]
            result[f"repeat_prototype{n}_x"] = result[f"b_{n}_x"][ix]
    return result


def geometric_controls(source, family, mode):
    n = 200 if family == "imagenet" else 512
    x = preprocess(source["test_x"], source["cal_x"], mode)[:n]
    y = preprocess(source["test_y"], source["cal_y"], mode)[:n]
    output = {"primary": geometry_statistics(x, y), "repeated_views": {}}
    if family == "imagenet":
        for axis, values in (("x", x), ("y", y)):
            repeat = preprocess(source[f"repeat_{axis}"], source[f"cal_{axis}"], mode)
            output["repeated_views"][axis] = geometry_statistics(
                values, repeat, permutations=0
            )
        output["prototype_sizes"] = {}
        for size in PROTOTYPES:
            a = preprocess(source[f"test_prototype{size}_x"], source["cal_x"], mode)
            b = preprocess(source[f"repeat_prototype{size}_x"], source["cal_x"], mode)
            output["prototype_sizes"][str(size)] = {
                "crossmodal": geometry_statistics(a, y, permutations=0),
                "image_repeat": geometry_statistics(a, b, permutations=0),
            }
    else:
        a = preprocess(source["test_first2_y"], source["cal_y"], mode)[:n]
        b = preprocess(source["test_remaining_y"], source["cal_y"], mode)[:n]
        output["repeated_views"]["y"] = geometry_statistics(a, b, permutations=0)
        output["caption_counts"] = {
            str(count): geometry_statistics(
                x,
                preprocess(source[f"test_caption{count}_y"], source["cal_y"], mode)[:n],
                permutations=0,
            )
            for count in (1, 3, 5)
        }
    return output


def pair_scores(x, positive, negative):
    cosine = F.cosine_similarity(x, positive) - F.cosine_similarity(x, negative)
    distance = (x - negative).square().sum(1) - (x - positive).square().sum(1)
    return {
        "cosine": (cosine.gt(0).double() + 0.5 * cosine.eq(0)),
        "distance": (distance.gt(0).double() + 0.5 * distance.eq(0)),
    }


def aro_controls(fit, source, mode, aro, bootstrap):
    x = preprocess(aro["test_x"], source["cal_x"], mode)
    yp = preprocess(aro["test_y_positive"], source["cal_y"], mode)
    yn = preprocess(aro["test_y_negative"], source["cal_y"], mode)
    scores = {"original_coordinates": pair_scores(x, yp, yn)}
    xf, _ = apply_frame(fit["fx"], x)
    yp, _ = apply_frame(fit["fy"], yp)
    yn, _ = apply_frame(fit["fy"], yn)
    predicted = xf @ fit["q"]
    scores["paired_fit"] = pair_scores(predicted, yp, yn)
    scores["shuffled_fit"] = pair_scores(xf @ fit["null_q"], yp, yn)
    tasks = DATA["aro"]["tasks"]
    order = torch.arange(len(x))
    gen = torch.Generator().manual_seed(SEED)
    for task in sorted(set(tasks)):
        indices = torch.tensor([i for i, value in enumerate(tasks) if value == task])
        order[indices] = indices[torch.randperm(len(indices), generator=gen)]
    scores["shuffled_image"] = pair_scores(predicted[order], yp, yn)
    output = {}
    for task in sorted(set(tasks)):
        select = torch.tensor([i for i, value in enumerate(tasks) if value == task])
        record = {
            label: {metric: float(v[select].mean()) for metric, v in metrics.items()}
            for label, metrics in scores.items()
        }
        if bootstrap:
            draws = torch.randint(
                len(select),
                (2000, len(select)),
                generator=torch.Generator().manual_seed(SEED),
            )
            intervals = {}
            for metric in ("cosine", "distance"):
                p = scores["paired_fit"][metric][select]
                null = scores["shuffled_image"][metric][select]
                intervals[metric] = {
                    "accuracy_95": p[draws]
                    .mean(1)
                    .quantile(torch.tensor([0.025, 0.975], dtype=p.dtype))
                    .tolist(),
                    "advantage_over_shuffled_image_95": (p - null)[draws]
                    .mean(1)
                    .quantile(torch.tensor([0.025, 0.975], dtype=p.dtype))
                    .tolist(),
                }
            record["bootstrap"] = intervals
        output[task] = record
    return output


def mapping_controls(source, target, aro, family, mode, layer, readout):
    xfit = preprocess(source["fit_x"], source["cal_x"], mode)
    yfit = preprocess(source["fit_y"], source["cal_y"], mode)
    sizes = (512, 2048, 8192) if family == "coco" else (600,)
    output = {}
    for n in sizes:
        spectra = spectral_fit(xfit[:n]), spectral_fit(yfit[:n])
        for dimension in DIMS:
            fit = fit_pair(xfit[:n], yfit[:n], dimension, SEED, spectra=spectra)
            key = f"fit{n}-dim{dimension}"
            if fit is None:
                output[key] = {
                    "valid": False,
                    "fit_points": n,
                    "requested_dimension": dimension,
                    "reason": "constant or rank-deficient subspace",
                }
                continue
            record = {"valid": True, "fit_points": n, **describe_fit(fit)}
            for split in ("fit", "dev", "test", "transfer_test"):
                values = target if split == "transfer_test" else source
                prefix = "test" if split == "transfer_test" else split
                x = preprocess(values[f"{prefix}_x"], source["cal_x"], mode)
                y = preprocess(values[f"{prefix}_y"], source["cal_y"], mode)
                if split == "fit":
                    x, y = x[:n], y[:n]
                result, errors, null_errors, baseline = evaluate_pair(fit, x, y)
                if layer == "final_norm" and split in ("test", "transfer_test"):
                    result["bootstrap"] = error_bootstrap(errors, null_errors, baseline)
                record[split] = result
            if family == "imagenet":
                y = preprocess(source["test_y"], source["cal_y"], mode)
                record["prototype_test"] = {}
                for size in PROTOTYPES:
                    x = preprocess(
                        source[f"test_prototype{size}_x"], source["cal_x"], mode
                    )
                    record["prototype_test"][str(size)] = evaluate_pair(fit, x, y)[0]
                x = preprocess(source["test_x"], source["cal_x"], mode)
                yb = preprocess(source["repeat_y"], source["cal_y"], mode)
                record["wording_shift_test"] = evaluate_pair(fit, x, yb)[0]
            elif n == 8192:
                x = preprocess(source["test_x"], source["cal_x"], mode)
                record["caption_test"] = {}
                for count in (1, 3, 5):
                    y = preprocess(
                        source[f"test_caption{count}_y"], source["cal_y"], mode
                    )
                    record["caption_test"][str(count)] = evaluate_pair(fit, x, y)[0]
                record["aro"] = aro_controls(
                    fit, source, mode, aro, bootstrap=layer == "final_norm"
                )
                # Save frozen frames for perturbation tests; never fit on perturbed test data.
                if CONTEXT["profile"] == "native" and layer in (
                    "13",
                    "23",
                    "final_norm",
                ):
                    path = (
                        Path(CONTEXT["root"])
                        / "fitted-maps"
                        / CONTEXT["state"]
                        / f"layer-{layer}-{readout}-{mode}-dim{dimension}.pt"
                    )
                    path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        {
                            k: (
                                {kk: vv for kk, vv in v.items() if kk != "fit"}
                                if k in ("fx", "fy")
                                else v
                            )
                            for k, v in fit.items()
                        },
                        path,
                    )
            if (
                family == "coco"
                and CONTEXT["profile"] == "native"
                and layer in ("13", "23", "final_norm")
                and readout in ("content_mean", "query_native")
                and dimension in ("full", 32)
            ):
                repeats = []
                for seed in (SEED + 1, SEED + 2):
                    gen = torch.Generator().manual_seed(seed)
                    ix = (
                        torch.randint(len(xfit), (n,), generator=gen)
                        if n == 8192
                        else torch.randperm(len(xfit), generator=gen)[:n]
                    )
                    repeated = fit_pair(xfit[ix], yfit[ix], dimension, seed)
                    x = preprocess(source["test_x"], source["cal_x"], mode)
                    y = preprocess(source["test_y"], source["cal_y"], mode)
                    repeats.append(
                        {
                            "seed": seed,
                            "test": evaluate_pair(repeated, x, y)[0],
                            **describe_fit(repeated),
                        }
                    )
                record["repeat_fits"] = repeats
            output[key] = record
    return output


def analyze_one(task):
    layer, readout = task
    torch.set_num_threads(CONTEXT["threads"])
    path = (
        Path(CONTEXT["root"])
        / "analysis"
        / CONTEXT["state"]
        / CONTEXT["profile"]
        / f"layer-{layer}-{readout}.json"
    )
    if path.exists():
        return str(path)
    started = time.monotonic()
    layer_index, pool = LAYERS.index(layer), READOUTS[readout]
    bundles = {
        family: layer_values(family, layer_index, pool)
        for family in ("imagenet", "coco", "aro")
    }
    row = {
        "state": CONTEXT["state"],
        "profile": CONTEXT["profile"],
        "layer": layer,
        "readout": readout,
        "families": {},
    }
    for family in ("imagenet", "coco"):
        source = bundles[family]
        target = bundles["coco" if family == "imagenet" else "imagenet"]
        row["families"][family] = {}
        for mode in MODES:
            row["families"][family][mode] = {
                "geometry": geometric_controls(source, family, mode),
                "mappings": mapping_controls(
                    source, target, bundles["aro"], family, mode, layer, readout
                ),
            }
    row["seconds"] = time.monotonic() - started
    write_json(path, row)
    return str(path)


def summarize(root):
    rows = [
        json.loads(p.read_text())
        for p in sorted((root / "analysis").glob("*/*/layer-*.json"))
    ]
    flat, selected, curves = [], defaultdict(list), defaultdict(list)
    for row in rows:
        identity = {k: row[k] for k in ("state", "profile", "layer", "readout")}
        for family, modes in row["families"].items():
            for mode, values in modes.items():
                g = values["geometry"]["primary"]
                curve_key = (row["state"], row["profile"], row["readout"], family, mode)
                if g["valid"]:
                    curves[curve_key].append(g)
                for name, mapping in values["mappings"].items():
                    record = {
                        **identity,
                        "family": family,
                        "mode": mode,
                        "mapping": name,
                        "fit_points": mapping["fit_points"],
                        "valid": mapping["valid"],
                        **g.get("scores", {}),
                    }
                    if mapping["valid"]:
                        record.update(
                            {
                                k: mapping[k]
                                for k in (
                                    "dimension",
                                    "rotation_identified",
                                    "rank_x",
                                    "rank_y",
                                )
                            }
                        )
                        for split in ("fit", "dev", "test", "transfer_test"):
                            record[f"{split}_r2"] = mapping[split]["paired"]["r2"]
                            record[f"{split}_shuffled_r2"] = mapping[split][
                                "shuffled_fit"
                            ]["r2"]
                        record["retained_x"] = mapping["test"]["variance_retained_x"]
                        record["retained_y"] = mapping["test"]["variance_retained_y"]
                        selected[(*curve_key, mapping["fit_points"])].append(record)
                    flat.append(record)
    if flat:
        columns = list(dict.fromkeys(k for r in flat for k in r))
        with (root / "geometry-v4.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(flat)
    selection = [
        max(
            options,
            key=lambda r: (r["dev_r2"], -LAYERS.index(r["layer"]), -r["dimension"]),
        )
        for options in selected.values()
    ]
    maxima = []
    for key, curve in curves.items():
        for metric in curve[0]["scores"]:
            null = (
                torch.tensor([g["null_samples"][metric] for g in curve]).max(0).values
            )
            observed = max(g["scores"][metric] for g in curve)
            maxima.append(
                {
                    "state": key[0],
                    "profile": key[1],
                    "readout": key[2],
                    "family": key[3],
                    "mode": key[4],
                    "metric": metric,
                    "observed_max": observed,
                    "null_max_q95": float(null.quantile(0.95)),
                    "p": (1 + int(null.ge(observed).sum())) / (1 + len(null)),
                    "valid_layers": len(curve),
                }
            )
    write_json(root / "results-geometry-v4.json", {"rows": rows})
    write_json(root / "dev-selected-geometry-v4.json", selection)
    write_json(root / "layer-search-null-v4.json", maxima)
    return len(rows)


def main():
    global DATA, CONTEXT
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--state", default="final_ema")
    p.add_argument("--profiles", default="native,bare,neutral")
    p.add_argument("--layers", default=",".join(LAYERS))
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--summarize-only", action="store_true")
    p.add_argument("--skip-summary", action="store_true")
    a = p.parse_args()
    torch.set_num_threads(1)
    freeze_analysis(a.output_dir)
    if a.prepare_only:
        emit("analysis_protocol_frozen_v4")
        return
    if not a.summarize_only:
        for profile in a.profiles.split(","):
            DATA = {
                family: torch.load(
                    a.output_dir / "views" / a.state / profile / f"{family}.pt",
                    weights_only=True,
                    mmap=True,
                    map_location="cpu",
                )
                for family in ("imagenet", "coco", "aro")
            }
            assert all(
                v["state"] == a.state and v["profile"] == profile for v in DATA.values()
            )
            CONTEXT = {
                "root": str(a.output_dir),
                "state": a.state,
                "profile": profile,
                "threads": a.threads,
            }
            tasks = [
                (layer, readout)
                for layer in a.layers.split(",")
                for readout in READOUTS
            ]
            emit(
                "analysis_started_v4", state=a.state, profile=profile, tasks=len(tasks)
            )
            with mp.get_context("fork").Pool(a.workers) as pool:
                for i, _ in enumerate(pool.imap_unordered(analyze_one, tasks), 1):
                    if i % 5 == 0 or i == len(tasks):
                        emit(
                            "analysis_progress_v4",
                            state=a.state,
                            profile=profile,
                            completed=i,
                            total=len(tasks),
                        )
    if not a.skip_summary:
        emit("analysis_summarized_v4", rows=summarize(a.output_dir))


if __name__ == "__main__":
    main()
