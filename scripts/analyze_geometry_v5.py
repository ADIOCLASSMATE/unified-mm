"""Full common-protocol V5 geometry, held-out mappings and disjoint ARO tests."""

import argparse
import csv
import json
import multiprocessing as mp
import time
from collections import defaultdict
from pathlib import Path

import torch

from scripts.analyze_unified_geometry_v3 import apply_frame, error_bootstrap, preprocess
from scripts.analyze_unified_geometry_v4 import geometric_controls, pair_scores
from scripts.freeze_geometry_v5_analysis import freeze
from scripts.geometry_v4_math import describe_fit, evaluate_pair, spectral_fit
from scripts.geometry_v5_math import fit_pair
from scripts.geometry_v5_protocol import (
    SETTINGS,
    canonical_mode,
    layer_values,
    load_pair_views,
)
from scripts.prepare_geometry_v5_assets import RUN, emit, write_json

DATA, CONTEXT = {}, {}
MODES = ("centered_euclidean", "unit_sphere")
DIMS = (32, 128, 512, "full")
SEED = 20260909


def aro_controls(fit, source, mode, aro, tasks, bootstrap):
    mode = canonical_mode(mode)
    x = preprocess(aro["test_x"], source["cal_x"], mode)
    yp = preprocess(aro["test_y_positive"], source["cal_y"], mode)
    yn = preprocess(aro["test_y_negative"], source["cal_y"], mode)
    scores = {}
    if x.shape[-1] == yp.shape[-1]:
        scores["original_coordinates"] = pair_scores(x, yp, yn)
    xf, _ = apply_frame(fit["fx"], x)
    yp, _ = apply_frame(fit["fy"], yp)
    yn, _ = apply_frame(fit["fy"], yn)
    prediction = xf @ fit["q"]
    scores["paired_fit"] = pair_scores(prediction, yp, yn)
    scores["shuffled_fit"] = pair_scores(xf @ fit["null_q"], yp, yn)
    order = torch.arange(len(x))
    gen = torch.Generator().manual_seed(SEED)
    for task in sorted(set(tasks)):
        ix = torch.tensor([i for i, value in enumerate(tasks) if value == task])
        order[ix] = ix[torch.randperm(len(ix), generator=gen)]
    scores["shuffled_image"] = pair_scores(prediction[order], yp, yn)
    summary = {}
    for task in sorted(set(tasks)):
        ix = torch.tensor([i for i, value in enumerate(tasks) if value == task])
        row = {
            label: {metric: float(v[ix].mean()) for metric, v in values.items()}
            for label, values in scores.items()
        }
        row["points"] = len(ix)
        if bootstrap:
            draws = torch.randint(
                len(ix), (2000, len(ix)), generator=torch.Generator().manual_seed(SEED)
            )
            intervals = {}
            for metric in ("cosine", "distance"):
                paired = scores["paired_fit"][metric][ix]
                null = scores["shuffled_image"][metric][ix]
                quantiles = torch.tensor([0.025, 0.975], dtype=paired.dtype)
                intervals[metric] = {
                    "accuracy_95": paired[draws].mean(1).quantile(quantiles).tolist(),
                    "advantage_over_shuffled_image_95": (paired - null)[draws]
                    .mean(1)
                    .quantile(quantiles)
                    .tolist(),
                }
            row["bootstrap"] = intervals
        summary[task] = row
    return summary, scores


def mapping_controls(source, target, aro, family, mode, final, identity):
    canonical = canonical_mode(mode)
    xfit = preprocess(source["fit_x"], source["cal_x"], canonical)
    yfit = preprocess(source["fit_y"], source["cal_y"], canonical)
    result, arrays = {}, {}
    for n in (600,) if family == "imagenet" else (512, 2048, 8192):
        spectra = spectral_fit(xfit[:n]), spectral_fit(yfit[:n])
        for dimension in DIMS:
            key = f"fit{n}-dim{dimension}"
            fit = fit_pair(xfit[:n], yfit[:n], dimension, SEED, spectra=spectra)
            if fit is None:
                result[key] = {
                    "valid": False,
                    "fit_points": n,
                    "requested_dimension": dimension,
                    "reason": "unequal original widths cannot define full rotation"
                    if dimension == "full" and xfit.shape[-1] != yfit.shape[-1]
                    else "constant or source-rank-deficient subspace",
                }
                continue
            record = {
                "valid": True,
                "fit_points": n,
                "requested_dimension": dimension,
                "original_width_x": xfit.shape[-1],
                "original_width_y": yfit.shape[-1],
                **describe_fit(fit),
                **fit["diagnostics"],
                "rotation_identified": fit["rotation_identified"],
            }
            vectors = {}
            for split in ("fit", "dev", "test", "transfer_test"):
                values = target if split == "transfer_test" else source
                prefix = "test" if split == "transfer_test" else split
                x = preprocess(values[f"{prefix}_x"], source["cal_x"], canonical)
                y = preprocess(values[f"{prefix}_y"], source["cal_y"], canonical)
                if split == "fit":
                    x, y = x[:n], y[:n]
                summary, errors, null, baseline = evaluate_pair(fit, x, y)
                if final and split in {"test", "transfer_test"}:
                    summary["bootstrap"] = error_bootstrap(errors, null, baseline)
                    vectors[split] = {
                        "errors": errors,
                        "shuffled_errors": null,
                        "baseline": baseline,
                    }
                record[split] = summary
            if family == "imagenet":
                y = preprocess(source["test_y"], source["cal_y"], canonical)
                record["prototype_test"] = {
                    str(size): evaluate_pair(
                        fit,
                        preprocess(
                            source[f"test_prototype{size}_x"],
                            source["cal_x"],
                            canonical,
                        ),
                        y,
                    )[0]
                    for size in (1, 3, 5, 10, 15)
                }
                record["wording_shift_test"] = evaluate_pair(
                    fit,
                    preprocess(source["test_x"], source["cal_x"], canonical),
                    preprocess(source["repeat_y"], source["cal_y"], canonical),
                )[0]
            elif n == 8192:
                x = preprocess(source["test_x"], source["cal_x"], canonical)
                record["caption_test"] = {
                    str(count): evaluate_pair(
                        fit,
                        x,
                        preprocess(
                            source[f"test_caption{count}_y"], source["cal_y"], canonical
                        ),
                    )[0]
                    for count in (1, 3, 5)
                }
                record["aro"], aro_vectors = aro_controls(
                    fit, source, mode, aro, DATA["aro"]["tasks"], final
                )
                if final:
                    vectors["aro"] = aro_vectors
            if final:
                map_path = (
                    Path(CONTEXT["root"])
                    / "fitted-maps"
                    / CONTEXT["setting"]
                    / f"{identity}-{family}-{mode}-{key}.pt"
                )
                map_path.parent.mkdir(parents=True, exist_ok=True)
                bundle = {
                    k: (
                        {kk: vv for kk, vv in v.items() if kk != "fit"}
                        if k in {"fx", "fy"}
                        else v
                    )
                    for k, v in fit.items()
                }
                bundle.update(
                    {
                        "cal_x": source["cal_x"],
                        "cal_y": source["cal_y"],
                        "mode": mode,
                        "source_family": family,
                        "target_adaptation": False,
                        "fit_points": n,
                    }
                )
                temporary = map_path.with_suffix(".tmp")
                torch.save(bundle, temporary)
                temporary.replace(map_path)
                record["frozen_map"] = str(map_path)
                arrays[key] = vectors
            result[key] = record
    return result, arrays


def analyze_one(task):
    index, readout_name = task
    torch.set_num_threads(CONTEXT["threads"])
    setting = CONTEXT["layout"]
    pair, readout = setting["layer_pairs"][index], setting["readouts"][readout_name]
    identity = f"pair-{index:02d}-{readout_name}"
    destination = (
        Path(CONTEXT["root"]) / "analysis" / CONTEXT["setting"] / f"{identity}.json"
    )
    if destination.exists():
        old = json.loads(destination.read_text())
        assert (
            old["schema"] == "geometry_v5_layer_result_1"
            and old["pair"] == pair
            and old["readout_specification"] == readout
        )
        return str(destination)
    started = time.monotonic()
    bundles = {
        family: layer_values(DATA[family], family, pair, readout) for family in DATA
    }
    final = pair["kind"] in {"final_norm", "native_endpoint"}
    row = {
        "schema": "geometry_v5_layer_result_1",
        "setting": CONTEXT["setting"],
        "pair_index": index,
        "pair": pair,
        "readout": readout_name,
        "readout_specification": readout,
        "families": {},
    }
    vectors = {}
    for family in ("imagenet", "coco"):
        source, target = (
            bundles[family],
            bundles["coco" if family == "imagenet" else "imagenet"],
        )
        row["families"][family], vectors[family] = {}, {}
        for mode in MODES:
            geometry = geometric_controls(source, family, canonical_mode(mode))
            mapping, saved = mapping_controls(
                source, target, bundles["aro"], family, mode, final, identity
            )
            row["families"][family][mode] = {"geometry": geometry, "mappings": mapping}
            vectors[family][mode] = saved
    if final:
        path = (
            Path(CONTEXT["root"])
            / "sample-statistics"
            / CONTEXT["setting"]
            / f"{identity}.pt"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        torch.save(
            {
                "schema": "geometry_v5_sample_statistics_1",
                "identity": identity,
                "setting": CONTEXT["setting"],
                "aro_tasks": DATA["aro"]["tasks"],
                "aro_groups": DATA["aro"]["groups"],
                "statistics": vectors,
            },
            temporary,
        )
        temporary.replace(path)
        row["sample_statistics"] = str(path)
    row["seconds"] = time.monotonic() - started
    write_json(destination, row)
    emit(
        "v5_layer_analysis_complete",
        setting=CONTEXT["setting"],
        pair_index=index,
        readout=readout_name,
        seconds=row["seconds"],
    )
    return str(destination)


def summarize(root):
    rows = [
        json.loads(path.read_text())
        for path in sorted((root / "analysis").glob("*/*.json"))
    ]
    flat, choices, curves = [], defaultdict(list), defaultdict(list)
    for row in rows:
        base = {
            "setting": row["setting"],
            "pair_index": row["pair_index"],
            "readout": row["readout"],
            **row["pair"],
        }
        for family, modes in row["families"].items():
            for mode, values in modes.items():
                geo = values["geometry"]["primary"]
                curve_key = (row["setting"], row["readout"], family, mode)
                if geo["valid"]:
                    curves[curve_key].append(geo)
                for key, mapping in values["mappings"].items():
                    record = {
                        **base,
                        "family": family,
                        "mode": mode,
                        "mapping": key,
                        "fit_points": mapping["fit_points"],
                        "requested_dimension": mapping["requested_dimension"],
                        "valid": mapping["valid"],
                        **geo.get("scores", {}),
                    }
                    if mapping["valid"]:
                        for field in (
                            "dimension",
                            "rotation_identified",
                            "rank_x",
                            "rank_y",
                            "cross_covariance_rank",
                            "cross_covariance_condition",
                        ):
                            record[field] = mapping[field]
                        for split in ("fit", "dev", "test", "transfer_test"):
                            record[f"{split}_r2"] = mapping[split]["paired"]["r2"]
                            record[f"{split}_shuffled_r2"] = mapping[split][
                                "shuffled_fit"
                            ]["r2"]
                        record["retained_x"], record["retained_y"] = (
                            mapping["test"]["variance_retained_x"],
                            mapping["test"]["variance_retained_y"],
                        )
                        if mapping["requested_dimension"] != "full":
                            choices[(*curve_key, mapping["fit_points"])].append(record)
                    flat.append(record)
    if flat:
        with (root / "geometry-v5.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(dict.fromkeys(key for row in flat for key in row)),
            )
            writer.writeheader()
            writer.writerows(flat)
    selected = [
        max(values, key=lambda r: (r["dev_r2"], -r["pair_index"], -r["dimension"]))
        for values in choices.values()
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
                    "setting": key[0],
                    "readout": key[1],
                    "family": key[2],
                    "mode": key[3],
                    "metric": metric,
                    "observed_max": observed,
                    "null_max_q95": float(null.quantile(0.95)),
                    "p": (1 + int(null.ge(observed).sum())) / (1 + len(null)),
                    "valid_layer_pairs": len(curve),
                }
            )
    write_json(root / "dev-selected-geometry-v5.json", selected)
    write_json(root / "layer-search-null-v5.json", maxima)
    write_json(
        root / "analysis-index-v5.json",
        {
            "rows": len(rows),
            "settings": sorted({row["setting"] for row in rows}),
            "completeness": "coverage must be checked against comparison-contract; this index alone is not a completion claim",
        },
    )


def main():
    global DATA, CONTEXT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=RUN)
    parser.add_argument("--settings", default=",".join(SETTINGS))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--pair-indices", default="all")
    parser.add_argument("--readouts", default="all")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Partitioned CPU work: defer shared summary writes until all partitions finish",
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    contract = freeze(args.output_dir)
    if args.summarize_only:
        summarize(args.output_dir)
        return
    for setting in args.settings.split(","):
        layout = contract["settings"][setting]
        DATA = {
            family: load_pair_views(args.output_dir, setting, family)
            for family in ("imagenet", "coco", "aro")
        }
        for data in DATA.values():
            for key in ("layers_x", "layers_y", "pools_x", "pools_y"):
                assert data[key] == layout[key]
        assert len(DATA["aro"]["groups"]) == 868
        CONTEXT = {
            "root": str(args.output_dir),
            "setting": setting,
            "layout": layout,
            "threads": args.threads,
        }
        indices = (
            range(len(layout["layer_pairs"]))
            if args.pair_indices == "all"
            else [int(v) for v in args.pair_indices.split(",")]
        )
        readouts = (
            layout["readouts"] if args.readouts == "all" else args.readouts.split(",")
        )
        tasks = [(index, readout) for index in indices for readout in readouts]
        with mp.get_context("fork").Pool(args.workers) as pool:
            list(pool.imap_unordered(analyze_one, tasks))
        emit("v5_setting_analysis_finished", setting=setting, rows=len(tasks))
    if not args.no_summary:
        summarize(args.output_dir)


if __name__ == "__main__":
    main()
