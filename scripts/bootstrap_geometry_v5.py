"""Conditional endpoint uncertainty and source-dev-only selection for V5.

No test, transfer or ARO statistic participates in endpoint selection. Bootstrap
units are semantic identities, not individual captions. Fits are frozen during
the bootstrap; training-seed and fit/dev-sample uncertainty are not estimated.
"""

import argparse
import json
import multiprocessing as mp
from pathlib import Path

import torch

from scripts.analyze_geometry_v5 import DIMS, MODES, aro_controls
from scripts.analyze_unified_geometry_v3 import error_bootstrap, preprocess
from scripts.geometry_v4_math import describe_fit, evaluate_pair
from scripts.geometry_v5_math import fit_pair
from scripts.geometry_v5_protocol import (
    SETTINGS,
    canonical_mode,
    layer_values,
    load_pair_views,
)
from utils.research.geometry_v5_assets import RUN, emit, write_json

DATA, CONTEXT = {}, {}
SEED = 20260908


def atomic_torch_save(value, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def freeze_statistics(root):
    contract = {
        "schema": "geometry_v5_statistics_contract_1",
        "bootstrap_repeats": 2000,
        "bootstrap_seed": SEED,
        "endpoint_selection": "source-dev R2, then shallower pair, then smaller dimension; common 32/128/512 only; separately per readout, mode and fit size",
        "fixed_endpoints": "final_norm or trained native endpoint; all declared dimensions, including explicitly invalid ones",
        "conditional_uncertainty": "test identities resampled; no refit, dev reselection, model retraining, or target adaptation",
        "knn_uncertainty": "resample query identities with frozen original candidate graph; not a bootstrap of a newly sampled candidate pool",
        "rsa_cka_inference": "199 identity permutations and max-over-layer null in main analysis, not independent distance-pair bootstrap",
        "paired_model_differences": "same semantic-identity draw for both models; each R2 retains its own representation/subspace denominator",
        "primary_contrast": "B native vs each baseline, common content_mean, final_norm, centered_euclidean, 32 dimensions, ImageNet600 and COCO8192 fits; remaining contrasts descriptive sensitivity analyses",
        "multiplicity": "95% pointwise intervals, not experiment-wide simultaneous intervals; no test-based winner selection",
        "aro": "868 confirmed COCO-ID-disjoint original images, bootstrap separately within task; no ARO fitting or selection",
    }
    path = root / "statistics-contract.json"
    if path.exists():
        assert json.loads(path.read_text()) == contract
    else:
        write_json(path, contract)
    return contract


def test_groups(data):
    return data["groups"][data["test_indices"]]


def endpoint_plan(rows, layout):
    actual = {(row["pair_index"], row["readout"]) for row in rows}
    expected = {
        (index, readout)
        for index in range(len(layout["layer_pairs"]))
        for readout in layout["readouts"]
    }
    assert actual == expected and len(rows) == len(expected), (
        "Incomplete layer curve cannot select endpoints"
    )
    plans = []
    for readout in layout["readouts"]:
        curve = [row for row in rows if row["readout"] == readout]
        final = [
            row
            for row in curve
            if row["pair"]["kind"] in {"final_norm", "native_endpoint"}
        ]
        assert len(final) == 1
        for family in ("imagenet", "coco"):
            for mode in MODES:
                for n in (600,) if family == "imagenet" else (512, 2048, 8192):
                    base = {
                        "readout": readout,
                        "family": family,
                        "mode": mode,
                        "fit_points": n,
                    }
                    for dimension in DIMS:
                        key = f"fit{n}-dim{dimension}"
                        plans.append(
                            {
                                **base,
                                "endpoint": "fixed",
                                "row": final[0],
                                "mapping": key,
                            }
                        )
                    candidates = [
                        (row, name, mapping)
                        for row in curve
                        for name, mapping in row["families"][family][mode][
                            "mappings"
                        ].items()
                        if mapping["valid"]
                        and mapping["fit_points"] == n
                        and mapping["requested_dimension"] != "full"
                    ]
                    if candidates:
                        row, key, _ = max(
                            candidates,
                            key=lambda v: (
                                v[2]["dev"]["paired"]["r2"],
                                -v[0]["pair_index"],
                                -v[2]["dimension"],
                            ),
                        )
                        plans.append(
                            {
                                **base,
                                "endpoint": "dev_selected",
                                "row": row,
                                "mapping": key,
                            }
                        )
                    else:
                        plans.append(
                            {
                                **base,
                                "endpoint": "dev_selected",
                                "row": None,
                                "mapping": "unavailable",
                            }
                        )
    return plans


def knn_anchors(x, y, ks=(5, 10, 20)):
    n = len(x)
    if min(float((x - x.mean(0)).norm()), float((y - y.mean(0)).norm())) <= 1e-10:
        return {}
    distances = [torch.cdist(v.double(), v.double()) for v in (x, y)]
    for distance in distances:
        distance.fill_diagonal_(float("inf"))
    order = [distance.argsort(dim=1, stable=True) for distance in distances]
    result = {}
    for k in ks:
        assert n > k
        ax, ay = (
            torch.zeros(n, n, dtype=torch.bool),
            torch.zeros(n, n, dtype=torch.bool),
        )
        ax.scatter_(1, order[0][:, :k], True)
        ay.scatter_(1, order[1][:, :k], True)
        result[f"knn_{k}"] = (ax & ay).sum(1).double() / k
    return result


def mean_interval(values, repeats=2000):
    draws = torch.randint(
        len(values),
        (repeats, len(values)),
        generator=torch.Generator().manual_seed(SEED),
    )
    return (
        values[draws]
        .mean(1)
        .quantile(torch.tensor([0.025, 0.975], dtype=values.dtype))
        .tolist()
    )


def stripped_fit(fit, source, mode, family, n):
    result = {
        key: {k: v for k, v in value.items() if k != "fit"}
        if key in {"fx", "fy"}
        else value
        for key, value in fit.items()
    }
    result.update(
        {
            "cal_x": source["cal_x"],
            "cal_y": source["cal_y"],
            "mode": mode,
            "source_family": family,
            "target_adaptation": False,
            "fit_points": n,
        }
    )
    return result


def process_endpoint(plan):
    torch.set_num_threads(1)
    root, setting = Path(CONTEXT["root"]), CONTEXT["setting"]
    row, family, mode = plan["row"], plan["family"], plan["mode"]
    identity = f"{plan['endpoint']}-{plan['readout']}-{family}-{mode}-{plan['mapping']}"
    path = root / "endpoints" / setting / f"{identity}.json"
    if path.exists():
        old = json.loads(path.read_text())
        assert old["schema"] == "geometry_v5_endpoint_1"
        if old["valid"]:
            assert (
                Path(old["sample_statistics"]).exists()
                and Path(old["frozen_map"]).exists()
            )
        return str(path)
    result = {
        "schema": "geometry_v5_endpoint_1",
        "setting": setting,
        **{k: v for k, v in plan.items() if k != "row"},
    }
    if row is None:
        result.update(
            {
                "valid": False,
                "reason": "No valid source-dev candidate at any predeclared common dimension",
            }
        )
        write_json(path, result)
        return str(path)
    expected = row["families"][family][mode]["mappings"][plan["mapping"]]
    result.update(
        {
            "valid": expected["valid"],
            "pair_index": row["pair_index"],
            "pair": row["pair"],
            "readout_specification": row["readout_specification"],
            "requested_dimension": expected["requested_dimension"],
        }
    )
    if not expected["valid"]:
        result["reason"] = expected["reason"]
        write_json(path, result)
        return str(path)
    bundles = {
        f: layer_values(data, f, row["pair"], row["readout_specification"])
        for f, data in DATA.items()
    }
    source = bundles[family]
    canonical = canonical_mode(mode)
    if plan["endpoint"] == "fixed":
        map_path = Path(expected["frozen_map"])
        fit = torch.load(map_path, weights_only=True, map_location="cpu")
        assert (
            fit["source_family"] == family
            and fit["mode"] == mode
            and not fit["target_adaptation"]
        )
    else:
        n = plan["fit_points"]
        fit = fit_pair(
            preprocess(source["fit_x"][:n], source["cal_x"], canonical),
            preprocess(source["fit_y"][:n], source["cal_y"], canonical),
            expected["requested_dimension"],
        )
        assert fit is not None
        map_path = root / "selected-maps" / setting / f"{identity}.pt"
        atomic_torch_save(stripped_fit(fit, source, mode, family, n), map_path)
    result.update(
        {
            **describe_fit(fit),
            **fit["diagnostics"],
            "rotation_identified": fit["rotation_identified"],
            "dev_r2": expected["dev"]["paired"]["r2"],
            "frozen_map": str(map_path),
            "test_source": str(
                root
                / "analysis"
                / setting
                / f"pair-{row['pair_index']:02d}-{plan['readout']}.json"
            ),
        }
    )
    # Re-evaluate dev as well: selected-map reconstruction must match the frozen curve.
    check = evaluate_pair(
        fit,
        preprocess(source["dev_x"], source["cal_x"], canonical),
        preprocess(source["dev_y"], source["cal_y"], canonical),
    )[0]
    assert abs(check["paired"]["r2"] - result["dev_r2"]) < 1e-10
    arrays = {}
    for split, target_family in (
        ("test", family),
        ("transfer_test", "coco" if family == "imagenet" else "imagenet"),
    ):
        values = bundles[target_family]
        x = preprocess(values["test_x"], source["cal_x"], canonical)
        y = preprocess(values["test_y"], source["cal_y"], canonical)
        summary, errors, null, baseline = evaluate_pair(fit, x, y)
        assert abs(summary["paired"]["r2"] - expected[split]["paired"]["r2"]) < 1e-10
        summary["bootstrap"] = error_bootstrap(errors, null, baseline)
        result[split] = summary
        arrays[split] = {
            "groups": test_groups(DATA[target_family]),
            "target_family": target_family,
            "errors": errors,
            "shuffled_errors": null,
            "baseline": baseline,
        }
    n = 200 if family == "imagenet" else 512
    anchors = knn_anchors(
        preprocess(source["test_x"], source["cal_x"], canonical)[:n],
        preprocess(source["test_y"], source["cal_y"], canonical)[:n],
    )
    expected_geometry = row["families"][family][mode]["geometry"]["primary"]
    assert bool(anchors) == expected_geometry["valid"]
    result["geometry"] = {
        "valid": bool(anchors),
        "points": n,
        "scores": expected_geometry.get("scores", {}),
    }
    if anchors:
        for key, values in anchors.items():
            assert abs(float(values.mean()) - expected_geometry["scores"][key]) < 1e-12
        result["geometry"]["knn_query_bootstrap"] = {
            key: mean_interval(value) for key, value in anchors.items()
        }
        arrays["geometry"] = {
            "groups": test_groups(DATA[family])[:n],
            "anchors": anchors,
        }
    if family == "coco" and plan["fit_points"] == 8192:
        result["aro"], scores = aro_controls(
            fit, source, mode, bundles["aro"], DATA["aro"]["tasks"], True
        )
        arrays["aro"] = {
            "groups": DATA["aro"]["groups"],
            "tasks": DATA["aro"]["tasks"],
            "scores": scores,
        }
        for task, values in result["aro"].items():
            for control in ("paired_fit", "shuffled_fit", "shuffled_image"):
                for metric in ("cosine", "distance"):
                    assert (
                        abs(
                            values[control][metric]
                            - expected["aro"][task][control][metric]
                        )
                        < 1e-12
                    )
    statistics_path = root / "endpoint-statistics" / setting / f"{identity}.pt"
    atomic_torch_save(
        {
            "schema": "geometry_v5_endpoint_statistics_1",
            "setting": setting,
            "identity": identity,
            "arrays": arrays,
        },
        statistics_path,
    )
    result["sample_statistics"] = str(statistics_path)
    write_json(path, result)
    emit(
        "v5_endpoint_complete",
        setting=setting,
        endpoint=plan["endpoint"],
        readout=plan["readout"],
        pair=row["pair_index"],
        source=family,
        mode=mode,
        mapping=plan["mapping"],
    )
    return str(path)


def main():
    global DATA, CONTEXT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=RUN)
    parser.add_argument("--settings", default=",".join(SETTINGS))
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--freeze-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    freeze_statistics(args.output_dir)
    if args.freeze_only:
        return
    contract = json.loads((args.output_dir / "comparison-contract.json").read_text())
    for setting in args.settings.split(","):
        rows = [
            json.loads(path.read_text())
            for path in sorted((args.output_dir / "analysis" / setting).glob("*.json"))
        ]
        plans = endpoint_plan(rows, contract["settings"][setting])
        DATA = {
            family: load_pair_views(args.output_dir, setting, family)
            for family in ("imagenet", "coco", "aro")
        }
        CONTEXT = {"root": str(args.output_dir), "setting": setting}
        with mp.get_context("fork").Pool(args.workers) as pool:
            results = list(pool.imap_unordered(process_endpoint, plans))
        write_json(
            args.output_dir / "audits" / f"endpoints-{setting}.json",
            {
                "schema": "geometry_v5_endpoint_coverage_1",
                "setting": setting,
                "expected": len(plans),
                "paths": sorted(results),
                "conditional_on_frozen_fit_and_dev_choice": True,
            },
        )
        emit("v5_endpoint_setting_complete", setting=setting, endpoints=len(results))


if __name__ == "__main__":
    main()
