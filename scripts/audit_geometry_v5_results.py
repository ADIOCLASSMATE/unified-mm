"""Verify complete V5 curves, frozen maps, endpoint identities and sensitivity coverage.

This is a result audit, not a full-goal completion claim (report/plots/resource
cleanup require separate evidence). Invalid/negative outcomes are retained.
"""

import argparse
import json
import math
from pathlib import Path

import torch

from scripts.analyze_geometry_v5 import DIMS, MODES
from scripts.bootstrap_geometry_v5 import endpoint_plan
from scripts.geometry_v5_protocol import SETTINGS
from scripts.prepare_geometry_v5_assets import RUN, emit, write_json

METRICS = {
    "linear_cka",
    "distance_pearson",
    "rsa_spearman",
    "knn_5",
    "knn_10",
    "knn_20",
}
TASKS = {"aro_vg_attribution": 451, "aro_vg_relation": 417}


def read(path):
    return json.loads(Path(path).read_text())


def finite_tree(value):
    if isinstance(value, float):
        assert math.isfinite(value)
    elif isinstance(value, dict):
        for child in value.values():
            finite_tree(child)
    elif isinstance(value, list):
        for child in value:
            finite_tree(child)


def interval(value):
    assert len(value) == 2 and value[0] <= value[1]
    assert all(math.isfinite(x) for x in value)


def geometry(value, points, permutations):
    assert value["points"] == points
    if not value["valid"]:
        assert "reason" in value
        return
    assert set(value["scores"]) == METRICS
    if permutations:
        assert set(value["null_samples"]) == METRICS
        for metric in METRICS:
            null = value["null_samples"][metric]
            assert len(null) == 199
            p = (1 + sum(x >= value["scores"][metric] for x in null)) / 200
            assert abs(p - value["null_summary"][metric]["tail_p"]) < 1e-12


def mapping(value, family, n, dimension, final):
    assert value["fit_points"] == n and value["requested_dimension"] == dimension
    if not value["valid"]:
        assert value["reason"]
        return False
    d = value["dimension"]
    assert d == dimension or dimension == "full"
    if dimension == "full":
        assert d == value["original_width_x"] == value["original_width_y"]
    assert 0 <= value["cross_covariance_rank"] <= d
    assert len(value["cross_covariance_singular_values"]) == d
    # Match the pre-existing numerical bound in geometry_v5_math.rotation.
    assert value["orthogonality_max_abs"] < 1e-8
    if value["rotation_identified"]:
        assert (
            min(value["rank_x"], value["rank_y"], value["cross_covariance_rank"]) >= d
        )
        assert n > d
    counts = {
        "fit": n,
        "dev": 200 if family == "imagenet" else 1024,
        "test": 200 if family == "imagenet" else 2048,
        "transfer_test": 2048 if family == "imagenet" else 200,
    }
    for split, count in counts.items():
        row = value[split]
        assert row["points"] == count
        for control in ("paired", "shuffled_fit"):
            assert abs(row[control]["r2"] - (1 - row[control]["nrmse"] ** 2)) < 1e-9
        if final and split in {"test", "transfer_test"}:
            for ci in row["bootstrap"].values():
                interval(ci)
    if family == "imagenet":
        assert set(value["prototype_test"]) == {"1", "3", "5", "10", "15"}
        assert value["wording_shift_test"]["points"] == 200
    elif n == 8192:
        assert set(value["caption_test"]) == {"1", "3", "5"}
        assert set(value["aro"]) == set(TASKS)
        for task, count in TASKS.items():
            row = value["aro"][task]
            assert row["points"] == count
            for control in ("paired_fit", "shuffled_fit", "shuffled_image"):
                assert set(row[control]) == {"cosine", "distance"}
                assert all(0 <= v <= 1 for v in row[control].values())
            if final:
                for metric in ("cosine", "distance"):
                    for ci in row["bootstrap"][metric].values():
                        interval(ci)
    return True


def expected_groups(samples, family, robust=False):
    rows = samples[f"{family}_images"]
    if family == "imagenet":
        return sorted(
            {
                r["group"]
                for r in rows
                if r["mapping_split"] == "test" and (not robust or r["robust"])
            }
        )
    return [
        r["group"] for r in rows if r["split"] == "test" and (not robust or r["robust"])
    ]


def audit_setting(root, setting, layout, samples):
    paths = sorted((root / "analysis" / setting).glob("*.json"))
    expected = {
        f"pair-{i:02d}-{readout}.json"
        for i in range(len(layout["layer_pairs"]))
        for readout in layout["readouts"]
    }
    assert {p.name for p in paths} == expected
    rows, valid, invalid = [], 0, 0
    for path in paths:
        row = read(path)
        finite_tree(row)
        assert (
            row["schema"] == "geometry_v5_layer_result_1" and row["setting"] == setting
        )
        assert row["pair"] == layout["layer_pairs"][row["pair_index"]]
        assert row["readout_specification"] == layout["readouts"][row["readout"]]
        final = row["pair"]["kind"] in {"final_norm", "native_endpoint"}
        assert set(row["families"]) == {"imagenet", "coco"}
        for family in ("imagenet", "coco"):
            assert set(row["families"][family]) == set(MODES)
            for mode in MODES:
                bundle = row["families"][family][mode]
                geometry(
                    bundle["geometry"]["primary"],
                    200 if family == "imagenet" else 512,
                    True,
                )
                if family == "imagenet":
                    assert set(bundle["geometry"]["prototype_sizes"]) == {
                        "1",
                        "3",
                        "5",
                        "10",
                        "15",
                    }
                else:
                    assert set(bundle["geometry"]["caption_counts"]) == {"1", "3", "5"}
                sizes = (600,) if family == "imagenet" else (512, 2048, 8192)
                assert set(bundle["mappings"]) == {
                    f"fit{n}-dim{d}" for n in sizes for d in DIMS
                }
                for n in sizes:
                    for dimension in DIMS:
                        good = mapping(
                            bundle["mappings"][f"fit{n}-dim{dimension}"],
                            family,
                            n,
                            dimension,
                            final,
                        )
                        valid += good
                        invalid += not good
        rows.append(row)
    plans = endpoint_plan(rows, layout)
    expected_endpoints = {
        f"{p['endpoint']}-{p['readout']}-{p['family']}-{p['mode']}-{p['mapping']}.json": p
        for p in plans
    }
    endpoint_paths = sorted((root / "endpoints" / setting).glob("*.json"))
    assert {p.name for p in endpoint_paths} == set(expected_endpoints)
    coverage = read(root / "audits" / f"endpoints-{setting}.json")
    assert coverage["expected"] == len(plans) and set(coverage["paths"]) == {
        str(p) for p in endpoint_paths
    }
    endpoints, maps_checked = [], set()
    for path in endpoint_paths:
        row, plan = read(path), expected_endpoints[path.name]
        finite_tree(row)
        assert row["setting"] == setting and row["endpoint"] == plan["endpoint"]
        if plan["row"] is None:
            assert not row["valid"]
        else:
            assert row["pair_index"] == plan["row"]["pair_index"]
            source = plan["row"]["families"][row["family"]][row["mode"]]["mappings"][
                row["mapping"]
            ]
            assert row["valid"] == source["valid"]
        endpoints.append(row)
        if not row["valid"]:
            continue
        fit = torch.load(row["frozen_map"], map_location="cpu", weights_only=True)
        assert fit["mode"] == row["mode"] and fit["source_family"] == row["family"]
        assert (
            fit["fit_points"] == row["fit_points"] and fit["target_adaptation"] is False
        )
        d = row["dimension"]
        for key in ("q", "null_q"):
            assert fit[key].shape == (d, d) and bool(torch.isfinite(fit[key]).all())
        assert (
            fit["diagnostics"]["orthogonality_max_abs"] == row["orthogonality_max_abs"]
        )
        maps_checked.add(row["frozen_map"])
        arrays = torch.load(
            row["sample_statistics"], map_location="cpu", weights_only=True
        )["arrays"]
        for split in ("test", "transfer_test"):
            arr = arrays[split]
            assert arr["groups"].tolist() == expected_groups(
                samples, arr["target_family"]
            )
            assert len(arr["errors"]) == len(arr["baseline"]) == len(arr["groups"])
            observed = float(
                1
                - arr["errors"].double().sum()
                / arr["baseline"].double().sum().clamp_min(1e-20)
            )
            assert abs(observed - row[split]["paired"]["r2"]) < 1e-10
            assert abs(observed - source[split]["paired"]["r2"]) < 1e-10
            for ci in row[split]["bootstrap"].values():
                interval(ci)
        if row["family"] == "coco" and row["fit_points"] == 8192:
            arr = arrays["aro"]
            assert arr["groups"].tolist() == [r["group"] for r in samples["aro_images"]]
            assert arr["tasks"] == [r["task"] for r in samples["aro_images"]]
            for task, n in TASKS.items():
                ix = torch.tensor([i for i, t in enumerate(arr["tasks"]) if t == task])
                assert len(ix) == n
                for control in ("paired_fit", "shuffled_fit", "shuffled_image"):
                    for metric in ("cosine", "distance"):
                        mean = float(arr["scores"][control][metric][ix].double().mean())
                        assert abs(mean - row["aro"][task][control][metric]) < 1e-10
    robust_count = 0
    if setting in {"b_native", "f_native", "janusflow_generation", "showo2_generation"}:
        wanted = [
            r
            for r in endpoints
            if r["valid"]
            and r["fit_points"] == (600 if r["family"] == "imagenet" else 8192)
        ]
        robust = read(root / "audits" / f"robustness-{setting}.json")
        assert (
            robust["expected_endpoints"] == len(wanted)
            and robust["rows_per_endpoint"] == 6
        )
        expected_names = {
            f"{r['endpoint']}-{r['readout']}-{r['family']}-{r['mode']}-{r['mapping']}.json"
            for r in wanted
        }
        robust_paths = sorted((root / "robustness" / setting).glob("*.json"))
        assert {p.name for p in robust_paths} == expected_names
        assert set(robust["paths"]) == {str(p) for p in robust_paths}
        profiles = (
            {"native_sigma1", "native_sigma2", "native_mean"}
            if setting[0] in "bf"
            else {"seed1", "seed2", "image_midpoint"}
        )
        for path in robust_paths:
            result = read(path)
            finite_tree(result)
            assert (
                not result["target_recalibration"]
                and not result["fit_or_choice_from_perturbed_test"]
            )
            assert {(r["target_family"], r["profile"]) for r in result["rows"]} == {
                (f, p) for f in ("imagenet", "coco") for p in profiles
            }
            for row in result["rows"]:
                assert row["groups"] == expected_groups(
                    samples, row["target_family"], robust=True
                )
                assert row["points"] == (
                    32 if row["target_family"] == "imagenet" else 512
                )
                interval(row["error_reduction_95"])
                if row["profile"] == "image_midpoint":
                    assert (
                        row["text_reused_unmodified"]
                        and row["within_modal_change"]["y"]["relative_rms"] == 0
                    )
                robust_count += 1
    result = {
        "schema": "geometry_v5_setting_result_audit_1",
        "status": "passed",
        "setting": setting,
        "layer_readout_rows": len(rows),
        "valid_mappings": valid,
        "invalid_mappings_explicit": invalid,
        "endpoints": len(endpoints),
        "frozen_maps_checked": len(maps_checked),
        "robustness_rows": robust_count,
        "scope": "all declared settings/readouts/dimensions/splits, endpoint identity/selection/score reconstruction, strict ARO and robustness coverage; not a score threshold",
    }
    write_json(root / "audits" / f"results-{setting}.json", result)
    emit(
        "v5_result_setting_audit_passed",
        **{k: v for k, v in result.items() if k != "scope"},
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=RUN)
    parser.add_argument("--settings", default=",".join(SETTINGS))
    args = parser.parse_args()
    torch.set_num_threads(1)
    contract, samples = (
        read(args.output_dir / "comparison-contract.json"),
        read(args.output_dir / "samples.json"),
    )
    for setting in args.settings.split(","):
        audit_setting(args.output_dir, setting, contract["settings"][setting], samples)
