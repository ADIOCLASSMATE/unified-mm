"""Frozen-source-map sigma, posterior, fixed-noise and midpoint sensitivity."""

import argparse
import json
import multiprocessing as mp
from pathlib import Path

import torch
import torch.nn.functional as F

from scripts.analyze_geometry_v4_robustness import (
    paired_change_interval,
    perturbation_view,
)
from scripts.analyze_unified_geometry_v3 import geometry_statistics, preprocess
from scripts.bootstrap_geometry_v5 import atomic_torch_save, test_groups
from scripts.geometry_v4_math import evaluate_pair
from scripts.geometry_v5_protocol import canonical_mode, layer_values, load_pair_views
from scripts.prepare_cross_model_geometry_v5 import V4
from scripts.prepare_geometry_v5_assets import RUN, emit, write_json
from scripts.prepare_geometry_v5_views import grouped

DATA, PERTURBED, CONTEXT = {}, {}, {}
SETTINGS = ("b_native", "f_native", "janusflow_generation", "showo2_generation")


def profiles_for(setting):
    return (
        ("native_sigma1", "native_sigma2", "native_mean")
        if setting in {"b_native", "f_native"}
        else ("seed1", "seed2", "image_midpoint")
    )


def load_flow_subset(root, model, profile, dataset, all_rows):
    selected = [i for i, row in enumerate(all_rows) if row["robust"]]
    lookup = {original: i for i, original in enumerate(selected)}
    contract = json.loads((root / "adapter-contracts" / f"{model}.json").read_text())
    result, coverage = None, []
    for rank in range(16):
        path = (
            root
            / "features"
            / model
            / "generation"
            / profile
            / f"{dataset}-rank-{rank:02d}-of-16.pt"
        )
        value = torch.load(path, weights_only=True, map_location="cpu", mmap=True)
        indices = value["indices"].tolist()
        assert indices == selected[rank::16] and value["contract"] == contract
        assert (
            value["profile"] == profile
            and value["path"] == "generation"
            and value["dataset"] == dataset
        )
        assert (
            value["training_updates"] == 0
            and value["audit"]["opposite_modality_replacement_exact"]
        )
        features = value["features"]
        assert bool(torch.isfinite(features).all())
        if result is None:
            result = torch.empty(
                len(selected), *features.shape[1:], dtype=features.dtype
            )
        result[torch.tensor([lookup[i] for i in indices])] = features
        coverage.extend(indices)
    assert sorted(coverage) == selected
    return result, [all_rows[i] for i in selected]


def robust_view(root, setting, profile, family):
    if setting in {"b_native", "f_native"}:
        return perturbation_view(
            V4 if setting == "b_native" else root / "f-v4", profile, family
        )
    destination = root / "robust-views" / setting / profile / f"{family}.pt"
    if destination.exists():
        result = torch.load(
            destination, weights_only=True, map_location="cpu", mmap=True
        )
        assert result["schema"] == "geometry_v5_flow_robust_views_1"
        assert (
            result["setting"] == setting
            and result["profile"] == profile
            and result["family"] == family
        )
        return result
    samples = json.loads((root / "samples.json").read_text())
    image_rows = samples[f"{family}_images"]
    groups = list(
        dict.fromkeys(
            row["group"]
            for row in image_rows
            if row["robust"]
            and (
                row["mapping_split"] == "test"
                if family == "imagenet"
                else row["split"] == "test"
            )
        )
    )
    assert len(groups) == (32 if family == "imagenet" else 512)
    result = {
        "schema": "geometry_v5_flow_robust_views_1",
        "setting": setting,
        "profile": profile,
        "family": family,
        "groups": torch.tensor(groups),
    }
    for axis, modality in (("x", "images"), ("y", "texts")):
        if profile == "image_midpoint" and axis == "y":
            continue  # Explicitly use the unchanged main text condition below.
        features, rows = load_flow_subset(
            root,
            setting.split("_")[0],
            profile,
            f"{family}_{modality}",
            samples[f"{family}_{modality}"],
        )
        result[f"test_{axis}"] = grouped(
            features,
            rows,
            groups,
            lambda row: row["split"] == "a" if family == "imagenet" else True,
        )
    atomic_torch_save(result, destination)
    return result


def analyze_endpoint(endpoint):
    torch.set_num_threads(1)
    root, setting = Path(CONTEXT["root"]), CONTEXT["setting"]
    name = Path(endpoint["sample_statistics"]).stem
    destination = root / "robustness" / setting / f"{name}.json"
    if destination.exists():
        old = json.loads(destination.read_text())
        assert old["schema"] == "geometry_v5_robustness_endpoint_1"
        return str(destination)
    fit = torch.load(endpoint["frozen_map"], weights_only=True, map_location="cpu")
    assert (
        not fit["target_adaptation"]
        and fit["source_family"] == endpoint["family"]
        and fit["mode"] == endpoint["mode"]
    )
    pair, readout = endpoint["pair"], endpoint["readout_specification"]
    mode = canonical_mode(endpoint["mode"])
    rows = []
    for family in ("imagenet", "coco"):
        native = layer_values(DATA[family], family, pair, readout)
        lookup = {
            group: i for i, group in enumerate(test_groups(DATA[family]).tolist())
        }
        for profile in profiles_for(setting):
            change = PERTURBED[profile, family]
            indices = torch.tensor(
                [lookup[group] for group in change["groups"].tolist()]
            )
            before, after = {}, {}
            for axis in ("x", "y"):
                original = native[f"test_{axis}"][indices]
                perturbed = change.get(f"test_{axis}")
                altered = (
                    original
                    if perturbed is None
                    else perturbed[
                        :, pair[f"index_{axis}"], readout[f"pool_{axis}"]
                    ].double()
                )
                before[axis] = preprocess(original, fit[f"cal_{axis}"], mode)
                after[axis] = preprocess(altered, fit[f"cal_{axis}"], mode)
            baseline, be, _, denominator = evaluate_pair(fit, before["x"], before["y"])
            perturbed, pe, _, _ = evaluate_pair(fit, after["x"], after["y"])
            row = {
                "target_family": family,
                "profile": profile,
                "points": len(indices),
                "groups": change["groups"].tolist(),
                "original": baseline,
                "perturbed": perturbed,
                "error_reduction_over_original_denominator": float(
                    (be - pe).sum() / denominator.sum().clamp_min(1e-20)
                ),
                "error_reduction_95": paired_change_interval(be, pe, denominator),
                "geometry_original": geometry_statistics(
                    before["x"], before["y"], permutations=0
                ),
                "geometry_perturbed": geometry_statistics(
                    after["x"], after["y"], permutations=0
                ),
                "within_modal_change": {
                    axis: {
                        "mean_cosine": float(
                            F.cosine_similarity(before[axis], after[axis]).mean()
                        ),
                        "relative_rms": float(
                            (
                                (after[axis] - before[axis]).square().sum()
                                / before[axis].square().sum().clamp_min(1e-20)
                            ).sqrt()
                        ),
                    }
                    for axis in ("x", "y")
                },
                "text_reused_unmodified": profile == "image_midpoint",
            }
            rows.append(row)
    result = {
        "schema": "geometry_v5_robustness_endpoint_1",
        "setting": setting,
        "endpoint": {
            key: endpoint[key]
            for key in (
                "endpoint",
                "readout",
                "pair_index",
                "pair",
                "family",
                "mode",
                "mapping",
                "fit_points",
                "dimension",
                "rotation_identified",
                "frozen_map",
            )
        },
        "target_recalibration": False,
        "fit_or_choice_from_perturbed_test": False,
        "rows": rows,
        "interpretation": "B/F sigma intervention also moves the first native image query spatial slot; it is not pure content order. Flow seed changes concern fixed semantic-independent initial noise; image_midpoint changes only image condition, with the unmodified main text representation.",
    }
    write_json(destination, result)
    emit(
        "v5_robustness_endpoint_complete",
        setting=setting,
        endpoint=name,
        rows=len(rows),
    )
    return str(destination)


def freeze_robustness(root):
    contract = {
        "schema": "geometry_v5_robustness_contract_1",
        "settings": list(SETTINGS),
        "endpoints": "all declared readouts, fixed final norm at all valid dimensions plus source-dev-selected common-budget endpoint",
        "source_fit_points": {"imagenet": 600, "coco": 8192},
        "targets": {"imagenet_test_classes": 32, "coco_test_scenes": 512},
        "subset": "exact V4 prospectively designated robust semantic identities; no score filtering",
        "profiles": {setting: list(profiles_for(setting)) for setting in SETTINGS},
        "parameters": "source main calibration means, PCA, fit means, global RMS and Q all frozen; no target recentering",
        "paired_error_change": "same native-target denominator; positive means perturbation reduced error, negative means degradation",
        "selection": "no robustness test result used to select settings, seeds, time or readout",
    }
    path = root / "robustness-contract.json"
    if path.exists():
        assert json.loads(path.read_text()) == contract
    else:
        write_json(path, contract)


def main():
    global DATA, PERTURBED, CONTEXT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=RUN)
    parser.add_argument("--settings", default=",".join(SETTINGS))
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--freeze-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    freeze_robustness(args.output_dir)
    if args.freeze_only:
        return
    for setting in args.settings.split(","):
        audit = json.loads(
            (args.output_dir / "audits" / f"endpoints-{setting}.json").read_text()
        )
        all_endpoints = [json.loads(Path(path).read_text()) for path in audit["paths"]]
        endpoints = [
            row
            for row in all_endpoints
            if row["valid"]
            and row["fit_points"] == (600 if row["family"] == "imagenet" else 8192)
        ]
        DATA = {
            family: load_pair_views(args.output_dir, setting, family)
            for family in ("imagenet", "coco")
        }
        PERTURBED = {
            (profile, family): robust_view(args.output_dir, setting, profile, family)
            for profile in profiles_for(setting)
            for family in DATA
        }
        CONTEXT = {"root": str(args.output_dir), "setting": setting}
        with mp.get_context("fork").Pool(args.workers) as pool:
            results = list(pool.imap_unordered(analyze_endpoint, endpoints))
        write_json(
            args.output_dir / "audits" / f"robustness-{setting}.json",
            {
                "schema": "geometry_v5_robustness_coverage_1",
                "setting": setting,
                "expected_endpoints": len(endpoints),
                "paths": sorted(results),
                "rows_per_endpoint": 6,
            },
        )
        emit("v5_robustness_setting_complete", setting=setting, endpoints=len(results))


if __name__ == "__main__":
    main()
