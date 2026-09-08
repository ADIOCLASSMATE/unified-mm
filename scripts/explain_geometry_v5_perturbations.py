"""Post-hoc error decomposition; no target refit, recentering, or revised scores.

Run after formal robustness results. Decompose their exact original squared-error
change into a mean-residual term and a centered-residual term. Target means here
describe the error algebra only; they are never used to improve a prediction.
"""

import argparse
import json
from pathlib import Path

import torch

from scripts.analyze_geometry_v5_robustness import profiles_for, robust_view
from scripts.bootstrap_geometry_v5 import test_groups
from scripts.geometry_v5_protocol import layer_values, load_pair_views
from scripts.prepare_geometry_v5_assets import RUN, emit, write_json


def read(path):
    return json.loads(Path(path).read_text())


def residual_decomposition(before, after):
    assert before.shape == after.shape and before.ndim == 2
    mb, ma = before.mean(0), after.mean(0)
    mean_change = len(before) * (ma.square().sum() - mb.square().sum())
    centered_change = (after - ma).square().sum() - (before - mb).square().sum()
    torch.testing.assert_close(
        mean_change + centered_change,
        after.square().sum() - before.square().sum(),
        atol=1e-8,
        rtol=1e-10,
    )
    return float(mean_change), float(centered_change)


def project(value, cal, frame):
    value = value.double() - cal - frame["mean"]
    if frame["projection"] is not None:
        value = value @ frame["projection"]
    return value / frame["radius"]


def main(root):
    torch.set_num_threads(1)
    results = []
    settings = ("b_native", "f_native", "janusflow_generation", "showo2_generation")
    for setting in settings:
        data = load_pair_views(root, setting, "coco")
        lookup = {g: i for i, g in enumerate(test_groups(data).tolist())}
        altered_views = {
            p: robust_view(root, setting, p, "coco") for p in profiles_for(setting)
        }
        endpoints = [read(p) for p in (root / "endpoints" / setting).glob("*.json")]
        endpoints = [
            e
            for e in endpoints
            if e["family"] == "coco"
            and e["fit_points"] == 8192
            and e["mode"] == "centered_euclidean"
            and e["readout"] in {"content_mean", "native_task"}
            and (e["endpoint"] == "dev_selected" or e["requested_dimension"] == 32)
        ]
        assert len(endpoints) == 4 and all(e["valid"] for e in endpoints)
        for endpoint in endpoints:
            evidence = (
                root
                / "robustness"
                / setting
                / (Path(endpoint["sample_statistics"]).stem + ".json")
            )
            expected = {
                r["profile"]: r
                for r in read(evidence)["rows"]
                if r["target_family"] == "coco"
            }
            fit = torch.load(
                endpoint["frozen_map"], map_location="cpu", weights_only=True
            )
            assert not fit["target_adaptation"] and fit["mode"] == "centered_euclidean"
            pair, readout = endpoint["pair"], endpoint["readout_specification"]
            main_values = layer_values(data, "coco", pair, readout)
            for profile, view in altered_views.items():
                assert view["groups"].tolist() == expected[profile]["groups"]
                indices = torch.tensor([lookup[g] for g in view["groups"].tolist()])
                before, after, raw = {}, {}, {}
                for axis in ("x", "y"):
                    original = main_values[f"test_{axis}"][indices]
                    altered = (
                        view[f"test_{axis}"][
                            :, pair[f"index_{axis}"], readout[f"pool_{axis}"]
                        ].double()
                        if f"test_{axis}" in view
                        else original
                    )
                    delta = altered - original
                    energy = float(delta.square().sum())
                    translation = delta.mean(0)
                    signal = (original - original.mean(0)).square().sum()
                    raw[axis] = {
                        "delta_energy": energy,
                        "translation_fraction_of_delta_energy": float(
                            len(delta) * translation.square().sum()
                        )
                        / energy
                        if energy
                        else None,
                        "centered_deformation_relative_rms": float(
                            (
                                (delta - translation).square().sum()
                                / signal.clamp_min(1e-20)
                            ).sqrt()
                        ),
                    }
                    before[axis] = project(
                        original, fit[f"cal_{axis}"], fit[f"f{axis}"]
                    )
                    after[axis] = project(altered, fit[f"cal_{axis}"], fit[f"f{axis}"])
                rb, ra = (
                    before["x"] @ fit["q"] - before["y"],
                    after["x"] @ fit["q"] - after["y"],
                )
                denominator = float(before["y"].square().sum())
                point = float(rb.square().sum() - ra.square().sum()) / denominator
                assert (
                    abs(
                        point
                        - expected[profile]["error_reduction_over_original_denominator"]
                    )
                    < 1e-7
                )
                mean, centered = residual_decomposition(rb, ra)
                results.append(
                    {
                        "setting": setting,
                        "endpoint": endpoint["endpoint"],
                        "readout": endpoint["readout"],
                        "pair": pair,
                        "dimension": endpoint["dimension"],
                        "profile": profile,
                        "points": len(indices),
                        "formal_robustness_evidence": str(evidence),
                        "error_reduction_recomputed": point,
                        "mean_residual_error_increase_over_native_denominator": mean
                        / denominator,
                        "centered_residual_error_increase_over_native_denominator": centered
                        / denominator,
                        "raw_change": raw,
                    }
                )
        emit("v5_perturbation_decomposition_setting", setting=setting)
    assert len(results) == 48
    write_json(
        root / "audits/perturbation-error-decomposition.json",
        {
            "status": "passed",
            "schema": "geometry_v5_posthoc_error_decomposition_1",
            "scope": "COCO512 robustness subset; COCO8192 Euclidean source fit; content mean and native task; fixed32 plus independently dev-selected endpoints for all four controlled settings",
            "posthoc_explanatory": True,
            "target_refit": False,
            "predictions_recentered": False,
            "formal_results_modified": False,
            "all_original_error_changes_independently_recomputed": True,
            "sign": "Positive decomposed term means more error; two terms sum to negative formal error reduction. This is error decomposition, not a target-adapted score.",
            "rows": results,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=RUN)
    main(parser.parse_args().output_dir)
