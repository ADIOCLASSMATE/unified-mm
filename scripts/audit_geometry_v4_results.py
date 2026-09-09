"""Strict completeness/structure audit for the V4 analysis, without score thresholds."""

import argparse
import itertools
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.research.representation_protocol import emit, write_json


def finite_tree(value, path="result"):
    if isinstance(value, dict):
        for key, item in value.items():
            finite_tree(item, f"{path}.{key}")
    elif isinstance(value, list):
        for i, item in enumerate(value):
            finite_tree(item, f"{path}[{i}]")
    elif isinstance(value, float):
        assert math.isfinite(value), path


def validate_mapping(mapping, expected_points):
    assert isinstance(mapping["valid"], bool)
    if not mapping["valid"]:
        return
    dim = mapping["dimension"]
    assert dim in (32, 128, 512, 1024)
    assert 0 < mapping["rank_x"] <= min(1024, mapping["fit_points"] - 1)
    assert 0 < mapping["rank_y"] <= min(1024, mapping["fit_points"] - 1)
    assert mapping["rotation_identified"] == (
        min(mapping["rank_x"], mapping["rank_y"]) >= dim
    )
    for axis in ("x", "y"):
        assert 0 < mapping[f"fit_variance_retained_{axis}"] < 1 + 1e-8
    for split, points in expected_points.items():
        score = mapping[split]
        assert score["points"] == points
        for label in ("paired", "shuffled_fit"):
            assert abs(score[label]["r2"] - (1 - score[label]["nrmse"] ** 2)) < 1e-9
        if "bootstrap" in score:
            for ci in score["bootstrap"].values():
                assert len(ci) == 2 and ci[0] <= ci[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    root = a.output_dir
    protocol = json.loads((root / "protocol.json").read_text())
    analysis_protocol = json.loads((root / "analysis-protocol.json").read_text())
    assert (
        json.loads((root / "feature-audit-v4.json").read_text())["status"] == "passed"
    )
    states_profiles = [
        (state, profile)
        for state, profiles in protocol["states_profiles"].items()
        for profile in profiles
        if protocol["profiles"][profile]["subset"] == "all"
    ]
    keys = {
        (*sp, layer, readout)
        for sp, layer, readout in itertools.product(
            states_profiles, protocol["layers"], protocol["pools"]
        )
    }
    rows = json.loads((root / "results-geometry-v4.json").read_text())["rows"]
    actual = {(r["state"], r["profile"], r["layer"], r["readout"]) for r in rows}
    assert actual == keys and len(rows) == len(keys) == 630
    mappings, invalid, repeats, aro, intervals = 0, 0, 0, 0, 0
    for row in rows:
        finite_tree(row)
        for family in ("imagenet", "coco"):
            for mode in analysis_protocol["modes"]:
                result = row["families"][family][mode]
                geometry = result["geometry"]
                g = geometry["primary"]
                assert g["points"] == (200 if family == "imagenet" else 512)
                if g["valid"]:
                    assert set(g["scores"]) == {
                        "linear_cka",
                        "distance_pearson",
                        "rsa_spearman",
                        "knn_5",
                        "knn_10",
                        "knn_20",
                    }
                    assert all(len(v) == 199 for v in g["null_samples"].values())
                if family == "imagenet":
                    assert set(geometry["prototype_sizes"]) == {
                        "1",
                        "3",
                        "5",
                        "10",
                        "15",
                    }
                    assert set(geometry["repeated_views"]) == {"x", "y"}
                    sizes = (600,)
                else:
                    assert set(geometry["caption_counts"]) == {"1", "3", "5"}
                    assert set(geometry["repeated_views"]) == {"y"}
                    sizes = (512, 2048, 8192)
                expected_maps = {
                    f"fit{n}-dim{dim}"
                    for n, dim in itertools.product(sizes, ("full", 32, 128, 512))
                }
                assert set(result["mappings"]) == expected_maps
                for mapping in result["mappings"].values():
                    mappings += 1
                    n = mapping["fit_points"]
                    points = {
                        "fit": n,
                        "dev": 200 if family == "imagenet" else 1024,
                        "test": 200 if family == "imagenet" else 2048,
                        "transfer_test": 2048 if family == "imagenet" else 200,
                    }
                    validate_mapping(mapping, points)
                    if not mapping["valid"]:
                        invalid += 1
                        continue
                    if row["layer"] == "final_norm":
                        assert all(
                            "bootstrap" in mapping[s] for s in ("test", "transfer_test")
                        )
                        intervals += 2
                    if family == "imagenet":
                        assert set(mapping["prototype_test"]) == {
                            "1",
                            "3",
                            "5",
                            "10",
                            "15",
                        }
                        assert "wording_shift_test" in mapping
                    if family == "coco" and n == 8192:
                        assert set(mapping["caption_test"]) == {"1", "3", "5"}
                        assert len(mapping["aro"]) == 2
                        for record in mapping["aro"].values():
                            for control in (
                                "original_coordinates",
                                "paired_fit",
                                "shuffled_fit",
                                "shuffled_image",
                            ):
                                assert all(
                                    0 <= value <= 1
                                    for value in record[control].values()
                                )
                            if row["layer"] == "final_norm":
                                assert "bootstrap" in record
                        aro += 1
                    if (
                        family == "coco"
                        and row["profile"] == "native"
                        and row["layer"] in ("13", "23", "final_norm")
                        and row["readout"] in ("content_mean", "query_native")
                        and mapping["dimension"] in (32, 1024)
                    ):
                        assert [r["seed"] for r in mapping["repeat_fits"]] == [
                            20260910,
                            20260911,
                        ]
                        repeats += 1
    robust = json.loads((root / "robustness-geometry-v4.json").read_text())
    finite_tree(robust)
    assert len(robust["rows"]) == 216
    assert (
        len(
            {
                tuple(
                    r[k]
                    for k in (
                        "layer",
                        "readout",
                        "source_family",
                        "target_family",
                        "profile",
                        "mode",
                    )
                )
                for r in robust["rows"]
            }
        )
        == 216
    )
    selections = json.loads((root / "dev-selected-geometry-v4.json").read_text())
    assert len(selections) == 168
    for chosen in selections:
        candidates = []
        for row in rows:
            if all(row[k] == chosen[k] for k in ("state", "profile", "readout")):
                for mapping in row["families"][chosen["family"]][chosen["mode"]][
                    "mappings"
                ].values():
                    if (
                        mapping["valid"]
                        and mapping["fit_points"] == chosen["fit_points"]
                    ):
                        candidates.append(mapping["dev"]["paired"]["r2"])
        assert chosen["dev_r2"] == max(candidates)
    layer_null = json.loads((root / "layer-search-null-v4.json").read_text())
    assert len(layer_null) == 504
    finite_tree(layer_null)
    disjoint = json.loads((root / "aro-disjoint-v4.json").read_text())
    finite_tree(disjoint)
    assert len(disjoint["rows"]) == 168
    assert sum(disjoint["cohort_sizes_by_task"]["verified_disjoint"].values()) == 868
    assert sum(disjoint["cohort_sizes_by_task"]["no_known_overlap"].values()) == 1924
    assert len(disjoint["ema_final_norm_cross_covariance"]) == 24
    position = json.loads((root / "position-exploration-v4.json").read_text())
    finite_tree(position)
    assert len(position["within_modality"]) == 24
    assert len(position["crossmodal_frozen_map"]) == 12
    selected_intervals = json.loads(
        (root / "selected-uncertainty-v4.json").read_text()
    )["rows"]
    finite_tree(selected_intervals)
    assert len(selected_intervals) == 4
    for record in selected_intervals:
        match = [
            s
            for s in selections
            if all(
                s[k] == record[k]
                for k in (
                    "state",
                    "profile",
                    "readout",
                    "family",
                    "mode",
                    "layer",
                    "mapping",
                )
            )
        ]
        assert (
            len(match) == 1
            and abs(match[0]["test_r2"] - record["test"]["paired"]["r2"]) < 1e-10
        )
    cleanup = json.loads((root / "notebook-cleanup.json").read_text())
    assert cleanup["status_after_stop"] == "STOPPED" and not cleanup["notebook_deleted"]
    write_json(
        root / "analysis-audit-v4.json",
        {
            "status": "passed",
            "layer_records": len(rows),
            "mapping_records": mappings,
            "expected_unfit_subspaces": invalid,
            "mapping_test_intervals": intervals,
            "aro_map_settings": aro,
            "repeated_fit_settings": repeats,
            "robustness_records": len(robust["rows"]),
            "dev_selected_settings": len(selections),
            "max_layer_null_records": len(layer_null),
            "identity_disjoint_aro_map_settings": len(disjoint["rows"]),
            "posthoc_position_records": 36,
            "selected_endpoint_intervals": 4,
            "notebook_cleanup": "STOPPED, object and shared artifacts retained",
            "note": "Completion and numerical/structural audit, not a threshold-based success declaration. rotation_identified denotes input/output sample-span coverage, not a cross-covariance condition guarantee.",
        },
    )
    emit("analysis_audit_passed_v4", rows=len(rows), maps=mappings)


if __name__ == "__main__":
    main()
