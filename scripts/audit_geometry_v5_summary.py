"""Verify full CSV, dev-choice and layer-search-null summaries against every source row."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import torch

from scripts.prepare_geometry_v5_assets import RUN, emit, write_json


def read(path):
    return json.loads(Path(path).read_text())


def main(root):
    torch.set_num_threads(1)
    contract = read(root / "comparison-contract.json")
    expected, curves = {}, defaultdict(list)
    row_count = 0
    for setting, layout in contract["settings"].items():
        paths = list((root / "analysis" / setting).glob("*.json"))
        assert len(paths) == len(layout["layer_pairs"]) * len(layout["readouts"])
        actual = set()
        for path in paths:
            row = read(path)
            key = row["pair_index"], row["readout"]
            assert key not in actual
            actual.add(key)
            assert (
                row["setting"] == setting
                and row["pair"] == layout["layer_pairs"][key[0]]
            )
            assert row["readout_specification"] == layout["readouts"][key[1]]
            row_count += 1
            for family, modes in row["families"].items():
                for mode, bundle in modes.items():
                    gkey = setting, key[1], family, mode
                    geometry = bundle["geometry"]["primary"]
                    if geometry["valid"]:
                        curves[gkey].append(geometry)
                    for mapping, result in bundle["mappings"].items():
                        ckey = setting, key[0], key[1], family, mode, mapping
                        assert ckey not in expected
                        expected[ckey] = result, geometry
        assert actual == {
            (i, r)
            for i in range(len(layout["layer_pairs"]))
            for r in layout["readouts"]
        }
    index = read(root / "analysis-index-v5.json")
    assert index["rows"] == row_count == 1171
    assert set(index["settings"]) == set(contract["settings"])
    observed, choices = set(), defaultdict(list)
    with (root / "geometry-v5.csv").open() as handle:
        for row in csv.DictReader(handle):
            key = (
                row["setting"],
                int(row["pair_index"]),
                row["readout"],
                row["family"],
                row["mode"],
                row["mapping"],
            )
            assert key not in observed and key in expected
            observed.add(key)
            source, geo = expected[key]
            assert row["valid"] == str(source["valid"])
            assert int(row["fit_points"]) == source["fit_points"]
            assert row["requested_dimension"] == str(source["requested_dimension"])
            if geo["valid"]:
                for metric, value in geo["scores"].items():
                    assert float(row[metric]) == value
            if source["valid"]:
                for split in ("fit", "dev", "test", "transfer_test"):
                    assert float(row[f"{split}_r2"]) == source[split]["paired"]["r2"]
                    assert (
                        float(row[f"{split}_shuffled_r2"])
                        == source[split]["shuffled_fit"]["r2"]
                    )
                if source["requested_dimension"] != "full":
                    gkey = key[0], key[2], key[3], key[4], source["fit_points"]
                    choices[gkey].append(row)
    assert observed == set(expected) and len(observed) == 37472
    selected = read(root / "dev-selected-geometry-v5.json")
    assert len(selected) == len(choices)
    selected_keys = set()
    for row in selected:
        key = (
            row["setting"],
            row["readout"],
            row["family"],
            row["mode"],
            row["fit_points"],
        )
        assert key not in selected_keys
        selected_keys.add(key)
        best = max(
            choices[key],
            key=lambda v: (
                float(v["dev_r2"]),
                -int(v["pair_index"]),
                -int(v["dimension"]),
            ),
        )
        assert (
            row["pair_index"] == int(best["pair_index"])
            and row["mapping"] == best["mapping"]
        )
        assert row["dev_r2"] == float(best["dev_r2"])
    assert selected_keys == set(choices)
    nulls = read(root / "layer-search-null-v5.json")
    expected_null = {
        (key, metric) for key, curve in curves.items() for metric in curve[0]["scores"]
    }
    observed_null = set()
    for row in nulls:
        key = row["setting"], row["readout"], row["family"], row["mode"]
        pair = key, row["metric"]
        assert pair not in observed_null and pair in expected_null
        observed_null.add(pair)
        curve = curves[key]
        null = (
            torch.tensor([g["null_samples"][row["metric"]] for g in curve])
            .max(0)
            .values
        )
        point = max(g["scores"][row["metric"]] for g in curve)
        assert row["observed_max"] == point and row["valid_layer_pairs"] == len(curve)
        assert row["null_max_q95"] == float(null.quantile(0.95))
        assert row["p"] == (1 + int(null.ge(point).sum())) / (1 + len(null))
    assert observed_null == expected_null
    result = {
        "schema": "geometry_v5_summary_audit_1",
        "status": "passed",
        "layer_readout_rows": row_count,
        "csv_rows": len(observed),
        "dev_selections": len(selected_keys),
        "layer_search_null_tests": len(observed_null),
        "csv_exactly_matches_all_curve_scores": True,
        "selection_recomputed_from_dev_only": True,
        "all_max_layer_nulls_recomputed": True,
    }
    write_json(root / "audits/summary-tables.json", result)
    emit("v5_summary_audit_passed", **result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=RUN)
    main(parser.parse_args().output_dir)
