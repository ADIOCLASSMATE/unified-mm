"""Conditional test uncertainty for the dev-selected native-query endpoints."""

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_geometry_v4_robustness import native_arrays
from scripts.analyze_unified_geometry_v3 import error_bootstrap, preprocess
from scripts.geometry_v4_math import describe_fit, evaluate_pair, fit_pair
from utils.research.representation_protocol import emit, write_json
from scripts.probe_unified_semantics_v2 import LAYERS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    root = a.output_dir
    torch.set_num_threads(1)
    rows = [
        json.loads(p.read_text())
        for p in (root / "analysis/final_ema/native").glob("layer-*-query_native.json")
    ]
    assert len(rows) == 30
    data = {
        family: torch.load(
            root / "views/final_ema/native" / f"{family}.pt",
            mmap=True,
            weights_only=True,
        )
        for family in ("imagenet", "coco")
    }
    records = []
    for family, n in (("imagenet", 600), ("coco", 8192)):
        for mode in ("centered_euclidean", "centered_unit_sphere"):
            candidates = [
                (r, name, m)
                for r in rows
                for name, m in r["families"][family][mode]["mappings"].items()
                if m["valid"] and m["fit_points"] == n
            ]
            selected, name, expected = max(
                candidates,
                key=lambda x: (
                    x[2]["dev"]["paired"]["r2"],
                    -LAYERS.index(x[0]["layer"]),
                    -x[2]["dimension"],
                ),
            )
            layer = LAYERS.index(selected["layer"])
            source = native_arrays(data[family], family, layer, 2)
            target_family = "coco" if family == "imagenet" else "imagenet"
            target = native_arrays(data[target_family], target_family, layer, 2)
            fit = fit_pair(
                preprocess(source["fit_x"], source["cal_x"], mode),
                preprocess(source["fit_y"], source["cal_y"], mode),
                "full" if expected["dimension"] == 1024 else expected["dimension"],
                20260909,
            )
            record = {
                "state": "final_ema",
                "profile": "native",
                "readout": "query_native",
                "family": family,
                "mode": mode,
                "layer": selected["layer"],
                "mapping": name,
                "dev_r2": expected["dev"]["paired"]["r2"],
                **describe_fit(fit),
            }
            for split, values in (("test", source), ("transfer_test", target)):
                x = preprocess(values["test_x"], source["cal_x"], mode)
                y = preprocess(values["test_y"], source["cal_y"], mode)
                scores, errors, null_errors, baseline = evaluate_pair(fit, x, y)
                assert (
                    abs(scores["paired"]["r2"] - expected[split]["paired"]["r2"])
                    < 1e-10
                )
                scores["bootstrap"] = error_bootstrap(errors, null_errors, baseline)
                record[split] = scores
            records.append(record)
    write_json(
        root / "selected-uncertainty-v4.json",
        {
            "rows": records,
            "note": "Conditional 2000-repeat test-group bootstrap after independent dev-only selection. Does not include new training runs or uncertainty from resampling fit/dev.",
        },
    )
    emit("selected_uncertainty_complete_v4", endpoints=len(records))


if __name__ == "__main__":
    main()
