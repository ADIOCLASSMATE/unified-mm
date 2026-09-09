"""Audit exact cross-model contrast coverage and recompute every paired interval."""

import argparse
import json
import multiprocessing as mp
from pathlib import Path

from scripts.audit_geometry_v5_results import finite_tree
from scripts.compare_geometry_v5 import compare_endpoints, key_for, load_arrays
from utils.research.geometry_v5_assets import RUN, emit, write_json


def read(path):
    return json.loads(Path(path).read_text())


def audit_one(row):
    a, b = [read(p) for p in row["endpoint_files"]]
    key = (a["setting"], b["setting"], key_for(a))
    finite_tree(row)
    # Re-read semantic arrays and recompute every paired interval.
    assert compare_endpoints(a, b, row["endpoint_files"]) == row
    return key, row["primary_contrast"]


def main(root, workers=1):
    import torch

    torch.set_num_threads(1)
    contract = read(root / "comparison-contract.json")
    endpoint_sets = {}
    for setting, layout in contract["settings"].items():
        audit = read(root / "audits" / f"results-{setting}.json")
        assert audit["status"] == "passed"
        assert audit["layer_readout_rows"] == len(layout["layer_pairs"]) * len(
            layout["readouts"]
        )
        rows = [read(p) for p in (root / "endpoints" / setting).glob("*.json")]
        endpoint_sets[setting] = {
            key_for(r): r
            for r in rows
            if r["valid"] and r["requested_dimension"] != "full"
        }
    contrasts = [
        ("b_native", other)
        for other in (
            "f_native",
            "dinov2_qwen",
            "mae_qwen",
            "siglip",
            "janusflow_understanding",
            "janusflow_generation",
            "showo2_understanding",
            "showo2_generation",
        )
    ]
    contrasts += [
        ("b_bare", "f_bare"),
        ("b_neutral", "f_neutral"),
        ("b_native", "b_bare"),
        ("b_native", "b_neutral"),
        ("f_native", "f_bare"),
        ("f_native", "f_neutral"),
    ]
    expected = set()
    for a, b in contrasts:
        expected.update(
            (a, b, key) for key in endpoint_sets[a].keys() & endpoint_sets[b].keys()
        )
    for key, row in endpoint_sets["siglip_native"].items():
        akey = key_for(row, "content_mean")
        if akey in endpoint_sets["b_native"]:
            expected.add(("b_native", "siglip_native", akey))
    report = read(root / "paired-model-differences-v5.json")
    assert report["bootstrap"] == 2000 and report["seed"] == 20260908
    observed, primary = set(), 0
    assert workers > 0
    with mp.get_context("fork").Pool(
        workers, initializer=torch.set_num_threads, initargs=(1,)
    ) as pool:
        for i, (key, is_primary) in enumerate(
            pool.imap(audit_one, report["rows"], chunksize=4)
        ):
            assert key not in observed and key in expected
            observed.add(key)
            primary += is_primary
            if i % 128 == 0:
                emit(
                    "v5_paired_audit_progress",
                    completed=i + 1,
                    total=len(report["rows"]),
                )
    assert (
        observed == expected and primary == 16
    )  # 8 external/F comparisons × 2 sources.
    load_arrays.cache_clear()
    result = {
        "schema": "geometry_v5_paired_audit_1",
        "status": "passed",
        "contrasts_recomputed": len(observed),
        "primary_contrasts": primary,
        "identities_and_intervals_recomputed": True,
        "expected_setting_pairs": contrasts + [("b_native", "siglip_native")],
        "bootstrap": 2000,
    }
    write_json(root / "audits/paired-model-differences.json", result)
    emit("v5_paired_audit_passed", **result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=RUN)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    main(args.output_dir, args.workers)
