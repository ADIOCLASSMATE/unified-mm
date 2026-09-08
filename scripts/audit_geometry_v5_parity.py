"""Check all B V5 curves against the authoritative V4 baseline, including strict ARO."""

import argparse
import json
from pathlib import Path

from scripts.prepare_cross_model_geometry_v5 import V4
from scripts.prepare_geometry_v5_assets import RUN, emit, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=RUN)
    args = parser.parse_args()
    readouts = {
        "content_mean": "content_mean",
        "content_last": "content_last_sigma",
        "native_task": "query_native",
    }
    strict = json.loads((V4 / "aro-disjoint-v4.json").read_text())["rows"]
    strict = {
        (r["profile"], r["readout"], r["mode"], r["dimension"]): r["cohorts"][
            "verified_disjoint"
        ]
        for r in strict
        if r["state"] == "final_ema"
    }
    geometries, comparisons, aro_comparisons, largest = 0, 0, 0, 0.0
    checked = []
    for profile in ("native", "bare", "neutral"):
        files = sorted((args.output_dir / "analysis" / f"b_{profile}").glob("*.json"))
        assert len(files) == 90
        for path in files:
            new = json.loads(path.read_text())
            readout = readouts[new["readout"]]
            old_path = (
                V4
                / "analysis/final_ema"
                / profile
                / f"layer-{new['pair']['layer_x']}-{readout}.json"
            )
            old = json.loads(old_path.read_text())
            for family, modes in new["families"].items():
                for mode, value in modes.items():
                    old_mode = "centered_unit_sphere" if mode == "unit_sphere" else mode
                    source = old["families"][family][old_mode]
                    assert value["geometry"] == source["geometry"], (path, family, mode)
                    geometries += 1
                    for key, mapping in value["mappings"].items():
                        reference = source["mappings"][key]
                        assert mapping["valid"] == reference["valid"]
                        if not mapping["valid"]:
                            continue
                        for split in ("fit", "dev", "test", "transfer_test"):
                            for control in ("paired", "shuffled_fit"):
                                error = abs(
                                    mapping[split][control]["r2"]
                                    - reference[split][control]["r2"]
                                )
                                assert error < 1e-10, (
                                    path,
                                    family,
                                    mode,
                                    key,
                                    split,
                                    control,
                                    error,
                                )
                                largest = max(largest, error)
                                comparisons += 1
                        if (
                            new["pair"]["kind"] == "final_norm"
                            and family == "coco"
                            and mapping["fit_points"] == 8192
                        ):
                            aro = strict[
                                profile,
                                readout,
                                old_mode,
                                str(mapping["requested_dimension"]),
                            ]
                            for task, scores in mapping["aro"].items():
                                for control in (
                                    "paired_fit",
                                    "shuffled_fit",
                                    "shuffled_image",
                                    "original_coordinates",
                                ):
                                    for metric in ("cosine", "distance"):
                                        assert (
                                            scores[control][metric]
                                            == aro[task][control][metric]
                                        )
                                        aro_comparisons += 1
            checked.append({"v5": str(path), "v4": str(old_path)})
    result = {
        "status": "passed",
        "layer_readout_rows": len(checked),
        "geometry_bundles_exact": geometries,
        "mapping_r2_comparisons": comparisons,
        "maximum_absolute_r2_difference": largest,
        "strict_aro_accuracy_comparisons": aro_comparisons,
        "records": checked,
        "intentional_difference": "V5 additionally requires cross-covariance rank for rotation_identified; original 2000-ARO curves are replaced only in new V5 analysis by the already audited 868-ID cohort",
    }
    write_json(args.output_dir / "audits" / "b-v4-parity.json", result)
    emit(
        "v5_b_v4_parity_passed",
        **{key: value for key, value in result.items() if key != "records"},
    )


if __name__ == "__main__":
    main()
