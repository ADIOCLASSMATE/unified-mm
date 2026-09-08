"""Re-evaluate frozen V4 maps on ARO cohorts audited against COCO image IDs."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.analyze_unified_geometry_v4 as primary
from scripts.analyze_unified_geometry_v3 import apply_frame, preprocess
from scripts.geometry_v4_math import fit_pair, spectral_fit
from scripts.probe_unified_representations import emit, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    root = a.output_dir
    torch.set_num_threads(1)
    audit = json.loads((root / "sample-audit-v4.json").read_text())
    details = audit["aro_identity_details"]
    cohorts = {
        "verified_disjoint": torch.tensor(
            [r["index"] for r in details if r["category"] == "verified_coco_disjoint"]
        ),
        "no_known_overlap": torch.tensor(
            [r["index"] for r in details if r["category"] != "known_overlap"]
        ),
    }
    sizes = {
        name: dict(Counter(details[i]["task"] for i in indices.tolist()))
        for name, indices in cohorts.items()
    }
    rows, conditioning = [], []
    settings = [("final_ema", profile) for profile in ("native", "bare", "neutral")]
    settings += [
        (state, "native") for state in ("final_raw", "init42", "init43", "init44")
    ]
    for state, profile in settings:
        source = torch.load(
            root / "views" / state / profile / "coco.pt", weights_only=True, mmap=True
        )
        aro = torch.load(
            root / "views" / state / profile / "aro.pt", weights_only=True, mmap=True
        )
        for readout, pool in primary.READOUTS.items():
            data = {k: source[k][-1, pool].double() for k in ("cal_x", "cal_y")}
            for mode in primary.MODES:
                xf = preprocess(source["fit_x"][:, -1, pool], data["cal_x"], mode)
                yf = preprocess(source["fit_y"][:, -1, pool], data["cal_y"], mode)
                spectra = None
                for dim in primary.DIMS:
                    saved = (
                        root
                        / "fitted-maps"
                        / state
                        / f"layer-final_norm-{readout}-{mode}-dim{dim}.pt"
                    )
                    if profile == "native" and saved.exists():
                        fit = torch.load(saved, weights_only=True)
                    else:
                        if spectra is None:
                            spectra = spectral_fit(xf), spectral_fit(yf)
                        fit = fit_pair(xf, yf, dim, primary.SEED, spectra=spectra)
                    row = {
                        "state": state,
                        "profile": profile,
                        "layer": "final_norm",
                        "readout": readout,
                        "mode": mode,
                        "dimension": str(dim),
                        "cohorts": {},
                    }
                    for name, indices in cohorts.items():
                        tasks = [aro["tasks"][i] for i in indices.tolist()]
                        primary.DATA = {"aro": {"tasks": tasks}}
                        values = {
                            key: aro[key][indices, -1, pool].double()
                            for key in ("test_x", "test_y_positive", "test_y_negative")
                        }
                        row["cohorts"][name] = primary.aro_controls(
                            fit, data, mode, values, bootstrap=True
                        )
                    rows.append(row)
                    if state == "final_ema" and profile == "native":
                        x, _ = apply_frame(fit["fx"], xf)
                        y, _ = apply_frame(fit["fy"], yf)
                        singular = torch.linalg.svdvals(x.T @ y)
                        condition = float(singular[0] / singular[-1])
                        conditioning.append(
                            {
                                "readout": readout,
                                "mode": mode,
                                "dimension": str(dim),
                                "fit_points": 8192,
                                "cross_covariance_rank_at_relative_1e_6": int(
                                    (singular > singular[0] * 1e-6).sum()
                                ),
                                "cross_covariance_condition_number": condition,
                                "orthogonality_error_per_sqrt_dimension": float(
                                    (
                                        fit["q"].T @ fit["q"] - torch.eye(len(singular))
                                    ).norm()
                                    / len(singular) ** 0.5
                                ),
                            }
                        )
            emit(
                "aro_disjoint_readout_done_v4",
                state=state,
                profile=profile,
                readout=readout,
            )
    write_json(
        root / "aro-disjoint-v4.json",
        {
            "schema": "geometry_v4_aro_identity_control_1",
            "status": "post-audit identity isolation, no score-based exclusions or map refitting on ARO",
            "cohort_sizes_by_task": sizes,
            "source_identity_categories": audit["aro_identity_categories"],
            "rows": rows,
            "ema_final_norm_cross_covariance": conditioning,
            "note": "verified_disjoint requires a known official COCO identity absent from all V4 COCO splits. no_known_overlap also includes unlinked VG images; no content-level decontamination is claimed. All image shuffles stay within the selected cohort and task.",
        },
    )
    emit("aro_disjoint_analysis_complete_v4", maps=len(rows), sizes=sizes)


if __name__ == "__main__":
    main()
