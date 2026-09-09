"""Post-hoc, calibration-only explanation of native-query slot sensitivity."""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_unified_geometry_v3 import (
    apply_frame,
    error_metrics,
    geometry_statistics,
    preprocess,
)
from scripts.geometry_v4_math import describe_fit, evaluate_pair, fit_pair, spectral_fit
from scripts.prepare_geometry_v4_views import grouped
from utils.research.representation_protocol import emit, write_json

PROFILES = ("native_sigma1", "native_sigma2", "native_mean")
MODES = ("centered_euclidean", "centered_unit_sphere")
SEED = 20260909


def calibration(root, profile, samples):
    path = root / "views/final_ema" / profile / "coco-calibration.pt"
    if path.exists():
        return torch.load(path, mmap=True, weights_only=True)
    groups = [r["group"] for r in samples["coco_images"] if r["split"] == "cal"]
    data = {"groups": torch.tensor(groups), "profile": profile}
    for modality, axis in (("images", "x"), ("texts", "y")):
        rows = samples[f"coco_{modality}"]
        selected = [i for i, r in enumerate(rows) if r["split"] == "cal"]
        lookup = {v: i for i, v in enumerate(selected)}
        features = torch.empty(len(selected), 30, 3, 1024, dtype=torch.bfloat16)
        found = []
        for shard_path in sorted(
            (root / "final_ema" / profile).glob(f"coco_{modality}-rank-*-of-16.pt")
        ):
            shard = torch.load(shard_path, mmap=True, weights_only=True)
            chosen = [
                (i, int(j)) for i, j in enumerate(shard["indices"]) if int(j) in lookup
            ]
            if chosen:
                features[[lookup[j] for _, j in chosen]] = shard["features"][
                    [i for i, _ in chosen]
                ]
                found.extend(j for _, j in chosen)
        assert sorted(found) == selected
        data[axis] = grouped(
            features, [rows[i] for i in selected], groups, lambda r: True
        )
    torch.save(data, path)
    return data


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    root = a.output_dir
    torch.set_num_threads(1)
    samples = json.loads((root / "samples.json").read_text())
    cal = {
        profile: calibration(root, profile, samples)
        for profile in ("native", *PROFILES)
    }
    native = torch.load(
        root / "views/final_ema/native/coco.pt", mmap=True, weights_only=True
    )
    changed = {
        profile: torch.load(
            root / "views/final_ema" / profile / "coco.pt", mmap=True, weights_only=True
        )
        for profile in PROFILES
    }
    native_groups = native["groups"][native["test_indices"]][:512]
    assert all(
        torch.equal(native_groups, value["groups"]) for value in changed.values()
    )
    within, cross = [], []
    for readout, pool in (("content_mean", 0), ("query_native", 2)):
        for profile in PROFILES:
            for mode in MODES:
                adjusted = {}
                for axis in ("x", "y"):
                    bc, pc = (
                        cal["native"][axis][:, -1, pool].double(),
                        cal[profile][axis][:, -1, pool].double(),
                    )
                    bt, pt = (
                        native[f"test_{axis}"][:512, -1, pool].double(),
                        changed[profile][f"test_{axis}"][:, -1, pool].double(),
                    )
                    base_mean, changed_mean = bc.mean(0), pc.mean(0)
                    delta = changed_mean - base_mean
                    adjusted[axis] = pt - delta
                    xf, yf = (
                        preprocess(bc, base_mean, mode),
                        preprocess(pc, changed_mean, mode),
                    )
                    xt, yt = (
                        preprocess(bt, base_mean, mode),
                        preprocess(pt, changed_mean, mode),
                    )
                    frozen_yt = preprocess(pt, base_mean, mode)
                    row = {
                        "readout": readout,
                        "profile": profile,
                        "mode": mode,
                        "axis": axis,
                        "calibration_scenes": 512,
                        "test_scenes": 512,
                        "native_mean_cosine": float(
                            F.cosine_similarity(xt, frozen_yt).mean()
                        ),
                        "own_calibrated_mean_cosine": float(
                            F.cosine_similarity(xt, yt).mean()
                        ),
                        "geometry": geometry_statistics(xt, yt, permutations=0),
                        "mean_shift_over_native_rms": float(
                            delta.norm()
                            / (bc - base_mean).square().sum(1).mean().sqrt()
                        ),
                        "maps": {},
                    }
                    spectra = spectral_fit(xf), spectral_fit(yf)
                    for dim in ("full", 32):
                        fit = fit_pair(xf, yf, dim, SEED, spectra=spectra)
                        record = {
                            **describe_fit(fit),
                            "test": evaluate_pair(fit, xt, yt)[0],
                        }
                        if dim == "full":
                            xb, _ = apply_frame(fit["fx"], xt)
                            yb, _ = apply_frame(fit["fy"], yt)
                            record["identity_after_calibration_and_rms"] = (
                                error_metrics(xb, yb)[0]
                            )
                        row["maps"][str(dim)] = record
                    within.append(row)
                record = {
                    "readout": readout,
                    "profile": profile,
                    "mode": mode,
                    "maps": {},
                }
                for dim in ("full", 32, 128, 512):
                    fit = torch.load(
                        root
                        / "fitted-maps/final_ema"
                        / f"layer-final_norm-{readout}-{mode}-dim{dim}.pt",
                        weights_only=True,
                    )
                    values = {}
                    for label in (
                        "native",
                        "perturbed",
                        "calibration_translation_only",
                    ):
                        coordinates = {}
                        for axis in ("x", "y"):
                            raw = (
                                native[f"test_{axis}"][:512, -1, pool]
                                if label == "native"
                                else changed[profile][f"test_{axis}"][:, -1, pool]
                                if label == "perturbed"
                                else adjusted[axis]
                            )
                            coordinates[axis] = preprocess(
                                raw, native[f"cal_{axis}"][-1, pool], mode
                            )
                        values[label] = evaluate_pair(
                            fit, coordinates["x"], coordinates["y"]
                        )[0]
                    record["maps"][str(dim)] = values
                cross.append(record)
            emit("position_diagnostic_done_v4", readout=readout, profile=profile)
    write_json(
        root / "position-exploration-v4.json",
        {
            "schema": "geometry_v4_posthoc_position_1",
            "seed": SEED,
            "status": "post-hoc explanatory analysis, prompted by prospective sigma2 sensitivity; not a new confirmatory endpoint",
            "first_image_query_slots_0_based": {
                "native": [7, 8],
                "native_sigma1": [7, 7],
                "native_sigma2": [14, 13],
            },
            "calibration": "Only the 512 disjoint COCO calibration scenes; fixed 512 test scenes. Translation is estimated in raw coordinates before any unit-sphere normalization. No 8192-fit map is refitted.",
            "within_modality": within,
            "crossmodal_frozen_map": cross,
        },
    )


if __name__ == "__main__":
    main()
