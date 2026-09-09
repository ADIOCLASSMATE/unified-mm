"""Frozen-native-map sensitivity to token order and VAE posterior sampling."""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_unified_geometry_v3 import geometry_statistics, preprocess
from scripts.analyze_unified_semantics_v2 import load_dataset
from scripts.geometry_v4_math import describe_fit, evaluate_pair, fit_pair, spectral_fit
from scripts.prepare_geometry_v4_views import grouped
from utils.research.representation_protocol import emit, write_json
from scripts.probe_unified_semantics_v2 import LAYERS

SEED = 20260909
PROFILES = ("native_sigma1", "native_sigma2", "native_mean")
READOUTS = ("content_mean", "content_last_sigma", "query_native")
MODES = ("centered_euclidean", "centered_unit_sphere")
DIMS = ("full", 32, 128, 512)


def perturbation_view(root, profile_name, family):
    path = root / "views/final_ema" / profile_name / f"{family}.pt"
    if path.exists():
        value = torch.load(path, mmap=True, weights_only=True)
        assert value["schema"] == "geometry_v4_perturbation_views_1"
        assert value["profile"] == profile_name and value["family"] == family
        return value
    protocol = json.loads((root / "protocol.json").read_text())
    samples = json.loads((root / "samples.json").read_text())
    profile = protocol["profiles"][profile_name]
    image_rows = samples[f"{family}_images"]
    groups = list(
        dict.fromkeys(
            r["group"]
            for r in image_rows
            if r["robust"]
            and (
                r["mapping_split"] == "test"
                if family == "imagenet"
                else r["split"] == "test"
            )
        )
    )
    assert len(groups) == (32 if family == "imagenet" else 512)
    data = {
        "schema": "geometry_v4_perturbation_views_1",
        "state": "final_ema",
        "profile": profile_name,
        "family": family,
        "groups": torch.tensor(groups),
    }
    checks = None
    for modality, axis in (("images", "x"), ("texts", "y")):
        features, rows, current = load_dataset(
            root / "final_ema" / profile_name,
            f"{family}_{modality}",
            samples[f"{family}_{modality}"],
            profile,
            "final_ema",
        )
        assert features is not None and (checks is None or checks == current)
        checks = current
        data[f"test_{axis}"] = grouped(
            features,
            rows,
            groups,
            lambda r: r["split"] == "a" if family == "imagenet" else True,
        )
    data["state_checks"] = checks
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(data, temporary)
    temporary.replace(path)
    emit("perturbation_views_ready_v4", profile=profile_name, family=family)
    return data


def native_arrays(data, family, layer, pool):
    values = {
        f"cal_{axis}": data[f"cal_{axis}"][layer, pool].double() for axis in ("x", "y")
    }
    for split in ("fit", "test"):
        for axis in ("x", "y"):
            raw = (
                data["a_15_x" if axis == "x" else "a_y"][data[f"{split}_indices"]]
                if family == "imagenet"
                else data[f"{split}_{axis}"]
            )
            values[f"{split}_{axis}"] = raw[:, layer, pool].double()
    return values


def paired_change_interval(base_errors, changed_errors, baseline, repeats=2000):
    draws = torch.randint(
        len(baseline),
        (repeats, len(baseline)),
        generator=torch.Generator().manual_seed(SEED),
    )
    # A common native-target denominator isolates the perturbation's error change.
    delta = (base_errors - changed_errors)[draws].sum(1) / baseline[draws].sum(1)
    return delta.quantile(torch.tensor([0.025, 0.975], dtype=delta.dtype)).tolist()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--layers", default="13,23,final_norm")
    a = p.parse_args()
    root = a.output_dir
    torch.set_num_threads(1)
    native = {
        family: torch.load(
            root / "views/final_ema/native" / f"{family}.pt",
            mmap=True,
            weights_only=True,
        )
        for family in ("imagenet", "coco")
    }
    perturbed = {
        (profile, family): perturbation_view(root, profile, family)
        for profile in PROFILES
        for family in native
    }
    records = []
    for layer in a.layers.split(","):
        li = LAYERS.index(layer)
        for pool, readout in enumerate(READOUTS):
            base = {
                family: native_arrays(d, family, li, pool)
                for family, d in native.items()
            }
            for source_family, source in base.items():
                for mode in MODES:
                    xf = preprocess(source["fit_x"], source["cal_x"], mode)
                    yf = preprocess(source["fit_y"], source["cal_y"], mode)
                    spectra = spectral_fit(xf), spectral_fit(yf)
                    fits = {
                        str(dim): fit_pair(xf, yf, dim, SEED, spectra=spectra)
                        for dim in DIMS
                    }
                    for target_family, target in base.items():
                        original_groups = native[target_family]["groups"][
                            native[target_family]["test_indices"]
                        ].tolist()
                        lookup = {g: i for i, g in enumerate(original_groups)}
                        for profile in PROFILES:
                            changed = perturbed[profile, target_family]
                            ix = torch.tensor(
                                [lookup[g] for g in changed["groups"].tolist()]
                            )
                            bx = preprocess(target["test_x"][ix], source["cal_x"], mode)
                            by = preprocess(target["test_y"][ix], source["cal_y"], mode)
                            px = preprocess(
                                changed["test_x"][:, li, pool], source["cal_x"], mode
                            )
                            py = preprocess(
                                changed["test_y"][:, li, pool], source["cal_y"], mode
                            )
                            row = {
                                "layer": layer,
                                "readout": readout,
                                "source_family": source_family,
                                "target_family": target_family,
                                "profile": profile,
                                "mode": mode,
                                "points": len(ix),
                                "calibration_and_map": "frozen unperturbed native source; no target recalibration",
                                "geometry_native": geometry_statistics(
                                    bx, by, permutations=0
                                ),
                                "geometry_perturbed": geometry_statistics(
                                    px, py, permutations=0
                                ),
                                "within_modal_change": {},
                                "maps": {},
                            }
                            for axis, before, after in (("x", bx, px), ("y", by, py)):
                                row["within_modal_change"][axis] = {
                                    "mean_cosine": float(
                                        F.cosine_similarity(before, after).mean()
                                    ),
                                    "relative_rms": float(
                                        (
                                            (after - before).square().sum()
                                            / before.square().sum()
                                        ).sqrt()
                                    ),
                                }
                            for dimension, fit in fits.items():
                                if fit is None:
                                    row["maps"][dimension] = {"valid": False}
                                    continue
                                original, be, _, baseline = evaluate_pair(fit, bx, by)
                                changed_result, pe, _, _ = evaluate_pair(fit, px, py)
                                record = {
                                    "valid": True,
                                    **describe_fit(fit),
                                    "native": original,
                                    "perturbed": changed_result,
                                    "error_reduction_over_native_denominator": float(
                                        (be - pe).sum() / baseline.sum()
                                    ),
                                }
                                if layer == "final_norm":
                                    record["error_reduction_95_interval"] = (
                                        paired_change_interval(be, pe, baseline)
                                    )
                                row["maps"][dimension] = record
                            records.append(row)
            emit("perturbation_readout_done_v4", layer=layer, readout=readout)
    write_json(
        root / "robustness-geometry-v4.json",
        {
            "schema": "geometry_v4_robustness_1",
            "seed": SEED,
            "rows": records,
            "note": "Sensitivity on fixed cohorts. Different native-vs-perturbed R2 denominators are supplemented by paired error changes over the same native target denominator. These are not new independent training runs.",
        },
    )
    emit("perturbation_analysis_complete_v4", rows=len(records))


if __name__ == "__main__":
    main()
