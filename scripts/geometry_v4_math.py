"""Reusable fit-only spectral frames for large-sample V4 orthogonal maps."""

import torch

from scripts.analyze_unified_geometry_v3 import (
    apply_frame,
    error_metrics,
    solve_rotation,
)


def spectral_fit(values):
    values = values.double()
    mean = values.mean(0)
    centered = values - mean
    covariance = centered.T @ centered
    energy, basis = torch.linalg.eigh(covariance)
    energy, basis = energy.flip(0).clamp_min(0), basis.flip(1)
    total = energy.sum()
    if float(total) <= 1e-18:
        return None
    rank = min(len(values) - 1, int((energy > energy[0] * 1e-12).sum()))
    condition = float((energy[0] / energy[rank - 1]).sqrt()) if rank else None
    return {
        "mean": mean,
        "centered": centered,
        "energy": energy,
        "basis": basis,
        "total": total,
        "rank": rank,
        "condition_number_nonzero": condition,
        "effective_dimension": float(total.square() / energy.square().sum()),
    }


def spectral_frame(spectrum, dimension):
    if spectrum is None:
        return None
    dim = spectrum["centered"].shape[1] if dimension == "full" else int(dimension)
    if dimension != "full" and spectrum["rank"] < dim:
        return None
    projection = None if dimension == "full" else spectrum["basis"][:, :dim]
    z = (
        spectrum["centered"]
        if projection is None
        else spectrum["centered"] @ projection
    )
    radius = z.square().sum(1).mean().sqrt()
    return {
        "mean": spectrum["mean"],
        "projection": projection,
        "radius": radius,
        "fit": z / radius,
        "dimension": dim,
        "numerical_rank": spectrum["rank"],
        "participation_ratio": spectrum["effective_dimension"],
        "condition_number_nonzero": spectrum["condition_number_nonzero"],
        "retained_variance_fit": float(z.square().sum() / spectrum["total"]),
    }


def fit_pair(x, y, dimension, seed, spectra=None):
    sx, sy = spectra if spectra is not None else (spectral_fit(x), spectral_fit(y))
    fx, fy = spectral_frame(sx, dimension), spectral_frame(sy, dimension)
    if fx is None or fy is None:
        return None
    q, scale = solve_rotation(fx["fit"], fy["fit"])
    order = torch.randperm(len(x), generator=torch.Generator().manual_seed(seed))
    null_q, null_scale = solve_rotation(fx["fit"], fy["fit"][order])
    return {
        "fx": fx,
        "fy": fy,
        "q": q,
        "scale": scale,
        "null_q": null_q,
        "null_scale": null_scale,
    }


def evaluate_pair(fit, x, y):
    x, retained_x = apply_frame(fit["fx"], x.double())
    y, retained_y = apply_frame(fit["fy"], y.double())
    paired, errors, baseline = error_metrics(x @ fit["q"], y, fit["scale"])
    null, null_errors, _ = error_metrics(x @ fit["null_q"], y, fit["null_scale"])
    return (
        {
            "points": len(x),
            "paired": paired,
            "shuffled_fit": null,
            "variance_retained_x": retained_x,
            "variance_retained_y": retained_y,
        },
        errors,
        null_errors,
        baseline,
    )


def describe_fit(fit):
    return {
        "dimension": fit["fx"]["dimension"],
        "rank_x": fit["fx"]["numerical_rank"],
        "rank_y": fit["fy"]["numerical_rank"],
        "rotation_identified": min(
            fit["fx"]["numerical_rank"], fit["fy"]["numerical_rank"]
        )
        >= fit["fx"]["dimension"],
        "condition_number_x": fit["fx"]["condition_number_nonzero"],
        "condition_number_y": fit["fy"]["condition_number_nonzero"],
        "effective_dimension_x": fit["fx"]["participation_ratio"],
        "effective_dimension_y": fit["fy"]["participation_ratio"],
        "fit_variance_retained_x": fit["fx"]["retained_variance_fit"],
        "fit_variance_retained_y": fit["fy"]["retained_variance_fit"],
        "paired_scale": float(fit["scale"]),
        "shuffled_scale": float(fit["null_scale"]),
    }
