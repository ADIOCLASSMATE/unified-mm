"""Orthogonal subspace fits with explicit cross-covariance identifiability."""

import torch

from scripts.geometry_v4_math import spectral_fit, spectral_frame


def rotation(x, y):
    assert x.shape == y.shape, "A full-space rotation requires equal dimensions"
    cross = x.T @ y
    u, singular, vh = torch.linalg.svd(cross, full_matrices=False)
    q = u @ vh
    orthogonal_error = float(
        (q.T @ q - torch.eye(q.shape[0], dtype=q.dtype)).abs().max()
    )
    assert orthogonal_error < 1e-8
    scale = singular.sum() / x.square().sum().clamp_min(1e-20)
    rank = (
        int((singular > singular[0] * 1e-6).sum()) if float(singular[0]) > 1e-20 else 0
    )
    diagnostics = {
        "cross_covariance_rank": rank,
        "cross_covariance_condition": float(singular[0] / singular[-1])
        if float(singular[-1]) > 0
        else None,
        "cross_covariance_condition_nonzero": float(singular[0] / singular[rank - 1])
        if rank
        else None,
        "cross_covariance_singular_values": singular.tolist(),
        "cross_covariance_rank_relative_tolerance": 1e-6,
        "orthogonality_max_abs": orthogonal_error,
    }
    return q, scale, diagnostics


def fit_pair(x, y, dimension, seed=20260909, spectra=None):
    if dimension == "full" and x.shape[-1] != y.shape[-1]:
        return None
    sx, sy = spectra if spectra is not None else (spectral_fit(x), spectral_fit(y))
    fx, fy = spectral_frame(sx, dimension), spectral_frame(sy, dimension)
    if fx is None or fy is None:
        return None
    q, scale, diagnostics = rotation(fx["fit"], fy["fit"])
    order = torch.randperm(len(x), generator=torch.Generator().manual_seed(seed))
    nq, ns, null_diagnostics = rotation(fx["fit"], fy["fit"][order])
    return {
        "fx": fx,
        "fy": fy,
        "q": q,
        "scale": scale,
        "null_q": nq,
        "null_scale": ns,
        "diagnostics": diagnostics,
        "null_diagnostics": null_diagnostics,
        "rotation_identified": min(
            fx["numerical_rank"],
            fy["numerical_rank"],
            diagnostics["cross_covariance_rank"],
        )
        >= fx["dimension"],
    }
