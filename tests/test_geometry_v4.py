import torch

from scripts.analyze_geometry_v4_robustness import paired_change_interval
from scripts.analyze_unified_geometry_v3 import fitting_frame
from scripts.analyze_unified_geometry_v4 import pair_scores
from scripts.audit_geometry_v4_results import finite_tree
from scripts.geometry_v4_math import (
    evaluate_pair,
    fit_pair,
    spectral_fit,
    spectral_frame,
)
from scripts.prepare_geometry_v4_views import grouped


def test_spectral_frame_matches_svd_subspace_geometry():
    torch.manual_seed(711)
    x = torch.randn(100, 24, dtype=torch.float64)
    a = fitting_frame(x, 12)
    b = spectral_frame(spectral_fit(x), 12)
    torch.testing.assert_close(a["fit"] @ a["fit"].T, b["fit"] @ b["fit"].T)
    assert abs(a["retained_variance_fit"] - b["retained_variance_fit"]) < 1e-12


def test_identified_full_rotation_generalizes_to_new_points():
    torch.manual_seed(93)
    x = torch.randn(150, 24, dtype=torch.float64)
    q, _ = torch.linalg.qr(torch.randn(24, 24, dtype=torch.float64))
    y = x @ q * 3 + 8
    fit = fit_pair(x[:100], y[:100], "full", 51)
    result, *_ = evaluate_pair(fit, x[100:], y[100:])
    assert result["paired"]["nrmse"] < 1e-10
    assert result["shuffled_fit"]["nrmse"] > 0.5


def test_rank_is_bounded_by_centered_sample_count():
    torch.manual_seed(17)
    spectrum = spectral_fit(torch.randn(12, 24, dtype=torch.float64))
    assert spectrum["rank"] == 11
    assert spectral_frame(spectrum, 12) is None
    assert spectral_frame(spectrum, "full") is not None


def test_raw_grouping_preserves_all_three_readouts():
    x = torch.zeros(3, 30, 3, 4, dtype=torch.bfloat16)
    x[0, :, :, 0], x[1, :, :, 1], x[2] = 2, 8, 50
    rows = [
        {"group": 7, "view": "a"},
        {"group": 7, "view": "a"},
        {"group": 7, "view": "b"},
    ]
    result = grouped(x, rows, [7], lambda r: r["view"] == "a")
    for pool in range(3):
        torch.testing.assert_close(
            result[0, 0, pool], torch.tensor([1.0, 4.0, 0.0, 0.0])
        )


def test_no_signal_map_is_not_made_positive_by_training_fit():
    gen = torch.Generator().manual_seed(94)
    x = torch.randn(200, 16, generator=gen, dtype=torch.float64)
    y = torch.randn(200, 16, generator=gen, dtype=torch.float64)
    fit = fit_pair(x[:100], y[:100], "full", 51)
    result, *_ = evaluate_pair(fit, x[100:], y[100:])
    assert result["paired"]["r2"] < 0


def test_hard_negative_ties_receive_half_credit():
    x = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    scores = pair_scores(x, x, x)
    for value in scores.values():
        torch.testing.assert_close(value, torch.full((2,), 0.5, dtype=torch.float64))


def test_paired_perturbation_interval_has_common_native_denominator():
    baseline = torch.ones(20, dtype=torch.float64)
    before = torch.full((20,), 0.2, dtype=torch.float64)
    after = torch.full((20,), 0.3, dtype=torch.float64)
    low, high = paired_change_interval(before, after, baseline, repeats=50)
    assert abs(low + 0.1) < 1e-12 and abs(high + 0.1) < 1e-12


def test_result_finiteness_audit_does_not_reject_negative_r2():
    finite_tree({"r2": -100.0, "invalid": None, "values": [0.0, 1.0]})
    import pytest

    with pytest.raises(AssertionError):
        finite_tree({"r2": float("nan")})
