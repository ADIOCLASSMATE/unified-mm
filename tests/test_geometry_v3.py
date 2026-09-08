import torch

from scripts.analyze_unified_geometry_v3 import (
    alignment_experiment,
    apply_frame,
    average_groups,
    error_metrics,
    fitting_frame,
    geometry_statistics,
    preprocess,
    ranks_with_ties,
)


def paired_bundles(dimension=16):
    gen = torch.Generator().manual_seed(71)
    q, _ = torch.linalg.qr(
        torch.randn(dimension, dimension, dtype=torch.float64, generator=gen)
    )
    bias = torch.randn(dimension, dtype=torch.float64, generator=gen)
    source, target = {}, {}
    for bundle, shift in ((source, 0), (target, 3)):
        bundle.update(cal_x=torch.zeros(dimension), cal_y=bias)
        for split, count in (("fit", 100), ("dev", 30), ("test", 50)):
            x = (
                torch.randn(count, dimension, dtype=torch.float64, generator=gen)
                + shift
            )
            bundle[f"{split}_x"] = x
            bundle[f"{split}_y"] = 2.5 * x @ q + bias
    return source, target


def test_exact_similarity_is_recovered_on_unseen_and_shifted_domain_points():
    source, target = paired_bundles()
    result = alignment_experiment(source, target, "centered_euclidean", "full")
    assert result["valid"]
    for split in ("fit", "dev", "test", "transfer_test"):
        assert result[split]["paired"]["nrmse"] < 1e-10
    assert result["test"]["shuffled_fit"]["nrmse"] > 0.5


def test_geometry_is_invariant_to_rotation_translation_and_global_scale():
    source, _ = paired_bundles()
    result = geometry_statistics(source["test_x"], source["test_y"], permutations=9)
    for score in result["scores"].values():
        assert score > 0.999999
    assert result["null_summary"]["knn_10"]["mean"] < 0.5


def test_nonuniform_stretch_is_not_accepted_as_rotation():
    source, target = paired_bundles()
    stretch = torch.diag(torch.linspace(0.1, 5, 16, dtype=torch.float64))
    for bundle in (source, target):
        bundle["cal_y"] = torch.zeros(16)
        for split in ("fit", "dev", "test"):
            bundle[f"{split}_y"] = bundle[f"{split}_x"] @ stretch
    result = alignment_experiment(source, target, "centered_euclidean", "full")
    assert result["test"]["paired"]["nrmse"] > 0.3


def test_constant_representations_are_invalid_not_perfectly_aligned():
    result = geometry_statistics(torch.ones(40, 8), torch.ones(40, 8), permutations=9)
    assert not result["valid"]
    assert fitting_frame(torch.ones(100, 16), "full") is None


def test_fit_frame_has_no_access_to_test_data():
    source, _ = paired_bundles()
    fit = preprocess(source["fit_x"], source["cal_x"], "centered_euclidean")
    frame = fitting_frame(fit, 8)
    apply_frame(frame, source["test_x"])
    first = frame["projection"].clone()
    apply_frame(frame, source["test_x"] * 1000)
    torch.testing.assert_close(first, frame["projection"])
    assert 0 < frame["retained_variance_fit"] < 1


def test_spherical_and_euclidean_preprocessing_are_distinct():
    x = torch.tensor([[1.0, 0.0], [10.0, 0.0]])
    cal = torch.zeros(2)
    assert not torch.equal(
        preprocess(x, cal, "centered_euclidean")[0],
        preprocess(x, cal, "centered_euclidean")[1],
    )
    torch.testing.assert_close(
        preprocess(x, cal, "centered_unit_sphere")[0],
        preprocess(x, cal, "centered_unit_sphere")[1],
    )


def test_error_r2_keeps_negative_values():
    result, _, _ = error_metrics(-torch.eye(5), torch.eye(5))
    assert result["r2"] == -3
    assert result["nrmse"] == 2


def test_tied_ranks_preserve_float64_dtype():
    actual = ranks_with_ties(torch.tensor([4, 1, 1, 9], dtype=torch.float64))
    torch.testing.assert_close(
        actual, torch.tensor([3, 1.5, 1.5, 4], dtype=torch.float64)
    )


def test_group_prototypes_average_raw_features_and_preserve_requested_order():
    features = torch.zeros(4, 30, 4, 3, dtype=torch.bfloat16)
    features[0, :, 0, 0] = 2
    features[1, :, 0, 1] = 10
    features[2, :, 2, 2] = 7
    features[3] = 100
    rows = [
        {"group": 11, "split": "fit"},
        {"group": 11, "split": "fit"},
        {"group": 22, "split": "fit"},
        {"group": 22, "split": "test"},
    ]
    result = average_groups(features, rows, [22, 11], lambda r: r["split"] == "fit")
    torch.testing.assert_close(result[0, 0, 1], torch.tensor([0.0, 0.0, 7.0]))
    torch.testing.assert_close(result[1, 0, 0], torch.tensor([1.0, 5.0, 0.0]))
