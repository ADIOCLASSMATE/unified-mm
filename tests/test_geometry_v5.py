import pytest
import torch

from scripts.analyze_unified_geometry_v3 import geometry_statistics
from scripts.bootstrap_geometry_v5 import endpoint_plan, knn_anchors
from scripts.check_geometry_v5_batch_layers import layer_comparison
from scripts.compare_geometry_v5 import compare_endpoints, paired_mean, paired_r2
from scripts.explain_geometry_v5_perturbations import residual_decomposition
from scripts.extract_geometry_v5_flow import eligible_indices
from scripts.geometry_v5_flow import FlowAdapter, MaskCollector, pack, readout_masks
from scripts.geometry_v5_math import fit_pair, rotation
from scripts.geometry_v5_protocol import canonical_mode, relative_layer_pairs
from scripts.prepare_geometry_v5_views import grouped


def test_fp32_storage_and_input_pooling_preserve_fixed_noise_across_padding():
    gen = torch.Generator().manual_seed(191)
    noise = torch.randn(9, 7, generator=gen)
    sequences = [torch.cat((torch.randn(n, 7, generator=gen), noise)) for n in (2, 5)]
    masks = [readout_masks(n + 9, range(n), slots=range(n, n + 9)) for n in (2, 5)]
    values, pooled, _ = pack(sequences, masks, "cpu")
    collector = MaskCollector(
        torch.nn.ModuleList([torch.nn.Identity()]), storage_dtype=torch.float32
    )
    collector.stable_input_pooling = True
    collector.begin_masks(pooled)
    collector.collect(values)
    collector.collect(values)
    result = collector.finish(values)
    assert result.dtype == torch.float32 and result.shape == (2, 3, 4, 7)
    assert torch.equal(result[0, 0, 2:], result[1, 0, 2:])
    assert torch.equal(result[0, 0, 2], noise.mean(0))


def test_grouped_raw_before_normalizing_and_arbitrary_shape():
    rows = [{"group": 2}, {"group": 1}, {"group": 2}]
    values = torch.tensor([[3.0, 0.0], [5.0, 7.0], [0.0, 1.0]])
    assert torch.equal(
        grouped(values, rows, [1, 2], lambda row: True),
        torch.tensor([[5.0, 7.0], [1.5, 0.5]]),
    )
    with pytest.raises(AssertionError):
        grouped(values, rows, [3], lambda row: True)


def test_modes_fail_closed_and_spherical_alias():
    assert canonical_mode("unit_sphere") == "centered_unit_sphere"
    with pytest.raises(KeyError):
        canonical_mode("sphere_typo")


def test_relative_depth_covers_both_encoders_without_selecting_scores():
    x = [str(i) for i in range(13)] + ["final_norm"]
    y = [str(i) for i in range(29)] + ["final_norm"]
    pairs = relative_layer_pairs(x, y)
    assert len(pairs) == 30
    assert pairs[14]["index_x"] == 6 and pairs[14]["index_y"] == 14
    assert pairs[-2]["index_x"] == 12 and pairs[-1]["index_x"] == 13
    assert pairs[-1]["kind"] == "final_norm"


def test_content_masks_exclude_prompt_noise_and_markers():
    masks = readout_masks(20, [2, 3], slots=range(6, 15))
    assert masks.sum(1).tolist() == [2, 1, 9, 1]
    assert masks[3].nonzero().flatten().tolist() == [10]
    assert not bool(masks[:, :2].any())
    assert not bool(masks[:2, 6:].any())


def test_omni_mask_is_causal_outside_image_and_padding_hidden():
    sequences = [torch.ones(8, 2), torch.ones(6, 2)]
    masks = [readout_masks(8, [1, 2], boundary=7), readout_masks(6, [1], boundary=5)]
    embeds, pooled, attention = pack(sequences, masks, "cpu", [(3, 3), (2, 2)])
    assert embeds.shape == (2, 8, 2) and pooled.shape == (2, 3, 8)
    assert attention[0, 0, 3, 5] == 0  # image bidirectional
    assert torch.isneginf(attention[0, 0, 2, 5])  # text cannot see future image
    assert torch.isneginf(attention[1, :, :, 6:]).all()  # padded keys never visible
    assert not torch.isneginf(
        attention[1, 0, 7, :6]
    ).all()  # no fully masked padding row


def test_noise_is_fixed_across_semantic_items_and_seed_specific():
    adapter = FlowAdapter.__new__(FlowAdapter)
    adapter.device, adapter.dtype, adapter.latent_shape = (
        "cpu",
        torch.bfloat16,
        (2, 3, 3),
    )
    a, b = adapter.noise(3, 7), adapter.noise(1, 7)
    assert torch.equal(a[0], a[1]) and torch.equal(a[:1], b)
    assert not torch.equal(b, adapter.noise(1, 8))


def test_calibration_and_robustness_selection_cannot_consume_test_scores():
    rows = [
        {"split": "fit", "robust": False},
        {"split": "cal", "robust": True},
        {"split": "test", "robust": True},
    ]
    assert eligible_indices(rows, "coco_images", "main", True, 32) == [1]
    assert eligible_indices(rows, "aro_images", "main", True, 32) == []
    assert eligible_indices(rows, "coco_images", "seed1", False, 32) == [1, 2]


def test_different_widths_cannot_be_full_rotation_but_share_subspaces():
    generator = torch.Generator().manual_seed(11)
    z = torch.randn(100, 4, generator=generator, dtype=torch.float64)
    ax = torch.linalg.qr(torch.randn(12, 4, generator=generator, dtype=torch.float64)).Q
    ay = torch.linalg.qr(torch.randn(16, 4, generator=generator, dtype=torch.float64)).Q
    x, y = z @ ax.T, z @ ay.T
    assert fit_pair(x, y, "full") is None
    fitted = fit_pair(x, y, 4)
    assert fitted["rotation_identified"]
    assert torch.allclose(
        fitted["fx"]["fit"] @ fitted["q"], fitted["fy"]["fit"], atol=1e-10
    )


def test_cross_covariance_rank_not_implied_by_each_side_rank():
    x = torch.zeros(20, 4, dtype=torch.float64)
    y = torch.zeros_like(x)
    x[:4] = torch.eye(4)
    y[:2, :2] = torch.eye(2)
    y[4:6, 2:] = torch.eye(2)
    assert torch.linalg.matrix_rank(x) == torch.linalg.matrix_rank(y) == 4
    _, _, info = rotation(x, y)
    assert info["cross_covariance_rank"] == 2


def test_knn_anchor_bootstrap_uses_original_candidate_graph():
    g = torch.Generator().manual_seed(73)
    x, y = (torch.randn(32, 6, generator=g, dtype=torch.float64) for _ in range(2))
    anchors = knn_anchors(x, y)
    reference = geometry_statistics(x, y, permutations=0)
    for key, values in anchors.items():
        assert len(values) == 32
        assert float(values.mean()) == pytest.approx(reference["scores"][key])
    assert knn_anchors(x, torch.ones_like(y)) == {}


def test_paired_r2_identity_and_different_denominators():
    a = {
        "groups": torch.tensor([4, 7, 9]),
        "target_family": "coco",
        "errors": torch.tensor([1.0, 2.0, 3.0]),
        "shuffled_errors": torch.tensor([4.0, 4.0, 4.0]),
        "baseline": torch.tensor([4.0, 4.0, 4.0]),
    }
    b = {
        **a,
        "errors": torch.tensor([2.0, 4.0, 6.0]),
        "baseline": torch.tensor([4.0, 4.0, 4.0]),
    }
    stats = paired_r2(a, b, repeats=100)
    assert stats["a_r2"] == 0.5 and stats["b_r2"] == 0.0
    assert stats["a_minus_b_r2"] == 0.5
    assert paired_r2(a, a, repeats=100)["a_minus_b_r2_95"] == [0.0, 0.0]
    b["groups"] = torch.tensor([7, 4, 9])
    with pytest.raises(AssertionError):
        paired_r2(a, b, repeats=100)
    result = paired_mean(torch.ones(6), torch.zeros(6), repeats=100)
    assert result["a_minus_b_95"] == [1.0, 1.0]


def test_paired_r2_uses_each_models_own_denominator():
    a = {
        "groups": torch.tensor([4, 7, 9]),
        "target_family": "coco",
        "errors": torch.tensor([1.0, 2.0, 3.0]),
        "shuffled_errors": torch.tensor([4.0, 5.0, 6.0]),
        "baseline": torch.tensor([4.0, 8.0, 12.0]),
    }
    b = {
        **a,
        **{k: a[k] * 8 for k in ("errors", "shuffled_errors", "baseline")},
    }
    result = paired_r2(a, b, repeats=100)
    assert result["a_r2"] == result["b_r2"] == 0.75
    assert result["a_minus_b_r2_95"] == [0.0, 0.0]
    assert result["a_minus_b_advantage_over_shuffled_fit_95"] == [0.0, 0.0]


def test_primary_contrast_excludes_prompt_and_native_pooler_sensitivities(monkeypatch):
    values = {
        "groups": torch.tensor([4, 7, 9]),
        "target_family": "imagenet",
        "errors": torch.ones(3),
        "shuffled_errors": torch.ones(3) * 2,
        "baseline": torch.ones(3) * 4,
    }
    monkeypatch.setattr(
        "scripts.compare_geometry_v5.load_arrays",
        lambda path: {"test": values, "transfer_test": values},
    )
    a = {
        "setting": "b_native",
        "endpoint": "fixed",
        "readout": "content_mean",
        "family": "imagenet",
        "mode": "centered_euclidean",
        "requested_dimension": 32,
        "dimension": 32,
        "fit_points": 600,
        "pair": {"kind": "final_norm"},
        "sample_statistics": "synthetic",
    }
    b = {**a, "setting": "f_native"}
    assert compare_endpoints(a, b, ["synthetic", "synthetic"])["primary_contrast"]
    for setting in ("b_bare", "b_neutral"):
        assert not compare_endpoints(a, {**b, "setting": setting}, ["a", "b"])[
            "primary_contrast"
        ]
    native = {**b, "setting": "siglip_native", "readout": "native_endpoint"}
    result = compare_endpoints(a, native, ["a", "native"])
    assert not result["primary_contrast"] and not result["common_content_readout"]


def test_endpoint_selection_excludes_full_and_never_uses_test():
    rows = []
    for layer in range(2):
        families = {}
        for family, sizes in (("imagenet", (600,)), ("coco", (512, 2048, 8192))):
            modes = {}
            for mode in ("centered_euclidean", "unit_sphere"):
                mappings = {}
                for n in sizes:
                    for dimension in (32, 128, 512, "full"):
                        mappings[f"fit{n}-dim{dimension}"] = {
                            "valid": True,
                            "fit_points": n,
                            "requested_dimension": dimension,
                            "dimension": 1024 if dimension == "full" else dimension,
                            "dev": {
                                "paired": {"r2": 999 if dimension == "full" else layer}
                            },
                            "test": {"paired": {"r2": 999 * (1 - layer)}},
                        }
                modes[mode] = {"mappings": mappings}
            families[family] = modes
        rows.append(
            {
                "pair_index": layer,
                "pair": {"kind": "final_norm" if layer else "input"},
                "readout": "content_mean",
                "families": families,
            }
        )
    layout = {"layer_pairs": [0, 1], "readouts": ["content_mean"]}
    plans = endpoint_plan(rows, layout)
    assert len(plans) == 40
    chosen = [plan for plan in plans if plan["endpoint"] == "dev_selected"]
    assert len(chosen) == 8
    assert all(
        plan["row"]["pair_index"] == 1 and plan["mapping"].endswith("dim32")
        for plan in chosen
    )
    with pytest.raises(AssertionError):
        endpoint_plan(rows[:1], layout)


def test_raw_cosine_can_hide_error_relative_to_semantic_variation():
    g = torch.Generator().manual_seed(3)
    singles = 1000 + torch.randn(32, 3, 2, 16, generator=g)
    batch = singles + 3 * torch.randn(singles.shape, generator=g)
    result = layer_comparison(batch, singles)
    assert result["formal_flattened_threshold_passed"]
    assert (
        max(v for row in result["cal_semantic_relative_rms_by_layer_pool"] for v in row)
        > 2
    )
    exact = layer_comparison(singles, singles)
    assert all(
        v == 0 for row in exact["cal_semantic_relative_rms_by_layer_pool"] for v in row
    )


def test_residual_error_decomposition_separates_translation_from_deformation():
    g = torch.Generator().manual_seed(161)
    before = torch.randn(50, 6, generator=g, dtype=torch.float64)
    mean, centered = residual_decomposition(before, before + 2.0)
    assert abs(centered) < 1e-10
    assert mean == pytest.approx(
        float((before + 2).square().sum() - before.square().sum())
    )
    zero_mean = before - before.mean(0)
    mean, centered = residual_decomposition(zero_mean, zero_mean * 2)
    assert abs(mean) < 1e-10
    assert centered == pytest.approx(float(3 * zero_mean.square().sum()))
