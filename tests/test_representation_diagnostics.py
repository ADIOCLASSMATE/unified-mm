import torch

from scripts.analyze_unified_representations import retrieval, ridge_fit, ridge_predict


def test_retrieval_accepts_all_reference_captions():
    images = torch.eye(20)
    captions = images.repeat_interleave(5, 0)
    result = retrieval(
        images, captions, torch.arange(20), torch.arange(20).repeat_interleave(5)
    )
    assert result["i2t_r1"] == 100.0
    assert result["t2i_r1"] == 100.0


def test_collapsed_features_do_not_report_perfect_retrieval():
    images, captions = torch.ones(20, 8), torch.ones(100, 8)
    result = retrieval(
        images, captions, torch.arange(20), torch.arange(20).repeat_interleave(5)
    )
    assert abs(result["i2t_r1"] - 5.0) < 1e-5
    assert abs(result["t2i_r1"] - 5.0) < 1e-5
    assert abs(result["cosine_margin"]) < 1e-6


def test_ridge_recovers_a_held_out_linear_relation():
    generator = torch.Generator().manual_seed(17)
    train = torch.randn(200, 16, generator=generator)
    test = torch.randn(40, 16, generator=generator)
    weight = torch.randn(16, 12, generator=generator)
    bias = torch.randn(12, generator=generator)
    fitted = ridge_fit(train, train @ weight + bias, strength=1e-6)
    torch.testing.assert_close(
        ridge_predict(fitted, test), test @ weight + bias, atol=1e-4, rtol=1e-4
    )
