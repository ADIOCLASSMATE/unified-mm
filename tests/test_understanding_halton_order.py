import os
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import pytest
import torch
from utils.evaluation.multimodal_likelihood import build_image_sigma, image_order_mc_seed, encode_candidate_mc
from utils.image_token_order import halton_image_positions
from models.modeling_model.modeling_selfless_generation import SelflessGenerationMixin


@pytest.mark.parametrize("side", [2, 4, 6, 16])
def test_understanding_sigma_is_inverse_of_generation_reveal_order(side):
    count = side * side
    positions = SelflessGenerationMixin._halton_image_order(count, side, torch.device("cpu"))
    ranks = build_image_sigma(count, order="spatial_halton", seed=42)
    assert sorted(ranks) == list(range(count))
    assert [ranks[p] for p in positions.tolist()] == list(range(count))
    assert ranks == build_image_sigma(count, order="spatial_halton", seed=1234)


def test_halton_known_prefix_preserves_spatial_coordinates():
    assert halton_image_positions(256, 16)[:5] == (133, 74, 193, 39, 172)
    with pytest.raises(ValueError, match="square"):
        build_image_sigma(6, order="spatial_halton", seed=42)


def test_random_order_is_unchanged_and_reproducible():
    generator = torch.Generator().manual_seed(42)
    expected = torch.rand(256, generator=generator).argsort().tolist()
    assert build_image_sigma(256, order="random", seed=42) == expected


def test_shifted_halton_mc64_has_distinct_reproducible_complete_orders():
    orders = [build_image_sigma(256, order="spatial_halton_shifted",
                               seed=image_order_mc_seed(424242, 6000000000, i)) for i in range(64)]
    assert len(set(map(tuple, orders))) == 64
    assert all(sorted(order) == list(range(256)) for order in orders)
    assert orders[0] == build_image_sigma(256, order="spatial_halton_shifted",
                                         seed=image_order_mc_seed(424242, 6000000000, 0))
    assert orders[0] != build_image_sigma(256, order="spatial_halton", seed=42)


def test_shifted_halton_candidate_texts_share_the_same_mc_orders():
    from test_multimodal_likelihood_benchmarks import CharacterTokenizer, example
    kwargs = dict(image_tokens=16, boi_token_id=101, eoi_token_id=102,
                  image_mask_token_id=103, max_length=128,
                  image_sigma_order="spatial_halton_shifted", seed=424242, mc_samples=64)
    a = encode_candidate_mc(CharacterTokenizer(), example(), 0, **kwargs)
    b = encode_candidate_mc(CharacterTokenizer(), example(), 1, **kwargs)
    for x, y in zip(a, b):
        assert x.sigma[x.image_start:x.image_start+16] == y.sigma[y.image_start:y.image_start+16]
