import inspect

import torch

from utils.showo2_maskgit import (
    build_showo2_allowed_mask,
    build_showo2_position_ids,
    sample_maskgit_training_masks,
)


def test_packed_position_ids_are_1d_and_reset_per_segment():
    token_types = torch.tensor([[2, 0, 1, 1, 2, 0, 1, 1, 3]])
    segment_ids = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1, -1]])
    positions = build_showo2_position_ids(token_types, segment_ids)
    assert positions.tolist() == [[0, 1, 2, 3, 0, 1, 2, 3, 0]]


def test_text_is_causal_with_diagonal_and_image_span_is_bidirectional():
    # special, text, image x3, special, padding
    token_types = torch.tensor([[2, 0, 1, 1, 1, 2, 3]])
    allowed = build_showo2_allowed_mask(token_types)

    assert allowed[0, 1].tolist() == [True, True, False, False, False, False, False]
    assert allowed[0, 2].tolist() == [True, True, True, True, True, False, False]
    assert allowed[0, 4].tolist() == [True, True, True, True, True, False, False]
    assert allowed[0, 5].tolist() == [True, True, True, True, True, True, False]
    assert not allowed[0, 6].any()


def test_packed_segments_cannot_cross_attend():
    token_types = torch.tensor([[2, 0, 1, 1, 2, 0, 1, 1, 3]])
    segment_ids = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1, -1]])
    allowed = build_showo2_allowed_mask(
        token_types,
        segment_ids=segment_ids,
    )

    assert allowed[0, 6, 7]
    assert allowed[0, 7, 6]
    assert allowed[0, 6, 4]
    assert allowed[0, 6, 5]
    assert not allowed[0, 6, 2]
    assert not allowed[0, 2, 6]


def test_image_cfg_dropout_removes_text_context_only_for_image_queries():
    token_types = torch.tensor([[2, 0, 1, 1, 1, 2]])
    allowed = build_showo2_allowed_mask(
        token_types,
        image_uncond_rows=torch.tensor([True]),
    )

    assert allowed[0, 2].tolist() == [False, False, True, True, True, False]
    assert allowed[0, 4].tolist() == [False, False, True, True, True, False]
    assert allowed[0, 1, 0]
    assert allowed[0, 1, 1]


def test_maskgit_training_masks_are_vectorized_and_image_only():
    token_types = torch.tensor(
        [
            [2, 1, 1, 1, 1, 2],
            [2, 1, 1, 1, 1, 2],
        ]
    )
    span_table = torch.tensor([[0, 0, 1, 5], [1, 1, 1, 5]])
    torch.manual_seed(11)
    visible, loss_mask, ratios = sample_maskgit_training_masks(
        token_types,
        span_table,
        image_tokens_per_img=4,
    )

    image_mask = token_types.eq(1)
    assert ratios.shape == (2,)
    assert torch.all(loss_mask.sum(dim=1).ge(1))
    assert not (loss_mask & ~image_mask).any()
    assert torch.equal(visible, image_mask & ~loss_mask)
    assert ".item()" not in inspect.getsource(sample_maskgit_training_masks)


def test_validation_maskgit_mask_is_fixed_and_deterministic():
    token_types = torch.tensor([[2, 1, 1, 1, 1, 2]])
    span_table = torch.tensor([[0, 0, 1, 5]])
    flow_sigma = torch.tensor([[0.0, 4.0, 1.0, 3.0, 2.0, 5.0]])
    first = sample_maskgit_training_masks(
        token_types,
        span_table,
        image_tokens_per_img=4,
        validation_mask_ratio=0.5,
        flow_sigma=flow_sigma,
    )
    torch.manual_seed(999)
    second = sample_maskgit_training_masks(
        token_types,
        span_table,
        image_tokens_per_img=4,
        validation_mask_ratio=0.5,
        flow_sigma=flow_sigma,
    )

    for first_tensor, second_tensor in zip(first, second):
        torch.testing.assert_close(first_tensor, second_tensor)
    assert first[1][0].tolist() == [False, False, True, False, True, False]
