from types import SimpleNamespace

import torch

from scripts.evaluate_imagenet1k_i2t_clip import (
    build_i2t_prefix,
    generate_batch,
    select_balanced_validation_indices,
)


class _Tokenizer:
    eos_token_id = 9

    def encode(self, text, add_special_tokens=False):
        assert text == "describe"
        assert add_special_tokens is False
        return [10, 11]

    def decode(self, tokens, skip_special_tokens=True):
        assert skip_special_tokens is True
        return " ".join(f"tok-{token}" for token in tokens)


def test_i2t_prefix_orders_eoi_before_image_and_caption_queries():
    input_ids, token_types, sigma, image_start = build_i2t_prefix(
        _Tokenizer(),
        text_prefix="describe",
        boi_token_id=20,
        eoi_token_id=21,
        image_mask_token_id=22,
        image_tokens=4,
    )

    assert input_ids.tolist() == [10, 11, 20, 22, 22, 22, 22, 21]
    assert token_types.tolist() == [0, 0, 2, 1, 1, 1, 1, 2]
    assert image_start == 3
    assert sigma.tolist() == [0.0, 1.0, 2.0, 4.0, 5.0, 6.0, 7.0, 3.0]
    assert float(sigma[image_start : image_start + 4].min()) > float(sigma[-1])


class _Backbone:
    def __call__(self, X0_input_ids, **kwargs):
        del kwargs
        batch, length = X0_input_ids.shape
        return SimpleNamespace(
            last_hidden_state=torch.zeros(batch, length, 1)
        )


class _Model:
    def __init__(self):
        self.config = SimpleNamespace(
            boi_token_id=20,
            eoi_token_id=21,
            image_mask_token_id=22,
            mask_token_id=23,
            im_end_token_id=None,
        )
        self.model = _Backbone()
        self.calls = 0

    def lm_head(self, hidden):
        logits = torch.full((hidden.shape[0], 32), -100.0)
        logits[:, 7 if self.calls == 0 else 9] = 100.0
        self.calls += 1
        return logits


def test_i2t_generation_stops_on_eos_and_decodes_only_caption_tokens():
    texts, token_ids, reasons = generate_batch(
        _Model(),
        _Tokenizer(),
        torch.zeros(2, 4, 3),
        text_prefix="describe",
        max_new_tokens=4,
        temperature=0.0,
        device=torch.device("cpu"),
    )

    assert token_ids == [[7], [7]]
    assert texts == ["tok-7", "tok-7"]
    assert reasons == ["eos", "eos"]


def test_balanced_holdout_selection_round_robins_synsets():
    dataset = SimpleNamespace(
        img_ids=torch.tensor([1, 2, 3, 4, 5, 6]),
        synsets={1: "b", 2: "a", 3: "b", 4: "a", 5: "b", 6: "a"},
    )

    selected = select_balanced_validation_indices(
        dataset,
        [4, 2, 0, 5, 1, 3],
        4,
    )

    assert selected == [5, 4, 1, 2]
