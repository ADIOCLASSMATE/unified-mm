from types import SimpleNamespace

import torch

from pretrain.train_selfless_flow import (
    _build_i2t_generation_prefix,
    _generate_i2t_caption_batch,
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


class _Model:
    def __init__(self):
        self.config = SimpleNamespace(
            boi_token_id=20,
            eoi_token_id=21,
            image_mask_token_id=22,
            mask_token_id=23,
            im_end_token_id=None,
        )
        self.calls = []

    def generate(self, task, **kwargs):
        self.calls.append((task, kwargs))
        input_ids = kwargs["input_ids"]
        suffix = torch.tensor(
            [[7, 9]] * input_ids.shape[0],
            device=input_ids.device,
            dtype=torch.long,
        )
        return torch.cat([input_ids, suffix], dim=1), {
            "backbone_kv_cache_enabled": kwargs["use_cache"],
        }


def test_i2t_qualitative_prefix_places_complete_image_before_caption_query():
    input_ids, token_types, sigma, image_start = (
        _build_i2t_generation_prefix(
            _Tokenizer(),
            text_prefix="describe",
            boi_token_id=20,
            eoi_token_id=21,
            image_mask_token_id=22,
            image_tokens=4,
        )
    )

    assert input_ids.tolist() == [10, 11, 20, 22, 22, 22, 22, 21]
    assert token_types.tolist() == [0, 0, 2, 1, 1, 1, 1, 2]
    assert image_start == 3
    assert sigma.tolist() == [0.0, 1.0, 2.0, 4.0, 5.0, 6.0, 7.0, 3.0]


def test_i2t_qualitative_generation_stops_on_eos_and_decodes_caption_only():
    texts, token_ids, reasons = _generate_i2t_caption_batch(
        _Model(),
        _Tokenizer(),
        torch.zeros(2, 4, 3),
        text_prefix="describe",
        max_new_tokens=4,
        temperature=0.0,
    )

    assert token_ids == [[7], [7]]
    assert texts == ["tok-7", "tok-7"]
    assert reasons == ["eos", "eos"]


def test_i2t_qualitative_generation_delegates_to_unified_cached_api():
    model = _Model()
    _generate_i2t_caption_batch(
        model,
        _Tokenizer(),
        torch.zeros(1, 4, 3),
        text_prefix="describe",
        max_new_tokens=1,
        temperature=0.0,
    )
    task, kwargs = model.calls[0]
    assert task == "i2t"
    assert kwargs["use_cache"] is True
    assert kwargs["return_trace"] is True
    assert "attention_mask" not in kwargs
