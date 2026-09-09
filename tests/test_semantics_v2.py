from types import SimpleNamespace

import torch

from scripts.analyze_unified_semantics_v2 import (
    calibrated,
    geometry,
    grouped_means,
    pair_accuracy,
    ranks_with_ties,
)
from scripts.audit_unified_semantics_v2 import grouped_retrieval_null
from scripts.probe_unified_semantics_v2 import PROFILES, Collector, make_batch


def test_calibration_does_not_use_test_examples():
    rows = [{"split": "cal"}, {"split": "cal"}, {"split": "test"}]
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [100.0, -100.0]])
    _, mean = calibrated(x, rows)
    x[2] *= 1000
    _, changed_mean = calibrated(x, rows)
    torch.testing.assert_close(mean, torch.tensor([2.0, 3.0]))
    torch.testing.assert_close(mean, changed_mean)


def test_caption_grouping_accepts_variable_counts_and_noncontiguous_ids():
    x = torch.tensor([[1.0], [3.0], [10.0], [5.0]])
    result = grouped_means(x, torch.tensor([8, 8, 2, 8]), torch.tensor([8, 2]))
    torch.testing.assert_close(result, torch.tensor([[3.0], [10.0]]))


def test_pairwise_ties_receive_half_credit():
    assert pair_accuracy(torch.ones(20), torch.ones(20)) == 50.0
    assert pair_accuracy(torch.ones(20), torch.zeros(20)) == 100.0


def test_ranks_average_ties():
    torch.testing.assert_close(
        ranks_with_ties(torch.tensor([2.0, 1.0, 2.0, 4.0])),
        torch.tensor([2.5, 1.0, 2.5, 4.0]),
    )


def test_cka_is_not_a_test_of_identical_coordinates():
    gen = torch.Generator().manual_seed(42)
    x = torch.randn(40, 8, generator=gen)
    q, _ = torch.linalg.qr(torch.randn(8, 8, generator=gen))
    result = geometry(x, x @ q + 5, permutations=9)
    assert result["linear_cka"] > 0.9999
    assert result["permutation_cka_mean"] < 0.6


def test_retrieval_permutation_preserves_variable_caption_groups():
    image_groups = torch.tensor([17, 81, 4])
    text_groups = torch.tensor([17, 81, 81, 4, 4, 4])
    scores = image_groups[:, None].eq(text_groups[None, :]).float()
    control = grouped_retrieval_null(scores, image_groups, text_groups, repeats=99)
    for direction in ("i2t", "t2i"):
        assert control[direction]["observed_r1"] == 100.0
        assert 15 < control[direction]["shuffled_mean_r1"] < 55


class Tokenizer:
    eos_token_id = 7

    def encode(self, text, add_special_tokens=False):
        return list(range(10, 10 + len(text.split())))


class Cache:
    def sample(self, image_id):
        return torch.full((256, 16), float(image_id))


def dense_attention(**kwargs):
    sigma, segments = kwargs["sigma"], kwargs["segment_ids"]
    valid = segments.ge(0)
    allowed = valid[:, :, None] & valid[:, None, :]
    strict = allowed & (sigma[:, None, :] < sigma[:, :, None])
    diagonal = allowed & (sigma[:, None, :] <= sigma[:, :, None])
    return ~strict[:, None], ~diagonal[:, None]


def batch(rows, modality, profile, monkeypatch):
    import utils.evaluation.multimodal_likelihood as likelihood

    monkeypatch.setattr(likelihood, "build_attention_masks", dense_attention)
    cfg = SimpleNamespace(
        boi_token_id=2,
        eoi_token_id=3,
        image_mask_token_id=4,
        dual_stream_attention_contract="xlnet_content_diagonal",
    )
    return make_batch(
        rows,
        modality,
        Tokenizer(),
        cfg,
        Cache(),
        torch.device("cpu"),
        20260907,
        PROFILES[profile],
    )


def test_native_image_query_has_no_visible_target_latents(monkeypatch):
    rows = [{"image_id": 1, "text": "a red cat"}, {"image_id": 937, "text": "the dog"}]
    values, mask, last, query = batch(rows, "text", "native", monkeypatch)
    local_positions = []
    for i in range(2):
        image_positions = values["token_types"][i].eq(1)
        assert values["attention_mask"][i, 0, query[i], image_positions].all()
        assert not values["image_latent_mask"][i].any()
        assert values["token_types"][i, query[i]] == 1
        assert mask[i, last[i]]
        local_positions.append(int(query[i] - values["image_span_table"][i, 2]))
    assert local_positions[0] == local_positions[1]


def test_content_last_sigma_sees_all_image_data(monkeypatch):
    values, mask, last, query = batch([{"image_id": 1}], "image", "bare", monkeypatch)
    assert int(mask.sum()) == 256
    assert not values["content_attention_mask"][0, 0, last[0], mask[0]].any()
    assert not values["attention_mask"][0, 0, query[0], mask[0]].any()
    assert values["attention_mask"][0, 0, query[0], query[0]]
    assert values["token_types"][0, query[0]] == 0


class Block(torch.nn.Module):
    def forward(self, x0, xt):
        return x0 + 0.1, xt + x0.mean(1, keepdim=True)


class Backbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([Block(), Block()])
        self.norm = torch.nn.Identity()

    def forward(self, x0, xt):
        for layer in self.layers:
            x0, xt = layer(x0, xt)
        return x0, xt


def test_mask_intervention_hook_changes_only_xt_and_preserves_x0():
    backbone = Backbone()
    collector = Collector(backbone)
    x0, xt = torch.randn(2, 4, 8), torch.zeros(2, 4, 8)
    mask = torch.tensor([[True, True, True, False]] * 2)
    last, query = torch.tensor([2, 2]), torch.tensor([3, 3])
    collector.begin(mask, last, query, audit=True)
    backbone(x0, xt)
    normal, reference = collector.finish(), collector.x0_values
    collector.begin(mask, last, query, torch.ones(8), audit=True, reference=reference)
    backbone(x0, xt)
    changed = collector.finish()
    assert torch.equal(normal[:, :, :2], changed[:, :, :2])
    assert not torch.equal(normal[:, :, 2], changed[:, :, 2])
    assert xt.eq(0).all()
