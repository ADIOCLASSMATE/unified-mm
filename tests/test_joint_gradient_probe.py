import math

import pytest
import torch
from torch import nn

from utils.joint_gradient_probe import (
    build_lambda_text_candidates,
    checkpoint_bundle_sha256,
    measure_gradient_probe_batch,
    optimizer_parameter_role,
    summarize_probe_batches,
)
from utils.selfless_flow_optimizer import learning_rate_for_parameter


def test_checkpoint_bundle_sha256_is_order_invariant_and_content_sensitive(tmp_path):
    first = tmp_path / "a.bin"
    second = tmp_path / "b.bin"
    first.write_bytes(b"first")
    second.write_bytes(b"second")

    digest = checkpoint_bundle_sha256([second, first], root=tmp_path)
    assert digest == checkpoint_bundle_sha256([first, second], root=tmp_path)

    second.write_bytes(b"changed")
    assert digest != checkpoint_bundle_sha256([first, second], root=tmp_path)


class _Body(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(8, 4)
        self.layers = nn.ModuleList([nn.Linear(4, 4, bias=False)])
        self.image_token_embedder = nn.Linear(4, 4, bias=False)


class _ToyJointModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _Body()
        self.lm_head = nn.Linear(4, 8, bias=False)
        self.lm_head.weight = self.model.embed_tokens.weight
        self.image_flow_condition_proj = nn.Linear(4, 4, bias=False)
        self.image_flow_head = nn.Linear(4, 4, bias=False)

    def losses(self):
        tokens = torch.tensor([1, 2, 3])
        hidden = self.model.layers[0](self.model.embed_tokens(tokens)).mean(0)
        text = self.lm_head(hidden).square().mean()
        image_input = self.model.image_token_embedder(hidden)
        image = self.image_flow_head(
            self.image_flow_condition_proj(image_input)
        ).square().mean()
        counts = {
            "text_tokens": torch.tensor(3),
            "image_tokens": torch.tensor(4),
        }
        return {"text_loss": text, "image_loss": image}, counts


def test_optimizer_roles_cover_tied_head_projector_flow_and_backbone():
    expected = {
        "model.layers.0.weight": "backbone",
        "model.embed_tokens.weight": "tied_lm_head_embedding",
        "lm_head.weight": "tied_lm_head_embedding",
        "model.image_token_embedder.z_proj.weight": "image_projector",
        "image_flow_condition_proj.weight": "image_projector",
        "image_flow_head.net.input_proj.weight": "flow_head",
    }
    assert {name: optimizer_parameter_role(name) for name in expected} == expected
    assert learning_rate_for_parameter(
        "lm_head.weight",
        backbone_lr=1e-5,
        flow_lr=4e-5,
        projector_lr=4e-5,
        special_token_lr=1e-5,
    ) == 1e-5


def test_probe_runs_separate_backwards_and_clears_gradients():
    torch.manual_seed(7)
    model = _ToyJointModel()
    row = measure_gradient_probe_batch(
        model,
        model.losses,
        special_token_ids=[1, 2],
        reset_seed=lambda: torch.manual_seed(11),
    )
    assert row["text_tokens"] == 3
    assert row["image_tokens"] == 4
    assert row["shared_backbone"]["g_text"] > 0
    assert row["shared_backbone"]["g_image"] > 0
    assert -1.0 <= row["shared_backbone"]["cosine"] <= 1.0
    assert row["gradient_norms"]["lm_head"]["text"] > 0
    assert row["gradient_norms"]["special_token_embedding"]["text"] > 0
    assert all(parameter.grad is None for parameter in model.parameters())


def test_probe_summary_and_lambda_candidates_cover_reference_points():
    template = {
        "loss_text_unweighted": 2.0,
        "loss_image_unweighted": 0.5,
        "shared_backbone": {
            "g_text": 2.0,
            "g_image": 0.2,
            "ratio_g_image_over_g_text": 0.1,
            "cosine": -0.25,
        },
        "gradient_norms": {
            group: {"text": 1.0, "image": 2.0}
            for group in (
                "backbone",
                "lm_head",
                "special_token_embedding",
                "image_projector",
                "flow_head",
            )
        },
    }
    summary = summarize_probe_batches([template] * 16)
    assert summary["ratio_g_image_over_g_text"]["median"] == pytest.approx(0.1)
    assert summary["task_conflict"]["persistent_negative"] is True
    assert summary["lambda_text"]["center"] == pytest.approx(0.1)
    assert summary["lambda_text"]["candidates"] == [0.05, 0.1, 0.2]

    clipped = build_lambda_text_candidates(10.0)
    assert clipped["center"] == 0.4
    assert all(math.isfinite(value) for value in clipped["candidates"])
    assert {0.05, 0.1, 0.2}.issubset(clipped["candidates"])


def test_formal_summary_rejects_too_few_batches():
    with pytest.raises(ValueError, match="16-32"):
        summarize_probe_batches([])
