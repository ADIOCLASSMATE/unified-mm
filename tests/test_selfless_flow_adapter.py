from copy import deepcopy

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from utils.selfless_flow_adapter import load_image_flow_adapter


class _Body(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(10, 4)
        self.layers = nn.ModuleList([nn.Linear(4, 4)])
        self.norm = nn.LayerNorm(4)
        self.image_token_embedder = nn.Linear(4, 4)


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _Body()
        self.lm_head = nn.Linear(4, 10, bias=False)
        self.lm_head.weight = self.model.embed_tokens.weight
        self.image_flow_condition_proj = nn.Linear(4, 4)
        self.image_flow_head = nn.Linear(4, 4)

    @property
    def image_token_embedder(self):
        return self.model.image_token_embedder


def _config():
    return OmegaConf.create(
        {
            "model": {
                "mask_token_id": 6,
                "boi_token_id": 7,
                "eoi_token_id": 8,
                "image_mask_token_id": 9,
            }
        }
    )


def _adapter_state(model):
    def filled(module, value):
        return {
            name: torch.full_like(tensor, value)
            for name, tensor in module.state_dict().items()
        }

    return {
        "image_flow_head": filled(model.image_flow_head, 1.0),
        "image_flow_condition_proj": filled(
            model.image_flow_condition_proj, 2.0
        ),
        "image_token_embedder": filled(model.image_token_embedder, 3.0),
        "special_token_ids": {
            "mask": 6,
            "boi": 7,
            "eoi": 8,
            "image_mask": 9,
        },
        "special_token_embeddings": {
            "mask": torch.full((4,), 4.0),
            "boi": torch.full((4,), 5.0),
            "eoi": torch.full((4,), 6.0),
            "image_mask": torch.full((4,), 7.0),
        },
    }


def test_adapter_only_load_preserves_text_backbone_and_ordinary_vocab(tmp_path):
    torch.manual_seed(11)
    model = _Model()
    backbone_before = deepcopy(model.model.layers.state_dict())
    norm_before = deepcopy(model.model.norm.state_dict())
    ordinary_row_before = model.model.embed_tokens.weight[0].detach().clone()
    path = tmp_path / "adapter.pt"
    torch.save(_adapter_state(model), path)

    report = load_image_flow_adapter(model, path, _config())

    assert report["format"] == "finalized_pt_adapter"
    assert report["loaded_modules"]["image_flow_head"]["numel"] == 20
    assert model.model.layers.state_dict().keys() == backbone_before.keys()
    for name, value in model.model.layers.state_dict().items():
        assert torch.equal(value, backbone_before[name])
    for name, value in model.model.norm.state_dict().items():
        assert torch.equal(value, norm_before[name])
    assert torch.equal(model.model.embed_tokens.weight[0], ordinary_row_before)
    assert torch.equal(model.model.embed_tokens.weight[6], torch.full((4,), 4.0))
    assert torch.equal(model.model.embed_tokens.weight[9], torch.full((4,), 7.0))
    assert torch.equal(model.image_flow_head.weight, torch.ones_like(model.image_flow_head.weight))
    assert torch.equal(
        model.image_flow_condition_proj.weight,
        torch.full_like(model.image_flow_condition_proj.weight, 2.0),
    )
    assert torch.equal(
        model.image_token_embedder.weight,
        torch.full_like(model.image_token_embedder.weight, 3.0),
    )


def test_adapter_only_load_rejects_tokenizer_id_mismatch(tmp_path):
    model = _Model()
    state = _adapter_state(model)
    state["special_token_ids"]["boi"] = 5
    path = tmp_path / "adapter.pt"
    torch.save(state, path)

    with pytest.raises(ValueError, match="special-token ids"):
        load_image_flow_adapter(model, path, _config())
