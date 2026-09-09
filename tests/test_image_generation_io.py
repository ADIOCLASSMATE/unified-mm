import pytest
import torch

from utils.image_generation_io import load_model_state


@pytest.mark.parametrize("damage", ["missing", "unexpected", "shape", "empty", "not_tensor"])
def test_full_model_state_rejects_incomplete_input_before_mutating_model(tmp_path, damage):
    model = torch.nn.Linear(2, 2)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    state = {key: torch.full_like(value, 7) for key, value in before.items()}
    if damage == "missing":
        state.pop("bias")
    elif damage == "unexpected":
        state["other.weight"] = torch.ones(1)
    elif damage == "shape":
        state["bias"] = torch.ones(3)
    elif damage == "empty":
        state.clear()
    else:
        state["bias"] = "not a tensor"
    path = tmp_path / "model.pt"
    torch.save({"module": state}, path)
    with pytest.raises(ValueError, match="(full-model state|full-model state dictionary)"):
        load_model_state(model, path)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


def test_complete_legacy_checkpoint_restores_every_parameter(tmp_path):
    source, target = torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)
    path = tmp_path / "pytorch_model/mp_rank_00_model_states.pt"
    path.parent.mkdir()
    torch.save({"module": source.state_dict()}, path)
    report = load_model_state(target, tmp_path)
    assert report["missing"] == report["unexpected"] == []
    for key, value in source.state_dict().items():
        torch.testing.assert_close(value, target.state_dict()[key], rtol=0, atol=0)


def test_complete_model_state_allows_omitted_tied_alias(tmp_path):
    model = torch.nn.Module()
    model.embed = torch.nn.Embedding(3, 2)
    model.head = torch.nn.Linear(2, 3, bias=False)
    model.head.weight = model.embed.weight
    path = tmp_path / "tied.pt"
    torch.save({"embed.weight": torch.full((3, 2), 4.)}, path)
    report = load_model_state(model, path)
    assert report["restored_tied_aliases"] == {"head.weight": "embed.weight"}
    assert model.head.weight is model.embed.weight
    torch.testing.assert_close(model.head.weight, torch.full((3, 2), 4.))
