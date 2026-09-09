import json
from types import SimpleNamespace

import pytest

from utils.checkpoint_transaction import (
    prepare_checkpoint, publish_checkpoint, validate_checkpoint_inventory,
    write_checkpoint_inventory,
)

ACCELERATOR = SimpleNamespace(is_main_process=True, num_processes=1)


def _write_payload(directory, *, world_size=2):
    directory.mkdir()
    (directory / "metadata.json").write_text("{}")
    for rank in range(world_size):
        for name in (f"random_states_{rank}.pkl", f"data_state_rank_{rank:05d}.pt"):
            (directory / name).write_bytes(b"saved state")


def test_incomplete_rank_payload_cannot_be_committed(tmp_path):
    _write_payload(tmp_path / "partial", world_size=1)
    with pytest.raises(ValueError, match="random_states_1"):
        write_checkpoint_inventory(tmp_path / "partial", world_size=2, mixed_data=True, ema=False)


@pytest.mark.parametrize("damage", ["missing", "truncated", "missing_inventory"])
def test_resume_rejects_damage_to_published_payload(tmp_path, damage):
    destination = tmp_path / "checkpoint-2"
    staging = prepare_checkpoint(destination, ACCELERATOR, global_step=2)
    _write_payload(staging)
    write_checkpoint_inventory(staging, world_size=2, mixed_data=True, ema=False)
    (staging / "checkpoint_complete.json").write_text(json.dumps({
        "schema": "selfless_caption_checkpoint_complete_v2", "global_step": 2,
    }))
    publish_checkpoint(staging, destination, ACCELERATOR, global_step=2)
    validate_checkpoint_inventory(destination)
    target = destination / "data_state_rank_00001.pt"
    if damage == "missing_inventory":
        (destination / "checkpoint_files.json").unlink()
    elif damage == "missing":
        target.unlink()
    else:
        target.write_bytes(b"bad")
    from pretrain.train_selfless_flow import _validate_checkpoint_complete
    with pytest.raises((ValueError, FileNotFoundError)):
        _validate_checkpoint_complete(destination, expected_global_step=2)


def test_publication_failure_keeps_previous_checkpoint(tmp_path, monkeypatch):
    old = tmp_path / "checkpoint-1"
    old.mkdir()
    (old / "state.bin").write_bytes(b"last complete state")
    destination = tmp_path / "checkpoint-2"
    staging = prepare_checkpoint(destination, ACCELERATOR, global_step=2)
    _write_payload(staging)
    write_checkpoint_inventory(staging, world_size=2, mixed_data=True, ema=False)
    (staging / "checkpoint_complete.json").write_text('{"global_step": 2}')
    def fail(*_):
        raise OSError("injected publish failure")
    monkeypatch.setattr("utils.checkpoint_transaction.os.replace", fail)
    with pytest.raises(OSError, match="injected"):
        publish_checkpoint(staging, destination, ACCELERATOR, global_step=2)
    assert not destination.exists()
    assert (old / "state.bin").read_bytes() == b"last complete state"


@pytest.mark.parametrize("damage", ["missing_metadata", "missing_rank_state", "truncated_rank_state"])
def test_restore_preflight_rejects_damage_before_model_or_optimizer_load(tmp_path, damage):
    from utils.training_checkpoint import restore_training_state

    directory = tmp_path / "checkpoint-2"
    _write_payload(directory, world_size=1)
    (directory / "metadata.json").write_text(json.dumps({"global_step": 2, "world_size": 1}))
    write_checkpoint_inventory(directory, world_size=1, mixed_data=True, ema=False)
    (directory / "checkpoint_complete.json").write_text(json.dumps({
        "schema": "selfless_caption_checkpoint_complete_v2", "global_step": 2,
    }))
    if damage == "missing_metadata":
        (directory / "metadata.json").unlink()
    elif damage == "missing_rank_state":
        (directory / "data_state_rank_00000.pt").unlink()
    else:
        (directory / "data_state_rank_00000.pt").write_bytes(b"bad")
    loaded = []
    accelerator = SimpleNamespace(is_main_process=True, num_processes=1, load_state=loaded.append)
    config = SimpleNamespace(experiment=SimpleNamespace(resume_from_checkpoint=str(directory)))
    with pytest.raises((RuntimeError, ValueError, FileNotFoundError)):
        restore_training_state(config=config, accelerator=accelerator, train_dataloader=None,
                               mixed_source_training=True, config_contract={}, ema=None,
                               ema_update_after_step=0)
    assert loaded == [], "damaged published state must be rejected before loading mutable training state"
