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
