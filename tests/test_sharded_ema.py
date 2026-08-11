import json
from pathlib import Path

import pytest
import torch

from utils.sharded_ema import (
    RankShardedEMA,
    build_sharded_ema_layout,
    load_ema_manifest,
    merge_sharded_ema_state_dict,
)


class _Accelerator:
    def __init__(self, rank):
        self.process_index = rank
        self.is_main_process = rank == 0

    @staticmethod
    def wait_for_everyone():
        return None


class _TiedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(11, 5)
        self.proj = torch.nn.Linear(5, 11, bias=False)
        self.proj.weight = self.embed.weight
        self.extra = torch.nn.Parameter(torch.arange(23, dtype=torch.float32))
        self.register_buffer("counter", torch.tensor(3, dtype=torch.int64))


def _make_emas(model, *, world_size=3, decay=0.75, update_after_step=0):
    layout = build_sharded_ema_layout(
        model,
        world_size=world_size,
        chunk_numel=7,
    )
    emas = []
    for rank in range(world_size):
        ema = RankShardedEMA(
            layout,
            rank=rank,
            decay=decay,
            update_after_step=update_after_step,
        )
        ema.bind(model)
        ema.initialize_from_model(global_step=0)
        emas.append(ema)
    return layout, emas


def _merge_memory(layout, emas):
    canonical = {
        name: torch.empty(
            metadata["shape"],
            dtype=getattr(torch, metadata["ema_dtype"]),
        )
        for name, metadata in layout["tensors"].items()
    }
    seen = set()
    for ema in emas:
        for chunk_id, value in ema.shards.items():
            assert chunk_id not in seen
            seen.add(chunk_id)
            chunk = layout["chunks"][chunk_id]
            canonical[chunk["tensor"]].view(-1).narrow(
                0, chunk["offset"], chunk["numel"]
            ).copy_(value.cpu())
    assert seen == set(layout["chunks"])
    return {
        name: canonical[layout["canonical_for_name"][name]]
        for name in layout["state_keys"]
    }


def _save_all(emas, directory: Path, step: int):
    # The fake barrier is a no-op, so write non-main ranks before rank 0 writes
    # the manifest commit marker.
    for ema in reversed(emas):
        ema.save_checkpoint(directory, _Accelerator(ema.rank), global_step=step)


def test_layout_is_deterministic_complete_balanced_and_tied():
    torch.manual_seed(1)
    model = _TiedModel()
    first = build_sharded_ema_layout(model, world_size=3, chunk_numel=7)
    second = build_sharded_ema_layout(model, world_size=3, chunk_numel=7)
    assert first == second
    assert first["canonical_for_name"]["embed.weight"] == first["canonical_for_name"]["proj.weight"]

    owned = [
        chunk_id
        for rank in range(3)
        for chunk_id, chunk in first["chunks"].items()
        if chunk["owner"] == rank
    ]
    assert len(owned) == len(set(owned))
    assert set(owned) == set(first["chunks"])
    assert max(first["rank_bytes"]) - min(first["rank_bytes"]) <= 7 * 4


def test_sharded_update_matches_full_fp32_reference_and_tied_updates_once():
    torch.manual_seed(2)
    model = _TiedModel().to(dtype=torch.bfloat16)
    layout, emas = _make_emas(model, decay=0.9)
    reference = {}
    for name in layout["tensors"]:
        source = model.state_dict()[name].detach()
        reference[name] = (
            source.float().clone() if source.dtype.is_floating_point else source.clone()
        )

    with torch.no_grad():
        model.embed.weight.fill_(10.0)
        model.extra.fill_(6.0)
        model.counter.fill_(9)
    for name, value in reference.items():
        source = model.state_dict()[name]
        if value.dtype.is_floating_point:
            value.mul_(0.9).add_(source.float(), alpha=0.1)
        else:
            value.copy_(source)
    for ema in emas:
        ema.maybe_update(1)

    merged = _merge_memory(layout, emas)
    for name, expected in reference.items():
        torch.testing.assert_close(merged[name], expected, rtol=0, atol=0)
    assert merged["embed.weight"].data_ptr() == merged["proj.weight"].data_ptr()


def test_same_dtype_ema_shards_do_not_alias_live_model_storage():
    model = _TiedModel()
    layout, emas = _make_emas(model, world_size=2)
    state = model.state_dict(keep_vars=True)
    for ema in emas:
        for chunk_id, shard in ema.shards.items():
            source_name = layout["chunks"][chunk_id]["tensor"]
            assert (
                shard.untyped_storage().data_ptr()
                != state[source_name].untyped_storage().data_ptr()
            )


def test_delayed_start_sync_semantics():
    model = _TiedModel()
    _, emas = _make_emas(model, decay=0.5, update_after_step=3)
    before = _merge_memory(emas[0].layout, emas)
    with torch.no_grad():
        model.embed.weight.fill_(4.0)
    for ema in emas:
        assert not ema.maybe_update(2)
    unchanged = _merge_memory(emas[0].layout, emas)
    torch.testing.assert_close(unchanged["embed.weight"], before["embed.weight"], rtol=0, atol=0)
    for ema in emas:
        assert ema.maybe_update(3)
    synced = _merge_memory(emas[0].layout, emas)
    torch.testing.assert_close(
        synced["embed.weight"], torch.full_like(synced["embed.weight"], 4.0), rtol=0, atol=0
    )


def test_checkpoint_resume_and_cpu_merge_are_strict(tmp_path):
    model = _TiedModel()
    layout, emas = _make_emas(model, world_size=2, decay=0.5)
    with torch.no_grad():
        model.embed.weight.add_(2.0)
        model.extra.mul_(3.0)
    for ema in emas:
        ema.maybe_update(1)
    expected = _merge_memory(layout, emas)
    checkpoint = tmp_path / "checkpoint-1"
    _save_all(emas, checkpoint, 1)

    manifest = load_ema_manifest(checkpoint)
    assert manifest["runtime"]["global_step"] == 1
    assert manifest["runtime"]["started"] is True
    assert len(list(checkpoint.glob("ema_shard_rank_*.safetensors"))) == 2
    merged = merge_sharded_ema_state_dict(checkpoint)
    assert list(merged) == list(model.state_dict())
    for name in expected:
        torch.testing.assert_close(merged[name], expected[name], rtol=0, atol=0)

    restored = []
    for rank in range(2):
        ema = RankShardedEMA(layout, rank=rank, decay=0.5, update_after_step=0)
        ema.bind(model)
        ema.load_checkpoint(checkpoint, _Accelerator(rank), expected_global_step=1)
        restored.append(ema)
    restored_state = _merge_memory(layout, restored)
    for name in expected:
        torch.testing.assert_close(restored_state[name], expected[name], rtol=0, atol=0)

    wrong_layout = build_sharded_ema_layout(model, world_size=1, chunk_numel=7)
    wrong = RankShardedEMA(wrong_layout, rank=0, decay=0.5, update_after_step=0)
    wrong.bind(model)
    with pytest.raises(RuntimeError, match="same world size"):
        wrong.load_checkpoint(checkpoint, _Accelerator(0), expected_global_step=1)

    (checkpoint / "ema_shard_rank_00001.safetensors").unlink()
    with pytest.raises(FileNotFoundError, match="Missing EMA shard"):
        merge_sharded_ema_state_dict(checkpoint)


def test_legacy_ema_state_is_rejected(tmp_path):
    legacy_dir = tmp_path / "legacy"
    legacy_dir.mkdir()
    torch.save({"state_dict": _TiedModel().state_dict()}, legacy_dir / "ema_state.pt")
    with pytest.raises(FileNotFoundError, match="Legacy ema_state.pt checkpoints are not supported"):
        load_ema_manifest(legacy_dir)


def test_manifest_tampering_is_detected(tmp_path):
    model = _TiedModel()
    _, emas = _make_emas(model, world_size=1)
    _save_all(emas, tmp_path, 0)
    path = tmp_path / "ema_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["rank_bytes"][0] += 1
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        load_ema_manifest(tmp_path)


def test_training_checkpoint_commit_marker_is_strict_and_invalidated(tmp_path):
    from pretrain.train_selfless_flow import (
        _begin_checkpoint_write,
        _validate_checkpoint_complete,
    )

    marker = tmp_path / "checkpoint_complete.json"
    marker.write_text(
        json.dumps(
            {
                "schema": "selfless_caption_checkpoint_complete_v1",
                "global_step": 7,
            }
        )
    )
    _validate_checkpoint_complete(tmp_path, expected_global_step=7)
    with pytest.raises(RuntimeError, match="Invalid caption checkpoint completion marker"):
        _validate_checkpoint_complete(tmp_path, expected_global_step=8)

    _begin_checkpoint_write(tmp_path, accelerator=_Accelerator(rank=0))
    assert not marker.exists()
    with pytest.raises(RuntimeError, match="incomplete checkpoint"):
        _validate_checkpoint_complete(tmp_path, expected_global_step=7)
