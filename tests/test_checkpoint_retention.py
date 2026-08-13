import json
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

import pretrain.train_selfless_flow as training
from pretrain.train_selfless_flow import _image_flow_adapter_save_enabled
from utils.utils import rotate_checkpoints_for_save


def _mkdirs(root: Path, *names: str) -> None:
    for name in names:
        (root / name).mkdir()


def test_rotation_excludes_destination_created_early_by_non_main_rank(tmp_path: Path):
    _mkdirs(
        tmp_path,
        "checkpoint-10",
        "checkpoint-20",
        "checkpoint-30",
        "checkpoint-40",
    )

    removed = rotate_checkpoints_for_save(
        tmp_path,
        3,
        current_checkpoint_name="checkpoint-40",
    )

    assert [path.name for path in removed] == ["checkpoint-10"]
    assert sorted(path.name for path in tmp_path.glob("checkpoint-*")) == [
        "checkpoint-20",
        "checkpoint-30",
        "checkpoint-40",
    ]


def test_rotation_keeps_space_for_destination_not_created_yet(tmp_path: Path):
    _mkdirs(tmp_path, "checkpoint-10", "checkpoint-20", "checkpoint-30")

    rotate_checkpoints_for_save(
        tmp_path,
        3,
        current_checkpoint_name="checkpoint-40",
    )
    (tmp_path / "checkpoint-40").mkdir()

    assert sorted(path.name for path in tmp_path.glob("checkpoint-*")) == [
        "checkpoint-20",
        "checkpoint-30",
        "checkpoint-40",
    ]


def test_rotation_rejects_nonpositive_limit(tmp_path: Path):
    with pytest.raises(ValueError, match="must be positive"):
        rotate_checkpoints_for_save(
            tmp_path,
            0,
            current_checkpoint_name="checkpoint-10",
        )


def test_ordinary_save_keeps_milestones_plus_latest_three_ordinary(tmp_path: Path):
    _mkdirs(
        tmp_path,
        "checkpoint-80",
        "checkpoint-90",
        "checkpoint-100",
        "checkpoint-110",
        "checkpoint-120",
        "checkpoint-130",
        # Simulate the destination being created early by a non-main rank.
        "checkpoint-140",
    )

    removed = rotate_checkpoints_for_save(
        tmp_path,
        3,
        current_checkpoint_name="checkpoint-140",
        milestone_every_steps=100,
    )

    assert [path.name for path in removed] == [
        "checkpoint-80",
        "checkpoint-90",
        "checkpoint-110",
    ]
    assert sorted(path.name for path in tmp_path.glob("checkpoint-*")) == [
        "checkpoint-100",
        "checkpoint-120",
        "checkpoint-130",
        "checkpoint-140",
    ]


def test_milestone_save_does_not_consume_a_rolling_slot(tmp_path: Path):
    _mkdirs(
        tmp_path,
        "checkpoint-100",
        "checkpoint-160",
        "checkpoint-170",
        "checkpoint-180",
        "checkpoint-200",
    )

    removed = rotate_checkpoints_for_save(
        tmp_path,
        3,
        current_checkpoint_name="checkpoint-200",
        milestone_every_steps=100,
    )

    assert removed == []
    assert sorted(path.name for path in tmp_path.glob("checkpoint-*")) == [
        "checkpoint-100",
        "checkpoint-160",
        "checkpoint-170",
        "checkpoint-180",
        "checkpoint-200",
    ]


def test_next_ordinary_save_preserves_all_milestones(tmp_path: Path):
    _mkdirs(
        tmp_path,
        "checkpoint-100",
        "checkpoint-160",
        "checkpoint-170",
        "checkpoint-180",
        "checkpoint-200",
    )

    removed = rotate_checkpoints_for_save(
        tmp_path,
        3,
        current_checkpoint_name="checkpoint-210",
        milestone_every_steps=100,
    )
    (tmp_path / "checkpoint-210").mkdir()

    assert [path.name for path in removed] == ["checkpoint-160"]
    assert sorted(path.name for path in tmp_path.glob("checkpoint-*")) == [
        "checkpoint-100",
        "checkpoint-170",
        "checkpoint-180",
        "checkpoint-200",
        "checkpoint-210",
    ]


def test_intermediate_adapter_can_be_disabled_while_final_remains_enabled():
    config = OmegaConf.create(
        {
            "experiment": {
                "save_image_flow_adapter": False,
                "save_final_image_flow_adapter": True,
            },
            "training": {
                "save_image_flow_adapter": True,
            }
        }
    )

    assert not _image_flow_adapter_save_enabled(config, final=False)
    assert _image_flow_adapter_save_enabled(config, final=True)


def test_periodic_ema_eval_export_is_complete_bf16_and_self_describing(
    tmp_path: Path,
    monkeypatch,
):
    shared = torch.randn(2, 3, dtype=torch.float32)
    full_state = {
        "model.embed_tokens.weight": shared,
        "lm_head.weight": shared,
        "image_flow_head.weight": torch.randn(3, 3, dtype=torch.float32),
        "position_ids": torch.arange(3, dtype=torch.int64),
    }
    manifest = {
        "runtime": {"global_step": 20},
        "world_size": 4,
        "layout_fingerprint": "layout-sha256",
    }
    ema_dir = tmp_path / "checkpoint-20"
    ema_dir.mkdir()
    (ema_dir / "ema_manifest.json").write_text("{}")

    monkeypatch.setattr(
        training,
        "merge_sharded_ema_state_dict",
        lambda path: dict(full_state),
    )
    monkeypatch.setattr(training, "load_ema_manifest", lambda path: manifest)

    class FakeModel:
        def save_pretrained(self, path, *, state_dict, safe_serialization):
            path.mkdir(parents=True)
            assert safe_serialization
            assert set(state_dict) == set(full_state)
            assert state_dict["model.embed_tokens.weight"].dtype == torch.bfloat16
            assert state_dict["model.embed_tokens.weight"] is state_dict["lm_head.weight"]
            assert state_dict["position_ids"].dtype == torch.int64
            (path / "model.safetensors").write_bytes(b"complete-model")
            (path / "config.json").write_text(
                json.dumps({"dtype": "float32", "torch_dtype": "float32"})
            )

    class FakeTokenizer:
        def save_pretrained(self, path):
            (path / "tokenizer.json").write_text("{}")

    class FakeAccelerator:
        is_main_process = True

        @staticmethod
        def unwrap_model(model):
            return model

        @staticmethod
        def wait_for_everyone():
            return None

    class FakeEMA:
        started = True

    config = OmegaConf.create(
        {
            "experiment": {"output_dir": str(tmp_path)},
            "training": {"ema_save_hf_model": True},
        }
    )
    training._save_ema_hf_model(
        FakeEMA(),
        FakeModel(),
        FakeTokenizer(),
        config,
        FakeAccelerator(),
        20,
        ema_dir,
        floating_dtype=torch.bfloat16,
        save_name="hf_model-20-ema-eval",
        export_kind="evaluation",
    )

    export = tmp_path / "hf_model-20-ema-eval"
    metadata = json.loads((export / "ema_export_metadata.json").read_text())
    hf_config = json.loads((export / "config.json").read_text())
    assert metadata == {
        "schema": "selfless_ema_hf_export_v1",
        "export_kind": "evaluation",
        "floating_dtype": "bfloat16",
        "source_ema_directory": str(ema_dir),
        "source_global_step": 20,
        "source_world_size": 4,
        "layout_fingerprint": "layout-sha256",
        "state_key_count": len(full_state),
    }
    assert hf_config["dtype"] == "bfloat16"
    assert hf_config["torch_dtype"] == "bfloat16"
