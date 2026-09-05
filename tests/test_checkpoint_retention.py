import json
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf
from safetensors.torch import save_file

import pretrain.train_selfless_flow as training
from pretrain.train_selfless_flow import _image_flow_adapter_save_enabled
from utils.utils import checkpoint_save_due, rotate_checkpoints_for_save


def _mkdirs(root: Path, *names: str) -> None:
    for name in names:
        (root / name).mkdir()


def _write_test_safetensors(path: Path, state_dict: dict[str, torch.Tensor]) -> int:
    """Write one representative tensor for each tied-storage group."""

    stored = {}
    seen_ids = set()
    for name, tensor in state_dict.items():
        identity = id(tensor)
        if identity in seen_ids:
            continue
        seen_ids.add(identity)
        stored[name] = tensor
    save_file(stored, str(path))
    return len(stored)


@pytest.mark.parametrize(
    "config_path",
    sorted((Path(__file__).resolve().parents[1] / "configs/selfless").glob("*.yaml")),
    ids=lambda path: path.stem,
)
def test_training_configs_follow_project_artifact_defaults(config_path):
    config = OmegaConf.load(config_path)
    # Unified image tasks and pure-text controls share this reference cadence.
    steps_per_epoch = int(config.training.get("optimizer_steps_per_epoch", 1_251))
    experiment = config.experiment

    assert int(experiment.checkpoints_total_limit) == 3
    assert int(experiment.checkpoint_milestone_every) == 100 * steps_per_epoch
    assert int(experiment.save_ema_eval_every) == 20 * steps_per_epoch
    assert experiment.save_model_with_ema_eval is True
    assert experiment.ema_eval_dtype == "bf16"
    assert config.training.use_ema is True
    assert config.training.ema_save_hf_model is True


def test_off_cadence_milestones_are_saved_and_survive_rolling_checkpoints(tmp_path):
    saved_steps = []

    class Accelerator:
        is_main_process = True

        @staticmethod
        def wait_for_everyone():
            pass

        @staticmethod
        def save_state(directory):
            directory.mkdir()
            (directory / "training-state.pt").write_bytes(b"state")
            saved_steps.append(int(directory.name.removeprefix("checkpoint-")))

    config = OmegaConf.create(
        {
            "experiment": {
                "output_dir": str(tmp_path),
                "save_every": 2_000,
                "checkpoints_total_limit": 3,
                "checkpoint_milestone_every": 125_100,
            },
            "model": {},
        }
    )
    for step in range(260_001):
        if checkpoint_save_due(
            step,
            save_every=config.experiment.save_every,
            milestone_every_steps=config.experiment.checkpoint_milestone_every,
        ):
            training.save_checkpoint(None, config, Accelerator(), step)

    assert 0 not in saved_steps
    assert saved_steps.count(125_100) == 1
    assert saved_steps.count(250_200) == 1
    assert sorted(path.name for path in tmp_path.glob("checkpoint-*")) == [
        "checkpoint-125100",
        "checkpoint-250200",
        "checkpoint-256000",
        "checkpoint-258000",
        "checkpoint-260000",
    ]
    for path in tmp_path.glob("checkpoint-*"):
        assert (path / "training-state.pt").is_file()
        metadata = json.loads((path / "metadata.json").read_text())
        assert metadata["global_step"] == int(path.name.removeprefix("checkpoint-"))


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


def test_four_rolling_saves_leave_exactly_the_latest_three(tmp_path: Path):
    for step in (10, 20, 30, 40):
        name = f"checkpoint-{step}"
        rotate_checkpoints_for_save(
            tmp_path,
            3,
            current_checkpoint_name=name,
            milestone_every_steps=0,
        )
        checkpoint = tmp_path / name
        checkpoint.mkdir()
        (checkpoint / "checkpoint_complete.json").write_text("{}")

    assert sorted(path.name for path in tmp_path.glob("checkpoint-*")) == [
        "checkpoint-20",
        "checkpoint-30",
        "checkpoint-40",
    ]


def test_begin_checkpoint_write_removes_stale_partial_destination(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint-20"
    checkpoint.mkdir()
    (checkpoint / "stale-rank-state.pt").write_bytes(b"partial")
    (checkpoint / "checkpoint_complete.json").write_text("stale")

    class Accelerator:
        is_main_process = True

        @staticmethod
        def wait_for_everyone():
            return None

    training._begin_checkpoint_write(
        checkpoint,
        accelerator=Accelerator(),
    )

    assert not checkpoint.exists()


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
        "layout_validation": "readable_field_equality",
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
            _write_test_safetensors(path / "model.safetensors", state_dict)
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
        "source_ema_directory_retained": True,
        "source_global_step": 20,
        "source_world_size": 4,
        "layout_validation": "readable_field_equality",
        "state_key_count": len(full_state),
        "stored_weight_key_count": 3,
    }
    assert hf_config["dtype"] == "bfloat16"
    assert hf_config["torch_dtype"] == "bfloat16"


def test_periodic_current_model_export_is_complete_atomic_and_idempotent(
    tmp_path: Path,
):
    global_step = 12_510
    save_calls = []
    full_state = {
        "weight": torch.randn(2, 3, dtype=torch.float32),
        "position_ids": torch.arange(3, dtype=torch.int64),
    }

    class FakeModel:
        def save_pretrained(
            self,
            path,
            *,
            save_function,
            state_dict,
            safe_serialization,
        ):
            del save_function
            save_calls.append(Path(path))
            path.mkdir(parents=True)
            assert safe_serialization
            assert state_dict["weight"].dtype == torch.bfloat16
            assert state_dict["position_ids"].dtype == torch.int64
            _write_test_safetensors(path / "model.safetensors", state_dict)
            (path / "config.json").write_text(
                json.dumps({"dtype": "float32", "torch_dtype": "float32"})
            )

    class FakeTokenizer:
        @staticmethod
        def save_pretrained(path):
            (path / "tokenizer.json").write_text("{}")

    class FakeAccelerator:
        is_main_process = True

        @staticmethod
        def get_state_dict(model):
            del model
            return dict(full_state)

        @staticmethod
        def unwrap_model(model):
            return model

        @staticmethod
        def save(*args, **kwargs):
            del args, kwargs

        @staticmethod
        def wait_for_everyone():
            return None

    config = OmegaConf.create(
        {
            "experiment": {
                "output_dir": str(tmp_path),
                "ema_eval_dtype": "bf16",
            },
            "training": {},
        }
    )
    args = (
        FakeModel(),
        FakeTokenizer(),
        config,
        FakeAccelerator(),
        global_step,
    )
    training._save_model_hf_for_evaluation(*args)

    export = tmp_path / f"hf_model-{global_step}-eval"
    assert save_calls == [tmp_path / f".{export.name}.partial"]
    assert not (tmp_path / f".{export.name}.partial").exists()
    assert (export / "model.safetensors").is_file()
    assert (export / "config.json").is_file()
    assert (export / "tokenizer.json").is_file()
    metadata = json.loads(
        (export / "model_export_metadata.json").read_text()
    )
    assert metadata == {
        "schema": "selfless_model_hf_export_v1",
        "export_kind": "evaluation",
        "floating_dtype": "bfloat16",
        "source_global_step": global_step,
        "state_key_count": len(full_state),
        "stored_weight_key_count": len(full_state),
    }
    hf_config = json.loads((export / "config.json").read_text())
    assert hf_config["dtype"] == "bfloat16"
    assert hf_config["torch_dtype"] == "bfloat16"

    # Reaching the same step again after resuming an older rolling checkpoint
    # keeps the already-complete artifact instead of overwriting it.
    training._save_model_hf_for_evaluation(*args)
    assert len(save_calls) == 1


def test_periodic_ema_eval_export_uses_ephemeral_state_off_checkpoint_cadence(
    tmp_path: Path,
    monkeypatch,
):
    global_step = 12_510
    source_checkpoint = tmp_path / f"checkpoint-{global_step}"
    staged_directories = []

    class FakeEMA:
        started = True

        def __init__(self):
            self.global_step = global_step

        def save_checkpoint(self, directory, accelerator, *, global_step):
            directory = Path(directory)
            staged_directories.append(directory)
            directory.mkdir(parents=True)
            (directory / "ema_manifest.json").write_text("{}")
            accelerator.wait_for_everyone()
            return directory / "ema_manifest.json"

    monkeypatch.setattr(
        training,
        "merge_sharded_ema_state_dict",
        lambda path: {"weight": torch.ones(2, dtype=torch.float32)},
    )
    monkeypatch.setattr(
        training,
        "load_ema_manifest",
        lambda path: {
            "runtime": {"global_step": global_step},
            "world_size": 64,
            "layout_validation": "readable_field_equality",
        },
    )
    monkeypatch.setattr(training.logger, "info", lambda *args, **kwargs: None)

    class FakeModel:
        def save_pretrained(self, path, *, state_dict, safe_serialization):
            path.mkdir(parents=True)
            assert safe_serialization
            assert state_dict["weight"].dtype == torch.bfloat16
            _write_test_safetensors(path / "model.safetensors", state_dict)
            (path / "config.json").write_text(
                json.dumps({"dtype": "float32", "torch_dtype": "float32"})
            )

    class FakeTokenizer:
        @staticmethod
        def save_pretrained(path):
            (path / "tokenizer.json").write_text("{}")

    class FakeAccelerator:
        is_main_process = True

        @staticmethod
        def unwrap_model(model):
            return model

        @staticmethod
        def wait_for_everyone():
            return None

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
        global_step,
        source_checkpoint,
        floating_dtype=torch.bfloat16,
        save_name=f"hf_model-{global_step}-ema-eval",
        export_kind="evaluation",
    )

    export = tmp_path / f"hf_model-{global_step}-ema-eval"
    assert [path.name for path in staged_directories] == [
        f".hf_model-{global_step}-ema-eval.ema-state.partial"
    ]
    assert not source_checkpoint.exists()
    assert not staged_directories[0].exists()
    assert not (tmp_path / f".{export.name}.partial").exists()
    assert (export / "model.safetensors").is_file()
    assert (export / "config.json").is_file()
    assert (export / "tokenizer.json").is_file()
    assert (export / "ema_export_metadata.json").is_file()
    metadata = json.loads((export / "ema_export_metadata.json").read_text())
    assert metadata["source_ema_directory_retained"] is False

    training._save_ema_hf_model(
        FakeEMA(),
        FakeModel(),
        FakeTokenizer(),
        config,
        FakeAccelerator(),
        global_step,
        source_checkpoint,
        floating_dtype=torch.bfloat16,
        save_name=f"hf_model-{global_step}-ema-eval",
        export_kind="evaluation",
    )
    assert len(staged_directories) == 1

    _mkdirs(tmp_path, "checkpoint-8000", "checkpoint-10000", "checkpoint-12000")
    rotate_checkpoints_for_save(
        tmp_path,
        3,
        current_checkpoint_name="checkpoint-14000",
    )
    (tmp_path / "checkpoint-14000").mkdir()
    assert sorted(path.name for path in tmp_path.glob("checkpoint-*")) == [
        "checkpoint-10000",
        "checkpoint-12000",
        "checkpoint-14000",
    ]


def test_current_model_export_broadcasts_rank0_failure_and_cleans_partial(
    tmp_path: Path,
    monkeypatch,
):
    broadcasts = []

    def record_broadcast(objects, *, from_process):
        assert from_process == 0
        broadcasts.append(dict(objects[0]))
        return objects

    monkeypatch.setattr(training, "broadcast_object_list", record_broadcast)

    class FakeModel:
        @staticmethod
        def save_pretrained(path, **kwargs):
            del kwargs
            path.mkdir(parents=True)
            (path / "partial").write_text("partial")
            raise OSError("injected write failure")

    class FakeAccelerator:
        is_main_process = True
        num_processes = 64

        @staticmethod
        def get_state_dict(model):
            del model
            return {"weight": torch.ones(1)}

        @staticmethod
        def unwrap_model(model):
            return model

        @staticmethod
        def save(*args, **kwargs):
            del args, kwargs

    config = OmegaConf.create(
        {
            "experiment": {
                "output_dir": str(tmp_path),
                "ema_eval_dtype": "bf16",
            },
            "training": {},
        }
    )
    with pytest.raises(RuntimeError, match="injected write failure"):
        training._save_model_hf_for_evaluation(
            FakeModel(),
            object(),
            config,
            FakeAccelerator(),
            12_510,
        )

    assert broadcasts[-1]["ok"] is False
    assert not (tmp_path / ".hf_model-12510-eval.partial").exists()
    assert not (tmp_path / "hf_model-12510-eval").exists()


def test_non_main_rank_never_materializes_the_complete_current_model(
    tmp_path: Path,
    monkeypatch,
):
    outcomes = iter(
        [
            {"ok": True, "result": "write"},
            {"ok": True, "result": "saved"},
        ]
    )

    def receive_rank0_outcome(objects, *, from_process):
        assert from_process == 0
        assert objects == [None]
        objects[0] = next(outcomes)
        return objects

    monkeypatch.setattr(
        training,
        "broadcast_object_list",
        receive_rank0_outcome,
    )

    class NonMainAccelerator:
        is_main_process = False
        num_processes = 64

        @staticmethod
        def get_state_dict(model):
            del model
            raise AssertionError("non-main rank must not gather the full model")

    config = OmegaConf.create(
        {
            "experiment": {
                "output_dir": str(tmp_path),
                "ema_eval_dtype": "bf16",
            },
            "training": {},
        }
    )
    training._save_model_hf_for_evaluation(
        object(),
        object(),
        config,
        NonMainAccelerator(),
        12_510,
    )


def test_evaluation_pair_manifest_commits_only_two_valid_exports(tmp_path: Path):
    global_step = 12_510
    dtype_name = "bfloat16"
    exports = (
        (
            tmp_path / f"hf_model-{global_step}-eval",
            "model_export_metadata.json",
            "selfless_model_hf_export_v1",
        ),
        (
            tmp_path / f"hf_model-{global_step}-ema-eval",
            "ema_export_metadata.json",
            "selfless_ema_hf_export_v1",
        ),
    )
    for directory, metadata_name, schema in exports:
        directory.mkdir()
        _write_test_safetensors(
            directory / "model.safetensors",
            {"weight": torch.ones(2, dtype=torch.bfloat16)},
        )
        (directory / "config.json").write_text(
            json.dumps({"dtype": dtype_name, "torch_dtype": dtype_name})
        )
        (directory / "tokenizer.json").write_text("{}")
        (directory / metadata_name).write_text(
            json.dumps(
                {
                    "schema": schema,
                    "export_kind": "evaluation",
                    "floating_dtype": dtype_name,
                    "source_global_step": global_step,
                    "state_key_count": 1,
                    "stored_weight_key_count": 1,
                }
            )
        )

    class FakeAccelerator:
        is_main_process = True

    config = OmegaConf.create(
        {
            "experiment": {
                "output_dir": str(tmp_path),
                "ema_eval_dtype": "bf16",
            },
            "training": {},
        }
    )
    training._publish_evaluation_model_pair_manifest(
        config,
        FakeAccelerator(),
        global_step,
    )

    manifest_path = tmp_path / f"hf_model-{global_step}-eval-pair.json"
    assert json.loads(manifest_path.read_text()) == {
        "schema": "selfless_evaluation_model_pair_v1",
        "complete": True,
        "global_step": global_step,
        "floating_dtype": dtype_name,
        "current_model_directory": f"hf_model-{global_step}-eval",
        "ema_model_directory": f"hf_model-{global_step}-ema-eval",
    }
    assert not (tmp_path / f".{manifest_path.name}.partial").exists()

    # A resume that reaches the same step keeps the exact committed pair.
    original = manifest_path.read_bytes()
    training._publish_evaluation_model_pair_manifest(
        config,
        FakeAccelerator(),
        global_step,
    )
    assert manifest_path.read_bytes() == original

    # A non-empty but corrupt weight file can never be accepted as complete.
    (exports[0][0] / "model.safetensors").write_bytes(b"truncated")
    with pytest.raises(RuntimeError, match="Invalid safetensors export"):
        training._publish_evaluation_model_pair_manifest(
            config,
            FakeAccelerator(),
            global_step,
        )
