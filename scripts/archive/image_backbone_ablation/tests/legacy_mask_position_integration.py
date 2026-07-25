"""Archived tests for the completed Q-factor study; not part of CI."""
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

import scripts.evaluate_single_stream_fid_is as evaluator
from pretrain.train_selfless_flow import (
    _reinitialize_image_modules,
    _uses_paired_image_initialization,
)
from scripts.image_embedder_confirmation_protocol import canonical_sha256
from scripts.image_mask_position_ablation_protocol import Q_FACTOR_PHASE
from utils.utils import save_checkpoint


def _digest(character):
    return str(character) * 64


def _declaration():
    return {
        "declaration_sha256": _digest("a"),
        "study_manifest_sha256": _digest("b"),
        "parent_summary_sha256": _digest("c"),
        "config_contract_sha256": _digest("d"),
        "source_manifest_sha256": _digest("e"),
        "runtime_source_manifest_sha256": _digest("e"),
        "evaluator_rng_contract_sha256": _digest("f"),
        "config_contract": {
            "schema": "test_q_factor_config_contract_v1",
            "resolved_config": {"bound": True},
        },
    }


def _q_config(tmp_path, *, variant_id="E2-Q0", seed=43, max_train_steps=10):
    parent_id = "E2b" if variant_id.startswith("E2b-") else "E2"
    return OmegaConf.create(
        {
            "experiment": {
                "output_dir": str(tmp_path),
                "checkpoints_total_limit": None,
                "ablation_phase": Q_FACTOR_PHASE,
                "ablation_id": variant_id,
                "parent_ablation_id": parent_id,
                "q_factor_protocol": _declaration(),
                "q_factor_provenance_path": str(
                    tmp_path / "q_factor_training_provenance.json"
                ),
                "q_factor_provenance_sha256": "",
            },
            "model": {
                "reinitialize_image_modules": True,
                "image_query_stage_mode": "none",
                "image_observed_position_mode": "none",
                "image_mask_position_mode": "none",
                "image_rope_mode": "row_col_2d",
                "image_space_to_depth_factor": 1,
            },
            "training": {
                "seed": seed,
                "dataloader_shuffle_seed": seed,
                "max_train_steps": max_train_steps,
            },
        }
    )


class _RandomReset:
    def __init__(self):
        self.value = None

    def initialize_weights(self):
        self.value = torch.randn(5)


class _PairedResetModel:
    def __init__(self):
        self.image_flow_head = SimpleNamespace(net=_RandomReset())
        self.embedder_value = None
        self.condition_value = None

    def reset_image_token_embedder(self):
        self.embedder_value = torch.randn(5)

    def _reset_image_flow_condition_proj(self):
        self.condition_value = torch.randn(5)

    def values(self):
        return (
            self.image_flow_head.net.value,
            self.embedder_value,
            self.condition_value,
        )


def test_pretrain_paired_initialization_recognizes_q_configs(tmp_path):
    q1_config = _q_config(tmp_path, variant_id="E2-Q1", seed=44)
    q0_config = _q_config(tmp_path, variant_id="E2-Q0", seed=44)
    plain_config = OmegaConf.create(
        {
            "experiment": {"ablation_phase": "screen"},
            "model": {"reinitialize_image_modules": True},
            "training": {"seed": 44},
        }
    )
    assert _uses_paired_image_initialization(q1_config)
    assert _uses_paired_image_initialization(q0_config)
    assert not _uses_paired_image_initialization(plain_config)

    first = _PairedResetModel()
    second = _PairedResetModel()
    torch.manual_seed(1)
    torch.rand(1_000)
    assert _reinitialize_image_modules(first, q1_config)
    torch.manual_seed(999)
    torch.rand(17)
    assert _reinitialize_image_modules(second, q0_config)

    for first_value, second_value in zip(first.values(), second.values()):
        assert torch.equal(first_value, second_value)


class _SavingAccelerator:
    is_main_process = True

    def save_state(self, path):
        Path(path).mkdir(parents=True, exist_ok=False)


def test_save_checkpoint_writes_all_q_provenance_bindings(tmp_path):
    config = _q_config(tmp_path, variant_id="E2b-Q0", seed=45, max_train_steps=7)
    config.experiment.q_factor_provenance_sha256 = _digest("9")
    save_checkpoint(None, config, _SavingAccelerator(), global_step=7)

    metadata = json.loads(
        (tmp_path / "checkpoint-7/metadata.json").read_text(encoding="utf-8")
    )
    declaration = _declaration()
    assert metadata["global_step"] == 7
    assert metadata["model_config"] == OmegaConf.to_container(
        config.model,
        resolve=True,
    )
    assert metadata["q_factor_provenance"] == {
        "path": str(config.experiment.q_factor_provenance_path),
        "sha256": _digest("9"),
        "declaration_sha256": declaration["declaration_sha256"],
        "study_manifest_sha256": declaration["study_manifest_sha256"],
        "config_contract_sha256": declaration["config_contract_sha256"],
        "source_manifest_sha256": declaration["source_manifest_sha256"],
    }
    assert "confirmation_provenance" not in metadata


def _provenance(declaration):
    payload = {
        "schema": "test_q_factor_training_provenance_v1",
        "q_factor_id": "E2-Q0",
        "parent_ablation_id": "E2",
        "training_seed": 43,
        "architecture": {
            "parent_ablation_id": "E2",
            "observed_position_mode": "none",
            "mask_position_mode": "none",
            "query_stage_mode": "none",
            "rope_mode": "row_col_2d",
            "space_to_depth_factor": 1,
        },
        "q_factor_declaration": declaration,
        "q_factor_declaration_sha256": declaration["declaration_sha256"],
        "study_manifest_sha256": declaration["study_manifest_sha256"],
        "parent_summary_sha256": declaration["parent_summary_sha256"],
        "config_contract_sha256": declaration["config_contract_sha256"],
        "runtime_source_manifest_sha256": declaration[
            "runtime_source_manifest_sha256"
        ],
        "source_manifest_sha256": declaration["source_manifest_sha256"],
        "initial_state": {
            "contract": {"schema": "test_init_v1"},
            "image_modules": {
                "parameter_count": 3,
                "parameter_schema_sha256": _digest("1"),
                "state_sha256": _digest("2"),
            },
            "special_token_names_and_ids": [["image_mask", 8]],
            "special_token_rows_sha256": _digest("3"),
        },
        "train_data": {
            "contract": {"schema": "test_order_v1"},
            "dataloader_shuffle_seed": 43,
            "initial_generator_state_sha256": _digest("4"),
            "dataloader_base_seed": 123,
            "dataset_length": 90_000,
            "epoch0_ordered_sample_identity_sha256": _digest("5"),
            "augmentation_contract": {"schema": "test_aug_v1"},
            "epoch0_augmentation_decisions_sha256": _digest("6"),
            "augmentation_seed": 43,
            "latent_hflip_probability": 0.5,
            "batch_size_per_rank": 32,
            "total_batch_size": 256,
            "drop_last": True,
            "num_workers": 8,
            "persistent_workers": True,
            "input_files": {},
        },
        "base_model": {"manifest_sha256": _digest("7")},
        "runtime_context": {
            "schema": "test_runtime_v1",
            "world_size": 8,
        },
    }
    payload["provenance_sha256"] = canonical_sha256(payload)
    return payload


def _strict_test_provenance_loader(
    path,
    *,
    expected_sha256,
    variant_id,
    seed,
    config,
):
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("missing or invalid Q-factor training provenance") from exc
    stored = payload.pop("provenance_sha256", None)
    if stored != expected_sha256 or canonical_sha256(payload) != stored:
        raise ValueError("Q-factor training provenance digest mismatch")
    payload["provenance_sha256"] = stored
    assert variant_id == "E2-Q0"
    assert seed == 43
    assert config.experiment.ablation_phase == Q_FACTOR_PHASE
    return payload


def _prepare_evaluator_case(tmp_path, monkeypatch):
    config = _q_config(tmp_path, variant_id="E2-Q0", seed=43, max_train_steps=10)
    declaration = _declaration()
    provenance = _provenance(declaration)
    provenance_path = Path(config.experiment.q_factor_provenance_path)
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    config.experiment.q_factor_provenance_sha256 = provenance["provenance_sha256"]

    checkpoint = tmp_path / "checkpoint-10"
    checkpoint.mkdir()
    checkpoint_binding = {
        "path": str(provenance_path),
        "sha256": provenance["provenance_sha256"],
        "declaration_sha256": declaration["declaration_sha256"],
        "study_manifest_sha256": declaration["study_manifest_sha256"],
        "config_contract_sha256": declaration["config_contract_sha256"],
        "source_manifest_sha256": declaration["source_manifest_sha256"],
    }
    metadata = {
        "global_step": 10,
        "model_config": OmegaConf.to_container(config.model, resolve=True),
        "q_factor_provenance": checkpoint_binding,
    }
    metadata_path = checkpoint / "metadata.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    (checkpoint / "ema_state.pt").write_bytes(b"ema")

    hf_model = tmp_path / "hf_model-final-ema"
    hf_model.mkdir()
    (hf_model / "model.safetensors").write_bytes(b"weights")
    hf_provenance_path = hf_model / provenance_path.name
    hf_provenance_path.write_text(json.dumps(provenance), encoding="utf-8")

    calls = []

    def validate_q(config_to_validate):
        calls.append(("q", str(config_to_validate.experiment.ablation_id)))

    def reject_legacy(*_args, **_kwargs):
        pytest.fail("Q-factor artifacts were dispatched to the legacy validator")

    def validate_declaration(
        declaration_payload,
        *,
        variant_id,
        seed,
        config_contract,
    ):
        assert variant_id == "E2-Q0"
        assert seed == 43
        assert config_contract == declaration["config_contract"]
        assert declaration_payload == declaration
        return declaration

    monkeypatch.setattr(evaluator, "validate_q_factor_config", validate_q)
    monkeypatch.setattr(evaluator, "validate_ablation_config", reject_legacy)
    monkeypatch.setattr(
        evaluator,
        "q_factor_config_contract",
        lambda _config: declaration["config_contract"],
    )
    monkeypatch.setattr(
        evaluator,
        "validate_q_factor_declaration",
        validate_declaration,
    )
    monkeypatch.setattr(
        evaluator,
        "load_and_validate_q_factor_training_provenance",
        _strict_test_provenance_loader,
    )
    monkeypatch.setattr(
        evaluator,
        "training_protocol_metadata",
        lambda _config, final_global_step: {
            "schema": "test_training_protocol_v1",
            "training_seed": 43,
            "final_global_step": final_global_step,
        },
    )
    return SimpleNamespace(
        config=config,
        config_path=tmp_path / "config.yaml",
        hf_model=hf_model,
        provenance=provenance,
        provenance_path=provenance_path,
        hf_provenance_path=hf_provenance_path,
        checkpoint=checkpoint,
        metadata=metadata,
        metadata_path=metadata_path,
        calls=calls,
    )


def _load_evaluator_artifacts(case):
    return evaluator.image_embedder_training_artifacts(
        case.config,
        config_path=str(case.config_path),
        model_path=str(case.hf_model),
    )


def test_evaluator_dispatches_q_and_binds_both_provenance_copies(
    tmp_path,
    monkeypatch,
):
    case = _prepare_evaluator_case(tmp_path, monkeypatch)
    protocol = _load_evaluator_artifacts(case)

    assert case.calls == [("q", "E2-Q0")]
    assert protocol["q_factor"]["provenance_sha256"] == case.provenance[
        "provenance_sha256"
    ]
    assert protocol["q_factor"]["provenance"]["q_factor_id"] == "E2-Q0"
    assert protocol["q_factor"]["provenance"]["architecture"][
        "mask_position_mode"
    ] == "none"
    assert protocol["artifacts"]["q_factor_provenance_path"] == str(
        case.provenance_path
    )
    assert protocol["artifacts"]["q_factor_hf_provenance_path"] == str(
        case.hf_provenance_path
    )
    assert len(protocol["artifacts"]["checkpoint_metadata_sha256"]) == 64
    assert len(protocol["artifacts"]["hf_model_weights_sha256"]) == 64


def test_evaluator_rejects_missing_or_tampered_checkpoint_binding(
    tmp_path,
    monkeypatch,
):
    case = _prepare_evaluator_case(tmp_path, monkeypatch)
    case.metadata_path.unlink()
    with pytest.raises(ValueError, match="Missing or invalid final checkpoint metadata"):
        _load_evaluator_artifacts(case)

    tampered = dict(case.metadata)
    tampered["q_factor_provenance"] = dict(tampered["q_factor_provenance"])
    tampered["q_factor_provenance"]["config_contract_sha256"] = _digest("0")
    case.metadata_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="Q-factor provenance binding"):
        _load_evaluator_artifacts(case)


def test_evaluator_rejects_missing_or_tampered_hf_provenance(
    tmp_path,
    monkeypatch,
):
    case = _prepare_evaluator_case(tmp_path, monkeypatch)
    case.hf_provenance_path.unlink()
    with pytest.raises(ValueError, match="missing or invalid Q-factor"):
        _load_evaluator_artifacts(case)

    tampered = dict(case.provenance)
    tampered["training_seed"] = 44
    case.hf_provenance_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="digest mismatch"):
        _load_evaluator_artifacts(case)


def test_evaluator_main_statically_enables_q_pairing_and_reports_q_architecture():
    source = inspect.getsource(evaluator.main)
    assert "or is_q_factor_config(config)" in source
    for field in (
        '"parent_ablation_id"',
        '"q_factor_id"',
        '"mask_query_position_factor"',
        '"image_mask_position_mode"',
    ):
        assert field in source
