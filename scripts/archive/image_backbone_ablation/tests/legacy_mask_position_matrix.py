"""Archived tests for the completed Q-factor matrix; not part of CI."""
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

import pytest
from omegaconf import OmegaConf

import scripts.image_mask_position_ablation_matrix as matrix_module
from scripts.image_embedder_confirmation_protocol import (
    EVALUATOR_RNG_CONTRACT_SHA256,
)
from scripts.image_mask_position_ablation_matrix import (
    DEFAULT_BASE_CONFIG,
    Q_FACTOR_SEQUENCE_LENGTH,
    build_q_factor_config,
    validate_q_factor_config,
)
from scripts.image_mask_position_ablation_protocol import (
    PARENT_CONFIRMATION_IDS,
    PARENT_PAIRING_GATE_SCHEMA,
    PARENT_SUMMARY_SCHEMA,
    Q_FACTOR_IDS,
    Q_FACTOR_PHASE,
    Q_FACTOR_SEEDS,
    Q_FACTOR_VARIANTS,
    q_factor_config_contract,
    q_factor_run_slug,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def parent_summary(tmp_path):
    seeds = sorted(Q_FACTOR_SEEDS)
    pairing_gate = {
        "schema": PARENT_PAIRING_GATE_SCHEMA,
        "validated_runs": len(PARENT_CONFIRMATION_IDS) * len(seeds),
        "evaluator_rng_contract_sha256": EVALUATOR_RNG_CONTRACT_SHA256,
    }
    payload = {
        "schema": PARENT_SUMMARY_SCHEMA,
        "expected": "confirmation",
        "runs": [
            {"id": variant_id, "training_seed": seed}
            for variant_id in PARENT_CONFIRMATION_IDS
            for seed in seeds
        ],
        "aggregates": [{"id": variant_id} for variant_id in PARENT_CONFIRMATION_IDS],
        "confirmation_scope_manifest": {
            "candidate_ids": list(PARENT_CONFIRMATION_IDS),
            "required_space_to_depth_factor": 1,
        },
        "confirmation_pairing_gate": pairing_gate,
    }
    path = tmp_path / "confirmation_summary.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_matrix_is_exact_independent_two_by_two_factorial():
    assert Q_FACTOR_IDS == ("E2b-Q1", "E2b-Q0", "E2-Q1", "E2-Q0")
    assert Q_FACTOR_SEEDS == frozenset((43, 44, 45))
    assert asdict(Q_FACTOR_VARIANTS["E2b-Q1"]) == {
        "parent_ablation_id": "E2b",
        "observed_position_mode": "additive_2d",
        "mask_position_mode": "additive_2d",
        "query_stage_mode": "none",
        "rope_mode": "row_col_2d",
        "space_to_depth_factor": 1,
    }
    assert Q_FACTOR_VARIANTS["E2b-Q0"].mask_position_mode == "none"
    assert Q_FACTOR_VARIANTS["E2-Q1"].observed_position_mode == "none"
    assert Q_FACTOR_VARIANTS["E2-Q1"].mask_position_mode == "additive_2d"
    assert Q_FACTOR_VARIANTS["E2-Q0"].observed_position_mode == "none"
    assert Q_FACTOR_VARIANTS["E2-Q0"].mask_position_mode == "none"


@pytest.mark.parametrize("variant_id", Q_FACTOR_IDS)
@pytest.mark.parametrize("seed", sorted(Q_FACTOR_SEEDS))
def test_all_twelve_configs_are_strict_and_use_independent_slugs(
    variant_id,
    seed,
    parent_summary,
):
    config = build_q_factor_config(
        variant_id,
        seed,
        parent_summary_json=parent_summary,
    )
    variant = Q_FACTOR_VARIANTS[variant_id]
    slug = f"selfless-flow-image-embedder-qf-{variant_id.lower()}-seed{seed}"

    assert q_factor_run_slug(variant_id, seed) == slug
    assert config.experiment.project == slug
    assert config.experiment.ablation_id == variant_id
    assert config.experiment.parent_ablation_id == variant.parent_ablation_id
    assert config.experiment.ablation_phase == Q_FACTOR_PHASE
    assert config.training.seed == config.training.dataloader_shuffle_seed == seed
    assert config.evaluation.seed == 42
    assert config.evaluation.checkpoint == f"output/{slug}/hf_model-final-ema"

    assert config.model.image_query_stage_mode == "none"
    assert config.model.image_observed_position_mode == variant.observed_position_mode
    assert config.model.image_mask_position_mode == variant.mask_position_mode
    assert config.model.image_rope_mode == "row_col_2d"
    assert config.model.image_space_to_depth_factor == 1
    assert config.model.image_tokens_per_img == 256
    assert config.model.image_latent_dim == 16
    assert config.dataset.params.image_space_to_depth_factor == 1
    assert (
        config.dataset.params.max_seq_length
        == config.dataset.params.pad_to_length
        == config.dataset.preprocessing.max_seq_length
        == Q_FACTOR_SEQUENCE_LENGTH
    )

    declaration = config.experiment.q_factor_protocol
    assert declaration.q_factor_id == variant_id
    assert declaration.training_seed == seed
    assert declaration.run_slug == slug
    assert OmegaConf.to_container(declaration.config_contract, resolve=True) == (
        q_factor_config_contract(config)
    )
    validate_q_factor_config(config, variant_id, seed)


def test_builder_computes_config_contract_before_declaration(
    monkeypatch,
    parent_summary,
):
    events = []
    original_contract = matrix_module.q_factor_config_contract
    original_declaration = matrix_module.build_q_factor_declaration

    def tracked_contract(*args, **kwargs):
        events.append("config_contract")
        return original_contract(*args, **kwargs)

    def tracked_declaration(*args, **kwargs):
        events.append("declaration")
        return original_declaration(*args, **kwargs)

    monkeypatch.setattr(matrix_module, "q_factor_config_contract", tracked_contract)
    monkeypatch.setattr(
        matrix_module,
        "build_q_factor_declaration",
        tracked_declaration,
    )
    build_q_factor_config("E2-Q0", 43, parent_summary_json=parent_summary)
    assert events[:2] == ["config_contract", "declaration"]


def test_builder_and_validator_reject_seed_42(parent_summary):
    with pytest.raises(ValueError, match="training seed"):
        build_q_factor_config("E2-Q0", 42, parent_summary_json=parent_summary)

    config = build_q_factor_config("E2-Q0", 43, parent_summary_json=parent_summary)
    config.training.seed = 42
    with pytest.raises(ValueError, match="training seed"):
        validate_q_factor_config(config, "E2-Q0")


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        ("model.image_query_stage_mode", "fixed_sincos", "architecture mismatch"),
        ("model.image_rope_mode", "sequence_1d", "architecture mismatch"),
        ("model.image_space_to_depth_factor", 2, "architecture mismatch"),
        ("training.dataloader_shuffle_seed", 44, "dataloader_shuffle_seed"),
    ],
)
def test_validator_rejects_architecture_and_shuffle_drift(
    path,
    value,
    message,
    parent_summary,
):
    config = build_q_factor_config("E2b-Q0", 43, parent_summary_json=parent_summary)
    OmegaConf.update(config, path, value)
    with pytest.raises(ValueError, match=message):
        validate_q_factor_config(config, "E2b-Q0")


def test_validator_rejects_other_resolved_config_contract_drift(parent_summary):
    config = build_q_factor_config("E2-Q1", 44, parent_summary_json=parent_summary)
    config.optimizer.params.beta2 = 0.9
    with pytest.raises(ValueError, match="optimizer.params.beta2"):
        validate_q_factor_config(config, "E2-Q1")


def test_base_config_and_entrypoint_shells_exist_and_parse():
    assert Path(DEFAULT_BASE_CONFIG).is_file()
    scripts = (
        REPO_ROOT / "script/ablation/train_image_mask_position_ablation.sh",
        REPO_ROOT / "script/ablation/evaluate_image_mask_position_ablation.sh",
    )
    for script in scripts:
        assert script.is_file()
        subprocess.run(["bash", "-n", str(script)], check=True)

    train_text = scripts[0].read_text(encoding="utf-8")
    eval_text = scripts[1].read_text(encoding="utf-8")
    assert "ALLOW_EXISTING_RUN_DIR" in train_text
    assert "selfless-flow-image-embedder-qf-${ID_LOWER}-seed${TRAINING_SEED}" in train_text
    assert "ALLOW_EXISTING_METRICS" in eval_text
    assert "--require_image_embedder_ablation_protocol" in eval_text
