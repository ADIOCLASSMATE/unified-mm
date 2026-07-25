"""Archived tests for the historical screening matrix; not part of CI."""
import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from scripts.image_embedder_ablation_matrix import (
    DEFAULT_BASE_CONFIG,
    FLOW_HEAD_INVARIANTS,
    VARIANTS,
    build_ablation_config,
    normalize_variant_id,
    validate_ablation_config,
)
from scripts.image_embedder_confirmation_protocol import (
    CONFIRMATION_SEEDS,
    canonical_sha256,
)


EXPECTED_IDS = (
    "E0",
    "E1",
    "E2a",
    "E2b",
    "E2",
    "E3",
    "E4a",
    "E4b",
    "E4",
    "E5",
    "E6a",
    "E6b",
    "E6",
    "E7a",
    "E7b",
    "E7",
)
CONFIRMATION_IDS = ("E0", "E1", "E2b", "E2", "E3", "E4b", "E4", "E6b", "E6", "E7a")


@pytest.fixture
def confirmation_screen(tmp_path):
    manifest = {
        "schema": "selfless_flow_image_embedder_confirmation_candidates_v1",
        "screen_summary_schema": "selfless_flow_image_embedder_ablation_summary_v3",
        "screen_training_seed": 42,
        "confirmation_training_seeds": [43, 44, 45],
        "near_best_fid_margin": 1.0,
        "speed_advantage_ratio_vs_e0": 1.5,
        "near_best_fid_ids": ["E1", "E2b", "E2", "E4b", "E4"],
        "fid_is_pareto_ids": ["E2"],
        "speed_pareto_ids_meeting_threshold": ["E3", "E6b", "E6", "E7a"],
        "candidate_ids": list(CONFIRMATION_IDS),
    }
    payload = {
        "schema": "selfless_flow_image_embedder_ablation_summary_v3",
        "expected": "expanded",
        "runs": [{"id": value, "training_seed": 42} for value in EXPECTED_IDS],
        "aggregates": [{"id": value} for value in EXPECTED_IDS],
        "confirmation_candidate_manifest": manifest,
    }
    path = tmp_path / "expanded.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_matrix_contains_atomic_controls_and_full_factorial():
    assert tuple(VARIANTS) == EXPECTED_IDS
    assert VARIANTS["E2a"].observed_position_mode == "none"
    assert VARIANTS["E2a"].rope_mode == "sequence_1d"
    assert VARIANTS["E2b"].observed_position_mode == "additive_2d"
    assert VARIANTS["E2b"].rope_mode == "row_col_2d"
    assert VARIANTS["E2"].observed_position_mode == "none"
    assert VARIANTS["E2"].rope_mode == "row_col_2d"
    assert VARIANTS["E4a"].query_stage_mode == "fixed_sincos"
    assert VARIANTS["E4a"].observed_position_mode == "none"
    assert VARIANTS["E4a"].rope_mode == "sequence_1d"
    assert VARIANTS["E4b"].query_stage_mode == "fixed_sincos"
    assert VARIANTS["E4b"].observed_position_mode == "additive_2d"
    assert VARIANTS["E4b"].rope_mode == "row_col_2d"
    assert VARIANTS["E6a"].space_to_depth_factor == 2
    assert VARIANTS["E6a"].observed_position_mode == "none"
    assert VARIANTS["E6a"].rope_mode == "sequence_1d"
    assert VARIANTS["E6b"].space_to_depth_factor == 2
    assert VARIANTS["E6b"].observed_position_mode == "additive_2d"
    assert VARIANTS["E6b"].rope_mode == "row_col_2d"
    assert VARIANTS["E7a"].query_stage_mode == "fixed_sincos"
    assert VARIANTS["E7a"].space_to_depth_factor == 2
    assert VARIANTS["E7a"].observed_position_mode == "none"
    assert VARIANTS["E7a"].rope_mode == "sequence_1d"
    assert VARIANTS["E7b"].query_stage_mode == "fixed_sincos"
    assert VARIANTS["E7b"].space_to_depth_factor == 2
    assert VARIANTS["E7b"].observed_position_mode == "additive_2d"
    assert VARIANTS["E7b"].rope_mode == "row_col_2d"


@pytest.mark.parametrize("variant_id", EXPECTED_IDS)
def test_every_resolved_config_freezes_flow_head_and_derives_layout(variant_id):
    config = build_ablation_config(variant_id, 137)
    validate_ablation_config(config, variant_id)
    assert config.experiment.ablation_id == variant_id
    assert config.training.seed == 137
    assert config.dataset.params.split_seed == 42
    assert config.evaluation.seed == 42
    for key, expected in FLOW_HEAD_INVARIANTS.items():
        assert config.model[key] == expected

    factor = VARIANTS[variant_id].space_to_depth_factor
    expected = (256, 16, 320) if factor == 1 else (64, 64, 128)
    actual = (
        config.model.image_tokens_per_img,
        config.model.image_latent_dim,
        config.dataset.params.pad_to_length,
    )
    assert actual == expected
    assert config.dataset.params.image_tokens_per_img == expected[0]
    assert config.dataset.params.image_latent_dim == expected[1]
    assert config.dataset.params.image_space_to_depth_factor == factor


def test_unknown_id_is_rejected():
    with pytest.raises(ValueError, match="Unknown ablation ID"):
        normalize_variant_id("E8")


def test_base_config_is_repo_local_and_exists():
    assert Path(DEFAULT_BASE_CONFIG).is_file()


def test_confirmation_config_is_manifest_gated_and_uses_paired_shuffle_seed(
    confirmation_screen,
):
    screen = confirmation_screen
    config = build_ablation_config(
        "E2",
        43,
        confirmation_screen_json=screen,
    )
    assert config.experiment.ablation_phase == "confirmation"
    assert config.training.seed == config.training.dataloader_shuffle_seed == 43
    declaration = config.experiment.confirmation_protocol
    assert declaration.candidate_manifest_sha256 == canonical_sha256(
        OmegaConf.to_container(declaration.candidate_manifest, resolve=True)
    )
    validate_ablation_config(config, "E2")


@pytest.mark.parametrize("seed", sorted(CONFIRMATION_SEEDS))
def test_every_preregistered_confirmation_seed_is_accepted(seed, confirmation_screen):
    config = build_ablation_config(
        "E0",
        seed,
        confirmation_screen_json=confirmation_screen,
    )
    assert config.training.dataloader_shuffle_seed == seed


def test_confirmation_rejects_wrong_seed_non_candidate_and_tampered_screen(
    tmp_path,
    confirmation_screen,
):
    screen = confirmation_screen
    with pytest.raises(ValueError, match="confirmation training seed"):
        build_ablation_config("E0", 42, confirmation_screen_json=screen)
    with pytest.raises(ValueError, match="not in the frozen"):
        build_ablation_config("E2a", 43, confirmation_screen_json=screen)

    payload = json.loads(screen.read_text(encoding="utf-8"))
    payload["confirmation_candidate_manifest"]["candidate_ids"].append("E2a")
    tampered = tmp_path / "screen.json"
    tampered.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="selector union"):
        build_ablation_config("E0", 43, confirmation_screen_json=tampered)


def test_confirmation_validator_rejects_shuffle_seed_drift(confirmation_screen):
    config = build_ablation_config(
        "E0",
        43,
        confirmation_screen_json=confirmation_screen,
    )
    config.training.dataloader_shuffle_seed = 44
    with pytest.raises(ValueError, match="dataloader_shuffle_seed"):
        validate_ablation_config(config, "E0")
