import json

import pytest

from models.modeling_model.image_flow_position import FLOW_HEAD_POSITION_SPECS
from scripts.flow_head_position_ablation import (
    SUPPORTED_FLOW_HEAD_POSITION_VARIANTS,
    build_ablation_config,
    validate_ablation_config,
)
from scripts.summarize_flow_head_position_ablation import summarize


@pytest.mark.parametrize("variant", SUPPORTED_FLOW_HEAD_POSITION_VARIANTS)
def test_generated_screen_configs_are_closed_and_fingerprinted(variant):
    config = build_ablation_config(variant, 42)
    validate_ablation_config(config, variant)
    assert config.model.image_backbone_variant == "E2-Q1"
    assert (
        config.model.image_flow_query_position_mode
        == FLOW_HEAD_POSITION_SPECS[variant].query_position_mode
    )
    assert config.training.dataloader_shuffle_seed == 42
    config.model.image_flow_rope_rotate_value = True
    with pytest.raises(ValueError):
        validate_ablation_config(config, variant)


def test_launcher_only_config_selector_does_not_change_fingerprint():
    config = build_ablation_config("FH0", 42)
    expected = config.experiment.config_fingerprint
    config.config = "configs/ablation/flow_head_position/screen/FH0_s42.yaml"
    validate_ablation_config(config, "FH0")
    assert config.experiment.config_fingerprint == expected


def _write_metrics(tmp_path, variant, *, fid, score, early):
    validation = tmp_path / f"{variant}-validation.json"
    validation.write_text(
        json.dumps(
            {
                "schema": "selfless_flow_validation_metrics_v1",
                "global_step": 35920,
                "training_seed": 42,
                "ablation_id": variant,
                "metrics": {
                    "val/flow/context_0_v_mse": early,
                    "val/flow/context_1_v_mse": early,
                    "val/flow/v_mse": early + 0.1,
                },
            }
        )
    )
    metrics = tmp_path / f"{variant}-metrics.json"
    metrics.write_text(
        json.dumps(
            {
                "official_protocol": True,
                "architecture": {
                    "ablation_id": variant,
                    "flow_head": {
                        "position_contract": FLOW_HEAD_POSITION_SPECS[
                            variant
                        ].as_contract((80, 80))
                    },
                },
                "strategies": {
                    "spatial_halton": {
                        "fid": fid,
                        "inception_score_mean": score,
                        "inception_score_std": 0.1,
                        "generation_wall_seconds": 10.0,
                        "generation_samples_per_second": 1000.0,
                    }
                },
                "distributed": {"peak_cuda_allocated_mib": 100.0},
                "parameters": {"flow_head": 123},
                "training_protocol": {
                    "flow_head_position": {
                        "validation_metrics_path": str(validation),
                        "provenance": {
                            "ablation_id": variant,
                            "phase": "screen",
                            "training_seed": 42,
                            "architecture": {
                                "flow_head_position": FLOW_HEAD_POSITION_SPECS[
                                    variant
                                ].as_contract((80, 80))
                            },
                            "initial_parameter_schema_sha256": "a" * 64,
                            "initial_parameter_state_sha256": "b" * 64,
                            "runtime_source_manifest_sha256": "c" * 64,
                            "config_fingerprint": variant,
                            "provenance_sha256": variant,
                        },
                    }
                },
            }
        )
    )
    return metrics


def test_screen_selector_is_the_preregistered_union(tmp_path):
    values = {
        "FH0": (10.0, 10.0, 1.0),
        "FH1": (10.8, 10.5, 1.1),
        "FH2": (12.0, 9.0, 0.8),
        "FH3": (10.4, 10.2, 0.9),
        "FH4": (14.0, 8.0, 1.2),
    }
    paths = [
        _write_metrics(
            tmp_path,
            variant,
            fid=values[variant][0],
            score=values[variant][1],
            early=values[variant][2],
        )
        for variant in SUPPORTED_FLOW_HEAD_POSITION_VARIANTS
    ]
    payload = summarize(paths, phase="screen")
    selector = payload["selector"]
    assert selector["mandatory_ids"] == ["FH0"]
    assert selector["near_best_fid_ids"] == ["FH0", "FH1", "FH3"]
    assert selector["early_loss_guardrail_ids"] == ["FH3"]
    assert selector["selected_ids"] == ["FH0", "FH1", "FH3"]
