import pytest
from omegaconf import OmegaConf

from models.modeling_model.image_flow_position import (
    DEFAULT_FLOW_HEAD_POSITION_VARIANT,
    FLOW_HEAD_POSITION_SPECS,
    SUPPORTED_FLOW_HEAD_POSITION_VARIANTS,
    resolve_flow_head_position_config,
)


def _config(position: str):
    return OmegaConf.create(
        {
            "model": {
                "image_flow_head_arch": "contextual",
                "image_flow_head_variant": "DF1",
                "image_flow_width": 1280,
                "image_flow_latent_mixer_heads": 8,
                "image_flow_position_variant": position,
                "image_flow_rope_axis_dims": [80, 80],
                "image_flow_rope_rotate_value": False,
            }
        }
    )


def test_active_position_interface_contains_only_the_two_df1_baselines():
    assert DEFAULT_FLOW_HEAD_POSITION_VARIANT == "FH4"
    assert SUPPORTED_FLOW_HEAD_POSITION_VARIANTS == ("FH0", "FH4")
    assert {
        name: (
            spec.query_position_mode,
            spec.context_position_mode,
            spec.rope_mode,
        )
        for name, spec in FLOW_HEAD_POSITION_SPECS.items()
    } == {
        "FH0": ("additive_2d", "additive_2d", "none"),
        "FH4": ("none", "none", "row_col_2d"),
    }


def test_production_sized_implicit_contract_defaults_to_fh4():
    config = _config("FH4")
    del config.model.image_flow_position_variant
    spec, axis_dims = resolve_flow_head_position_config(config)
    assert spec == FLOW_HEAD_POSITION_SPECS["FH4"]
    assert axis_dims == (80, 80)


@pytest.mark.parametrize("position", SUPPORTED_FLOW_HEAD_POSITION_VARIANTS)
def test_single_position_enum_resolves_the_complete_baseline_contract(position):
    config = _config(position)
    spec, axis_dims = resolve_flow_head_position_config(config)
    assert spec == FLOW_HEAD_POSITION_SPECS[position]
    assert axis_dims == (80, 80)
    assert config.model.image_flow_query_position_mode == spec.query_position_mode
    assert config.model.image_flow_context_position_mode == spec.context_position_mode
    assert config.model.image_flow_rope_mode == spec.rope_mode


@pytest.mark.parametrize("removed", ["FH1", "FH2", "FH3"])
def test_archived_position_variants_are_rejected_by_the_active_interface(removed):
    with pytest.raises(ValueError, match="Unsupported image_flow_position_variant"):
        resolve_flow_head_position_config(_config(removed))


def test_free_form_hybrid_position_flags_are_not_an_active_interface():
    config = _config("FH4")
    config.model.image_flow_query_position_mode = "additive_2d"
    config.model.image_flow_context_position_mode = "additive_2d"
    config.model.image_flow_rope_mode = "row_col_2d"
    with pytest.raises(ValueError, match="conflicts with explicit"):
        resolve_flow_head_position_config(config)
