from pathlib import Path

from omegaconf import OmegaConf

from scripts.archive.backbone_flow_head_joint_ablation.backbone_flow_head_joint_ablation import (
    BACKBONE_VARIANTS,
    CELLS,
    CONCEPTUAL_ORDER,
    FLOW_POSITION_VARIANTS,
    build_config,
    cell_id,
    control_fingerprint,
    select_winner,
    validate_config,
)
from pretrain.train_selfless_flow import (
    _validation_flat_query_mixer_context,
    _validation_sequence_mixer_context,
)
from models.modeling_model.image_flow_loss import FlowLoss
from models.modeling_model.image_flow_position import FLOW_HEAD_POSITION_SPECS
import torch


SOURCE_DIGEST = "a" * 64


def test_joint_matrix_is_the_closed_three_by_two_product():
    assert BACKBONE_VARIANTS == ("E2-Q1", "E2-Q0", "E2b-Q0")
    assert FLOW_POSITION_VARIANTS == ("FH0", "FH4")
    assert CELLS == (
        "E2-Q1__DF1-FH0",
        "E2-Q1__DF1-FH4",
        "E2-Q0__DF1-FH0",
        "E2-Q0__DF1-FH4",
        "E2b-Q0__DF1-FH0",
        "E2b-Q0__DF1-FH4",
    )


def test_generated_configs_change_only_the_two_factors_and_run_identity():
    configs = [
        build_config(
            backbone,
            position,
            source_manifest_sha256=SOURCE_DIGEST,
        )
        for backbone in BACKBONE_VARIANTS
        for position in FLOW_POSITION_VARIANTS
    ]
    assert len({control_fingerprint(config) for config in configs}) == 1
    for config in configs:
        contract = validate_config(config)
        assert contract["cell_id"] in CELLS
        assert config.training.seed == 42
        assert config.training.dataloader_shuffle_seed == 42
        assert config.model.image_flow_head_variant == "DF1"
        assert not any(
            key in config.model
            for key in (
                "image_flow_query_position_mode",
                "image_flow_context_position_mode",
                "image_flow_rope_mode",
                "image_flow_rope_axis_dims",
                "image_flow_rope_rotate_value",
            )
        )


def test_validator_rejects_a_hidden_low_level_position_override():
    config = build_config(
        "E2-Q0",
        "FH4",
        source_manifest_sha256=SOURCE_DIGEST,
    )
    config.model.image_flow_rope_mode = "row_col_2d"
    config.experiment.config_fingerprint = "invalid"
    try:
        validate_config(config)
    except ValueError as error:
        assert "retired low-level position knob" in str(error)
    else:
        raise AssertionError("hidden low-level override was accepted")


def _synthetic_rows(fid_by_cell, is_by_cell):
    return [
        {
            "cell_id": cell,
            "fid": float(fid_by_cell[cell]),
            "inception_score_mean": float(is_by_cell[cell]),
        }
        for cell in CELLS
    ]


def test_selector_prefers_end_to_end_pure_rope_inside_quality_band():
    fid = {cell: 23.0 for cell in CELLS}
    score = {cell: 64.0 for cell in CELLS}
    fid["E2-Q1__DF1-FH0"] = 22.7
    score["E2-Q1__DF1-FH0"] = 64.4
    decision = select_winner(_synthetic_rows(fid, score))
    assert decision["selected"] == "E2-Q0__DF1-FH4"
    assert decision["fallback_used"] is False


def test_selector_does_not_hide_a_material_quality_gap():
    fid = {cell: 24.0 for cell in CELLS}
    score = {cell: 63.0 for cell in CELLS}
    fid["E2-Q1__DF1-FH0"] = 22.0
    score["E2-Q1__DF1-FH0"] = 65.0
    decision = select_winner(_synthetic_rows(fid, score))
    assert decision["selected"] == "E2-Q1__DF1-FH0"


def test_conceptual_order_is_total_and_starts_with_pure_rope():
    assert set(CONCEPTUAL_ORDER) == set(CELLS)
    assert CONCEPTUAL_ORDER[0] == cell_id("E2-Q0", "FH4")


def test_active_full_training_config_uses_selected_joint_default():
    config = OmegaConf.load(
        Path("configs/selfless/imagenet_flow_full_from_qwen3base.yaml")
    )
    assert config.model.image_backbone_variant == "E2-Q0"
    assert config.model.image_flow_head_variant == "DF1"
    assert config.model.image_flow_position_variant == "FH4"


def _tiny_df1_flow(position: str) -> FlowLoss:
    spec = FLOW_HEAD_POSITION_SPECS[position]
    return FlowLoss(
        target_channels=4,
        z_channels=8,
        depth=2,
        width=16,
        num_sampling_steps=1,
        mlp_ratio=1.0,
        image_tokens_per_img=4,
        latent_mixer_heads=4,
        position_variant=position,
        query_position_mode=spec.query_position_mode,
        context_position_mode=spec.context_position_mode,
        rope_mode=spec.rope_mode,
        rope_axis_dims=(2, 2),
        flow_head_variant="DF1",
    ).eval()


def test_periodic_validation_supplies_df1_content_conditions():
    torch.manual_seed(7)
    target = torch.randn(4, 4)
    conditions = torch.randn(4, 8)
    sigma = torch.tensor([3.0, 0.0, 2.0, 1.0])
    positions = torch.arange(4)

    flat = _validation_flat_query_mixer_context(
        target,
        sigma,
        positions,
        conditions,
    )
    assert flat["context_conditions"].shape == (4, 4, 8)
    for query_idx in range(4):
        torch.testing.assert_close(
            flat["context_conditions"][query_idx],
            conditions,
        )

    for position in ("FH0", "FH4"):
        sampled = _tiny_df1_flow(position).sample(
            conditions,
            num_steps=1,
            **flat,
        )
        assert sampled.shape == target.shape
        assert torch.isfinite(sampled).all()


def test_periodic_validation_probe_context_preserves_token_conditions():
    torch.manual_seed(8)
    target = torch.randn(4, 4)
    conditions = torch.randn(4, 8)
    sigma = torch.tensor([3.0, 0.0, 2.0, 1.0])
    positions = torch.arange(4)
    sequence = _validation_sequence_mixer_context(
        target,
        sigma,
        positions,
        conditions,
    )
    assert sequence["context_conditions"].shape == (1, 4, 8)
    torch.testing.assert_close(
        sequence["context_conditions"][0],
        conditions,
    )
    velocity = _tiny_df1_flow("FH4").velocity(
        torch.randn(1, 4, 4),
        torch.full((1, 4), 0.5),
        conditions.unsqueeze(0),
        **sequence,
    )
    assert velocity.shape == (1, 4, 4)
    assert torch.isfinite(velocity).all()
