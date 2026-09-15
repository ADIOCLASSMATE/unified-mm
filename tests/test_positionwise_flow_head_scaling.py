import pytest
import torch
from omegaconf import OmegaConf

from models.modeling_model.image_flow_loss_positionwise import PositionwiseFlowLoss
from scripts.launch_unified_flow_head_scaling import launch_plan
from utils.positionwise_flow_head_scaling import (
    B_HEAD_PARAMETERS, HEAD_PARAMETERS, WIDTHS, config_path, validate_scaling_config,
)


@pytest.mark.parametrize("depth", [16, 30])
def test_f_scaling_counts_real_mlp_parameters_and_matches_b_budget(depth):
    config = OmegaConf.load(config_path(depth))
    report = validate_scaling_config(config)
    with torch.device("meta"):
        head = PositionwiseFlowLoss(target_channels=16, z_channels=1024,
                                   width=WIDTHS[depth], depth=depth,
                                   num_sampling_steps=10, grad_checkpointing=True)
    count = sum(p.numel() for p in head.parameters())
    assert count == HEAD_PARAMETERS[depth] == report["head_parameters"]
    assert abs(count - B_HEAD_PARAMETERS[depth]) / B_HEAD_PARAMETERS[depth] < 0.005
    assert len(head.net.res_blocks) == depth
    assert head.net.position_contract()["cross_token_attention"] is False
    assert head.net.position_contract()["uses_content_latents"] is False


@pytest.mark.parametrize("field,value", [
    ("optimizer.params.flow_learning_rate", 1e-4),
    ("training.max_train_steps", 100),
    ("training.from_scratch", True),
    ("training.seed", 43),
    ("model.image_flow_batch_mul", 1),
    ("model.image_flow_width", 1936),
    ("model.image_flow_share_content", True),
    ("model.image_flow_grad_checkpointing", False),
    ("model.positionwise_reference_flow_depth", 8),
    ("model.flow_head_attention_contract", "xlnet_content_diagonal"),
    ("dataset.params.sources.t2i.micro_batch_size", 8),
])
def test_f_scaling_rejects_budget_architecture_or_data_drift(field, value):
    config = OmegaConf.load(config_path(16))
    OmegaConf.update(config, field, value)
    with pytest.raises(ValueError, match=field):
        validate_scaling_config(config)


def test_f_formal_and_smoke_use_same_per_rank_recipe_and_f_validator():
    environment = {"PET_NNODES": "4", "PET_NODE_RANK": "2", "PET_NPROC_PER_NODE": "16",
                   "PET_MASTER_ADDR": "10.0.0.1", "PET_MASTER_PORT": "29500"}
    formal = launch_plan(30, smoke=False, label="r1", steps=12, environment=environment, ablation="f")
    smoke = launch_plan(30, smoke=True, label="r1", steps=12, environment={}, ablation="f")
    assert formal["world_size"] == 64 and formal["rank"] == 2
    assert smoke["world_size"] == 16
    assert formal["per_rank_batches"] == smoke["per_rank_batches"] == {"climbmix": 4, "t2i": 16, "i2t": 16}
    assert formal["image_flow_batch_mul"] == smoke["image_flow_batch_mul"] == 4
    assert formal["preflight"][formal["preflight"].index("--ablation") + 1] == "f"
    assert "--flow-head-scaling" in formal["preflight"]
    assert f"config={config_path(30)}" in formal["command"]
    assert "experiment.resume_from_checkpoint=none" in formal["command"]
    assert not any(s.startswith("training.stop_after_steps=") for s in formal["command"])
    assert "training.stop_after_steps=12" in smoke["command"]
    assert formal["output_root"] != smoke["output_root"]


def test_scaling_rejects_unsupported_ablation():
    with pytest.raises(ValueError, match="only for B and F"):
        launch_plan(16, smoke=True, label="r1", steps=12, environment={}, ablation="d")
