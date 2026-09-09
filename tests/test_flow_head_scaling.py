import pytest
import torch
from omegaconf import OmegaConf

from models.modeling_model.image_flow_loss import FlowLoss
from scripts.launch_unified_flow_head_scaling import launch_plan
from utils.flow_head_scaling import HEAD_PARAMETERS, config_path, validate_scaling_config


@pytest.mark.parametrize("depth", [16, 30])
def test_scaling_preserves_recipe_and_counts_actual_parameters(depth):
    config = OmegaConf.load(config_path(depth))
    report = validate_scaling_config(config)
    with torch.device("meta"):
        head = FlowLoss(target_channels=16, z_channels=1024, width=1280, depth=depth,
                        num_sampling_steps=10, flow_head_attention_contract="xlnet_content_diagonal")
    assert sum(p.numel() for p in head.parameters()) == HEAD_PARAMETERS[depth]
    assert report["head_parameters"] < 600_000_000
    assert report["max_train_steps"] == 95415


@pytest.mark.parametrize("field,value", [
    ("optimizer.params.flow_learning_rate", 1e-4),
    ("model.image_flow_width", 2048),
    ("model.image_flow_batch_mul", 1),
    ("model.flow_condition_contract", "backbone_xt_shared_query_content"),
    ("training.trainable_scope", "image_flow_head"),
    ("training.max_train_steps", 100),
    ("dataset.params.sources.t2i.micro_batch_size", 8),
])
def test_scaling_rejects_recipe_drift(field, value):
    config = OmegaConf.load(config_path(16))
    OmegaConf.update(config, field, value)
    with pytest.raises(ValueError, match=field):
        validate_scaling_config(config)


def test_activation_checkpointing_is_required_by_the_scaling_protocol():
    config = OmegaConf.load(config_path(30))
    config.model.image_flow_grad_checkpointing = True
    assert validate_scaling_config(config)["image_flow_grad_checkpointing"] is True
    config.model.image_flow_grad_checkpointing = False
    with pytest.raises(ValueError, match="image_flow_grad_checkpointing=true"):
        validate_scaling_config(config)


def test_formal_and_smoke_launch_use_the_same_local_workload():
    environment = {"PET_NNODES": "4", "PET_NODE_RANK": "2", "PET_NPROC_PER_NODE": "16",
                   "PET_MASTER_ADDR": "10.0.0.1", "PET_MASTER_PORT": "29500"}
    formal = launch_plan(30, smoke=False, label="r1", steps=12, environment=environment)
    smoke = launch_plan(30, smoke=True, label="r1", steps=12, environment={})
    assert formal["world_size"] == 64 and formal["rank"] == 2
    assert smoke["world_size"] == 16 and smoke["rank"] == 0
    assert formal["per_rank_batches"] == smoke["per_rank_batches"] == {
        "climbmix": 4, "t2i": 16, "i2t": 16}
    assert formal["image_flow_batch_mul"] == smoke["image_flow_batch_mul"] == 4
    assert not any(item.startswith("training.stop_after_steps=") for item in formal["command"])
    assert "training.stop_after_steps=12" in smoke["command"]
    assert formal["output_root"] != smoke["output_root"]


def test_formal_launch_rejects_wrong_world_size():
    with pytest.raises(ValueError, match="PET_NNODES=4"):
        launch_plan(16, smoke=False, label="r1", steps=12,
                    environment={"PET_NNODES": "1", "PET_NODE_RANK": "0"})
