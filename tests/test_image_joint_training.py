import pytest
from omegaconf import OmegaConf

from scripts.launch_unified_image_joint import launch_plan
from utils.image_joint_training import CONFIG, validate_config


def test_joint_exposure_matches_b_and_each_only_run():
    contract = validate_config(OmegaConf.load(CONFIG))
    for source in ("t2i", "i2t"):
        only = OmegaConf.load(f"configs/selfless/unified_b_{source}_only_matched_ascend16.yaml")
        assert contract["cumulative_positions_per_task"] == only.training.target_physical_tokens
        assert contract["max_train_steps"] == only.training.max_train_steps
    b = OmegaConf.load("configs/selfless/unified_baseline_100b_ascend_64npu.yaml")
    for source in ("t2i", "i2t"):
        assert 64 * b.dataset.params.sources[source].micro_batch_size == contract["global_rows_per_task_per_step"]
    assert contract["total_physical_positions"] == 2 * contract["cumulative_positions_per_task"]


@pytest.mark.parametrize("key,value", [
    ("dataset.params.schedule", ["climbmix", "t2i", "climbmix", "i2t"]),
    ("experiment.resume_from_checkpoint", "output/old/checkpoint-95415"),
    ("model.model_path", "output/old/hf_model-final"),
    ("optimizer.params.learning_rate", 0.0006),
    ("training.gradient_accumulation_steps", 4),
])
def test_joint_rejects_scientific_drift(key, value):
    config = OmegaConf.load(CONFIG)
    OmegaConf.update(config, key, value)
    with pytest.raises(ValueError, match="differs from B-matched recipe"):
        validate_config(config)


def test_formal_launcher_uses_two_nodes_and_distinct_machine_ranks():
    for rank in (0, 1):
        plan = launch_plan(smoke=False, label="test", steps=12, environment={
            "PET_NODE_RANK": str(rank), "PET_NNODES": "2", "PET_NPROC_PER_NODE": "16",
            "PET_MASTER_ADDR": "10.0.0.1", "PET_MASTER_PORT": "23456",
        })
        command = plan["command"]
        assert command[command.index("--machine_rank") + 1] == str(rank)
        assert command[command.index("--num_processes") + 1] == "32"
        assert command[command.index("--num_machines") + 1] == "2"
        assert "experiment.resume_from_checkpoint=none" in command
        assert not any(arg.startswith("training.stop_after_steps=") for arg in command)
    with pytest.raises(ValueError, match="PET_NNODES=2"):
        launch_plan(smoke=False, label="test", steps=12, environment={"PET_NNODES": "1"})


def test_smoke_preserves_per_rank_shape_and_has_separate_output():
    plan = launch_plan(smoke=True, label="test", steps=12, environment={})
    assert plan["world_size"] == 16 and plan["gradient_accumulation_steps"] == 2
    assert plan["per_rank_batches"] == {"t2i": 32, "i2t": 32}
    assert plan["output_root"].endswith("-smoke-test")
    assert "training.stop_after_steps=12" in plan["command"]
