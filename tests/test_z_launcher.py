import pytest

from scripts.launch_z import launch_plan


@pytest.mark.parametrize("rank", range(4))
@pytest.mark.parametrize("platform_processes", [None, "0", "16"])
def test_formal_platform_launch_always_starts_64_workers(rank, platform_processes):
    environment = {"PET_NODE_RANK": str(rank), "PET_NNODES": "4",
                   "PET_MASTER_ADDR": "10.0.0.1", "PET_MASTER_PORT": "29500"}
    if platform_processes is not None:
        environment["PET_NPROC_PER_NODE"] = platform_processes
    plan = launch_plan(smoke=False, label="r1", steps=12, environment=environment)
    command = plan["command"]
    assert plan["rank"] == rank and plan["world_size"] == 64
    assert plan["platform_nproc_per_node"] == int(platform_processes or 0)
    assert command[command.index("--num_processes") + 1] == "64"
    assert command[command.index("--num_machines") + 1] == "4"
    assert command[command.index("--machine_rank") + 1] == str(rank)
    assert plan["preflight"][plan["preflight"].index("--require-npu-count") + 1] == "16"
    assert not any(value.startswith("training.stop_after_steps=") for value in command)


@pytest.mark.parametrize("override", [{"PET_NNODES": "1"}, {"PET_NODE_RANK": "4"},
                                     {"PET_NPROC_PER_NODE": "8"}])
def test_formal_platform_launch_rejects_wrong_allocation(override):
    environment = {"PET_NODE_RANK": "0", "PET_NNODES": "4", "PET_NPROC_PER_NODE": "0",
                   "PET_MASTER_ADDR": "10.0.0.1", "PET_MASTER_PORT": "29500", **override}
    with pytest.raises(ValueError, match="PET_NODE_RANK=0..3, PET_NNODES=4"):
        launch_plan(smoke=False, label="r1", steps=12, environment=environment)
