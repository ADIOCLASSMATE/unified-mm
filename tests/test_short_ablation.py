"""Short runs must finish decay, validate at the endpoint, and start independently."""
import pytest
import torch
from omegaconf import OmegaConf

from scripts.launch_short_ablation import launch_plan
from utils.short_ablation_protocol import ARMS, ROOT, config_path, validate_short_config
from utils.wsd_schedule import get_wsd_schedule


def environment(rank=0):
    return dict(PET_NODE_RANK=str(rank), PET_NNODES="4", PET_NPROC_PER_NODE="0",
                PET_MASTER_ADDR="127.0.0.1", PET_MASTER_PORT="29500")


@pytest.mark.parametrize("arm", [arm for arm in ARMS if arm != "y"])
def test_short_config_keeps_model_data_optimizer_and_scales_ema(arm):
    config = OmegaConf.load(ROOT / config_path(arm))
    base = OmegaConf.load(ROOT / config_path(arm, base=True))
    contract = validate_short_config(arm, config)
    for section in ("model", "dataset", "optimizer"):
        assert OmegaConf.to_container(config[section], resolve=True) == OmegaConf.to_container(base[section], resolve=True)
    assert base.training.ema_decay == 0.9999
    assert config.training.ema_decay == 0.9997
    n, original_n = config.training.max_train_steps, base.training.max_train_steps
    d, original_d = config.training.ema_decay, base.training.ema_decay
    assert n * (1 - d) == pytest.approx(original_n * (1 - original_d), rel=1e-3)
    assert d ** n == pytest.approx(original_d ** original_n, rel=1e-3)
    assert (1 / (1 - d) / config.lr_scheduler.params.decay_steps) == pytest.approx(
        1 / (1 - original_d) / base.lr_scheduler.params.decay_steps, rel=1e-3)
    generation = config.experiment.validation_generation
    assert generation.enabled and generation.weights == "raw" and generation.samples == 16
    assert generation.seed == 42 and generation.cfg == 3.5
    assert generation.steps == 10 and generation.solver == "heun"
    assert config.experiment.downstream_validation == base.experiment.downstream_validation
    assert config.training.max_train_steps == config.training.stop_after_steps == 31800
    assert config.training.target_text_tokens == 33328435200
    assert contract["validation_steps"] == list(range(3180, 31801, 3180))
    assert config.experiment.project != base.experiment.project
    assert config.experiment.resume_from_checkpoint == "none"
    assert config.evaluation.checkpoint.endswith(f"{config.experiment.project}/hf_model-final-ema")
    config.model.lambda_text *= 2
    with pytest.raises(ValueError, match="budget/EMA/validation"):
        validate_short_config(arm, config)


@pytest.mark.parametrize("arm", ARMS)
def test_scheduler_reaches_floor_at_short_endpoint(arm):
    config = OmegaConf.load(ROOT / config_path(arm))
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    scheduler = get_wsd_schedule(optimizer, config.lr_scheduler.params.warmup_steps,
                                 config.lr_scheduler.params.decay_steps, config.training.max_train_steps,
                                 config.lr_scheduler.params.min_lr_scale)
    factor = scheduler.lr_lambdas[0]
    assert factor(0) == 0
    assert factor(199) == 1
    assert factor(23850) == 1
    assert factor(27825) == pytest.approx(.55)
    assert factor(31800) == pytest.approx(.1)


@pytest.mark.parametrize("arm", ARMS)
def test_all_four_nodes_launch_the_same_fresh_short_run(arm):
    plans = [launch_plan(arm, environment=environment(rank)) for rank in range(4)]
    for rank, plan in enumerate(plans):
        assert plan["rank"] == rank
        assert plan["world_size"] == 64
        assert plan["contract"]["optimizer_steps"] == 31800
        assert plan["output_root"] == plans[0]["output_root"]
        assert f"config={config_path(arm)}" in plan["command"]
        assert "experiment.resume_from_checkpoint=none" in plan["command"]
        assert ("--assets" in plan["preflight"]) == (rank == 0)
    base = OmegaConf.load(ROOT / config_path("z" if arm == "y" else arm, base=True))
    with pytest.raises(ValueError, match="exact short-budget run"):
        launch_plan(arm, environment=environment(),
                    resume=str(ROOT / "output" / base.experiment.project / "checkpoint-2000"))


def test_short_run_rejects_wrong_node_count():
    invalid = environment()
    invalid["PET_NNODES"] = "2"
    with pytest.raises(ValueError, match="PET_NNODES=4"):
        launch_plan("z", environment=invalid)


def test_short_s2_pair_differs_only_in_text_attention():
    single = OmegaConf.load(ROOT / config_path("s2-single"))
    double = OmegaConf.load(ROOT / config_path("s2-text2stream"))
    model = OmegaConf.to_container(single.model, resolve=True)
    assert model["dual_stream_attention_contract"] == "showo2_omni_attention"
    model["dual_stream_attention_contract"] = "showo2_text_two_stream"
    assert model == OmegaConf.to_container(double.model, resolve=True)
    assert single.dataset == double.dataset
    assert single.training == double.training
    assert single.optimizer == double.optimizer
    assert single.lr_scheduler == double.lr_scheduler
    assert single.experiment.downstream_validation == double.experiment.downstream_validation
