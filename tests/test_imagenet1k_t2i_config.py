from copy import deepcopy

import pytest
from omegaconf import OmegaConf

from scripts.validate_ascend_imagenet1k_t2i import validate_config


CONFIGS = {
    "baseline": (
        "configs/selfless/"
        "imagenet1k_t2i_baseline_80ep_ascend_64npu_bs1024.yaml"
    ),
    "positionwise_head": (
        "configs/selfless/"
        "imagenet1k_t2i_positionwise_head_80ep_ascend_64npu_bs1024.yaml"
    ),
    "seq_sigma": (
        "configs/selfless/"
        "imagenet1k_t2i_seq_sigma_80ep_ascend_64npu_bs1024.yaml"
    ),
}
CONFIG_PATH = CONFIGS["baseline"]


def test_t2i_baseline_config_preserves_formal_training_contract():
    config = OmegaConf.load(CONFIG_PATH)
    report = validate_config(config, world_size=64)

    assert report["global_batch"] == 1024
    assert report["gradient_accumulation_steps"] == 1
    assert report["max_optimizer_steps"] == 96_160
    assert report["task_modes"] == ["t2i"]
    assert config.model.lambda_text == pytest.approx(0.0)
    assert config.model.lambda_image == pytest.approx(1.0)
    assert config.training.num_train_epochs == 80
    assert config.dataset.params.caption_sequence_modes == ["t2i"]
    assert int(config.model.image_flow_num_sampling_steps) == 10
    assert int(config.evaluation.sampling_steps) == 10


def test_t2i_baseline_config_rejects_i2t_rows():
    config = OmegaConf.load(CONFIG_PATH)
    changed = deepcopy(config)
    changed.dataset.params.caption_sequence_modes = ["t2i", "i2t"]

    with pytest.raises(RuntimeError, match="caption_sequence_modes"):
        validate_config(changed, world_size=64)


def test_t2i_baseline_config_rejects_nonzero_text_loss():
    config = OmegaConf.load(CONFIG_PATH)
    config.model.lambda_text = 0.05

    with pytest.raises(RuntimeError, match="lambda_text"):
        validate_config(config, world_size=64)


@pytest.mark.parametrize("variant", tuple(CONFIGS))
def test_t2i_variant_configs_preserve_shared_training_contract(variant):
    config = OmegaConf.load(CONFIGS[variant])
    report = validate_config(config, world_size=64, variant=variant)

    assert report["variant"] == variant
    assert report["global_batch"] == 1024
    assert report["gradient_accumulation_steps"] == 1
    assert report["max_optimizer_steps"] == 96_160
    assert config.training.num_train_epochs == 80
    assert config.optimizer.params.learning_rate == pytest.approx(2e-5)
    assert config.dataset.params.caption_sequence_modes == ["t2i"]


def test_positionwise_t2i_config_preserves_architecture_identity():
    config = OmegaConf.load(CONFIGS["positionwise_head"])
    report = validate_config(config, world_size=64, variant="positionwise_head")

    assert report["architecture_variant"] == "positionwise_selfless"
    assert report["image_sigma_order"] == "random"
    assert report["generation_strategy"] == "spatial_halton"


def test_seq_sigma_t2i_config_preserves_order_identity():
    config = OmegaConf.load(CONFIGS["seq_sigma"])
    report = validate_config(config, world_size=64, variant="seq_sigma")

    assert report["architecture_variant"] == "selfless_contextual"
    assert report["image_sigma_order"] == "sequential"
    assert report["generation_strategy"] == "sequential"


def test_t2i_variant_validator_rejects_cross_wired_config():
    config = OmegaConf.load(CONFIGS["positionwise_head"])

    with pytest.raises(RuntimeError, match="experiment_project"):
        validate_config(config, world_size=64, variant="seq_sigma")


def _without_variant_identity(config):
    value = OmegaConf.to_container(config, resolve=False)
    value["experiment"]["project"] = "<variant-project>"
    value["experiment"]["name"] = "<variant-name>"
    value["experiment"]["validation_single_stream_order_strategies"] = [
        "<variant-order>"
    ]
    value["model"]["model_path"] = "<variant-class-ema>"
    value["model"]["architecture_variant"] = "<variant-architecture>"
    value["dataset"]["params"]["image_sigma_order"] = "<variant-order>"
    value["evaluation"]["checkpoint"] = "<variant-output-ema>"
    value["evaluation"]["strategies"] = "<variant-order>"
    return value


@pytest.mark.parametrize("variant", ("positionwise_head", "seq_sigma"))
def test_t2i_variants_match_baseline_outside_variant_identity(variant):
    baseline = _without_variant_identity(OmegaConf.load(CONFIGS["baseline"]))
    candidate = _without_variant_identity(OmegaConf.load(CONFIGS[variant]))

    assert candidate == baseline
