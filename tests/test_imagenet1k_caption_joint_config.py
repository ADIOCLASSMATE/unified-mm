from copy import deepcopy

import pytest
from omegaconf import OmegaConf

from scripts.validate_ascend_imagenet1k_caption_joint import validate_config


CONFIG_PATH = (
    "configs/selfless/imagenet1k_caption_joint_10ep_ascend16_b1024.yaml"
)


def test_selected_joint_config_preserves_training_contract():
    config = OmegaConf.load(CONFIG_PATH)
    report = validate_config(config, world_size=16)

    assert report["global_batch"] == 1024
    assert report["gradient_accumulation_steps"] == 4
    assert report["max_optimizer_steps"] == 12020
    assert config.model.lambda_text == pytest.approx(0.05)
    assert config.model.lambda_image == pytest.approx(1.0)
    assert config.optimizer.params.backbone_learning_rate == pytest.approx(2e-5)
    assert config.optimizer.params.special_token_learning_rate == pytest.approx(2e-5)
    assert config.optimizer.params.projector_learning_rate == pytest.approx(2e-5)
    assert config.optimizer.params.flow_learning_rate == pytest.approx(2e-5)
    assert config.experiment.log_grad_norm_every % config.experiment.log_every == 0


def test_selected_joint_config_rejects_sweep_override():
    config = OmegaConf.load(CONFIG_PATH)
    changed = deepcopy(config)
    changed.optimizer.params.backbone_learning_rate = 1e-5

    with pytest.raises(RuntimeError, match="backbone_learning_rate"):
        validate_config(changed, world_size=16)


def test_selected_joint_config_rejects_unlogged_gradient_norm_interval():
    config = OmegaConf.load(CONFIG_PATH)
    config.experiment.log_grad_norm_every = 1202

    with pytest.raises(RuntimeError, match="log_grad_norm_every"):
        validate_config(config, world_size=16)
