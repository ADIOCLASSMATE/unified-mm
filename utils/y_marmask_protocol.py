"""Mask-distribution-only Y comparison, stopped at 2000 on the original LR curve."""
from omegaconf import OmegaConf

from utils.y_protocol import expected_config as baseline_config, parameter_report  # noqa: F401

CONFIG = "configs/selfless/unified_y_marmask_33b_ascend64.yaml"
RUN = "unified-y-marmask-0p6b-33b-imagenet-split-s42-r1"


def expected_config():
    config = baseline_config()
    config.experiment.project = config.experiment.name = RUN
    config.experiment.identity = dict(id="y_marmask_2000_r1", label="Y-MARmask · step 2000",
                                      group="ablation", purpose="formal")
    config.model.y_mask_distribution = "mar_truncnorm"
    config.model.y_empty_visible_prob = 0.
    config.training.stop_after_steps = 2000
    config.evaluation.checkpoint = f"output/{RUN}/hf_model-final-ema"
    return config


def validate_joint_dit_config(config):
    if OmegaConf.to_container(config, resolve=True) != OmegaConf.to_container(expected_config(), resolve=True):
        raise ValueError("Y-MARmask differs from its masking-only 2000-step comparison")
    return dict(schema="y_marmask_2000_v1", method="Y-MARmask", run_project=RUN,
                world_size=64, optimizer_steps=2000, lr_schedule_steps=31800,
                mask_distribution="mar_truncnorm", mask_min=.7, mask_std=.25,
                empty_visible_prob=0., sampling_steps=10, reveal_steps=8)
