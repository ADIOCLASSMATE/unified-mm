"""Z with B's complete flow-head modules used as a single noisy-image stream."""
from omegaconf import OmegaConf

from utils import joint_dit_protocol as z

CONFIG = "configs/selfless/unified_z_b_head_100b_ascend64.yaml"
RUN = "unified-z-b-head-0p6b-100b-imagenet-split-s42-r1"


def expected_config():
    config = z.expected_config()
    config.experiment.project = config.experiment.name = RUN
    config.experiment.identity = dict(id="z-b-head", label="Z + B head (single stream)",
                                      group="ablation", purpose="formal")
    config.model.joint_dit_head_type = "b_single_stream"
    del config.model.joint_dit_head_dim
    del config.model.joint_dit_intermediate
    config.evaluation.checkpoint = f"output/{RUN}/hf_model-final-ema"
    return config


def validate_joint_dit_config(config):
    if OmegaConf.to_container(config, resolve=True) != OmegaConf.to_container(expected_config(), resolve=True):
        raise ValueError("Z+B configuration differs from its Z-aligned experiment contract")
    report = z.validate_joint_dit_config(z.expected_config())
    return {**report, "schema": "joint_b_head_experiment_v1", "method": "Z + B head (single stream)",
            "run_project": RUN, "head_type": "b_single_stream", "head_input": "proj(x_t)",
            "head_adaln": "proj(h) + time_embed(t)", "head_kv_source": "current_noisy_stream",
            "head_attention": "bidirectional", "head_num_heads": 8, "head_mlp_ratio": 1.0,
            "head_normalization": "B LayerNorm", "head_initialization": "B"}


def parameter_report(config):
    report = z.parameter_report(config)
    if report["roles"]["flow_head"] != report["baseline_b_flow_parameters"]:
        raise ValueError("Z+B must use exactly B's flow-head parameter count")
    return report
