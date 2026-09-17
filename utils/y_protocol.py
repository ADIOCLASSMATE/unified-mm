"""Y's 31,800-update recipe, aligned with the current Z experiment."""
from pathlib import Path

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
CONFIG = "configs/selfless/unified_y_33b_ascend64.yaml"
RUN = "unified-y-0p6b-33b-imagenet-split-s42-r1"


def expected_config():
    config = OmegaConf.load(ROOT / "configs/selfless/unified_z_33b_ascend64.yaml")
    config.experiment.project = config.experiment.name = RUN
    config.experiment.identity = dict(id="y_33b_r1", label="Y · 33B", group="ablation", purpose="formal")
    config.experiment.validation_single_stream_order_strategies = ["random"]
    config.model.architecture_variant = "selfless_y"
    config.model.flow_head_attention_contract = "not_applicable"
    config.model.image_flow_width = 1936
    config.model.y_empty_visible_prob = .1
    config.model.y_reveal_steps = 8
    del config.model.joint_dit_head_dim
    del config.model.joint_dit_intermediate
    config.evaluation.checkpoint = f"output/{RUN}/hf_model-final-ema"
    config.evaluation.strategies = "random"
    return config


def validate_joint_dit_config(config):
    # Shared acceptance tools use this entry point for the Z-derived recipes.
    if OmegaConf.to_container(config, resolve=True) != OmegaConf.to_container(expected_config(), resolve=True):
        raise ValueError("Y configuration differs from its Z-aligned 33B recipe")
    return dict(schema="y_masked_token_flow_v1", method="Y", run_project=RUN,
                world_size=64, optimizer_steps=31800, flow_mc_samples=4,
                head="positionwise_adaln_mlp", visible_content="bidirectional",
                image_loss="unknown_per_image_mean", empty_visible_prob=.1,
                reveal_steps=8, solver="heun", sampling_steps=10,
                backbone_calls_per_image_batch=8, head_calls_per_image_batch=160)


def parameter_report(config):
    import torch
    from transformers import AutoConfig
    from models.modeling_model.modeling_y import YConfig, YForCausalLM
    from utils.selfless_flow_optimizer import optimizer_parameter_role
    payload = AutoConfig.from_pretrained(config.model.model_path, local_files_only=True).to_dict()
    payload.update(OmegaConf.to_container(config.model, resolve=True))
    payload.pop("model_type", None)
    payload.update(mask_token_id=151669, boi_token_id=151670, eoi_token_id=151671, image_mask_token_id=151672)
    with torch.device("meta"):
        model = YForCausalLM(YConfig(**payload))
    roles = {}
    for name, parameter in model.named_parameters():
        role = optimizer_parameter_role(name)
        roles[role] = roles.get(role, 0) + parameter.numel()
    return dict(roles=roles, total_parameters_before_vocab_resize=sum(roles.values()),
                reference_b_flow_parameters=164072976,
                flow_parameter_relative_difference=roles["flow_head"] / 164072976 - 1)
