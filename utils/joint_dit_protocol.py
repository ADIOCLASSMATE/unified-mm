"""Experiment Z: tied image sigma and a single-stream whole-image DiT."""
from pathlib import Path

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = ROOT / "configs/selfless/unified_baseline_100b_ascend_64npu.yaml"
CONFIG = "configs/selfless/unified_z_100b_ascend64.yaml"
RUN = "unified-z-0p6b-100b-imagenet-split-s42-r1"


def expected_config():
    config = OmegaConf.load(BASE_CONFIG)
    config.experiment.project = RUN
    config.experiment.name = RUN
    config.experiment.identity = dict(id="z", label="Z", group="ablation", purpose="formal")
    config.experiment.save_final_image_flow_adapter = False
    config.experiment.validation_single_stream_order_strategies = ["joint"]
    config.experiment.validation_generation = dict(enabled=True, samples=16, seed=42,
        prompt_file="configs/protocols/unified_qualitative_prompts_v1.json",
        cfg=3.5, steps=10, solver="heun", vae_module_root="public/code/mar",
        vae_path="public/vae/mar-kl16/kl16.ckpt", vae_scaling_factor=0.2325)
    config.model.architecture_variant = "selfless_joint_dit"
    config.model.flow_head_attention_contract = "joint_bidirectional"
    config.model.flow_condition_contract = "backbone_xt_fixed"
    config.model.training_image_sigma_order = "joint"
    config.model.image_flow_grad_checkpointing = True
    config.model.joint_dit_head_dim = 64
    config.model.joint_dit_intermediate = 1472
    config.training.save_image_flow_adapter = False
    config.training.ema_save_adapter = False
    config.dataset.params.image.image_sigma_order = "joint"
    config.evaluation.checkpoint = f"output/{RUN}/hf_model-final-ema"
    config.evaluation.strategies = "joint"
    return config


def validate_joint_dit_config(config):
    expected = expected_config()
    if OmegaConf.to_container(config, resolve=True) != OmegaConf.to_container(expected, resolve=True):
        raise ValueError("Z configuration differs from its B-aligned experiment contract")
    return {"schema": "joint_dit_experiment_v1", "method": "Z", "run_project": RUN,
            "world_size": 64, "optimizer_steps": 95415, "flow_mc_samples": 4,
            "backbone_streams": 2, "head_streams": 1, "image_sigma_order": "joint",
            "backbone_calls_per_image_batch": 1, "head_calls_per_image_batch": 20,
            "solver": "heun", "sampling_steps": 10,
            "time_in_backbone": False, "shared_time_per_image": True,
            "cfg_branches_batched": True, "image_input_noise_strength": 0.01,
            "validation_generation": {"every_validation": True, "images": 16, "weights": "ema",
                                      "seed": 42, "cfg": 3.5, "solver": "heun", "steps": 10}}


def parameter_report(config):
    import torch
    from transformers import AutoConfig
    from models.modeling_model.modeling_joint_dit import JointDiTConfig, JointDiTForCausalLM
    from utils.selfless_flow_optimizer import optimizer_parameter_role
    payload = AutoConfig.from_pretrained(config.model.model_path, local_files_only=True).to_dict()
    payload.update(OmegaConf.to_container(config.model, resolve=True))
    payload.pop("model_type", None)
    payload.update(mask_token_id=151669, boi_token_id=151670, eoi_token_id=151671, image_mask_token_id=151672)
    with torch.device("meta"):
        model = JointDiTForCausalLM(JointDiTConfig(**payload))
    roles = {}
    for name, parameter in model.named_parameters():
        role = optimizer_parameter_role(name)
        roles[role] = roles.get(role, 0) + parameter.numel()
    error = roles["flow_head"] / 164072976 - 1
    if abs(error) > .005:
        raise ValueError(f"Joint DiT flow parameter budget differs from B by {error:.2%}")
    return {"roles": roles, "total_parameters_before_vocab_resize": sum(roles.values()),
            "baseline_b_flow_parameters": 164072976, "flow_parameter_relative_difference": error}
