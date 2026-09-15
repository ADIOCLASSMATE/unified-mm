"""B with S2-single-style conditioning, retaining both B streams and masks."""
from pathlib import Path

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = ROOT / "configs/selfless/unified_baseline_100b_ascend_64npu.yaml"
CONFIG = "configs/selfless/unified_b_s2_modulation_100b_ascend64.yaml"
RUN = "unified-b-s2-modulation-0p6b-100b-imagenet-split-s42-r1"


def expected_config():
    config = OmegaConf.load(BASE_CONFIG)
    config.experiment.project = RUN
    config.experiment.name = RUN
    config.experiment.identity = dict(id="b-s2-modulation", label="B + S2-single modulation",
                                      group="ablation", purpose="formal")
    config.model.image_flow_conditioning_mode = "s2_input"
    config.experiment.validation_generation = dict(enabled=True, samples=16, seed=42,
        prompt_file="configs/protocols/unified_qualitative_prompts_v1.json",
        cfg=3.5, steps=10, solver="heun", vae_module_root="public/code/mar",
        vae_path="public/vae/mar-kl16/kl16.ckpt", vae_scaling_factor=0.2325)
    config.evaluation.checkpoint = f"output/{RUN}/hf_model-final-ema"
    return config


def validate_b_s2_modulation_config(config):
    if OmegaConf.to_container(config, resolve=True) != OmegaConf.to_container(expected_config(), resolve=True):
        raise ValueError("B + S2 modulation configuration differs from its B-aligned contract")
    return dict(schema="b_s2_modulation_experiment_v1", method="B + S2-single modulation",
        run_project=RUN, world_size=64, optimizer_steps=95415, flow_mc_samples=4,
        backbone_streams=2, head_streams=2, image_sigma_order="random",
        conditioning_mode="s2_input", query_input="latent_proj(xt) + cond_proj(backbone_xt)",
        content_input="latent_proj(x0) + cond_proj(backbone_x0)",
        query_adaln="time_only", content_adaln="endpoint_time_only",
        query_visibility="sigma_lt", content_visibility="sigma_leq",
        time_in_backbone=False, shared_time_per_image=False,
        solver="heun", sampling_steps=10, image_input_noise_strength=0.01,
        validation_generation=dict(every_validation=True, images=16, weights="ema",
                                   seed=42, cfg=3.5, solver="heun", steps=10,
                                   order="spatial_halton", use_cache=True))


def parameter_report(config):
    import torch
    from transformers import AutoConfig, Qwen3Config
    from models.modeling_model.modeling_selfless_flow import Qwen3ForCausalLM
    from utils.selfless_flow_optimizer import optimizer_parameter_role

    payload = AutoConfig.from_pretrained(config.model.model_path, local_files_only=True).to_dict()
    payload.update(OmegaConf.to_container(config.model, resolve=True))
    payload.pop("model_type", None)
    payload.update(mask_token_id=151669, boi_token_id=151670, eoi_token_id=151671, image_mask_token_id=151672)
    with torch.device("meta"):
        model = Qwen3ForCausalLM(Qwen3Config(**payload))
    roles = {}
    for name, parameter in model.named_parameters():
        role = optimizer_parameter_role(name)
        roles[role] = roles.get(role, 0) + parameter.numel()
    if roles["flow_head"] != 164072976:
        raise ValueError(f"B modulation ablation changed head parameter count: {roles['flow_head']}")
    return dict(roles=roles, total_parameters_before_vocab_resize=sum(roles.values()),
                baseline_b_flow_parameters=164072976, flow_parameter_relative_difference=0)
