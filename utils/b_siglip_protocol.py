"""B's unchanged training/flow contract with one extra pretrained visual branch."""
from pathlib import Path

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/protocols/unified_b_siglip_100b_ascend64.yaml"


def config_path(variant="b-siglip"):
    if variant != "b-siglip":
        raise ValueError(f"Unknown B + SigLIP variant: {variant}")
    return "configs/selfless/unified_b_siglip_100b_ascend64.yaml"


def validate_b_siglip_config(config):
    protocol = OmegaConf.load(PROTOCOL)
    for field, expected in protocol.frozen_from_b.items():
        actual = OmegaConf.select(config, field)
        if actual != expected:
            raise ValueError(f"B + SigLIP frozen contract differs at {field}: {actual!r} != {expected!r}")
    required = {"architecture_variant": "selfless_siglip", "training_objective": "selfless_dual_stream",
        "b_siglip_width": 1152, "b_siglip_intermediate": 4304, "b_siglip_depth": 26,
        "b_siglip_heads": 16, "b_siglip_gradient_checkpointing": False, "b_siglip_initialization_seed": 42,
        "image_flow_share_content": True,
        "b_siglip_visibility_contract": "same_image_native_x0_sigma_causal"}
    for field, expected in required.items():
        if config.model.get(field) != expected:
            raise ValueError(f"B + SigLIP architecture mismatch: {field}")
    if config.optimizer.params.semantic_learning_rate != 2e-6:
        raise ValueError("SigLIP pretrained LR must match S2-dual-siglip: 2e-6")
    if (config.experiment.get("save_final_image_flow_adapter") or config.experiment.get("save_image_flow_adapter")
            or config.training.get("save_image_flow_adapter")):
        raise ValueError("B + SigLIP requires complete checkpoints including its semantic branch")
    return {"variant": "b-siglip", "run_project": str(config.experiment.project),
        "project": "random-sequence-language-modeling", "frozen_fields_checked": len(protocol.frozen_from_b),
        "world_size": 64, "optimizer_steps": 95415, "flow_mc_samples": 4,
        "nominal_text_targets_per_step": 1048064, "image_exposures_per_task": 97704960}


def parameter_report(config):
    import torch
    from transformers import AutoConfig
    from models.modeling_model.modeling_selfless_siglip import SelflessSiglipConfig, SelflessSiglipForCausalLM
    from utils.selfless_flow_optimizer import optimizer_parameter_role
    payload = AutoConfig.from_pretrained(config.model.model_path, local_files_only=True).to_dict()
    payload.update(OmegaConf.to_container(config.model, resolve=True))
    payload.pop("model_type", None)
    payload.update(mask_token_id=151669, boi_token_id=151670, eoi_token_id=151671, image_mask_token_id=151672)
    with torch.device("meta"):
        model = SelflessSiglipForCausalLM(SelflessSiglipConfig(**payload))
    roles = {}
    for name, parameter in model.named_parameters():
        role = optimizer_parameter_role(name)
        roles[role] = roles.get(role, 0) + parameter.numel()
    assert roles["flow_head"] == 164072976, "B's flow head must remain exactly unchanged"
    extra = sum(p.numel() for name, p in model.named_parameters()
                if name.startswith(("model.semantic_encoder.", "model.semantic_input_proj.", "model.image_fusion.")))
    return {"total_parameters_before_vocab_resize": sum(roles.values()), "roles": roles,
            "extra_visual_parameters": extra, "baseline_b_flow_parameters": 164072976}
