"""Frozen B-aligned training contract for the two Show-o2-style experiments."""
from pathlib import Path
from copy import deepcopy

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/protocols/showo2_unified_100b_ascend64.yaml"


def config_path(variant):
    if variant not in {"single", "dual-siglip", "single-text-two-stream"}:
        raise ValueError(f"Unknown S2 arm: {variant}")
    return f"configs/selfless/unified_s2_{variant.replace('-', '_')}_100b_ascend64.yaml"


def _plain(value):
    return OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value


def validate_s2_config(config):
    protocol = OmegaConf.load(PROTOCOL)
    text_two_stream = config.model.get("dual_stream_attention_contract") == "showo2_text_two_stream"
    for field, expected in protocol.frozen_from_b.items():
        actual = OmegaConf.select(config, field)
        if _plain(actual) != _plain(expected):
            raise ValueError(f"S2/B frozen contract differs at {field}: {actual!r} != {expected!r}")
    required = {"architecture_variant": "showo2_unified", "training_objective": "showo2_full_image_flow",
        "dual_stream_attention_contract": "showo2_text_two_stream" if text_two_stream else "showo2_omni_attention",
        "flow_head_attention_contract": "showo2_omni_attention",
        "flow_condition_contract": "backbone_noisy_image_hidden", "s2_semantic_depth": 26,
        "s2_semantic_width": 1152, "s2_semantic_intermediate": 4304, "s2_semantic_heads": 16,
        "s2_flow_intermediate": 1472, "s2_flow_head_dim": 64, "s2_initialization_seed": 42}
    for field, expected in required.items():
        if config.model.get(field) != expected:
            raise ValueError(f"S2 architecture mismatch: {field}")
    if float(config.optimizer.params.semantic_learning_rate) != 2e-6:
        raise ValueError("SigLIP parameters require their declared 2e-6 LR")
    if config.model.get("s2_mc_batch_size", 1) not in (1, 2, 4):
        raise ValueError("S2 MC execution batch size must be 1, 2 or 4")
    if config.model.get("s2_backbone_checkpoint_every", 1) not in (1, 2):
        raise ValueError("S2 backbone checkpoint interval must be 1 or 2")
    for field in ("training.use_gradient_checkpointing", "model.image_flow_grad_checkpointing",
                  "model.s2_full_prediction_checkpointing", "model.s2_semantic_gradient_checkpointing"):
        if type(OmegaConf.select(config, field, default=True)) is not bool:
            raise ValueError(f"S2 checkpointing control must be boolean: {field}")
    if config.training.get("save_image_flow_adapter") or config.experiment.get("save_final_image_flow_adapter"):
        raise ValueError("S2 saves complete HF/raw/EMA states; Selfless-only adapters are incompatible")
    variant = "dual-siglip" if bool(config.model.s2_use_siglip) else "single"
    if text_two_stream:
        if config.model.s2_use_siglip:
            raise ValueError("S2 text two-stream ablation requires the single visual frontend")
        variant = "single-text-two-stream"
    return {"variant": variant, "run_project": str(config.experiment.project),
            "project": protocol.projects[variant], "frozen_fields_checked": len(protocol.frozen_from_b),
            "world_size": 64, "optimizer_steps": 95415, "flow_mc_samples": 4,
            "nominal_text_targets_per_step": 1048064, "image_exposures_per_task": 97704960}


def parameter_report(config):
    import torch
    from models.modeling_model.modeling_showo2_unified import Showo2UnifiedConfig, Showo2UnifiedForCausalLM
    from transformers import AutoConfig
    payload = AutoConfig.from_pretrained(config.model.model_path, local_files_only=True).to_dict()
    payload.update(OmegaConf.to_container(config.model, resolve=True))
    payload.pop("model_type", None)
    payload.update(mask_token_id=151669, boi_token_id=151670, eoi_token_id=151671, image_mask_token_id=151672)
    cfg = Showo2UnifiedConfig(**payload)
    with torch.device("meta"):
        model = Showo2UnifiedForCausalLM(cfg)
    from utils.selfless_flow_optimizer import optimizer_parameter_role
    roles = {}
    for name, p in model.named_parameters():
        role = optimizer_parameter_role(name)
        roles[role] = roles.get(role, 0) + p.numel()
    reference = 164072976
    error = abs(roles["flow_head"] / reference - 1)
    if error > .01:
        raise ValueError(f"S2 flow head differs from B's parameter budget by {error:.2%}")
    return {"total_parameters_before_vocab_resize": sum(roles.values()), "roles": roles,
            "baseline_b_flow_parameters": reference, "flow_parameter_relative_difference": error}


def validate_s2_infra_migration(saved, current):
    """Allow only an explicit S2 execution change, retaining all training state.

    MC batching changes BF16 reduction order, so this is an audited numerical
    continuation, not a promise of bitwise equivalence to the previous infra.
    Every model/data/optimizer/world-size setting outside these execution controls
    remains subject to the strict readable resume contract.
    """
    old, new = deepcopy(saved), deepcopy(current)
    for contract in (old, new):
        if not isinstance(contract, dict) or contract.get("model", {}).get("architecture_variant") != "showo2_unified":
            raise RuntimeError("Infra migration is restricted to Show-o2 unified checkpoints")
        if contract["model"].get("training_objective") != "showo2_full_image_flow":
            raise RuntimeError("Infra migration requires the unchanged S2 objective")
    controls = {"model": {"s2_mc_batch_size": 1, "s2_full_prediction_checkpointing": True,
                "image_flow_grad_checkpointing": True, "s2_semantic_gradient_checkpointing": True,
                "s2_backbone_checkpoint_every": 1},
                "training": {"use_gradient_checkpointing": True}}
    changes = []
    for section, fields in controls.items():
        for field, default in fields.items():
            before, after = old[section].pop(field, default), new[section].pop(field, default)
            if field == "s2_mc_batch_size":
                if type(before) is not int or type(after) is not int or before not in (1, 2, 4) or after not in (1, 2, 4):
                    raise RuntimeError("Infra migration requires 1, 2 or 4 MC draws per execution batch")
            elif field == "s2_backbone_checkpoint_every":
                if type(before) is not int or type(after) is not int or before not in (1, 2) or after not in (1, 2):
                    raise RuntimeError("Infra migration backbone checkpoint interval must be 1 or 2")
            elif type(before) is not bool or type(after) is not bool:
                raise RuntimeError("Infra migration checkpointing controls must be boolean")
            if before != after:
                changes.append({"field": f"{section}.{field}", "before": before, "after": after})
    if old != new:
        raise RuntimeError("S2 infra resume differs outside the allowed execution controls")
    return changes
