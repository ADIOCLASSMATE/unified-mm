"""The two depth-only B scaling arms, with the complete baseline recipe."""

from __future__ import annotations

from omegaconf import OmegaConf
from utils.experiment_registry import experiment_identity


BASE_CONFIG = "configs/selfless/unified_baseline_100b_ascend_64npu.yaml"
HEAD_PARAMETERS = {16: 321_543_696, 30: 597_117_456}
NON_HEAD_PARAMETERS = 597_116_928


def config_path(depth: int) -> str:
    if depth not in HEAD_PARAMETERS:
        raise ValueError("flow-head scaling depth must be 16 or 30")
    return f"configs/selfless/unified_b_x0_flow_depth{depth}_100b_ascend64.yaml"


def run_project(depth: int) -> str:
    return f"unified-b-x0content-flowdepth{depth}-0p6b-100b-imagenet-split-s42-r1"


def validate_scaling_config(config) -> dict:
    """Reject any scientific change beyond head depth and activation storage."""
    depth = int(config.model.image_flow_depth)
    if depth not in HEAD_PARAMETERS:
        raise ValueError("flow-head scaling depth must be 16 or 30")
    checkpointing = config.model.get("image_flow_grad_checkpointing", False)
    if checkpointing is not True:
        raise ValueError("flow-head scaling requires image_flow_grad_checkpointing=true")
    if config.model.get("image_flow_share_content", False) is not True:
        raise ValueError("flow-head scaling requires image_flow_share_content=true")
    expected = OmegaConf.load(BASE_CONFIG)
    expected.model.image_flow_depth = depth
    expected.model.image_flow_grad_checkpointing = checkpointing
    expected.model.image_flow_share_content = True
    expected.experiment.project = run_project(depth)
    expected.experiment.name = run_project(depth)
    expected.evaluation.checkpoint = f"output/{run_project(depth)}/hf_model-final-ema"
    actual_payload = OmegaConf.to_container(config, resolve=True)
    expected_payload = OmegaConf.to_container(expected, resolve=True)
    # Presentation metadata cannot affect this complete scientific comparison.
    experiment_identity(run_project(depth), actual_payload)
    actual_payload["experiment"].pop("identity", None)

    def differences(actual, reference, prefix=""):
        if isinstance(actual, dict) and isinstance(reference, dict):
            result = []
            for key in sorted(actual.keys() | reference.keys()):
                path = f"{prefix}.{key}" if prefix else key
                if key not in actual or key not in reference:
                    result.append(path)
                else:
                    result.extend(differences(actual[key], reference[key], path))
            return result
        return [] if actual == reference else [prefix]

    changed = differences(actual_payload, expected_payload)
    if changed:
        raise ValueError("scaling arm differs from B recipe: " + ", ".join(changed))
    return {
        "schema": "unified_b_x0_flow_depth_scaling_v1",
        "baseline_config": BASE_CONFIG,
        "run_project": run_project(depth),
        "depth": depth,
        "width": 1280,
        "head_parameters": HEAD_PARAMETERS[depth],
        "total_parameters": HEAD_PARAMETERS[depth] + NON_HEAD_PARAMETERS,
        "image_flow_grad_checkpointing": checkpointing,
        "image_flow_share_content": True,
        "scientific_difference": "model.image_flow_depth",
        "world_size": 64,
        "max_train_steps": 95415,
        "target_text_tokens": 100_000_000_000,
    }
