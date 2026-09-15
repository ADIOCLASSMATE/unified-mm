"""F scaling arms matched to the B depth-16 and depth-30 parameter budgets."""

from __future__ import annotations

from omegaconf import OmegaConf

from utils.experiment_registry import experiment_identity
from utils.flow_head_scaling import BASE_CONFIG, HEAD_PARAMETERS as B_HEAD_PARAMETERS
from utils.flow_head_scaling import NON_HEAD_PARAMETERS


WIDTHS = {16: 1960, 30: 1968}
HEAD_PARAMETERS = {16: 321_655_616, 30: 595_579_792}
MAX_PARAMETER_RELATIVE_ERROR = 0.005


def config_path(depth: int) -> str:
    if depth not in WIDTHS:
        raise ValueError("F flow-head scaling depth must be 16 or 30")
    return f"configs/selfless/unified_f_on_b_flow_depth{depth}_100b_ascend64.yaml"


def run_project(depth: int) -> str:
    return f"unified-f-on-b-flowdepth{depth}-0p6b-100b-imagenet-split-s42-r1"


def expected_config(depth: int):
    config_path(depth)
    config = OmegaConf.load(BASE_CONFIG)
    config.model.architecture_variant = "positionwise_flow_head_on_b"
    config.model.flow_head_attention_contract = "not_applicable"
    config.model.flow_condition_contract = "not_applicable"
    config.model.image_flow_depth = depth
    config.model.image_flow_width = WIDTHS[depth]
    config.model.image_flow_grad_checkpointing = True
    config.model.image_flow_share_content = False
    config.model.positionwise_reference_flow_width = 1280
    config.model.positionwise_reference_flow_depth = depth
    config.model.positionwise_max_parameter_relative_error = MAX_PARAMETER_RELATIVE_ERROR
    config.experiment.project = run_project(depth)
    config.experiment.name = run_project(depth)
    config.evaluation.checkpoint = f"output/{run_project(depth)}/hf_model-final-ema"
    return config


def _differences(actual, reference, prefix=""):
    if isinstance(actual, dict) and isinstance(reference, dict):
        result = []
        for key in sorted(actual.keys() | reference.keys()):
            path = f"{prefix}.{key}" if prefix else key
            if key not in actual or key not in reference:
                result.append(path)
            else:
                result.extend(_differences(actual[key], reference[key], path))
        return result
    return [] if actual == reference else [prefix]


def validate_scaling_config(config) -> dict:
    depth = int(config.model.image_flow_depth)
    expected = expected_config(depth)
    actual = OmegaConf.to_container(config, resolve=True)
    reference = OmegaConf.to_container(expected, resolve=True)
    experiment_identity(run_project(depth), actual)
    actual["experiment"].pop("identity", None)
    reference["experiment"].pop("identity", None)
    changed = _differences(actual, reference)
    if changed:
        raise ValueError("F scaling arm differs from the matched recipe: " + ", ".join(changed))
    width = WIDTHS[depth]
    count = (3 + 5 * depth) * width**2 + (1318 + 7 * depth) * width + 16
    relative_error = abs(count - B_HEAD_PARAMETERS[depth]) / B_HEAD_PARAMETERS[depth]
    if count != HEAD_PARAMETERS[depth] or relative_error > MAX_PARAMETER_RELATIVE_ERROR:
        raise ValueError("F scaling head parameter budget is not matched to B")
    return {
        "schema": "unified_f_on_b_flow_head_scaling_v1",
        "baseline_config": BASE_CONFIG,
        "baseline_f_run": "unified-f-on-b-0p6b-100b-imagenet-split-s42-r1",
        "run_project": run_project(depth),
        "architecture": "positionwise_flow_head_on_b",
        "depth": depth, "width": width,
        "head_parameters": count,
        "total_parameters": count + NON_HEAD_PARAMETERS,
        "reference_depth": depth, "reference_width": 1280,
        "reference_head_parameters": B_HEAD_PARAMETERS[depth],
        "parameter_relative_error": relative_error,
        "max_parameter_relative_error": MAX_PARAMETER_RELATIVE_ERROR,
        "image_flow_grad_checkpointing": True,
        "image_flow_share_content": False,
        "scientific_difference_from_f": ["model.image_flow_depth", "model.image_flow_width"],
        "world_size": 64, "max_train_steps": 95415,
        "target_text_tokens": 100_000_000_000,
    }
