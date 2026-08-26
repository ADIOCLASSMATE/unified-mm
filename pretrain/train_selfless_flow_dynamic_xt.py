"""Dedicated training entry for the Dynamic-XT ImageNet successor backbone."""

from __future__ import annotations

import warnings

import torch

from models.modeling_model.modeling_selfless_flow_dynamic_xt import (
    DynamicXtQwen3ForCausalLM,
    SelflessFlowDynamicXtConfig,
)
from pretrain.train_selfless_flow import main as run_selfless_training
from utils.utils import load_model_tokenizer

DYNAMIC_XT_CONTRACT = "backbone_single_flow_state_v2"


def load_dynamic_xt_model_tokenizer(
    config,
    logger=None,
    model_dtype: torch.dtype = torch.bfloat16,
):
    contract = str(config.model.get("dynamic_xt_contract", ""))
    if contract != DYNAMIC_XT_CONTRACT:
        raise ValueError(
            "Dynamic-XT training requires model.dynamic_xt_contract="
            f"{DYNAMIC_XT_CONTRACT!r}, got {contract!r}"
        )
    if config.model.get("architecture_variant", None) is not None:
        raise ValueError(
            "Dynamic-XT must use its dedicated model class, not architecture_variant"
        )
    model, tokenizer = load_model_tokenizer(
        config=config,
        logger=logger,
        model_dtype=model_dtype,
        model_class=DynamicXtQwen3ForCausalLM,
        model_config_class=SelflessFlowDynamicXtConfig,
    )
    if logger is not None:
        logger.info(
            "Dynamic-XT contract: model_type=%s, extra_parameters=%s",
            model.model_type,
            f"{model.dynamic_xt_parameter_count():,}",
        )
    return model, tokenizer


if __name__ == "__main__":
    warnings.filterwarnings(
        "error",
        message="None of the inputs have requires_grad=True",
    )
    run_selfless_training(model_loader=load_dynamic_xt_model_tokenizer)
