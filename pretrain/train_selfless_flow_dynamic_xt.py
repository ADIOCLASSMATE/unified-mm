"""Strict training entry for unified ablation D on baseline B."""

from __future__ import annotations

import warnings

import torch

from models.modeling_model.modeling_selfless_flow_dynamic_xt import (
    DYNAMIC_XT_ARCHITECTURE,
    DYNAMIC_XT_ATTENTION_CONTRACT,
    DYNAMIC_XT_FLOW_BATCH_MUL,
    DYNAMIC_XT_FLOW_HEAD_ATTENTION_CONTRACT,
    DYNAMIC_XT_T2I_GRADIENT_CHECKPOINTING,
)
from pretrain.train_selfless_flow import main as run_selfless_training
from utils.utils import load_model_tokenizer

def load_dynamic_xt_model_tokenizer(
    config,
    logger=None,
    model_dtype: torch.dtype = torch.bfloat16,
):
    architecture = str(
        config.model.get("architecture_variant", "")
    ).strip().lower()
    if architecture != DYNAMIC_XT_ARCHITECTURE:
        raise ValueError(
            "Ablation D requires model.architecture_variant='dynamic_xt', "
            f"got {architecture!r}"
        )
    objective = str(config.model.get("training_objective", "")).strip().lower()
    if objective != "selfless_dual_stream":
        raise ValueError(
            "Ablation D requires model.training_objective="
            "'selfless_dual_stream'"
        )
    attention_contract = str(
        config.model.get("dual_stream_attention_contract", "")
    ).strip().lower()
    if attention_contract != DYNAMIC_XT_ATTENTION_CONTRACT:
        raise ValueError(
            "Ablation D is based on B and requires "
            "model.dual_stream_attention_contract="
            f"'{DYNAMIC_XT_ATTENTION_CONTRACT}', got {attention_contract!r}"
        )
    flow_head_attention_contract = str(
        config.model.get("flow_head_attention_contract", "")
    ).strip().lower()
    if (
        flow_head_attention_contract
        != DYNAMIC_XT_FLOW_HEAD_ATTENTION_CONTRACT
    ):
        raise ValueError(
            "Ablation D is based on corrected B and requires "
            "model.flow_head_attention_contract="
            f"'{DYNAMIC_XT_FLOW_HEAD_ATTENTION_CONTRACT}', got "
            f"{flow_head_attention_contract!r}"
        )
    flow_batch_mul = int(config.model.get("image_flow_batch_mul", -1))
    if flow_batch_mul != DYNAMIC_XT_FLOW_BATCH_MUL:
        raise ValueError(
            "Ablation D must preserve image_flow_batch_mul=4, "
            f"got {flow_batch_mul}"
        )
    if bool(config.training.get("use_gradient_checkpointing", False)):
        raise ValueError(
            "Ablation D keeps global gradient checkpointing disabled; only "
            "its T2I Dynamic-XT decoder layers may be checkpointed"
        )
    selective_checkpointing = bool(
        config.model.get("dynamic_xt_t2i_gradient_checkpointing", False)
    )
    if selective_checkpointing is not DYNAMIC_XT_T2I_GRADIENT_CHECKPOINTING:
        raise ValueError(
            "Ablation D requires "
            "model.dynamic_xt_t2i_gradient_checkpointing=true"
        )
    model, tokenizer = load_model_tokenizer(
        config=config,
        logger=logger,
        model_dtype=model_dtype,
    )
    if getattr(model, "architecture_variant", None) != DYNAMIC_XT_ARCHITECTURE:
        raise TypeError(
            "Dynamic-XT loader selected the wrong model implementation: "
            f"{type(model).__name__}"
        )
    if logger is not None:
        logger.info(
            "Ablation D contract: model_type=%s, base=B, "
            "image_flow_batch_mul=%s, t2i_only_checkpointing=%s, "
            "extra_parameters=%s",
            model.model_type,
            model.image_flow_batch_mul,
            selective_checkpointing,
            f"{model.dynamic_xt_parameter_count():,}",
        )
    return model, tokenizer


if __name__ == "__main__":
    warnings.filterwarnings(
        "error",
        message="None of the inputs have requires_grad=True",
    )
    run_selfless_training(model_loader=load_dynamic_xt_model_tokenizer)
