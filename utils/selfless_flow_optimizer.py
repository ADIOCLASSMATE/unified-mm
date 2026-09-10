"""Parameter-group rules for Selfless-Flow AdamW."""


NO_DECAY_NAME_FRAGMENTS = (
    "bias",
    "layer_norm.weight",
    "layernorm.weight",
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "norm.weight",
    "embed_tokens.weight",
    "lm_head.weight",
)


def optimizer_parameter_role(name: str) -> str:
    """Return the non-overlapping LR role for a canonical parameter name."""

    if name.startswith("model.semantic_encoder."):
        return "semantic_pretrained"
    if name.startswith(("model.semantic_input_proj.", "model.image_fusion.", "model.image_time_embedder.")):
        return "image_projector"
    if name.startswith("image_flow_head."):
        return "flow_head"
    if "image_token_embedder" in name or name.startswith(
        "image_flow_condition_proj."
    ):
        return "image_projector"
    if name in {"model.embed_tokens.weight", "lm_head.weight"}:
        # Qwen ties these names to one complete matrix.  AdamW cannot assign a
        # separate LR to only the special-token rows of a single parameter.
        return "tied_lm_head_embedding"
    return "backbone"


def learning_rate_for_parameter(
    name: str,
    *,
    backbone_lr: float,
    flow_lr: float,
    projector_lr: float,
    special_token_lr: float,
    semantic_lr: float | None = None,
) -> float:
    role = optimizer_parameter_role(name)
    return {
        "backbone": float(backbone_lr),
        "flow_head": float(flow_lr),
        "image_projector": float(projector_lr),
        "tied_lm_head_embedding": float(special_token_lr),
        "semantic_pretrained": float(backbone_lr if semantic_lr is None else semantic_lr),
    }[role]


def weight_decay_for_parameter(
    name: str,
    global_weight_decay: float,
    flow_weight_decay: float,
) -> float:
    """Apply decay to flow matrices, but never to bias/norm parameters."""

    if any(fragment in name for fragment in NO_DECAY_NAME_FRAGMENTS):
        return 0.0
    if optimizer_parameter_role(name) == "image_projector":
        return 0.0
    if name.startswith("model.semantic_encoder.") and (
        "layer_norm1.weight" in name or "layer_norm2.weight" in name or "position_embedding.weight" in name
    ):
        return 0.0
    if name.startswith("image_flow_head."):
        return float(flow_weight_decay)
    if (
        "image_token_embedder" in name
        or name.startswith("image_flow_condition_proj.")
    ):
        return 0.0
    return float(global_weight_decay)
