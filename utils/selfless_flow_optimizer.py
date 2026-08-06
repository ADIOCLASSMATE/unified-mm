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


def weight_decay_for_parameter(
    name: str,
    global_weight_decay: float,
    flow_weight_decay: float,
) -> float:
    """Apply decay to flow matrices, but never to bias/norm parameters."""

    if any(fragment in name for fragment in NO_DECAY_NAME_FRAGMENTS):
        return 0.0
    if name.startswith("image_flow_head."):
        return float(flow_weight_decay)
    if (
        "image_token_embedder" in name
        or name.startswith("image_flow_condition_proj.")
    ):
        return 0.0
    return float(global_weight_decay)
