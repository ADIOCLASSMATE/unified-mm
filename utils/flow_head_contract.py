"""Shared interpretation of contextual and position-wise flow-head contracts."""

from __future__ import annotations


POSITIONWISE_ON_B_ARCHITECTURE = "positionwise_flow_head_on_b"
POSITIONWISE_FLOW_HEAD_ATTENTION_CONTRACT = "not_applicable"
CONTEXTUAL_FLOW_HEAD_ATTENTION_CONTRACTS = {
    "selfless_strict",
    "xlnet_content_diagonal",
}


def _config_value(config, key: str, default=None):
    getter = getattr(config, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(config, key, default)


def validate_flow_head_attention_contract(config, *, label: str = "model") -> str:
    """Return the normalized contract, preserving legacy checkpoint meaning."""

    architecture = str(
        _config_value(config, "architecture_variant", "selfless_contextual")
    ).strip().lower()
    default_contract = (
        POSITIONWISE_FLOW_HEAD_ATTENTION_CONTRACT
        if architecture == POSITIONWISE_ON_B_ARCHITECTURE
        else "selfless_strict"
    )
    contract = str(
        _config_value(config, "flow_head_attention_contract", default_contract)
    ).strip().lower()
    if architecture == "showo2_unified":
        if contract != "showo2_omni_attention":
            raise ValueError("Show-o2 requires flow_head_attention_contract=showo2_omni_attention")
    elif architecture == POSITIONWISE_ON_B_ARCHITECTURE:
        if contract != POSITIONWISE_FLOW_HEAD_ATTENTION_CONTRACT:
            raise ValueError(
                f"{label} architecture {architecture!r} has no flow-head "
                "attention and requires flow_head_attention_contract="
                f"{POSITIONWISE_FLOW_HEAD_ATTENTION_CONTRACT!r}, got "
                f"{contract!r}"
            )
    elif contract not in CONTEXTUAL_FLOW_HEAD_ATTENTION_CONTRACTS:
        raise ValueError(
            f"{label} has unsupported flow_head_attention_contract={contract!r} "
            f"for architecture_variant={architecture!r}"
        )
    return contract


def flow_head_attention_report(config) -> dict[str, object]:
    """Describe attention without pretending F owns a content stream."""

    architecture = str(
        _config_value(config, "architecture_variant", "selfless_contextual")
    ).strip().lower()
    contract = validate_flow_head_attention_contract(config)
    if architecture == "showo2_unified":
        return {"applicable": True, "architecture": "showo2_modulated_single_stream",
                "flow_head_attention_contract": contract, "cross_token_attention": True,
                "query_stream_input": "backbone_noisy_image_hidden", "query_stream_diagonal": True,
                "content_stream_input": None, "content_stream_diagonal": None}
    if architecture == POSITIONWISE_ON_B_ARCHITECTURE:
        return {
            "applicable": False,
            "architecture": "positionwise_adaln_mlp",
            "flow_head_attention_contract": contract,
            "cross_token_attention": False,
            "query_stream_input": "x_t",
            "query_stream_diagonal": None,
            "content_stream_input": None,
            "content_stream_diagonal": None,
        }
    return {
        "applicable": True,
        "architecture": "contextual_dual_stream",
        "flow_head_attention_contract": contract,
        "cross_token_attention": True,
        "query_stream_input": "x_t",
        "query_stream_diagonal": False,
        "content_stream_input": "backbone_content_latent",
        "content_stream_diagonal": contract == "xlnet_content_diagonal",
    }


__all__ = [
    "CONTEXTUAL_FLOW_HEAD_ATTENTION_CONTRACTS",
    "POSITIONWISE_FLOW_HEAD_ATTENTION_CONTRACT",
    "POSITIONWISE_ON_B_ARCHITECTURE",
    "flow_head_attention_report",
    "validate_flow_head_attention_contract",
]
