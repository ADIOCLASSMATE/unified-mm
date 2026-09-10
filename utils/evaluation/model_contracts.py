"""Checkpoint-specific scoring semantics shared by training and evaluation."""

S2_ATTENTION = "showo2_omni_attention"
S2_SCORING = "showo2_next_token_ar_target_aligned_v1"


def scoring_contract(attention_contract, legacy_contract):
    return S2_SCORING if attention_contract == S2_ATTENTION else legacy_contract


def image_order_for_scoring(model_config, configured_order):
    # S2 observes the entire image block and has no random-order distribution.
    if getattr(model_config, "architecture_variant", None) == "showo2_unified":
        return "sequential"
    return configured_order
