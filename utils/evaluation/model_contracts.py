"""Checkpoint-specific scoring semantics shared by training and evaluation."""

S2_ATTENTION = "showo2_omni_attention"
S2_SCORING = "showo2_next_token_ar_target_aligned_v1"


def scoring_contract(attention_contract, legacy_contract):
    return S2_SCORING if attention_contract == S2_ATTENTION else legacy_contract


def image_order_for_scoring(model_config, configured_order):
    if getattr(model_config, "architecture_variant", None) == "selfless_joint_dit":
        return "joint"
    # S2 observes the entire image block and has no random-order distribution.
    if getattr(model_config, "architecture_variant", None) == "showo2_unified":
        return "sequential"
    return configured_order


def validate_image_generation_report(report, attention_contract, *, strategy=None, architecture_variant=None):
    """Require the public image generator and its architecture-specific cache mode."""
    is_joint = (architecture_variant or report.get("architecture_variant")) == "selfless_joint_dit"
    use_cache = attention_contract != S2_ATTENTION and not is_joint
    if report.get("generation_entry") != "model.generate" or report.get("use_cache") is not use_cache:
        raise ValueError("held-out generation entry/cache differs from the model contract")
    strategies = report.get("strategies", {})
    selected = strategies if strategy is None else {strategy: strategies.get(strategy, {})}
    if not selected:
        raise ValueError("held-out generation report has no strategies")
    for name, trace in selected.items():
        if trace.get("backbone_kv_cache_enabled") is not use_cache:
            raise ValueError(f"held-out generation cache differs from the model contract: {name}")
        expected_mode = "joint_dit_full_image_flow" if is_joint else "showo2_full_image_flow"
        if not use_cache and trace.get("generation_mode") != expected_mode:
            raise ValueError(f"held-out generation did not use its full-image flow contract: {name}")
        if is_joint and (trace.get("backbone_calls") != 1 or trace.get("flow_head_calls") != trace.get("steps")):
            raise ValueError("Joint DiT must run one backbone pass and one head evaluation per step")
    return use_cache


def validate_formal_image_order_scoring(manifest, scoring):
    """Accept exact S2 AR scores or the formal Selfless MC64 estimator."""
    is_s2 = manifest.get("dual_stream_attention_contract") == S2_ATTENTION
    is_joint = manifest.get("architecture_variant") == "selfless_joint_dit"
    if is_joint:
        for value in (manifest, scoring):
            if (value.get("architecture_variant") != "selfless_joint_dit"
                    or value.get("dual_stream_attention_contract") != "xlnet_content_diagonal"
                    or value.get("image_sigma_order") != "joint"
                    or value.get("image_order_mc_contract") != "not_applicable_joint_image"):
                raise ValueError("Joint DiT scoring must use the fixed whole-image contract")
    if is_s2:
        if (
            scoring.get("dual_stream_attention_contract") != S2_ATTENTION
            or manifest.get("scoring_contract") != S2_SCORING
            or scoring.get("contract") != S2_SCORING
            or any(value.get("image_order_mc_contract") != "not_applicable_full_image_ar"
                   for value in (manifest, scoring))
        ):
            raise ValueError("S2 benchmark scoring must use the exact full-image AR contract")
    elif scoring.get("dual_stream_attention_contract") == S2_ATTENTION:
        raise ValueError("benchmark manifest and scoring architecture differ")
    expected = 1 if is_s2 or is_joint else 64
    for value in (manifest, scoring):
        if int(value.get("mc_samples", -1)) != expected:
            raise ValueError(f"formal benchmark requires mc_samples={expected} for its model contract")
    return expected
