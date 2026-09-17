"""Checkpoint-specific scoring semantics shared by training and evaluation."""

S2_ATTENTION = "showo2_omni_attention"
S2_SCORING = "showo2_next_token_ar_target_aligned_v1"
S2_TEXT_TWO_STREAM_ATTENTION = "showo2_text_two_stream"
S2_TEXT_TWO_STREAM_SCORING = "showo2_text_query_same_position_v1"
S2_ATTENTION_CONTRACTS = frozenset({S2_ATTENTION, S2_TEXT_TWO_STREAM_ATTENTION})


def scoring_contract(attention_contract, legacy_contract):
    if attention_contract == S2_TEXT_TWO_STREAM_ATTENTION:
        return S2_TEXT_TWO_STREAM_SCORING
    return S2_SCORING if attention_contract == S2_ATTENTION else legacy_contract


def image_order_for_scoring(model_config, configured_order):
    if getattr(model_config, "architecture_variant", None) in {"selfless_joint_dit", "selfless_y"}:
        return "joint"
    # S2 observes the entire image block and has no random-order distribution.
    if getattr(model_config, "architecture_variant", None) == "showo2_unified":
        return "sequential"
    return configured_order


def validate_image_generation_report(report, attention_contract, *, strategy=None, architecture_variant=None):
    """Require the public image generator and its architecture-specific cache mode."""
    is_joint = (architecture_variant or report.get("architecture_variant")) == "selfless_joint_dit"
    is_y = (architecture_variant or report.get("architecture_variant")) == "selfless_y"
    use_cache = attention_contract not in S2_ATTENTION_CONTRACTS and not (is_joint or is_y)
    if report.get("generation_entry") != "model.generate" or report.get("use_cache") is not use_cache:
        raise ValueError("held-out generation entry/cache differs from the model contract")
    strategies = report.get("strategies", {})
    selected = strategies if strategy is None else {strategy: strategies.get(strategy, {})}
    if not selected:
        raise ValueError("held-out generation report has no strategies")
    for name, trace in selected.items():
        if trace.get("backbone_kv_cache_enabled") is not use_cache:
            raise ValueError(f"held-out generation cache differs from the model contract: {name}")
        expected_mode = "y_masked_token_flow" if is_y else "joint_dit_full_image_flow" if is_joint else "showo2_full_image_flow"
        if not use_cache and trace.get("generation_mode") != expected_mode:
            raise ValueError(f"held-out generation did not use its full-image flow contract: {name}")
        if is_y:
            rounds = int(trace.get("reveal_steps", 0))
            solver = trace.get("solver")
            expected_calls = rounds * int(trace.get("steps", 0)) * (2 if solver == "heun" else 1)
            if rounds <= 0 or solver not in {"heun", "euler"} or trace.get("backbone_calls") != rounds or trace.get("flow_head_calls") != expected_calls:
                raise ValueError("Y must refresh the backbone once per reveal round")
        if is_joint:
            solver = trace.get("solver", report.get("solver", report.get("flow_solver")))
            expected_calls = int(trace.get("steps", 0)) * (2 if solver == "heun" else 1)
            if solver not in {"euler", "heun"} or expected_calls <= 0 or trace.get("backbone_calls") != 1 or trace.get("flow_head_calls") != expected_calls:
                raise ValueError("Joint DiT must run one backbone pass and solver-specific head evaluations")
    return use_cache


def validate_formal_image_order_scoring(manifest, scoring):
    """Accept exact S2 AR scores or the formal Selfless MC64 estimator."""
    attention = manifest.get("dual_stream_attention_contract")
    is_s2 = attention in S2_ATTENTION_CONTRACTS
    architecture = manifest.get("architecture_variant")
    is_joint = architecture in {"selfless_joint_dit", "selfless_y"}
    if is_joint:
        for value in (manifest, scoring):
            if (value.get("architecture_variant") != architecture
                    or value.get("dual_stream_attention_contract") != "xlnet_content_diagonal"
                    or value.get("image_sigma_order") != "joint"
                    or value.get("image_order_mc_contract") != "not_applicable_joint_image"):
                raise ValueError("Joint DiT scoring must use the fixed whole-image contract")
    if is_s2:
        if (
            scoring.get("dual_stream_attention_contract") != attention
            or manifest.get("scoring_contract") != scoring_contract(attention, None)
            or scoring.get("contract") != scoring_contract(attention, None)
            or any(value.get("image_order_mc_contract") != "not_applicable_full_image_ar"
                   for value in (manifest, scoring))
        ):
            raise ValueError("S2 benchmark scoring must use the exact full-image AR contract")
    elif scoring.get("dual_stream_attention_contract") in S2_ATTENTION_CONTRACTS:
        raise ValueError("benchmark manifest and scoring architecture differ")
    expected = 1 if is_s2 or is_joint else 64
    for value in (manifest, scoring):
        if int(value.get("mc_samples", -1)) != expected:
            raise ValueError(f"formal benchmark requires mc_samples={expected} for its model contract")
    return expected
