"""Inference-only reveal-order policies; scores are proxies, not probabilities."""

CONFIDENCE_BLOCK_SIZE = 16
CONFIDENCE_PROBE_DT = 0.1
CONFIDENCE_STRATEGIES = {
    "confidence_cfg": "Within each next 16 Halton positions, lower normalized conditional/unconditional velocity disagreement first.",
    "confidence_cfg_reverse": "Same disagreement score, higher first; a direction control for guidance sensitivity.",
    "confidence_stability": "Within each next 16 Halton positions, lower normalized guided velocity change over a 0.1 Euler probe first.",
    "confidence_halton": "Run the same two velocity probes but retain Halton order; cache correctness and compute control.",
}


def checkpoint_generation_contract(config):
    """Readable checkpoint-owned fields, including legacy default semantics."""
    def get(key, default):
        return config.get(key, default) if isinstance(config, dict) else getattr(config, key, default)
    architecture = get("architecture_variant", "selfless_contextual")
    positionwise = architecture == "positionwise_flow_head_on_b"
    return {
        "architecture": architecture,
        "backbone_attention": get("dual_stream_attention_contract", "selfless_strict"),
        "flow_attention": get("flow_head_attention_contract", "not_applicable" if positionwise else "selfless_strict"),
        "flow_condition": ("not_applicable" if positionwise else get("dynamic_xt_flow_condition_contract",
            get("flow_condition_contract", "backbone_xt_shared_query_content"))),
        "training_image_sigma_order": get("training_image_sigma_order", "random"),
    }


def order_policy(strategy):
    if strategy in CONFIDENCE_STRATEGIES:
        return {
            "description": CONFIDENCE_STRATEGIES[strategy],
            "candidate_order": "spatial_halton", "candidate_count": CONFIDENCE_BLOCK_SIZE,
            "refresh_every_reveals": CONFIDENCE_BLOCK_SIZE,
            "probe_t": 0.0, "probe_dt": CONFIDENCE_PROBE_DT,
            "score_epsilon": 1e-8, "calibrated_probability": False,
            "uses_target_latents": False, "uses_training_sigma_order": False,
            "noise": "same canonical per-position noise as final decoding",
            "commit": "Only completed content is cached; probe proposals are discarded.",
        }
    return {"description": strategy, "uses_target_latents": False,
            "uses_training_sigma_order": strategy in {"sigma", "sigma_replay", "causal_sigma"}}


def confidence_scores(strategy, conditional_velocity, unconditional_velocity, *, cfg, next_guided_velocity=None):
    """One scalar per candidate, with all arithmetic in FP32."""
    vc, vu = conditional_velocity.float(), unconditional_velocity.float()
    if strategy == "confidence_stability":
        if next_guided_velocity is None:
            raise ValueError("stability scoring requires the second velocity")
        v = vu + float(cfg) * (vc - vu)
        return (next_guided_velocity.float() - v).square().mean(-1) / (v.square().mean(-1) + 1e-8)
    return (vc - vu).square().mean(-1) / (0.5 * (vc.square().mean(-1) + vu.square().mean(-1)) + 1e-8)
