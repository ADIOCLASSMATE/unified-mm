"""Generation behavior for ablation F (position-wise flow head on B).

The generic B cache/reveal state machine remains the source of truth.  This
module only owns F's model-specific trace contract, keeping inference identity
out of both the baseline model and the F training implementation.
"""

from __future__ import annotations

import torch


class PositionwiseFlowOnBGenerationMixin:
    """F-only generation metadata over the baseline-B state machine."""

    @torch.no_grad()
    def generate_image(self, *args, **kwargs):
        return_trace = bool(kwargs.get("return_trace", False))
        result = super().generate_image(*args, **kwargs)
        if not return_trace or result is None:
            return result
        generated, trace = result
        trace["architecture_variant"] = "positionwise_flow_head_on_b"
        trace["flow_head_architecture"] = "positionwise_adaln_mlp"
        trace["flow_head_attention_contract"] = "not_applicable"
        trace["flow_head_content_stream"] = False
        trace["flow_head_consumes_prior_latents"] = False
        trace["flow_content_cache_peak_bytes_per_sample"] = 0
        trace["flow_cfg_content_cache_divergence_by_layer"] = None
        trace["flow_head_parameter_count"] = int(
            self.positionwise_flow_head_parameter_count
        )
        trace["reference_contextual_flow_head_parameter_count"] = int(
            self.reference_contextual_flow_head_parameter_count
        )
        return generated, trace


__all__ = ["PositionwiseFlowOnBGenerationMixin"]
