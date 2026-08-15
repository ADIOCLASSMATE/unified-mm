"""Selfless-backbone ablation with a strictly position-wise flow head.

This file deliberately owns a separate model class.  The production
``Qwen3ForCausalLM`` and contextual flow-head implementation are not modified
or parameterized by this experiment.
"""

from __future__ import annotations

import torch.nn as nn

from .image_flow_loss_positionwise import PositionwiseFlowLoss
from .modeling_selfless_flow import (
    Qwen3ForCausalLM as ContextualQwen3ForCausalLM,
)
from .modeling_selfless_flow import Qwen3Model, Qwen3PreTrainedModel


class PositionwiseFlowQwen3ForCausalLM(ContextualQwen3ForCausalLM):
    """Random-sigma selfless backbone plus MAR/NextStep-style flow MLP."""

    architecture_variant = "positionwise_selfless"

    def __init__(self, config):
        # Construct the shared backbone directly so the large contextual head is
        # never instantiated and discarded during ablation startup.
        Qwen3PreTrainedModel.__init__(self, config)
        self.model = Qwen3Model(config)
        self.vocab_size = config.vocab_size
        self.image_latent_dim = int(getattr(config, "image_latent_dim", 4))
        self.image_flow_batch_mul = int(
            getattr(config, "image_flow_batch_mul", 1)
        )
        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
        )
        self.image_flow_condition_proj = nn.Linear(
            config.hidden_size,
            config.hidden_size,
            bias=True,
        )
        self.image_flow_head = PositionwiseFlowLoss(
            target_channels=self.image_latent_dim,
            z_channels=config.hidden_size,
            width=getattr(config, "image_flow_width", 1280),
            depth=getattr(config, "image_flow_depth", 8),
            num_sampling_steps=getattr(
                config,
                "image_flow_num_sampling_steps",
                "10",
            ),
            grad_checkpointing=getattr(
                config,
                "image_flow_grad_checkpointing",
                False,
            ),
            time_scale=getattr(config, "image_flow_time_scale", 1000.0),
            time_sampling=getattr(
                config,
                "image_flow_time_sampling",
                "logit_normal",
            ),
            logit_mean=getattr(config, "image_flow_logit_mean", 0.0),
            logit_std=getattr(config, "image_flow_logit_std", 1.0),
            time_eps=getattr(config, "image_flow_time_eps", 1.0e-4),
            uniform_mix=getattr(
                config,
                "image_flow_time_uniform_mix",
                0.1,
            ),
            solver=getattr(config, "image_flow_solver", "heun"),
            image_tokens_per_img=getattr(config, "image_tokens_per_img", 256),
        )

        self.post_init()
        self.reset_backbone_attention_output_gates()
        self.reset_image_modules()

    def sample_image_latents_single_stream(self, *args, **kwargs):
        return_trace = bool(kwargs.get("return_trace", False))
        result = super().sample_image_latents_single_stream(*args, **kwargs)
        if not return_trace or result is None:
            return result
        generated, trace = result
        trace["flow_head_architecture"] = "positionwise_adaln_mlp"
        trace["flow_head_consumes_prior_latents"] = False
        trace["flow_content_cache_peak_bytes_per_sample"] = 0
        trace["flow_cfg_content_cache_divergence_by_layer"] = None
        return generated, trace


__all__ = ["PositionwiseFlowQwen3ForCausalLM"]
