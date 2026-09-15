"""B's flow-head modules with a single, bidirectional noisy-image stream."""
from functools import partial

import torch
from torch.utils.checkpoint import checkpoint

from .image_flow_loss import ContextualFlowTransformerHead


class JointSingleStreamBHead(ContextualFlowTransformerHead):
    """Reuse every B head parameter; each layer's Q/K/V read the same x_t stream.

    B's separate Q/KV LayerNorms, Q-only attention modulation, gated SiLU MLP,
    2D RoPE and final AdaLN are retained. K/V are rebuilt from the current
    noisy hidden state at every layer and ODE evaluation. No clean target
    content or second hidden stream is constructed.
    """

    def __init__(self, config):
        super().__init__(
            in_channels=config.image_latent_dim,
            model_channels=config.image_flow_width,
            out_channels=config.image_latent_dim,
            z_channels=config.hidden_size,
            num_res_blocks=config.image_flow_depth,
            grad_checkpointing=getattr(config, "image_flow_grad_checkpointing", True),
            image_tokens_per_img=config.image_tokens_per_img,
            endpoint_time=getattr(config, "image_flow_time_scale", 1000.0),
            flow_head_attention_contract="xlnet_content_diagonal",
            conditioning_mode="adaln",
        )
        self.flow_head_attention_contract = "joint_bidirectional"

    def position_contract(self):
        return {**super().position_contract(), "architecture": "single_stream_b_head",
                "cross_token_attention": True, "shared_time_per_image": True,
                "uses_clean_target_content": False}

    @staticmethod
    def cache_contract():
        return {"schema": "joint_b_head_fixed_condition_v1", "content_cache": False,
                "backbone_condition_cached": True, "refresh_backbone_each_velocity": False,
                "head_kv_source": "current_noisy_stream", "conditioning": "backbone_plus_time_adaln"}

    @staticmethod
    def _single_stream_block(x, modulation, *, block, positions, rope, mask, layout):
        kv = block.prepare_cross_cache(x, positions, rope, input_layout=layout)
        return block(x, modulation, layer_cache=kv, context_mask=mask,
                     query_positions=positions, query_rope=rope)

    def forward(self, x_t, times, condition):
        if x_t.ndim != 3 or x_t.shape[:2] != condition.shape[:2] or x_t.shape[1] != self.image_tokens_per_img:
            raise ValueError("Joint B head requires aligned, complete [batch, image_tokens, channels] grids")
        if times.shape != (x_t.shape[0],):
            raise ValueError("Joint B head requires one shared time per image")
        dtype = self.input_proj.weight.dtype
        hidden = self.input_proj(x_t.to(dtype))
        modulation = self.cond_embed(condition.to(dtype)) + self._shape_time(
            times * self.endpoint_time, times.shape)[:, None, :]
        positions = torch.arange(self.image_tokens_per_img, device=hidden.device)[None].expand(hidden.shape[0], -1)
        rope = self._build_rope(positions, dtype)
        mask = self.blocks[0].prepare_context_mask(
            None, hidden.shape[0], self.image_tokens_per_img, self.image_tokens_per_img, hidden.device)
        layout = "BSND" if hidden.device.type == "npu" else "BNSD"
        for block in self.blocks:
            forward = partial(self._single_stream_block, block=block, positions=positions,
                              rope=rope, mask=mask, layout=layout)
            if self.grad_checkpointing and self.training and torch.is_grad_enabled():
                hidden = checkpoint(forward, hidden, modulation, use_reentrant=False)
            else:
                hidden = forward(hidden, modulation)
        return self.final_layer(hidden, modulation)
