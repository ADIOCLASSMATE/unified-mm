import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from torch.utils.checkpoint import checkpoint

from .image_position_utils import (
    apply_local_row_col_rope,
    build_2d_sincos_position_embedding,
)


def _xavier_uniform_init_fp32_(tensor: torch.Tensor):
    if tensor.is_meta:
        return
    with torch.no_grad():
        value = torch.empty(tensor.shape, device=tensor.device, dtype=torch.float32)
        torch.nn.init.xavier_uniform_(value)
        tensor.copy_(value.to(dtype=tensor.dtype))


def _normal_init_fp32_(tensor: torch.Tensor, mean: float = 0.0, std: float = 1.0):
    if tensor.is_meta:
        return
    with torch.no_grad():
        value = torch.empty(tensor.shape, device=tensor.device, dtype=torch.float32)
        torch.nn.init.normal_(value, mean=mean, std=std)
        tensor.copy_(value.to(dtype=tensor.dtype))


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_freq = t_freq.to(device=self.mlp[0].weight.device, dtype=self.mlp[0].weight.dtype)
        return self.mlp(t_freq)


class FinalLayer(nn.Module):
    def __init__(self, model_channels, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(model_channels, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(model_channels, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(model_channels, 2 * model_channels, bias=True),
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


class ContextualFlowBlock(nn.Module):
    def __init__(
        self,
        channels,
        num_heads=8,
        mlp_ratio=2.0,
        dropout=0.0,
        rope_mode="none",
        rope_axis_dims=(80, 80),
        image_tokens_per_img=256,
    ):
        super().__init__()
        self.channels = int(channels)
        self.num_heads = int(num_heads)
        self.mlp_ratio = float(mlp_ratio)
        self.dropout = float(dropout)
        if self.num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if self.channels % self.num_heads != 0:
            raise ValueError(f"channels={channels} must be divisible by num_heads={num_heads}")
        self.head_dim = self.channels // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.rope_mode = str(rope_mode)
        self.rope_axis_dims = tuple(int(item) for item in rope_axis_dims)
        self.image_tokens_per_img = int(image_tokens_per_img)

        self.cross_q_norm = nn.LayerNorm(self.channels, eps=1e-6)
        self.cross_kv_norm = nn.LayerNorm(self.channels, eps=1e-6)
        self.cross_q = nn.Linear(self.channels, self.channels)
        self.cross_k = nn.Linear(self.channels, self.channels)
        self.cross_v = nn.Linear(self.channels, self.channels)
        self.cross_out = nn.Linear(self.channels, self.channels)

        self.mlp_norm = nn.LayerNorm(self.channels, eps=1e-6)
        hidden = int(self.channels * self.mlp_ratio)
        if hidden <= 0:
            raise ValueError(f"mlp_ratio must produce a positive hidden size, got {mlp_ratio}")
        self.mlp = nn.Sequential(
            nn.Linear(self.channels, hidden),
            nn.SiLU(),
            nn.Linear(hidden, self.channels),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.channels, 6 * self.channels),
        )
        self.last_gate_abs_mean = None
        self.last_gate_abs_per_token = None
        self.last_attention_gate_abs_per_token = None
        self.last_mlp_gate_abs_per_token = None
        self.collect_attention_diagnostics = False
        self.last_attention_entropy_per_token = None
        self.last_attention_distance_per_token = None
        self.last_update_rms_per_token = None

    def _split_heads(self, x):
        batch_size, seq_len, _ = x.shape
        return x.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x):
        batch_size, _, seq_len, _ = x.shape
        return x.transpose(1, 2).reshape(batch_size, seq_len, self.channels)

    def _format_context_mask(self, context_mask, batch_size, query_len, context_len, device):
        if context_mask is None:
            return torch.ones(batch_size, query_len, context_len, device=device, dtype=torch.bool)
        context_mask = context_mask.to(device=device, dtype=torch.bool)
        if context_mask.dim() == 2:
            context_mask = context_mask.unsqueeze(1)
        if context_mask.shape[1] == 1 and query_len != 1:
            context_mask = context_mask.expand(batch_size, query_len, context_len)
        if context_mask.shape != (batch_size, query_len, context_len):
            raise ValueError(
                f"context_mask must have shape {(batch_size, query_len, context_len)}, "
                f"got {tuple(context_mask.shape)}"
            )
        return context_mask

    def _maybe_apply_rope(self, x, positions):
        if self.rope_mode == "none":
            return x
        if self.rope_mode != "row_col_2d":
            raise ValueError(f"Unknown flow-head rope_mode={self.rope_mode!r}.")
        if positions is None:
            raise ValueError(
                "row_col_2d flow-head RoPE requires explicit local positions."
            )
        return apply_local_row_col_rope(
            x,
            positions,
            image_tokens_per_img=self.image_tokens_per_img,
            axis_dims=self.rope_axis_dims,
        )

    def prepare_cross_cache(self, context_hidden, context_positions=None):
        context_hidden = self.cross_kv_norm(context_hidden)
        k = self._split_heads(self.cross_k(context_hidden))
        k = self._maybe_apply_rope(k, context_positions)
        v = self._split_heads(self.cross_v(context_hidden))
        return {
            "k": k,
            "v": v,
            "context_positions": context_positions,
            "k_rotation_count": int(self.rope_mode != "none"),
        }

    def _record_attention_diagnostics(
        self,
        q,
        k,
        context_mask,
        query_positions,
        context_positions,
    ):
        if not self.collect_attention_diagnostics:
            return
        batch_size, _, query_len, _ = q.shape
        context_len = k.shape[2]
        mask = self._format_context_mask(
            context_mask,
            batch_size,
            query_len,
            context_len,
            q.device,
        )
        has_context = mask.any(dim=-1)
        safe_mask = mask.clone()
        safe_mask[~has_context] = True
        scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * self.scale
        scores = scores.masked_fill(~safe_mask.unsqueeze(1), float("-inf"))
        probabilities = torch.softmax(scores, dim=-1)
        probabilities = probabilities * has_context[:, None, :, None]
        entropy = -(
            probabilities
            * probabilities.clamp_min(torch.finfo(torch.float32).tiny).log()
        ).sum(dim=-1)
        self.last_attention_entropy_per_token = entropy.mean(dim=1).detach()

        if query_positions is None or context_positions is None:
            self.last_attention_distance_per_token = None
            return
        side = int(math.isqrt(self.image_tokens_per_img))
        query_positions = query_positions.to(device=q.device, dtype=torch.long)
        context_positions = context_positions.to(device=q.device, dtype=torch.long)
        query_row = torch.div(query_positions, side, rounding_mode="floor")
        query_col = query_positions.remainder(side)
        context_row = torch.div(context_positions, side, rounding_mode="floor")
        context_col = context_positions.remainder(side)
        distance = (
            (query_row.unsqueeze(-1) - context_row.unsqueeze(1)).float().pow(2)
            + (query_col.unsqueeze(-1) - context_col.unsqueeze(1)).float().pow(2)
        ).sqrt()
        mean_probability = probabilities.mean(dim=1)
        self.last_attention_distance_per_token = (
            (mean_probability * distance).sum(dim=-1).detach()
        )

    def _cross_attention(
        self,
        x,
        layer_cache,
        context_mask,
        query_positions=None,
        context_block_mask=None,
        use_flex_attention=False,
    ):
        if layer_cache is None:
            return None
        k = layer_cache.get("k")
        v = layer_cache.get("v")
        if k is None or v is None:
            return None
        batch_size, query_len, _ = x.shape
        if k.shape[0] != batch_size:
            raise ValueError(f"context cache batch {k.shape[0]} must match query batch {batch_size}")
        context_len = k.shape[2]
        if context_len == 0:
            return None

        q = self._split_heads(self.cross_q(x))
        q = self._maybe_apply_rope(q, query_positions)
        attn_dtype = q.dtype
        k = k.to(device=x.device, dtype=attn_dtype)
        v = v.to(device=x.device, dtype=attn_dtype)
        self._record_attention_diagnostics(
            q,
            k,
            context_mask,
            query_positions,
            layer_cache.get("context_positions"),
        )
        if use_flex_attention:
            out = flex_attention(
                query=q,
                key=k,
                value=v,
                score_mod=None,
                block_mask=context_block_mask,
                scale=self.scale,
                enable_gqa=False,
                return_lse=False,
            )
            if self.dropout > 0.0 and self.training:
                out = F.dropout(out, p=self.dropout, training=True)
            return self.cross_out(self._merge_heads(out))

        context_mask = self._format_context_mask(
            context_mask,
            batch_size,
            query_len,
            context_len,
            x.device,
        )
        has_context = context_mask.any(dim=-1)
        safe_mask = context_mask
        if not bool(has_context.all().item()):
            safe_mask = context_mask.clone()
            safe_mask[~has_context] = True
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=safe_mask.unsqueeze(1),
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
            scale=self.scale,
        )
        if not bool(has_context.all().item()):
            out = out * has_context[:, None, :, None].to(dtype=out.dtype)
        return self.cross_out(self._merge_heads(out))

    def forward(
        self,
        x,
        y,
        layer_cache=None,
        context_mask=None,
        query_positions=None,
        context_block_mask=None,
        use_flex_attention=False,
        include_mlp=True,
    ):
        self.last_attention_entropy_per_token = None
        self.last_attention_distance_per_token = None
        input_x = x
        (
            shift_cross,
            scale_cross,
            gate_cross,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.adaLN_modulation(y).chunk(6, dim=-1)

        if layer_cache is not None:
            h = modulate(self.cross_q_norm(x), shift_cross, scale_cross)
            mixed = self._cross_attention(
                h,
                layer_cache,
                context_mask,
                query_positions=query_positions,
                context_block_mask=context_block_mask,
                use_flex_attention=use_flex_attention,
            )
            if mixed is not None:
                x = x + gate_cross * mixed

        if include_mlp:
            h = modulate(self.mlp_norm(x), shift_mlp, scale_mlp)
            x = x + gate_mlp * self.mlp(h)
        attention_gate = gate_cross.detach().float().abs().mean(dim=-1)
        mlp_gate = gate_mlp.detach().float().abs().mean(dim=-1)
        gate_per_token = (
            torch.stack([attention_gate, mlp_gate], dim=0).mean(dim=0)
            if include_mlp
            else attention_gate
        )
        self.last_attention_gate_abs_per_token = attention_gate
        self.last_mlp_gate_abs_per_token = mlp_gate if include_mlp else None
        self.last_gate_abs_per_token = gate_per_token
        self.last_gate_abs_mean = gate_per_token.mean()
        self.last_update_rms_per_token = (
            (x.detach().float() - input_x.detach().float())
            .pow(2)
            .mean(dim=-1)
            .sqrt()
        )
        return x


class TokenFlowBlock(nn.Module):
    """Pointwise AdaLN-MLP block with no token-to-token communication."""

    def __init__(self, channels, mlp_ratio=1.0):
        super().__init__()
        self.channels = int(channels)
        self.mlp_ratio = float(mlp_ratio)
        hidden = int(self.channels * self.mlp_ratio)
        if hidden <= 0:
            raise ValueError(f"mlp_ratio must produce a positive hidden size, got {mlp_ratio}")

        self.mlp_norm = nn.LayerNorm(self.channels, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(self.channels, hidden),
            nn.SiLU(),
            nn.Linear(hidden, self.channels),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.channels, 3 * self.channels),
        )
        self.last_gate_abs_mean = None

    def forward(self, x, y):
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(y).chunk(3, dim=-1)
        h = modulate(self.mlp_norm(x), shift_mlp, scale_mlp)
        x = x + gate_mlp * self.mlp(h)
        self.last_gate_abs_mean = gate_mlp.detach().float().abs().mean()
        return x


class TokenFlowMLPHead(nn.Module):
    """MAR/NextStep-style flow head applied independently to each latent token.

    The causal backbone condition ``c`` is the only carrier of sequence context.
    This module contains no attention, positional mixing, or clean-latent context
    path: each output token is a function only of its own ``x_t``, timestep, and
    same-position backbone condition.
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        z_channels,
        num_res_blocks,
        grad_checkpointing=False,
        mlp_ratio=1.0,
        zero_init_gate=True,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.model_channels = int(model_channels)
        self.out_channels = int(out_channels)
        self.num_res_blocks = int(num_res_blocks)
        self.grad_checkpointing = bool(grad_checkpointing)
        self.mlp_ratio = float(mlp_ratio)
        self.zero_init_gate = bool(zero_init_gate)

        self.time_embed = TimestepEmbedder(self.model_channels)
        self.cond_embed = nn.Linear(z_channels, self.model_channels)
        self.input_proj = nn.Linear(self.in_channels, self.model_channels)
        self.blocks = nn.ModuleList(
            [
                TokenFlowBlock(
                    self.model_channels,
                    mlp_ratio=self.mlp_ratio,
                )
                for _ in range(self.num_res_blocks)
            ]
        )
        self.final_layer = FinalLayer(self.model_channels, self.out_channels)
        self.last_gate_abs_mean = None
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                _xavier_uniform_init_fp32_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm) and module.elementwise_affine:
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        self.apply(_basic_init)
        _normal_init_fp32_(self.time_embed.mlp[0].weight, std=0.02)
        _normal_init_fp32_(self.time_embed.mlp[2].weight, std=0.02)

        if self.zero_init_gate:
            for block in self.blocks:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def _shape_time(self, t, batch_shape):
        expected = int(math.prod(batch_shape))
        if t.numel() != expected:
            raise ValueError(
                f"t must contain one value per latent token ({expected}), got shape {tuple(t.shape)}"
            )
        embedded = self.time_embed(t.to(device=self.input_proj.weight.device).reshape(-1))
        return embedded.view(*batch_shape, self.model_channels)

    def forward(self, x, t, c):
        if x.dim() not in {2, 3}:
            raise ValueError(f"expected [B,D] or [B,Q,D], got {tuple(x.shape)}")
        model_dtype = self.input_proj.weight.dtype
        model_device = self.input_proj.weight.device
        x = x.to(device=model_device, dtype=model_dtype)
        c = c.to(device=model_device, dtype=model_dtype)
        batch_shape = x.shape[:-1]
        if c.shape[:-1] != batch_shape:
            raise ValueError(
                f"condition batch shape must match latent batch shape {tuple(batch_shape)}, "
                f"got {tuple(c.shape[:-1])}"
            )

        x = self.input_proj(x)
        y = self._shape_time(t, batch_shape) + self.cond_embed(c)
        gate_stats = []
        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.blocks:
                x = checkpoint(block, x, y, use_reentrant=False)
                if block.last_gate_abs_mean is not None:
                    gate_stats.append(block.last_gate_abs_mean)
        else:
            for block in self.blocks:
                x = block(x, y)
                if block.last_gate_abs_mean is not None:
                    gate_stats.append(block.last_gate_abs_mean)
        self.last_gate_abs_mean = torch.stack(gate_stats).mean() if gate_stats else None
        return self.final_layer(x, y)


class ContextualFlowTransformerHead(nn.Module):
    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        z_channels,
        num_res_blocks,
        grad_checkpointing=False,
        mlp_ratio=1.0,
        latent_mixer_heads=8,
        latent_mixer_dropout=0.0,
        latent_mixer_zero_init_gate=True,
        image_tokens_per_img=256,
        query_position_mode="additive_2d",
        context_position_mode="additive_2d",
        rope_mode="none",
        rope_axis_dims=(80, 80),
        rope_rotate_value=False,
        position_variant="FH0",
        flow_head_variant="DF0",
        endpoint_time=1000.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.grad_checkpointing = grad_checkpointing
        self.mlp_ratio = float(mlp_ratio)
        self.num_heads = int(latent_mixer_heads)
        self.dropout = float(latent_mixer_dropout)
        self.zero_init_gate = bool(latent_mixer_zero_init_gate)
        self.image_tokens_per_img = int(image_tokens_per_img)
        self.query_position_mode = str(query_position_mode)
        self.context_position_mode = str(context_position_mode)
        self.rope_mode = str(rope_mode)
        self.rope_axis_dims = tuple(int(item) for item in rope_axis_dims)
        self.rope_rotate_value = bool(rope_rotate_value)
        self.position_variant = str(position_variant)
        self.flow_head_variant = self._normalize_flow_head_variant(flow_head_variant)
        self.endpoint_time = float(endpoint_time)
        self.dynamic_content = self.flow_head_variant in {"DF1", "DF2"}
        self.content_uses_mlp = self.flow_head_variant == "DF1"
        if self.query_position_mode not in {"none", "additive_2d"}:
            raise ValueError(
                f"Unknown flow query_position_mode={self.query_position_mode!r}."
            )
        if self.context_position_mode not in {"none", "additive_2d"}:
            raise ValueError(
                f"Unknown flow context_position_mode={self.context_position_mode!r}."
            )
        if self.rope_mode not in {"none", "row_col_2d"}:
            raise ValueError(f"Unknown flow rope_mode={self.rope_mode!r}.")
        if self.rope_rotate_value:
            raise ValueError("Flow-head RoPE never rotates V.")
        if self.dynamic_content:
            dynamic_position_contracts = {
                "FH0": ("additive_2d", "additive_2d", "none"),
                "FH1": ("additive_2d", "additive_2d", "row_col_2d"),
                "FH4": ("none", "none", "row_col_2d"),
            }
            expected = dynamic_position_contracts.get(self.position_variant)
            actual = (
                self.query_position_mode,
                self.context_position_mode,
                self.rope_mode,
            )
            if expected is None or actual != expected:
                raise ValueError(
                    f"{self.flow_head_variant} position contract must be one of "
                    f"{dynamic_position_contracts}, got "
                    f"{self.position_variant}={actual}."
                )

        self.time_embed = TimestepEmbedder(model_channels)
        self.cond_embed = nn.Linear(z_channels, model_channels)
        self.input_proj = nn.Linear(in_channels, model_channels)
        pos_embed = torch.empty(self.image_tokens_per_img, self.model_channels)
        self.register_buffer("image_pos_embed", pos_embed.clone())
        self.blocks = nn.ModuleList(
            [
                ContextualFlowBlock(
                    model_channels,
                    num_heads=self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                    dropout=self.dropout,
                    rope_mode=self.rope_mode,
                    rope_axis_dims=self.rope_axis_dims,
                    image_tokens_per_img=self.image_tokens_per_img,
                )
                for _ in range(num_res_blocks)
            ]
        )
        self.final_layer = FinalLayer(model_channels, out_channels)
        self.last_gate_abs_mean = None
        self.last_gate_abs_per_token = None
        self.last_attention_entropy_per_token = None
        self.last_attention_distance_per_token = None
        self.last_query_attention_gate_abs_per_token = None
        self.last_query_mlp_gate_abs_per_token = None
        self.last_content_attention_gate_abs_per_token = None
        self.last_content_mlp_gate_abs_per_token = None
        self.last_content_update_rms_per_token = None
        self.last_content_relative_update_per_token = None
        self.last_content_query_cosine_per_token = None
        self.initialize_weights()

    @staticmethod
    def _normalize_flow_head_variant(value):
        normalized = str(value or "DF0").strip().upper().replace("_", "")
        if normalized not in {"DF0", "DF1", "DF2"}:
            raise ValueError(
                f"Unknown image_flow_head_variant={value!r}; expected DF0, DF1, or DF2."
            )
        return normalized

    def set_attention_diagnostics(self, enabled: bool) -> None:
        enabled = bool(enabled)
        for block in self.blocks:
            block.collect_attention_diagnostics = enabled

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                _xavier_uniform_init_fp32_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                if module.elementwise_affine:
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)

        self.apply(_basic_init)
        _normal_init_fp32_(self.time_embed.mlp[0].weight, std=0.02)
        _normal_init_fp32_(self.time_embed.mlp[2].weight, std=0.02)
        self._reset_position_buffer()

        if self.zero_init_gate:
            for block in self.blocks:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def _reset_position_buffer(self):
        device = self.input_proj.weight.device
        if device.type == "meta":
            device = None
        pos_embed = build_2d_sincos_position_embedding(
            self.image_tokens_per_img,
            self.model_channels,
            device=device,
        )
        self.image_pos_embed = pos_embed.clone()

    def _shape_time(self, t, batch_shape):
        t = t.to(device=self.input_proj.weight.device)
        if len(batch_shape) == 1:
            return self.time_embed(t.reshape(-1))
        return self.time_embed(t.reshape(-1)).view(*batch_shape, self.model_channels)

    def _positions(self, positions, batch_size, seq_len, device):
        if positions is None:
            base = torch.arange(seq_len, device=device, dtype=torch.long)
            return base.unsqueeze(0).expand(batch_size, seq_len)
        positions = positions.to(device=device, dtype=torch.long)
        if positions.dim() == 1:
            if positions.numel() == seq_len:
                positions = positions.unsqueeze(0).expand(batch_size, seq_len)
            elif seq_len == 1 and positions.numel() == batch_size:
                positions = positions.view(batch_size, 1)
            else:
                raise ValueError(
                    f"positions has shape {tuple(positions.shape)}, expected [{seq_len}] "
                    f"or [{batch_size}] for seq_len=1"
                )
        if positions.shape != (batch_size, seq_len):
            raise ValueError(f"positions must have shape {(batch_size, seq_len)}, got {tuple(positions.shape)}")
        if positions.numel() > 0:
            min_pos = int(positions.min().item())
            max_pos = int(positions.max().item())
            if min_pos < 0 or max_pos >= self.image_tokens_per_img:
                raise ValueError(
                    f"flow positions must be in [0, {self.image_tokens_per_img}), "
                    f"got min={min_pos}, max={max_pos}"
                )
        return positions

    def _lookup_pos_embed(self, positions, dtype):
        flat_positions = positions.reshape(-1)
        pos_embed = self.image_pos_embed.to(device=positions.device, dtype=dtype)
        values = pos_embed.index_select(0, flat_positions)
        return values.reshape(positions.shape + (self.model_channels,))

    def _ensure_sequence(self, x, positions=None):
        squeeze = False
        if x.dim() == 2:
            x = x.unsqueeze(1)
            squeeze = True
        if x.dim() != 3:
            raise ValueError(f"expected [B,D] or [B,Q,D], got {tuple(x.shape)}")
        batch_size, seq_len, _ = x.shape
        positions = self._positions(positions, batch_size, seq_len, x.device)
        return x, positions, squeeze

    def position_contract(self):
        return {
            "schema": "selfless_flow_head_position_v1",
            "variant": self.position_variant,
            "A_q": int(self.query_position_mode == "additive_2d"),
            "A_c": int(self.context_position_mode == "additive_2d"),
            "R_f": int(self.rope_mode == "row_col_2d"),
            "query_position_mode": self.query_position_mode,
            "context_position_mode": self.context_position_mode,
            "rope_mode": self.rope_mode,
            "rope_axis_dims": list(self.rope_axis_dims),
            "rotate_value": False,
        }

    def cache_contract(self):
        return {
            "schema": "selfless_flow_head_content_cache_v1",
            "flow_head_variant": self.flow_head_variant,
            "content_update": (
                "static"
                if self.flow_head_variant == "DF0"
                else "shared_attention_mlp"
                if self.content_uses_mlp
                else "shared_attention"
            ),
            "strict_context": True,
            "query_writes_cache": False,
            "position_contract": self.position_contract(),
        }

    @staticmethod
    def _position_digest(positions):
        weights = torch.arange(
            1,
            positions.shape[1] + 1,
            device=positions.device,
            dtype=torch.long,
        )
        return torch.stack(
            [
                positions.sum(dim=1),
                (positions * weights).sum(dim=1),
                positions.min(dim=1).values,
                positions.max(dim=1).values,
            ],
            dim=1,
        )

    def _validate_latent_mixer_cache(self, cache):
        expected = self.position_contract()
        actual = cache.get("position_contract")
        if actual != expected:
            raise ValueError(
                "Flow-head context cache position contract mismatch: "
                f"expected={expected}, actual={actual}."
            )
        layers = cache.get("layers")
        if not isinstance(layers, list) or len(layers) != len(self.blocks):
            raise ValueError(
                "Flow-head context cache must contain one layer cache per block."
            )
        if self.dynamic_content:
            actual_cache_contract = cache.get("cache_contract")
            expected_cache_contract = self.cache_contract()
            if actual_cache_contract != expected_cache_contract:
                raise ValueError(
                    "Dynamic flow-head content cache contract mismatch: "
                    f"expected={expected_cache_contract}, actual={actual_cache_contract}."
                )

    def _initial_content_hidden(self, context_latents, context_positions):
        content_hidden = self.input_proj(context_latents)
        if self.context_position_mode == "additive_2d":
            content_hidden = content_hidden + self._lookup_pos_embed(
                context_positions, content_hidden.dtype
            )
        return content_hidden

    def _content_condition(self, context_conditions):
        embedded = self.cond_embed(context_conditions)
        endpoint = torch.full(
            context_conditions.shape[:-1],
            self.endpoint_time,
            device=context_conditions.device,
            dtype=torch.float32,
        )
        return self._shape_time(endpoint, context_conditions.shape[:-1]) + embedded

    @staticmethod
    def _clone_stat(value):
        return value if value is None else value.clone()

    def _capture_block_stats(self, block):
        return {
            "gate": self._clone_stat(block.last_gate_abs_per_token),
            "attention_gate": self._clone_stat(
                block.last_attention_gate_abs_per_token
            ),
            "mlp_gate": self._clone_stat(block.last_mlp_gate_abs_per_token),
            "attention_entropy": self._clone_stat(
                block.last_attention_entropy_per_token
            ),
            "attention_distance": self._clone_stat(
                block.last_attention_distance_per_token
            ),
            "update_rms": self._clone_stat(block.last_update_rms_per_token),
        }

    @staticmethod
    def _mean_stat(stats, key):
        values = [item[key] for item in stats if item.get(key) is not None]
        return torch.stack(values).mean(dim=0) if values else None

    def _publish_stream_stats(self, query_stats, content_stats, content_inputs=None, content_outputs=None, query_outputs=None):
        self.last_gate_abs_per_token = self._mean_stat(query_stats, "gate")
        self.last_gate_abs_mean = (
            self.last_gate_abs_per_token.mean()
            if self.last_gate_abs_per_token is not None
            else None
        )
        self.last_attention_entropy_per_token = self._mean_stat(
            query_stats, "attention_entropy"
        )
        self.last_attention_distance_per_token = self._mean_stat(
            query_stats, "attention_distance"
        )
        self.last_query_attention_gate_abs_per_token = self._mean_stat(
            query_stats, "attention_gate"
        )
        self.last_query_mlp_gate_abs_per_token = self._mean_stat(
            query_stats, "mlp_gate"
        )
        self.last_content_attention_gate_abs_per_token = self._mean_stat(
            content_stats, "attention_gate"
        )
        self.last_content_mlp_gate_abs_per_token = self._mean_stat(
            content_stats, "mlp_gate"
        )
        self.last_content_update_rms_per_token = self._mean_stat(
            content_stats, "update_rms"
        )
        if content_inputs is not None and content_outputs is not None:
            denominator = (
                content_inputs.detach().float().pow(2).mean(dim=-1).sqrt()
                .clamp_min(torch.finfo(torch.float32).tiny)
            )
            self.last_content_relative_update_per_token = (
                (content_outputs.detach().float() - content_inputs.detach().float())
                .pow(2)
                .mean(dim=-1)
                .sqrt()
                / denominator
            )
        else:
            self.last_content_relative_update_per_token = None
        if content_outputs is not None and query_outputs is not None:
            self.last_content_query_cosine_per_token = F.cosine_similarity(
                content_outputs.detach().float(),
                query_outputs.detach().float(),
                dim=-1,
            )
        else:
            self.last_content_query_cosine_per_token = None

    @staticmethod
    def _format_context_mask(context_mask, batch_size, query_len, context_len, device):
        if context_mask is None:
            return None
        context_mask = context_mask.to(device=device, dtype=torch.bool)
        if context_mask.dim() == 2:
            context_mask = context_mask.unsqueeze(1)
        if context_mask.shape[1] == 1 and query_len != 1:
            context_mask = context_mask.expand(batch_size, query_len, context_len)
        if context_mask.shape != (batch_size, query_len, context_len):
            raise ValueError(
                f"context_mask must have shape {(batch_size, query_len, context_len)}, "
                f"got {tuple(context_mask.shape)}"
            )
        return context_mask

    def _build_context_block_mask(self, context_mask, batch_size, query_len, context_len, device):
        context_mask = self._format_context_mask(
            context_mask,
            batch_size,
            query_len,
            context_len,
            device,
        )
        if context_mask is None:
            return None, None

        def mask_mod(b, h, q_idx, kv_idx):
            return context_mask[b, q_idx, kv_idx]

        block_mask = create_block_mask(
            mask_mod,
            B=batch_size,
            H=None,
            Q_LEN=query_len,
            KV_LEN=context_len,
            device=device,
        )
        return context_mask, block_mask

    def prepare_latent_mixer_cache(
        self,
        context_latents=None,
        context_mask=None,
        context_positions=None,
        context_conditions=None,
    ):
        if context_latents is None:
            return None
        if context_latents.dim() != 3:
            raise ValueError(f"context_latents must be [B,K,D], got {tuple(context_latents.shape)}")
        if context_latents.shape[1] == 0:
            return None
        model_dtype = self.input_proj.weight.dtype
        model_device = self.input_proj.weight.device
        context_latents = context_latents.to(device=model_device, dtype=model_dtype)
        batch_size, context_len, _ = context_latents.shape
        context_positions = self._positions(context_positions, batch_size, context_len, model_device)
        context_hidden = self._initial_content_hidden(
            context_latents, context_positions
        )
        if context_mask is not None:
            context_mask = context_mask.to(device=model_device, dtype=torch.bool)
        if self.dynamic_content:
            if context_conditions is None:
                raise ValueError(
                    f"{self.flow_head_variant} cache construction requires "
                    "context_conditions for every content token."
                )
            context_conditions = context_conditions.to(
                device=model_device, dtype=model_dtype
            )
            if context_conditions.shape[:2] != (batch_size, context_len):
                raise ValueError(
                    "context_conditions must match context_latents batch/sequence "
                    f"shape {(batch_size, context_len)}, got "
                    f"{tuple(context_conditions.shape[:2])}."
                )
            content_y = self._content_condition(context_conditions)
            content_mask = self._format_context_mask(
                context_mask,
                batch_size,
                context_len,
                context_len,
                model_device,
            )
            if content_mask is None:
                raise ValueError(
                    f"{self.flow_head_variant} full cache construction requires "
                    "an explicit strict content mask."
                )
            content_mask, content_block_mask = self._build_context_block_mask(
                content_mask,
                batch_size,
                context_len,
                context_len,
                model_device,
            )
            layers = []
            for block in self.blocks:
                layer_cache = block.prepare_cross_cache(
                    context_hidden,
                    context_positions=context_positions,
                )
                layers.append(layer_cache)
                context_hidden = block(
                    context_hidden,
                    content_y,
                    layer_cache=layer_cache,
                    context_mask=content_mask,
                    query_positions=context_positions,
                    context_block_mask=content_block_mask,
                    use_flex_attention=True,
                    include_mlp=self.content_uses_mlp,
                )
        else:
            layers = [
                block.prepare_cross_cache(
                    context_hidden,
                    context_positions=context_positions,
                )
                for block in self.blocks
            ]
        return {
            "layers": layers,
            "context_mask": context_mask,
            "context_positions": context_positions,
            "position_digest": self._position_digest(context_positions),
            "position_contract": self.position_contract(),
            "cache_contract": self.cache_contract(),
        }

    def empty_latent_mixer_cache(self, batch_size=1):
        if not self.dynamic_content:
            raise ValueError("Incremental content caches are only defined for DF1/DF2.")
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}.")
        device = self.input_proj.weight.device
        dtype = self.input_proj.weight.dtype
        empty_positions = torch.empty(batch_size, 0, device=device, dtype=torch.long)
        layers = []
        for block in self.blocks:
            layers.append(
                {
                    "k": torch.empty(
                        batch_size,
                        block.num_heads,
                        0,
                        block.head_dim,
                        device=device,
                        dtype=dtype,
                    ),
                    "v": torch.empty(
                        batch_size,
                        block.num_heads,
                        0,
                        block.head_dim,
                        device=device,
                        dtype=dtype,
                    ),
                    "context_positions": empty_positions,
                    "k_rotation_count": int(block.rope_mode != "none"),
                }
            )
        return {
            "layers": layers,
            "context_mask": torch.empty(
                batch_size, 1, 0, device=device, dtype=torch.bool
            ),
            "context_positions": empty_positions,
            "position_digest": None,
            "position_contract": self.position_contract(),
            "cache_contract": self.cache_contract(),
        }

    def append_latent_mixer_cache(
        self,
        cache,
        *,
        context_latents,
        context_conditions,
        context_positions,
    ):
        if not self.dynamic_content:
            raise ValueError("Incremental content-cache append is only valid for DF1/DF2.")
        model_device = self.input_proj.weight.device
        model_dtype = self.input_proj.weight.dtype
        context_latents = context_latents.to(
            device=model_device, dtype=model_dtype
        )
        context_conditions = context_conditions.to(
            device=model_device, dtype=model_dtype
        )
        if context_latents.dim() == 2:
            context_latents = context_latents.unsqueeze(1)
        if context_conditions.dim() == 2:
            context_conditions = context_conditions.unsqueeze(1)
        if context_latents.dim() != 3 or context_latents.shape[1] != 1:
            raise ValueError(
                "append_latent_mixer_cache accepts exactly one new content token "
                f"per row, got {tuple(context_latents.shape)}."
            )
        if context_conditions.shape[:2] != context_latents.shape[:2]:
            raise ValueError(
                "context_conditions must match the appended latent batch/sequence."
            )
        batch_size = context_latents.shape[0]
        context_positions = self._positions(
            context_positions, batch_size, 1, model_device
        )
        if cache is None:
            cache = self.empty_latent_mixer_cache(batch_size)
        self._validate_latent_mixer_cache(cache)
        if cache["layers"][0]["k"].shape[0] != batch_size:
            raise ValueError("cache batch size must match appended content batch size.")

        hidden = self._initial_content_hidden(context_latents, context_positions)
        content_y = self._content_condition(context_conditions)
        previous_len = cache["layers"][0]["k"].shape[2]
        previous_mask = torch.ones(
            batch_size,
            1,
            previous_len,
            device=model_device,
            dtype=torch.bool,
        )
        updated_layers = []
        for layer_idx, block in enumerate(self.blocks):
            previous_layer = cache["layers"][layer_idx]
            new_layer = block.prepare_cross_cache(
                hidden, context_positions=context_positions
            )
            hidden = block(
                hidden,
                content_y,
                layer_cache=previous_layer if previous_len else None,
                context_mask=previous_mask,
                query_positions=context_positions,
                use_flex_attention=False,
                include_mlp=self.content_uses_mlp,
            )
            updated_layers.append(
                {
                    "k": torch.cat([previous_layer["k"], new_layer["k"]], dim=2),
                    "v": torch.cat([previous_layer["v"], new_layer["v"]], dim=2),
                    "context_positions": torch.cat(
                        [
                            previous_layer["context_positions"],
                            context_positions,
                        ],
                        dim=1,
                    ),
                    "k_rotation_count": int(block.rope_mode != "none"),
                }
            )
        positions = torch.cat(
            [cache["context_positions"], context_positions], dim=1
        )
        return {
            "layers": updated_layers,
            "context_mask": torch.ones(
                batch_size,
                1,
                previous_len + 1,
                device=model_device,
                dtype=torch.bool,
            ),
            "context_positions": positions,
            "position_digest": self._position_digest(positions),
            "position_contract": self.position_contract(),
            "cache_contract": self.cache_contract(),
        }

    def stack_latent_mixer_caches(self, caches):
        if not caches:
            return None
        for cache in caches:
            self._validate_latent_mixer_cache(cache)
            if cache["layers"][0]["k"].shape[0] != 1:
                raise ValueError("stack_latent_mixer_caches expects batch-one caches.")
        device = self.input_proj.weight.device
        dtype = self.input_proj.weight.dtype
        max_len = max(cache["layers"][0]["k"].shape[2] for cache in caches)
        batch_size = len(caches)
        positions = torch.zeros(
            batch_size, max_len, device=device, dtype=torch.long
        )
        mask = torch.zeros(
            batch_size, 1, max_len, device=device, dtype=torch.bool
        )
        layers = []
        for layer_idx, block in enumerate(self.blocks):
            k = torch.zeros(
                batch_size,
                block.num_heads,
                max_len,
                block.head_dim,
                device=device,
                dtype=dtype,
            )
            v = torch.zeros_like(k)
            for row, cache in enumerate(caches):
                length = cache["layers"][layer_idx]["k"].shape[2]
                if length:
                    k[row, :, :length] = cache["layers"][layer_idx]["k"][0]
                    v[row, :, :length] = cache["layers"][layer_idx]["v"][0]
                    if layer_idx == 0:
                        positions[row, :length] = cache["context_positions"][0]
                        mask[row, 0, :length] = True
            layers.append(
                {
                    "k": k,
                    "v": v,
                    "context_positions": positions,
                    "k_rotation_count": int(block.rope_mode != "none"),
                }
            )
        return {
            "layers": layers,
            "context_mask": mask,
            "context_positions": positions,
            "position_digest": None,
            "position_contract": self.position_contract(),
            "cache_contract": self.cache_contract(),
        }

    def forward(
        self,
        x,
        t,
        c,
        context_latents=None,
        context_mask=None,
        query_positions=None,
        context_positions=None,
        context_conditions=None,
        latent_mixer_cache=None,
    ):
        model_dtype = self.input_proj.weight.dtype
        x = x.to(device=self.input_proj.weight.device, dtype=model_dtype)
        c = c.to(device=x.device, dtype=model_dtype)
        batch_shape = x.shape[:-1]
        x, query_positions, squeeze = self._ensure_sequence(x, query_positions)
        x = self.input_proj(x)
        if self.query_position_mode == "additive_2d":
            x = x + self._lookup_pos_embed(query_positions, x.dtype)
        t = self._shape_time(t, batch_shape)
        raw_c = c
        c = self.cond_embed(raw_c)
        y = t + c
        if y.dim() == 2:
            y = y.unsqueeze(1)

        use_direct_context = latent_mixer_cache is None and context_latents is not None
        if self.dynamic_content and use_direct_context:
            context_latents = context_latents.to(device=x.device, dtype=model_dtype)
            if context_latents.dim() != 3:
                raise ValueError(
                    f"context_latents must be [B,K,D], got {tuple(context_latents.shape)}"
                )
            batch_size, context_len, _ = context_latents.shape
            if x.shape[:2] != (batch_size, context_len):
                raise ValueError(
                    f"{self.flow_head_variant} training requires aligned content/query "
                    f"streams, got content={tuple(context_latents.shape[:2])}, "
                    f"query={tuple(x.shape[:2])}."
            )
            if context_conditions is None:
                context_conditions = raw_c
            context_conditions = context_conditions.to(
                device=x.device, dtype=model_dtype
            )
            if context_conditions.shape[:2] != (batch_size, context_len):
                raise ValueError(
                    "context_conditions must align with the dynamic content stream."
                )
            context_positions = self._positions(
                context_positions, batch_size, context_len, x.device
            )
            content = self._initial_content_hidden(
                context_latents, context_positions
            )
            initial_content = content
            content_y = self._content_condition(context_conditions)
            content_mask, content_block_mask = self._build_context_block_mask(
                context_mask,
                batch_size,
                context_len,
                context_len,
                x.device,
            )
            if content_mask is None:
                raise ValueError(
                    f"{self.flow_head_variant} training requires an explicit "
                    "strict sigma-causal mask."
                )
            query_stats = []
            content_stats = []
            for block in self.blocks:
                if self.grad_checkpointing and not torch.jit.is_scripting():
                    def _dual_step(content_hidden, query_hidden):
                        layer_cache = block.prepare_cross_cache(
                            content_hidden,
                            context_positions=context_positions,
                        )
                        next_content = block(
                            content_hidden,
                            content_y,
                            layer_cache=layer_cache,
                            context_mask=content_mask,
                            query_positions=context_positions,
                            context_block_mask=content_block_mask,
                            use_flex_attention=True,
                            include_mlp=self.content_uses_mlp,
                        )
                        next_query = block(
                            query_hidden,
                            y,
                            layer_cache=layer_cache,
                            context_mask=content_mask,
                            query_positions=query_positions,
                            context_block_mask=content_block_mask,
                            use_flex_attention=True,
                        )
                        return next_content, next_query

                    content, x = checkpoint(
                        _dual_step,
                        content,
                        x,
                        use_reentrant=False,
                    )
                else:
                    layer_cache = block.prepare_cross_cache(
                        content,
                        context_positions=context_positions,
                    )
                    content = block(
                        content,
                        content_y,
                        layer_cache=layer_cache,
                        context_mask=content_mask,
                        query_positions=context_positions,
                        context_block_mask=content_block_mask,
                        use_flex_attention=True,
                        include_mlp=self.content_uses_mlp,
                    )
                    content_stats.append(self._capture_block_stats(block))
                    x = block(
                        x,
                        y,
                        layer_cache=layer_cache,
                        context_mask=content_mask,
                        query_positions=query_positions,
                        context_block_mask=content_block_mask,
                        use_flex_attention=True,
                    )
                    query_stats.append(self._capture_block_stats(block))
            self._publish_stream_stats(
                query_stats,
                content_stats,
                content_inputs=initial_content,
                content_outputs=content,
                query_outputs=x,
            )
            out = self.final_layer(x, y)
            return out.squeeze(1) if squeeze else out

        if use_direct_context:
            latent_mixer_cache = self.prepare_latent_mixer_cache(
                context_latents=context_latents,
                context_mask=context_mask,
                context_positions=context_positions,
                context_conditions=context_conditions,
            )
        context_layers = None
        context_mask = None
        if latent_mixer_cache is not None:
            self._validate_latent_mixer_cache(latent_mixer_cache)
            context_layers = latent_mixer_cache.get("layers")
            context_mask = latent_mixer_cache.get("context_mask")
        context_block_mask = None
        use_flex_attention = (
            bool(context_layers)
            and context_layers[0]["k"].shape[2] > 0
            and (self.training or use_direct_context)
        )
        if use_flex_attention:
            batch_size, query_len, _ = x.shape
            context_len = context_layers[0]["k"].shape[2]
            context_mask, context_block_mask = self._build_context_block_mask(
                context_mask,
                batch_size,
                query_len,
                context_len,
                x.device,
            )
        gate_stats = []
        gate_token_stats = []
        attention_entropy_stats = []
        attention_distance_stats = []
        if self.grad_checkpointing and not torch.jit.is_scripting():
            for layer_idx, block in enumerate(self.blocks):
                layer_cache = None if context_layers is None else context_layers[layer_idx]
                x = checkpoint(
                    block,
                    x,
                    y,
                    layer_cache,
                    context_mask,
                    query_positions,
                    context_block_mask,
                    use_flex_attention,
                    use_reentrant=False,
                )
                if block.last_gate_abs_mean is not None:
                    gate_stats.append(block.last_gate_abs_mean)
                if block.last_gate_abs_per_token is not None:
                    gate_token_stats.append(block.last_gate_abs_per_token)
                if block.last_attention_entropy_per_token is not None:
                    attention_entropy_stats.append(
                        block.last_attention_entropy_per_token
                    )
                if block.last_attention_distance_per_token is not None:
                    attention_distance_stats.append(
                        block.last_attention_distance_per_token
                    )
        else:
            for layer_idx, block in enumerate(self.blocks):
                layer_cache = None if context_layers is None else context_layers[layer_idx]
                x = block(
                    x,
                    y,
                    layer_cache=layer_cache,
                    context_mask=context_mask,
                    query_positions=query_positions,
                    context_block_mask=context_block_mask,
                    use_flex_attention=use_flex_attention,
                )
                if block.last_gate_abs_mean is not None:
                    gate_stats.append(block.last_gate_abs_mean)
                if block.last_gate_abs_per_token is not None:
                    gate_token_stats.append(block.last_gate_abs_per_token)
                if block.last_attention_entropy_per_token is not None:
                    attention_entropy_stats.append(
                        block.last_attention_entropy_per_token
                    )
                if block.last_attention_distance_per_token is not None:
                    attention_distance_stats.append(
                        block.last_attention_distance_per_token
                    )
        self.last_gate_abs_mean = torch.stack(gate_stats).mean() if gate_stats else None
        self.last_gate_abs_per_token = (
            torch.stack(gate_token_stats).mean(dim=0)
            if gate_token_stats
            else None
        )
        self.last_attention_entropy_per_token = (
            torch.stack(attention_entropy_stats).mean(dim=0)
            if attention_entropy_stats
            else None
        )
        self.last_attention_distance_per_token = (
            torch.stack(attention_distance_stats).mean(dim=0)
            if attention_distance_stats
            else None
        )
        self.last_query_attention_gate_abs_per_token = self._mean_stat(
            [
                {
                    "attention_gate": block.last_attention_gate_abs_per_token,
                }
                for block in self.blocks
            ],
            "attention_gate",
        )
        self.last_query_mlp_gate_abs_per_token = self._mean_stat(
            [
                {
                    "mlp_gate": block.last_mlp_gate_abs_per_token,
                }
                for block in self.blocks
            ],
            "mlp_gate",
        )
        self.last_content_attention_gate_abs_per_token = None
        self.last_content_mlp_gate_abs_per_token = None
        self.last_content_update_rms_per_token = None
        self.last_content_relative_update_per_token = None
        self.last_content_query_cosine_per_token = None
        out = self.final_layer(x, y)
        return out.squeeze(1) if squeeze else out


class FlowLoss(nn.Module):
    """Rectified-flow loss with a selectable velocity-head architecture.

    Uses the standard flow-matching convention: t=0 is noise and t=1 is data.
    """

    def __init__(
        self,
        target_channels,
        z_channels,
        depth,
        width,
        num_sampling_steps,
        grad_checkpointing=False,
        time_scale=1000.0,
        time_sampling="logit_normal",
        logit_mean=0.0,
        logit_std=1.0,
        time_eps=1.0e-4,
        uniform_mix=0.1,
        solver="heun",
        mlp_ratio=1.0,
        image_tokens_per_img=256,
        latent_mixer_heads=8,
        latent_mixer_dropout=0.0,
        latent_mixer_zero_init_gate=True,
        head_arch="contextual",
        query_position_mode="additive_2d",
        context_position_mode="additive_2d",
        rope_mode="none",
        rope_axis_dims=(80, 80),
        rope_rotate_value=False,
        position_variant="FH0",
        flow_head_variant="DF0",
    ):
        super().__init__()
        self.in_channels = int(target_channels)
        self.num_sampling_steps = int(num_sampling_steps)
        self.time_scale = float(time_scale)
        self.time_sampling = str(time_sampling or "logit_normal").lower()
        self.logit_mean = float(logit_mean)
        self.logit_std = float(logit_std)
        self.time_eps = float(time_eps)
        self.uniform_mix = float(uniform_mix)
        self.solver = str(solver or "heun").lower()
        self.mlp_ratio = float(mlp_ratio)
        self.head_arch = self._normalize_head_arch(head_arch)
        self.uses_latent_mixer = self.head_arch == "contextual"
        self.flow_head_variant = (
            ContextualFlowTransformerHead._normalize_flow_head_variant(
                flow_head_variant
            )
        )
        if self.head_arch != "contextual" and self.flow_head_variant != "DF0":
            raise ValueError(
                f"{self.flow_head_variant} requires image_flow_head_arch=contextual."
            )
        self._guidance_diagnostic_sums = {}
        self._guidance_diagnostic_counts = {}
        if self.num_sampling_steps <= 0:
            raise ValueError(f"num_sampling_steps must be positive, got {num_sampling_steps}")
        if not 0.0 <= self.uniform_mix <= 1.0:
            raise ValueError(f"uniform_mix must be in [0, 1], got {uniform_mix}")
        if not 0.0 <= self.time_eps < 0.5:
            raise ValueError(f"time_eps must be in [0, 0.5), got {time_eps}")

        if self.uses_latent_mixer:
            self.net = ContextualFlowTransformerHead(
                in_channels=self.in_channels,
                model_channels=width,
                out_channels=self.in_channels,
                z_channels=z_channels,
                num_res_blocks=depth,
                grad_checkpointing=grad_checkpointing,
                mlp_ratio=self.mlp_ratio,
                latent_mixer_heads=latent_mixer_heads,
                latent_mixer_dropout=latent_mixer_dropout,
                latent_mixer_zero_init_gate=latent_mixer_zero_init_gate,
                image_tokens_per_img=image_tokens_per_img,
                query_position_mode=query_position_mode,
                context_position_mode=context_position_mode,
                rope_mode=rope_mode,
                rope_axis_dims=rope_axis_dims,
                rope_rotate_value=rope_rotate_value,
                position_variant=position_variant,
                flow_head_variant=self.flow_head_variant,
                endpoint_time=self.time_scale,
            )
        else:
            self.net = TokenFlowMLPHead(
                in_channels=self.in_channels,
                model_channels=width,
                out_channels=self.in_channels,
                z_channels=z_channels,
                num_res_blocks=depth,
                grad_checkpointing=grad_checkpointing,
                mlp_ratio=self.mlp_ratio,
                zero_init_gate=latent_mixer_zero_init_gate,
            )
        self.last_forward_stats = {}

    @staticmethod
    def _normalize_head_arch(head_arch: str) -> str:
        value = str(head_arch or "contextual").strip().lower().replace("-", "_")
        if value in {"contextual", "latent_mixer", "cross_attention", "cross_attn"}:
            return "contextual"
        if value in {"token_mlp", "tokenwise_mlp", "per_token", "mar", "nextstep", "no_cross_attention"}:
            return "token_mlp"
        raise ValueError(
            f"Unknown image_flow_head_arch={head_arch!r}; expected contextual or token_mlp."
        )

    @staticmethod
    def _rms_stat(x):
        return x.detach().float().pow(2).mean().sqrt()

    def _sample_times(self, batch_size: int, device) -> torch.Tensor:
        if self.time_sampling in {"uniform", "rand", "random"}:
            t = torch.rand(batch_size, device=device)
        elif self.time_sampling in {"logit_normal", "lognorm", "logistic_normal"}:
            logits = torch.randn(batch_size, device=device) * self.logit_std + self.logit_mean
            t = torch.sigmoid(logits)
            if self.uniform_mix > 0.0:
                use_uniform = torch.rand(batch_size, device=device) < self.uniform_mix
                t = torch.where(use_uniform, torch.rand(batch_size, device=device), t)
        else:
            raise ValueError(
                f"Unknown image_flow_time_sampling={self.time_sampling!r}; expected logit_normal or uniform."
            )
        return t.clamp(self.time_eps, 1.0 - self.time_eps)

    def _scale_time(self, t: torch.Tensor) -> torch.Tensor:
        return t.to(dtype=torch.float32) * self.time_scale

    def _context_to_device(self, context_kwargs, device, dtype):
        out = {}
        for key, value in context_kwargs.items():
            if value is None:
                out[key] = None
            elif key.endswith("positions"):
                out[key] = value.to(device=device)
            elif key == "context_mask":
                out[key] = value.to(device=device, dtype=torch.bool)
            else:
                out[key] = value.to(device=device, dtype=dtype)
        return out

    def _cache_to_device(self, cache, device, dtype):
        if cache is None:
            return None
        def _convert(value, key=None):
            if value is None:
                return None
            if isinstance(value, dict):
                return {sub_key: _convert(sub_value, sub_key) for sub_key, sub_value in value.items()}
            if isinstance(value, list):
                return [_convert(item, key) for item in value]
            if isinstance(value, tuple):
                return tuple(_convert(item, key) for item in value)
            if key == "context_mask":
                return value.to(device=device, dtype=torch.bool)
            if key is not None and (
                key.endswith("positions") or key == "position_digest"
            ):
                return value.to(device=device)
            if not isinstance(value, torch.Tensor):
                return value
            return value.to(device=device, dtype=dtype)

        return _convert(cache)

    def prepare_latent_mixer_cache(
        self,
        context_latents: torch.Tensor | None = None,
        context_mask: torch.Tensor | None = None,
        context_positions: torch.Tensor | None = None,
        context_conditions: torch.Tensor | None = None,
    ):
        if not self.uses_latent_mixer:
            return None
        if context_latents is None:
            return None
        model_dtype = self.net.input_proj.weight.dtype
        model_device = self.net.input_proj.weight.device
        context_latents = context_latents.to(device=model_device, dtype=model_dtype)
        if context_mask is not None:
            context_mask = context_mask.to(device=model_device, dtype=torch.bool)
        if context_positions is not None:
            context_positions = context_positions.to(device=model_device)
        if context_conditions is not None:
            context_conditions = context_conditions.to(
                device=model_device, dtype=model_dtype
            )
        return self.net.prepare_latent_mixer_cache(
            context_latents=context_latents,
            context_mask=context_mask,
            context_positions=context_positions,
            context_conditions=context_conditions,
        )

    def empty_latent_mixer_cache(self, batch_size=1):
        if not self.uses_latent_mixer:
            return None
        return self.net.empty_latent_mixer_cache(batch_size=batch_size)

    def append_latent_mixer_cache(self, cache, **kwargs):
        if not self.uses_latent_mixer:
            return None
        return self.net.append_latent_mixer_cache(cache, **kwargs)

    def stack_latent_mixer_caches(self, caches):
        if not self.uses_latent_mixer:
            return None
        return self.net.stack_latent_mixer_caches(caches)

    def velocity(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        z: torch.Tensor,
        *,
        context_latents: torch.Tensor | None = None,
        context_mask: torch.Tensor | None = None,
        query_positions: torch.Tensor | None = None,
        context_positions: torch.Tensor | None = None,
        context_conditions: torch.Tensor | None = None,
        latent_mixer_cache: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        model_dtype = self.net.input_proj.weight.dtype
        model_device = self.net.input_proj.weight.device
        x_t = x_t.to(device=model_device, dtype=model_dtype)
        z = z.to(device=model_device, dtype=model_dtype)
        t = t.to(device=model_device)
        if not self.uses_latent_mixer:
            return self.net(x_t, self._scale_time(t), c=z)

        context_kwargs = self._context_to_device(
            {
                "context_latents": context_latents,
                "context_mask": context_mask,
                "query_positions": query_positions,
                "context_positions": context_positions,
                "context_conditions": context_conditions,
            },
            model_device,
            model_dtype,
        )
        latent_mixer_cache = self._cache_to_device(
            latent_mixer_cache,
            model_device,
            model_dtype,
        )
        return self.net(
            x_t,
            self._scale_time(t),
            c=z,
            latent_mixer_cache=latent_mixer_cache,
            **context_kwargs,
        )

    def _training_context(self, target, sigma, image_positions, context_latents=None):
        if not self.uses_latent_mixer:
            return {}
        if target.dim() != 3:
            return {}
        if context_latents is None:
            context_latents = target
        else:
            context_latents = context_latents.to(device=target.device, dtype=target.dtype)
        batch_size, seq_len, _ = target.shape
        if sigma is None:
            order = torch.arange(seq_len, device=target.device, dtype=torch.float32)
            sigma = order.unsqueeze(0).expand(batch_size, seq_len)
        sigma = sigma.to(device=target.device, dtype=torch.float32)
        context_mask = sigma.unsqueeze(1) < sigma.unsqueeze(2)
        if image_positions is None:
            image_positions = torch.arange(seq_len, device=target.device, dtype=torch.long).unsqueeze(0).expand(batch_size, seq_len)
        return {
            "context_latents": context_latents,
            "context_mask": context_mask,
            "query_positions": image_positions,
            "context_positions": image_positions,
            "context_conditions": None,
        }

    def forward(self, target, z, mask=None, sigma=None, image_positions=None, context_latents=None):
        model_dtype = self.net.input_proj.weight.dtype
        model_device = self.net.input_proj.weight.device
        target = target.to(device=model_device)
        target_model = target.to(dtype=model_dtype)
        target_float = target.float()
        z = z.to(device=model_device, dtype=model_dtype)
        if mask is not None:
            mask = mask.to(device=model_device, dtype=model_dtype)
        if sigma is not None:
            sigma = sigma.to(device=model_device)
        if image_positions is not None:
            image_positions = image_positions.to(device=model_device)
        if context_latents is not None:
            context_latents = context_latents.to(device=model_device, dtype=model_dtype)

        batch_shape = target_model.shape[:-1]
        t = self._sample_times(int(math.prod(batch_shape)), model_device).view(batch_shape)
        noise = torch.randn(target_float.shape, device=model_device, dtype=torch.float32)
        t_view = t.unsqueeze(-1).float()
        x_t_float = (1.0 - t_view) * noise + t_view * target_float
        v_target = target_float - noise
        context_kwargs = self._training_context(target_model, sigma, image_positions, context_latents=context_latents)
        v_pred = self.velocity(x_t_float.to(dtype=model_dtype), t, z, **context_kwargs)
        token_loss = (v_pred.float() - v_target).pow(2).mean(dim=-1)
        if mask is not None:
            token_loss = (token_loss * mask.float()).sum() / mask.float().sum().clamp_min(1.0)
            loss_mean = token_loss
        else:
            loss_mean = token_loss.mean()

        stats = {
            "flow/loss": loss_mean.detach().float(),
            "flow/v_mse": loss_mean.detach().float(),
            "flow/t_mean": t.detach().float().mean(),
            "flow/t_min": t.detach().float().min(),
            "flow/t_max": t.detach().float().max(),
            "flow/x_t_rms": self._rms_stat(x_t_float),
            "flow/v_target_rms": self._rms_stat(v_target),
            "flow/v_pred_rms": self._rms_stat(v_pred),
        }
        if target_model.dim() == 3 and sigma is not None and token_loss.dim() == 2:
            sigma_values = sigma.to(device=token_loss.device, dtype=torch.float32)
            strict_context = sigma_values.unsqueeze(1) < sigma_values.unsqueeze(2)
            context_counts = strict_context.sum(dim=-1)
            ranks = context_counts
            denominator = max(int(sigma_values.shape[1]) - 1, 1)
            reveal_fractions = ranks.float() / float(denominator)
            bucket_edges = (
                (0.0, 0.25),
                (0.25, 0.5),
                (0.5, 0.75),
                (0.75, 1.000001),
            )
            for lower, upper in bucket_edges:
                bucket_mask = (reveal_fractions >= lower) & (reveal_fractions < upper)
                if bucket_mask.any():
                    tag = f"{int(lower * 100):02d}_{min(100, int(upper * 100)):02d}"
                    stats[f"flow/reveal_{tag}_v_mse"] = (
                        token_loss[bucket_mask].mean().detach().float()
                    )
            context_buckets = (
                ("0", 0, 0),
                ("1", 1, 1),
                ("2_4", 2, 4),
                ("5_16", 5, 16),
                ("17_64", 17, 64),
                ("65_plus", 65, None),
            )
            gate_per_token = getattr(self.net, "last_gate_abs_per_token", None)
            query_attention_gate = getattr(
                self.net, "last_query_attention_gate_abs_per_token", None
            )
            query_mlp_gate = getattr(
                self.net, "last_query_mlp_gate_abs_per_token", None
            )
            content_attention_gate = getattr(
                self.net, "last_content_attention_gate_abs_per_token", None
            )
            content_mlp_gate = getattr(
                self.net, "last_content_mlp_gate_abs_per_token", None
            )
            attention_entropy = getattr(
                self.net, "last_attention_entropy_per_token", None
            )
            attention_distance = getattr(
                self.net, "last_attention_distance_per_token", None
            )
            for tag, lower, upper in context_buckets:
                bucket_mask = context_counts >= lower
                if upper is not None:
                    bucket_mask &= context_counts <= upper
                if not bucket_mask.any():
                    continue
                stats[f"flow/context_{tag}_v_mse"] = (
                    token_loss[bucket_mask].mean().detach().float()
                )
                if (
                    isinstance(gate_per_token, torch.Tensor)
                    and gate_per_token.shape == token_loss.shape
                ):
                    stats[f"flow/context_{tag}_gate_abs"] = (
                        gate_per_token[bucket_mask].mean().detach().float()
                    )
                for metric_name, metric_value in (
                    ("query_attention_gate_abs", query_attention_gate),
                    ("query_mlp_gate_abs", query_mlp_gate),
                    ("content_attention_gate_abs", content_attention_gate),
                    ("content_mlp_gate_abs", content_mlp_gate),
                ):
                    if (
                        isinstance(metric_value, torch.Tensor)
                        and metric_value.shape == token_loss.shape
                    ):
                        stats[f"flow/context_{tag}_{metric_name}"] = (
                            metric_value[bucket_mask].mean().detach().float()
                        )
                if (
                    isinstance(attention_entropy, torch.Tensor)
                    and attention_entropy.shape == token_loss.shape
                ):
                    stats[f"flow/context_{tag}_attention_entropy"] = (
                        attention_entropy[bucket_mask].mean().detach().float()
                    )
                if (
                    isinstance(attention_distance, torch.Tensor)
                    and attention_distance.shape == token_loss.shape
                ):
                    stats[f"flow/context_{tag}_attention_distance"] = (
                        attention_distance[bucket_mask].mean().detach().float()
                    )
        gate_stat = getattr(self.net, "last_gate_abs_mean", None)
        if gate_stat is not None:
            stats["flow/head_residual_gate"] = gate_stat.detach().float()
            if self.uses_latent_mixer:
                stats["flow/latent_mixer_gate"] = gate_stat.detach().float()
        for metric_name, attribute in (
            ("content_update_rms", "last_content_update_rms_per_token"),
            ("content_relative_update", "last_content_relative_update_per_token"),
            ("content_query_cosine", "last_content_query_cosine_per_token"),
        ):
            value = getattr(self.net, attribute, None)
            if isinstance(value, torch.Tensor):
                stats[f"flow/{metric_name}"] = value.mean().detach().float()
        stats["flow/nonfinite_count"] = torch.zeros(
            (), device=loss_mean.device, dtype=torch.float32
        )
        self.last_forward_stats = stats
        return loss_mean

    def estimate_x0(self, x_t: torch.Tensor, t: torch.Tensor, z: torch.Tensor, **context_kwargs) -> torch.Tensor:
        v = self.velocity(x_t, t, z, **context_kwargs)
        t_view = t.view(*t.shape, *([1] * (x_t.ndim - t.ndim))).to(dtype=x_t.dtype)
        return x_t + (1.0 - t_view) * v

    @staticmethod
    def _duplicate_context(context_kwargs):
        def _duplicate_value(value):
            if value is None:
                return None
            if isinstance(value, dict):
                return {key: _duplicate_value(item) for key, item in value.items()}
            if isinstance(value, list):
                return [_duplicate_value(item) for item in value]
            if isinstance(value, tuple):
                return tuple(_duplicate_value(item) for item in value)
            if not isinstance(value, torch.Tensor):
                return value
            return torch.cat([value, value], dim=0)

        out = {}
        for key, value in context_kwargs.items():
            out[key] = _duplicate_value(value)
        return out

    def _guided_velocity(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        z: torch.Tensor,
        cfg: float,
        context_kwargs,
        *,
        context_is_paired: bool = False,
        debug_check=None,
        ode_step: int | None = None,
        debug_phase: str = "",
    ) -> torch.Tensor:
        z_is_paired = z.shape[0] == x.shape[0] * 2
        if cfg == 1.0 and not z_is_paired:
            velocity = self.velocity(x, t, z, **context_kwargs)
            if debug_check is not None:
                debug_check(
                    f"{debug_phase}velocity",
                    velocity,
                    ode_step,
                    {"state": x, "condition": z},
                )
            return velocity
        x_pair = torch.cat([x, x], dim=0)
        t_pair = torch.cat([t, t], dim=0)
        paired_context_kwargs = context_kwargs if context_is_paired else self._duplicate_context(context_kwargs)
        v_pair = self.velocity(x_pair, t_pair, z, **paired_context_kwargs)
        v_cond, v_uncond = torch.chunk(v_pair, 2, dim=0)
        velocity_delta = v_cond - v_uncond
        self._record_guidance_delta(velocity_delta, context_kwargs)
        scaled_velocity_delta = float(cfg) * velocity_delta
        guided_velocity = v_uncond + scaled_velocity_delta
        if debug_check is not None:
            references = {"state": x, "condition": z}
            debug_check(f"{debug_phase}paired_velocity", v_pair, ode_step, references)
            debug_check(f"{debug_phase}conditional_velocity", v_cond, ode_step, references)
            debug_check(f"{debug_phase}unconditional_velocity", v_uncond, ode_step, references)
            debug_check(f"{debug_phase}velocity_delta", velocity_delta, ode_step, references)
            debug_check(f"{debug_phase}scaled_velocity_delta", scaled_velocity_delta, ode_step, references)
            debug_check(f"{debug_phase}guided_velocity", guided_velocity, ode_step, references)
        return guided_velocity

    @staticmethod
    def _context_bucket_tag(counts: torch.Tensor, tag: str) -> torch.Tensor:
        bounds = {
            "0": (0, 0),
            "1": (1, 1),
            "2_4": (2, 4),
            "5_16": (5, 16),
            "17_64": (17, 64),
            "65_plus": (65, None),
        }
        lower, upper = bounds[tag]
        mask = counts >= lower
        return mask if upper is None else mask & (counts <= upper)

    def reset_guidance_diagnostics(self) -> None:
        self._guidance_diagnostic_sums = {}
        self._guidance_diagnostic_counts = {}

    def _record_guidance_delta(self, velocity_delta, context_kwargs) -> None:
        if velocity_delta.ndim < 2:
            return
        per_token = velocity_delta.detach().float().pow(2).mean(dim=-1).sqrt()
        cache = context_kwargs.get("latent_mixer_cache")
        context_mask = cache.get("context_mask") if isinstance(cache, dict) else None
        if isinstance(context_mask, torch.Tensor):
            if context_mask.shape[0] == per_token.shape[0] * 2:
                context_mask = context_mask[: per_token.shape[0]]
            counts = context_mask.to(dtype=torch.bool).sum(dim=-1)
            if counts.shape[-1] == 1 and per_token.ndim == 2:
                counts = counts.expand_as(per_token)
        else:
            counts = torch.zeros_like(per_token, dtype=torch.long)
        if counts.shape != per_token.shape:
            counts = counts.reshape(per_token.shape)
        for tag in ("0", "1", "2_4", "5_16", "17_64", "65_plus"):
            mask = self._context_bucket_tag(counts, tag)
            if not mask.any():
                continue
            value_sum = per_token[mask].sum()
            value_count = mask.sum().to(dtype=torch.float32)
            self._guidance_diagnostic_sums[tag] = (
                self._guidance_diagnostic_sums.get(tag, value_sum.new_zeros(()))
                + value_sum
            )
            self._guidance_diagnostic_counts[tag] = (
                self._guidance_diagnostic_counts.get(
                    tag, value_count.new_zeros(())
                )
                + value_count
            )

    def guidance_diagnostics(self) -> dict[str, dict[str, torch.Tensor]]:
        return {
            tag: {
                "sum": self._guidance_diagnostic_sums[tag],
                "count": self._guidance_diagnostic_counts[tag],
            }
            for tag in self._guidance_diagnostic_sums
        }

    @staticmethod
    def _scheduled_cfg(cfg: float, schedule: str | None, progress: float) -> float:
        cfg = float(cfg)
        if cfg == 1.0:
            return 1.0
        schedule = str(schedule or "constant").lower()
        if schedule in {"constant", "none", "off", ""}:
            return cfg
        progress = max(0.0, min(1.0, float(progress)))
        if schedule == "linear":
            return 1.0 + (cfg - 1.0) * progress
        raise ValueError(f"Unknown image flow cfg_schedule={schedule!r}; expected constant or linear.")

    def sample(
        self,
        z,
        temperature=1.0,
        cfg=1.0,
        cfg_schedule="constant",
        solver=None,
        num_steps=None,
        return_trace=False,
        *,
        context_latents: torch.Tensor | None = None,
        context_mask: torch.Tensor | None = None,
        query_positions: torch.Tensor | None = None,
        context_positions: torch.Tensor | None = None,
        context_conditions: torch.Tensor | None = None,
        latent_mixer_cache: dict | None = None,
        latent_mixer_cache_is_paired: bool = False,
        initial_noise: torch.Tensor | None = None,
        debug_finite: bool = False,
        debug_label: str = "",
    ):
        def _max_abs(tensor: torch.Tensor) -> float | None:
            finite_values = tensor[torch.isfinite(tensor)].float()
            if not finite_values.numel():
                return None
            return float(finite_values.abs().max().item())

        def _debug_check(
            name: str,
            tensor: torch.Tensor,
            ode_step: int | None = None,
            references: dict[str, torch.Tensor] | None = None,
        ) -> None:
            if not debug_finite:
                return
            finite = torch.isfinite(tensor)
            if bool(finite.all()):
                return
            nonfinite_rows = None
            if tensor.ndim >= 1:
                row_finite = finite.reshape(tensor.shape[0], -1).all(dim=1)
                nonfinite_rows = (~row_finite).nonzero(as_tuple=True)[0].detach().cpu().tolist()
            reference_max_abs = {
                key: _max_abs(value)
                for key, value in (references or {}).items()
            }
            raise FloatingPointError(
                "non-finite tensor during image-flow sampling: "
                f"label={debug_label!r}, component={name!r}, ode_step={ode_step}, "
                f"shape={tuple(tensor.shape)}, nonfinite_rows={nonfinite_rows}, "
                f"nonfinite={int((~finite).sum().item())}/{tensor.numel()}, "
                f"finite_max_abs={_max_abs(tensor)}, references_max_abs={reference_max_abs}"
            )

        model_dtype = self.net.input_proj.weight.dtype
        model_device = self.net.input_proj.weight.device
        z = z.to(device=model_device, dtype=model_dtype)
        _debug_check("flow_condition", z)
        if cfg != 1.0 and z.shape[0] % 2 != 0:
            raise ValueError(f"cfg != 1.0 requires paired conditional/unconditional conditions; got batch {z.shape[0]}")

        steps = int(num_steps or self.num_sampling_steps)
        if steps <= 0:
            raise ValueError(f"num_steps must be positive, got {steps}")
        solver = str(solver or self.solver).lower()
        x_shape = (z.shape[0] // 2, *z.shape[1:-1]) if cfg != 1.0 else z.shape[:-1]
        expected_noise_shape = (*x_shape, self.in_channels)
        temperature = float(temperature)
        if not math.isfinite(temperature):
            raise ValueError(f"temperature must be finite, got {temperature}")
        if initial_noise is None:
            x = torch.randn(
                *expected_noise_shape,
                device=z.device,
                dtype=torch.float32,
            )
        else:
            if not isinstance(initial_noise, torch.Tensor):
                raise TypeError(
                    "initial_noise must be a torch.Tensor when provided, "
                    f"got {type(initial_noise).__name__}"
                )
            if tuple(initial_noise.shape) != expected_noise_shape:
                raise ValueError(
                    "initial_noise must exactly match the unpaired flow-state shape "
                    f"{expected_noise_shape}, got {tuple(initial_noise.shape)}"
                )
            if not initial_noise.is_floating_point():
                raise TypeError(
                    "initial_noise must have a floating dtype, "
                    f"got {initial_noise.dtype}"
                )
            if not bool(torch.isfinite(initial_noise).all().item()):
                raise FloatingPointError("initial_noise contains non-finite values")
            x = initial_noise.to(device=z.device, dtype=torch.float32)
        x = x * temperature
        if not bool(torch.isfinite(x).all().item()):
            raise FloatingPointError(
                "temperature scaling produced non-finite initial_noise values"
            )
        _debug_check("initial_noise", x)
        if context_latents is not None:
            _debug_check("context_latents", context_latents)
        if self.uses_latent_mixer:
            if latent_mixer_cache is not None:
                latent_mixer_cache = self._cache_to_device(
                    latent_mixer_cache, z.device, z.dtype
                )
                query_positions = (
                    None
                    if query_positions is None
                    else query_positions.to(device=z.device)
                )
                context_kwargs = {
                    "query_positions": query_positions,
                    "latent_mixer_cache": latent_mixer_cache,
                }
            else:
                raw_context_kwargs = self._context_to_device(
                    {
                        "context_latents": context_latents,
                        "context_mask": context_mask,
                        "query_positions": query_positions,
                        "context_positions": context_positions,
                        "context_conditions": context_conditions,
                    },
                    z.device,
                    z.dtype,
                )
                latent_mixer_cache = self.prepare_latent_mixer_cache(
                    context_latents=raw_context_kwargs.get("context_latents"),
                    context_mask=raw_context_kwargs.get("context_mask"),
                    context_positions=raw_context_kwargs.get("context_positions"),
                    context_conditions=raw_context_kwargs.get(
                        "context_conditions"
                    ),
                )
                context_kwargs = {
                    "query_positions": raw_context_kwargs.get("query_positions"),
                    "latent_mixer_cache": latent_mixer_cache,
                }
        else:
            context_kwargs = {}
        context_is_paired = bool(latent_mixer_cache_is_paired)
        if cfg != 1.0:
            if context_is_paired:
                if context_kwargs.get("query_positions") is not None:
                    positions = context_kwargs["query_positions"]
                    if positions.shape[0] * 2 == z.shape[0]:
                        context_kwargs["query_positions"] = torch.cat(
                            [positions, positions], dim=0
                        )
            else:
                context_kwargs = self._duplicate_context(context_kwargs)
                context_is_paired = True
        times = torch.linspace(0.0, 1.0, steps + 1, device=z.device, dtype=torch.float32)

        for idx in range(steps):
            _debug_check("ode_state", x, idx, {"condition": z})
            t = times[idx].expand(x_shape)
            t_next = times[idx + 1].expand(x_shape)
            dt = (times[idx + 1] - times[idx]).float()
            cfg_t = self._scheduled_cfg(cfg, cfg_schedule, float(times[idx].item()))
            v = self._guided_velocity(
                x.to(dtype=model_dtype),
                t,
                z,
                cfg_t,
                context_kwargs,
                context_is_paired=context_is_paired,
                debug_check=_debug_check if debug_finite else None,
                ode_step=idx,
                debug_phase="predictor_",
            ).float()
            _debug_check("guided_velocity", v, idx, {"state": x, "condition": z})
            if solver == "euler":
                x = x + dt * v
                _debug_check("euler_state", x, idx, {"velocity": v, "condition": z})
            elif solver == "heun":
                x_euler = x + dt * v
                _debug_check("heun_euler_predictor", x_euler, idx, {"velocity": v, "condition": z})
                cfg_t_next = self._scheduled_cfg(cfg, cfg_schedule, float(times[idx + 1].item()))
                v_next = self._guided_velocity(
                    x_euler.to(dtype=model_dtype),
                    t_next,
                    z,
                    cfg_t_next,
                    context_kwargs,
                    context_is_paired=context_is_paired,
                    debug_check=_debug_check if debug_finite else None,
                    ode_step=idx,
                    debug_phase="corrector_",
                ).float()
                _debug_check(
                    "heun_corrector_velocity",
                    v_next,
                    idx,
                    {"predictor_state": x_euler, "condition": z},
                )
                x = x + 0.5 * dt * (v + v_next)
                _debug_check(
                    "heun_corrected_state",
                    x,
                    idx,
                    {"velocity": v, "corrector_velocity": v_next, "condition": z},
                )
            else:
                raise ValueError(f"Unknown image_flow_solver={solver!r}; expected heun or euler.")

        if return_trace:
            return x.to(dtype=model_dtype), {
                "solver": solver,
                "num_steps": steps,
                "cfg_schedule": str(cfg_schedule or "constant"),
            }
        return x.to(dtype=model_dtype)
