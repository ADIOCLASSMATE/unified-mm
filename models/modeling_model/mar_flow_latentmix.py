import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from torch.utils.checkpoint import checkpoint

from .image_position_utils import build_2d_sincos_position_embedding


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
    def __init__(self, channels, num_heads=8, mlp_ratio=2.0, dropout=0.0):
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

    def prepare_cross_cache(self, context_hidden):
        context_hidden = self.cross_kv_norm(context_hidden)
        k = self._split_heads(self.cross_k(context_hidden))
        v = self._split_heads(self.cross_v(context_hidden))
        return {"k": k, "v": v}

    def _cross_attention(self, x, layer_cache, context_mask, context_block_mask=None, use_flex_attention=False):
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
        attn_dtype = q.dtype
        k = k.to(device=x.device, dtype=attn_dtype)
        v = v.to(device=x.device, dtype=attn_dtype)
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
        context_block_mask=None,
        use_flex_attention=False,
    ):
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
                context_block_mask=context_block_mask,
                use_flex_attention=use_flex_attention,
            )
            if mixed is not None:
                x = x + gate_cross * mixed

        h = modulate(self.mlp_norm(x), shift_mlp, scale_mlp)
        x = x + gate_mlp * self.mlp(h)
        self.last_gate_abs_mean = torch.stack(
            [
                gate_cross.detach().float().abs().mean(),
                gate_mlp.detach().float().abs().mean(),
            ]
        ).mean()
        return x


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
                )
                for _ in range(num_res_blocks)
            ]
        )
        self.final_layer = FinalLayer(model_channels, out_channels)
        self.last_gate_abs_mean = None
        self.initialize_weights()

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
        context_hidden = self.input_proj(context_latents)
        context_hidden = context_hidden + self._lookup_pos_embed(context_positions, context_hidden.dtype)
        if context_mask is not None:
            context_mask = context_mask.to(device=model_device, dtype=torch.bool)
        return {
            "layers": [block.prepare_cross_cache(context_hidden) for block in self.blocks],
            "context_mask": context_mask,
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
        latent_mixer_cache=None,
    ):
        model_dtype = self.input_proj.weight.dtype
        x = x.to(device=self.input_proj.weight.device, dtype=model_dtype)
        c = c.to(device=x.device, dtype=model_dtype)
        batch_shape = x.shape[:-1]
        x, query_positions, squeeze = self._ensure_sequence(x, query_positions)
        x = self.input_proj(x)
        x = x + self._lookup_pos_embed(query_positions, x.dtype)
        t = self._shape_time(t, batch_shape)
        c = self.cond_embed(c)
        y = t + c
        if y.dim() == 2:
            y = y.unsqueeze(1)

        use_direct_context = latent_mixer_cache is None and context_latents is not None
        if use_direct_context:
            latent_mixer_cache = self.prepare_latent_mixer_cache(
                context_latents=context_latents,
                context_mask=context_mask,
                context_positions=context_positions,
            )
        context_layers = None
        context_mask = None
        if latent_mixer_cache is not None:
            context_layers = latent_mixer_cache.get("layers")
            context_mask = latent_mixer_cache.get("context_mask")
        context_block_mask = None
        use_flex_attention = bool(context_layers) and (self.training or use_direct_context)
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
        if self.grad_checkpointing and not torch.jit.is_scripting():
            for layer_idx, block in enumerate(self.blocks):
                layer_cache = None if context_layers is None else context_layers[layer_idx]
                x = checkpoint(
                    block,
                    x,
                    y,
                    layer_cache,
                    context_mask,
                    context_block_mask,
                    use_flex_attention,
                    use_reentrant=False,
                )
                if block.last_gate_abs_mean is not None:
                    gate_stats.append(block.last_gate_abs_mean)
        else:
            for layer_idx, block in enumerate(self.blocks):
                layer_cache = None if context_layers is None else context_layers[layer_idx]
                x = block(
                    x,
                    y,
                    layer_cache=layer_cache,
                    context_mask=context_mask,
                    context_block_mask=context_block_mask,
                    use_flex_attention=use_flex_attention,
                )
                if block.last_gate_abs_mean is not None:
                    gate_stats.append(block.last_gate_abs_mean)
        self.last_gate_abs_mean = torch.stack(gate_stats).mean() if gate_stats else None
        out = self.final_layer(x, y)
        return out.squeeze(1) if squeeze else out


class FlowLoss(nn.Module):
    """Rectified-flow loss with a contextual latent transformer velocity head."""

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
        if self.num_sampling_steps <= 0:
            raise ValueError(f"num_sampling_steps must be positive, got {num_sampling_steps}")
        if not 0.0 <= self.uniform_mix <= 1.0:
            raise ValueError(f"uniform_mix must be in [0, 1], got {uniform_mix}")
        if not 0.0 <= self.time_eps < 0.5:
            raise ValueError(f"time_eps must be in [0, 0.5), got {time_eps}")

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
        )
        self.last_forward_stats = {}

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
            return value.to(device=device, dtype=dtype)

        return _convert(cache)

    def prepare_latent_mixer_cache(
        self,
        context_latents: torch.Tensor | None = None,
        context_mask: torch.Tensor | None = None,
        context_positions: torch.Tensor | None = None,
    ):
        if context_latents is None:
            return None
        model_dtype = self.net.input_proj.weight.dtype
        model_device = self.net.input_proj.weight.device
        context_latents = context_latents.to(device=model_device, dtype=model_dtype)
        if context_mask is not None:
            context_mask = context_mask.to(device=model_device, dtype=torch.bool)
        if context_positions is not None:
            context_positions = context_positions.to(device=model_device)
        return self.net.prepare_latent_mixer_cache(
            context_latents=context_latents,
            context_mask=context_mask,
            context_positions=context_positions,
        )

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
        latent_mixer_cache: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        model_dtype = self.net.input_proj.weight.dtype
        model_device = self.net.input_proj.weight.device
        x_t = x_t.to(device=model_device, dtype=model_dtype)
        z = z.to(device=model_device, dtype=model_dtype)
        t = t.to(device=model_device)
        context_kwargs = self._context_to_device(
            {
                "context_latents": context_latents,
                "context_mask": context_mask,
                "query_positions": query_positions,
                "context_positions": context_positions,
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
        x_t_float = (1.0 - t_view) * target_float + t_view * noise
        v_target = noise - target_float
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
        gate_stat = getattr(self.net, "last_gate_abs_mean", None)
        if gate_stat is not None:
            stats["flow/latent_mixer_gate"] = gate_stat.detach().float()
        self.last_forward_stats = stats
        return loss_mean

    def estimate_x0(self, x_t: torch.Tensor, t: torch.Tensor, z: torch.Tensor, **context_kwargs) -> torch.Tensor:
        v = self.velocity(x_t, t, z, **context_kwargs)
        return x_t - t.view(*t.shape, *([1] * (x_t.ndim - t.ndim))).to(dtype=x_t.dtype) * v

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
    ) -> torch.Tensor:
        if cfg == 1.0:
            return self.velocity(x, t, z, **context_kwargs)
        x_pair = torch.cat([x, x], dim=0)
        t_pair = torch.cat([t, t], dim=0)
        paired_context_kwargs = context_kwargs if context_is_paired else self._duplicate_context(context_kwargs)
        v_pair = self.velocity(x_pair, t_pair, z, **paired_context_kwargs)
        v_cond, v_uncond = torch.chunk(v_pair, 2, dim=0)
        return v_uncond + float(cfg) * (v_cond - v_uncond)

    def sample(
        self,
        z,
        temperature=1.0,
        cfg=1.0,
        solver=None,
        num_steps=None,
        return_trace=False,
        *,
        context_latents: torch.Tensor | None = None,
        context_mask: torch.Tensor | None = None,
        query_positions: torch.Tensor | None = None,
        context_positions: torch.Tensor | None = None,
    ):
        model_dtype = self.net.input_proj.weight.dtype
        model_device = self.net.input_proj.weight.device
        z = z.to(device=model_device, dtype=model_dtype)
        if cfg != 1.0 and z.shape[0] % 2 != 0:
            raise ValueError(f"cfg != 1.0 requires paired conditional/unconditional conditions; got batch {z.shape[0]}")

        steps = int(num_steps or self.num_sampling_steps)
        if steps <= 0:
            raise ValueError(f"num_steps must be positive, got {steps}")
        solver = str(solver or self.solver).lower()
        x_shape = (z.shape[0] // 2, *z.shape[1:-1]) if cfg != 1.0 else z.shape[:-1]
        x = torch.randn(*x_shape, self.in_channels, device=z.device, dtype=torch.float32) * float(temperature)
        raw_context_kwargs = self._context_to_device(
            {
                "context_latents": context_latents,
                "context_mask": context_mask,
                "query_positions": query_positions,
                "context_positions": context_positions,
            },
            z.device,
            z.dtype,
        )
        latent_mixer_cache = self.prepare_latent_mixer_cache(
            context_latents=raw_context_kwargs.get("context_latents"),
            context_mask=raw_context_kwargs.get("context_mask"),
            context_positions=raw_context_kwargs.get("context_positions"),
        )
        context_kwargs = {
            "query_positions": raw_context_kwargs.get("query_positions"),
            "latent_mixer_cache": latent_mixer_cache,
        }
        context_is_paired = False
        if cfg != 1.0:
            context_kwargs = self._duplicate_context(context_kwargs)
            context_is_paired = True
        times = torch.linspace(1.0, 0.0, steps + 1, device=z.device, dtype=torch.float32)

        for idx in range(steps):
            t = times[idx].expand(x_shape)
            t_next = times[idx + 1].expand(x_shape)
            dt = (times[idx + 1] - times[idx]).float()
            v = self._guided_velocity(
                x.to(dtype=model_dtype),
                t,
                z,
                cfg,
                context_kwargs,
                context_is_paired=context_is_paired,
            ).float()
            if solver == "euler":
                x = x + dt * v
            elif solver == "heun":
                x_euler = x + dt * v
                v_next = self._guided_velocity(
                    x_euler.to(dtype=model_dtype),
                    t_next,
                    z,
                    cfg,
                    context_kwargs,
                    context_is_paired=context_is_paired,
                ).float()
                x = x + 0.5 * dt * (v + v_next)
            else:
                raise ValueError(f"Unknown image_flow_solver={solver!r}; expected heun or euler.")

        if return_trace:
            return x.to(dtype=model_dtype), {"solver": solver, "num_steps": steps}
        return x.to(dtype=model_dtype)
