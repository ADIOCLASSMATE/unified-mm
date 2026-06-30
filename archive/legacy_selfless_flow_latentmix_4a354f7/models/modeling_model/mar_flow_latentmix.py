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


class ResBlock(nn.Module):
    def __init__(self, channels, mlp_ratio=1.0):
        super().__init__()
        self.channels = channels
        self.mlp_ratio = float(mlp_ratio)
        self.intermediate_size = int(channels * self.mlp_ratio)
        if self.intermediate_size <= 0:
            raise ValueError(f"mlp_ratio must produce a positive hidden size, got {mlp_ratio}")

        self.in_ln = nn.LayerNorm(channels, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(channels, self.intermediate_size, bias=True),
            nn.SiLU(),
            nn.Linear(self.intermediate_size, channels, bias=True),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(channels, 3 * channels, bias=True),
        )

    def forward(self, x, y):
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(y).chunk(3, dim=-1)
        h = modulate(self.in_ln(x), shift_mlp, scale_mlp)
        h = self.mlp(h)
        return x + gate_mlp * h


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


class CausalLatentInputMixer(nn.Module):
    """Residual cross-attention from noisy query hidden states to earlier latent context."""

    def __init__(
        self,
        model_channels,
        num_heads=8,
        dropout=0.0,
        image_tokens_per_img=256,
        zero_init_gate=True,
    ):
        super().__init__()
        self.model_channels = int(model_channels)
        self.num_heads = int(num_heads)
        self.dropout = float(dropout)
        self.image_tokens_per_img = int(image_tokens_per_img)
        self.zero_init_gate = bool(zero_init_gate)
        if self.num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if self.model_channels % self.num_heads != 0:
            raise ValueError(
                f"model_channels={model_channels} must be divisible by num_heads={num_heads}"
            )
        self.head_dim = self.model_channels // self.num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(self.model_channels, self.model_channels)
        self.k_proj = nn.Linear(self.model_channels, self.model_channels)
        self.v_proj = nn.Linear(self.model_channels, self.model_channels)
        self.out_proj = nn.Linear(self.model_channels, self.model_channels)
        self.q_norm = nn.LayerNorm(self.model_channels, eps=1e-6)
        self.kv_norm = nn.LayerNorm(self.model_channels, eps=1e-6)
        pos_embed = torch.empty(self.image_tokens_per_img, self.model_channels)
        self.register_buffer("image_pos_embed", pos_embed.clone())
        self._reset_position_buffer()
        self.gate = nn.Parameter(torch.empty((), dtype=torch.float32))
        self._reset_gate()

    def _reset_gate(self):
        gate_init = 0.0 if self.zero_init_gate else 1.0
        with torch.no_grad():
            self.gate.fill_(gate_init)

    def _reset_position_buffer(self):
        device = self.q_proj.weight.device
        if device.type == "meta":
            device = None
        pos_embed = build_2d_sincos_position_embedding(
            self.image_tokens_per_img,
            self.model_channels,
            device=device,
        )
        self.image_pos_embed = pos_embed.clone()

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
            raise ValueError(
                f"positions must have shape {(batch_size, seq_len)}, got {tuple(positions.shape)}"
            )
        if positions.numel() > 0:
            min_pos = int(positions.min().item())
            max_pos = int(positions.max().item())
            if min_pos < 0 or max_pos >= self.image_tokens_per_img:
                raise ValueError(
                    f"latent mixer positions must be in [0, {self.image_tokens_per_img}), "
                    f"got min={min_pos}, max={max_pos}"
                )
        return positions

    def _lookup_pos_embed(self, positions, dtype):
        flat_positions = positions.reshape(-1)
        pos_embed = self.image_pos_embed.to(device=positions.device, dtype=dtype)
        values = pos_embed.index_select(0, flat_positions)
        return values.reshape(positions.shape + (self.model_channels,))

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

    def _project_query(self, query_hidden, query_positions):
        batch_size, query_len, _ = query_hidden.shape
        dtype = query_hidden.dtype
        device = query_hidden.device
        q_pos = self._positions(query_positions, batch_size, query_len, device)
        query_for_attn = query_hidden + self._lookup_pos_embed(q_pos, dtype)
        q = self.q_proj(self.q_norm(query_for_attn))
        return q.view(batch_size, query_len, self.num_heads, self.head_dim).transpose(1, 2)

    def _project_context(self, context_hidden, context_positions):
        batch_size, context_len, _ = context_hidden.shape
        dtype = context_hidden.dtype
        device = context_hidden.device
        kv_pos = self._positions(context_positions, batch_size, context_len, device)
        kv_for_attn = context_hidden + self._lookup_pos_embed(kv_pos, dtype)
        kv_for_attn = self.kv_norm(kv_for_attn)
        k = self.k_proj(kv_for_attn)
        v = self.v_proj(kv_for_attn)
        k = k.view(batch_size, context_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, context_len, self.num_heads, self.head_dim).transpose(1, 2)
        return k, v

    def prepare_context_cache(self, context_hidden, context_mask=None, context_positions=None):
        if context_hidden is None:
            return None
        if context_hidden.dim() != 3:
            raise ValueError(f"context_hidden must be [B,K,D], got {tuple(context_hidden.shape)}")
        batch_size, context_len, hidden_dim = context_hidden.shape
        if hidden_dim != self.model_channels:
            raise ValueError(
                f"context hidden dim {hidden_dim} must match model_channels={self.model_channels}"
            )
        if context_len == 0:
            return None

        device = context_hidden.device
        if context_mask is not None:
            context_mask = context_mask.to(device=device, dtype=torch.bool)
            if context_mask.dim() == 2:
                context_mask = context_mask.unsqueeze(1)
            if (
                context_mask.dim() != 3
                or context_mask.shape[0] != batch_size
                or context_mask.shape[-1] != context_len
            ):
                raise ValueError(
                    f"context_mask must have shape [B,Q,K] or [B,K] with "
                    f"B={batch_size}, K={context_len}; got {tuple(context_mask.shape)}"
                )
        k, v = self._project_context(context_hidden, context_positions)
        return {
            "k": k,
            "v": v,
            "context_mask": context_mask,
        }

    def apply_context_cache(self, query_hidden, context_cache=None, query_positions=None):
        if context_cache is None:
            return query_hidden

        squeeze_query = False
        if query_hidden.dim() == 2:
            query_hidden = query_hidden.unsqueeze(1)
            squeeze_query = True
        if query_hidden.dim() != 3:
            raise ValueError(f"query_hidden must be [N,D] or [B,Q,D], got {tuple(query_hidden.shape)}")

        k = context_cache.get("k")
        v = context_cache.get("v")
        if k is None or v is None:
            return query_hidden.squeeze(1) if squeeze_query else query_hidden
        if k.dim() != 4 or v.dim() != 4:
            raise ValueError("cached k/v must be [B,H,K,Dh]")
        batch_size, query_len, hidden_dim = query_hidden.shape
        if hidden_dim != self.model_channels:
            raise ValueError(
                f"query hidden dim {hidden_dim} must match model_channels={self.model_channels}"
            )
        if k.shape != v.shape:
            raise ValueError(f"cached k/v shapes must match, got {tuple(k.shape)} vs {tuple(v.shape)}")
        if k.shape[0] != batch_size or k.shape[1] != self.num_heads or k.shape[3] != self.head_dim:
            raise ValueError(
                f"cached k/v shape {tuple(k.shape)} is incompatible with "
                f"batch={batch_size}, heads={self.num_heads}, head_dim={self.head_dim}"
            )
        context_len = k.shape[2]
        if context_len == 0:
            return query_hidden.squeeze(1) if squeeze_query else query_hidden

        dtype = query_hidden.dtype
        device = query_hidden.device
        k = k.to(device=device, dtype=dtype)
        v = v.to(device=device, dtype=dtype)
        context_mask = self._format_context_mask(
            context_cache.get("context_mask"),
            batch_size,
            query_len,
            context_len,
            device,
        )
        has_context = context_mask.any(dim=-1)
        safe_mask = context_mask
        if not bool(has_context.all().item()):
            safe_mask = context_mask.clone()
            safe_mask[~has_context] = True

        q = self._project_query(query_hidden, query_positions)
        mixed = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=safe_mask.unsqueeze(1),
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
            scale=self.scale,
        )
        if not bool(has_context.all().item()):
            mixed = mixed * has_context[:, None, :, None].to(dtype=mixed.dtype)
        mixed = mixed.transpose(1, 2).reshape(batch_size, query_len, self.model_channels)
        mixed = self.out_proj(mixed)
        out = query_hidden + self.gate.to(dtype=dtype) * mixed
        return out.squeeze(1) if squeeze_query else out

    def _build_context_block_mask(self, context_mask, batch_size, query_len, context_len, device):
        def mask_mod(b, h, q_idx, kv_idx):
            return context_mask[b, q_idx, kv_idx]

        return create_block_mask(
            mask_mod,
            B=batch_size,
            H=None,
            Q_LEN=query_len,
            KV_LEN=context_len,
            device=device,
        )

    def forward(
        self,
        query_hidden,
        context_hidden=None,
        context_mask=None,
        query_positions=None,
        context_positions=None,
    ):
        if context_hidden is None:
            return query_hidden

        squeeze_query = False
        if query_hidden.dim() == 2:
            query_hidden = query_hidden.unsqueeze(1)
            squeeze_query = True
        if query_hidden.dim() != 3:
            raise ValueError(f"query_hidden must be [N,D] or [B,Q,D], got {tuple(query_hidden.shape)}")
        if context_hidden.dim() != 3:
            raise ValueError(f"context_hidden must be [B,K,D], got {tuple(context_hidden.shape)}")

        batch_size, query_len, _ = query_hidden.shape
        if query_hidden.shape[-1] != self.model_channels:
            raise ValueError(
                f"query hidden dim {query_hidden.shape[-1]} must match model_channels={self.model_channels}"
            )
        if context_hidden.shape[0] != batch_size:
            raise ValueError(
                f"context batch {context_hidden.shape[0]} must match query batch {batch_size}"
            )
        if context_hidden.shape[-1] != self.model_channels:
            raise ValueError(
                f"context hidden dim {context_hidden.shape[-1]} must match model_channels={self.model_channels}"
            )
        context_len = context_hidden.shape[1]
        if context_len == 0:
            return query_hidden.squeeze(1) if squeeze_query else query_hidden

        dtype = query_hidden.dtype
        device = query_hidden.device
        context_hidden = context_hidden.to(device=device, dtype=dtype)

        context_mask = self._format_context_mask(
            context_mask,
            batch_size,
            query_len,
            context_len,
            device,
        )
        q = self._project_query(query_hidden, query_positions)
        k, v = self._project_context(context_hidden, context_positions)

        block_mask = self._build_context_block_mask(
            context_mask,
            batch_size,
            query_len,
            context_len,
            device,
        )
        mixed = flex_attention(
            query=q,
            key=k,
            value=v,
            score_mod=None,
            block_mask=block_mask,
            scale=self.scale,
            enable_gqa=False,
            return_lse=False,
        )
        mixed = F.dropout(mixed, p=self.dropout, training=self.training) if self.dropout > 0.0 else mixed
        mixed = mixed.transpose(1, 2).reshape(batch_size, query_len, self.model_channels)
        mixed = self.out_proj(mixed)
        out = query_hidden + self.gate.to(dtype=dtype) * mixed
        return out.squeeze(1) if squeeze_query else out


class SimpleFlowMLPAdaLN(nn.Module):
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
        latent_mixer_enabled=True,
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
        self.latent_mixer_enabled = bool(latent_mixer_enabled)

        self.time_embed = TimestepEmbedder(model_channels)
        self.cond_embed = nn.Linear(z_channels, model_channels)
        self.input_proj = nn.Linear(in_channels, model_channels)
        if self.latent_mixer_enabled:
            self.input_mixer = CausalLatentInputMixer(
                model_channels=model_channels,
                num_heads=latent_mixer_heads,
                dropout=latent_mixer_dropout,
                image_tokens_per_img=image_tokens_per_img,
                zero_init_gate=latent_mixer_zero_init_gate,
            )
        else:
            self.input_mixer = None

        self.res_blocks = nn.ModuleList(
            [ResBlock(model_channels, mlp_ratio=self.mlp_ratio) for _ in range(num_res_blocks)]
        )
        self.final_layer = FinalLayer(model_channels, out_channels)
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

        for block in self.res_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)
        if self.input_mixer is not None:
            self.input_mixer._reset_position_buffer()
            self.input_mixer._reset_gate()

    def _shape_time(self, t, batch_shape):
        t = t.to(device=self.input_proj.weight.device)
        if len(batch_shape) == 1:
            return self.time_embed(t.reshape(-1))
        return self.time_embed(t.reshape(-1)).view(*batch_shape, self.model_channels)

    def prepare_latent_mixer_cache(
        self,
        context_latents=None,
        context_mask=None,
        context_positions=None,
    ):
        if self.input_mixer is None or context_latents is None:
            return None
        if context_latents.dim() != 3:
            raise ValueError(f"context_latents must be [B,K,D], got {tuple(context_latents.shape)}")
        if context_latents.shape[1] == 0:
            return None
        model_dtype = self.input_proj.weight.dtype
        model_device = self.input_proj.weight.device
        context_latents = context_latents.to(device=model_device, dtype=model_dtype)
        context_hidden = self.input_proj(context_latents)
        return self.input_mixer.prepare_context_cache(
            context_hidden,
            context_mask=context_mask,
            context_positions=context_positions,
        )

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
        x = self.input_proj(x)
        if self.input_mixer is not None:
            if latent_mixer_cache is not None:
                x = self.input_mixer.apply_context_cache(
                    x,
                    context_cache=latent_mixer_cache,
                    query_positions=query_positions,
                )
            elif context_latents is not None:
                context_latents = context_latents.to(device=x.device, dtype=model_dtype)
                context_hidden = self.input_proj(context_latents)
                x = self.input_mixer(
                    x,
                    context_hidden=context_hidden,
                    context_mask=context_mask,
                    query_positions=query_positions,
                    context_positions=context_positions,
                )
        t = self._shape_time(t, batch_shape)
        c = self.cond_embed(c)
        y = t + c

        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.res_blocks:
                x = checkpoint(block, x, y)
        else:
            for block in self.res_blocks:
                x = block(x, y)
        return self.final_layer(x, y)


class FlowLoss(nn.Module):
    """Rectified-flow loss with optional causal latent input mixing."""

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

        self.net = SimpleFlowMLPAdaLN(
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
        out = {}
        for key, value in cache.items():
            if value is None:
                out[key] = None
            elif key == "context_mask":
                out[key] = value.to(device=device, dtype=torch.bool)
            else:
                out[key] = value.to(device=device, dtype=dtype)
        return out

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
        mixer = getattr(self.net, "input_mixer", None)
        if mixer is not None:
            stats["flow/latent_mixer_gate"] = mixer.gate.detach().float()
        self.last_forward_stats = stats
        return loss_mean

    def estimate_x0(self, x_t: torch.Tensor, t: torch.Tensor, z: torch.Tensor, **context_kwargs) -> torch.Tensor:
        v = self.velocity(x_t, t, z, **context_kwargs)
        return x_t - t.view(*t.shape, *([1] * (x_t.ndim - t.ndim))).to(dtype=x_t.dtype) * v

    @staticmethod
    def _duplicate_context(context_kwargs):
        out = {}
        for key, value in context_kwargs.items():
            if value is None:
                out[key] = None
            elif isinstance(value, dict):
                out[key] = {
                    cache_key: (
                        None
                        if cache_value is None
                        else torch.cat([cache_value, cache_value], dim=0)
                    )
                    for cache_key, cache_value in value.items()
                }
            else:
                out[key] = torch.cat([value, value], dim=0)
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
        x_batch = z.shape[0] // 2 if cfg != 1.0 else z.shape[0]
        x = torch.randn(x_batch, self.in_channels, device=z.device, dtype=torch.float32) * float(temperature)
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
            t = times[idx].expand(x_batch)
            t_next = times[idx + 1].expand(x_batch)
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
