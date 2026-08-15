"""Position-wise rectified-flow head used only by architecture ablations.

The production flow head is intentionally contextual: its query stream attends
to a dynamically updated content stream.  This module is the control model.
Every latent token is processed by the same AdaLN MLP independently, matching
the per-token head shape used by MAR and NextStep.  Context tensors and image
positions are accepted for call-site compatibility but are never consumed.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


def _xavier_uniform_init_fp32_(tensor: torch.Tensor) -> None:
    if tensor.is_meta:
        return
    with torch.no_grad():
        value = torch.empty(tensor.shape, device=tensor.device, dtype=torch.float32)
        nn.init.xavier_uniform_(value)
        tensor.copy_(value.to(dtype=tensor.dtype))


def _normal_init_fp32_(
    tensor: torch.Tensor,
    mean: float = 0.0,
    std: float = 1.0,
) -> None:
    if tensor.is_meta:
        return
    with torch.no_grad():
        value = torch.empty(tensor.shape, device=tensor.device, dtype=torch.float32)
        nn.init.normal_(value, mean=mean, std=std)
        tensor.copy_(value.to(dtype=tensor.dtype))


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = int(frequency_embedding_size)
        self._frequency_cache: dict[tuple[str, int | None], torch.Tensor] = {}

    def _apply(self, fn, recurse: bool = True):
        self._frequency_cache.clear()
        return super()._apply(fn, recurse=recurse)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        original_shape = t.shape
        flat_t = t.reshape(-1)
        cache_key = (flat_t.device.type, flat_t.device.index)
        freqs = self._frequency_cache.get(cache_key)
        if freqs is None:
            half = self.frequency_embedding_size // 2
            freqs = torch.exp(
                -math.log(10000)
                * torch.arange(
                    half,
                    device=flat_t.device,
                    dtype=torch.float32,
                )
                / half
            )
            self._frequency_cache[cache_key] = freqs
        args = flat_t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.frequency_embedding_size % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])],
                dim=-1,
            )
        embedding = embedding.to(
            device=self.mlp[0].weight.device,
            dtype=self.mlp[0].weight.dtype,
        )
        return self.mlp(embedding).view(*original_shape, -1)


class PositionwiseResBlock(nn.Module):
    def __init__(self, channels: int, mlp_ratio: float = 1.0):
        super().__init__()
        hidden_size = int(channels * mlp_ratio)
        if hidden_size <= 0:
            raise ValueError(f"mlp_ratio must produce a positive width, got {mlp_ratio}")
        self.in_ln = nn.LayerNorm(channels, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, channels),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(channels, 3 * channels, bias=True),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        shift, scale, gate = self.adaLN_modulation(y).chunk(3, dim=-1)
        return x + gate * self.mlp(modulate(self.in_ln(x), shift, scale))


class PositionwiseFinalLayer(nn.Module):
    def __init__(self, model_channels: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(
            model_channels,
            elementwise_affine=False,
            eps=1e-6,
        )
        self.linear = nn.Linear(model_channels, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(model_channels, 2 * model_channels, bias=True),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(y).chunk(2, dim=-1)
        return self.linear(modulate(self.norm_final(x), shift, scale))


class PositionwiseFlowMLP(nn.Module):
    """Shared MLP applied independently to the last-but-one token axis."""

    def __init__(
        self,
        in_channels: int,
        model_channels: int,
        out_channels: int,
        z_channels: int,
        num_res_blocks: int,
        grad_checkpointing: bool = False,
        mlp_ratio: float = 1.0,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.model_channels = int(model_channels)
        self.out_channels = int(out_channels)
        self.num_res_blocks = int(num_res_blocks)
        self.grad_checkpointing = bool(grad_checkpointing)
        self.mlp_ratio = float(mlp_ratio)

        self.time_embed = TimestepEmbedder(self.model_channels)
        self.cond_embed = nn.Linear(z_channels, self.model_channels)
        self.input_proj = nn.Linear(self.in_channels, self.model_channels)
        self.res_blocks = nn.ModuleList(
            [
                PositionwiseResBlock(self.model_channels, self.mlp_ratio)
                for _ in range(self.num_res_blocks)
            ]
        )
        self.final_layer = PositionwiseFinalLayer(
            self.model_channels,
            self.out_channels,
        )
        self.initialize_weights()

    def initialize_weights(self) -> None:
        def _basic_init(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                _xavier_uniform_init_fp32_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm) and module.elementwise_affine:
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        self.apply(_basic_init)
        _normal_init_fp32_(self.time_embed.mlp[0].weight, std=0.02)
        _normal_init_fp32_(self.time_embed.mlp[2].weight, std=0.02)
        for block in self.res_blocks:
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)

    @staticmethod
    def position_contract() -> dict[str, object]:
        return {
            "schema": "positionwise_flow_head_v1",
            "architecture": "positionwise_adaln_mlp",
            "cross_token_attention": False,
            "uses_content_latents": False,
            "uses_image_position": False,
        }

    def cache_contract(self) -> dict[str, object]:
        return {
            "schema": "positionwise_flow_head_no_cache_v1",
            "content_cache": False,
            "position_contract": self.position_contract(),
        }

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor,
        **_unused_context,
    ) -> torch.Tensor:
        if x.shape[:-1] != t.shape or x.shape[:-1] != c.shape[:-1]:
            raise ValueError(
                "position-wise flow inputs must align on every non-channel axis: "
                f"x={tuple(x.shape)}, t={tuple(t.shape)}, c={tuple(c.shape)}"
            )
        model_device = self.input_proj.weight.device
        model_dtype = self.input_proj.weight.dtype
        x = self.input_proj(x.to(device=model_device, dtype=model_dtype))
        y = self.time_embed(t.to(device=model_device)) + self.cond_embed(
            c.to(device=model_device, dtype=model_dtype)
        )
        for block in self.res_blocks:
            if self.grad_checkpointing and self.training:
                x = checkpoint(block, x, y, use_reentrant=False)
            else:
                x = block(x, y)
        return self.final_layer(x, y)


class PositionwiseFlowLoss(nn.Module):
    """Vectorized per-token rectified-flow objective with no latent context."""

    def __init__(
        self,
        target_channels: int,
        z_channels: int,
        depth: int,
        width: int,
        num_sampling_steps: int | str,
        grad_checkpointing: bool = False,
        time_scale: float = 1000.0,
        time_sampling: str = "logit_normal",
        logit_mean: float = 0.0,
        logit_std: float = 1.0,
        time_eps: float = 1.0e-4,
        uniform_mix: float = 0.1,
        solver: str = "heun",
        image_tokens_per_img: int = 256,
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
        self.image_tokens_per_img = int(image_tokens_per_img)
        if self.num_sampling_steps <= 0:
            raise ValueError("num_sampling_steps must be positive")
        if not 0.0 <= self.uniform_mix <= 1.0:
            raise ValueError("uniform_mix must be in [0, 1]")
        if not 0.0 <= self.time_eps < 0.5:
            raise ValueError("time_eps must be in [0, 0.5)")
        self.net = PositionwiseFlowMLP(
            in_channels=self.in_channels,
            model_channels=width,
            out_channels=self.in_channels,
            z_channels=z_channels,
            num_res_blocks=depth,
            grad_checkpointing=grad_checkpointing,
        )
        self.last_forward_stats: dict[str, torch.Tensor] = {}
        self.collect_guidance_diagnostics = False

    def _sample_times(self, shape: torch.Size, device: torch.device) -> torch.Tensor:
        if self.time_sampling in {"uniform", "rand", "random"}:
            t = torch.rand(shape, device=device)
        elif self.time_sampling in {"logit_normal", "lognorm", "logistic_normal"}:
            logits = (
                torch.randn(shape, device=device) * self.logit_std + self.logit_mean
            )
            t = torch.sigmoid(logits)
            if self.uniform_mix > 0.0:
                use_uniform = torch.rand(shape, device=device) < self.uniform_mix
                t = torch.where(use_uniform, torch.rand(shape, device=device), t)
        else:
            raise ValueError(
                f"Unknown image_flow_time_sampling={self.time_sampling!r}; "
                "expected logit_normal or uniform."
            )
        return t.clamp(self.time_eps, 1.0 - self.time_eps)

    def velocity(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        z: torch.Tensor,
        **_unused_context,
    ) -> torch.Tensor:
        return self.net(x_t, t.float() * self.time_scale, z)

    def forward(
        self,
        target: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor | None = None,
        sigma: torch.Tensor | None = None,
        image_positions: torch.Tensor | None = None,
        context_latents: torch.Tensor | None = None,
        record_stats: bool = True,
    ) -> torch.Tensor:
        del sigma, image_positions, context_latents
        model_device = self.net.input_proj.weight.device
        model_dtype = self.net.input_proj.weight.dtype
        target_float = target.to(device=model_device, dtype=torch.float32)
        z = z.to(device=model_device, dtype=model_dtype)
        if target_float.shape[:-1] != z.shape[:-1]:
            raise ValueError(
                "target and flow condition must align for position-wise loss: "
                f"target={tuple(target_float.shape)}, z={tuple(z.shape)}"
            )
        t = self._sample_times(target_float.shape[:-1], model_device)
        noise = torch.randn_like(target_float)
        x_t = (1.0 - t.unsqueeze(-1)) * noise + t.unsqueeze(-1) * target_float
        v_target = target_float - noise
        v_pred = self.velocity(x_t.to(model_dtype), t, z)
        token_loss = (v_pred.float() - v_target).square().mean(dim=-1)
        if mask is not None:
            weights = mask.to(device=model_device, dtype=torch.float32)
            loss = (token_loss * weights).sum() / weights.sum().clamp_min(1.0)
        else:
            loss = token_loss.mean()
        if record_stats:
            self.last_forward_stats = {
                "flow/loss": loss.detach().float(),
                "flow/v_mse": loss.detach().float(),
                "flow/t_mean": t.detach().float().mean(),
                "flow/t_min": t.detach().float().min(),
                "flow/t_max": t.detach().float().max(),
                "flow/x_t_rms": x_t.detach().float().pow(2).mean().sqrt(),
                "flow/v_target_rms": v_target.detach().float().pow(2).mean().sqrt(),
                "flow/v_pred_rms": v_pred.detach().float().pow(2).mean().sqrt(),
                "flow/nonfinite_count": (~torch.isfinite(v_pred)).sum().float(),
                "flow/context_tokens_consumed": torch.zeros(
                    (), device=model_device, dtype=torch.float32
                ),
            }
        else:
            self.last_forward_stats = {}
        return loss

    def empty_latent_mixer_cache(self, batch_size: int = 1, capacity=None):
        del batch_size, capacity
        return None

    def append_latent_mixer_cache(self, cache, **_unused_context):
        del cache
        return None

    def stack_latent_mixer_caches(self, caches):
        del caches
        return None

    def prepare_latent_mixer_cache(self, **_unused_context):
        return None

    def set_attention_diagnostics(self, enabled: bool) -> None:
        del enabled

    def set_guidance_diagnostics(self, enabled: bool) -> None:
        self.collect_guidance_diagnostics = bool(enabled)

    def reset_guidance_diagnostics(self) -> None:
        return None

    @staticmethod
    def guidance_diagnostics() -> dict:
        return {}

    @staticmethod
    def _scheduled_cfg(cfg: float, schedule: str | None, progress: float) -> float:
        schedule = str(schedule or "constant").lower()
        if cfg == 1.0 or schedule in {"constant", "none", "off", ""}:
            return float(cfg)
        if schedule == "linear":
            progress = min(1.0, max(0.0, float(progress)))
            return 1.0 + (float(cfg) - 1.0) * progress
        raise ValueError(
            f"Unknown image flow cfg_schedule={schedule!r}; expected constant or linear."
        )

    def _guided_velocity(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        z: torch.Tensor,
        cfg: float,
    ) -> torch.Tensor:
        if cfg == 1.0 and z.shape[0] == x.shape[0]:
            return self.velocity(x, t, z)
        if z.shape[0] != x.shape[0] * 2:
            raise ValueError(
                "CFG requires conditional/unconditional conditions concatenated "
                f"on the first axis, got x={tuple(x.shape)}, z={tuple(z.shape)}"
            )
        paired_x = torch.cat([x, x], dim=0)
        paired_t = torch.cat([t, t], dim=0)
        conditional, unconditional = self.velocity(paired_x, paired_t, z).chunk(
            2,
            dim=0,
        )
        return unconditional + float(cfg) * (conditional - unconditional)

    @torch.no_grad()
    def sample(
        self,
        z: torch.Tensor,
        temperature: float = 1.0,
        cfg: float = 1.0,
        cfg_schedule: str = "constant",
        solver: str | None = None,
        num_steps: int | None = None,
        return_trace: bool = False,
        *,
        initial_noise: torch.Tensor | None = None,
        debug_finite: bool = False,
        debug_label: str = "",
        **_unused_context,
    ):
        model_device = self.net.input_proj.weight.device
        model_dtype = self.net.input_proj.weight.dtype
        z = z.to(device=model_device, dtype=model_dtype)
        x_shape = (z.shape[0] // 2, *z.shape[1:-1]) if cfg != 1.0 else z.shape[:-1]
        expected_shape = (*x_shape, self.in_channels)
        if initial_noise is None:
            x = torch.randn(expected_shape, device=model_device, dtype=torch.float32)
        else:
            if tuple(initial_noise.shape) != expected_shape:
                raise ValueError(
                    f"initial_noise must have shape {expected_shape}, got "
                    f"{tuple(initial_noise.shape)}"
                )
            if not initial_noise.is_floating_point():
                raise TypeError("initial_noise must have a floating dtype")
            x = initial_noise.to(device=model_device, dtype=torch.float32)
        x = x * float(temperature)
        if debug_finite and not bool(torch.isfinite(x).all().item()):
            raise FloatingPointError(
                f"non-finite initial state in position-wise flow sample {debug_label!r}"
            )
        steps = int(num_steps or self.num_sampling_steps)
        if steps <= 0:
            raise ValueError("num_steps must be positive")
        solver = str(solver or self.solver).lower()
        times = torch.linspace(
            0.0,
            1.0,
            steps + 1,
            device=model_device,
            dtype=torch.float32,
        )
        for step in range(steps):
            t = times[step].expand(x_shape)
            t_next = times[step + 1].expand(x_shape)
            dt = times[step + 1] - times[step]
            cfg_now = self._scheduled_cfg(cfg, cfg_schedule, step / steps)
            velocity = self._guided_velocity(
                x.to(model_dtype),
                t,
                z,
                cfg_now,
            ).float()
            if solver == "euler":
                x = x + dt * velocity
            elif solver == "heun":
                predictor = x + dt * velocity
                cfg_next = self._scheduled_cfg(
                    cfg,
                    cfg_schedule,
                    (step + 1) / steps,
                )
                next_velocity = self._guided_velocity(
                    predictor.to(model_dtype),
                    t_next,
                    z,
                    cfg_next,
                ).float()
                x = x + 0.5 * dt * (velocity + next_velocity)
            else:
                raise ValueError(
                    f"Unknown image_flow_solver={solver!r}; expected heun or euler."
                )
            if debug_finite and not bool(torch.isfinite(x).all().item()):
                raise FloatingPointError(
                    "non-finite state during position-wise flow sampling: "
                    f"label={debug_label!r}, step={step}"
                )
        output = x.to(dtype=model_dtype)
        if return_trace:
            return output, {
                "solver": solver,
                "num_steps": steps,
                "cfg_schedule": str(cfg_schedule),
            }
        return output
