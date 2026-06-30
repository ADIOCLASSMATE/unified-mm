import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mar_diffloss import SimpleMLPAdaLN


class FlowLoss(nn.Module):
    """Rectified-flow loss over per-token continuous image latents.

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

        self.net = SimpleMLPAdaLN(
            in_channels=self.in_channels,
            model_channels=width,
            out_channels=self.in_channels,
            z_channels=z_channels,
            num_res_blocks=depth,
            grad_checkpointing=grad_checkpointing,
            mlp_ratio=self.mlp_ratio,
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
                t_uniform = torch.rand(batch_size, device=device)
                t = torch.where(use_uniform, t_uniform, t)
        else:
            raise ValueError(
                f"Unknown image_flow_time_sampling={self.time_sampling!r}; "
                "expected logit_normal or uniform."
            )
        return t.clamp(self.time_eps, 1.0 - self.time_eps)

    def _scale_time(self, t: torch.Tensor) -> torch.Tensor:
        return t.to(dtype=torch.float32) * self.time_scale

    def velocity(self, x_t: torch.Tensor, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        model_dtype = self.net.input_proj.weight.dtype
        model_device = self.net.input_proj.weight.device
        x_t = x_t.to(device=model_device, dtype=model_dtype)
        z = z.to(device=model_device, dtype=model_dtype)
        t = t.to(device=model_device)
        return self.net(x_t, self._scale_time(t), c=z)

    def forward(self, target, z, mask=None):
        model_dtype = self.net.input_proj.weight.dtype
        model_device = self.net.input_proj.weight.device
        target = target.to(device=model_device, dtype=model_dtype)
        z = z.to(device=model_device, dtype=model_dtype)
        if mask is not None:
            mask = mask.to(device=model_device, dtype=target.dtype)

        t = self._sample_times(target.shape[0], target.device)
        noise = torch.randn_like(target)
        t_view = t.view(-1, *([1] * (target.ndim - 1))).to(dtype=target.dtype)
        x_t = (1.0 - t_view) * noise + t_view * target
        v_target = target - noise
        v_pred = self.velocity(x_t, t, z)
        loss = mean_flat((v_pred.float() - v_target.float()) ** 2)
        if mask is not None:
            mask = mask.float()
            loss = (loss * mask).sum() / mask.sum().clamp_min(1.0)

        loss_mean = loss.mean()
        self.last_forward_stats = {
            "flow/loss": loss_mean.detach().float(),
            "flow/v_mse": loss_mean.detach().float(),
            "flow/t_mean": t.detach().float().mean(),
            "flow/t_min": t.detach().float().min(),
            "flow/t_max": t.detach().float().max(),
            "flow/x_t_rms": self._rms_stat(x_t),
            "flow/v_target_rms": self._rms_stat(v_target),
            "flow/v_pred_rms": self._rms_stat(v_pred),
        }
        return loss_mean

    def estimate_x0(self, x_t: torch.Tensor, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        v = self.velocity(x_t, t, z)
        t_view = t.view(-1, *([1] * (x_t.ndim - 1))).to(dtype=x_t.dtype)
        return x_t + (1.0 - t_view) * v

    def sample(
        self,
        z,
        temperature=1.0,
        cfg=1.0,
        solver=None,
        num_steps=None,
        return_trace=False,
    ):
        model_dtype = self.net.input_proj.weight.dtype
        model_device = self.net.input_proj.weight.device
        z = z.to(device=model_device, dtype=model_dtype)
        if cfg != 1.0 and z.shape[0] % 2 != 0:
            raise ValueError(
                f"cfg != 1.0 requires paired conditional/unconditional conditions; got batch {z.shape[0]}"
            )

        steps = int(num_steps or self.num_sampling_steps)
        if steps <= 0:
            raise ValueError(f"num_steps must be positive, got {steps}")
        solver = str(solver or self.solver).lower()
        x_batch = z.shape[0] // 2 if cfg != 1.0 else z.shape[0]
        x = torch.randn(x_batch, self.in_channels, device=z.device, dtype=z.dtype) * float(temperature)
        times = torch.linspace(0.0, 1.0, steps + 1, device=z.device, dtype=torch.float32)

        for idx in range(steps):
            t = times[idx].expand(x_batch)
            t_next = times[idx + 1].expand(x_batch)
            dt = (times[idx + 1] - times[idx]).to(dtype=x.dtype)
            v = self._guided_velocity(x, t, z, cfg)
            if solver == "euler":
                x = x + dt * v
            elif solver == "heun":
                x_euler = x + dt * v
                v_next = self._guided_velocity(x_euler, t_next, z, cfg)
                x = x + 0.5 * dt * (v + v_next)
            else:
                raise ValueError(f"Unknown image_flow_solver={solver!r}; expected heun or euler.")

        if return_trace:
            return x, {"solver": solver, "num_steps": steps}
        return x

    def _guided_velocity(self, x: torch.Tensor, t: torch.Tensor, z: torch.Tensor, cfg: float) -> torch.Tensor:
        if cfg == 1.0:
            return self.velocity(x, t, z)
        x_pair = torch.cat([x, x], dim=0)
        t_pair = torch.cat([t, t], dim=0)
        v_pair = self.velocity(x_pair, t_pair, z)
        v_cond, v_uncond = torch.chunk(v_pair, 2, dim=0)
        return v_uncond + float(cfg) * (v_cond - v_uncond)


def mean_flat(tensor):
    return tensor.mean(dim=list(range(1, len(tensor.shape))))
