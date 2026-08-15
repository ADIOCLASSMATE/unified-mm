"""Small shared rectified-flow training-state helper.

This module owns only the behavior-equivalent RF interpolation used by both
the static flow loss and the isolated Dynamic-XT ablation.  It deliberately
contains no model, attention, or loss-normalization policy.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RectifiedFlowTrainingState:
    """Pre-sampled RF variables with ``t=0`` noise and ``t=1`` data."""

    t: torch.Tensor
    noise: torch.Tensor
    x_t: torch.Tensor
    v_target: torch.Tensor

    def slice(self, start: int, end: int) -> RectifiedFlowTrainingState:
        return RectifiedFlowTrainingState(
            t=self.t[start:end],
            noise=self.noise[start:end],
            x_t=self.x_t[start:end],
            v_target=self.v_target[start:end],
        )


def sample_rectified_flow_training_state(
    target: torch.Tensor,
    *,
    sample_times: Callable[[int, torch.device], torch.Tensor],
    device: torch.device,
) -> RectifiedFlowTrainingState:
    """Sample the exact RF state historically constructed inside ``FlowLoss``.

    Time remains per token because ``target.shape[:-1]`` is flattened before
    calling the existing time sampler.  Sampling order, FP32 interpolation,
    and the normal draw shape intentionally match the static implementation.
    """

    target_float = target.to(device=device, dtype=torch.float32)
    batch_shape = target_float.shape[:-1]
    sample_count = 1
    for dimension in batch_shape:
        sample_count *= int(dimension)
    t = sample_times(sample_count, device).view(batch_shape)
    noise = torch.randn(
        target_float.shape,
        device=device,
        dtype=torch.float32,
    )
    t_view = t.unsqueeze(-1).float()
    x_t = (1.0 - t_view) * noise + t_view * target_float
    return RectifiedFlowTrainingState(
        t=t,
        noise=noise,
        x_t=x_t,
        v_target=target_float - noise,
    )


__all__ = [
    "RectifiedFlowTrainingState",
    "sample_rectified_flow_training_state",
]
