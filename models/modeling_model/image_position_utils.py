import math

import torch


def sincos_1d_position_embedding(
    positions: torch.Tensor,
    dim: int,
    max_period: float = 10000.0,
) -> torch.Tensor:
    if dim <= 0:
        return torch.zeros((positions.numel(), 0), dtype=torch.float32, device=positions.device)
    n_freqs = (dim + 1) // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(n_freqs, dtype=torch.float32, device=positions.device)
        / max(n_freqs, 1)
    )
    args = positions.float().reshape(-1, 1) * freqs.reshape(1, -1)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)[:, :dim]


def build_2d_sincos_position_embedding(
    num_positions: int,
    dim: int,
    device=None,
) -> torch.Tensor:
    side = int(num_positions ** 0.5)
    if side * side != int(num_positions):
        raise ValueError(f"2D sin-cos image positions require a square grid, got {num_positions} tokens")
    positions = torch.arange(num_positions, dtype=torch.long, device=device)
    rows = positions.div(side, rounding_mode="floor").float()
    cols = (positions % side).float()
    row_dim = dim // 2
    col_dim = dim - row_dim
    return torch.cat(
        [
            sincos_1d_position_embedding(rows, row_dim),
            sincos_1d_position_embedding(cols, col_dim),
        ],
        dim=-1,
    )
