from __future__ import annotations

import torch
from torch import nn

from simple_spread_baseline.networks import build_mlp, orthogonal_init


class CentralizedCritic(nn.Module):
    def __init__(self, critic_input_dim: int, hidden_sizes: tuple[int, ...]) -> None:
        super().__init__()
        self.network = build_mlp(
            input_dim=critic_input_dim,
            hidden_sizes=hidden_sizes,
            output_dim=1,
            output_gain=1.0,
        )
        if isinstance(self.network[-1], nn.Linear):
            orthogonal_init(self.network[-1], gain=1.0)

    def forward(self, critic_input: torch.Tensor) -> torch.Tensor:
        return self.network(critic_input).squeeze(-1)
