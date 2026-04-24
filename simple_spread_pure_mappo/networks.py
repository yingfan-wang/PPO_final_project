from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Categorical

from simple_spread_baseline.config import ACTION_DIM
from simple_spread_baseline.networks import build_backbone, build_mlp, orthogonal_init


class PrimitiveActor(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        hidden_sizes: tuple[int, ...],
        action_dim: int = ACTION_DIM,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.backbone, hidden_output_dim = build_backbone(obs_dim, hidden_sizes)
        self.logits_head = nn.Linear(hidden_output_dim, action_dim)
        orthogonal_init(self.logits_head, gain=0.01)

    def _dist(self, obs: torch.Tensor) -> Categorical:
        features = self.backbone(obs)
        logits = self.logits_head(features)
        return Categorical(logits=logits)

    def sample(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dist = self._dist(obs)
        actions = torch.argmax(dist.logits, dim=-1) if deterministic else dist.sample()
        log_probs = dist.log_prob(actions)
        return actions, log_probs

    def log_prob(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self._dist(obs).log_prob(actions.long().view(-1))

    def entropy(self, obs: torch.Tensor) -> torch.Tensor:
        return self._dist(obs).entropy()


class CentralizedCritic(nn.Module):
    def __init__(
        self,
        critic_input_dim: int,
        hidden_sizes: tuple[int, ...],
    ) -> None:
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
