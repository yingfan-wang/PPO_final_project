from __future__ import annotations

import math

import torch
from torch import nn
from torch.distributions import Categorical

from simple_spread_baseline.config import N_AGENTS


def orthogonal_init(module: nn.Module, gain: float = 1.0) -> None:
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        nn.init.constant_(module.bias, 0.0)


def build_backbone(
    input_dim: int,
    hidden_sizes: tuple[int, ...],
    activation_cls: type[nn.Module] = nn.Tanh,
) -> tuple[nn.Sequential, int]:
    if not hidden_sizes:
        return nn.Identity(), input_dim

    layers: list[nn.Module] = []
    last_dim = input_dim
    for hidden_dim in hidden_sizes:
        linear = nn.Linear(last_dim, hidden_dim)
        orthogonal_init(linear, gain=math.sqrt(2.0))
        layers.append(linear)
        layers.append(activation_cls())
        last_dim = hidden_dim
    return nn.Sequential(*layers), last_dim


def build_mlp(
    input_dim: int,
    hidden_sizes: tuple[int, ...],
    output_dim: int,
    activation_cls: type[nn.Module] = nn.Tanh,
    output_gain: float = 0.01,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    last_dim = input_dim
    for hidden_dim in hidden_sizes:
        layers.append(nn.Linear(last_dim, hidden_dim))
        layers.append(activation_cls())
        last_dim = hidden_dim
    layers.append(nn.Linear(last_dim, output_dim))
    network = nn.Sequential(*layers)
    for layer in network:
        gain = math.sqrt(2.0) if isinstance(layer, nn.Linear) else 1.0
        orthogonal_init(layer, gain=gain)
    if isinstance(network[-1], nn.Linear):
        orthogonal_init(network[-1], gain=output_gain)
    return network


class LandmarkSelectorActor(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        hidden_sizes: tuple[int, ...],
        n_landmarks: int = N_AGENTS,
    ) -> None:
        super().__init__()
        self.n_landmarks = n_landmarks
        self.backbone, hidden_output_dim = build_backbone(obs_dim, hidden_sizes)
        self.logits_head = nn.Linear(hidden_output_dim, n_landmarks)
        orthogonal_init(self.logits_head, gain=0.01)

    def _dist(self, obs: torch.Tensor) -> Categorical:
        features = self.backbone(obs)
        logits = self.logits_head(features)
        return Categorical(logits=logits)

    def sample(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dist = self._dist(obs)
        targets = torch.argmax(dist.logits, dim=-1) if deterministic else dist.sample()
        log_probs = dist.log_prob(targets)
        return targets, log_probs

    def log_prob(self, obs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self._dist(obs).log_prob(targets.long().view(-1))

    def entropy(self, obs: torch.Tensor) -> torch.Tensor:
        return self._dist(obs).entropy()


class LocalActorCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        hidden_sizes: tuple[int, ...],
    ) -> None:
        super().__init__()
        self.actor = LandmarkSelectorActor(
            obs_dim=obs_dim,
            hidden_sizes=hidden_sizes,
            n_landmarks=N_AGENTS,
        )
        self.critic_backbone, hidden_output_dim = build_backbone(obs_dim, hidden_sizes)
        self.value_head = nn.Linear(hidden_output_dim, 1)
        orthogonal_init(self.value_head, gain=1.0)

    def act(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        targets, log_probs = self.actor.sample(obs, deterministic=deterministic)
        values = self.get_value(obs)
        return targets, log_probs, values

    def evaluate_actions(
        self, obs: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        log_probs = self.actor.log_prob(obs, targets)
        entropy = self.actor.entropy(obs)
        values = self.get_value(obs)
        return log_probs, entropy, values

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        features = self.critic_backbone(obs)
        return self.value_head(features).squeeze(-1)
