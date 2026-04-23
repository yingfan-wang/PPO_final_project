from __future__ import annotations

import math

import torch
from torch import nn
from torch.distributions import Categorical
from torch.distributions import Normal

from simple_spread_baseline.config import ACTION_DIM, POLICY_ACTION_DIM


def orthogonal_init(module: nn.Module, gain: float = 1.0) -> None:
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        nn.init.constant_(module.bias, 0.0)


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


class SquashedGaussianActor(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        hidden_sizes: tuple[int, ...],
        action_dim: int = POLICY_ACTION_DIM,
    ) -> None:
        super().__init__()
        self.eps = 1e-6
        self.action_dim = action_dim
        self.backbone, hidden_output_dim = build_backbone(obs_dim, hidden_sizes)
        self.mean_head = nn.Linear(hidden_output_dim, action_dim)
        orthogonal_init(self.mean_head, gain=0.01)
        self.log_std = nn.Parameter(torch.full((action_dim,), 0.0))

    def _dist(self, obs: torch.Tensor) -> Normal:
        features = self.backbone(obs)
        mean = self.mean_head(features)
        std = self.log_std.clamp(-5.0, 2.0).exp().expand_as(mean)
        return Normal(mean, std)

    def sample(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dist = self._dist(obs)
        pre_tanh = dist.mean if deterministic else dist.rsample()
        action = torch.tanh(pre_tanh)
        log_prob = self.log_prob(obs, action)
        return action, log_prob

    def log_prob(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        dist = self._dist(obs)
        clipped_actions = actions.clamp(-1.0 + self.eps, 1.0 - self.eps)
        pre_tanh = 0.5 * (
            torch.log1p(clipped_actions) - torch.log1p(-clipped_actions)
        )
        log_det = torch.log(1.0 - clipped_actions.pow(2) + self.eps)
        return (dist.log_prob(pre_tanh) - log_det).sum(dim=-1)

    def entropy(self, obs: torch.Tensor) -> torch.Tensor:
        return self._dist(obs).entropy().sum(dim=-1)

    @staticmethod
    def to_env_actions(policy_actions: torch.Tensor) -> torch.Tensor:
        env_actions = torch.zeros(
            (*policy_actions.shape[:-1], ACTION_DIM),
            dtype=policy_actions.dtype,
            device=policy_actions.device,
        )
        horizontal = policy_actions[..., 0]
        vertical = policy_actions[..., 1]
        env_actions[..., 1] = torch.relu(-horizontal)
        env_actions[..., 2] = torch.relu(horizontal)
        env_actions[..., 3] = torch.relu(-vertical)
        env_actions[..., 4] = torch.relu(vertical)
        return env_actions


class CategoricalActor(nn.Module):
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


class LocalActorCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        hidden_sizes: tuple[int, ...],
        continuous_actions: bool,
        action_dim: int = POLICY_ACTION_DIM,
    ) -> None:
        super().__init__()
        self.continuous_actions = continuous_actions
        if continuous_actions:
            self.actor = SquashedGaussianActor(
                obs_dim=obs_dim,
                hidden_sizes=hidden_sizes,
                action_dim=action_dim,
            )
        else:
            self.actor = CategoricalActor(
                obs_dim=obs_dim,
                hidden_sizes=hidden_sizes,
                action_dim=ACTION_DIM,
            )
        self.critic = build_mlp(obs_dim, hidden_sizes, 1, output_gain=1.0)
        if isinstance(self.critic[-1], nn.Linear):
            orthogonal_init(self.critic[-1], gain=1.0)

    def act(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        policy_actions, log_probs = self.actor.sample(obs, deterministic=deterministic)
        if self.continuous_actions:
            env_actions = self.actor.to_env_actions(policy_actions)
            stored_actions = policy_actions
        else:
            env_actions = policy_actions.unsqueeze(-1)
            stored_actions = env_actions
        values = self.critic(obs).squeeze(-1)
        return env_actions, stored_actions, log_probs, values

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        log_probs = self.actor.log_prob(obs, actions)
        entropy = self.actor.entropy(obs)
        values = self.critic(obs).squeeze(-1)
        return log_probs, entropy, values

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)
