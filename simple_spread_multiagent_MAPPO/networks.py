from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Categorical

from simple_spread_baseline.config import N_AGENTS
from simple_spread_baseline.networks import build_backbone, orthogonal_init
from simple_spread_multiagent_MAPPO.expert import ASSIGNMENTS


class JointAssignmentActor(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        hidden_sizes: tuple[int, ...],
        action_dim: int,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.backbone, hidden_output_dim = build_backbone(obs_dim, hidden_sizes)
        self.score_head = nn.Linear(hidden_output_dim, N_AGENTS * N_AGENTS)
        orthogonal_init(self.score_head, gain=0.01)
        self.register_buffer(
            "assignment_index_tensor",
            torch.as_tensor(ASSIGNMENTS, dtype=torch.long),
            persistent=False,
        )

    def _dist(self, obs: torch.Tensor) -> Categorical:
        features = self.backbone(obs)
        residual_scores = self.score_head(features).view(-1, N_AGENTS, N_AGENTS)
        landmark_vectors = obs[:, : N_AGENTS * N_AGENTS * 2].view(-1, N_AGENTS, N_AGENTS, 2)
        distance_prior_scores = -torch.linalg.vector_norm(landmark_vectors, dim=-1)
        pair_scores = distance_prior_scores + residual_scores
        logits = pair_scores[
            :,
            torch.arange(N_AGENTS, device=obs.device).unsqueeze(0),
            self.assignment_index_tensor,
        ].sum(dim=-1)
        return Categorical(logits=logits)

    def sample(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dist = self._dist(obs)
        actions = torch.argmax(dist.logits, dim=-1) if deterministic else dist.sample()
        log_probs = dist.log_prob(actions)
        return actions, log_probs

    def log_prob(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self._dist(obs).log_prob(actions.long())

    def entropy(self, obs: torch.Tensor) -> torch.Tensor:
        return self._dist(obs).entropy()
