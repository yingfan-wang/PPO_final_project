from __future__ import annotations

import numpy as np

from simple_spread_baseline.wrappers import (
    flatten_time_env_agent,
    flatten_time_env_agent_scalar,
)


class RolloutBuffer:
    def __init__(
        self,
        rollout_steps: int,
        num_envs: int,
        num_agents: int,
        obs_dim: int,
        action_dim: int,
        gamma: float,
        gae_lambda: float,
        action_dtype: np.dtype = np.float32,
    ) -> None:
        self.rollout_steps = rollout_steps
        self.num_envs = num_envs
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_dtype = np.dtype(action_dtype)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.reset()

    def reset(self) -> None:
        shape = (self.rollout_steps, self.num_envs, self.num_agents)
        self.observations = np.zeros(shape + (self.obs_dim,), dtype=np.float32)
        self.actions = np.zeros(shape + (self.action_dim,), dtype=self.action_dtype)
        self.log_probs = np.zeros(shape, dtype=np.float32)
        self.rewards = np.zeros(shape, dtype=np.float32)
        self.dones = np.zeros((self.rollout_steps, self.num_envs), dtype=np.float32)
        self.values = np.zeros(shape, dtype=np.float32)
        self.advantages = np.zeros(shape, dtype=np.float32)
        self.returns = np.zeros(shape, dtype=np.float32)
        self.position = 0

    def add(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        log_probs: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        values: np.ndarray,
    ) -> None:
        self.observations[self.position] = observations
        self.actions[self.position] = actions
        self.log_probs[self.position] = log_probs
        self.rewards[self.position] = rewards
        self.dones[self.position] = dones
        self.values[self.position] = values
        self.position += 1

    def compute_returns_and_advantages(self, last_values: np.ndarray) -> None:
        last_advantage = np.zeros((self.num_envs, self.num_agents), dtype=np.float32)
        for step in reversed(range(self.rollout_steps)):
            if step == self.rollout_steps - 1:
                next_values = last_values
            else:
                next_values = self.values[step + 1]
            next_non_terminal = 1.0 - self.dones[step][:, None]
            delta = (
                self.rewards[step]
                + self.gamma * next_values * next_non_terminal
                - self.values[step]
            )
            last_advantage = (
                delta
                + self.gamma * self.gae_lambda * next_non_terminal * last_advantage
            )
            self.advantages[step] = last_advantage
        self.returns = self.advantages + self.values

    def get_training_arrays(self) -> dict[str, np.ndarray]:
        advantages = flatten_time_env_agent_scalar(self.advantages)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return {
            "observations": flatten_time_env_agent(self.observations),
            "actions": flatten_time_env_agent(self.actions),
            "log_probs": flatten_time_env_agent_scalar(self.log_probs),
            "advantages": advantages.astype(np.float32),
            "returns": flatten_time_env_agent_scalar(self.returns),
            "values": flatten_time_env_agent_scalar(self.values),
        }
