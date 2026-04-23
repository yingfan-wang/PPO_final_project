from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import numpy as np
import torch
from torch import nn

from simple_spread_baseline.config import ACTION_DIM, BaselineConfig, N_AGENTS, OBS_DIM
from simple_spread_baseline.networks import LocalActorCritic
from simple_spread_baseline.utils import resolve_device

LANDMARK_SLICE_START = 4
LANDMARK_SLICE_END = LANDMARK_SLICE_START + N_AGENTS * 2
STOP_DISTANCE = 0.08


def explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    variance = np.var(y_true)
    if variance < 1e-8:
        return 0.0
    return float(1.0 - np.var(y_true - y_pred) / variance)


def flatten_time_env_agent(array: np.ndarray) -> np.ndarray:
    return array.reshape((-1,) + array.shape[3:])


def flatten_time_env_agent_scalar(array: np.ndarray) -> np.ndarray:
    return array.reshape(-1)


def target_indices_to_env_actions(
    observations: np.ndarray, target_indices: np.ndarray
) -> np.ndarray:
    target_indices = target_indices[..., 0]
    landmark_rel = observations[..., LANDMARK_SLICE_START:LANDMARK_SLICE_END].reshape(
        observations.shape[0], observations.shape[1], N_AGENTS, 2
    )
    env_idx = np.arange(observations.shape[0])[:, None]
    agent_idx = np.arange(observations.shape[1])[None, :]
    chosen_rel = landmark_rel[env_idx, agent_idx, target_indices]

    env_actions = np.zeros(
        (observations.shape[0], observations.shape[1], 1), dtype=np.int64
    )
    dx = chosen_rel[..., 0]
    dy = chosen_rel[..., 1]
    distance = np.linalg.norm(chosen_rel, axis=-1)
    horizontal_mask = np.abs(dx) > np.abs(dy)
    move_mask = distance >= STOP_DISTANCE

    env_actions[..., 0] = 0
    env_actions[..., 0] = np.where(
        move_mask & horizontal_mask & (dx < 0.0),
        1,
        env_actions[..., 0],
    )
    env_actions[..., 0] = np.where(
        move_mask & horizontal_mask & (dx >= 0.0),
        2,
        env_actions[..., 0],
    )
    env_actions[..., 0] = np.where(
        move_mask & ~horizontal_mask & (dy < 0.0),
        3,
        env_actions[..., 0],
    )
    env_actions[..., 0] = np.where(
        move_mask & ~horizontal_mask & (dy >= 0.0),
        4,
        env_actions[..., 0],
    )
    return env_actions


class RolloutBuffer:
    def __init__(
        self,
        rollout_steps: int,
        num_envs: int,
        num_agents: int,
        obs_dim: int,
        gamma: float,
        gae_lambda: float,
    ) -> None:
        self.rollout_steps = rollout_steps
        self.num_envs = num_envs
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.reset()

    def reset(self) -> None:
        shape = (self.rollout_steps, self.num_envs, self.num_agents)
        self.observations = np.zeros(shape + (self.obs_dim,), dtype=np.float32)
        self.targets = np.zeros(shape + (1,), dtype=np.int64)
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
        targets: np.ndarray,
        log_probs: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        values: np.ndarray,
    ) -> None:
        self.observations[self.position] = observations
        self.targets[self.position] = targets
        self.log_probs[self.position] = log_probs
        self.rewards[self.position] = rewards
        self.dones[self.position] = dones
        self.values[self.position] = values
        self.position += 1

    def compute_returns_and_advantages(self, last_values: np.ndarray) -> None:
        last_advantage = np.zeros((self.num_envs, self.num_agents), dtype=np.float32)
        for step in reversed(range(self.rollout_steps)):
            next_values = last_values if step == self.rollout_steps - 1 else self.values[step + 1]
            next_non_terminal = 1.0 - self.dones[step][:, None]
            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
            last_advantage = delta + self.gamma * self.gae_lambda * next_non_terminal * last_advantage
            self.advantages[step] = last_advantage
        self.returns = self.advantages + self.values

    def get_training_arrays(self) -> dict[str, np.ndarray]:
        advantages = flatten_time_env_agent_scalar(self.advantages)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return {
            "observations": flatten_time_env_agent(self.observations),
            "targets": flatten_time_env_agent(self.targets),
            "log_probs": flatten_time_env_agent_scalar(self.log_probs),
            "advantages": advantages.astype(np.float32),
            "returns": flatten_time_env_agent_scalar(self.returns),
            "values": flatten_time_env_agent_scalar(self.values),
        }


class PPOAgent:
    def __init__(
        self,
        config: BaselineConfig,
        obs_dim: int = OBS_DIM,
        action_dim: int = ACTION_DIM,
    ) -> None:
        if config.continuous_actions:
            raise ValueError("The baseline currently supports discrete actions only.")
        self.config = config
        self.obs_dim = obs_dim
        self.env_action_dim = 1
        self.n_targets = N_AGENTS
        self.action_dim = action_dim
        self.device = resolve_device(config.device)
        self.model = LocalActorCritic(obs_dim, config.hidden_sizes).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=config.learning_rate, eps=1e-5
        )

    def select_actions(
        self, observations: np.ndarray, deterministic: bool = False
    ) -> np.ndarray:
        flat_observations = observations.reshape(-1, self.obs_dim)
        obs_tensor = torch.as_tensor(flat_observations, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            target_indices, _, _ = self.model.act(obs_tensor, deterministic=deterministic)
        target_indices = target_indices.cpu().numpy().reshape(observations.shape[0], N_AGENTS, 1)
        return target_indices_to_env_actions(observations, target_indices)

    def rollout_step(
        self, observations: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        flat_observations = observations.reshape(-1, self.obs_dim)
        obs_tensor = torch.as_tensor(flat_observations, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            target_indices, log_probs, values = self.model.act(obs_tensor, deterministic=False)
        target_indices_np = target_indices.cpu().numpy().reshape(observations.shape[0], N_AGENTS, 1)
        env_actions = target_indices_to_env_actions(observations, target_indices_np)
        batch_shape = (observations.shape[0], N_AGENTS)
        return (
            env_actions,
            target_indices_np,
            log_probs.cpu().numpy().reshape(batch_shape),
            values.cpu().numpy().reshape(batch_shape),
        )

    def get_values(self, observations: np.ndarray) -> np.ndarray:
        flat_observations = observations.reshape(-1, self.obs_dim)
        obs_tensor = torch.as_tensor(flat_observations, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            values = self.model.get_value(obs_tensor)
        return values.cpu().numpy().reshape(observations.shape[0], N_AGENTS)

    def build_buffer(self) -> RolloutBuffer:
        return RolloutBuffer(
            rollout_steps=self.config.rollout_steps,
            num_envs=self.config.num_envs,
            num_agents=N_AGENTS,
            obs_dim=self.obs_dim,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
        )

    def update(self, buffer: RolloutBuffer) -> dict[str, float]:
        arrays = buffer.get_training_arrays()
        num_samples = arrays["advantages"].shape[0]
        minibatch_size = min(self.config.minibatch_size, num_samples)

        observations = torch.as_tensor(
            arrays["observations"], dtype=torch.float32, device=self.device
        )
        targets = torch.as_tensor(arrays["targets"], dtype=torch.long, device=self.device)
        old_log_probs = torch.as_tensor(
            arrays["log_probs"], dtype=torch.float32, device=self.device
        )
        advantages = torch.as_tensor(
            arrays["advantages"], dtype=torch.float32, device=self.device
        )
        returns = torch.as_tensor(arrays["returns"], dtype=torch.float32, device=self.device)

        metrics = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
        }
        num_updates = 0

        for _ in range(self.config.update_epochs):
            permutation = np.random.permutation(num_samples)
            for start in range(0, num_samples, minibatch_size):
                batch_indices = permutation[start : start + minibatch_size]

                batch_obs = observations[batch_indices]
                batch_targets = targets[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                new_log_probs, entropy, new_values = self.model.evaluate_actions(
                    batch_obs, batch_targets
                )
                log_ratio = new_log_probs - batch_old_log_probs
                ratio = torch.exp(log_ratio)

                unclipped = ratio * batch_advantages
                clipped = torch.clamp(
                    ratio, 1.0 - self.config.clip_coef, 1.0 + self.config.clip_coef
                ) * batch_advantages
                policy_loss = -torch.min(unclipped, clipped).mean()

                value_loss = 0.5 * (batch_returns - new_values).pow(2).mean()
                entropy_loss = entropy.mean()

                loss = (
                    policy_loss
                    + self.config.vf_coef * value_loss
                    - self.config.ent_coef * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                metrics["policy_loss"] += float(policy_loss.item())
                metrics["value_loss"] += float(value_loss.item())
                metrics["entropy"] += float(entropy_loss.item())
                metrics["approx_kl"] += float(((ratio - 1.0) - log_ratio).mean().item())
                metrics["clip_fraction"] += float(
                    ((ratio - 1.0).abs() > self.config.clip_coef).float().mean().item()
                )
                num_updates += 1

        for key in metrics:
            metrics[key] /= max(num_updates, 1)
        metrics["explained_variance"] = explained_variance(
            y_pred=arrays["values"],
            y_true=arrays["returns"],
        )
        return metrics

    def save(self, checkpoint_path: str | Path, metadata: dict | None = None) -> None:
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": {
                    field.name: getattr(self.config, field.name)
                    for field in fields(BaselineConfig)
                },
                "obs_dim": self.obs_dim,
                "action_dim": self.action_dim,
                "state_dict": self.model.state_dict(),
                "metadata": metadata or {},
            },
            checkpoint_path,
        )

    @classmethod
    def load(cls, checkpoint_path: str | Path, device: str = "cpu") -> "PPOAgent":
        payload = torch.load(checkpoint_path, map_location=resolve_device(device))
        config_dict = dict(payload["config"])
        config_dict["device"] = device
        config_dict["hidden_sizes"] = tuple(config_dict["hidden_sizes"])
        agent = cls(BaselineConfig(**config_dict), obs_dim=int(payload["obs_dim"]))
        agent.model.load_state_dict(payload["state_dict"])
        agent.model.to(agent.device)
        agent.model.eval()
        return agent
