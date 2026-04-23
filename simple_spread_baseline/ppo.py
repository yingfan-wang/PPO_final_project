from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch import nn
from dataclasses import fields

from simple_spread_baseline.config import (
    ACTION_DIM,
    DISCRETE_POLICY_ACTION_DIM,
    POLICY_ACTION_DIM,
    BaselineConfig,
    N_AGENTS,
    OBS_DIM,
)
from simple_spread_baseline.networks import LocalActorCritic
from simple_spread_baseline.rollout_buffer import RolloutBuffer
from simple_spread_baseline.utils import resolve_device


def explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    variance = np.var(y_true)
    if variance < 1e-8:
        return 0.0
    return float(1.0 - np.var(y_true - y_pred) / variance)


class PPOAgent:
    def __init__(
        self,
        config: BaselineConfig,
        obs_dim: int = OBS_DIM,
        action_dim: int = ACTION_DIM,
    ) -> None:
        self.config = config
        self.obs_dim = obs_dim
        self.continuous_actions = config.continuous_actions
        self.env_action_dim = action_dim if self.continuous_actions else 1
        self.policy_action_dim = (
            POLICY_ACTION_DIM if self.continuous_actions else DISCRETE_POLICY_ACTION_DIM
        )
        self.device = resolve_device(config.device)
        self.model = LocalActorCritic(
            obs_dim,
            config.hidden_sizes,
            continuous_actions=self.continuous_actions,
            action_dim=self.policy_action_dim,
        ).to(
            self.device
        )
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=config.learning_rate, eps=1e-5
        )

    def select_actions(
        self, observations: np.ndarray, deterministic: bool = False
    ) -> np.ndarray:
        flat_observations = observations.reshape(-1, self.obs_dim)
        obs_tensor = torch.as_tensor(flat_observations, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            env_actions, _, _, _ = self.model.act(
                obs_tensor, deterministic=deterministic
            )
        return env_actions.cpu().numpy().reshape(
            observations.shape[0], N_AGENTS, self.env_action_dim
        )

    def rollout_step(
        self, observations: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        flat_observations = observations.reshape(-1, self.obs_dim)
        obs_tensor = torch.as_tensor(flat_observations, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            env_actions, policy_actions, log_probs, values = self.model.act(
                obs_tensor, deterministic=False
            )
        batch_shape = (observations.shape[0], N_AGENTS)
        return (
            env_actions.cpu().numpy().reshape(batch_shape + (self.env_action_dim,)),
            policy_actions.cpu().numpy().reshape(batch_shape + (self.policy_action_dim,)),
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
            action_dim=self.policy_action_dim,
            action_dtype=np.float32 if self.continuous_actions else np.int64,
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
        action_dtype = torch.float32 if self.continuous_actions else torch.long
        actions = torch.as_tensor(arrays["actions"], dtype=action_dtype, device=self.device)
        old_log_probs = torch.as_tensor(
            arrays["log_probs"], dtype=torch.float32, device=self.device
        )
        advantages = torch.as_tensor(
            arrays["advantages"], dtype=torch.float32, device=self.device
        )
        returns = torch.as_tensor(arrays["returns"], dtype=torch.float32, device=self.device)
        old_values = torch.as_tensor(arrays["values"], dtype=torch.float32, device=self.device)

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
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                batch_old_values = old_values[batch_indices]

                new_log_probs, entropy, new_values = self.model.evaluate_actions(
                    batch_obs, batch_actions
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
                "action_dim": self.env_action_dim,
                "policy_action_dim": self.policy_action_dim,
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
        agent = cls(BaselineConfig(**config_dict))
        agent.model.load_state_dict(payload["state_dict"])
        agent.model.to(agent.device)
        agent.model.eval()
        return agent
