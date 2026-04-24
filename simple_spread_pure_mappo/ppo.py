from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import numpy as np
import torch
from torch import nn

from simple_spread_baseline.config import ACTION_DIM, N_AGENTS, OBS_DIM, STATE_DIM
from simple_spread_baseline.utils import resolve_device
from simple_spread_pure_mappo.config import PureMAPPOConfig
from simple_spread_pure_mappo.env import build_actor_inputs, build_critic_inputs
from simple_spread_pure_mappo.networks import CentralizedCritic, PrimitiveActor


def explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    variance = np.var(y_true)
    if variance < 1e-8:
        return 0.0
    return float(1.0 - np.var(y_true - y_pred) / variance)


def flatten_time_env_agent(array: np.ndarray) -> np.ndarray:
    return array.reshape((-1,) + array.shape[3:])


def flatten_time_env_agent_scalar(array: np.ndarray) -> np.ndarray:
    return array.reshape(-1)


class CentralizedRolloutBuffer:
    def __init__(
        self,
        rollout_steps: int,
        num_envs: int,
        actor_obs_dim: int,
        critic_obs_dim: int,
        gamma: float,
        gae_lambda: float,
    ) -> None:
        self.rollout_steps = rollout_steps
        self.num_envs = num_envs
        self.actor_obs_dim = actor_obs_dim
        self.critic_obs_dim = critic_obs_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.reset()

    def reset(self) -> None:
        shape = (self.rollout_steps, self.num_envs, N_AGENTS)
        self.actor_observations = np.zeros(shape + (self.actor_obs_dim,), dtype=np.float32)
        self.critic_observations = np.zeros(
            shape + (self.critic_obs_dim,), dtype=np.float32
        )
        self.actions = np.zeros(shape, dtype=np.int64)
        self.log_probs = np.zeros(shape, dtype=np.float32)
        self.rewards = np.zeros(shape, dtype=np.float32)
        self.dones = np.zeros((self.rollout_steps, self.num_envs), dtype=np.float32)
        self.values = np.zeros(shape, dtype=np.float32)
        self.advantages = np.zeros(shape, dtype=np.float32)
        self.returns = np.zeros(shape, dtype=np.float32)
        self.position = 0

    def add(
        self,
        actor_observations: np.ndarray,
        critic_observations: np.ndarray,
        actions: np.ndarray,
        log_probs: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        values: np.ndarray,
    ) -> None:
        self.actor_observations[self.position] = actor_observations
        self.critic_observations[self.position] = critic_observations
        self.actions[self.position] = actions.astype(np.int64)
        self.log_probs[self.position] = log_probs
        self.rewards[self.position] = rewards
        self.dones[self.position] = dones
        self.values[self.position] = values
        self.position += 1

    def compute_returns_and_advantages(self, last_values: np.ndarray) -> None:
        last_advantage = np.zeros((self.num_envs, N_AGENTS), dtype=np.float32)
        for step in reversed(range(self.rollout_steps)):
            next_values = last_values if step == self.rollout_steps - 1 else self.values[step + 1]
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
            "actor_observations": flatten_time_env_agent(self.actor_observations),
            "critic_observations": flatten_time_env_agent(self.critic_observations),
            "actions": flatten_time_env_agent_scalar(self.actions).astype(np.int64),
            "log_probs": flatten_time_env_agent_scalar(self.log_probs),
            "advantages": advantages.astype(np.float32),
            "returns": flatten_time_env_agent_scalar(self.returns),
            "values": flatten_time_env_agent_scalar(self.values),
        }


class PureMAPPOAgent:
    def __init__(
        self,
        config: PureMAPPOConfig,
        obs_dim: int = OBS_DIM,
        state_dim: int = STATE_DIM,
        action_dim: int = ACTION_DIM,
    ) -> None:
        if config.continuous_actions:
            raise ValueError("Pure MAPPO currently supports discrete primitive actions only.")
        self.config = config
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.actor_input_dim = obs_dim + (N_AGENTS if config.use_agent_id else 0)
        self.critic_input_dim = state_dim + (N_AGENTS if config.use_agent_id else 0)
        self.device = resolve_device(config.device)

        self.actor = PrimitiveActor(
            obs_dim=self.actor_input_dim,
            hidden_sizes=config.actor_hidden_sizes,
            action_dim=action_dim,
        ).to(self.device)
        self.critic = CentralizedCritic(
            critic_input_dim=self.critic_input_dim,
            hidden_sizes=config.critic_hidden_sizes,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=config.learning_rate,
            eps=1e-5,
        )

    def select_actions(
        self, observations: np.ndarray, deterministic: bool = False
    ) -> np.ndarray:
        actor_inputs = build_actor_inputs(
            observations,
            use_agent_id=self.config.use_agent_id,
        )
        flat_actor_inputs = actor_inputs.reshape(-1, self.actor_input_dim)
        obs_tensor = torch.as_tensor(
            flat_actor_inputs, dtype=torch.float32, device=self.device
        )
        with torch.no_grad():
            actions, _ = self.actor.sample(obs_tensor, deterministic=deterministic)
        return actions.cpu().numpy().reshape(observations.shape[0], N_AGENTS, 1).astype(
            np.int64
        )

    def rollout_step(
        self,
        observations: np.ndarray,
        joint_states: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        actor_inputs = build_actor_inputs(
            observations,
            use_agent_id=self.config.use_agent_id,
        )
        critic_inputs = build_critic_inputs(
            joint_states,
            use_agent_id=self.config.use_agent_id,
        )
        flat_actor_inputs = actor_inputs.reshape(-1, self.actor_input_dim)
        flat_critic_inputs = critic_inputs.reshape(-1, self.critic_input_dim)
        actor_tensor = torch.as_tensor(
            flat_actor_inputs, dtype=torch.float32, device=self.device
        )
        critic_tensor = torch.as_tensor(
            flat_critic_inputs, dtype=torch.float32, device=self.device
        )
        with torch.no_grad():
            actions, log_probs = self.actor.sample(actor_tensor, deterministic=False)
            values = self.critic(critic_tensor)

        batch_shape = (observations.shape[0], N_AGENTS)
        actions_np = actions.cpu().numpy().reshape(batch_shape).astype(np.int64)
        return (
            actions_np[..., None],
            actions_np,
            log_probs.cpu().numpy().reshape(batch_shape).astype(np.float32),
            values.cpu().numpy().reshape(batch_shape).astype(np.float32),
        )

    def get_values(self, joint_states: np.ndarray) -> np.ndarray:
        critic_inputs = build_critic_inputs(
            joint_states,
            use_agent_id=self.config.use_agent_id,
        )
        flat_critic_inputs = critic_inputs.reshape(-1, self.critic_input_dim)
        critic_tensor = torch.as_tensor(
            flat_critic_inputs, dtype=torch.float32, device=self.device
        )
        with torch.no_grad():
            values = self.critic(critic_tensor)
        return values.cpu().numpy().reshape(joint_states.shape[0], N_AGENTS).astype(
            np.float32
        )

    def build_buffer(self) -> CentralizedRolloutBuffer:
        return CentralizedRolloutBuffer(
            rollout_steps=self.config.rollout_steps,
            num_envs=self.config.num_envs,
            actor_obs_dim=self.actor_input_dim,
            critic_obs_dim=self.critic_input_dim,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
        )

    def update(self, buffer: CentralizedRolloutBuffer) -> dict[str, float]:
        arrays = buffer.get_training_arrays()
        num_samples = arrays["advantages"].shape[0]
        minibatch_size = min(self.config.minibatch_size, num_samples)

        actor_observations = torch.as_tensor(
            arrays["actor_observations"], dtype=torch.float32, device=self.device
        )
        critic_observations = torch.as_tensor(
            arrays["critic_observations"], dtype=torch.float32, device=self.device
        )
        actions = torch.as_tensor(arrays["actions"], dtype=torch.long, device=self.device)
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

                batch_actor_obs = actor_observations[batch_indices]
                batch_critic_obs = critic_observations[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                new_log_probs = self.actor.log_prob(batch_actor_obs, batch_actions)
                entropy = self.actor.entropy(batch_actor_obs)
                new_values = self.critic(batch_critic_obs)
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
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    self.config.max_grad_norm,
                )
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
                    for field in fields(PureMAPPOConfig)
                },
                "obs_dim": self.obs_dim,
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
                "actor_state_dict": self.actor.state_dict(),
                "critic_state_dict": self.critic.state_dict(),
                "metadata": metadata or {},
            },
            checkpoint_path,
        )

    @classmethod
    def load(cls, checkpoint_path: str | Path, device: str = "cpu") -> "PureMAPPOAgent":
        payload = torch.load(checkpoint_path, map_location=resolve_device(device))
        config_dict = dict(payload["config"])
        config_dict["device"] = device
        config_dict["actor_hidden_sizes"] = tuple(config_dict["actor_hidden_sizes"])
        config_dict["critic_hidden_sizes"] = tuple(config_dict["critic_hidden_sizes"])
        agent = cls(
            PureMAPPOConfig(**config_dict),
            obs_dim=int(payload["obs_dim"]),
            state_dim=int(payload["state_dim"]),
            action_dim=int(payload["action_dim"]),
        )
        agent.actor.load_state_dict(payload["actor_state_dict"])
        agent.critic.load_state_dict(payload["critic_state_dict"])
        agent.actor.to(agent.device)
        agent.critic.to(agent.device)
        agent.actor.eval()
        agent.critic.eval()
        return agent
