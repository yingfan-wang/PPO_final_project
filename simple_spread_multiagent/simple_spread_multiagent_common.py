"""Shared helpers for the Simple Spread multi-agent PPO extension.

This track implements a compact MAPPO-style variant:

- one shared actor for all agents
- decentralized execution from each agent's local observation
- a centralized critic that receives the ordered joint observation
- optional one-hot agent IDs for both actor and critic
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from gymnasium import spaces
from mpe2 import simple_spread_v3
from torch import nn
from torch.distributions import Categorical

BASE_DIR = Path(__file__).resolve().parent
RUNS_DIR = BASE_DIR / "runs"
N_AGENTS = 3
LOCAL_RATIO = 0.5
DEFAULT_MAX_CYCLES = 100
DEFAULT_NUM_VEC_ENVS = 4


def _seed_action_spaces(env, seed: Optional[int]) -> None:
    if seed is None:
        return

    for agent_idx, agent in enumerate(env.possible_agents):
        env.action_space(agent).seed(seed + agent_idx)


def make_parallel_env(
    max_cycles: int,
    terminate_on_success: bool,
    render_mode=None,
    seed: Optional[int] = None,
):
    """Create one unwrapped MPE2 Simple Spread parallel environment."""

    env = simple_spread_v3.parallel_env(
        N=N_AGENTS,
        local_ratio=LOCAL_RATIO,
        max_cycles=max_cycles,
        continuous_actions=False,
        render_mode=render_mode,
        terminate_on_success=terminate_on_success,
    )
    _seed_action_spaces(env, seed)
    return env


def infer_env_spaces(max_cycles: int, terminate_on_success: bool):
    """Return agent order plus local observation/action dimensions."""

    env = make_parallel_env(
        max_cycles=max_cycles,
        terminate_on_success=terminate_on_success,
        render_mode=None,
        seed=0,
    )
    try:
        agents = list(env.possible_agents)
        obs_space = env.observation_space(agents[0])
        action_space = env.action_space(agents[0])
        if not isinstance(obs_space, spaces.Box):
            raise TypeError(f"Expected Box observations, got {obs_space!r}.")
        if not isinstance(action_space, spaces.Discrete):
            raise TypeError(f"Expected Discrete actions, got {action_space!r}.")
        local_obs_dim = int(np.prod(obs_space.shape))
        action_dim = int(action_space.n)
    finally:
        env.close()

    return agents, local_obs_dim, action_dim


def mean_agent_reward(rewards) -> float:
    if not rewards:
        return 0.0
    return float(sum(rewards.values())) / N_AGENTS


def safe_std(values) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.size <= 1:
        return 0.0
    return float(values.std(ddof=1))


def resolve_checkpoint_path(checkpoint_path: Union[str, Path]) -> Path:
    """Accept explicit .pt paths or checkpoint stems."""

    raw_path = Path(checkpoint_path).expanduser()
    candidate_paths = [raw_path]
    if raw_path.suffix != ".pt":
        candidate_paths.append(raw_path.with_suffix(".pt"))

    for candidate_path in candidate_paths:
        if candidate_path.exists():
            return candidate_path.resolve()

    checked_paths = ", ".join(str(path) for path in candidate_paths)
    raise FileNotFoundError(f"Could not find a MAPPO checkpoint at: {checked_paths}")


def observations_to_arrays(
    observations: dict[str, np.ndarray],
    agents: list[str],
    local_obs_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a PettingZoo observation dict into local and joint arrays."""

    local_obs = np.zeros((len(agents), local_obs_dim), dtype=np.float32)
    for agent_idx, agent in enumerate(agents):
        if agent in observations:
            local_obs[agent_idx] = np.asarray(
                observations[agent],
                dtype=np.float32,
            ).reshape(-1)

    joint_obs = local_obs.reshape(-1).astype(np.float32, copy=False)
    return local_obs, joint_obs


def make_agent_id_matrix(n_envs: int, n_agents: int) -> np.ndarray:
    return np.tile(np.arange(n_agents, dtype=np.int64), (n_envs, 1))


def mlp(input_dim: int, hidden_sizes: list[int], output_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    last_dim = input_dim
    for hidden_size in hidden_sizes:
        layers.append(nn.Linear(last_dim, hidden_size))
        layers.append(nn.Tanh())
        last_dim = hidden_size
    layers.append(nn.Linear(last_dim, output_dim))
    return nn.Sequential(*layers)


def orthogonal_init(module: nn.Module) -> None:
    for submodule in module.modules():
        if isinstance(submodule, nn.Linear):
            gain = np.sqrt(2)
            if submodule.out_features == 1:
                gain = 1.0
            nn.init.orthogonal_(submodule.weight, gain=gain)
            nn.init.constant_(submodule.bias, 0.0)


class SharedCentralizedActorCritic(nn.Module):
    """Shared actor with a centralized value function."""

    def __init__(
        self,
        local_obs_dim: int,
        joint_obs_dim: int,
        action_dim: int,
        n_agents: int,
        hidden_sizes: list[int],
        use_agent_id: bool,
    ):
        super().__init__()
        self.local_obs_dim = local_obs_dim
        self.joint_obs_dim = joint_obs_dim
        self.action_dim = action_dim
        self.n_agents = n_agents
        self.hidden_sizes = list(hidden_sizes)
        self.use_agent_id = use_agent_id

        id_dim = n_agents if use_agent_id else 0
        self.actor = mlp(local_obs_dim + id_dim, hidden_sizes, action_dim)
        self.critic = mlp(joint_obs_dim + id_dim, hidden_sizes, 1)
        orthogonal_init(self)
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        nn.init.constant_(self.actor[-1].bias, 0.0)
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)
        nn.init.constant_(self.critic[-1].bias, 0.0)

    def _append_agent_id(
        self,
        features: torch.Tensor,
        agent_ids: torch.Tensor,
    ) -> torch.Tensor:
        if not self.use_agent_id:
            return features

        one_hot = torch.nn.functional.one_hot(
            agent_ids.long(),
            num_classes=self.n_agents,
        ).to(dtype=features.dtype, device=features.device)
        return torch.cat([features, one_hot], dim=-1)

    def distribution(
        self,
        local_obs: torch.Tensor,
        agent_ids: torch.Tensor,
    ) -> Categorical:
        actor_input = self._append_agent_id(local_obs, agent_ids)
        logits = self.actor(actor_input)
        return Categorical(logits=logits)

    def values(
        self,
        joint_obs: torch.Tensor,
        agent_ids: torch.Tensor,
    ) -> torch.Tensor:
        critic_input = self._append_agent_id(joint_obs, agent_ids)
        return self.critic(critic_input).squeeze(-1)

    def act(
        self,
        local_obs: torch.Tensor,
        joint_obs: torch.Tensor,
        agent_ids: torch.Tensor,
        deterministic: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self.distribution(local_obs, agent_ids)
        if deterministic:
            actions = dist.probs.argmax(dim=-1)
        else:
            actions = dist.sample()
        log_probs = dist.log_prob(actions)
        values = self.values(joint_obs, agent_ids)
        return actions, log_probs, values

    def evaluate_actions(
        self,
        local_obs: torch.Tensor,
        joint_obs: torch.Tensor,
        agent_ids: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self.distribution(local_obs, agent_ids)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        values = self.values(joint_obs, agent_ids)
        return log_probs, entropy, values


@dataclass
class CheckpointMetadata:
    agents: list[str]
    local_obs_dim: int
    joint_obs_dim: int
    action_dim: int
    n_agents: int
    hidden_sizes: list[int]
    use_agent_id: bool


def save_checkpoint(
    path: Union[str, Path],
    model: SharedCentralizedActorCritic,
    metadata: CheckpointMetadata,
    config: dict,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "metadata": metadata.__dict__,
            "config": config,
        },
        path,
    )


def _torch_load(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_checkpoint(
    checkpoint_path: Union[str, Path],
    device: Union[str, torch.device] = "cpu",
) -> tuple[SharedCentralizedActorCritic, CheckpointMetadata, dict, Path]:
    resolved_path = resolve_checkpoint_path(checkpoint_path)
    torch_device = device if isinstance(device, torch.device) else torch.device(device)
    checkpoint = _torch_load(resolved_path, torch_device)
    metadata = CheckpointMetadata(**checkpoint["metadata"])
    model = SharedCentralizedActorCritic(
        local_obs_dim=metadata.local_obs_dim,
        joint_obs_dim=metadata.joint_obs_dim,
        action_dim=metadata.action_dim,
        n_agents=metadata.n_agents,
        hidden_sizes=metadata.hidden_sizes,
        use_agent_id=metadata.use_agent_id,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(torch_device)
    model.eval()
    return model, metadata, checkpoint.get("config", {}), resolved_path


def tensorize_eval_batch(
    observations: dict[str, np.ndarray],
    metadata: CheckpointMetadata,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    local_obs, joint_obs = observations_to_arrays(
        observations=observations,
        agents=metadata.agents,
        local_obs_dim=metadata.local_obs_dim,
    )
    joint_obs = np.repeat(joint_obs[None, :], metadata.n_agents, axis=0)
    agent_ids = np.arange(metadata.n_agents, dtype=np.int64)

    return (
        torch.as_tensor(local_obs, dtype=torch.float32, device=device),
        torch.as_tensor(joint_obs, dtype=torch.float32, device=device),
        torch.as_tensor(agent_ids, dtype=torch.long, device=device),
    )


def predict_parallel_actions(
    model: SharedCentralizedActorCritic,
    metadata: CheckpointMetadata,
    observations: dict[str, np.ndarray],
    device: torch.device,
    deterministic: bool = True,
) -> dict[str, int]:
    local_obs, joint_obs, agent_ids = tensorize_eval_batch(
        observations=observations,
        metadata=metadata,
        device=device,
    )
    with torch.no_grad():
        actions, _, _ = model.act(
            local_obs=local_obs,
            joint_obs=joint_obs,
            agent_ids=agent_ids,
            deterministic=deterministic,
        )

    action_values = actions.cpu().numpy()
    return {
        agent: int(action_values[agent_idx])
        for agent_idx, agent in enumerate(metadata.agents)
        if agent in observations
    }


def evaluate_team_policy(
    model: SharedCentralizedActorCritic,
    metadata: CheckpointMetadata,
    seed: int,
    n_eval_episodes: int,
    max_cycles: int,
    terminate_on_success: bool,
    device: torch.device,
    deterministic: bool = True,
):
    """Evaluate by mean per-agent episode return, matching the baseline."""

    env = make_parallel_env(
        max_cycles=max_cycles,
        terminate_on_success=terminate_on_success,
        render_mode=None,
        seed=seed,
    )
    returns = []
    episode_lengths = []

    try:
        for episode_idx in range(n_eval_episodes):
            observations, _ = env.reset(seed=seed + episode_idx)
            episode_return = 0.0
            episode_length = 0

            while env.agents:
                actions = predict_parallel_actions(
                    model=model,
                    metadata=metadata,
                    observations=observations,
                    device=device,
                    deterministic=deterministic,
                )
                observations, rewards, terminations, truncations, _ = env.step(actions)
                episode_return += mean_agent_reward(rewards)
                episode_length += 1

                if not env.agents:
                    break
                if terminations and all(terminations.values()):
                    break
                if truncations and all(truncations.values()):
                    break

            returns.append(episode_return)
            episode_lengths.append(episode_length)
    finally:
        env.close()

    return (
        np.asarray(returns, dtype=np.float64),
        np.asarray(episode_lengths, dtype=np.int32),
    )
