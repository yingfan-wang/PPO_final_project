from __future__ import annotations

import numpy as np

from simple_spread_baseline.config import N_AGENTS, OBS_DIM, STATE_DIM
from simple_spread_baseline.env import EpisodeTracker, SimpleSpreadEnv, SimpleSpreadVectorEnv


def build_actor_inputs(
    local_observations: np.ndarray,
    *,
    use_agent_id: bool,
) -> np.ndarray:
    local_observations = np.asarray(local_observations, dtype=np.float32)
    if not use_agent_id:
        return local_observations.astype(np.float32)

    if local_observations.ndim == 2:
        return np.concatenate(
            [local_observations, np.eye(N_AGENTS, dtype=np.float32)],
            axis=-1,
        ).astype(np.float32)

    if local_observations.ndim != 3:
        raise ValueError(
            "Expected local observations with shape (agents, obs_dim) or "
            "(batch, agents, obs_dim)."
        )

    batch_size = local_observations.shape[0]
    agent_ids = np.broadcast_to(
        np.eye(N_AGENTS, dtype=np.float32),
        (batch_size, N_AGENTS, N_AGENTS),
    )
    return np.concatenate([local_observations, agent_ids], axis=-1).astype(np.float32)


def build_critic_inputs(
    joint_states: np.ndarray,
    *,
    use_agent_id: bool,
) -> np.ndarray:
    joint_states = np.asarray(joint_states, dtype=np.float32)
    single_env = joint_states.ndim == 1
    if single_env:
        joint_states = joint_states[None, :]
    if joint_states.ndim != 2:
        raise ValueError(
            "Expected joint states with shape (state_dim,) or (batch, state_dim)."
        )

    batch_size = joint_states.shape[0]
    repeated_states = np.repeat(joint_states[:, None, :], N_AGENTS, axis=1)
    if use_agent_id:
        agent_ids = np.broadcast_to(
            np.eye(N_AGENTS, dtype=np.float32),
            (batch_size, N_AGENTS, N_AGENTS),
        )
        repeated_states = np.concatenate([repeated_states, agent_ids], axis=-1)

    return repeated_states[0] if single_env else repeated_states.astype(np.float32)


__all__ = [
    "EpisodeTracker",
    "OBS_DIM",
    "STATE_DIM",
    "SimpleSpreadEnv",
    "SimpleSpreadVectorEnv",
    "build_actor_inputs",
    "build_critic_inputs",
]
