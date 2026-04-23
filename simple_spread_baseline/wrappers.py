from __future__ import annotations

import numpy as np


def flatten_time_env_agent(array: np.ndarray) -> np.ndarray:
    """Collapse [time, env, agent, ...] into [batch, ...]."""

    return array.reshape((-1,) + array.shape[3:])


def flatten_time_env_agent_scalar(array: np.ndarray) -> np.ndarray:
    """Collapse [time, env, agent] into [batch]."""

    return array.reshape(-1)


def repeat_agent_axis(array: np.ndarray, num_agents: int) -> np.ndarray:
    """Repeat [time, env, feat] arrays across the agent axis."""

    return np.repeat(array[:, :, None, :], repeats=num_agents, axis=2)

