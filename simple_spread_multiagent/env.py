from __future__ import annotations

import numpy as np

from simple_spread_baseline.config import N_AGENTS, OBS_DIM, STATE_DIM
from simple_spread_baseline.env import EpisodeTracker, SimpleSpreadEnv, SimpleSpreadVectorEnv

LANDMARK_REL_SLICE = slice(4, 10)
OTHER_AGENT_REL_SLICE = slice(10, 14)
ACTOR_INPUT_DIM = N_AGENTS * (
    (LANDMARK_REL_SLICE.stop - LANDMARK_REL_SLICE.start)
    + (OTHER_AGENT_REL_SLICE.stop - OTHER_AGENT_REL_SLICE.start)
)


def build_joint_observations(local_observations: np.ndarray) -> np.ndarray:
    return local_observations.reshape(local_observations.shape[0], OBS_DIM * N_AGENTS).astype(
        np.float32
    )


def build_actor_inputs(
    local_observations: np.ndarray, use_agent_id: bool | None = None
) -> np.ndarray:
    del use_agent_id
    landmark_features = local_observations[..., LANDMARK_REL_SLICE]
    other_agent_features = local_observations[..., OTHER_AGENT_REL_SLICE]
    return np.concatenate(
        [
            landmark_features.reshape(local_observations.shape[0], -1),
            other_agent_features.reshape(local_observations.shape[0], -1),
        ],
        axis=-1,
    ).astype(np.float32)


def build_critic_inputs(joint_states: np.ndarray) -> np.ndarray:
    if joint_states.ndim == 3:
        return build_joint_observations(joint_states)
    return joint_states.astype(np.float32)


__all__ = [
    "EpisodeTracker",
    "OBS_DIM",
    "STATE_DIM",
    "ACTOR_INPUT_DIM",
    "SimpleSpreadEnv",
    "SimpleSpreadVectorEnv",
    "build_joint_observations",
    "build_actor_inputs",
    "build_critic_inputs",
]
