from __future__ import annotations

from dataclasses import dataclass, field
import importlib
import os
from typing import Any

import numpy as np

try:
    agent_selector_module = importlib.import_module("pettingzoo.utils.agent_selector")
    if not hasattr(agent_selector_module, "AgentSelector") and hasattr(
        agent_selector_module, "agent_selector"
    ):
        agent_selector_module.AgentSelector = agent_selector_module.agent_selector
except ModuleNotFoundError:
    pass

from mpe2 import simple_spread_v3

from simple_spread_baseline.config import (
    ACTION_DIM,
    COVERAGE_THRESHOLD,
    DEFAULT_CONTINUOUS_ACTIONS,
    DEFAULT_LOCAL_RATIO,
    DEFAULT_MAX_CYCLES,
    N_AGENTS,
    OBS_DIM,
    STATE_DIM,
    SUCCESS_WINDOW,
)


@dataclass
class EpisodeTracker:
    n_agents: int = N_AGENTS
    success_window: int = SUCCESS_WINDOW
    agent_returns: np.ndarray = field(
        default_factory=lambda: np.zeros(N_AGENTS, dtype=np.float64)
    )
    step_count: int = 0
    collision_count: int = 0
    occupied_landmarks: list[int] = field(default_factory=list)
    min_distance_sums: list[float] = field(default_factory=list)

    def reset(self) -> None:
        self.agent_returns = np.zeros(self.n_agents, dtype=np.float64)
        self.step_count = 0
        self.collision_count = 0
        self.occupied_landmarks = []
        self.min_distance_sums = []

    def update(self, rewards: np.ndarray, step_metrics: dict[str, float]) -> None:
        self.agent_returns += rewards.astype(np.float64)
        self.step_count += 1
        self.collision_count += int(step_metrics["collision_pairs"])
        self.occupied_landmarks.append(int(step_metrics["occupied_landmarks"]))
        self.min_distance_sums.append(float(step_metrics["sum_min_dists"]))

    def summary(self) -> dict[str, float]:
        occupied = np.asarray(self.occupied_landmarks, dtype=np.float64)
        min_dists = np.asarray(self.min_distance_sums, dtype=np.float64)
        success_slice = occupied[-self.success_window :] if occupied.size else occupied
        return {
            "episode_return_mean": float(self.agent_returns.mean()),
            "episode_return_sum": float(self.agent_returns.sum()),
            "episode_length": int(self.step_count),
            "collision_count": int(self.collision_count),
            "avg_landmarks_covered": float(occupied.mean()) if occupied.size else 0.0,
            "full_coverage_near_end": float(
                np.any(success_slice >= self.n_agents)
            )
            if success_slice.size
            else 0.0,
            "full_coverage_final_step": float(occupied[-1] >= self.n_agents)
            if occupied.size
            else 0.0,
            "mean_sum_min_dists": float(min_dists.mean()) if min_dists.size else 0.0,
        }


class SimpleSpreadEnv:
    """Thin fixed-order wrapper around `mpe2.simple_spread_v3.parallel_env`."""

    def __init__(
        self,
        local_ratio: float = DEFAULT_LOCAL_RATIO,
        max_cycles: int = DEFAULT_MAX_CYCLES,
        continuous_actions: bool = DEFAULT_CONTINUOUS_ACTIONS,
        terminate_on_success: bool = False,
        curriculum: bool = False,
        render_mode: str | None = None,
    ) -> None:
        self.local_ratio = local_ratio
        self.max_cycles = max_cycles
        self.continuous_actions = continuous_actions
        self.terminate_on_success = terminate_on_success
        self.curriculum = curriculum
        self.render_mode = render_mode
        if render_mode != "human":
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        self.raw_env = simple_spread_v3.parallel_env(
            N=N_AGENTS,
            local_ratio=local_ratio,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            curriculum=curriculum,
            terminate_on_success=terminate_on_success,
            render_mode=render_mode,
        )
        self.agent_order = list(self.raw_env.possible_agents)
        self.obs_dim = OBS_DIM
        self.state_dim = STATE_DIM
        self.action_dim = ACTION_DIM if continuous_actions else 1

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        observations, infos = self.raw_env.reset(seed=seed)
        return self._ordered_observations(observations), self.state(), dict(infos)

    def step(
        self, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool, dict[str, Any]]:
        action_dict = {
            agent: self._coerce_action(actions[idx])
            for idx, agent in enumerate(self.agent_order)
        }
        observations, rewards, terminations, truncations, infos = self.raw_env.step(
            action_dict
        )
        done = (not self.raw_env.agents) or all(
            bool(terminations.get(agent, False) or truncations.get(agent, False))
            for agent in self.agent_order
        )
        reward_array = np.asarray(
            [float(rewards.get(agent, 0.0)) for agent in self.agent_order],
            dtype=np.float32,
        )
        info: dict[str, Any] = {
            "step_metrics": self.compute_step_metrics(),
            "terminations": {agent: bool(terminations.get(agent, False)) for agent in self.agent_order},
            "truncations": {agent: bool(truncations.get(agent, False)) for agent in self.agent_order},
            "raw_infos": dict(infos),
        }
        if done:
            next_obs = np.zeros((N_AGENTS, self.obs_dim), dtype=np.float32)
            next_state = np.zeros(self.state_dim, dtype=np.float32)
        else:
            next_obs = self._ordered_observations(observations)
            next_state = self.state()
        return next_obs, next_state, reward_array, done, info

    def state(self) -> np.ndarray:
        return np.asarray(self.raw_env.state(), dtype=np.float32)

    def render(self):
        return self.raw_env.render()

    def close(self) -> None:
        self.raw_env.close()

    def set_curriculum_stage(self, stage: int) -> None:
        if hasattr(self.raw_env.unwrapped, "set_curriculum_stage"):
            self.raw_env.unwrapped.set_curriculum_stage(stage)

    def compute_step_metrics(self) -> dict[str, float]:
        world = self.raw_env.unwrapped.world
        scenario = self.raw_env.unwrapped.scenario

        landmark_min_dists = []
        for landmark in world.landmarks:
            distances = [
                float(np.linalg.norm(agent.state.p_pos - landmark.state.p_pos))
                for agent in world.agents
            ]
            landmark_min_dists.append(min(distances))

        collision_pairs = 0
        for idx, first_agent in enumerate(world.agents):
            for second_agent in world.agents[idx + 1 :]:
                if scenario.is_collision(first_agent, second_agent):
                    collision_pairs += 1

        min_dists_array = np.asarray(landmark_min_dists, dtype=np.float64)
        return {
            "collision_pairs": float(collision_pairs),
            "occupied_landmarks": float(np.sum(min_dists_array < COVERAGE_THRESHOLD)),
            "sum_min_dists": float(min_dists_array.sum()),
        }

    def _ordered_observations(self, observations: dict[str, np.ndarray]) -> np.ndarray:
        return np.stack(
            [
                np.asarray(observations.get(agent, np.zeros(self.obs_dim)), dtype=np.float32)
                for agent in self.agent_order
            ],
            axis=0,
        )

    def _coerce_action(self, action: np.ndarray | float | int) -> np.ndarray | int:
        if self.continuous_actions:
            return np.asarray(action, dtype=np.float32)
        action_array = np.asarray(action, dtype=np.int64).reshape(-1)
        return int(action_array[0]) if action_array.size else int(action)


class SimpleSpreadVectorEnv:
    """Synchronous vector wrapper that keeps per-env episode bookkeeping."""

    def __init__(
        self,
        num_envs: int,
        local_ratio: float = DEFAULT_LOCAL_RATIO,
        max_cycles: int = DEFAULT_MAX_CYCLES,
        continuous_actions: bool = DEFAULT_CONTINUOUS_ACTIONS,
        terminate_on_success: bool = False,
        curriculum: bool = False,
        render_mode: str | None = None,
    ) -> None:
        self.num_envs = num_envs
        self.local_ratio = local_ratio
        self.max_cycles = max_cycles
        self.continuous_actions = continuous_actions
        self.terminate_on_success = terminate_on_success
        self.curriculum = curriculum
        self.render_mode = render_mode
        self.envs = [
            SimpleSpreadEnv(
                local_ratio=local_ratio,
                max_cycles=max_cycles,
                continuous_actions=continuous_actions,
                terminate_on_success=terminate_on_success,
                curriculum=curriculum,
                render_mode=render_mode,
            )
            for _ in range(num_envs)
        ]
        self.trackers = [EpisodeTracker() for _ in range(num_envs)]
        self.base_seed = 0
        self.reset_counts = [0 for _ in range(num_envs)]
        self.current_obs: np.ndarray | None = None
        self.current_state: np.ndarray | None = None

    @property
    def obs_dim(self) -> int:
        return self.envs[0].obs_dim

    @property
    def state_dim(self) -> int:
        return self.envs[0].state_dim

    @property
    def action_dim(self) -> int:
        return self.envs[0].action_dim

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        if seed is not None:
            self.base_seed = seed
            self.reset_counts = [0 for _ in range(self.num_envs)]
        observations = []
        states = []
        for env_idx, env in enumerate(self.envs):
            self.trackers[env_idx].reset()
            obs, state, _ = env.reset(seed=self._next_seed(env_idx))
            observations.append(obs)
            states.append(state)
        self.current_obs = np.stack(observations, axis=0)
        self.current_state = np.stack(states, axis=0)
        return self.current_obs.copy(), self.current_state.copy()

    def step(
        self, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        next_observations = []
        next_states = []
        reward_batch = []
        dones = []
        infos: list[dict[str, Any]] = []

        for env_idx, env in enumerate(self.envs):
            obs, state, rewards, done, info = env.step(actions[env_idx])
            self.trackers[env_idx].update(rewards, info["step_metrics"])
            if done:
                info["episode"] = self.trackers[env_idx].summary()
                self.trackers[env_idx].reset()
                obs, state, _ = env.reset(seed=self._next_seed(env_idx))
            next_observations.append(obs)
            next_states.append(state)
            reward_batch.append(rewards)
            dones.append(done)
            infos.append(info)

        self.current_obs = np.stack(next_observations, axis=0)
        self.current_state = np.stack(next_states, axis=0)
        return (
            self.current_obs.copy(),
            self.current_state.copy(),
            np.stack(reward_batch, axis=0),
            np.asarray(dones, dtype=np.float32),
            infos,
        )

    def close(self) -> None:
        for env in self.envs:
            env.close()

    def set_curriculum_stage(self, stage: int) -> None:
        for env in self.envs:
            env.set_curriculum_stage(stage)

    def _next_seed(self, env_idx: int) -> int:
        episode_seed = self.base_seed + env_idx * 10_000 + self.reset_counts[env_idx]
        self.reset_counts[env_idx] += 1
        return episode_seed
