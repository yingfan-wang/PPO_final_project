"""Shared helpers for the Simple Spread PPO baseline scripts."""

from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import supersuit as ss
from mpe2 import simple_spread_v3
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv, VecMonitor

BASE_DIR = Path(__file__).resolve().parent
RUNS_DIR = BASE_DIR / "runs"
N_AGENTS = 3
LOCAL_RATIO = 0.5
DEFAULT_MAX_CYCLES = 100
DEFAULT_NUM_VEC_ENVS = 4


class SB3VectorEnvAdapter(VecEnv):
    """SB3-compatible adapter for a Gymnasium/SuperSuit vector environment.

    SuperSuit's built-in SB3 wrapper does not forward reset-time seeds to the
    underlying ConcatVecEnv in the version installed for this project. This
    adapter keeps the wrapper small and explicit so PPO sees a standard SB3
    VecEnv while initial seeding remains reproducible.
    """

    def __init__(self, venv):
        self.venv = venv
        super().__init__(
            num_envs=venv.num_envs,
            observation_space=venv.observation_space,
            action_space=venv.action_space,
        )
        self.reset_infos = [{} for _ in range(self.num_envs)]
        self._saved_actions = None

    def _slot_to_env(self):
        if hasattr(self.venv, "vec_envs"):
            slot_envs = []
            for sub_env in self.venv.vec_envs:
                sub_env_num = getattr(sub_env, "num_envs", 1)
                slot_envs.extend([sub_env] * sub_env_num)
            if len(slot_envs) == self.num_envs:
                return slot_envs
        return [self.venv for _ in range(self.num_envs)]

    def reset(self):
        base_seed = next((seed for seed in self._seeds if seed is not None), None)
        options = next((option for option in self._options if option), None)
        observations, infos = self.venv.reset(seed=base_seed, options=options)
        self.reset_infos = infos
        self._reset_seeds()
        self._reset_options()
        return observations

    def step_async(self, actions: np.ndarray) -> None:
        self._saved_actions = actions
        self.venv.step_async(actions)

    def step_wait(self):
        observations, rewards, terminations, truncations, infos = self.venv.step_wait()
        new_infos = list(infos[:])
        for idx, info in enumerate(new_infos):
            info = dict(info)
            info["TimeLimit.truncated"] = bool(truncations[idx] and not terminations[idx])
            new_infos[idx] = info
        dones = np.logical_or(terminations, truncations)
        return observations, rewards, dones, new_infos

    def close(self) -> None:
        self.venv.close()

    def get_attr(self, attr_name: str, indices=None) -> list[Any]:
        slot_envs = self._slot_to_env()
        values = []
        for env_idx in self._get_indices(indices):
            env = slot_envs[env_idx]
            if hasattr(env, attr_name):
                values.append(getattr(env, attr_name))
            elif hasattr(self.venv, attr_name):
                values.append(getattr(self.venv, attr_name))
            else:
                raise AttributeError(attr_name)
        return values

    def set_attr(self, attr_name: str, value: Any, indices=None) -> None:
        slot_envs = self._slot_to_env()
        for env_idx in self._get_indices(indices):
            setattr(slot_envs[env_idx], attr_name, value)

    def env_method(self, method_name: str, *method_args, indices=None, **method_kwargs):
        slot_envs = self._slot_to_env()
        results = []
        for env_idx in self._get_indices(indices):
            method = getattr(slot_envs[env_idx], method_name)
            results.append(method(*method_args, **method_kwargs))
        return results

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False for _ in self._get_indices(indices)]

    def get_images(self):
        if getattr(self.venv, "render_mode", None) != "rgb_array":
            return [None for _ in range(self.num_envs)]
        frame = self.venv.render()
        return [frame for _ in range(self.num_envs)]


def resolve_model_path(model_path: Union[str, Path]) -> Path:
    """Accept either an explicit .zip path or the SB3 save stem."""

    raw_path = Path(model_path).expanduser()
    candidate_paths = [raw_path]

    if raw_path.suffix != ".zip":
        candidate_paths.append(raw_path.with_suffix(".zip"))

    for candidate_path in candidate_paths:
        if candidate_path.exists():
            return candidate_path.resolve()

    checked_paths = ", ".join(str(path) for path in candidate_paths)
    raise FileNotFoundError(f"Could not find a PPO checkpoint at: {checked_paths}")


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
    """Create one unwrapped PettingZoo parallel environment."""

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


def make_vec_env(
    seed: int,
    max_cycles: int,
    num_vec_envs: int,
    terminate_on_success: bool,
):
    """Wrap Simple Spread into the SB3 VecEnv interface used by PPO."""

    env = make_parallel_env(
        max_cycles=max_cycles,
        terminate_on_success=terminate_on_success,
        render_mode=None,
        seed=seed,
    )
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(
        env,
        num_vec_envs,
        num_cpus=1,
        base_class="gymnasium",
    )
    env = SB3VectorEnvAdapter(env)
    env = VecMonitor(env)

    # Seed after vectorization so the first SB3-triggered reset gives each
    # parallel environment copy a deterministic, offset seed.
    env.seed(seed)
    return env


def mean_agent_reward(rewards) -> float:
    if not rewards:
        return 0.0
    return float(sum(rewards.values())) / N_AGENTS


def predict_parallel_actions(model: PPO, observations, deterministic: bool = True):
    actions = {}
    for agent, agent_obs in observations.items():
        action, _ = model.predict(agent_obs, deterministic=deterministic)
        actions[agent] = action
    return actions


def evaluate_team_policy(
    model: PPO,
    seed: int,
    n_eval_episodes: int,
    max_cycles: int,
    terminate_on_success: bool,
    deterministic: bool = True,
):
    """Evaluate a shared policy using the mean reward across all agents."""

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
                    model,
                    observations,
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


def safe_std(values) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.size <= 1:
        return 0.0
    return float(values.std(ddof=1))
