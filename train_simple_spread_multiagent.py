from pathlib import Path
import supersuit as ss
import numpy as np
from pettingzoo.mpe import simple_spread_v3
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor, DummyVecEnv
from pettingzoo.utils import ParallelEnv

import gymnasium as gym
from gymnasium import spaces


class SpreadShapingWrapper(ParallelEnv):
    metadata = {}

    def __init__(self, env):
        self.env = env
        obs, _ = self.env.reset()
        self.agents = list(self.env.agents)
        self.possible_agents = list(self.env.possible_agents)
        self.observation_spaces = env.observation_spaces
        self.action_spaces = env.action_spaces
        self.render_mode = getattr(env, "render_mode", None)

    def reset(self, *args, **kwargs):
        obs, infos = self.env.reset(*args, **kwargs)
        self.agents = list(self.env.agents)
        return obs, infos

    def step(self, actions):
        # Only pass actions for agents still in the env
        active_actions = {a: actions[a] for a in self.env.agents if a in actions}
        obs, rewards, terminations, truncations, infos = self.env.step(active_actions)
        self.agents = list(self.env.agents)

        shaped_rewards = {}
        for agent in self.env.agents:
            agent_obs = obs[agent]
            landmark_positions = agent_obs[4:10]
            distances = [
                np.sqrt(landmark_positions[2*i]**2 + landmark_positions[2*i+1]**2)
                for i in range(3)
            ]
            nearest_dist = min(distances)

            other_agent_obs = agent_obs[10:14]
            agent_penalty = sum(
                -1.0 for i in range(2)
                if np.sqrt(other_agent_obs[2*i]**2 + other_agent_obs[2*i+1]**2) < 0.3
            )

            shaped_rewards[agent] = (
                rewards[agent]
                + 0.1 * np.mean(distances)
                + 3.0 * np.exp(-10.0 * nearest_dist)
                + agent_penalty
            )

        return obs, shaped_rewards, terminations, truncations, infos

    def observation_space(self, agent):
        return self.env.observation_space(agent)

    def action_space(self, agent):
        return self.env.action_space(agent)

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()


class SingleAgentWrapper:
    """Wraps the multi-agent env to expose only ONE agent's obs/actions to SB3."""

    def __init__(self, agent_id: str, render_mode=None):
        self.agent_id = agent_id
        self.render_mode = render_mode
        self._make_env(render_mode)

    def _make_env(self, render_mode=None):
        raw = simple_spread_v3.parallel_env(
            N=3, max_cycles=25, continuous_actions=True, render_mode=render_mode
        )
        self.wrapped = SpreadShapingWrapper(raw)
        obs, _ = self.wrapped.reset()
        self.observation_space = self.wrapped.observation_space(self.agent_id)
        self.action_space = self.wrapped.action_space(self.agent_id)
        self._last_obs = obs          # cache all agents' obs
        self._other_actions = {}      # will be filled by the coordinator
        self._done = False

    def set_other_actions(self, actions: dict):
        """Called by the training loop to inject the other agents' actions."""
        self._other_actions = actions

    def reset(self):
        obs, _ = self.wrapped.reset()
        self._last_obs = obs
        self._done = False
        return obs[self.agent_id], {}

    def step(self, action):
        all_actions = {**self._other_actions, self.agent_id: action}
        obs, rewards, terminations, truncations, infos = self.wrapped.step(all_actions)
        self._last_obs = obs
        done = terminations.get(self.agent_id, False) or truncations.get(self.agent_id, False)
        self._done = done
        return (
            obs.get(self.agent_id, np.zeros(self.observation_space.shape)),
            rewards.get(self.agent_id, 0.0),
            done,
            False,
            infos.get(self.agent_id, {}),
        )

    def render(self):
        return self.wrapped.render()

    def close(self):
        self.wrapped.close()

class CoordinatedMultiAgentEnv(gym.Env):
    def __init__(self, agent_id: str, other_models: dict, render_mode=None):
        super().__init__()
        self.agent_id = agent_id
        self.other_models = other_models
        self.render_mode = render_mode

        # Create a fresh raw parallel env (no wrapper)
        self.penv = simple_spread_v3.parallel_env(
            N=3, max_cycles=25, continuous_actions=True
        )
        obs, _ = self.penv.reset()
        self.observation_space = self.penv.observation_space(agent_id)
        self.action_space = self.penv.action_space(agent_id)
        self._last_full_obs = obs
        self.all_agent_ids = list(self.penv.possible_agents)

    def _shape_reward(self, agent, obs, base_reward):
        agent_obs = obs[agent]
        landmark_positions = agent_obs[4:10]
        distances = [
            np.sqrt(landmark_positions[2*i]**2 + landmark_positions[2*i+1]**2)
            for i in range(3)
        ]
        nearest_dist = min(distances)
        other_agent_obs = agent_obs[10:14]
        agent_penalty = sum(
            -1.0 for i in range(2)
            if np.sqrt(other_agent_obs[2*i]**2 + other_agent_obs[2*i+1]**2) < 0.3
        )
        return (
            base_reward
            + 0.1 * np.mean(distances)
            + 3.0 * np.exp(-10.0 * nearest_dist)
            + agent_penalty
        )

    def reset(self, seed=None, options=None):
        obs, infos = self.penv.reset()
        self._last_full_obs = obs
        return obs[self.agent_id], infos.get(self.agent_id, {})

    def step(self, action):
        all_actions = {}
        for a in self.penv.agents:
            if a == self.agent_id:
                all_actions[a] = action
            elif a in self.other_models and a in self._last_full_obs:
                other_obs = self._last_full_obs[a][np.newaxis]
                other_action, _ = self.other_models[a].predict(other_obs, deterministic=False)
                all_actions[a] = other_action[0]
            else:
                all_actions[a] = self.penv.action_space(a).sample()

        obs, rewards, terminations, truncations, infos = self.penv.step(all_actions)
        self._last_full_obs = obs

        done = terminations.get(self.agent_id, False) or truncations.get(self.agent_id, False)

        # Only shape reward if this agent still has an observation (not terminated)
        if self.agent_id in obs:
            reward = self._shape_reward(self.agent_id, obs, rewards.get(self.agent_id, 0.0))
        else:
            reward = rewards.get(self.agent_id, 0.0)

        return (
            obs.get(self.agent_id, np.zeros(self.observation_space.shape)),
            reward,
            done,
            False,
            infos.get(self.agent_id, {}),
        )

    def render(self):
        return self.penv.render()

    def close(self):
        self.penv.close()


def make_single_agent_vecenv(agent_id, other_models):
    def _make():
        return CoordinatedMultiAgentEnv(agent_id, other_models)
    vec = DummyVecEnv([_make])
    vec = VecMonitor(vec)
    return vec


def train_agents(total_timesteps=500_000, n_rounds=3):
    run_dir = Path("runs") / "multiagent_spread"
    run_dir.mkdir(parents=True, exist_ok=True)

    agent_ids = [f"agent_{i}" for i in range(3)]
    models: dict[str, PPO | None] = {a: None for a in agent_ids}

    for round_idx in range(n_rounds):
        print(f"\n{'='*50}")
        print(f"  Training round {round_idx + 1} / {n_rounds}")
        print(f"{'='*50}")

        for agent_id in agent_ids:
            print(f"\n--- Training {agent_id} ---")

            other_models = {
                a: m for a, m in models.items()
                if a != agent_id and m is not None
            }

            train_env = make_single_agent_vecenv(agent_id, other_models)

            if models[agent_id] is None:
                model = PPO(
                    policy="MlpPolicy",
                    env=train_env,
                    learning_rate=3e-4,
                    n_steps=512,
                    batch_size=256,
                    n_epochs=10,
                    gamma=0.99,
                    gae_lambda=0.95,
                    clip_range=0.2,
                    ent_coef=0.01,
                    vf_coef=0.5,
                    max_grad_norm=0.5,
                    verbose=1,
                    tensorboard_log=str(run_dir / "tb" / agent_id),
                )
            else:
                model = models[agent_id]
                model.set_env(train_env)

            model.learn(
                total_timesteps=total_timesteps // (n_rounds * len(agent_ids)),
                reset_num_timesteps=(round_idx == 0),
            )

            save_path = str(run_dir / f"{agent_id}_round{round_idx}")
            model.save(save_path)
            models[agent_id] = model
            train_env.close()

    for agent_id, model in models.items():
        model.save(str(run_dir / f"{agent_id}_final"))
        print(f"Final model saved: {agent_id}")

    return models


if __name__ == "__main__":
    train_agents(total_timesteps=500_000, n_rounds=3)