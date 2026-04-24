from pathlib import Path
import numpy as np
from pettingzoo.mpe import simple_spread_v3
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor, DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
import gymnasium as gym
from gymnasium import spaces
import shutil
from shaped_reward import TIMESTEPS
from shaped_reward import shape_reward
from shaped_reward import HYPERPARAMS

# ---------------------------------------------------------------------------
# Reward shaping (same as multiagent version)
# ---------------------------------------------------------------------------

# def shape_reward(agent_obs, base_reward):
#     landmark_positions = agent_obs[4:10]
#     distances = [
#         np.sqrt(landmark_positions[2*i]**2 + landmark_positions[2*i+1]**2)
#         for i in range(3)
#     ]
#     nearest_dist = min(distances)

#     other_agent_obs = agent_obs[10:14]
#     collision_penalty = 0.0
#     for i in range(2):
#         dx = other_agent_obs[2*i]
#         dy = other_agent_obs[2*i + 1]
#         dist = np.sqrt(dx**2 + dy**2)
#         if dist < 0.5:
#             collision_penalty -= 0.3 * (0.5 - dist) / 0.5

#     return (
#         base_reward
#         + 3.0 * np.exp(-10.0 * nearest_dist)
#         - 0.5 * nearest_dist
#         + collision_penalty
#     )


# ---------------------------------------------------------------------------
# Joint observation wrapper
# Each agent sees [own_obs | other_agent_1_obs | other_agent_2_obs]
# But still outputs only its own action and gets its own reward
# ---------------------------------------------------------------------------

class JointObsAgentEnv(gym.Env):
    """
    Same as SingleAgentEnv but observation is concatenation of ALL agents' obs.
    own obs (18) + agent_1 obs (18) + agent_2 obs (18) = 54 values total.
    Action and reward are still local to this agent only.
    """

    def __init__(self, agent_id: str, all_agent_ids: list, frozen_models: dict):
        super().__init__()
        self.agent_id = agent_id
        self.all_agent_ids = all_agent_ids  # ordered list so concatenation is consistent
        self.frozen_models = frozen_models

        self.penv = simple_spread_v3.parallel_env(
            N=3, max_cycles=25, continuous_actions=True
        )
        obs, _ = self.penv.reset()

        # Single agent obs size
        single_obs_space = self.penv.observation_space(agent_id)
        single_obs_size  = single_obs_space.shape[0]
        n_agents         = len(all_agent_ids)

        # Joint obs = all agents stacked → n_agents * single_obs_size
        joint_obs_size = n_agents * single_obs_size
        self.observation_space = spaces.Box(
            low  = -np.inf,
            high =  np.inf,
            shape = (joint_obs_size,),
            dtype = np.float32,
        )
        self.action_space = self.penv.action_space(agent_id)
        self._last_obs = obs

    def _make_joint_obs(self, obs: dict) -> np.ndarray:
        """
        Concatenate observations in a fixed order:
        [own_obs, other_agent_1_obs, other_agent_2_obs]
        Own obs always goes first so the agent knows which part is 'itself'.
        """
        parts = [obs[self.agent_id]]  # own obs first
        for a in self.all_agent_ids:
            if a != self.agent_id:
                parts.append(obs[a])
        return np.concatenate(parts, dtype=np.float32)

    def reset(self, seed=None, options=None):
        obs, infos = self.penv.reset()
        self._last_obs = obs
        return self._make_joint_obs(obs), infos.get(self.agent_id, {})

    def _make_joint_obs_for(self, obs: dict, for_agent: str) -> np.ndarray:
        """Build joint obs from a specific agent's perspective — own obs always first."""
        parts = [obs[for_agent]]
        for a in self.all_agent_ids:
            if a != for_agent:
                parts.append(obs[a])
        return np.concatenate(parts, dtype=np.float32)

    def step(self, action):
        all_actions = {}
        for a in self.penv.agents:
            if a == self.agent_id:
                all_actions[a] = action
            elif a in self.frozen_models and self.frozen_models[a] is not None:
                # build joint obs FROM THAT AGENT'S perspective, not self's
                frozen_joint = self._make_joint_obs_for(self._last_obs, a)[np.newaxis]
                frozen_action, _ = self.frozen_models[a].predict(frozen_joint, deterministic=False)
                all_actions[a] = frozen_action[0]
            else:
                all_actions[a] = self.penv.action_space(a).sample()

        obs, rewards, terminations, truncations, infos = self.penv.step(all_actions)
        self._last_obs = obs

        done = terminations.get(self.agent_id, False) or truncations.get(self.agent_id, False)

        # Still only shapes and returns THIS agent's reward — credit assignment unsolved
        if self.agent_id in obs:
            reward = shape_reward(obs[self.agent_id], rewards.get(self.agent_id, 0.0))
            joint_obs = self._make_joint_obs(obs)
        else:
            reward    = rewards.get(self.agent_id, 0.0)
            joint_obs = np.zeros(self.observation_space.shape, dtype=np.float32)

        return joint_obs, reward, done, False, infos.get(self.agent_id, {})

    def render(self):
        return self.penv.render()

    def close(self):
        self.penv.close()


def make_vecenv(agent_id, all_agent_ids, frozen_models):
    def _make():
        return JointObsAgentEnv(agent_id, all_agent_ids, frozen_models)
    return VecMonitor(DummyVecEnv([_make]))


# ---------------------------------------------------------------------------
# Training — same round-robin as multiagent, just with joint obs
# ---------------------------------------------------------------------------

def train_agents(total_timesteps=1_000_000, n_rounds=3):
    run_dir = Path("runs") / "joint_obs_spread"
    run_dir.mkdir(parents=True, exist_ok=True)

    agent_ids = [f"agent_{i}" for i in range(3)]
    models: dict[str, PPO | None] = {a: None for a in agent_ids}

    timesteps_per_agent = total_timesteps // (n_rounds * len(agent_ids))

    for round_idx in range(n_rounds):
        print(f"\n{'='*50}")
        print(f"  Round {round_idx + 1} / {n_rounds}  ({timesteps_per_agent} steps/agent)")
        print(f"{'='*50}")

        for agent_id in agent_ids:
            print(f"\n--- Training {agent_id} (joint obs, others frozen) ---")

            frozen = {a: m for a, m in models.items() if a != agent_id}

            train_env = make_vecenv(agent_id, agent_ids, frozen)
            eval_env  = make_vecenv(agent_id, agent_ids, frozen)

            eval_log_dir   = run_dir / agent_id / "eval_logs"
            best_model_dir = run_dir / agent_id / "best_model"
            eval_log_dir.mkdir(parents=True, exist_ok=True)
            best_model_dir.mkdir(parents=True, exist_ok=True)

            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=str(best_model_dir),
                log_path=str(eval_log_dir),
                eval_freq=5_000,
                n_eval_episodes=10,
                deterministic=True,
                render=False,
            )

            if models[agent_id] is None:
                # model = PPO(
                #     policy="MlpPolicy",
                #     env=train_env,
                #     learning_rate=3e-4,
                #     n_steps=512,
                #     batch_size=256,
                #     n_epochs=10,
                #     gamma=0.99,
                #     gae_lambda=0.95,
                #     clip_range=0.2,
                #     ent_coef=0.01,
                #     vf_coef=0.5,
                #     max_grad_norm=0.5,
                #     verbose=1,
                #     tensorboard_log=str(run_dir / "tb" / agent_id),
                # )
                model = PPO(
                    policy="MlpPolicy",
                    env=train_env,
                    learning_rate=HYPERPARAMS["learning_rate"],
                    n_steps=HYPERPARAMS["n_steps"],
                    batch_size=HYPERPARAMS["batch_size"],
                    n_epochs=HYPERPARAMS["n_epochs"],
                    gamma=HYPERPARAMS["gamma"],
                    gae_lambda=HYPERPARAMS["gae_lambda"],
                    clip_range=HYPERPARAMS["clip_range"],
                    ent_coef=HYPERPARAMS["ent_coef"],
                    vf_coef=HYPERPARAMS["vf_coef"],
                    max_grad_norm=HYPERPARAMS["max_grad_norm"],
                )
            else:
                model = models[agent_id]
                model.set_env(train_env)

            model.learn(
                total_timesteps=timesteps_per_agent,
                reset_num_timesteps=False,
                callback=eval_callback,
            )

            model.save(str(run_dir / f"{agent_id}_round{round_idx}"))
            models[agent_id] = model
            train_env.close()
            eval_env.close()

    for agent_id, model in models.items():
        model.save(str(run_dir / f"{agent_id}_final"))
        print(f"Saved: {run_dir}/{agent_id}_final")

    return models


if __name__ == "__main__":
    # train_agents(total_timesteps=1_000_000, n_rounds=3)
    train_agents(total_timesteps=TIMESTEPS, n_rounds=3)