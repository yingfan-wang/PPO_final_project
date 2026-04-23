from pathlib import Path
import numpy as np
from pettingzoo.mpe import simple_spread_v3
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor, DummyVecEnv
import gymnasium as gym
import shutil
from stable_baselines3.common.callbacks import EvalCallback
import random
from shaped_reward import shape_reward

# IPPO

# ---------------------------------------------------------------------------
# Reward shaping (shared logic)
# ---------------------------------------------------------------------------

# FIXED — cooperative shaping inspired by the TorchRL tutorial
# Reward is based on covering landmarks, not avoiding teammates

# def shape_reward(agent_obs, base_reward):
#     landmark_positions = agent_obs[4:10]
#     distances = [
#         np.sqrt(landmark_positions[2*i]**2 + landmark_positions[2*i+1]**2)
#         for i in range(3)
#     ]
#     nearest_dist = min(distances)

#     # Soft collision penalty — small enough to not override landmark-seeking
#     # Uses a smooth falloff instead of a hard threshold so agents are nudged
#     # away gently rather than incentivized to flee entirely
#     other_agent_obs = agent_obs[10:14]
#     collision_penalty = 0.0
#     for i in range(2):
#         dx = other_agent_obs[2*i]
#         dy = other_agent_obs[2*i + 1]
#         dist = np.sqrt(dx**2 + dy**2)
#         # Smooth penalty: only kicks in below 0.5, peaks at 0 distance
#         if dist < 0.5:
#             collision_penalty -= 0.3 * (0.5 - dist) / 0.5

#     return (
#         base_reward                                  # env's team reward (covers all landmarks)
#         + 3.0 * np.exp(-10.0 * nearest_dist)        # strong pull toward nearest landmark
#         - 0.5 * nearest_dist                         # linear pull to nearest landmark
#         + collision_penalty                          # soft nudge, not a flee incentive
#         # removed: 0.1 * np.mean(distances) ← this was rewarding spreading out!
#     )


# ---------------------------------------------------------------------------
# Single-agent gym wrapper — one agent trains, others use frozen models
# ---------------------------------------------------------------------------

class SingleAgentEnv(gym.Env):
    """
    Exposes one agent's obs/action/reward to SB3.
    The other two agents act via frozen PPO models (or randomly if None).
    Each instance owns its own parallel env so there are no shared-state issues.
    """

    def __init__(self, agent_id: str, frozen_models: dict):
        super().__init__()
        self.agent_id = agent_id
        self.frozen_models = frozen_models  # {agent_id: PPO | None}

        self.penv = simple_spread_v3.parallel_env(
            N=3, max_cycles=25, continuous_actions=True
        )
        obs, _ = self.penv.reset()
        self.observation_space = self.penv.observation_space(agent_id)
        self.action_space = self.penv.action_space(agent_id)
        self._last_obs = obs

    def reset(self, seed=None, options=None):
        obs, infos = self.penv.reset()
        self._last_obs = obs
        return obs[self.agent_id], infos.get(self.agent_id, {})

    def step(self, action):
        all_actions = {}
        for a in self.penv.agents:
            if a == self.agent_id:
                all_actions[a] = action
            elif a in self.frozen_models and self.frozen_models[a] is not None:
                frozen_obs = self._last_obs[a][np.newaxis]
                frozen_action, _ = self.frozen_models[a].predict(frozen_obs, deterministic=False)
                all_actions[a] = frozen_action[0]
            else:
                all_actions[a] = self.penv.action_space(a).sample()

        obs, rewards, terminations, truncations, infos = self.penv.step(all_actions)
        self._last_obs = obs

        done = terminations.get(self.agent_id, False) or truncations.get(self.agent_id, False)

        if self.agent_id in obs:
            reward = shape_reward(obs[self.agent_id], rewards.get(self.agent_id, 0.0))
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


def make_vecenv(agent_id, frozen_models):
    def _make():
        return SingleAgentEnv(agent_id, frozen_models)
    return VecMonitor(DummyVecEnv([_make]))


# ---------------------------------------------------------------------------
# Training — round-robin: train agent_0, freeze it, train agent_1, etc.
# ---------------------------------------------------------------------------

def train_agents(total_timesteps=500_000, n_rounds=3):
    run_dir = Path("runs") / "multiagent_spread"
    run_dir.mkdir(parents=True, exist_ok=True)

    agent_ids = [f"agent_{i}" for i in range(3)]
    # All models start as None → other agents act randomly until trained
    models: dict[str, PPO | None] = {a: None for a in agent_ids}

    timesteps_per_agent = total_timesteps // (n_rounds * len(agent_ids))

    for round_idx in range(n_rounds):
        order = agent_ids.copy()
        random.shuffle(order)         # ← different order every round
        print(f"Round {round_idx+1} order: {order}")

        for agent_id in order:        # ← use order instead of agent_ids
            print(f"\n--- Training {agent_id} (others frozen) ---")

            # Frozen models = everyone except the agent being trained
            frozen = {a: m for a, m in models.items() if a != agent_id}

            train_env = make_vecenv(agent_id, frozen)
            eval_env  = make_vecenv(agent_id, frozen)

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
                total_timesteps=timesteps_per_agent,
                reset_num_timesteps=False,
                callback=eval_callback, 
            )

            # Save checkpoint and update frozen copy for other agents
            model.save(str(run_dir / f"{agent_id}_round{round_idx}"))
            models[agent_id] = model
            train_env.close()

    # Save finals
    for agent_id, model in models.items():
        final_path = str(run_dir / f"{agent_id}_final")
        model.save(final_path)
        print(f"Saved: {final_path}")

    return models


# ---------------------------------------------------------------------------
# Duplicate best model → use one shared model for all 3 agents at test time
# ---------------------------------------------------------------------------

def duplicate_model_for_eval(source_agent="agent_0"):
    """
    Copies one agent's final model as a shared model for evaluation.
    Useful if you want a single model file driving all 3 dots.
    """
    run_dir = Path("runs") / "multiagent_spread"
    src = run_dir / f"{source_agent}_final.zip"
    dst = run_dir / "shared_eval_model.zip"
    shutil.copy(src, dst)
    print(f"Duplicated {src} → {dst}")


if __name__ == "__main__":
    train_agents(total_timesteps=1_500_000, n_rounds=3)
    # Optionally copy one model as a shared eval model:
    # duplicate_model_for_eval("agent_0")