from pathlib import Path
import supersuit as ss
from pettingzoo.mpe import simple_spread_v3
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import VecMonitor

from pettingzoo.utils import ParallelEnv
import numpy as np
from shaped_reward import shape_reward


# Shared pol param (naive)

class SpreadShapingWrapper(ParallelEnv):
    def __init__(self, env):
        self.env = env
        self.env.reset()
        self.agents = env.agents
        self.possible_agents = env.possible_agents
        self.observation_spaces = env.observation_spaces
        self.action_spaces = env.action_spaces
        self.metadata = env.metadata
        self.render_mode = env.render_mode

    def reset(self, *args, **kwargs):
        obs, infos = self.env.reset(*args, **kwargs)
        self.agents = self.env.agents
        return obs, infos

    def step(self, actions):
        obs, rewards, terminations, truncations, infos = self.env.step(actions)
        self.agents = self.env.agents

        if not self.env.agents:
            return obs, rewards, terminations, truncations, infos

        # shaped_rewards = {}
        # for agent in self.env.agents:
        #     agent_obs = obs[agent]
        #     N_landmarks = 3
        #     landmark_positions = agent_obs[4:10]

        #     distances = []
        #     for i in range(N_landmarks):
        #         dx = landmark_positions[2*i]
        #         dy = landmark_positions[2*i + 1]
        #         dist = np.sqrt(dx**2 + dy**2)
        #         distances.append(dist)

        #     nearest_dist = min(distances)

        #     # Soft collision penalty — smooth falloff, not a flee incentive
        #     other_agent_obs = agent_obs[10:14]
        #     collision_penalty = 0.0
        #     for i in range(2):
        #         dx = other_agent_obs[2*i]
        #         dy = other_agent_obs[2*i + 1]
        #         dist = np.sqrt(dx**2 + dy**2)
        #         if dist < 0.5:
        #             collision_penalty -= 0.3 * (0.5 - dist) / 0.5

        #     shaped_rewards[agent] = (
        #         rewards[agent]
        #         + (5.0 * np.exp(-10.0 * nearest_dist))   # strong pull to nearest landmark
        #         - (0.5 * nearest_dist)                    # linear pull to nearest landmark
        #         + collision_penalty                       # soft nudge only, not flee incentive
        #         # removed: 0.1 * np.mean(distances)       # ← was rewarding spreading out
        #     )
        shaped_rewards = {}
        for agent in self.env.agents:
            shaped_rewards[agent] = shape_reward(obs[agent], rewards[agent])

        return obs, shaped_rewards, terminations, truncations, infos

    def observation_space(self, agent):
        return self.env.observation_space(agent)

    def action_space(self, agent):
        return self.env.action_space(agent)

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()

def make_env():
    env = simple_spread_v3.parallel_env(
        N=3,
        max_cycles=25,
        continuous_actions=True,
        dynamic_rescaling=False
    )
    env = SpreadShapingWrapper(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 1, base_class="stable_baselines3")
    env = VecMonitor(env)
    return env


def main():
    run_dir = Path("runs") / "simple_spread"
    run_dir.mkdir(parents=True, exist_ok=True)

    train_env = make_env()
    eval_env  = make_env()

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=3e-4,
        n_steps=256,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.001,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log=str(run_dir / "tb"),
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(run_dir / "best_model"),
        log_path=str(run_dir / "eval_logs"),
        eval_freq=10_000,
        n_eval_episodes=10,
        deterministic=True,
        render=False,
    )

    # model.learn(total_timesteps=1_000_000, callback=eval_callback)
    model.learn(total_timesteps=500_000, callback=eval_callback)
    model.save(str(run_dir / "final_model"))

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()