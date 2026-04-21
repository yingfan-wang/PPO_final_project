from pathlib import Path
import numpy as np
from pettingzoo.mpe import simple_spread_v3
from stable_baselines3 import PPO
from train_multiagent import SpreadShapingWrapper


def evaluate(n_episodes=20, render=True):
    run_dir = Path("runs") / "multiagent_spread"
    agent_ids = [f"agent_{i}" for i in range(3)]

    models = {
        agent_id: PPO.load(str(run_dir / f"{agent_id}_final"))
        for agent_id in agent_ids
    }
    print("Loaded models:", list(models.keys()))

    raw_env = simple_spread_v3.parallel_env(
        N=3,
        max_cycles=25,
        continuous_actions=True,
        render_mode="human" if render else None,
    )
    env = SpreadShapingWrapper(raw_env)

    episode_rewards = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        ep_reward = {a: 0.0 for a in agent_ids}
        done = False

        while not done:
            actions = {}
            for agent_id in env.agents:
                agent_obs = obs[agent_id][np.newaxis]   # (1, obs_dim)
                action, _ = models[agent_id].predict(agent_obs, deterministic=True)
                actions[agent_id] = action[0]

            obs, rewards, terminations, truncations, _ = env.step(actions)

            for agent_id in agent_ids:
                ep_reward[agent_id] += rewards.get(agent_id, 0.0)

            done = all(
                terminations.get(a, False) or truncations.get(a, False)
                for a in agent_ids
            )

        mean_ep = np.mean(list(ep_reward.values()))
        episode_rewards.append(mean_ep)
        print(f"Episode {ep+1:3d} | Mean reward: {mean_ep:7.3f} | Per agent: { {k: f'{v:.2f}' for k, v in ep_reward.items()} }")

    print(f"\nOverall mean reward across {n_episodes} episodes: {np.mean(episode_rewards):.3f}")
    env.close()


if __name__ == "__main__":
    evaluate(n_episodes=20, render=True)