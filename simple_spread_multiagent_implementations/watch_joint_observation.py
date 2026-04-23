from pathlib import Path
import numpy as np
from pettingzoo.mpe import simple_spread_v3
from stable_baselines3 import PPO
from train_joint_observation import JointObsAgentEnv


def watch(n_episodes=20):
    run_dir  = Path("runs") / "joint_obs_spread"
    agent_ids = [f"agent_{i}" for i in range(3)]

    models = {
        a: PPO.load(str(run_dir / f"{a}_final"))
        for a in agent_ids
    }
    print("Loaded joint obs models")

    env = simple_spread_v3.parallel_env(
        N=3,
        max_cycles=25,
        continuous_actions=True,
        render_mode="human",
        dynamic_rescaling=False,
    )

    total_reward = 0.0
    runs = 0

    for ep in range(n_episodes):
        obs, _ = env.reset()
        ep_reward = {a: 0.0 for a in agent_ids}

        # Build joint obs for each agent: [own | other1 | other2]
        def make_joint(agent_id, obs_dict):
            parts = [obs_dict[agent_id]]
            for a in agent_ids:
                if a != agent_id:
                    parts.append(obs_dict[a])
            return np.concatenate(parts, dtype=np.float32)

        while env.agents:
            actions = {
                a: models[a].predict(
                    make_joint(a, obs)[np.newaxis], deterministic=True
                )[0][0]
                for a in env.agents
            }
            obs, rewards, terminations, truncations, _ = env.step(actions)
            for a in agent_ids:
                ep_reward[a] += rewards.get(a, 0.0)

        mean_ep = np.mean(list(ep_reward.values()))
        total_reward += mean_ep
        runs += 1
        print(f"Episode {ep+1:3d} | Mean reward: {mean_ep:7.3f} | Running avg: {total_reward/runs:.3f}")

    env.close()


if __name__ == "__main__":
    watch(n_episodes=20)