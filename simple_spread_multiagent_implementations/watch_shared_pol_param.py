from pathlib import Path
import numpy as np
from pettingzoo.mpe import simple_spread_v3
from stable_baselines3 import PPO
from train_shared_pol_param import shape_reward



def watch(mode="independent", n_episodes=20):
    """
    mode="independent" : loads agent_0_final, agent_1_final, agent_2_final
    mode="shared"      : loads shared_eval_model.zip for all 3 dots
    """
    run_dir = Path("runs") / "multiagent_spread"
    agent_ids = [f"agent_{i}" for i in range(3)]

    if mode == "shared":
        shared = PPO.load(str(run_dir / "shared_eval_model"))
        models = {a: shared for a in agent_ids}
        print("Running with shared/duplicated model for all 3 agents")
    else:
        models = {
            a: PPO.load(str(run_dir / f"{a}_final"))
            for a in agent_ids
        }
        print("Running with 3 independent models")

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

        while env.agents:
            actions = {
                a: models[a].predict(obs[a][np.newaxis], deterministic=True)[0][0]
                for a in env.agents
            }
            obs, rewards, terminations, truncations, _ = env.step(actions)
            for a in agent_ids:
                if a in obs:
                    ep_reward[a] += shape_reward(obs[a], rewards.get(a, 0.0))

        mean_ep = np.mean(list(ep_reward.values()))
        total_reward += mean_ep
        runs += 1
        print(f"Episode {ep+1:3d} | Mean reward: {mean_ep:7.3f} | Running avg: {total_reward/runs:.3f}")

    env.close()


if __name__ == "__main__":
    # Change to mode="shared" to test the duplicated single model
    watch(mode="independent", n_episodes=20)