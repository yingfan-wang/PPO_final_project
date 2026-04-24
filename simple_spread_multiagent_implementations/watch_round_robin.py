from pathlib import Path
import numpy as np
from pettingzoo.mpe import simple_spread_v3
from stable_baselines3 import PPO
# from train_simple_spread_multiagent import SpreadShapingWrapper
from train_round_robin import SpreadShapingWrapper

run_dir = Path("runs") / "multiagent_spread"
agent_ids = [f"agent_{i}" for i in range(3)]

models = {
    agent_id: PPO.load(str(run_dir / f"{agent_id}_final"))
    for agent_id in agent_ids
}

raw_env = simple_spread_v3.parallel_env(
    N=3,
    max_cycles=25,
    continuous_actions=True,
    render_mode="human",
    dynamic_rescaling=False,
)
env = SpreadShapingWrapper(raw_env)

obs, _ = env.reset()
total_reward = 0
runs = 0

for _ in range(1000):
    actions = {
        agent_id: models[agent_id].predict(obs[agent_id][np.newaxis], deterministic=True)[0][0]
        for agent_id in env.agents
    }

    obs, rewards, terminations, truncations, info = env.step(actions)

    done = all(
        terminations.get(a, False) or truncations.get(a, False)
        for a in agent_ids
    )

    if done:
        total_reward += np.mean(list(rewards.values()))
        runs += 1
        print("Average reward", total_reward / runs)
        obs, _ = env.reset()

env.close()