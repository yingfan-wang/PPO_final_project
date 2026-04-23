from pathlib import Path
import numpy as np
import torch
from pettingzoo.mpe import simple_spread_v3
from train_mappo import Actor


def watch(n_episodes=20):
    run_dir   = Path("runs") / "mappo_spread"
    agent_ids = ["agent_0", "agent_1", "agent_2"]

    actors = {}
    for a in agent_ids:
        actor = Actor(obs_dim=18, action_dim=5)
        actor.load_state_dict(torch.load(str(run_dir / f"{a}_actor.pt")))
        actor.eval()
        actors[a] = actor
    print("Loaded MAPPO actors")

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
            actions = {}
            for a in env.agents:
                obs_t  = torch.FloatTensor(obs[a]).unsqueeze(0)
                dist   = actors[a](obs_t)
                action = dist.mean  # deterministic
                actions[a] = action.squeeze(0).detach().numpy()

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