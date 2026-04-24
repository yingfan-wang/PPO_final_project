from pathlib import Path
import numpy as np
import torch

from pettingzoo.mpe import simple_spread_v3
from train_mappo_true import Actor   # IMPORTANT: use TRUE MAPPO actor


# ---------------------------------------------------------------------
# WATCH CONFIG
# ---------------------------------------------------------------------

RUN_DIR = Path("runs") / "mappo_TRUE_baseline"
AGENTS = ["agent_0", "agent_1", "agent_2"]


# ---------------------------------------------------------------------
# LOAD ACTORS
# ---------------------------------------------------------------------

def load_actors():
    actors = {}

    for a in AGENTS:
        actor = Actor(obs_dim=18, act_dim=5)

        path = RUN_DIR / f"{a}_actor.pt"

        if not path.exists():
            raise FileNotFoundError(
                f"Missing checkpoint for {a}: {path}\n"
                "Make sure you saved models after training."
            )

        actor.load_state_dict(torch.load(path, map_location="cpu"))
        actor.eval()
        actors[a] = actor

    print("✅ Loaded MAPPO actors")
    return actors


# ---------------------------------------------------------------------
# WATCH LOOP
# ---------------------------------------------------------------------

def watch(n_episodes=20, deterministic=True):

    actors = load_actors()

    env = simple_spread_v3.parallel_env(
        N=3,
        max_cycles=25,
        continuous_actions=True,
        render_mode="human",
        dynamic_rescaling=False,
    )

    print("🎥 Starting watch...")

    for ep in range(n_episodes):

        obs, _ = env.reset()
        ep_reward = {a: 0.0 for a in AGENTS}

        while env.agents:

            actions = {}

            for a in env.agents:
                obs_t = torch.tensor(obs[a], dtype=torch.float32).unsqueeze(0)

                with torch.no_grad():
                    dist = actors[a](obs_t)

                    if deterministic:
                        action = dist.mean
                    else:
                        action = dist.sample()

                actions[a] = action.squeeze(0).numpy()

            obs, rewards, terminations, truncations, _ = env.step(actions)

            for a in AGENTS:
                ep_reward[a] += rewards.get(a, 0.0)

        mean_ep = np.mean(list(ep_reward.values()))

        print(
            f"Episode {ep+1:3d} | "
            f"Mean reward: {mean_ep:7.3f}"
        )

    env.close()


# ---------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------

if __name__ == "__main__":
    watch(n_episodes=20)