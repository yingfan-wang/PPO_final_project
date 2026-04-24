import numpy as np
import imageio
import os
import torch
from pathlib import Path
from pettingzoo.mpe import simple_spread_v3
from stable_baselines3 import PPO
import torch.nn as nn


# =========================
# MAPPO Actor (MATCHES YOUR TRAINING)
# =========================
class Actor(nn.Module):
    def __init__(self, obs_dim=18, action_dim=5, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),  nn.Tanh(),
            nn.Linear(hidden, action_dim), nn.Tanh(),
        )
        self.log_std = nn.Parameter(torch.ones(action_dim) * -0.5)

    def forward(self, obs):
        mean = self.net(obs)
        log_std = torch.clamp(self.log_std, min=-4.0, max=0.5)
        std = log_std.exp().expand_as(mean)
        return torch.distributions.Normal(mean, std)


# =========================
# LOAD MODELS
# =========================
def load_models(config):
    mode = config["mode"]

    if mode == "shared":
        return PPO.load(config["model_path"])

    elif mode in ("multi", "joint"):
        return {
            a: PPO.load(path)
            for a, path in config["model_paths"].items()
        }

    elif mode == "mappo":
        actors = {}
        for a, path in config["actor_paths"].items():
            actor = Actor()
            actor.load_state_dict(torch.load(path, map_location="cpu"))
            actor.eval()
            actors[a] = actor
        return actors


# =========================
# GENERATE GIF
# =========================
def generate_gif(config, output_path, episodes=3, fps=15):
    env = simple_spread_v3.parallel_env(
        N=3,
        max_cycles=25,
        continuous_actions=True,
        render_mode="rgb_array"
    )

    models = load_models(config)
    agent_ids = ["agent_0", "agent_1", "agent_2"]

    frames = []

    for ep in range(episodes):
        obs, _ = env.reset()
        last_obs = dict(obs)

        while env.agents:
            frames.append(env.render())

            actions = {}

            if config["mode"] == "shared":
                for a in env.agents:
                    action, _ = models.predict(
                        obs[a][np.newaxis], deterministic=True
                    )
                    actions[a] = action[0]

            elif config["mode"] == "multi":
                for a in env.agents:
                    action, _ = models[a].predict(
                        obs[a][np.newaxis], deterministic=True
                    )
                    actions[a] = action[0]

            elif config["mode"] == "joint":
                for a in env.agents:
                    joint = np.concatenate(
                        [last_obs[a]] + [last_obs[x] for x in agent_ids if x != a],
                        dtype=np.float32
                    )
                    action, _ = models[a].predict(
                        joint[np.newaxis], deterministic=True
                    )
                    actions[a] = action[0]

            elif config["mode"] == "mappo":
                for a in env.agents:
                    obs_t = torch.FloatTensor(obs[a]).unsqueeze(0)
                    with torch.no_grad():
                        dist = models[a](obs_t)
                        action = (dist.mean.clamp(-1, 1) + 1) / 2
                    actions[a] = action.squeeze(0).numpy()

            obs, _, _, _, _ = env.step(actions)
            last_obs = {**last_obs, **obs}

    env.close()
    imageio.mimsave(output_path, frames, fps=fps)
    print(f"✓ Generated: {output_path}")


# =========================
# RUN ALL
# =========================
def main():
    os.makedirs("gifs", exist_ok=True)

    experiments = [
        {
            "label": "parameter_sharing",
            "mode": "shared",
            "model_path": "runs/simple_spread/final_model",
        },
        {
            "label": "round_robin",
            "mode": "multi",
            "model_paths": {
                "agent_0": "runs/round_robin/agent_0_final",
                "agent_1": "runs/round_robin/agent_1_final",
                "agent_2": "runs/round_robin/agent_2_final",
            },
        },
        {
            "label": "ippo",
            "mode": "multi",
            "model_paths": {
                "agent_0": "runs/multiagent_spread/agent_0_final",
                "agent_1": "runs/multiagent_spread/agent_1_final",
                "agent_2": "runs/multiagent_spread/agent_2_final",
            },
        },
        {
            "label": "joint_obs",
            "mode": "joint",
            "model_paths": {
                "agent_0": "runs/joint_obs_spread/agent_0_final",
                "agent_1": "runs/joint_obs_spread/agent_1_final",
                "agent_2": "runs/joint_obs_spread/agent_2_final",
            },
        },
        {
            "label": "mappo",
            "mode": "mappo",
            "actor_paths": {
                "agent_0": "runs/mappo_spread/agent_0_actor.pt",
                "agent_1": "runs/mappo_spread/agent_1_actor.pt",
                "agent_2": "runs/mappo_spread/agent_2_actor.pt",
            },
        },
        {
            "label": "MAPPO (True)",
            "mode": "mappo",
            "actor_paths": {
                "agent_0": "runs/mappo_TRUE_baseline/agent_0_actor.pt",
                "agent_1": "runs/mappo_TRUE_baseline/agent_1_actor.pt",
                "agent_2": "runs/mappo_TRUE_baseline/agent_2_actor.pt",
            },
        }
    ]

    for exp in experiments:
        print(f"\nProcessing: {exp['label']}")
        try:
            generate_gif(
                exp,
                f"gifs/{exp['label']}.gif",
                episodes=3,
                fps=15
            )
        except Exception as e:
            print(f"✗ Failed: {e}")

    print("\n✓ Done!")


if __name__ == "__main__":
    main()