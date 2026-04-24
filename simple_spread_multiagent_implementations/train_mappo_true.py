"""
TRUE MAPPO implementation for simple_spread (PettingZoo MPE)

Key properties:
- Centralized critic with TD learning
- Proper GAE advantage estimation
- Decentralized actors
- Separate run directory for comparison
"""

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from pettingzoo.mpe import simple_spread_v3
from shaped_reward import shape_reward, TIMESTEPS


# ---------------------------------------------------------------------
# RUN CONFIG (SEPARATE OUTPUTS)
# ---------------------------------------------------------------------

RUN_NAME = "mappo_TRUE_baseline"
RUN_DIR = Path("runs") / RUN_NAME
RUN_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cpu")


# ---------------------------------------------------------------------
# HYPERPARAMETERS (STANDARD MAPPO STYLE)
# ---------------------------------------------------------------------

cfg = {
    "actor_lr": 3e-4,
    "critic_lr": 3e-4,
    "log_std_lr": 1e-4,

    "n_steps": 128,
    "batch_size": 64,
    "n_epochs": 4,

    "gamma": 0.99,
    "gae_lambda": 0.95,

    "clip_eps": 0.2,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,

    "hidden": 128,
}


cfg["total_timesteps"] = TIMESTEPS


# ---------------------------------------------------------------------
# NETWORKS
# ---------------------------------------------------------------------

class Actor(nn.Module):
    def __init__(self, obs_dim=18, act_dim=5, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, act_dim), nn.Tanh(),
        )
        self.log_std = nn.Parameter(torch.ones(act_dim) * -0.5)

    def forward(self, x):
        mean = self.net(x)
        std = self.log_std.exp().expand_as(mean)
        return torch.distributions.Normal(mean, std)


class Critic(nn.Module):
    def __init__(self, n_agents=3, obs_dim=18, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_agents * obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------
# ROLLOUT BUFFER
# ---------------------------------------------------------------------

def compute_gae(rewards, values, dones, gamma, lam):
    adv = np.zeros_like(rewards)
    last_gae = 0

    for t in reversed(range(len(rewards))):
        next_val = 0 if t == len(rewards) - 1 else values[t + 1]
        next_non_terminal = 1.0 - dones[t]

        delta = rewards[t] + gamma * next_val * next_non_terminal - values[t]
        last_gae = delta + gamma * lam * next_non_terminal * last_gae
        adv[t] = last_gae

    returns = adv + values
    return adv, returns


# ---------------------------------------------------------------------
# ROLLOUT COLLECTION
# ---------------------------------------------------------------------

def collect(env, actors, critic):
    agents = env.possible_agents

    obs_buf = {a: [] for a in agents}
    act_buf = {a: [] for a in agents}
    logp_buf = {a: [] for a in agents}
    rew_buf = {a: [] for a in agents}
    done_buf = []

    joint_buf = []
    value_buf = []

    obs, _ = env.reset()

    for _ in range(cfg["n_steps"]):

        joint = np.concatenate([obs[a] for a in agents])
        joint_buf.append(joint)

        actions = {}

        for a in agents:
            o = torch.tensor(obs[a], dtype=torch.float32)

            with torch.no_grad():
                dist = actors[a](o)
                action = dist.sample()
                logp = dist.log_prob(action).sum()

            actions[a] = action.numpy()

            obs_buf[a].append(obs[a])
            act_buf[a].append(action.numpy())
            logp_buf[a].append(logp.item())

        next_obs, rewards, terms, truncs, _ = env.step(actions)

        done = all(terms.values()) or all(truncs.values())
        done_buf.append(float(done))

        for a in agents:
            r = rewards.get(a, 0.0)
            r = shape_reward(next_obs[a], r)
            rew_buf[a].append(r)

        obs = next_obs if not done else env.reset()[0]

    # critic values
    joint_t = torch.tensor(np.array(joint_buf), dtype=torch.float32)
    with torch.no_grad():
        values = critic(joint_t).numpy()

    advantages = {}
    returns = {}

    for a in agents:
        adv, ret = compute_gae(
            np.array(rew_buf[a]),
            values,
            np.array(done_buf),
            cfg["gamma"],
            cfg["gae_lambda"]
        )
        advantages[a] = adv
        returns[a] = ret

    return {
        "obs": obs_buf,
        "actions": act_buf,
        "logprobs": logp_buf,
        "advantages": advantages,
        "returns": returns,
        "joint": np.array(joint_buf),
    }


# ---------------------------------------------------------------------
# UPDATE
# ---------------------------------------------------------------------

def update(actors, critic, actor_opt, critic_opt, batch):

    agents = list(actors.keys())

    joint = torch.tensor(batch["joint"], dtype=torch.float32)

    for _ in range(cfg["n_epochs"]):
        idx = torch.randperm(len(joint))

        for start in range(0, len(joint), cfg["batch_size"]):
            mb = idx[start:start + cfg["batch_size"]]

            # ---------------- ACTORS ----------------
            for a in agents:
                obs = torch.tensor(batch["obs"][a], dtype=torch.float32)[mb]
                act = torch.tensor(batch["actions"][a], dtype=torch.float32)[mb]
                old_lp = torch.tensor(batch["logprobs"][a], dtype=torch.float32)[mb]
                adv = torch.tensor(batch["advantages"][a], dtype=torch.float32)[mb]

                adv = (adv - adv.mean()) / (adv.std() + 1e-6)

                dist = actors[a](obs)
                new_lp = dist.log_prob(act).sum(-1)
                entropy = dist.entropy().mean()

                ratio = (new_lp - old_lp).exp()

                s1 = ratio * adv
                s2 = ratio.clamp(1 - cfg["clip_eps"], 1 + cfg["clip_eps"]) * adv

                loss = -torch.min(s1, s2).mean() - cfg["ent_coef"] * entropy

                actor_opt[a].zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(actors[a].parameters(), cfg["max_grad_norm"])
                actor_opt[a].step()

            # ---------------- CRITIC ----------------
            vals = critic(joint[mb])

            target = torch.tensor(
                np.mean([batch["returns"][a][mb] for a in agents], axis=0),
                dtype=torch.float32
            )

            c_loss = cfg["vf_coef"] * nn.functional.mse_loss(vals, target)

            critic_opt.zero_grad()
            c_loss.backward()
            critic_opt.step()


# ---------------------------------------------------------------------
# TRAIN LOOP
# ---------------------------------------------------------------------

def train():

    env = simple_spread_v3.parallel_env(
        N=3, max_cycles=25, continuous_actions=True
    )

    agents = ["agent_0", "agent_1", "agent_2"]

    actors = {a: Actor() for a in agents}
    critic = Critic()

    actor_opt = {
        a: Adam([
            {"params": actors[a].net.parameters(), "lr": cfg["actor_lr"]},
            {"params": [actors[a].log_std], "lr": cfg["log_std_lr"]},
        ])
        for a in agents
    }

    critic_opt = Adam(critic.parameters(), lr=cfg["critic_lr"])

    updates = cfg["total_timesteps"] // cfg["n_steps"]

    for i in range(updates):
        batch = collect(env, actors, critic)
        update(actors, critic, actor_opt, critic_opt, batch)

        if i % 10 == 0:
            print(f"[TRUE MAPPO] update {i}/{updates}")


if __name__ == "__main__":
    train()