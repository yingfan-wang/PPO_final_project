"""
MAPPO for simple_spread (PettingZoo MPE)

- decentralized actors
- centralized critic
- uses reward shaping from shaped_reward.py
- stable hyperparameter separation
"""

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from pettingzoo.mpe import simple_spread_v3

from shaped_reward import shape_reward, TIMESTEPS, HYPERPARAMS as BASE_CFG


# ---------------------------------------------------------------------
# CONFIG (split cleanly)
# ---------------------------------------------------------------------

cfg = dict(BASE_CFG)  # only shared PPO-style values

# MAPPO-specific overrides / additions
cfg.update({
    "actor_lr": 3e-4,
    "critic_lr": 3e-4,
    "actor_logstd_lr": 1e-4,
    "clip_eps": 0.2,

    # ensure compatibility if missing in base
    "n_steps": cfg.get("n_steps", 512),
    "batch_size": cfg.get("batch_size", 256),
    "n_epochs": cfg.get("n_epochs", 10),
    "gamma": cfg.get("gamma", 0.99),
    "gae_lambda": cfg.get("gae_lambda", 0.95),
    "ent_coef": cfg.get("ent_coef", 0.01),
    "vf_coef": cfg.get("vf_coef", 0.5),
    "max_grad_norm": cfg.get("max_grad_norm", 0.5),
})

cfg["total_timesteps"] = TIMESTEPS
cfg["hidden"] = 128
cfg["eval_freq"] = 10
cfg["n_eval_episodes"] = 10


# ---------------------------------------------------------------------
# NETWORKS
# ---------------------------------------------------------------------

class Actor(nn.Module):
    def __init__(self, obs_dim=18, action_dim=5, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, action_dim), nn.Tanh(),
        )
        self.log_std = nn.Parameter(torch.ones(action_dim) * -0.5)

    def forward(self, obs):
        mean = self.net(obs)
        mean = torch.nan_to_num(mean, nan=0.0, posinf=1.0, neginf=-1.0)

        log_std = torch.clamp(self.log_std, -4.0, 0.5)
        std = log_std.exp().expand_as(mean)

        return torch.distributions.Normal(mean, std)


class CentralCritic(nn.Module):
    def __init__(self, n_agents=3, obs_dim=18, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_agents * obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------
# ROLLOUT
# ---------------------------------------------------------------------

def collect_rollout(env, actors, critic):
    agent_ids = list(env.possible_agents)

    all_obs = {a: [] for a in agent_ids}
    all_actions = {a: [] for a in agent_ids}
    all_logprobs = {a: [] for a in agent_ids}
    all_rewards = {a: [] for a in agent_ids}
    all_dones = []
    all_joint = []

    obs, _ = env.reset()

    for _ in range(cfg["n_steps"]):

        joint = np.concatenate([obs[a] for a in agent_ids], dtype=np.float32)
        all_joint.append(joint)

        actions = {}

        for a in agent_ids:
            obs_t = torch.FloatTensor(obs[a]).unsqueeze(0)

            with torch.no_grad():
                dist = actors[a](obs_t)
                action = (dist.sample().clamp(-1, 1) + 1) / 2
                logp = dist.log_prob(action).sum(-1)

            actions[a] = action.squeeze(0).numpy()

            all_obs[a].append(obs[a].copy())
            all_actions[a].append(actions[a].copy())
            all_logprobs[a].append(logp.item())

        obs_next, rewards, terminations, truncations, _ = env.step(actions)

        for a in agent_ids:
            r = rewards.get(a, 0.0)
            if a in obs_next:
                r = shape_reward(obs_next[a], r)
            all_rewards[a].append(r)

        done = all(
            terminations.get(a, False) or truncations.get(a, False)
            for a in agent_ids
        )
        all_dones.append(float(done))

        obs = obs_next if not done else env.reset()[0]

    return {
        "obs": {a: np.array(all_obs[a]) for a in agent_ids},
        "actions": {a: np.array(all_actions[a]) for a in agent_ids},
        "logprobs": {a: np.array(all_logprobs[a]) for a in agent_ids},
        "rewards": {a: np.array(all_rewards[a]) for a in agent_ids},
        "dones": np.array(all_dones),
        "joint_obs": np.array(all_joint),
    }


# ---------------------------------------------------------------------
# PPO UPDATE (MAPPO style)
# ---------------------------------------------------------------------

def ppo_update(actors, critic, actor_opts, critic_opt, batch):

    agent_ids = list(actors.keys())

    joint_obs = torch.FloatTensor(batch["joint_obs"])
    n = len(joint_obs)

    for _ in range(cfg["n_epochs"]):
        idx = torch.randperm(n)

        for start in range(0, n, cfg["batch_size"]):
            mb_idx = idx[start:start + cfg["batch_size"]]

            # ---------------- ACTORS ----------------
            for a in agent_ids:
                mb_obs = torch.FloatTensor(batch["obs"][a][mb_idx])
                mb_act = torch.FloatTensor(batch["actions"][a][mb_idx])
                mb_lp = torch.FloatTensor(batch["logprobs"][a][mb_idx])

                rewards = torch.FloatTensor(batch["rewards"][a][mb_idx])
                adv = (rewards - rewards.mean()) / (rewards.std() + 1e-6)

                dist = actors[a](mb_obs)
                new_lp = dist.log_prob(mb_act).sum(-1)
                entropy = dist.entropy().sum(-1).mean()

                ratio = (new_lp - mb_lp).exp()

                s1 = ratio * adv
                s2 = ratio.clamp(1 - cfg["clip_eps"], 1 + cfg["clip_eps"]) * adv

                loss = -torch.min(s1, s2).mean() - cfg["ent_coef"] * entropy

                actor_opts[a].zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(actors[a].parameters(), cfg["max_grad_norm"])
                actor_opts[a].step()

            # ---------------- CRITIC ----------------
            mb_joint = joint_obs[mb_idx]
            values = critic(mb_joint)

            target = torch.stack([
                torch.FloatTensor(batch["rewards"][a][mb_idx])
                for a in agent_ids
            ]).sum(dim=0)

            c_loss = cfg["vf_coef"] * nn.functional.mse_loss(values, target)

            critic_opt.zero_grad()
            c_loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), cfg["max_grad_norm"])
            critic_opt.step()


# ---------------------------------------------------------------------
# TRAIN LOOP
# ---------------------------------------------------------------------

def train():
    agent_ids = ["agent_0", "agent_1", "agent_2"]

    actors = {a: Actor(18, 5, cfg["hidden"]) for a in agent_ids}
    critic = CentralCritic(3, 18, cfg["hidden"])

    actor_opts = {
        a: Adam([
            {"params": actors[a].net.parameters(), "lr": cfg["actor_lr"]},
            {"params": [actors[a].log_std], "lr": cfg["actor_logstd_lr"]},
        ])
        for a in agent_ids
    }

    critic_opt = Adam(critic.parameters(), lr=cfg["critic_lr"])

    env = simple_spread_v3.parallel_env(N=3, max_cycles=25, continuous_actions=True)

    n_updates = cfg["total_timesteps"] // cfg["n_steps"]

    for update in range(n_updates):

        batch = collect_rollout(env, actors, critic)
        ppo_update(actors, critic, actor_opts, critic_opt, batch)

        if update % cfg["eval_freq"] == 0:
            print(f"update {update} complete")


if __name__ == "__main__":
    train()