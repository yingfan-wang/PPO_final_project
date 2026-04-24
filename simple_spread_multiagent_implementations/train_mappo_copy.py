"""
train_mappo.py
--------------
MAPPO implementation for simple_spread using PyTorch.
- 3 decentralised actors (each sees own obs only, 18 dim)
- 1 centralised critic (sees all agents' obs concatenated, 54 dim)
- Team GAE — one shared advantage signal from joint reward
- Saves evaluations.npz in same format as EvalCallback for plot_results.py

Usage:
    python train_mappo.py
"""

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from pettingzoo.mpe import simple_spread_v3
from shaped_reward import shape_reward, TIMESTEPS


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------

class Actor(nn.Module):
    """
    Decentralised actor — sees only own obs (18 dim) → outputs action (5 dim).
    Tanh output + affine rescale to [0, 1] at sample/eval time.
    """
    def __init__(self, obs_dim=18, action_dim=5, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),  nn.Tanh(),
            nn.Linear(hidden, action_dim), nn.Tanh(),
        )
        self.log_std = nn.Parameter(torch.ones(action_dim) * -0.5)

    def forward(self, obs):
        mean    = self.net(obs)
        mean    = torch.nan_to_num(mean, nan=0.0, posinf=1.0, neginf=-1.0)
        log_std = torch.clamp(self.log_std, min=-4.0, max=0.5)
        std     = log_std.exp().expand_as(mean)
        std     = torch.nan_to_num(std, nan=1e-4, posinf=1.0, neginf=1e-4)
        return torch.distributions.Normal(mean, std)


class CentralCritic(nn.Module):
    """
    Centralised critic — sees ALL agents' obs concatenated (54 dim) → scalar.
    Three hidden layers to handle the larger input.
    """
    def __init__(self, n_agents=3, obs_dim=18, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_agents * obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),              nn.Tanh(),
            nn.Linear(hidden, hidden // 2),         nn.Tanh(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, all_obs):
        return self.net(all_obs).squeeze(-1)


# ---------------------------------------------------------------------------
# Rollout collection
# ---------------------------------------------------------------------------

def collect_rollout(env, actors, critic, n_steps=1024, gamma=0.99, gae_lambda=0.95):
    agent_ids = list(env.possible_agents)

    all_obs      = {a: [] for a in agent_ids}
    all_actions  = {a: [] for a in agent_ids}
    all_logprobs = {a: [] for a in agent_ids}
    all_rewards  = {a: [] for a in agent_ids}
    all_dones    = []
    all_joint    = []

    obs, _ = env.reset()

    for _ in range(n_steps):
        joint = np.concatenate([obs[a] for a in agent_ids], dtype=np.float32)
        all_joint.append(joint)

        actions  = {}
        logprobs = {}
        for a in agent_ids:
            obs_t = torch.FloatTensor(obs[a]).unsqueeze(0)
            with torch.no_grad():
                dist   = actors[a](obs_t)
                action = (dist.sample().clamp(-1, 1) + 1) / 2
                logp   = dist.log_prob(action).sum(-1)
            actions[a]  = action.squeeze(0).numpy()
            logprobs[a] = logp.item()
            all_obs[a].append(obs[a].copy())
            all_actions[a].append(actions[a].copy())
            all_logprobs[a].append(logprobs[a])

        obs_next, rewards, terminations, truncations, _ = env.step(actions)

        for a in agent_ids:
            if a in obs_next:
                r = shape_reward(obs_next[a], rewards.get(a, 0.0))
            else:
                r = rewards.get(a, 0.0)
            all_rewards[a].append(r)

        done = all(
            terminations.get(a, False) or truncations.get(a, False)
            for a in agent_ids
        )
        all_dones.append(float(done))

        if done:
            obs, _ = env.reset()
        else:
            obs = obs_next

    # ── Team GAE — one shared advantage for all agents ───────────────────────
    # Average per-step reward across agents → single team signal
    team_rewards = [
        np.mean([all_rewards[a][t] for a in agent_ids])
        for t in range(n_steps)
    ]

    joint_t = torch.FloatTensor(np.array(all_joint))
    with torch.no_grad():
        values = critic(joint_t).numpy()

    adv_list = []
    gae      = 0.0
    for t in reversed(range(n_steps)):
        next_val = 0.0 if all_dones[t] else (values[t + 1] if t + 1 < n_steps else 0.0)
        delta    = team_rewards[t] + gamma * next_val * (1 - all_dones[t]) - values[t]
        gae      = delta + gamma * gae_lambda * (1 - all_dones[t]) * gae
        adv_list.insert(0, gae)

    ret_list = [a_ + v for a_, v in zip(adv_list, values[:n_steps])]

    # All agents share the same advantage estimate
    advantages = {a: np.array(adv_list) for a in agent_ids}
    returns    = {a: np.array(ret_list) for a in agent_ids}

    return {
        "obs":        {a: np.array(all_obs[a])      for a in agent_ids},
        "actions":    {a: np.array(all_actions[a])  for a in agent_ids},
        "logprobs":   {a: np.array(all_logprobs[a]) for a in agent_ids},
        "advantages": advantages,
        "returns":    returns,
        "joint_obs":  np.array(all_joint),
        "team_rewards": np.array(team_rewards),
    }


# ---------------------------------------------------------------------------
# PPO update
# ---------------------------------------------------------------------------

def ppo_update(actors, critic, actor_opts, critic_opt, batch,
               agent_ids, clip_eps=0.2, n_epochs=10, batch_size=256,
               ent_coef=0.01, vf_coef=0.5):

    joint_obs = torch.FloatTensor(batch["joint_obs"])
    n         = len(joint_obs)

    for _ in range(n_epochs):
        idx = torch.randperm(n)
        for start in range(0, n, batch_size):
            mb_idx = idx[start:start + batch_size].numpy()

            # Shared advantage for this minibatch
            mb_adv = torch.FloatTensor(batch["advantages"]["agent_0"][mb_idx])
            mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std().clamp(min=1e-6) + 1e-6)

            # ── Actor updates ────────────────────────────────────────────────
            for a in agent_ids:
                mb_obs = torch.FloatTensor(batch["obs"][a][mb_idx])
                mb_act = torch.FloatTensor(batch["actions"][a][mb_idx])
                mb_lp  = torch.FloatTensor(batch["logprobs"][a][mb_idx])

                dist    = actors[a](mb_obs)
                new_lp  = dist.log_prob(mb_act).sum(-1)
                entropy = dist.entropy().sum(-1).mean()

                ratio = (new_lp - mb_lp).exp()
                surr1 = ratio * mb_adv
                surr2 = ratio.clamp(1 - clip_eps, 1 + clip_eps) * mb_adv
                a_loss = -torch.min(surr1, surr2).mean() - ent_coef * entropy

                actor_opts[a].zero_grad()
                a_loss.backward()
                nn.utils.clip_grad_norm_(actors[a].parameters(), 0.5)
                actor_opts[a].step()

            # ── Critic update ────────────────────────────────────────────────
            mb_joint  = joint_obs[mb_idx]
            mb_ret    = torch.FloatTensor(batch["returns"]["agent_0"][mb_idx])
            values    = critic(mb_joint)
            c_loss    = vf_coef * nn.functional.mse_loss(values, mb_ret.detach())

            critic_opt.zero_grad()
            c_loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
            critic_opt.step()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(actors, n_episodes=10):
    env       = simple_spread_v3.parallel_env(N=3, max_cycles=25, continuous_actions=True)
    agent_ids = list(env.possible_agents)
    ep_rewards = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        ep_rew = {a: 0.0 for a in agent_ids}
        while env.agents:
            actions = {}
            for a in env.agents:
                obs_t = torch.FloatTensor(obs[a]).unsqueeze(0)
                with torch.no_grad():
                    dist   = actors[a](obs_t)
                    action = (dist.mean.clamp(-1, 1) + 1) / 2
                actions[a] = action.squeeze(0).numpy()
            obs, rewards, _, _, _ = env.step(actions)
            for a in agent_ids:
                ep_rew[a] += rewards.get(a, 0.0)
        ep_rewards.append(np.mean(list(ep_rew.values())))

    env.close()
    return np.array(ep_rewards)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(total_timesteps=TIMESTEPS, n_steps=1024, eval_freq=10):
    run_dir = Path("runs") / "mappo_spread"
    run_dir.mkdir(parents=True, exist_ok=True)

    agent_ids = ["agent_0", "agent_1", "agent_2"]
    obs_dim   = 18
    act_dim   = 5

    actors  = {a: Actor(obs_dim, act_dim, hidden=128) for a in agent_ids}
    critic  = CentralCritic(n_agents=3, obs_dim=obs_dim, hidden=128)

    # Separate lr for log_std to prevent it wandering
    actor_opts = {
        a: Adam([
            {"params": actors[a].net.parameters(), "lr": 1e-4},
            {"params": [actors[a].log_std],         "lr": 5e-5},
        ], lr=1e-4)
        for a in agent_ids
    }
    critic_opt = Adam(critic.parameters(), lr=3e-4)

    env = simple_spread_v3.parallel_env(N=3, max_cycles=25, continuous_actions=True)

    n_updates    = total_timesteps // n_steps
    eval_results = []
    timesteps    = []

    print(f"Training MAPPO for {total_timesteps:,} timesteps ({n_updates} updates)")

    for update in range(n_updates):
        batch = collect_rollout(env, actors, critic, n_steps=n_steps)
        ppo_update(actors, critic, actor_opts, critic_opt, batch, agent_ids)

        if (update + 1) % eval_freq == 0:
            ep_rews = evaluate(actors, n_episodes=10)
            t       = (update + 1) * n_steps
            eval_results.append(ep_rews)
            timesteps.append(t)
            print(f"  Update {update+1:4d}/{n_updates} | "
                  f"timestep {t:7d} | "
                  f"mean eval reward: {ep_rews.mean():.3f} ± {ep_rews.std():.3f}")

    env.close()

    # ── Save eval logs (same format as EvalCallback) ─────────────────────────
    eval_log_dir = run_dir / "eval_logs"
    eval_log_dir.mkdir(exist_ok=True)
    np.savez(
        str(eval_log_dir / "evaluations.npz"),
        timesteps  = np.array(timesteps),
        results    = np.array(eval_results),
        ep_lengths = np.zeros((len(eval_results), 10)),
    )
    print(f"Saved eval logs → {eval_log_dir}/evaluations.npz")

    # ── Save actor and critic weights ────────────────────────────────────────
    for a in agent_ids:
        torch.save(actors[a].state_dict(), str(run_dir / f"{a}_actor.pt"))
    torch.save(critic.state_dict(), str(run_dir / "critic.pt"))
    print(f"Saved models → {run_dir}/")

    return actors, critic


if __name__ == "__main__":
    train()