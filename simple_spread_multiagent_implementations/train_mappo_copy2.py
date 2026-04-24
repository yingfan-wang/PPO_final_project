"""
train_simple_spread_mappo.py
----------------------------
MAPPO implementation for simple_spread using PyTorch.
- 3 decentralised actors (each sees own obs only)
- 1 centralised critic (sees all agents' obs concatenated)
- Saves evaluations.npz in same format as other runs for plot_results.py
"""

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from pettingzoo.mpe import simple_spread_v3
from shaped_reward import shape_reward
from shaped_reward import TIMESTEPS


# ---------------------------------------------------------------------------
# Reward shaping (same as other implementations)
# ---------------------------------------------------------------------------

# def shape_reward(agent_obs, base_reward):
#     """Cooperative reward shaping - encourages covering landmarks, soft collision avoidance"""
#     landmark_positions = agent_obs[4:10]
#     distances = [
#         np.sqrt(landmark_positions[2*i]**2 + landmark_positions[2*i+1]**2)
#         for i in range(3)
#     ]
#     nearest_dist = min(distances)
    
#     # Soft collision penalty
#     other_agent_obs = agent_obs[10:14]
#     collision_penalty = 0.0
#     for i in range(2):
#         dx = other_agent_obs[2*i]
#         dy = other_agent_obs[2*i + 1]
#         dist = np.sqrt(dx**2 + dy**2)
#         if dist < 0.5:
#             collision_penalty -= 0.3 * (0.5 - dist) / 0.5
    
#     return (
#         base_reward
#         + 3.0 * np.exp(-10.0 * nearest_dist)        # strong pull toward nearest landmark
#         - 0.5 * nearest_dist                         # linear pull to nearest landmark
#         + collision_penalty                          # soft nudge, not flee incentive
#     )


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------

class Actor(nn.Module):
    """Decentralised actor — sees only own obs (18 values) → outputs action (5 values)."""
    def __init__(self, obs_dim=18, action_dim=5, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),  nn.Tanh(),
            nn.Linear(hidden, action_dim), nn.Tanh(),  # actions in [0, 1]
        )
        # self.log_std = nn.Parameter(torch.zeros(action_dim) - 1.0)  # smaller initial std
        self.log_std = nn.Parameter(torch.ones(action_dim) * -0.5)

    def forward(self, obs):
        mean = self.net(obs)
        mean = torch.nan_to_num(mean, nan=0.0, posinf=1.0, neginf=-1.0)
        # Clamp std to prevent numerical issues
        # std  = torch.clamp(self.log_std.exp(), min=1e-4, max=1.0).expand_as(mean)
        log_std = torch.clamp(self.log_std, min=-4.0, max=0.5)  # clamp BEFORE exp
        std = log_std.exp().expand_as(mean)
        std = torch.nan_to_num(std, nan=1e-4, posinf=1.0, neginf=1e-4)
        return torch.distributions.Normal(mean, std)


class CentralCritic(nn.Module):
    """Centralised critic — sees ALL agents' obs concatenated (54 values) → scalar value."""
    def __init__(self, n_agents=3, obs_dim=18, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_agents * obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),             nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, all_obs):
        # all_obs: (batch, n_agents * obs_dim)
        return self.net(all_obs).squeeze(-1)


# ---------------------------------------------------------------------------
# Rollout collection
# ---------------------------------------------------------------------------

def collect_rollout(env, actors, critic, n_steps=1024, gamma=0.99, gae_lambda=0.95):
    agent_ids = list(env.possible_agents)
    n_agents  = len(agent_ids)

    # Storage
    all_obs     = {a: [] for a in agent_ids}  # per agent local obs
    all_actions = {a: [] for a in agent_ids}
    all_logprobs= {a: [] for a in agent_ids}
    all_rewards = {a: [] for a in agent_ids}
    all_dones   = []
    all_joint   = []  # joint obs for critic

    obs, _ = env.reset()
    done   = False

    for _ in range(n_steps):
        joint = np.concatenate([obs[a] for a in agent_ids], dtype=np.float32)
        all_joint.append(joint)

        actions   = {}
        logprobs  = {}
        for a in agent_ids:
            obs_t  = torch.FloatTensor(obs[a]).unsqueeze(0)
            
            with torch.no_grad():
                dist   = actors[a](obs_t)
                # action = dist.sample()
                # # Clamp to valid range
                # action = torch.clamp(action, 0.0, 1.0)
                # logp   = dist.log_prob(action).sum(-1)
                action = (dist.sample().clamp(-1, 1) + 1) / 2
                logp   = dist.log_prob(action).sum(-1)
            
            actions[a]  = action.squeeze(0).numpy()
            logprobs[a] = logp.item()
            all_obs[a].append(obs[a].copy())
            all_actions[a].append(actions[a].copy())
            all_logprobs[a].append(logprobs[a])

        obs_next, rewards, terminations, truncations, _ = env.step(actions)

        # Apply reward shaping
        shaped = {}
        for a in agent_ids:
            if a in obs_next:
                shaped[a] = shape_reward(obs_next[a], rewards.get(a, 0.0))
            else:
                shaped[a] = rewards.get(a, 0.0)
            all_rewards[a].append(shaped[a])

        done = all(terminations.get(a, False) or truncations.get(a, False) for a in agent_ids)
        all_dones.append(float(done))

        if done:
            obs, _ = env.reset()
        else:
            obs = obs_next

    # GAE per agent using centralized value function
    advantages = {a: [] for a in agent_ids}
    returns    = {a: [] for a in agent_ids}

    joint_t = torch.FloatTensor(np.array(all_joint))
    with torch.no_grad():
        values  = critic(joint_t).numpy()

    for a in agent_ids:
        rews  = all_rewards[a]
        dones = all_dones
        adv   = []
        gae   = 0.0
        for t in reversed(range(len(rews))):
            # next_val = values[t + 1] if t + 1 < len(values) else 0.0
            next_val = 0.0 if dones[t] else (values[t + 1] if t + 1 < len(values) else 0.0)
            delta    = rews[t] + gamma * next_val * (1 - dones[t]) - values[t]
            gae      = delta + gamma * gae_lambda * (1 - dones[t]) * gae
            adv.insert(0, gae)
        ret = [a_ + v for a_, v in zip(adv, values[:len(adv)])]
        advantages[a] = adv
        returns[a]    = ret

    return {
        "obs":        {a: np.array(all_obs[a])      for a in agent_ids},
        "actions":    {a: np.array(all_actions[a])  for a in agent_ids},
        "logprobs":   {a: np.array(all_logprobs[a]) for a in agent_ids},
        "advantages": {a: np.array(advantages[a])   for a in agent_ids},
        "returns":    {a: np.array(returns[a])       for a in agent_ids},
        "joint_obs":  np.array(all_joint),
    }


# ---------------------------------------------------------------------------
# PPO update
# ---------------------------------------------------------------------------

def ppo_update(actors, critic, actor_opts, critic_opt, batch,
               agent_ids, clip_eps=0.2, n_epochs=10, batch_size=256,
               ent_coef=0.01, vf_coef=0.5):

    joint_obs = torch.FloatTensor(batch["joint_obs"])
    n = len(joint_obs)

    for _ in range(n_epochs):
        idx = torch.randperm(n)
        for start in range(0, n, batch_size):
            mb_idx = idx[start:start + batch_size]

            # Update each actor separately
            for a in agent_ids:
                mb_obs  = torch.FloatTensor(batch["obs"][a][mb_idx.numpy()])
                mb_act  = torch.FloatTensor(batch["actions"][a][mb_idx.numpy()])
                mb_lp   = torch.FloatTensor(batch["logprobs"][a][mb_idx.numpy()])
                mb_adv  = torch.FloatTensor(batch["advantages"][a][mb_idx.numpy()])

                # Normalise advantages
                # mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std().clamp(min=1e-6) + 1e-6)

                dist    = actors[a](mb_obs)
                new_lp  = dist.log_prob(mb_act).sum(-1)
                entropy = dist.entropy().sum(-1).mean()

                ratio   = (new_lp - mb_lp).exp()
                surr1   = ratio * mb_adv
                surr2   = ratio.clamp(1 - clip_eps, 1 + clip_eps) * mb_adv
                a_loss  = -torch.min(surr1, surr2).mean() - ent_coef * entropy

                actor_opts[a].zero_grad()
                a_loss.backward()
                nn.utils.clip_grad_norm_(actors[a].parameters(), 0.5)
                actor_opts[a].step()

            # Update critic with all agents' returns
            mb_joint = joint_obs[mb_idx]
            values   = critic(mb_joint)

            # Average the returns across agents for critic target
            critic_targets = []
            for a in agent_ids:
                mb_ret = torch.FloatTensor(batch["returns"][a][mb_idx.numpy()])
                critic_targets.append(mb_ret)
            # critic_targets = torch.stack(critic_targets).mean(dim=0)
            critic_targets = torch.stack(critic_targets).sum(dim=0)

            c_loss = vf_coef * nn.functional.mse_loss(values, critic_targets)
            
            critic_opt.zero_grad()
            c_loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
            critic_opt.step()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(actors, n_episodes=10):
    env = simple_spread_v3.parallel_env(N=3, max_cycles=25, continuous_actions=True)
    agent_ids = list(env.possible_agents)
    episode_rewards = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        ep_rew = {a: 0.0 for a in agent_ids}
        while env.agents:
            actions = {}
            for a in env.agents:
                obs_t  = torch.FloatTensor(obs[a]).unsqueeze(0)
                with torch.no_grad():
                    dist   = actors[a](obs_t)
                    # action = dist.mean.clamp(0.0, 1.0)  # deterministic at eval
                    action = (dist.mean.clamp(-1, 1) + 1) / 2
                actions[a] = action.squeeze(0).numpy()
            obs, rewards, terminations, truncations, _ = env.step(actions)
            for a in agent_ids:
                ep_rew[a] += rewards.get(a, 0.0)
        episode_rewards.append(np.mean(list(ep_rew.values())))

    env.close()
    return np.array(episode_rewards)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

# def train(total_timesteps=1_000_000, n_steps=1024, lr=3e-4):
def train(total_timesteps=TIMESTEPS, n_steps=1024, lr=1e-4):
    run_dir = Path("runs") / "mappo_spread"
    run_dir.mkdir(parents=True, exist_ok=True)

    agent_ids = ["agent_0", "agent_1", "agent_2"]
    obs_dim   = 18
    act_dim   = 5

    # One actor per agent, one shared centralised critic
    actors     = {a: Actor(obs_dim, act_dim)        for a in agent_ids}
    # critic     = CentralCritic(n_agents=3, obs_dim=obs_dim)
    critic = CentralCritic(n_agents=3, obs_dim=obs_dim, hidden=128)
    actor_opts = {
        a: Adam([
            {'params': actors[a].net.parameters(), 'lr': 1e-4},
            {'params': [actors[a].log_std], 'lr': 5e-5},  
        ])
        for a in agent_ids
    }
    critic_opt = Adam(critic.parameters(), lr=lr)

    env = simple_spread_v3.parallel_env(N=3, max_cycles=25, continuous_actions=True)

    n_updates    = total_timesteps // n_steps
    eval_freq    = 10  # evaluate every N updates
    eval_results = []  # list of (timestep, episode_rewards_array)
    timesteps    = []

    print(f"Training MAPPO for {total_timesteps} timesteps ({n_updates} updates)")

    for update in range(n_updates):
        batch = collect_rollout(env, actors, critic, n_steps=n_steps)
        ppo_update(actors, critic, actor_opts, critic_opt, batch, agent_ids)

        if (update + 1) % eval_freq == 0:
            ep_rews = evaluate(actors, n_episodes=10)
            t = (update + 1) * n_steps
            eval_results.append(ep_rews)
            timesteps.append(t)
            print(f"  Update {update+1:4d}/{n_updates} | "
                  f"timestep {t:7d} | "
                  f"mean eval reward: {ep_rews.mean():.3f}")

    env.close()

    # Save in same evaluations.npz format as EvalCallback
    eval_log_dir = run_dir / "eval_logs"
    eval_log_dir.mkdir(exist_ok=True)
    np.savez(
        str(eval_log_dir / "evaluations.npz"),
        timesteps = np.array(timesteps),
        results   = np.array(eval_results),   # (n_evals, n_episodes)
        ep_lengths= np.zeros((len(eval_results), 10)),  # placeholder
    )
    print(f"Saved eval logs → {eval_log_dir}/evaluations.npz")

    # Save actor weights
    for a in agent_ids:
        torch.save(actors[a].state_dict(), str(run_dir / f"{a}_actor.pt"))
    torch.save(critic.state_dict(), str(run_dir / "critic.pt"))
    print(f"Saved models → {run_dir}/")

    return actors, critic


if __name__ == "__main__":
    # train(total_timesteps=1_000_000)
    train(total_timesteps=TIMESTEPS)