"""
test_and_plot.py
----------------
Evaluates all 5 trained runs by running their final models cooperatively,
broken down by seed (1, 2, 3). Produces a PDF with per-seed and aggregate stats.

Usage:
    python test_and_plot.py
Output:
    test_results.pdf
"""

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from pettingzoo.mpe import simple_spread_v3
from stable_baselines3 import PPO


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

N_EPISODES   = 50
SEEDS        = [1, 2, 3]
SMOOTH_WINDOW = 5
OUTPUT_PDF   = "test_results.pdf"

RUNS = [
    {
        "label": "Parameter Sharing",
        "color": "#2196F3",
        "mode":  "shared",
        "model_path": "runs/simple_spread/final_model",
    },
    {
        "label": "Round-Robin Multi-Agent",
        "color": "#4CAF50",
        "mode":  "multi",
        "model_paths": {
            "agent_0": "runs/round_robin/agent_0_final",
            "agent_1": "runs/round_robin/agent_1_final",
            "agent_2": "runs/round_robin/agent_2_final",
        },
    },
    {
        "label": "IPPO",
        "color": "#F44336",
        "mode":  "multi",
        "model_paths": {
            "agent_0": "runs/multiagent_spread/agent_0_final",
            "agent_1": "runs/multiagent_spread/agent_1_final",
            "agent_2": "runs/multiagent_spread/agent_2_final",
        },
    },
    {
        "label": "Joint Observation",
        "color": "#9C27B0",
        "mode":  "joint",
        "model_paths": {
            "agent_0": "runs/joint_obs_spread/agent_0_final",
            "agent_1": "runs/joint_obs_spread/agent_1_final",
            "agent_2": "runs/joint_obs_spread/agent_2_final",
        },
    },
    {
        "label": "MAPPO",
        "color": "#00BCD4",
        "mode":  "mappo",
        "actor_paths": {
            "agent_0": "runs/mappo_spread/agent_0_actor.pt",
            "agent_1": "runs/mappo_spread/agent_1_actor.pt",
            "agent_2": "runs/mappo_spread/agent_2_actor.pt",
        },
    },
    # {
    #     "label": "MAPPO (True)",
    #     "mode": "mappo",
    #     "actor_paths": {
    #         "agent_0": "runs/mappo_TRUE_baseline/agent_0_actor.pt",
    #         "agent_1": "runs/mappo_TRUE_baseline/agent_1_actor.pt",
    #         "agent_2": "runs/mappo_TRUE_baseline/agent_2_actor.pt",
    #     },
    # }
]


# ---------------------------------------------------------------------------
# MAPPO actor (must match train_mappo.py architecture)
# ---------------------------------------------------------------------------

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
        mean = torch.nan_to_num(mean, nan=0.0, posinf=1.0, neginf=-1.0)
        log_std = torch.clamp(self.log_std, min=-4.0, max=0.5)
        std = log_std.exp().expand_as(mean)
        return torch.distributions.Normal(mean, std)


# ---------------------------------------------------------------------------
# EPISODE RUNNERS
# ---------------------------------------------------------------------------

def run_episode_shared(model, env):
    obs, _ = env.reset()
    total = 0.0
    while env.agents:
        actions = {}
        for a in env.agents:
            action, _ = model.predict(obs[a][np.newaxis], deterministic=True)
            actions[a] = action[0]
        obs, rewards, _, _, _ = env.step(actions)
        total += np.mean(list(rewards.values())) if rewards else 0.0
    return total


def run_episode_multi(models, env):
    obs, _ = env.reset()
    total = 0.0
    while env.agents:
        actions = {}
        for a in env.agents:
            action, _ = models[a].predict(obs[a][np.newaxis], deterministic=True)
            actions[a] = action[0]
        obs, rewards, _, _, _ = env.step(actions)
        total += np.mean(list(rewards.values())) if rewards else 0.0
    return total


def run_episode_joint(models, agent_ids, env):
    obs, _ = env.reset()
    last_obs = dict(obs)
    total = 0.0
    while env.agents:
        actions = {}
        for a in env.agents:
            joint = np.concatenate(
                [last_obs[a]] + [last_obs[x] for x in agent_ids if x != a],
                dtype=np.float32
            )
            action, _ = models[a].predict(joint[np.newaxis], deterministic=True)
            actions[a] = action[0]
        obs, rewards, _, _, _ = env.step(actions)
        last_obs = {**last_obs, **obs}
        total += np.mean(list(rewards.values())) if rewards else 0.0
    return total


def run_episode_mappo(actors, env):
    obs, _ = env.reset()
    total = 0.0
    while env.agents:
        actions = {}
        for a in env.agents:
            obs_t = torch.FloatTensor(obs[a]).unsqueeze(0)
            with torch.no_grad():
                dist = actors[a](obs_t)
                action = (dist.mean.clamp(-1, 1) + 1) / 2
            actions[a] = action.squeeze(0).numpy()
        obs, rewards, _, _, _ = env.step(actions)
        total += np.mean(list(rewards.values())) if rewards else 0.0
    return total


# ---------------------------------------------------------------------------
# LOAD MODELS
# ---------------------------------------------------------------------------

def load_models(run: dict):
    mode = run["mode"]
    if mode == "shared":
        p = Path(run["model_path"] + ".zip")
        if not p.exists():
            print(f"  Missing: {p}"); return None
        return PPO.load(run["model_path"])

    elif mode in ("multi", "joint"):
        models = {}
        for a, path in run["model_paths"].items():
            p = Path(path + ".zip")
            if not p.exists():
                print(f"  Missing: {p}"); return None
            models[a] = PPO.load(path)
        return models

    elif mode == "mappo":
        actors = {}
        for a, path in run["actor_paths"].items():
            p = Path(path)
            if not p.exists():
                print(f"  Missing: {p}"); return None
            actor = Actor()
            actor.load_state_dict(torch.load(path, map_location="cpu"))
            actor.eval()
            actors[a] = actor
        return actors


# ---------------------------------------------------------------------------
# EVALUATE PER SEED
# ---------------------------------------------------------------------------

def evaluate_run(run: dict, n_episodes: int, seeds: list) -> dict | None:
    mode      = run["mode"]
    agent_ids = ["agent_0", "agent_1", "agent_2"]

    models = load_models(run)
    if models is None:
        return None

    per_seed = {}  # {seed: np.array of episode rewards}

    for seed in seeds:
        print(f"\n  Seed {seed}...")
        env = simple_spread_v3.parallel_env(N=3, max_cycles=25, continuous_actions=True)
        env.reset(seed=seed)  # seed the RNG

        seed_rewards = []
        for ep in range(n_episodes):
            if mode == "shared":
                r = run_episode_shared(models, env)
            elif mode == "multi":
                r = run_episode_multi(models, env)
            elif mode == "joint":
                r = run_episode_joint(models, agent_ids, env)
            elif mode == "mappo":
                r = run_episode_mappo(models, env)

            seed_rewards.append(r)
            print(f"    ep {ep+1:3d}/{n_episodes} | reward: {r:8.3f} | "
                  f"seed avg: {np.mean(seed_rewards):8.3f}")

        env.close()
        arr = np.array(seed_rewards)
        per_seed[seed] = arr
        print(f"  Seed {seed} summary — mean: {arr.mean():.3f}  "
              f"std: {arr.std():.3f}  max: {arr.max():.3f}  min: {arr.min():.3f}")

    all_rewards = np.concatenate(list(per_seed.values()))
    return {
        "per_seed":   per_seed,
        "rewards":    all_rewards,
        "mean":       float(all_rewards.mean()),
        "std":        float(all_rewards.std()),
        "max":        float(all_rewards.max()),
        "min":        float(all_rewards.min()),
        "seed_means": {s: float(v.mean()) for s, v in per_seed.items()},
        "seed_stds":  {s: float(v.std())  for s, v in per_seed.items()},
        "seed_maxes": {s: float(v.max())  for s, v in per_seed.items()},
        "seed_mins":  {s: float(v.min())  for s, v in per_seed.items()},
    }


# ---------------------------------------------------------------------------
# SMOOTH HELPER
# ---------------------------------------------------------------------------

def smooth(values, window):
    if len(values) <= window:
        return values, np.arange(len(values))
    kernel = np.ones(window) / window
    s      = np.convolve(values, kernel, mode="valid")
    idx    = np.arange(window - 1, len(values))
    return s, idx


# ---------------------------------------------------------------------------
# PLOT
# ---------------------------------------------------------------------------

def make_pdf(loaded: list[dict]):
    with PdfPages(OUTPUT_PDF) as pdf:

        # ── Page 1: Raw + smoothed curves, one line per seed ─────────────────
        fig, (ax_raw, ax_sm) = plt.subplots(2, 1, figsize=(11, 8.5))
        fig.suptitle("Test Evaluation — Episode Reward per Seed",
                     fontsize=14, fontweight="bold")

        linestyles = ["solid", "dashed", "dotted"]

        for run in loaded:
            d = run["data"]
            for i, (seed, rewards) in enumerate(d["per_seed"].items()):
                eps = np.arange(1, len(rewards) + 1)
                ax_raw.plot(eps, rewards, alpha=0.2, color=run["color"],
                            linewidth=0.8, linestyle=linestyles[i])
                s, idx = smooth(rewards, SMOOTH_WINDOW)
                ax_sm.plot(eps[idx], s, alpha=0.5, color=run["color"],
                           linewidth=1.2, linestyle=linestyles[i],
                           label=f"{run['label']} s{seed}")

            # Bold aggregate line
            agg = d["rewards"]
            eps_agg = np.arange(1, len(agg) + 1)
            ax_raw.plot(eps_agg, agg, alpha=0.6, color=run["color"], linewidth=1.5)
            s_agg, idx_agg = smooth(agg, SMOOTH_WINDOW)
            ax_sm.plot(eps_agg[idx_agg], s_agg, color=run["color"],
                       linewidth=2.5, label=f"{run['label']} (all)")

        for ax, title in [
            (ax_raw, "Raw episode reward (faint lines = individual seeds)"),
            (ax_sm,  f"Smoothed reward — dashed/dotted = seeds, bold = all seeds combined"),
        ]:
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("Episode")
            ax.set_ylabel("Total reward")
            ax.grid(True, alpha=0.3)

        ax_sm.legend(fontsize=6, ncol=2)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # ── Page 2: Per-seed bar charts side by side ─────────────────────────
        fig, axes = plt.subplots(1, len(SEEDS), figsize=(11, 5), sharey=True)
        fig.suptitle("Test Evaluation — Mean Reward per Seed",
                     fontsize=14, fontweight="bold")

        for ax, seed in zip(axes, SEEDS):
            labels = [r["label"] for r in loaded]
            means  = [r["data"]["seed_means"][seed] for r in loaded]
            stds   = [r["data"]["seed_stds"][seed]  for r in loaded]
            colors = [r["color"] for r in loaded]
            x      = np.arange(len(labels))

            bars = ax.bar(x, means, yerr=stds, color=colors, alpha=0.75,
                          capsize=5, error_kw={"linewidth": 1.2})
            ax.set_title(f"Seed {seed}", fontsize=11)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=6, rotation=15, ha="right")
            ax.grid(True, alpha=0.3, axis="y")
            if ax == axes[0]:
                ax.set_ylabel("Mean reward")

            for bar, mean in zip(bars, means):
                offset = 0.2 if mean >= 0 else -1.5
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + offset,
                        f"{mean:.1f}", ha="center", va="bottom", fontsize=7)

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # ── Page 3: Distribution + aggregate bar ─────────────────────────────
        fig = plt.figure(figsize=(11, 8.5))
        fig.suptitle("Test Evaluation — Distribution & Aggregate",
                     fontsize=14, fontweight="bold")
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

        ax_hist = fig.add_subplot(gs[0, 0])
        ax_box  = fig.add_subplot(gs[0, 1])
        ax_bar  = fig.add_subplot(gs[1, :])

        box_data, box_labels, box_colors = [], [], []

        for run in loaded:
            rewards = run["data"]["rewards"]
            lbl     = run["label"]
            ax_hist.hist(rewards, bins=20, alpha=0.45,
                         color=run["color"], label=lbl, density=True)
            box_data.append(rewards)
            box_labels.append(lbl)
            box_colors.append(run["color"])

        ax_hist.set_title("Reward distribution\n(all seeds combined)", fontsize=10)
        ax_hist.set_xlabel("Reward")
        ax_hist.set_ylabel("Density")
        ax_hist.legend(fontsize=7)
        ax_hist.grid(True, alpha=0.3)

        bp = ax_box.boxplot(box_data, patch_artist=True, labels=box_labels)
        for patch, color in zip(bp["boxes"], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax_box.set_title("Reward spread\n(all seeds combined)", fontsize=10)
        ax_box.set_ylabel("Reward")
        ax_box.tick_params(axis="x", labelsize=7)
        ax_box.grid(True, alpha=0.3, axis="y")

        labels = [r["label"] for r in loaded]
        means  = [r["data"]["mean"] for r in loaded]
        stds   = [r["data"]["std"]  for r in loaded]
        colors = [r["color"] for r in loaded]
        x      = np.arange(len(labels))

        bars = ax_bar.bar(x, means, yerr=stds, color=colors, alpha=0.75,
                          capsize=6, error_kw={"linewidth": 1.5})
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(labels, fontsize=8)
        ax_bar.set_title(
            f"Aggregate mean ± std ({len(SEEDS)} seeds × {N_EPISODES} episodes)",
            fontsize=10)
        ax_bar.set_ylabel("Mean reward")
        ax_bar.grid(True, alpha=0.3, axis="y")

        for bar, mean in zip(bars, means):
            offset = 0.3 if mean >= 0 else -1.5
            ax_bar.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + offset,
                        f"{mean:.2f}", ha="center", va="bottom", fontsize=8)

        pdf.savefig(fig)
        plt.close(fig)

        # ── Page 4: Full summary table with per-seed breakdown ───────────────
        fig, ax = plt.subplots(figsize=(14, 5))
        fig.suptitle(
            f"Test Evaluation — Summary ({len(SEEDS)} seeds × {N_EPISODES} episodes)",
            fontsize=14, fontweight="bold")
        ax.axis("off")

        col_labels = (
            ["Run", "Overall\nmean", "Overall\nstd", "Overall\nmax", "Overall\nmin"]
            + [f"Seed {s}\nmean" for s in SEEDS]
            + [f"Seed {s}\nstd"  for s in SEEDS]
            + [f"Seed {s}\nmax"  for s in SEEDS]
            + [f"Seed {s}\nmin"  for s in SEEDS]
        )

        rows = []
        for run in loaded:
            d = run["data"]
            row = [
                run["label"],
                f"{d['mean']:.3f}",
                f"{d['std']:.3f}",
                f"{d['max']:.3f}",
                f"{d['min']:.3f}",
            ]
            for s in SEEDS:
                row.append(f"{d['seed_means'][s]:.3f}")
            for s in SEEDS:
                row.append(f"{d['seed_stds'][s]:.3f}")
            for s in SEEDS:
                row.append(f"{d['seed_maxes'][s]:.3f}")
            for s in SEEDS:
                row.append(f"{d['seed_mins'][s]:.3f}")
            rows.append(row)

        tbl = ax.table(cellText=rows, colLabels=col_labels,
                       loc="center", cellLoc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(7)
        tbl.scale(1.1, 2.0)

        for j in range(len(col_labels)):
            tbl[0, j].set_facecolor("#37474F")
            tbl[0, j].set_text_props(color="white", fontweight="bold")
        for i, run in enumerate(loaded):
            tbl[i + 1, 0].set_facecolor(run["color"])
            tbl[i + 1, 0].set_text_props(color="white", fontweight="bold")

        pdf.savefig(fig)
        plt.close(fig)

    print(f"\nSaved: {OUTPUT_PDF}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    loaded = []
    for run in RUNS:
        print(f"\n{'='*50}")
        print(f"  Evaluating: {run['label']}")
        print(f"{'='*50}")
        data = evaluate_run(run, N_EPISODES, SEEDS)
        if data:
            loaded.append({**run, "data": data})
            print(f"\n  Overall — mean: {data['mean']:.3f}  std: {data['std']:.3f}  "
                  f"max: {data['max']:.3f}  min: {data['min']:.3f}")
            for s in SEEDS:
                print(f"  Seed {s}   — mean: {data['seed_means'][s]:.3f}  "
                      f"std: {data['seed_stds'][s]:.3f}  "
                      f"max: {data['seed_maxes'][s]:.3f}  "
                      f"min: {data['seed_mins'][s]:.3f}")
        else:
            print(f"  Skipped (missing model files)")

    if not loaded:
        print("\nNo models found.")
    else:
        print(f"\nGenerating {OUTPUT_PDF} with {len(loaded)} run(s)...")
        make_pdf(loaded)