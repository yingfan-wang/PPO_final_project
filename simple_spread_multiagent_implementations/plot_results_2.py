"""
plot_results.py
---------------
Reads evaluations.npz from EvalCallback logs and exports a standardized
comparison PDF across all runs.

Works with:
  - runs/simple_spread/eval_logs/evaluations.npz
  - runs/multiagent_spread/agent_0/eval_logs/evaluations.npz  (after retrain)
  - runs/multiagent_spread/agent_1/eval_logs/evaluations.npz
  - runs/multiagent_spread/agent_2/eval_logs/evaluations.npz

Usage:
    python plot_results.py
Output:
    results.pdf
"""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages


# ---------------------------------------------------------------------------
# CONFIG — add/remove/rename runs here, nothing else needs to change
# ---------------------------------------------------------------------------

# RUNS = [
#     {
#         "label": "Shared Policy",
#         "color": "#2196F3",
#         "npz": "runs/simple_spread/eval_logs/evaluations.npz",
#     },
#     {
#         "label": "Multi PPO agent_0",
#         "color": "#F44336",
#         "npz": "runs/multiagent_spread/agent_0/eval_logs/evaluations.npz",
#     },
#     {
#         "label": "Multi PPO agent_1",
#         "color": "#4CAF50",
#         "npz": "runs/multiagent_spread/agent_1/eval_logs/evaluations.npz",
#     },
#     {
#         "label": "Multi PPO agent_2",
#         "color": "#FF9800",
#         "npz": "runs/multiagent_spread/agent_2/eval_logs/evaluations.npz",
#     },
#     {
#         "label": "Joint Obs agent_0",  # ← added
#         "color": "#9C27B0",
#         "npz": "runs/joint_obs_spread/agent_0/eval_logs/evaluations.npz",
#     },
#     {
#         "label": "Joint Obs agent_1",  # ← added
#         "color": "#E91E63",
#         "npz": "runs/joint_obs_spread/agent_1/eval_logs/evaluations.npz",
#     },
#     {
#         "label": "Joint Obs agent_2",  # ← added
#         "color": "#607D8B",
#         "npz": "runs/joint_obs_spread/agent_2/eval_logs/evaluations.npz",
#     },
# ]
# instead of plotting agent_0, agent_1, agent_2 separately
# average them into one "IPPO" line and one "Joint" line

# RUNS = [
#     {
#         "label": "Shared Policy",
#         "color": "#2196F3",
#         "npz_list": ["runs/simple_spread/eval_logs/evaluations.npz"],
#     },
#     {
#         "label": "IPPO",
#         "color": "#F44336",
#         "npz_list": [
#             "runs/multiagent_spread/agent_0/eval_logs/evaluations.npz",
#             "runs/multiagent_spread/agent_1/eval_logs/evaluations.npz",
#             "runs/multiagent_spread/agent_2/eval_logs/evaluations.npz",
#         ],
#     },
#     {
#         "label": "Joint Obs",
#         "color": "#9C27B0",
#         "npz_list": [
#             "runs/joint_obs_spread/agent_0/eval_logs/evaluations.npz",
#             "runs/joint_obs_spread/agent_1/eval_logs/evaluations.npz",
#             "runs/joint_obs_spread/agent_2/eval_logs/evaluations.npz",
#         ],
#     },
#     {
#         "label": "MAPPO",
#         "color": "#00BCD4",
#         "npz_list": ["runs/mappo_spread/eval_logs/evaluations.npz"],
#     },
# ]
RUNS = [
    {
        "label": "Parameter Sharing",
        "color": "#2196F3",
        "npz_list": [
            "runs/simple_spread/eval_logs/evaluations.npz",
        ],
    },
    {
        "label": "Round-Robin",
        "color": "#4CAF50",
        "npz_list": [
            "runs/round_robin/agent_0/eval_logs/evaluations.npz",
            "runs/round_robin/agent_1/eval_logs/evaluations.npz",
            "runs/round_robin/agent_2/eval_logs/evaluations.npz",
        ],
    },
    {
        "label": "IPPO",
        "color": "#F44336",
        "npz_list": [
            "runs/multiagent_spread/agent_0/eval_logs/evaluations.npz",
            "runs/multiagent_spread/agent_1/eval_logs/evaluations.npz",
            "runs/multiagent_spread/agent_2/eval_logs/evaluations.npz",
        ],
    },
    {
        "label": "Joint Observation",
        "color": "#9C27B0",
        "npz_list": [
            "runs/joint_obs_spread/agent_0/eval_logs/evaluations.npz",
            "runs/joint_obs_spread/agent_1/eval_logs/evaluations.npz",
            "runs/joint_obs_spread/agent_2/eval_logs/evaluations.npz",
        ],
    },
    {
        "label": "MAPPO",
        "color": "#00BCD4",
        "npz_list": [
            "runs/mappo_spread/eval_logs/evaluations.npz",
        ],
    },
    # {
    #     "label": "MAPPO (True)",
    #     "color": "#00BCD4",
    #     "npz_list": [
    #         "runs/mappo_TRUE_baseline/eval_logs/evaluations.npz",
    #     ],
    # }
    {
        "label": "Advanced MAPPO",
        "color": "#00BCD4",
        "npz_list": [
            "runs/mappo_controller_seed012_evaluations.npz",
        ],
    },
]

SMOOTH_WINDOW = 5      # over eval checkpoints (not raw episodes)
OUTPUT_PDF    = "results.pdf"


# ---------------------------------------------------------------------------
# LOAD
# ---------------------------------------------------------------------------

# def load_npz(path: str) -> dict | None:
#     p = Path(path)
#     if not p.exists():
#         print(f"  Missing: {path}")
#         return None
#     data = np.load(p)
#     # EvalCallback saves: timesteps, results (n_evals x n_episodes), ep_lengths
#     timesteps = data["timesteps"]                  # (n_evals,)
#     results   = data["results"]                    # (n_evals, n_episodes)
#     mean_r    = results.mean(axis=1)               # (n_evals,)
#     std_r     = results.std(axis=1)
#     return {"timesteps": timesteps, "mean": mean_r, "std": std_r, "raw": results}
    
def load_npz(run: dict) -> dict | None:
    all_means = []
    for path in run["npz_list"]:
        p = Path(path)
        if not p.exists():
            print(f"  Missing: {path}")
            continue
        data = np.load(p)
        all_means.append(data["results"].mean(axis=1))

    if not all_means:
        return None

    min_len = min(len(m) for m in all_means)
    stacked = np.stack([m[:min_len] for m in all_means])
    mean    = stacked.mean(axis=0)
    std     = stacked.std(axis=0)
    x       = np.linspace(0, 100, min_len)  # normalized x axis

    return {**run, "x": x, "mean": mean, "std": std}

def smooth(values, window):
    if len(values) <= window:
        return values, np.arange(len(values))
    kernel  = np.ones(window) / window
    s       = np.convolve(values, kernel, mode="valid")
    idx     = np.arange(window - 1, len(values))
    return s, idx


# ---------------------------------------------------------------------------
# PLOT
# ---------------------------------------------------------------------------

def make_pdf(loaded: list[dict]):
    with PdfPages(OUTPUT_PDF) as pdf:

        # ── Page 1: Mean reward over timesteps ─────────────────────────────
        fig, (ax_raw, ax_sm) = plt.subplots(2, 1, figsize=(11, 8.5))
        fig.suptitle("PPO Evaluation — Mean Episode Reward", fontsize=14, fontweight="bold")

        for run in loaded:
            d  = run["data"]
            # ts = d["timesteps"]
            ts = d["x"]
            ax_raw.plot(ts, d["mean"], alpha=0.35, color=run["color"], linewidth=1.0)
            ax_raw.fill_between(ts,
                                d["mean"] - d["std"],
                                d["mean"] + d["std"],
                                alpha=0.08, color=run["color"])

            s, idx = smooth(d["mean"], SMOOTH_WINDOW)
            ax_sm.plot(ts[idx], s, label=run["label"].replace("\n", " "),
                       color=run["color"], linewidth=2.0)

        for ax, title in [
            (ax_raw, "Raw eval mean reward (± 1 std)"),
            (ax_sm,  f"Smoothed eval mean reward (window={SMOOTH_WINDOW} checkpoints)"),
        ]:
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("Timesteps")
            ax.set_ylabel("Mean reward")
            ax.grid(True, alpha=0.3)
            ax.xaxis.set_major_formatter(
                matplotlib.ticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}k")
            )

        ax_sm.legend(fontsize=8)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # ── Page 2: Reward distribution + convergence ───────────────────────
        fig = plt.figure(figsize=(11, 8.5))
        fig.suptitle("PPO Evaluation — Distribution & Convergence", fontsize=14, fontweight="bold")
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

        ax_hist = fig.add_subplot(gs[0, 0])
        ax_box  = fig.add_subplot(gs[0, 1])
        ax_conv = fig.add_subplot(gs[1, :])

        tail_labels, tail_data, tail_colors = [], [], []

        for run in loaded:
            d    = run["data"]
            # Last 25% of eval checkpoints = converged behaviour
            n    = max(1, len(d["mean"]) // 4)
            # tail = d["raw"][-n:].flatten()
            tail = d["mean"][-n:]
            lbl  = run["label"].replace("\n", " ")

            ax_hist.hist(tail, bins=30, alpha=0.45, color=run["color"],
                         label=lbl, density=True)

            tail_labels.append(run["label"])
            tail_data.append(tail)
            tail_colors.append(run["color"])

            # Convergence: rolling std of mean reward (lower = more stable)
            if len(d["mean"]) >= SMOOTH_WINDOW:
                roll_std = [d["mean"][max(0,i-SMOOTH_WINDOW):i].std()
                            for i in range(SMOOTH_WINDOW, len(d["mean"]) + 1)]
                # ax_conv.plot(d["timesteps"][SMOOTH_WINDOW - 1:], roll_std,
                ax_conv.plot(d["x"][SMOOTH_WINDOW - 1:], roll_std,
                             label=lbl, color=run["color"], linewidth=1.8)

        ax_hist.set_title("Reward distribution\n(last 25% of evals)", fontsize=10)
        ax_hist.set_xlabel("Reward")
        ax_hist.set_ylabel("Density")
        ax_hist.legend(fontsize=7)
        ax_hist.grid(True, alpha=0.3)

        bp = ax_box.boxplot(tail_data, patch_artist=True, labels=tail_labels)
        for patch, color in zip(bp["boxes"], tail_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax_box.set_title("Reward spread\n(last 25% of evals)", fontsize=10)
        ax_box.set_ylabel("Reward")
        ax_box.tick_params(axis="x", labelsize=7)
        ax_box.grid(True, alpha=0.3, axis="y")

        ax_conv.set_title(f"Rolling reward std (window={SMOOTH_WINDOW}) — lower = more stable",
                          fontsize=10)
        ax_conv.set_xlabel("Timesteps")
        ax_conv.set_ylabel("Std of mean reward")
        ax_conv.legend(fontsize=8)
        ax_conv.grid(True, alpha=0.3)
        ax_conv.xaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}k")
        )

        pdf.savefig(fig)
        plt.close(fig)

        # ── Page 3: Summary stats table ─────────────────────────────────────
        fig, ax = plt.subplots(figsize=(11, 4))
        fig.suptitle("PPO Evaluation — Summary Statistics (last 25% of evals)",
                     fontsize=14, fontweight="bold")
        ax.axis("off")

        rows = []
        for run in loaded:
            d    = run["data"]
            n    = max(1, len(d["mean"]) // 4)
            # tail = d["raw"][-n:].flatten()
            tail = d["mean"][-n:]
            rows.append([
                run["label"].replace("\n", " "),
                # f"{d['timesteps'][-1]/1e3:.0f}k",
                f"{d['x'][-1]/1e3:.0f}k",
                f"{len(d['mean'])}",
                f"{tail.mean():.3f}",
                f"{tail.std():.3f}",
                f"{tail.max():.3f}",
                f"{tail.min():.3f}",
            ])

        col_labels = ["Run", "Total steps", "Eval checkpoints",
                      "Mean reward", "Std", "Max", "Min"]
        tbl = ax.table(cellText=rows, colLabels=col_labels,
                       loc="center", cellLoc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1.2, 1.9)

        for j in range(len(col_labels)):
            tbl[0, j].set_facecolor("#37474F")
            tbl[0, j].set_text_props(color="white", fontweight="bold")
        for i, run in enumerate(loaded):
            tbl[i + 1, 0].set_facecolor(run["color"])
            tbl[i + 1, 0].set_text_props(color="white", fontweight="bold")

        pdf.savefig(fig)
        plt.close(fig)

    print(f"Saved: {OUTPUT_PDF}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import matplotlib.ticker  # needed for formatter inside make_pdf

    print("Loading runs...")
    loaded = []
    for run in RUNS:
        print(f"  {run['label'].replace(chr(10), ' ')}...")
        # data = load_npz(run["npz"])
        data = load_npz(run)
        if data:
            loaded.append({**run, "data": data})
            # print(f"    {len(data['timesteps'])} eval checkpoints, "
            print(f"    {len(data['x'])} eval checkpoints, "
                  f"final mean reward: {data['mean'][-1]:.3f}")

    if not loaded:
        print("\nNo data found. Train at least one run first.")
    else:
        print(f"\nGenerating {OUTPUT_PDF} with {len(loaded)} run(s)...")
        make_pdf(loaded)