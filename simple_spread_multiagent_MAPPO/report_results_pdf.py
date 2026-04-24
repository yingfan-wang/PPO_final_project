from __future__ import annotations

import argparse
import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
MPLCONFIG_DIR = SCRIPT_DIR / ".mplconfig"
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

ROOT_DIR = SCRIPT_DIR.parent
DEFAULT_RUNS_DIR = SCRIPT_DIR / "runs"
DEFAULT_OUTPUT_PATH = (
    ROOT_DIR / "results" / "simple_spread" / "reports" / "simple_spread_multiagent_seed_report.pdf"
)
SMOOTH_WINDOW = 3


def load_npz(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    results = np.asarray(data["results"], dtype=np.float64)
    return {
        "timesteps": np.asarray(data["timesteps"], dtype=np.int64),
        "mean": results.mean(axis=1),
        "std": results.std(axis=1),
        "raw": results,
    }


def smooth(values: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    if len(values) <= window:
        return values, np.arange(len(values))
    kernel = np.ones(window) / window
    smoothed = np.convolve(values, kernel, mode="valid")
    indices = np.arange(window - 1, len(values))
    return smoothed, indices


def load_eval_summary(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--runs_dir", type=str, default=str(DEFAULT_RUNS_DIR))
    parser.add_argument("--output_path", type=str, default=str(DEFAULT_OUTPUT_PATH))
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir).expanduser()
    output_path = Path(args.output_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    loaded: list[dict[str, object]] = []
    for seed in args.seeds:
        run_dir = runs_dir / f"simple_spread_multiagent_seed{seed}"
        npz_path = run_dir / "eval_logs" / "evaluations.npz"
        csv_path = run_dir / "eval_logs" / "eval_metrics.csv"
        if not npz_path.exists():
            raise FileNotFoundError(f"Missing file: {npz_path}")
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing file: {csv_path}")
        loaded.append(
            {
                "label": f"seed {seed}",
                "color": plt.get_cmap("tab10")(seed % 10),
                "seed": seed,
                "data": load_npz(npz_path),
                "eval_df": load_eval_summary(csv_path),
            }
        )

    with PdfPages(output_path) as pdf:
        fig, (ax_raw, ax_sm) = plt.subplots(2, 1, figsize=(11, 8.5))
        fig.suptitle(
            "Structured Multi-Agent PPO — Evaluation Return",
            fontsize=14,
            fontweight="bold",
        )

        for run in loaded:
            data = run["data"]
            timesteps = data["timesteps"]
            color = run["color"]
            ax_raw.plot(timesteps, data["mean"], alpha=0.4, color=color, linewidth=1.0)
            ax_raw.fill_between(
                timesteps,
                data["mean"] - data["std"],
                data["mean"] + data["std"],
                alpha=0.10,
                color=color,
            )

            smoothed, indices = smooth(data["mean"], SMOOTH_WINDOW)
            ax_sm.plot(
                timesteps[indices],
                smoothed,
                color=color,
                linewidth=2.0,
                label=run["label"],
            )

        for ax, title in [
            (ax_raw, "Raw evaluation mean return (± 1 std)"),
            (ax_sm, f"Smoothed evaluation mean return (window={SMOOTH_WINDOW})"),
        ]:
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("Environment Steps")
            ax.set_ylabel("Mean Return")
            ax.grid(True, alpha=0.3)
            ax.xaxis.set_major_formatter(
                matplotlib.ticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}k")
            )
        ax_sm.legend(fontsize=8)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig = plt.figure(figsize=(11, 8.5))
        fig.suptitle(
            "Structured Multi-Agent PPO — Distribution & Stability",
            fontsize=14,
            fontweight="bold",
        )
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)
        ax_hist = fig.add_subplot(gs[0, 0])
        ax_box = fig.add_subplot(gs[0, 1])
        ax_conv = fig.add_subplot(gs[1, :])

        tail_data = []
        tail_labels = []
        tail_colors = []

        for run in loaded:
            data = run["data"]
            color = run["color"]
            label = run["label"]
            tail_checkpoints = max(1, len(data["mean"]) // 4)
            tail = data["raw"][-tail_checkpoints:].flatten()
            ax_hist.hist(tail, bins=30, alpha=0.45, color=color, density=True, label=label)
            tail_data.append(tail)
            tail_labels.append(label)
            tail_colors.append(color)

            if len(data["mean"]) >= SMOOTH_WINDOW:
                rolling_std = [
                    data["mean"][max(0, idx - SMOOTH_WINDOW):idx].std()
                    for idx in range(SMOOTH_WINDOW, len(data["mean"]) + 1)
                ]
                ax_conv.plot(
                    data["timesteps"][SMOOTH_WINDOW - 1:],
                    rolling_std,
                    color=color,
                    linewidth=1.8,
                    label=label,
                )

        ax_hist.set_title("Reward distribution (last 25% of evals)", fontsize=10)
        ax_hist.set_xlabel("Reward")
        ax_hist.set_ylabel("Density")
        ax_hist.legend(fontsize=8)
        ax_hist.grid(True, alpha=0.3)

        bp = ax_box.boxplot(tail_data, patch_artist=True, tick_labels=tail_labels)
        for patch, color in zip(bp["boxes"], tail_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax_box.set_title("Reward spread (last 25% of evals)", fontsize=10)
        ax_box.set_ylabel("Reward")
        ax_box.grid(True, alpha=0.3, axis="y")

        ax_conv.set_title("Rolling std of eval mean return", fontsize=10)
        ax_conv.set_xlabel("Environment Steps")
        ax_conv.set_ylabel("Std")
        ax_conv.grid(True, alpha=0.3)
        ax_conv.legend(fontsize=8)
        ax_conv.xaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}k")
        )

        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(11, 4.5))
        fig.suptitle(
            "Structured Multi-Agent PPO — Final Evaluation Summary",
            fontsize=14,
            fontweight="bold",
        )
        ax.axis("off")

        rows = []
        for run in loaded:
            final_row = run["eval_df"].iloc[-1]
            rows.append(
                [
                    run["label"],
                    f"{int(final_row['env_steps'])/1e3:.0f}k",
                    f"{final_row['mean_episode_return']:.3f}",
                    f"{final_row['std_episode_return']:.3f}",
                    f"{final_row['mean_episode_length']:.2f}",
                    f"{final_row['mean_collision_count']:.2f}",
                    f"{final_row['mean_landmarks_covered']:.2f}",
                    f"{100.0 * final_row['success_near_end_rate']:.1f}%",
                ]
            )

        col_labels = [
            "Run",
            "Total steps",
            "Mean return",
            "Std",
            "Ep length",
            "Collisions",
            "Landmarks covered",
            "Success near end",
        ]
        table = ax.table(cellText=rows, colLabels=col_labels, loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.15, 1.9)
        for idx in range(len(col_labels)):
            table[0, idx].set_facecolor("#37474F")
            table[0, idx].set_text_props(color="white", fontweight="bold")
        for row_idx, run in enumerate(loaded, start=1):
            table[row_idx, 0].set_facecolor(run["color"])
            table[row_idx, 0].set_text_props(color="white", fontweight="bold")

        pdf.savefig(fig)
        plt.close(fig)

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    import matplotlib.ticker

    main()
