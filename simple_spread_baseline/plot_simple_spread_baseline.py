import argparse
import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
MPLCONFIG_DIR = SCRIPT_DIR / ".mplconfig"
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from simple_spread_baseline_common import BASE_DIR

ROOT_DIR = BASE_DIR.parent


def load_npz_evals(npz_path: Path, x_axis: str, metric: str):
    data = np.load(npz_path, allow_pickle=True)

    if x_axis not in data.files:
        available = ", ".join(data.files)
        raise KeyError(f"{npz_path} does not contain '{x_axis}'. Available: {available}")

    x_values = data[x_axis]
    if metric == "mean_return":
        y_values = np.asarray(data["results"], dtype=np.float64).mean(axis=1)
    else:
        y_values = np.asarray(data["ep_lengths"], dtype=np.float64).mean(axis=1)

    return pd.DataFrame({"x": x_values, "value": y_values})


def load_legacy_csv_evals(csv_path: Path):
    df = pd.read_csv(csv_path)
    if "timesteps" not in df.columns or "mean_return" not in df.columns:
        raise ValueError(f"Legacy eval log is missing required columns: {csv_path}")

    # Old logs sometimes have duplicate timestep rows because multiple training
    # attempts appended to the same file. Keep the last entry for each timestep.
    df = df.groupby("timesteps", as_index=False).last()
    return pd.DataFrame({"x": df["timesteps"], "value": df["mean_return"]})


def load_seed_curve(seed: int, x_axis: str, metric: str):
    run_dir = BASE_DIR / "runs" / f"simple_spread_baseline_seed{seed}"
    npz_path = run_dir / "eval_logs" / "evaluations.npz"
    if npz_path.exists():
        return load_npz_evals(npz_path=npz_path, x_axis=x_axis, metric=metric)

    csv_path = run_dir / "eval_log.csv"
    if csv_path.exists():
        if metric != "mean_return":
            raise ValueError(
                "Legacy CSV logs only contain mean returns. Re-run training with "
                "the new script to plot episode length metrics."
            )
        if x_axis != "timesteps":
            raise ValueError(
                "Legacy CSV logs only contain raw SB3 timesteps. Re-run training "
                "with the new script to plot alternate x-axis semantics."
            )
        return load_legacy_csv_evals(csv_path)

    raise FileNotFoundError(
        f"Missing eval logs for seed {seed}: checked {npz_path} and {csv_path}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--x_axis",
        type=str,
        choices=["timesteps", "parallel_env_steps", "per_env_steps"],
        default="timesteps",
    )
    parser.add_argument(
        "--metric",
        type=str,
        choices=["mean_return", "mean_episode_length"],
        default="mean_return",
    )
    parser.add_argument("--no_show", action="store_true")
    parser.add_argument("--output_path", type=str, default=None)
    args = parser.parse_args()

    curves = []
    for seed in args.seeds:
        df = load_seed_curve(seed=seed, x_axis=args.x_axis, metric=args.metric)
        df["seed"] = seed
        curves.append(df)

    all_df = pd.concat(curves, ignore_index=True)
    pivot = all_df.pivot_table(index="x", columns="seed", values="value", aggfunc="last")
    mean_curve = pivot.mean(axis=1)
    std_curve = pivot.std(axis=1, ddof=0)

    results_dir = ROOT_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))

    for seed in args.seeds:
        plt.plot(pivot.index, pivot[seed], alpha=0.5, label=f"seed {seed}")

    plt.plot(pivot.index, mean_curve, linewidth=3, label="mean")
    plt.fill_between(
        pivot.index,
        mean_curve - std_curve,
        mean_curve + std_curve,
        alpha=0.2,
        label="mean ± std",
    )

    x_labels = {
        "timesteps": "SB3 Agent-Slot Timesteps",
        "parallel_env_steps": "Parallel Environment Steps",
        "per_env_steps": "Per-Environment Steps",
    }
    y_labels = {
        "mean_return": "Evaluation Return",
        "mean_episode_length": "Evaluation Episode Length",
    }
    plot_titles = {
        "mean_return": "PPO on Simple Spread",
        "mean_episode_length": "PPO on Simple Spread (Episode Length)",
    }

    plt.xlabel(x_labels[args.x_axis])
    plt.ylabel(y_labels[args.metric])
    plt.title(plot_titles[args.metric])
    plt.legend()
    plt.tight_layout()

    if args.output_path is None:
        if args.metric == "mean_return" and args.x_axis == "timesteps":
            save_path = results_dir / "simple_spread_baseline_3seed_curve.png"
        else:
            save_path = (
                results_dir
                / f"simple_spread_baseline_{args.metric}_{args.x_axis}.png"
            )
    else:
        save_path = Path(args.output_path).expanduser()
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(save_path, dpi=200)
    if not args.no_show:
        plt.show()
    plt.close()

    print(f"Saved figure to: {save_path}")
    print("\nFinal evaluation summary:")
    final_row = pivot.iloc[-1]
    print(final_row)
    print(f"\nFinal mean: {final_row.mean():.2f}")
    print(f"Final std:  {final_row.std(ddof=0):.2f}")


if __name__ == "__main__":
    main()
