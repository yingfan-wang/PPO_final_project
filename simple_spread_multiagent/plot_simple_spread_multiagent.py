"""Plot Simple Spread SB3 adaptation curves, optionally against baseline."""

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

from simple_spread_multiagent_common import BASE_DIR

ROOT_DIR = BASE_DIR.parent
BASELINE_DIR = ROOT_DIR / "simple_spread_baseline"
RUN_PREFIX = "simple_spread_sb3_adaptation"


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


def load_seed_curve(root_dir: Path, run_prefix: str, seed: int, x_axis: str, metric: str):
    npz_path = (
        root_dir
        / "runs"
        / f"{run_prefix}_seed{seed}"
        / "eval_logs"
        / "evaluations.npz"
    )
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing eval log: {npz_path}")
    return load_npz_evals(npz_path=npz_path, x_axis=x_axis, metric=metric)


def build_pivot(
    root_dir: Path,
    run_prefix: str,
    seeds: list[int],
    x_axis: str,
    metric: str,
):
    curves = []
    for seed in seeds:
        df = load_seed_curve(
            root_dir=root_dir,
            run_prefix=run_prefix,
            seed=seed,
            x_axis=x_axis,
            metric=metric,
        )
        df["seed"] = seed
        curves.append(df)

    all_df = pd.concat(curves, ignore_index=True)
    return all_df.pivot_table(index="x", columns="seed", values="value", aggfunc="last")


def plot_pivot(pivot, label_prefix: str, seeds: list[int], alpha: float):
    for seed in seeds:
        plt.plot(
            pivot.index,
            pivot[seed],
            alpha=alpha,
            linewidth=1.5,
            label=f"{label_prefix} seed {seed}",
        )

    mean_curve = pivot.mean(axis=1)
    std_curve = pivot.std(axis=1, ddof=0)
    plt.plot(pivot.index, mean_curve, linewidth=3, label=f"{label_prefix} mean")
    plt.fill_between(
        pivot.index,
        mean_curve - std_curve,
        mean_curve + std_curve,
        alpha=0.15,
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
    parser.add_argument("--compare_baseline", action="store_true")
    parser.add_argument("--no_show", action="store_true")
    parser.add_argument("--output_path", type=str, default=None)
    args = parser.parse_args()

    adaptation_pivot = build_pivot(
        root_dir=BASE_DIR,
        run_prefix=RUN_PREFIX,
        seeds=args.seeds,
        x_axis=args.x_axis,
        metric=args.metric,
    )
    baseline_pivot = None
    if args.compare_baseline:
        baseline_pivot = build_pivot(
            root_dir=BASELINE_DIR,
            run_prefix="simple_spread_baseline",
            seeds=args.seeds,
            x_axis=args.x_axis,
            metric=args.metric,
        )

    results_dir = ROOT_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plot_pivot(adaptation_pivot, "SB3 adaptation", args.seeds, alpha=0.45)
    if baseline_pivot is not None:
        plot_pivot(baseline_pivot, "baseline", args.seeds, alpha=0.25)

    x_labels = {
        "timesteps": "Agent-Slot Timesteps",
        "parallel_env_steps": "Parallel Environment Steps",
        "per_env_steps": "Per-Environment Steps",
    }
    y_labels = {
        "mean_return": "Evaluation Return",
        "mean_episode_length": "Evaluation Episode Length",
    }
    title = "Traced SB3 Adaptation on Simple Spread"
    if args.compare_baseline:
        title = "Traced SB3 Adaptation vs Baseline on Simple Spread"
    if args.metric == "mean_episode_length":
        title += " (Episode Length)"

    plt.xlabel(x_labels[args.x_axis])
    plt.ylabel(y_labels[args.metric])
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    if args.output_path is None:
        if args.compare_baseline:
            save_name = "simple_spread_sb3_adaptation_vs_baseline_3seed_curve.png"
        elif args.metric == "mean_return" and args.x_axis == "timesteps":
            save_name = "simple_spread_sb3_adaptation_3seed_curve.png"
        else:
            save_name = f"simple_spread_sb3_adaptation_{args.metric}_{args.x_axis}.png"
        save_path = results_dir / save_name
    else:
        save_path = Path(args.output_path).expanduser()
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(save_path, dpi=200)
    if not args.no_show:
        plt.show()
    plt.close()

    print(f"Saved figure to: {save_path}")
    print("\nFinal SB3 adaptation evaluation summary:")
    final_row = adaptation_pivot.iloc[-1]
    print(final_row)
    print(f"\nFinal SB3 adaptation mean: {final_row.mean():.2f}")
    print(f"Final SB3 adaptation std:  {final_row.std(ddof=0):.2f}")

    if baseline_pivot is not None:
        baseline_final = baseline_pivot.iloc[-1]
        print("\nFinal baseline evaluation summary:")
        print(baseline_final)
        print(f"\nFinal baseline mean: {baseline_final.mean():.2f}")
        print(f"Final baseline std:  {baseline_final.std(ddof=0):.2f}")


if __name__ == "__main__":
    main()
