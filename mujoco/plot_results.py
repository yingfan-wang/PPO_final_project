import argparse
import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
DEFAULT_RUNS_DIR = SCRIPT_DIR / "runs"
DEFAULT_RESULTS_DIR = ROOT_DIR / "results" / "mujoco" / "curves"
MPLCONFIG_DIR = SCRIPT_DIR / ".mplconfig"
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_evals(npz_path: Path):
    """Load SB3 evaluation logs and convert them into one mean-return curve."""

    data = np.load(npz_path, allow_pickle=True)
    timesteps = data["timesteps"]
    results = data["results"]  # shape: [num_evals, n_eval_episodes]
    mean_returns = results.mean(axis=1)
    return pd.DataFrame({"timesteps": timesteps, "mean_return": mean_returns})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", type=str, default="HalfCheetah-v5")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--runs_dir", type=str, default=str(DEFAULT_RUNS_DIR))
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--no_show", action="store_true")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir).expanduser()
    results_dir = DEFAULT_RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    curves = []
    for seed in args.seeds:
        # EvalCallback saves one evaluations.npz per training run.
        npz_path = runs_dir / f"{args.env_id}_seed{seed}" / "eval_logs" / "evaluations.npz"
        if not npz_path.exists():
            raise FileNotFoundError(f"Missing file: {npz_path}")
        df = load_evals(npz_path)
        df["seed"] = seed
        curves.append(df)

    all_df = pd.concat(curves, ignore_index=True)

    # Align runs by timestep so we can compute per-timestep mean and std
    # across different random seeds.
    pivot = all_df.pivot(index="timesteps", columns="seed", values="mean_return")
    mean_curve = pivot.mean(axis=1)
    std_curve = pivot.std(axis=1)

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

    plt.xlabel("Timesteps")
    plt.ylabel("Evaluation Return")
    plt.title(f"PPO on {args.env_id}")
    plt.legend()
    plt.tight_layout()

    if args.output_path is None:
        save_path = results_dir / f"{args.env_id}_3seed_curve.png"
    else:
        save_path = Path(args.output_path).expanduser()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200)
    if not args.no_show:
        plt.show()
    plt.close()

    print(f"Saved figure to: {save_path}")
    print("\nFinal evaluation summary:")
    # The last row corresponds to the final saved evaluation for each seed.
    final_row = pivot.iloc[-1]
    print(final_row)
    print(f"\nFinal mean: {final_row.mean():.2f}")
    print(f"Final std:  {final_row.std():.2f}")


if __name__ == "__main__":
    main()
