import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_evals(npz_path: Path):
    data = np.load(npz_path, allow_pickle=True)
    timesteps = data["timesteps"]
    results = data["results"]  # shape: [num_evals, n_eval_episodes]
    mean_returns = results.mean(axis=1)
    return pd.DataFrame({"timesteps": timesteps, "mean_return": mean_returns})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", type=str, default="HalfCheetah-v5")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    curves = []
    for seed in args.seeds:
        npz_path = Path("runs") / f"{args.env_id}_seed{seed}" / "eval_logs" / "evaluations.npz"
        if not npz_path.exists():
            raise FileNotFoundError(f"Missing file: {npz_path}")
        df = load_evals(npz_path)
        df["seed"] = seed
        curves.append(df)

    all_df = pd.concat(curves, ignore_index=True)

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

    save_path = results_dir / f"{args.env_id}_3seed_curve.png"
    plt.savefig(save_path, dpi=200)
    plt.show()

    print(f"Saved figure to: {save_path}")
    print("\nFinal evaluation summary:")
    final_row = pivot.iloc[-1]
    print(final_row)
    print(f"\nFinal mean: {final_row.mean():.2f}")
    print(f"Final std:  {final_row.std():.2f}")


if __name__ == "__main__":
    main()