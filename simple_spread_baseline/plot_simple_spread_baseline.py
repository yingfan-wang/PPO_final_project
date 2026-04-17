import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()

    curves = []
    for seed in args.seeds:
        csv_path = BASE_DIR / "runs" / f"simple_spread_sb3_seed{seed}" / "eval_log.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing file: {csv_path}")
        df = pd.read_csv(csv_path)
        df["seed"] = seed
        curves.append(df)

    all_df = pd.concat(curves, ignore_index=True)
    pivot = all_df.pivot(index="timesteps", columns="seed", values="mean_return")

    mean_curve = pivot.mean(axis=1)
    std_curve = pivot.std(axis=1)

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

    plt.xlabel("Timesteps")
    plt.ylabel("Evaluation Mean Return")
    plt.title("Simple Spread SB3 PPO Baseline")
    plt.legend()
    plt.tight_layout()

    save_path = results_dir / "simple_spread_sb3_baseline_curve.png"
    plt.savefig(save_path, dpi=200)
    plt.show()

    print(f"Saved figure to: {save_path}")
    print(f"Final mean across seeds: {mean_curve.iloc[-1]:.3f}")
    print(f"Final std across seeds:  {std_curve.iloc[-1]:.3f}")


if __name__ == "__main__":
    main()
