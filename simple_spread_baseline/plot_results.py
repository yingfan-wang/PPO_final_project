from __future__ import annotations

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

ROOT_DIR = SCRIPT_DIR.parent
DEFAULT_RUNS_DIR = SCRIPT_DIR / "runs"
DEFAULT_RESULTS_DIR = ROOT_DIR / "results"


def load_evals(npz_path: Path) -> pd.DataFrame:
    data = np.load(npz_path, allow_pickle=True)
    return pd.DataFrame(
        {
            "env_steps": data["timesteps"],
            "mean_return": data["results"].mean(axis=1),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--runs_dir", type=str, default=str(DEFAULT_RUNS_DIR))
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--no_show", action="store_true")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir).expanduser()
    results_dir = DEFAULT_RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for seed in args.seeds:
        npz_path = (
            runs_dir
            / f"simple_spread_baseline_seed{seed}"
            / "eval_logs"
            / "evaluations.npz"
        )
        if not npz_path.exists():
            raise FileNotFoundError(f"Missing file: {npz_path}")
        frame = load_evals(npz_path)
        frame["seed"] = seed
        frames.append(frame)

    all_df = pd.concat(frames, ignore_index=True)
    pivot = all_df.pivot(index="env_steps", columns="seed", values="mean_return")
    mean_curve = pivot.mean(axis=1)
    std_curve = pivot.std(axis=1).fillna(0.0)

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
    plt.xlabel("Environment Steps")
    plt.ylabel("Evaluation Return")
    plt.title("PPO Baseline on Simple Spread")
    plt.legend()
    plt.tight_layout()

    if args.output_path is None:
        save_path = results_dir / "simple_spread_baseline_3seed_curve.png"
    else:
        save_path = Path(args.output_path).expanduser()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200)
    if not args.no_show:
        plt.show()
    plt.close()

    final_row = pivot.iloc[-1]
    print(f"Saved figure to: {save_path}")
    print("\nFinal evaluation summary:")
    print(final_row)
    print(f"\nFinal mean: {final_row.mean():.2f}")
    final_std = 0.0 if len(final_row) == 1 else float(final_row.std(ddof=1))
    print(f"Final std:  {final_std:.2f}")


if __name__ == "__main__":
    main()
