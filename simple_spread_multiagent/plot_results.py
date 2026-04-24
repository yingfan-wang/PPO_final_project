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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT_DIR = SCRIPT_DIR.parent
DEFAULT_RUNS_DIR = SCRIPT_DIR / "runs"
DEFAULT_RESULTS_DIR = ROOT_DIR / "results" / "simple_spread" / "curves"
DEFAULT_EXPORT_NPZ_PATH = ROOT_DIR / "runs" / "mappo_controller_spread" / "eval_logs" / "evaluations.npz"


def load_npz(npz_path: Path) -> dict[str, np.ndarray]:
    data = np.load(npz_path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def load_evals(npz_path: Path) -> pd.DataFrame:
    data = load_npz(npz_path)
    return pd.DataFrame(
        {
            "env_steps": data["timesteps"],
            "mean_return": data["results"].mean(axis=1),
        }
    )


def aggregate_seed_eval_logs(
    npz_paths: list[Path],
    seeds: list[int] | None = None,
) -> dict[str, np.ndarray]:
    if not npz_paths:
        raise ValueError("Expected at least one evaluations.npz path to aggregate.")

    payloads = [load_npz(path) for path in npz_paths]
    reference_steps = payloads[0]["timesteps"]
    for path, payload in zip(npz_paths[1:], payloads[1:]):
        if not np.array_equal(payload["timesteps"], reference_steps):
            raise ValueError(f"Mismatched timesteps in {path}")

    if seeds is None:
        seeds = list(range(len(npz_paths)))
    if len(seeds) != len(npz_paths):
        raise ValueError("Number of seeds must match number of evaluation files.")

    aggregated: dict[str, np.ndarray] = {
        "timesteps": reference_steps,
        "env_steps": payloads[0].get("env_steps", reference_steps),
        "agent_steps": payloads[0].get("agent_steps", reference_steps),
        "results": np.concatenate([payload["results"] for payload in payloads], axis=1),
        "ep_lengths": np.concatenate([payload["ep_lengths"] for payload in payloads], axis=1),
        "seeds": np.asarray(seeds, dtype=np.int64),
        "seed_mean_returns": np.stack(
            [payload["results"].mean(axis=1) for payload in payloads],
            axis=1,
        ),
    }

    for metric in [
        "collision_counts",
        "avg_landmarks_covered",
        "success_near_end",
        "mean_sum_min_dists",
    ]:
        if all(metric in payload for payload in payloads):
            aggregated[metric] = np.concatenate(
                [payload[metric] for payload in payloads],
                axis=1,
            )

    return aggregated


def save_aggregated_eval_npz(
    npz_paths: list[Path],
    output_path: Path,
    seeds: list[int] | None = None,
) -> Path:
    aggregated = aggregate_seed_eval_logs(npz_paths, seeds=seeds)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **aggregated)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--runs_dir", type=str, default=str(DEFAULT_RUNS_DIR))
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument(
        "--export_legacy_npz_path",
        type=str,
        default=str(DEFAULT_EXPORT_NPZ_PATH),
        help="Optional path for a teammate-compatible aggregated evaluations.npz export.",
    )
    parser.add_argument("--no_show", action="store_true")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir).expanduser()
    results_dir = DEFAULT_RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    npz_paths = []
    for seed in args.seeds:
        npz_path = (
            runs_dir
            / f"simple_spread_multiagent_seed{seed}"
            / "eval_logs"
            / "evaluations.npz"
        )
        if not npz_path.exists():
            raise FileNotFoundError(f"Missing file: {npz_path}")
        npz_paths.append(npz_path)
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
    plt.title("Centralized-Critic PPO on Simple Spread")
    plt.legend()
    plt.tight_layout()

    if args.output_path is None:
        save_path = results_dir / "simple_spread_multiagent_3seed_curve.png"
    else:
        save_path = Path(args.output_path).expanduser()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200)
    if not args.no_show:
        plt.show()
    plt.close()

    export_path = None
    if args.export_legacy_npz_path:
        export_path = save_aggregated_eval_npz(
            npz_paths=npz_paths,
            output_path=Path(args.export_legacy_npz_path).expanduser(),
            seeds=args.seeds,
        )

    final_row = pivot.iloc[-1]
    print(f"Saved figure to: {save_path}")
    if export_path is not None:
        print(f"Saved aggregated eval NPZ to: {export_path}")
    print("\nFinal evaluation summary:")
    print(final_row)
    print(f"\nFinal mean: {final_row.mean():.2f}")
    final_std = 0.0 if len(final_row) == 1 else float(final_row.std(ddof=1))
    print(f"Final std:  {final_std:.2f}")


if __name__ == "__main__":
    main()
