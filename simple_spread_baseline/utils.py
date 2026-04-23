from __future__ import annotations

import argparse
import csv
import json
import os
import random
from pathlib import Path
from typing import Iterable

BASE_DIR = Path(__file__).resolve().parent
MPLCONFIG_DIR = BASE_DIR / ".mplconfig"
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))

import matplotlib.pyplot as plt
import numpy as np
import torch

from simple_spread_baseline.config import N_AGENTS
from simple_spread_baseline.env import EpisodeTracker, SimpleSpreadEnv


def str2bool(value):
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Could not parse boolean value from {value!r}.")


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, indent=2, sort_keys=True)
        file_obj.write("\n")


def append_csv_row(path: Path, fieldnames: list[str], row: dict) -> None:
    ensure_parent(path)
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def resolve_checkpoint_path(model_path: str | Path) -> Path:
    raw_path = Path(model_path).expanduser()
    candidates = [raw_path]
    if raw_path.suffix != ".pt":
        candidates.append(raw_path.with_suffix(".pt"))
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    checked = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Could not find checkpoint at any of: {checked}")


def plot_eval_curve(
    env_steps: Iterable[float],
    mean_returns: Iterable[float],
    output_path: Path,
    title: str,
) -> None:
    env_steps = np.asarray(list(env_steps), dtype=np.float64)
    mean_returns = np.asarray(list(mean_returns), dtype=np.float64)
    if env_steps.size == 0:
        return
    ensure_parent(output_path)
    plt.figure(figsize=(8, 5))
    plt.plot(env_steps, mean_returns, linewidth=2.5)
    plt.xlabel("Environment Steps")
    plt.ylabel("Mean Episodic Return")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_evaluations_npz(
    output_path: Path,
    env_steps: list[int],
    episode_returns: list[list[float]],
    episode_lengths: list[list[int]],
    collision_counts: list[list[float]],
    avg_landmarks_covered: list[list[float]],
    success_near_end: list[list[float]],
    mean_sum_min_dists: list[list[float]],
) -> None:
    ensure_parent(output_path)
    np.savez(
        output_path,
        timesteps=np.asarray(env_steps, dtype=np.int64),
        env_steps=np.asarray(env_steps, dtype=np.int64),
        agent_steps=np.asarray(env_steps, dtype=np.int64) * N_AGENTS,
        results=np.asarray(episode_returns, dtype=np.float64),
        ep_lengths=np.asarray(episode_lengths, dtype=np.int64),
        collision_counts=np.asarray(collision_counts, dtype=np.float64),
        avg_landmarks_covered=np.asarray(avg_landmarks_covered, dtype=np.float64),
        success_near_end=np.asarray(success_near_end, dtype=np.float64),
        mean_sum_min_dists=np.asarray(mean_sum_min_dists, dtype=np.float64),
    )


def summarize_episode_records(records: list[dict[str, float]]) -> dict[str, float]:
    if not records:
        return {
            "mean_episode_return": 0.0,
            "std_episode_return": 0.0,
            "mean_episode_length": 0.0,
            "mean_collision_count": 0.0,
            "mean_landmarks_covered": 0.0,
            "success_near_end_rate": 0.0,
            "mean_sum_min_dists": 0.0,
        }

    def collect(name: str) -> np.ndarray:
        return np.asarray([float(record[name]) for record in records], dtype=np.float64)

    returns = collect("episode_return_mean")
    return {
        "mean_episode_return": float(returns.mean()),
        "std_episode_return": float(returns.std(ddof=1)) if returns.size > 1 else 0.0,
        "mean_episode_length": float(collect("episode_length").mean()),
        "mean_collision_count": float(collect("collision_count").mean()),
        "mean_landmarks_covered": float(collect("avg_landmarks_covered").mean()),
        "success_near_end_rate": float(collect("full_coverage_near_end").mean()),
        "mean_sum_min_dists": float(collect("mean_sum_min_dists").mean()),
    }


def run_policy_episodes(
    agent,
    episodes: int,
    seed: int,
    local_ratio: float,
    max_cycles: int,
    continuous_actions: bool,
    terminate_on_success: bool,
    deterministic: bool,
    render_mode: str | None = None,
) -> list[dict[str, float]]:
    env = SimpleSpreadEnv(
        local_ratio=local_ratio,
        max_cycles=max_cycles,
        continuous_actions=continuous_actions,
        terminate_on_success=terminate_on_success,
        render_mode=render_mode,
    )
    records: list[dict[str, float]] = []
    try:
        for episode_idx in range(episodes):
            tracker = EpisodeTracker()
            observations, _, _ = env.reset(seed=seed + episode_idx)
            done = False
            while not done:
                action_batch = agent.select_actions(
                    observations[None, ...], deterministic=deterministic
                )[0]
                if render_mode is not None:
                    env.render()
                observations, _, rewards, done, info = env.step(action_batch)
                tracker.update(rewards, info["step_metrics"])
            episode_record = tracker.summary()
            episode_record["episode_index"] = float(episode_idx)
            records.append(episode_record)
    finally:
        env.close()
    return records


def record_policy_episode_frames(
    agent,
    *,
    seed: int,
    local_ratio: float,
    max_cycles: int,
    continuous_actions: bool,
    terminate_on_success: bool,
    deterministic: bool,
    fps: int = 6,
    hold_last_seconds: float = 2.0,
    pad_to_max_cycles: bool = True,
) -> tuple[list[np.ndarray], dict[str, float]]:
    env = SimpleSpreadEnv(
        local_ratio=local_ratio,
        max_cycles=max_cycles,
        continuous_actions=continuous_actions,
        terminate_on_success=terminate_on_success,
        render_mode="rgb_array",
    )
    tracker = EpisodeTracker()
    frames: list[np.ndarray] = []
    try:
        observations, _, _ = env.reset(seed=seed)
        first_frame = env.render()
        if first_frame is not None:
            frames.append(np.asarray(first_frame).copy())
        done = False
        while not done:
            action_batch = agent.select_actions(
                observations[None, ...], deterministic=deterministic
            )[0]
            observations, _, rewards, done, info = env.step(action_batch)
            tracker.update(rewards, info["step_metrics"])
            frame = env.render()
            if frame is not None:
                frames.append(np.asarray(frame).copy())
    finally:
        env.close()

    if frames:
        if pad_to_max_cycles:
            target_frame_count = max_cycles + 1
            while len(frames) < target_frame_count:
                frames.append(frames[-1].copy())
        hold_last_frames = max(0, int(round(max(0.0, hold_last_seconds) * max(1, fps))))
        for _ in range(hold_last_frames):
            frames.append(frames[-1].copy())

    return frames, tracker.summary()


def save_animation(path: Path, frames: list[np.ndarray], fps: int = 6) -> None:
    if not frames:
        raise ValueError("Cannot save animation with no frames.")
    ensure_parent(path)
    import imageio.v2 as imageio

    suffix = path.suffix.lower()
    if suffix == ".gif":
        imageio.mimsave(path, frames, duration=1.0 / max(1, fps), loop=0)
        return
    if suffix == ".webp":
        from PIL import Image

        pil_frames = [Image.fromarray(np.asarray(frame)) for frame in frames]
        pil_frames[0].save(
            path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=max(1, int(round(1000 / max(1, fps)))),
            loop=0,
            format="WEBP",
            quality=90,
            method=6,
        )
        return
    raise ValueError(f"Unsupported animation format for {path}. Use .gif or .webp.")


def save_gif(path: Path, frames: list[np.ndarray], fps: int = 6) -> None:
    save_animation(path, frames, fps=fps)
