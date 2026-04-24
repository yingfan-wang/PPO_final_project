from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from simple_spread_baseline.ppo import PPOAgent
from simple_spread_baseline.utils import record_policy_episode_frames, save_animation
from simple_spread_multiagent_MAPPO.ma_ppo import MAPPOAgent

RESULTS_DIR = ROOT_DIR / "results" / "simple_spread" / "animations"


def add_title_band(frame: np.ndarray, left_label: str, right_label: str) -> np.ndarray:
    image = Image.fromarray(np.asarray(frame))
    title_height = 32
    canvas = Image.new("RGB", (image.width, image.height + title_height), color="white")
    canvas.paste(image, (0, title_height))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((10, 10), left_label, fill="black", font=font)
    right_x = image.width // 2 + 10
    draw.text((right_x, 10), right_label, fill="black", font=font)
    return np.asarray(canvas)


def combine_frames(
    left_frames: list[np.ndarray],
    right_frames: list[np.ndarray],
    *,
    left_label: str,
    right_label: str,
) -> list[np.ndarray]:
    frame_count = min(len(left_frames), len(right_frames))
    combined: list[np.ndarray] = []
    for idx in range(frame_count):
        left = np.asarray(left_frames[idx])
        right = np.asarray(right_frames[idx])
        merged = np.concatenate([left, right], axis=1)
        combined.append(
            add_title_band(merged, left_label=left_label, right_label=right_label)
        )
    return combined


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max_cycles", type=int, default=25)
    parser.add_argument("--local_ratio", type=float, default=0.5)
    parser.add_argument("--terminate_on_success", action="store_true", default=True)
    return parser.parse_args()


def load_agents(device: str) -> tuple[PPOAgent, MAPPOAgent]:
    baseline = PPOAgent.load(
        ROOT_DIR
        / "simple_spread_baseline"
        / "runs"
        / "simple_spread_baseline_seed0"
        / "final_model.pt",
        device=device,
    )
    multiagent = MAPPOAgent.load(
        ROOT_DIR
        / "simple_spread_multiagent_MAPPO"
        / "runs"
        / "simple_spread_multiagent_seed0"
        / "final_model.pt",
        device=device,
    )
    return baseline, multiagent


def checkpoint_path(track_dir: str, seed: int, *, run_prefix: str | None = None) -> Path:
    effective_run_prefix = track_dir if run_prefix is None else run_prefix
    return (
        ROOT_DIR
        / track_dir
        / "runs"
        / f"{effective_run_prefix}_seed{seed}"
        / "final_model.pt"
    )


def record_frames_for_seed(
    *,
    seed: int,
    fps: int,
    device: str,
    max_cycles: int,
    local_ratio: float,
    terminate_on_success: bool,
    hold_last_seconds: float,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    baseline = PPOAgent.load(
        checkpoint_path("simple_spread_baseline", seed),
        device=device,
    )
    multiagent = MAPPOAgent.load(
        checkpoint_path(
            "simple_spread_multiagent_MAPPO",
            seed,
            run_prefix="simple_spread_multiagent",
        ),
        device=device,
    )
    baseline_frames, _ = record_policy_episode_frames(
        agent=baseline,
        seed=seed,
        local_ratio=local_ratio,
        max_cycles=max_cycles,
        continuous_actions=False,
        terminate_on_success=terminate_on_success,
        deterministic=True,
        fps=fps,
        hold_last_seconds=hold_last_seconds,
        pad_to_max_cycles=True,
    )
    multiagent_frames, _ = record_policy_episode_frames(
        agent=multiagent,
        seed=seed,
        local_ratio=local_ratio,
        max_cycles=max_cycles,
        continuous_actions=False,
        terminate_on_success=terminate_on_success,
        deterministic=True,
        fps=fps,
        hold_last_seconds=hold_last_seconds,
        pad_to_max_cycles=True,
    )
    return baseline_frames, multiagent_frames


def save_track_gif(path: Path, frames: list[np.ndarray], fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_animation(path, frames, fps=fps)


def main() -> None:
    args = parse_args()
    baseline_dir = RESULTS_DIR / "baseline"
    multiagent_dir = RESULTS_DIR / "multiagent"
    side_by_side_dir = RESULTS_DIR / "side_by_side"

    for seed in args.seeds:
        for variant_name, hold_last_seconds in [("default", 2.0), ("long", 4.0)]:
            baseline_frames, multiagent_frames = record_frames_for_seed(
                seed=seed,
                fps=args.fps,
                device=args.device,
                max_cycles=args.max_cycles,
                local_ratio=args.local_ratio,
                terminate_on_success=args.terminate_on_success,
                hold_last_seconds=hold_last_seconds,
            )
            side_by_side_frames = combine_frames(
                baseline_frames,
                multiagent_frames,
                left_label=f"Baseline seed {seed}",
                right_label=f"Multi-agent seed {seed}",
            )

            suffix = "" if variant_name == "default" else "_long"
            save_track_gif(
                baseline_dir / f"simple_spread_baseline_seed{seed}{suffix}.gif",
                baseline_frames,
                fps=args.fps,
            )
            save_track_gif(
                multiagent_dir / f"simple_spread_multiagent_seed{seed}{suffix}.gif",
                multiagent_frames,
                fps=args.fps,
            )
            save_track_gif(
                side_by_side_dir / f"simple_spread_side_by_side_seed{seed}{suffix}.gif",
                side_by_side_frames,
                fps=args.fps,
            )
            print(
                f"Saved seed {seed} {variant_name} GIFs under {RESULTS_DIR}",
                flush=True,
            )


if __name__ == "__main__":
    main()
