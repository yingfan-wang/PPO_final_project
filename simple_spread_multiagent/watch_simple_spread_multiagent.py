"""Watch or frame-dump a saved Simple Spread MAPPO checkpoint."""

import argparse
from pathlib import Path
from typing import Optional

import torch

from simple_spread_multiagent_common import (
    DEFAULT_MAX_CYCLES,
    load_checkpoint,
    make_parallel_env,
    mean_agent_reward,
    predict_parallel_actions,
)


def resolve_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_frame(frame, frame_dir: Path, episode_idx: int, frame_idx: int) -> None:
    import os

    base_dir = Path(__file__).resolve().parent
    mplconfig_dir = base_dir / ".mplconfig"
    mplconfig_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mplconfig_dir))

    import matplotlib.pyplot as plt

    episode_dir = frame_dir / f"episode_{episode_idx:03d}"
    episode_dir.mkdir(parents=True, exist_ok=True)
    frame_path = episode_dir / f"frame_{frame_idx:04d}.png"
    plt.imsave(frame_path, frame)


def set_render_fps(env, fps: float) -> None:
    target_fps = max(1, int(round(fps)))
    if hasattr(env, "metadata") and isinstance(env.metadata, dict):
        env.metadata["render_fps"] = target_fps
    if hasattr(env, "unwrapped") and hasattr(env.unwrapped, "metadata"):
        if isinstance(env.unwrapped.metadata, dict):
            env.unwrapped.metadata["render_fps"] = target_fps


def watch(
    checkpoint_path: str,
    seed: int,
    episodes: int,
    max_cycles: int,
    fps: float,
    render_mode: str,
    frame_dir: Optional[str],
    deterministic: bool,
    terminate_on_success: bool,
    device_name: str,
):
    if episodes <= 0:
        raise ValueError("--episodes must be positive.")
    if fps <= 0:
        raise ValueError("--fps must be positive.")
    if render_mode == "rgb_array" and frame_dir is None:
        raise ValueError("--render_mode rgb_array requires --frame_dir.")
    if frame_dir is not None and render_mode != "rgb_array":
        raise ValueError("--frame_dir requires --render_mode rgb_array.")

    device = resolve_device(device_name)
    model, metadata, _, resolved_checkpoint_path = load_checkpoint(
        checkpoint_path,
        device=device,
    )
    env = make_parallel_env(
        max_cycles=max_cycles,
        terminate_on_success=terminate_on_success,
        render_mode=render_mode,
        seed=seed,
    )
    if render_mode == "human":
        set_render_fps(env, fps)
    capture_dir = None if frame_dir is None else Path(frame_dir).expanduser().resolve()
    if capture_dir is not None:
        capture_dir.mkdir(parents=True, exist_ok=True)

    print(f"Checkpoint: {resolved_checkpoint_path}")

    try:
        for episode_idx in range(episodes):
            observations, _ = env.reset(seed=seed + episode_idx)
            episode_return = 0.0
            episode_length = 0
            frame_idx = 0

            frame = env.render()
            if capture_dir is not None and frame is not None:
                save_frame(frame, capture_dir, episode_idx + 1, frame_idx)
                frame_idx += 1

            while env.agents:
                actions = predict_parallel_actions(
                    model=model,
                    metadata=metadata,
                    observations=observations,
                    device=device,
                    deterministic=deterministic,
                )

                observations, rewards, terminations, truncations, _ = env.step(actions)
                episode_return += mean_agent_reward(rewards)
                episode_length += 1

                if render_mode == "rgb_array":
                    frame = env.render()
                else:
                    frame = None
                if frame is not None and capture_dir is not None:
                    save_frame(frame, capture_dir, episode_idx + 1, frame_idx)
                    frame_idx += 1

                if not env.agents:
                    break
                if terminations and all(terminations.values()):
                    break
                if truncations and all(truncations.values()):
                    break

            summary = (
                f"Episode {episode_idx + 1}: return = {episode_return:.3f}, "
                f"length = {episode_length}"
            )
            if capture_dir is not None:
                summary += f", frames = {capture_dir / f'episode_{episode_idx + 1:03d}'}"
            print(summary)

    except KeyboardInterrupt:
        print("\nStopped watch loop.")
    except Exception as exc:
        message = str(exc).lower()
        if "display surface quit" in message or "video system not initialized" in message:
            print("\nViewer window closed.")
        elif "cannot connect to display" in message or "no available video device" in message:
            print(
                "\nNo display is available for the human viewer. "
                "Use --render_mode rgb_array --frame_dir <dir> to save frames."
            )
        else:
            raise
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", "--model_path", dest="checkpoint_path", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max_cycles", type=int, default=DEFAULT_MAX_CYCLES)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--render_mode",
        type=str,
        choices=["human", "rgb_array"],
        default="human",
    )
    parser.add_argument("--frame_dir", type=str, default=None)
    parser.set_defaults(deterministic=True)
    parser.add_argument(
        "--deterministic",
        dest="deterministic",
        action="store_true",
    )
    parser.add_argument(
        "--stochastic",
        dest="deterministic",
        action="store_false",
    )
    parser.set_defaults(terminate_on_success=True)
    parser.add_argument(
        "--terminate_on_success",
        dest="terminate_on_success",
        action="store_true",
    )
    parser.add_argument(
        "--no_terminate_on_success",
        dest="terminate_on_success",
        action="store_false",
    )
    args = parser.parse_args()

    watch(
        checkpoint_path=args.checkpoint_path,
        seed=args.seed,
        episodes=args.episodes,
        max_cycles=args.max_cycles,
        fps=args.fps,
        render_mode=args.render_mode,
        frame_dir=args.frame_dir,
        deterministic=args.deterministic,
        terminate_on_success=args.terminate_on_success,
        device_name=args.device,
    )
