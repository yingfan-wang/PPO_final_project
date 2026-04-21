"""Watch or frame-dump a saved Simple Spread SB3 adaptation checkpoint."""

import argparse
import time
from pathlib import Path
from typing import Optional

from stable_baselines3 import PPO

from simple_spread_multiagent_common import (
    DEFAULT_MAX_CYCLES,
    LOCAL_RATIO,
    make_parallel_env,
    mean_agent_reward,
    predict_parallel_actions,
    resolve_model_path,
)


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


def pause_after_success(
    env,
    render_mode: str,
    capture_dir: Optional[Path],
    episode_idx: int,
    frame_idx: int,
    fps: float,
    success_pause_seconds: float,
) -> int:
    if success_pause_seconds <= 0:
        return frame_idx

    frame_interval = 1.0 / max(1.0, fps)
    frame_count = max(1, int(round(success_pause_seconds * fps)))

    if render_mode == "rgb_array":
        for _ in range(frame_count):
            frame = env.render()
            if frame is not None and capture_dir is not None:
                save_frame(frame, capture_dir, episode_idx, frame_idx)
                frame_idx += 1
        return frame_idx

    deadline = time.monotonic() + success_pause_seconds
    while time.monotonic() < deadline:
        env.render()
        time.sleep(frame_interval)
    return frame_idx


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
    local_ratio: float,
    device: str,
    success_pause_seconds: float,
    episode_seconds: float,
):
    if episodes <= 0:
        raise ValueError("--episodes must be positive.")
    if fps <= 0:
        raise ValueError("--fps must be positive.")
    if success_pause_seconds < 0:
        raise ValueError("--success_pause_seconds cannot be negative.")
    if episode_seconds < 0:
        raise ValueError("--episode_seconds cannot be negative.")
    if render_mode == "rgb_array" and frame_dir is None:
        raise ValueError("--render_mode rgb_array requires --frame_dir.")
    if frame_dir is not None and render_mode != "rgb_array":
        raise ValueError("--frame_dir requires --render_mode rgb_array.")
    if not 0.0 <= local_ratio <= 1.0:
        raise ValueError("--local_ratio must be in [0, 1].")

    resolved_model_path = resolve_model_path(checkpoint_path)
    model = PPO.load(resolved_model_path, device=device)
    env = make_parallel_env(
        max_cycles=max_cycles,
        terminate_on_success=terminate_on_success,
        local_ratio=local_ratio,
        render_mode=render_mode,
        seed=seed,
    )
    if render_mode == "human":
        set_render_fps(env, fps)
    capture_dir = None if frame_dir is None else Path(frame_dir).expanduser().resolve()
    if capture_dir is not None:
        capture_dir.mkdir(parents=True, exist_ok=True)

    print(f"Checkpoint: {resolved_model_path}")
    print(f"Local ratio: {local_ratio}")

    try:
        for episode_idx in range(episodes):
            observations, _ = env.reset(seed=seed + episode_idx)
            episode_return = 0.0
            episode_length = 0
            frame_idx = 0
            ended_by_success = False
            episode_deadline = None
            if episode_seconds > 0:
                episode_deadline = time.monotonic() + episode_seconds

            def episode_time_expired() -> bool:
                return (
                    episode_deadline is not None
                    and time.monotonic() >= episode_deadline
                )

            frame = env.render()
            if capture_dir is not None and frame is not None:
                save_frame(frame, capture_dir, episode_idx + 1, frame_idx)
                frame_idx += 1

            while env.agents:
                actions = predict_parallel_actions(
                    model=model,
                    observations=observations,
                    deterministic=deterministic,
                )

                observations, rewards, terminations, truncations, _ = env.step(actions)
                episode_return += mean_agent_reward(rewards)
                episode_length += 1

                frame = env.render()
                if frame is not None and capture_dir is not None:
                    save_frame(frame, capture_dir, episode_idx + 1, frame_idx)
                    frame_idx += 1

                if episode_time_expired():
                    break
                ended_by_success = (
                    bool(terminations)
                    and all(terminations.values())
                    and not (truncations and all(truncations.values()))
                )

                if not env.agents:
                    break
                if terminations and all(terminations.values()):
                    break
                if truncations and all(truncations.values()):
                    break

            if ended_by_success and not episode_time_expired():
                frame_idx = pause_after_success(
                    env=env,
                    render_mode=render_mode,
                    capture_dir=capture_dir,
                    episode_idx=episode_idx + 1,
                    frame_idx=frame_idx,
                    fps=fps,
                    success_pause_seconds=success_pause_seconds,
                )

            summary = (
                f"Episode {episode_idx + 1}: return = {episode_return:.3f}, "
                f"length = {episode_length}"
            )
            if ended_by_success and success_pause_seconds > 0:
                summary += f", success pause = {success_pause_seconds:.1f}s"
            if capture_dir is not None:
                summary += f", frames = {capture_dir / f'episode_{episode_idx + 1:03d}'}"
            print(summary)
            if episode_time_expired():
                print(f"Episode time limit reached ({episode_seconds:.1f}s).")

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
    parser.add_argument(
        "--checkpoint_path",
        "--model_path",
        dest="checkpoint_path",
        required=True,
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max_cycles", type=int, default=DEFAULT_MAX_CYCLES)
    parser.add_argument("--local_ratio", type=float, default=LOCAL_RATIO)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--success_pause_seconds", type=float, default=0.0)
    parser.add_argument(
        "--episode_seconds",
        "--watch_seconds",
        dest="episode_seconds",
        type=float,
        default=10.0,
    )
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
        local_ratio=args.local_ratio,
        device=args.device,
        success_pause_seconds=args.success_pause_seconds,
        episode_seconds=args.episode_seconds,
    )
