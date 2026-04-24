from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from simple_spread_baseline.utils import (
    ensure_parent,
    record_policy_episode_frames,
    resolve_checkpoint_path,
    run_policy_episodes,
    save_animation,
    str2bool,
)
from simple_spread_multiagent.ma_ppo import MAPPOAgent


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--local_ratio", type=float, default=0.5)
    parser.add_argument("--max_cycles", type=int, default=25)
    parser.add_argument("--continuous_actions", type=str2bool, default=None)
    parser.add_argument("--terminate_on_success", type=str2bool, default=None)
    parser.add_argument("--render_mode", type=str, default="human")
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--output_format", type=str, default="gif")
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--hold_last_seconds", type=float, default=2.0)
    parser.add_argument("--pad_to_max_cycles", type=str2bool, default=True)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = resolve_checkpoint_path(args.model_path)
    agent = MAPPOAgent.load(model_path, device=args.device)
    continuous_actions = (
        agent.config.continuous_actions
        if args.continuous_actions is None
        else args.continuous_actions
    )
    terminate_on_success = (
        agent.config.terminate_on_success
        if args.terminate_on_success is None
        else args.terminate_on_success
    )
    if args.save_dir:
        save_dir = Path(args.save_dir).expanduser()
        ensure_parent(save_dir / ".keep")
        output_suffix = args.output_format.lower().lstrip(".")
        for episode_idx in range(args.episodes):
            frames, record = record_policy_episode_frames(
                agent=agent,
                seed=args.seed + episode_idx,
                local_ratio=args.local_ratio,
                max_cycles=args.max_cycles,
                continuous_actions=continuous_actions,
                terminate_on_success=terminate_on_success,
                deterministic=True,
                fps=args.fps,
                hold_last_seconds=args.hold_last_seconds,
                pad_to_max_cycles=args.pad_to_max_cycles,
            )
            animation_path = save_dir / f"episode_{episode_idx + 1:02d}.{output_suffix}"
            save_animation(animation_path, frames, fps=args.fps)
            print(
                f"Episode {episode_idx + 1}: return={record['episode_return_mean']:.3f}, "
                f"collisions={record['collision_count']}, "
                f"covered={record['avg_landmarks_covered']:.2f}, "
                f"success_near_end={bool(record['full_coverage_near_end'])}, "
                f"saved={animation_path}"
            )
        return
    records = run_policy_episodes(
        agent=agent,
        episodes=args.episodes,
        seed=args.seed,
        local_ratio=args.local_ratio,
        max_cycles=args.max_cycles,
        continuous_actions=continuous_actions,
        terminate_on_success=terminate_on_success,
        deterministic=True,
        render_mode=args.render_mode,
    )
    for episode_idx, record in enumerate(records, start=1):
        print(
            f"Episode {episode_idx}: return={record['episode_return_mean']:.3f}, "
            f"collisions={record['collision_count']}, "
            f"covered={record['avg_landmarks_covered']:.2f}, "
            f"success_near_end={bool(record['full_coverage_near_end'])}"
        )


if __name__ == "__main__":
    main()
