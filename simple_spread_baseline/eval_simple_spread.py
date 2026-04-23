from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from simple_spread_baseline.ppo import PPOAgent
from simple_spread_baseline.utils import (
    resolve_checkpoint_path,
    run_policy_episodes,
    summarize_episode_records,
    str2bool,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--local_ratio", type=float, default=0.5)
    parser.add_argument("--max_cycles", type=int, default=25)
    parser.add_argument("--continuous_actions", type=str2bool, default=None)
    parser.add_argument("--terminate_on_success", type=str2bool, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = resolve_checkpoint_path(args.model_path)
    agent = PPOAgent.load(model_path, device=args.device)
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
    records = run_policy_episodes(
        agent=agent,
        episodes=args.episodes,
        seed=args.seed,
        local_ratio=args.local_ratio,
        max_cycles=args.max_cycles,
        continuous_actions=continuous_actions,
        terminate_on_success=terminate_on_success,
        deterministic=True,
    )
    summary = summarize_episode_records(records)
    print(f"Checkpoint: {model_path}")
    print(f"Episodes: {args.episodes}")
    print(f"Mean episodic return: {summary['mean_episode_return']:.3f}")
    print(f"Std episodic return:  {summary['std_episode_return']:.3f}")
    print(f"Mean episode length:  {summary['mean_episode_length']:.2f}")
    print(f"Mean collisions:      {summary['mean_collision_count']:.2f}")
    print(f"Mean landmarks cover: {summary['mean_landmarks_covered']:.2f}")
    print(f"Success near end:     {summary['success_near_end_rate']:.2%}")
    print(f"Mean sum min dists:   {summary['mean_sum_min_dists']:.3f}")


if __name__ == "__main__":
    main()
