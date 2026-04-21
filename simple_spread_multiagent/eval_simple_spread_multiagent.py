"""Evaluate a saved Simple Spread SB3 adaptation checkpoint."""

import argparse

from stable_baselines3 import PPO

from simple_spread_multiagent_common import (
    DEFAULT_MAX_CYCLES,
    LOCAL_RATIO,
    evaluate_team_policy,
    resolve_model_path,
    safe_std,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_path",
        "--model_path",
        dest="checkpoint_path",
        required=True,
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--max_cycles", type=int, default=DEFAULT_MAX_CYCLES)
    parser.add_argument("--local_ratio", type=float, default=LOCAL_RATIO)
    parser.add_argument("--device", type=str, default="auto")
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

    if args.episodes <= 0:
        raise ValueError("--episodes must be positive.")
    if not 0.0 <= args.local_ratio <= 1.0:
        raise ValueError("--local_ratio must be in [0, 1].")

    model_path = resolve_model_path(args.checkpoint_path)
    model = PPO.load(model_path, device=args.device)
    returns, episode_lengths = evaluate_team_policy(
        model=model,
        seed=args.seed,
        n_eval_episodes=args.episodes,
        max_cycles=args.max_cycles,
        terminate_on_success=args.terminate_on_success,
        local_ratio=args.local_ratio,
        deterministic=args.deterministic,
    )

    print(f"Checkpoint: {model_path}")
    print(f"Local ratio: {args.local_ratio}")
    print(f"Episodes: {args.episodes}")
    print(f"Mean return: {returns.mean():.2f}")
    print(f"Std return:  {safe_std(returns):.2f}")
    print(f"Min return:  {returns.min():.2f}")
    print(f"Max return:  {returns.max():.2f}")
    print(f"Mean length: {episode_lengths.mean():.2f}")
    print(f"Std length:  {safe_std(episode_lengths):.2f}")


if __name__ == "__main__":
    main()
