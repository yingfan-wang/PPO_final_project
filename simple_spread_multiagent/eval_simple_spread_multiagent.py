"""Evaluate a saved Simple Spread MAPPO checkpoint."""

import argparse

import torch

from simple_spread_multiagent_common import (
    DEFAULT_MAX_CYCLES,
    evaluate_team_policy,
    load_checkpoint,
    safe_std,
)


def resolve_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", "--model_path", dest="checkpoint_path", required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--max_cycles", type=int, default=DEFAULT_MAX_CYCLES)
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

    device = resolve_device(args.device)
    model, metadata, _, resolved_checkpoint_path = load_checkpoint(
        args.checkpoint_path,
        device=device,
    )
    returns, episode_lengths = evaluate_team_policy(
        model=model,
        metadata=metadata,
        seed=args.seed,
        n_eval_episodes=args.episodes,
        max_cycles=args.max_cycles,
        terminate_on_success=args.terminate_on_success,
        device=device,
        deterministic=args.deterministic,
    )

    print(f"Checkpoint: {resolved_checkpoint_path}")
    print(f"Episodes: {args.episodes}")
    print(f"Mean return: {returns.mean():.2f}")
    print(f"Std return:  {safe_std(returns):.2f}")
    print(f"Min return:  {returns.min():.2f}")
    print(f"Max return:  {returns.max():.2f}")
    print(f"Mean length: {episode_lengths.mean():.2f}")
    print(f"Std length:  {safe_std(episode_lengths):.2f}")


if __name__ == "__main__":
    main()
