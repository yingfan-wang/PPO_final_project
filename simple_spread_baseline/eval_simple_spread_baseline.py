import argparse

from stable_baselines3 import PPO

from simple_spread_baseline_common import (
    DEFAULT_MAX_CYCLES,
    evaluate_team_policy,
    resolve_model_path,
    safe_std,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--max_cycles", type=int, default=DEFAULT_MAX_CYCLES)
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

    model_path = resolve_model_path(args.model_path)
    model = PPO.load(model_path)
    returns, episode_lengths = evaluate_team_policy(
        model=model,
        seed=args.seed,
        n_eval_episodes=args.episodes,
        max_cycles=args.max_cycles,
        terminate_on_success=args.terminate_on_success,
    )

    print(f"Model: {model_path}")
    print(f"Episodes: {args.episodes}")
    print(f"Mean return: {returns.mean():.2f}")
    print(f"Std return:  {safe_std(returns):.2f}")
    print(f"Min return:  {returns.min():.2f}")
    print(f"Max return:  {returns.max():.2f}")
    print(f"Mean length: {episode_lengths.mean():.2f}")
    print(f"Std length:  {safe_std(episode_lengths):.2f}")


if __name__ == "__main__":
    main()
