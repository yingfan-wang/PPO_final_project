import argparse
import csv
from pathlib import Path

import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from mpe2 import simple_spread_v3

BASE_DIR = Path(__file__).resolve().parent
N_AGENTS = 3
LOCAL_RATIO = 0.5
MAX_CYCLES = 25


def make_parallel_env(seed: int, render_mode=None):
    env = simple_spread_v3.parallel_env(
        N=N_AGENTS,
        local_ratio=LOCAL_RATIO,
        max_cycles=MAX_CYCLES,
        continuous_actions=False,
        render_mode=render_mode,
    )
    env.reset(seed=seed)
    return env


def make_vec_env(seed: int):
    env = make_parallel_env(seed=seed, render_mode=None)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 1, num_cpus=1, base_class="stable_baselines3")
    return env


def evaluate_model(model, seed: int, n_eval_episodes: int = 10):
    env = make_parallel_env(seed=seed, render_mode=None)
    episode_returns = []

    for ep in range(n_eval_episodes):
        obs, infos = env.reset(seed=seed + ep)
        ep_return = 0.0

        while env.agents:
            actions = {}
            for agent in env.agents:
                action, _ = model.predict(obs[agent], deterministic=True)
                actions[agent] = action

            obs, rewards, terminations, truncations, infos = env.step(actions)
            ep_return += sum(rewards.values()) / N_AGENTS

            if all(terminations.values()) or all(truncations.values()):
                break

        episode_returns.append(ep_return)

    env.close()
    mean_return = sum(episode_returns) / len(episode_returns)
    return mean_return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--eval_freq", type=int, default=20_000)
    parser.add_argument("--eval_episodes", type=int, default=10)
    args = parser.parse_args()

    run_dir = BASE_DIR / "runs" / f"simple_spread_sb3_seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    csv_path = run_dir / "eval_log.csv"
    model_path = run_dir / "final_model"

    env = make_vec_env(seed=args.seed)

    model = PPO(
        MlpPolicy,
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        seed=args.seed,
        tensorboard_log=str(run_dir / "tb")
    )

    if not csv_path.exists():
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timesteps", "mean_return"])

    steps_done = 0
    chunk = args.eval_freq

    while steps_done < args.timesteps:
        learn_steps = min(chunk, args.timesteps - steps_done)
        model.learn(total_timesteps=learn_steps, reset_num_timesteps=False)
        steps_done += learn_steps

        mean_return = evaluate_model(
            model=model,
            seed=args.seed + 1000,
            n_eval_episodes=args.eval_episodes,
        )

        print(f"[seed {args.seed}] timesteps={steps_done}, eval_mean_return={mean_return:.3f}")

        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([steps_done, mean_return])

    model.save(str(model_path))
    env.close()

    print(f"Saved model to {model_path}.zip")
    print(f"Saved eval log to {csv_path}")


if __name__ == "__main__":
    main()
