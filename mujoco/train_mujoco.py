import argparse
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor


def make_env(env_id: str, seed: int):
    """Create one monitored environment instance with a fixed seed."""

    def _init():
        env = gym.make(env_id)
        # Monitor records episode returns/lengths so SB3 can log them cleanly.
        env = Monitor(env)
        # Seed both reset() and the action space for more repeatable runs.
        env.reset(seed=seed)
        env.action_space.seed(seed)
        return env

    return _init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", type=str, default="HalfCheetah-v5")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--eval_freq", type=int, default=10_000)
    parser.add_argument("--n_eval_episodes", type=int, default=10)
    args = parser.parse_args()

    # Each seed gets its own directory so models, eval logs, and TensorBoard
    # files stay grouped together under runs/<env>_seed<seed>/.
    run_dir = Path("runs") / f"{args.env_id}_seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    train_env = DummyVecEnv([make_env(args.env_id, args.seed)])
    train_env = VecMonitor(train_env)

    # Use a different seed for evaluation to avoid measuring on the exact same
    # episode seeds seen during training rollouts.
    eval_env = DummyVecEnv([make_env(args.env_id, args.seed + 1000)])
    eval_env = VecMonitor(eval_env)

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        seed=args.seed,
        tensorboard_log=str(run_dir / "tb"),
    )

    # EvalCallback periodically evaluates the current policy, writes
    # evaluations.npz, and saves the best checkpoint seen so far.
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(run_dir / "best_model"),
        log_path=str(run_dir / "eval_logs"),
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True,
        render=False,
    )

    # final_model.zip is the last checkpoint after all requested timesteps.
    model.learn(total_timesteps=args.timesteps, callback=eval_callback)
    model.save(str(run_dir / "final_model"))

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
