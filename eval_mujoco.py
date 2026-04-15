import argparse

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--env_id", type=str, default="HalfCheetah-v5")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()

    model = PPO.load(args.model_path)

    returns = []
    for ep in range(args.episodes):
        env = gym.make(args.env_id)
        env = Monitor(env)
        obs, info = env.reset(seed=args.seed + ep)

        done = False
        ep_return = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += reward
            done = terminated or truncated

        returns.append(ep_return)
        env.close()

    returns = np.array(returns, dtype=np.float64)

    print(f"Episodes: {args.episodes}")
    print(f"Mean return: {returns.mean():.2f}")
    print(f"Std return:  {returns.std(ddof=1):.2f}")
    print(f"Min return:  {returns.min():.2f}")
    print(f"Max return:  {returns.max():.2f}")


if __name__ == "__main__":
    main()