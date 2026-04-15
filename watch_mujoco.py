import argparse
import gymnasium as gym
from stable_baselines3 import PPO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--env_id", type=str, default="HalfCheetah-v5")
    parser.add_argument("--episodes", type=int, default=3)
    args = parser.parse_args()

    model = PPO.load(args.model_path)

    env = gym.make(args.env_id, render_mode="human")

    for ep in range(args.episodes):
        obs, info = env.reset(seed=ep)
        done = False
        ep_return = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += reward
            done = terminated or truncated

        print(f"Episode {ep+1}: return = {ep_return:.2f}")

    env.close()


if __name__ == "__main__":
    main()