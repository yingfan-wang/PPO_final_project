import argparse
import gymnasium as gym
from stable_baselines3 import PPO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--env_id", type=str, default="HalfCheetah-v5")
    parser.add_argument("--episodes", type=int, default=3)
    args = parser.parse_args()

    # Load a checkpoint produced by train_mujoco.py.
    model = PPO.load(args.model_path)

    # render_mode="human" opens the environment viewer so we can watch the
    # trained policy act in real time.
    env = gym.make(args.env_id, render_mode="human")

    for ep in range(args.episodes):
        # Reset once per episode and vary the seed slightly across episodes.
        obs, info = env.reset(seed=ep)
        done = False
        ep_return = 0.0

        while not done:
            # Deterministic actions make the visual rollout easier to compare
            # across repeated runs of the same saved model.
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += reward
            done = terminated or truncated

        # Print the total return so the visual rollout has a numeric summary too.
        print(f"Episode {ep+1}: return = {ep_return:.2f}")

    env.close()


if __name__ == "__main__":
    main()
