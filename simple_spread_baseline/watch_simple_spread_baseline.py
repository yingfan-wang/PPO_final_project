import argparse
import time

from stable_baselines3 import PPO
from mpe2 import simple_spread_v3

N_AGENTS = 3
LOCAL_RATIO = 0.5


def watch(model_path: str, seed: int, episodes: int, max_cycles: int, fps: float):
    if fps <= 0:
        raise ValueError("--fps must be positive.")

    env = simple_spread_v3.parallel_env(
        N=N_AGENTS,
        local_ratio=LOCAL_RATIO,
        max_cycles=max_cycles,
        continuous_actions=False,
        render_mode="human",
    )

    model = PPO.load(model_path)
    frame_delay = 1.0 / fps

    for ep in range(episodes):
        obs, infos = env.reset(seed=seed + ep)
        ep_return = 0.0
        env.render()
        time.sleep(frame_delay)

        while env.agents:
            actions = {}
            for agent in env.agents:
                action, _ = model.predict(obs[agent], deterministic=True)
                actions[agent] = action

            obs, rewards, terminations, truncations, infos = env.step(actions)
            ep_return += sum(rewards.values()) / N_AGENTS
            env.render()
            time.sleep(frame_delay)

            if all(terminations.values()) or all(truncations.values()):
                break

        print(f"Episode {ep + 1}: return = {ep_return:.3f}")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max_cycles", type=int, default=50)
    parser.add_argument("--fps", type=float, default=10.0)
    args = parser.parse_args()

    watch(
        model_path=args.model_path,
        seed=args.seed,
        episodes=args.episodes,
        max_cycles=args.max_cycles,
        fps=args.fps,
    )
