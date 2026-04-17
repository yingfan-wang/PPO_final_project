import argparse

from stable_baselines3 import PPO
from mpe2 import simple_spread_v3

N_AGENTS = 3
LOCAL_RATIO = 0.5
MAX_CYCLES = 25


def evaluate(model_path: str, seed: int, episodes: int):
    env = simple_spread_v3.parallel_env(
        N=N_AGENTS,
        local_ratio=LOCAL_RATIO,
        max_cycles=MAX_CYCLES,
        continuous_actions=False,
        render_mode=None,
    )

    model = PPO.load(model_path)
    returns = []

    for ep in range(episodes):
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

        returns.append(ep_return)
        print(f"Episode {ep + 1}/{episodes}: return = {ep_return:.3f}")

    env.close()

    mean_return = sum(returns) / len(returns)
    print("\n===== Final Evaluation =====")
    print(f"Episodes: {episodes}")
    print(f"Mean return: {mean_return:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=20)
    args = parser.parse_args()

    evaluate(
        model_path=args.model_path,
        seed=args.seed,
        episodes=args.episodes,
    )
