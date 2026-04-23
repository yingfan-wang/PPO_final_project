from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from simple_spread_baseline.config import BaselineConfig, N_AGENTS, RUN_PREFIX
from simple_spread_baseline.env import SimpleSpreadVectorEnv
from simple_spread_baseline.ppo import PPOAgent
from simple_spread_baseline.utils import (
    append_csv_row,
    plot_eval_curve,
    run_policy_episodes,
    save_evaluations_npz,
    set_seed,
    str2bool,
    summarize_episode_records,
    write_json,
)

BASE_DIR = Path(__file__).resolve().parent
RUNS_DIR = BASE_DIR / "runs"


def parse_args() -> BaselineConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--eval_freq", type=int, default=10_000)
    parser.add_argument("--n_eval_episodes", type=int, default=10)
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--rollout_steps", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_coef", type=float, default=0.2)
    parser.add_argument("--ent_coef", type=float, default=0.0)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--update_epochs", type=int, default=10)
    parser.add_argument("--minibatch_size", type=int, default=256)
    parser.add_argument("--local_ratio", type=float, default=0.5)
    parser.add_argument("--max_cycles", type=int, default=25)
    parser.add_argument("--continuous_actions", type=str2bool, default=False)
    parser.add_argument("--terminate_on_success", type=str2bool, default=False)
    parser.add_argument("--curriculum", type=str2bool, default=False)
    parser.add_argument("--curriculum_switch_step", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()
    return BaselineConfig(
        seed=args.seed,
        timesteps=args.timesteps,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        num_envs=args.num_envs,
        rollout_steps=args.rollout_steps,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_coef=args.clip_coef,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
        local_ratio=args.local_ratio,
        max_cycles=args.max_cycles,
        continuous_actions=args.continuous_actions,
        terminate_on_success=args.terminate_on_success,
        curriculum=args.curriculum,
        curriculum_switch_step=args.curriculum_switch_step,
        device=args.device,
    )


def validate_config(config: BaselineConfig) -> None:
    if config.timesteps <= 0 or config.eval_freq <= 0:
        raise ValueError("--timesteps and --eval_freq must be positive.")
    if config.n_eval_episodes <= 0 or config.num_envs <= 0:
        raise ValueError("--n_eval_episodes and --num_envs must be positive.")
    if config.rollout_steps <= 0 or config.minibatch_size <= 0:
        raise ValueError("--rollout_steps and --minibatch_size must be positive.")
    if config.local_ratio < 0.0 or config.local_ratio > 1.0:
        raise ValueError("--local_ratio must be in [0, 1].")
    if config.curriculum_switch_step < 0:
        raise ValueError("--curriculum_switch_step must be non-negative.")
    total_samples = config.rollout_steps * config.num_envs * N_AGENTS
    if config.minibatch_size > total_samples:
        raise ValueError(
            "--minibatch_size cannot exceed one rollout worth of agent samples "
            f"({total_samples})."
        )


def main() -> None:
    config = parse_args()
    validate_config(config)
    set_seed(config.seed)

    run_dir = RUNS_DIR / f"{RUN_PREFIX}_seed{config.seed}"
    eval_dir = run_dir / "eval_logs"
    best_dir = run_dir / "best_model"
    eval_dir.mkdir(parents=True, exist_ok=True)
    best_dir.mkdir(parents=True, exist_ok=True)

    effective_timesteps = (
        ((config.timesteps + config.rollout_steps * config.num_envs - 1) // (config.rollout_steps * config.num_envs))
        * config.rollout_steps
        * config.num_envs
    )
    run_config = config.to_dict()
    run_config["requested_env_steps"] = config.timesteps
    run_config["effective_env_steps"] = effective_timesteps
    run_config["effective_agent_steps"] = effective_timesteps * N_AGENTS
    curriculum_switch_step = (
        max(config.timesteps // 2, config.rollout_steps * config.num_envs)
        if config.curriculum and config.curriculum_switch_step <= 0
        else config.curriculum_switch_step
    )
    run_config["curriculum"] = config.curriculum
    run_config["curriculum_switch_step"] = curriculum_switch_step if config.curriculum else None
    run_config["method"] = (
        "Naive PPO transfer to Simple Spread with a shared local-observation "
        "actor-critic and no centralized critic."
    )
    run_config["actor_input"] = "One local observation per agent."
    run_config["critic_input"] = "The same one-agent local observation."
    run_config["credit_assignment"] = "Environment reward only; no joint state."
    run_config["action_space"] = (
        "continuous_box_5" if config.continuous_actions else "discrete_5"
    )
    write_json(run_dir / "run_config.json", run_config)

    envs = SimpleSpreadVectorEnv(
        num_envs=config.num_envs,
        local_ratio=config.local_ratio,
        max_cycles=config.max_cycles,
        continuous_actions=config.continuous_actions,
        terminate_on_success=config.terminate_on_success,
        curriculum=config.curriculum,
    )
    if config.curriculum:
        envs.set_curriculum_stage(0)
    agent = PPOAgent(config)
    observations, _ = envs.reset(seed=config.seed)

    train_fields = [
        "update",
        "env_steps",
        "agent_steps",
        "policy_loss",
        "value_loss",
        "entropy",
        "approx_kl",
        "clip_fraction",
        "explained_variance",
        "rollout_episode_return_mean",
        "rollout_collision_count_mean",
        "rollout_landmarks_covered_mean",
    ]
    eval_fields = [
        "env_steps",
        "agent_steps",
        "mean_episode_return",
        "std_episode_return",
        "mean_episode_length",
        "mean_collision_count",
        "mean_landmarks_covered",
        "success_near_end_rate",
        "mean_sum_min_dists",
    ]

    env_steps = 0
    update_idx = 0
    next_eval_step = config.eval_freq
    best_eval_return = float("-inf")
    eval_env_steps: list[int] = []
    eval_returns: list[list[float]] = []
    eval_lengths: list[list[int]] = []
    eval_collisions: list[list[float]] = []
    eval_coverage: list[list[float]] = []
    eval_success: list[list[float]] = []
    eval_min_dists: list[list[float]] = []
    curriculum_switched = not config.curriculum

    def run_evaluation(eval_seed: int) -> float:
        nonlocal best_eval_return
        records = run_policy_episodes(
            agent=agent,
            episodes=config.n_eval_episodes,
            seed=eval_seed,
            local_ratio=config.local_ratio,
            max_cycles=config.max_cycles,
            continuous_actions=config.continuous_actions,
            terminate_on_success=config.terminate_on_success,
            deterministic=True,
        )
        summary = summarize_episode_records(records)
        append_csv_row(
            eval_dir / "eval_metrics.csv",
            eval_fields,
            {
                "env_steps": env_steps,
                "agent_steps": env_steps * N_AGENTS,
                **summary,
            },
        )
        eval_env_steps.append(env_steps)
        eval_returns.append([record["episode_return_mean"] for record in records])
        eval_lengths.append([int(record["episode_length"]) for record in records])
        eval_collisions.append([record["collision_count"] for record in records])
        eval_coverage.append([record["avg_landmarks_covered"] for record in records])
        eval_success.append([record["full_coverage_near_end"] for record in records])
        eval_min_dists.append([record["mean_sum_min_dists"] for record in records])
        save_evaluations_npz(
            eval_dir / "evaluations.npz",
            eval_env_steps,
            eval_returns,
            eval_lengths,
            eval_collisions,
            eval_coverage,
            eval_success,
            eval_min_dists,
        )
        plot_eval_curve(
            env_steps=eval_env_steps,
            mean_returns=[sum(values) / len(values) for values in eval_returns],
            output_path=run_dir / "eval_curve.png",
            title="Simple Spread Baseline PPO",
        )
        if summary["mean_episode_return"] > best_eval_return:
            best_eval_return = summary["mean_episode_return"]
            agent.save(
                best_dir / "best_model.pt",
                metadata={"env_steps": env_steps, "summary": summary},
            )
        return summary["mean_episode_return"]

    try:
        while env_steps < config.timesteps:
            buffer = agent.build_buffer()
            rollout_records: list[dict[str, float]] = []
            for _ in range(config.rollout_steps):
                env_actions, policy_actions, log_probs, values = agent.rollout_step(observations)
                next_observations, _, rewards, dones, infos = envs.step(env_actions)
                buffer.add(
                    observations,
                    policy_actions,
                    log_probs,
                    rewards,
                    dones,
                    values,
                )
                observations = next_observations
                env_steps += config.num_envs
                if not curriculum_switched and env_steps >= curriculum_switch_step:
                    envs.set_curriculum_stage(1)
                    curriculum_switched = True
                for info in infos:
                    if "episode" in info:
                        rollout_records.append(info["episode"])

            last_values = agent.get_values(observations)
            buffer.compute_returns_and_advantages(last_values)
            train_metrics = agent.update(buffer)
            rollout_summary = summarize_episode_records(rollout_records)
            append_csv_row(
                run_dir / "train_metrics.csv",
                train_fields,
                {
                    "update": update_idx,
                    "env_steps": env_steps,
                    "agent_steps": env_steps * N_AGENTS,
                    **train_metrics,
                    "rollout_episode_return_mean": rollout_summary["mean_episode_return"],
                    "rollout_collision_count_mean": rollout_summary["mean_collision_count"],
                    "rollout_landmarks_covered_mean": rollout_summary["mean_landmarks_covered"],
                },
            )

            while env_steps >= next_eval_step:
                run_evaluation(config.seed + 100_000 + len(eval_env_steps) * 1_000)
                next_eval_step += config.eval_freq

            update_idx += 1

        if not eval_env_steps or eval_env_steps[-1] != env_steps:
            run_evaluation(config.seed + 200_000)

        agent.save(run_dir / "final_model.pt", metadata={"env_steps": env_steps})
    finally:
        envs.close()

    print(f"Run directory: {run_dir}")
    print(f"Effective environment steps: {env_steps}")
    print(f"Effective agent steps: {env_steps * N_AGENTS}")
    print(f"Best evaluation mean return: {best_eval_return:.3f}")


if __name__ == "__main__":
    main()
