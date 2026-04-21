"""Train the traced naive SB3 PPO adaptation for MPE2 Simple Spread.

This intentionally uses the same agent-slot PPO path as the saved naive
baseline run that performed best in this workspace:

- one shared SB3 MlpPolicy over vectorized agent slots
- actor and critic see only one agent's local observation
- greedy nearest-landmark local rewards by default
- no agent IDs, centralized critic, or joint observation input
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MPLCONFIG_DIR = BASE_DIR / ".mplconfig"
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import set_random_seed

from simple_spread_multiagent_common import (
    DEFAULT_MAX_CYCLES,
    DEFAULT_NUM_VEC_ENVS,
    LOCAL_RATIO,
    N_AGENTS,
    RUNS_DIR,
    evaluate_team_policy,
    make_vec_env,
)

RUN_PREFIX = "simple_spread_sb3_adaptation"

MUJOCO_PPO_KWARGS = {
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.0,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
}


class SimpleSpreadEvalCallback(BaseCallback):
    """Periodic team-return evaluation and best-checkpoint saving."""

    def __init__(
        self,
        run_dir: Path,
        eval_freq: int,
        n_eval_episodes: int,
        eval_seed: int,
        max_cycles: int,
        terminate_on_success: bool,
        local_ratio: float,
        n_agents: int,
        agent_slots: int,
    ):
        super().__init__(verbose=1)
        self.run_dir = run_dir
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.eval_seed = eval_seed
        self.max_cycles = max_cycles
        self.terminate_on_success = terminate_on_success
        self.local_ratio = local_ratio
        self.n_agents = n_agents
        self.agent_slots = agent_slots

        self.best_mean_return = -np.inf
        self.next_eval_timestep = eval_freq
        self.timesteps = []
        self.results = []
        self.ep_lengths = []

    def _init_callback(self) -> None:
        (self.run_dir / "best_model").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "eval_logs").mkdir(parents=True, exist_ok=True)

    def _save_eval_logs(self) -> None:
        timesteps = np.array(self.timesteps, dtype=np.int64)
        np.savez(
            self.run_dir / "eval_logs" / "evaluations.npz",
            timesteps=timesteps,
            parallel_env_steps=timesteps // self.n_agents,
            per_env_steps=timesteps // self.agent_slots,
            results=np.array(self.results, dtype=np.float64),
            ep_lengths=np.array(self.ep_lengths, dtype=np.int64),
        )

    def _run_evaluation(self) -> None:
        returns, episode_lengths = evaluate_team_policy(
            model=self.model,
            seed=self.eval_seed + len(self.timesteps) * 10_000,
            n_eval_episodes=self.n_eval_episodes,
            max_cycles=self.max_cycles,
            terminate_on_success=self.terminate_on_success,
            local_ratio=self.local_ratio,
            deterministic=True,
        )
        mean_return = float(returns.mean())

        self.timesteps.append(self.num_timesteps)
        self.results.append(returns.tolist())
        self.ep_lengths.append(episode_lengths.tolist())
        self._save_eval_logs()

        self.logger.record("eval/mean_per_agent_return", mean_return)
        self.logger.record("eval/mean_episode_length", float(episode_lengths.mean()))

        if mean_return > self.best_mean_return:
            self.best_mean_return = mean_return
            self.model.save(str(self.run_dir / "best_model" / "best_model"))
            if self.verbose >= 1:
                print(
                    f"New best mean per-agent return: {mean_return:.3f} "
                    f"at timestep {self.num_timesteps}"
                )
        elif self.verbose >= 1:
            print(
                f"Eval mean per-agent return: {mean_return:.3f} "
                f"at timestep {self.num_timesteps}"
            )

    def _on_step(self) -> bool:
        while self.num_timesteps >= self.next_eval_timestep:
            self._run_evaluation()
            self.next_eval_timestep += self.eval_freq
        return True

    def _on_training_end(self) -> None:
        if not self.timesteps or self.timesteps[-1] != self.num_timesteps:
            self._run_evaluation()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timesteps", type=int, default=3_000_000)
    parser.add_argument("--eval_freq", type=int, default=50_000)
    parser.add_argument(
        "--n_eval_episodes",
        "--eval_episodes",
        dest="n_eval_episodes",
        type=int,
        default=4,
    )
    parser.add_argument("--max_cycles", type=int, default=DEFAULT_MAX_CYCLES)
    parser.add_argument("--local_ratio", type=float, default=LOCAL_RATIO)
    parser.add_argument("--num_vec_envs", type=int, default=DEFAULT_NUM_VEC_ENVS)
    parser.add_argument(
        "--reward_mode",
        type=str,
        choices=["greedy_local", "environment"],
        default="greedy_local",
    )
    parser.add_argument("--device", type=str, default="auto")
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
    return parser.parse_args()


def validate_args(args) -> None:
    positive_ints = {
        "--timesteps": args.timesteps,
        "--eval_freq": args.eval_freq,
        "--n_eval_episodes": args.n_eval_episodes,
        "--max_cycles": args.max_cycles,
        "--num_vec_envs": args.num_vec_envs,
    }
    for name, value in positive_ints.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive.")

    if not 0.0 <= args.local_ratio <= 1.0:
        raise ValueError("--local_ratio must be in [0, 1].")


def save_run_config(run_dir: Path, args, agent_slots: int, rollout_size: int) -> None:
    effective_timesteps = (
        ((args.timesteps + rollout_size - 1) // rollout_size) * rollout_size
    )
    config = {
        **vars(args),
        "n_agents": N_AGENTS,
        "agent_slots": agent_slots,
        "rollout_size": rollout_size,
        "effective_total_timesteps": effective_timesteps,
        "per_env_steps_per_rollout": MUJOCO_PPO_KWARGS["n_steps"],
        "parallel_env_steps_per_rollout": (
            MUJOCO_PPO_KWARGS["n_steps"] * args.num_vec_envs
        ),
        "method": "traced naive SB3 PPO adaptation over vectorized agent slots",
        "trace_source": (
            "Matched to the saved simple_spread_baseline_seed0 setup, which had "
            "the strongest saved seed-0 evaluation behavior in this workspace."
        ),
        "actor_observation": (
            "The SB3 policy receives one agent's local MPE2 observation."
        ),
        "critic_observation": (
            "The SB3 value function receives the same one-agent local observation."
        ),
        "credit_assignment": (
            "Each vectorized agent slot uses a greedy local reward by default. "
            "SB3 does not know which joint Simple Spread state the slot came "
            "from, nor whether another agent is chasing the same landmark."
        ),
        "coordination_note": (
            "There are no agent IDs, per-agent networks, centralized critic, "
            "or joint observation input."
        ),
        "training_reward": (
            "greedy_local replaces each agent's environment reward with "
            "-distance_to_nearest_landmark minus local collision penalties. "
            "environment uses the raw MPE2 reward."
        ),
        "evaluation_reward": (
            "Evaluation always uses the unwrapped Simple Spread environment "
            "mean per-agent return."
        ),
        "timestep_note": (
            "stable-baselines3 counts one timestep per agent slot, so one joint "
            "Simple Spread step across all agents adds n_agents timesteps per "
            "parallel environment copy."
        ),
        "ppo_impl": "stable_baselines3.PPO",
        "ppo_source": "The PPO constructor uses the same policy and hyperparameters as mujoco/train_mujoco.py.",
        "mujoco_ppo_kwargs": MUJOCO_PPO_KWARGS,
    }
    with open(run_dir / "run_config.json", "w", encoding="utf-8") as file_obj:
        json.dump(config, file_obj, indent=2, sort_keys=True)
        file_obj.write("\n")


def main():
    args = parse_args()
    validate_args(args)
    set_random_seed(args.seed)

    run_dir = RUNS_DIR / f"{RUN_PREFIX}_seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    train_env = make_vec_env(
        seed=args.seed,
        max_cycles=args.max_cycles,
        num_vec_envs=args.num_vec_envs,
        terminate_on_success=args.terminate_on_success,
        local_ratio=args.local_ratio,
        greedy_local_rewards=args.reward_mode == "greedy_local",
    )
    try:
        rollout_size = MUJOCO_PPO_KWARGS["n_steps"] * train_env.num_envs
        effective_timesteps = (
            ((args.timesteps + rollout_size - 1) // rollout_size) * rollout_size
        )
        save_run_config(
            run_dir=run_dir,
            args=args,
            agent_slots=train_env.num_envs,
            rollout_size=rollout_size,
        )

        model = PPO(
            policy="MlpPolicy",
            env=train_env,
            **MUJOCO_PPO_KWARGS,
            verbose=1,
            seed=args.seed,
            device=args.device,
            tensorboard_log=str(run_dir / "tb"),
        )

        eval_callback = SimpleSpreadEvalCallback(
            run_dir=run_dir,
            eval_freq=args.eval_freq,
            n_eval_episodes=args.n_eval_episodes,
            eval_seed=args.seed + 1000,
            max_cycles=args.max_cycles,
            terminate_on_success=args.terminate_on_success,
            local_ratio=args.local_ratio,
            n_agents=N_AGENTS,
            agent_slots=train_env.num_envs,
        )

        print(
            "Training Simple Spread SB3 adaptation with the traced naive "
            "MuJoCo-style PPO setup over "
            f"{train_env.num_envs} vectorized agent slots "
            f"({args.num_vec_envs} env copies x {N_AGENTS} agents)."
        )
        print(
            f"Environment setting: max_cycles={args.max_cycles}, "
            f"terminate_on_success={args.terminate_on_success}, "
            f"local_ratio={args.local_ratio}"
        )
        if args.reward_mode == "greedy_local":
            print(
                "Training rewards: greedy local nearest-landmark rewards with "
                "local collision penalties."
            )
        else:
            print("Training rewards: raw MPE2 Simple Spread environment rewards.")
        print("Policy/value observations: one local observation from one agent slot.")
        print("No agent IDs, per-agent networks, centralized critic, or joint input.")
        print(
            f"Rollout size: {rollout_size} agent-slot timesteps per PPO update "
            f"({MUJOCO_PPO_KWARGS['n_steps']} SB3 steps x "
            f"{train_env.num_envs} slots)."
        )
        if effective_timesteps != args.timesteps:
            print(
                f"Requested {args.timesteps} timesteps, rounded up to "
                f"{effective_timesteps} for a full SB3 rollout."
            )
        print(f"Evaluation cadence: every {args.eval_freq} agent-slot timesteps.")

        model.learn(total_timesteps=effective_timesteps, callback=eval_callback)
        model.save(str(run_dir / "final_model"))

        print(f"Saved final model to {run_dir / 'final_model.zip'}")
        print(f"Saved best model to {run_dir / 'best_model' / 'best_model.zip'}")
        print(f"Saved eval logs to {run_dir / 'eval_logs' / 'evaluations.npz'}")
        print(f"Saved run config to {run_dir / 'run_config.json'}")
    finally:
        train_env.close()


if __name__ == "__main__":
    main()
