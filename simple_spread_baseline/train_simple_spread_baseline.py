"""Straightforward PPO baseline for MPE2 Simple Spread.

This baseline intentionally keeps the learning setup simple:

- one shared PPO policy for all agents
- each agent acts only from its own local observation
- no communication channel
- no centralized critic

The goal is to provide a clean SB3 baseline that is easy to compare against the
MuJoCo scripts in this repo, while still evaluating checkpoints using the true
team objective of the cooperative environment.
"""

import argparse
import json
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MPLCONFIG_DIR = BASE_DIR / ".mplconfig"
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))

import numpy as np
from torch import nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import set_random_seed

from simple_spread_baseline_common import (
    DEFAULT_MAX_CYCLES,
    DEFAULT_NUM_VEC_ENVS,
    LOCAL_RATIO,
    N_AGENTS,
    RUNS_DIR,
    evaluate_team_policy,
    make_vec_env,
)

STANDARD_POLICY_KWARGS = {
    "activation_fn": nn.Tanh,
    "net_arch": {"pi": [64, 64], "vf": [64, 64]},
    "ortho_init": True,
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
    parser.add_argument("--eval_freq", type=int, default=20_000)
    parser.add_argument(
        "--n_eval_episodes",
        "--eval_episodes",
        dest="n_eval_episodes",
        type=int,
        default=12,
    )
    parser.add_argument("--max_cycles", type=int, default=DEFAULT_MAX_CYCLES)
    parser.add_argument("--num_vec_envs", type=int, default=DEFAULT_NUM_VEC_ENVS)
    parser.add_argument("--n_steps", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=256)
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


def validate_args(args):
    if args.timesteps <= 0:
        raise ValueError("--timesteps must be positive.")
    if args.eval_freq <= 0:
        raise ValueError("--eval_freq must be positive.")
    if args.n_eval_episodes <= 0:
        raise ValueError("--n_eval_episodes must be positive.")
    if args.max_cycles <= 0:
        raise ValueError("--max_cycles must be positive.")
    if args.num_vec_envs <= 0:
        raise ValueError("--num_vec_envs must be positive.")
    if args.n_steps <= 0:
        raise ValueError("--n_steps must be positive.")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive.")
    rollout_size = args.n_steps * args.num_vec_envs * N_AGENTS
    if args.batch_size > rollout_size:
        raise ValueError(
            f"--batch_size ({args.batch_size}) cannot exceed one rollout "
            f"({rollout_size} = n_steps * num_vec_envs * n_agents)."
        )
    if rollout_size % args.batch_size != 0:
        raise ValueError(
            f"--batch_size ({args.batch_size}) must divide the rollout size "
            f"({rollout_size} = n_steps * num_vec_envs * n_agents) so PPO "
            "minibatches stay consistent across updates."
        )


def save_run_config(run_dir: Path, args, agent_slots: int, rollout_size: int) -> None:
    effective_timesteps = (
        ((args.timesteps + rollout_size - 1) // rollout_size) * rollout_size
    )
    config = {
        **vars(args),
        "n_agents": N_AGENTS,
        "local_ratio": LOCAL_RATIO,
        "agent_slots": agent_slots,
        "rollout_size": rollout_size,
        "effective_total_timesteps": effective_timesteps,
        "per_env_steps_per_rollout": args.n_steps,
        "parallel_env_steps_per_rollout": args.n_steps * args.num_vec_envs,
        "timestep_note": (
            "stable-baselines3 counts one timestep per agent slot, so one joint "
            "Simple Spread step across all agents adds n_agents timesteps per "
            "parallel environment copy."
        ),
        "ppo_impl": "stable_baselines3.PPO",
        "policy_kwargs": {
            "activation_fn": "Tanh",
            "net_arch": {"pi": [64, 64], "vf": [64, 64]},
            "ortho_init": True,
        },
        "hardcoded_ppo_settings": {
            "clip_range_vf": None,
            "normalize_advantage": True,
            "ent_coef": 0.0,
            "use_sde": False,
            "sde_sample_freq": -1,
            "target_kl": None,
        },
    }
    with open(run_dir / "run_config.json", "w", encoding="utf-8") as file_obj:
        json.dump(config, file_obj, indent=2, sort_keys=True)
        file_obj.write("\n")


def main():
    args = parse_args()
    validate_args(args)
    set_random_seed(args.seed)

    run_dir = RUNS_DIR / f"simple_spread_baseline_seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    train_env = make_vec_env(
        seed=args.seed,
        max_cycles=args.max_cycles,
        num_vec_envs=args.num_vec_envs,
        terminate_on_success=args.terminate_on_success,
    )
    try:
        rollout_size = args.n_steps * train_env.num_envs
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
            learning_rate=3e-4,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            clip_range_vf=None,
            normalize_advantage=True,
            ent_coef=0.0,
            vf_coef=0.5,
            max_grad_norm=0.5,
            use_sde=False,
            sde_sample_freq=-1,
            target_kl=None,
            verbose=1,
            seed=args.seed,
            device=args.device,
            tensorboard_log=str(run_dir / "tb"),
            policy_kwargs=STANDARD_POLICY_KWARGS,
        )

        eval_callback = SimpleSpreadEvalCallback(
            run_dir=run_dir,
            eval_freq=args.eval_freq,
            n_eval_episodes=args.n_eval_episodes,
            eval_seed=args.seed + 1000,
            max_cycles=args.max_cycles,
            terminate_on_success=args.terminate_on_success,
            n_agents=N_AGENTS,
            agent_slots=train_env.num_envs,
        )

        print(
            "Training Simple Spread PPO baseline with "
            f"{train_env.num_envs} vectorized agent slots "
            f"({args.num_vec_envs} env copies x {N_AGENTS} agents)."
        )
        print(
            f"Environment setting: max_cycles={args.max_cycles}, "
            f"terminate_on_success={args.terminate_on_success}"
        )
        print(
            "Timestep semantics: stable-baselines3 counts one timestep per agent "
            f"slot, so one joint environment step across all {N_AGENTS} agents "
            f"advances {train_env.num_envs} timesteps with {args.num_vec_envs} "
            "parallel env copies."
        )
        print(
            f"Rollout size: {rollout_size} agent-slot timesteps per PPO update "
            f"({args.n_steps} steps per environment copy)."
        )
        print(
            "Policy/value network: hard-coded SB3 PPO MLP with separate "
            "64x64 tanh heads and standard PPO loss settings."
        )
        if effective_timesteps != args.timesteps:
            print(
                f"Requested {args.timesteps} timesteps, which PPO rounds up to "
                f"{effective_timesteps} so training ends on a full rollout."
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
