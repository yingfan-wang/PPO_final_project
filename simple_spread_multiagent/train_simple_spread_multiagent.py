"""MAPPO-style PPO extension for MPE2 Simple Spread.

This implementation keeps the actor decentralized and shared across agents,
while training a centralized critic on the ordered joint observation. It is
intended to be compared directly against the local-observation SB3 PPO baseline.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Optional

BASE_DIR = Path(__file__).resolve().parent
MPLCONFIG_DIR = BASE_DIR / ".mplconfig"
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))

import numpy as np
import torch

from simple_spread_multiagent_common import (
    DEFAULT_MAX_CYCLES,
    DEFAULT_NUM_VEC_ENVS,
    LOCAL_RATIO,
    N_AGENTS,
    RUNS_DIR,
    CheckpointMetadata,
    SharedCentralizedActorCritic,
    evaluate_team_policy,
    infer_env_spaces,
    make_agent_id_matrix,
    make_parallel_env,
    mean_agent_reward,
    observations_to_arrays,
    save_checkpoint,
)


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class SimpleSpreadVectorRunner:
    """Small synchronous vector runner over independent parallel MPE2 envs."""

    def __init__(
        self,
        seed: int,
        max_cycles: int,
        num_vec_envs: int,
        terminate_on_success: bool,
        agents: list[str],
        local_obs_dim: int,
        device: torch.device,
        gamma: float,
        reward_mode: str,
    ):
        self.seed = seed
        self.max_cycles = max_cycles
        self.num_vec_envs = num_vec_envs
        self.terminate_on_success = terminate_on_success
        self.agents = agents
        self.local_obs_dim = local_obs_dim
        self.joint_obs_dim = local_obs_dim * len(agents)
        self.n_agents = len(agents)
        self.device = device
        self.gamma = gamma
        self.reward_mode = reward_mode

        self.envs = [
            make_parallel_env(
                max_cycles=max_cycles,
                terminate_on_success=terminate_on_success,
                render_mode=None,
                seed=seed + env_idx * 10_000,
            )
            for env_idx in range(num_vec_envs)
        ]
        self.current_observations = []
        for env_idx, env in enumerate(self.envs):
            observations, _ = env.reset(seed=seed + env_idx * 10_000)
            self.current_observations.append(observations)

        self.agent_ids = make_agent_id_matrix(num_vec_envs, self.n_agents)
        self.episode_returns = np.zeros(num_vec_envs, dtype=np.float64)
        self.episode_lengths = np.zeros(num_vec_envs, dtype=np.int64)
        self.completed_returns: list[float] = []
        self.completed_lengths: list[int] = []

    def close(self) -> None:
        for env in self.envs:
            env.close()

    def current_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        local_obs = np.zeros(
            (self.num_vec_envs, self.n_agents, self.local_obs_dim),
            dtype=np.float32,
        )
        joint_obs = np.zeros(
            (self.num_vec_envs, self.n_agents, self.joint_obs_dim),
            dtype=np.float32,
        )

        for env_idx, observations in enumerate(self.current_observations):
            env_local, env_joint = observations_to_arrays(
                observations=observations,
                agents=self.agents,
                local_obs_dim=self.local_obs_dim,
            )
            local_obs[env_idx] = env_local
            joint_obs[env_idx] = np.repeat(
                env_joint[None, :],
                self.n_agents,
                axis=0,
            )

        return local_obs, joint_obs

    def _values_for_observations(
        self,
        model: SharedCentralizedActorCritic,
        observations: dict[str, np.ndarray],
    ) -> Optional[np.ndarray]:
        if not all(agent in observations for agent in self.agents):
            return None

        local_obs, joint_obs = observations_to_arrays(
            observations=observations,
            agents=self.agents,
            local_obs_dim=self.local_obs_dim,
        )
        del local_obs
        joint_obs = np.repeat(joint_obs[None, :], self.n_agents, axis=0)
        agent_ids = np.arange(self.n_agents, dtype=np.int64)

        with torch.no_grad():
            values = model.values(
                joint_obs=torch.as_tensor(
                    joint_obs,
                    dtype=torch.float32,
                    device=self.device,
                ),
                agent_ids=torch.as_tensor(
                    agent_ids,
                    dtype=torch.long,
                    device=self.device,
                ),
            )
        return values.cpu().numpy()

    def collect_rollout(
        self,
        model: SharedCentralizedActorCritic,
        n_steps: int,
    ) -> dict[str, np.ndarray]:
        local_obs_buffer = np.zeros(
            (n_steps, self.num_vec_envs, self.n_agents, self.local_obs_dim),
            dtype=np.float32,
        )
        joint_obs_buffer = np.zeros(
            (n_steps, self.num_vec_envs, self.n_agents, self.joint_obs_dim),
            dtype=np.float32,
        )
        actions_buffer = np.zeros(
            (n_steps, self.num_vec_envs, self.n_agents),
            dtype=np.int64,
        )
        log_probs_buffer = np.zeros(
            (n_steps, self.num_vec_envs, self.n_agents),
            dtype=np.float32,
        )
        values_buffer = np.zeros(
            (n_steps, self.num_vec_envs, self.n_agents),
            dtype=np.float32,
        )
        rewards_buffer = np.zeros(
            (n_steps, self.num_vec_envs, self.n_agents),
            dtype=np.float32,
        )
        dones_buffer = np.zeros(
            (n_steps, self.num_vec_envs, self.n_agents),
            dtype=np.float32,
        )
        agent_ids_buffer = np.zeros(
            (n_steps, self.num_vec_envs, self.n_agents),
            dtype=np.int64,
        )

        for step_idx in range(n_steps):
            local_obs, joint_obs = self.current_arrays()
            agent_ids = self.agent_ids.copy()

            flat_local_obs = local_obs.reshape(-1, self.local_obs_dim)
            flat_joint_obs = joint_obs.reshape(-1, self.joint_obs_dim)
            flat_agent_ids = agent_ids.reshape(-1)

            with torch.no_grad():
                actions, log_probs, values = model.act(
                    local_obs=torch.as_tensor(
                        flat_local_obs,
                        dtype=torch.float32,
                        device=self.device,
                    ),
                    joint_obs=torch.as_tensor(
                        flat_joint_obs,
                        dtype=torch.float32,
                        device=self.device,
                    ),
                    agent_ids=torch.as_tensor(
                        flat_agent_ids,
                        dtype=torch.long,
                        device=self.device,
                    ),
                    deterministic=False,
                )

            actions_np = actions.cpu().numpy().reshape(
                self.num_vec_envs,
                self.n_agents,
            )
            log_probs_np = log_probs.cpu().numpy().reshape(
                self.num_vec_envs,
                self.n_agents,
            )
            values_np = values.cpu().numpy().reshape(
                self.num_vec_envs,
                self.n_agents,
            )

            local_obs_buffer[step_idx] = local_obs
            joint_obs_buffer[step_idx] = joint_obs
            actions_buffer[step_idx] = actions_np
            log_probs_buffer[step_idx] = log_probs_np
            values_buffer[step_idx] = values_np
            agent_ids_buffer[step_idx] = agent_ids

            for env_idx, env in enumerate(self.envs):
                action_dict = {
                    agent: int(actions_np[env_idx, agent_idx])
                    for agent_idx, agent in enumerate(self.agents)
                    if agent in env.agents
                }
                observations, rewards, terminations, truncations, _ = env.step(
                    action_dict
                )

                if self.reward_mode == "team":
                    reward_values = np.full(
                        self.n_agents,
                        mean_agent_reward(rewards),
                        dtype=np.float32,
                    )
                else:
                    reward_values = np.array(
                        [float(rewards.get(agent, 0.0)) for agent in self.agents],
                        dtype=np.float32,
                    )
                done_values = np.array(
                    [
                        bool(terminations.get(agent, False))
                        or bool(truncations.get(agent, False))
                        for agent in self.agents
                    ],
                    dtype=np.float32,
                )

                env_done = (not env.agents) or bool(done_values.all())
                timed_out = (
                    env_done
                    and all(bool(truncations.get(agent, False)) for agent in self.agents)
                    and not any(
                        bool(terminations.get(agent, False)) for agent in self.agents
                    )
                )
                if timed_out:
                    terminal_values = self._values_for_observations(model, observations)
                    if terminal_values is not None:
                        reward_values += self.gamma * terminal_values.astype(np.float32)

                rewards_buffer[step_idx, env_idx] = reward_values
                dones_buffer[step_idx, env_idx] = done_values
                self.episode_returns[env_idx] += mean_agent_reward(rewards)
                self.episode_lengths[env_idx] += 1

                if env_done:
                    self.completed_returns.append(float(self.episode_returns[env_idx]))
                    self.completed_lengths.append(int(self.episode_lengths[env_idx]))
                    reset_observations, _ = env.reset()
                    self.current_observations[env_idx] = reset_observations
                    self.episode_returns[env_idx] = 0.0
                    self.episode_lengths[env_idx] = 0
                else:
                    self.current_observations[env_idx] = observations

        final_local_obs, final_joint_obs = self.current_arrays()
        flat_final_joint_obs = final_joint_obs.reshape(-1, self.joint_obs_dim)
        flat_final_agent_ids = self.agent_ids.reshape(-1)
        with torch.no_grad():
            last_values = model.values(
                joint_obs=torch.as_tensor(
                    flat_final_joint_obs,
                    dtype=torch.float32,
                    device=self.device,
                ),
                agent_ids=torch.as_tensor(
                    flat_final_agent_ids,
                    dtype=torch.long,
                    device=self.device,
                ),
            )
        del final_local_obs

        return {
            "local_obs": local_obs_buffer,
            "joint_obs": joint_obs_buffer,
            "agent_ids": agent_ids_buffer,
            "actions": actions_buffer,
            "old_log_probs": log_probs_buffer,
            "values": values_buffer,
            "rewards": rewards_buffer,
            "dones": dones_buffer,
            "last_values": last_values.cpu().numpy().reshape(
                self.num_vec_envs,
                self.n_agents,
            ),
        }


def compute_gae(
    rollout: dict[str, np.ndarray],
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    rewards = rollout["rewards"]
    dones = rollout["dones"]
    values = rollout["values"]
    last_values = rollout["last_values"]

    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_gae = np.zeros_like(last_values, dtype=np.float32)

    for step_idx in reversed(range(rewards.shape[0])):
        if step_idx == rewards.shape[0] - 1:
            next_values = last_values
        else:
            next_values = values[step_idx + 1]

        next_non_terminal = 1.0 - dones[step_idx]
        delta = rewards[step_idx] + gamma * next_values * next_non_terminal
        delta -= values[step_idx]
        last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        advantages[step_idx] = last_gae

    returns = advantages + values
    return advantages, returns


def flatten_rollout(
    rollout: dict[str, np.ndarray],
    advantages: np.ndarray,
    returns: np.ndarray,
) -> dict[str, np.ndarray]:
    return {
        "local_obs": rollout["local_obs"].reshape(-1, rollout["local_obs"].shape[-1]),
        "joint_obs": rollout["joint_obs"].reshape(-1, rollout["joint_obs"].shape[-1]),
        "agent_ids": rollout["agent_ids"].reshape(-1),
        "actions": rollout["actions"].reshape(-1),
        "old_log_probs": rollout["old_log_probs"].reshape(-1),
        "advantages": advantages.reshape(-1),
        "returns": returns.reshape(-1),
        "old_values": rollout["values"].reshape(-1),
    }


def ppo_update(
    model: SharedCentralizedActorCritic,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, np.ndarray],
    batch_size: int,
    n_epochs: int,
    clip_range: float,
    ent_coef: float,
    vf_coef: float,
    max_grad_norm: float,
    target_kl: Optional[float],
    device: torch.device,
) -> dict[str, float]:
    batch_count = batch["actions"].shape[0]
    advantages = batch["advantages"].astype(np.float32)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    stats = {
        "policy_loss": [],
        "value_loss": [],
        "entropy": [],
        "approx_kl": [],
        "clip_fraction": [],
    }

    for _ in range(n_epochs):
        indices = np.random.permutation(batch_count)
        for start in range(0, batch_count, batch_size):
            minibatch_indices = indices[start : start + batch_size]

            local_obs = torch.as_tensor(
                batch["local_obs"][minibatch_indices],
                dtype=torch.float32,
                device=device,
            )
            joint_obs = torch.as_tensor(
                batch["joint_obs"][minibatch_indices],
                dtype=torch.float32,
                device=device,
            )
            agent_ids = torch.as_tensor(
                batch["agent_ids"][minibatch_indices],
                dtype=torch.long,
                device=device,
            )
            actions = torch.as_tensor(
                batch["actions"][minibatch_indices],
                dtype=torch.long,
                device=device,
            )
            old_log_probs = torch.as_tensor(
                batch["old_log_probs"][minibatch_indices],
                dtype=torch.float32,
                device=device,
            )
            minibatch_advantages = torch.as_tensor(
                advantages[minibatch_indices],
                dtype=torch.float32,
                device=device,
            )
            minibatch_returns = torch.as_tensor(
                batch["returns"][minibatch_indices],
                dtype=torch.float32,
                device=device,
            )

            log_probs, entropy, values = model.evaluate_actions(
                local_obs=local_obs,
                joint_obs=joint_obs,
                agent_ids=agent_ids,
                actions=actions,
            )
            log_ratio = log_probs - old_log_probs
            ratio = torch.exp(log_ratio)

            unclipped_policy_loss = minibatch_advantages * ratio
            clipped_policy_loss = minibatch_advantages * torch.clamp(
                ratio,
                1.0 - clip_range,
                1.0 + clip_range,
            )
            policy_loss = -torch.min(
                unclipped_policy_loss,
                clipped_policy_loss,
            ).mean()
            value_loss = torch.nn.functional.mse_loss(values, minibatch_returns)
            entropy_loss = -entropy.mean()
            loss = policy_loss + vf_coef * value_loss + ent_coef * entropy_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                approx_kl = ((ratio - 1.0) - log_ratio).mean()
                clip_fraction = (
                    (torch.abs(ratio - 1.0) > clip_range).float().mean()
                )

            stats["policy_loss"].append(float(policy_loss.detach().cpu()))
            stats["value_loss"].append(float(value_loss.detach().cpu()))
            stats["entropy"].append(float(entropy.mean().detach().cpu()))
            stats["approx_kl"].append(float(approx_kl.detach().cpu()))
            stats["clip_fraction"].append(float(clip_fraction.detach().cpu()))

        if target_kl is not None and float(np.mean(stats["approx_kl"][-1:])) > target_kl:
            break

    return {key: float(np.mean(values)) for key, values in stats.items()}


def format_table_value(value) -> str:
    if value == "":
        return ""
    if isinstance(value, int):
        return str(value)
    value = float(value)
    if np.isnan(value):
        return "nan"
    if value != 0.0 and (abs(value) >= 10_000 or abs(value) < 0.001):
        return f"{value:.3e}"
    return f"{value:.3f}"


def print_training_table(
    update_idx: int,
    n_updates: int,
    total_timesteps: int,
    start_time: float,
    recent_return: float,
    recent_length: float,
    train_stats: dict[str, float],
) -> None:
    elapsed = max(time.time() - start_time, 1e-9)
    fps = int(total_timesteps / elapsed)
    rows = [
        ("rollout/", ""),
        ("    ep_len_mean", recent_length),
        ("    ep_rew_mean", recent_return),
        ("time/", ""),
        ("    fps", fps),
        ("    iterations", update_idx),
        ("    total_updates", n_updates),
        ("    total_timesteps", total_timesteps),
        ("train/", ""),
        ("    approx_kl", train_stats["approx_kl"]),
        ("    clip_fraction", train_stats["clip_fraction"]),
        ("    entropy_loss", -train_stats["entropy"]),
        ("    policy_gradient_loss", train_stats["policy_loss"]),
        ("    value_loss", train_stats["value_loss"]),
    ]
    left_width = max(len(name) for name, _ in rows)
    right_width = max(len(format_table_value(value)) for _, value in rows)
    divider = "-" * (left_width + right_width + 7)

    print(divider)
    for name, value in rows:
        print(
            f"| {name:<{left_width}} | "
            f"{format_table_value(value):>{right_width}} |"
        )
    print(divider)


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
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_range", type=float, default=0.2)
    parser.add_argument("--ent_coef", type=float, default=0.0)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--target_kl", type=float, default=None)
    parser.add_argument("--hidden_sizes", type=int, nargs="+", default=[64, 64])
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument(
        "--reward_mode",
        type=str,
        choices=["team", "individual"],
        default="team",
    )
    parser.set_defaults(use_agent_id=True)
    parser.add_argument("--use_agent_id", dest="use_agent_id", action="store_true")
    parser.add_argument("--no_agent_id", dest="use_agent_id", action="store_false")
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
        "--n_steps": args.n_steps,
        "--batch_size": args.batch_size,
        "--n_epochs": args.n_epochs,
        "--log_interval": args.log_interval,
    }
    for name, value in positive_ints.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive.")

    if not 0.0 < args.gamma <= 1.0:
        raise ValueError("--gamma must be in (0, 1].")
    if not 0.0 < args.gae_lambda <= 1.0:
        raise ValueError("--gae_lambda must be in (0, 1].")
    if args.clip_range <= 0.0:
        raise ValueError("--clip_range must be positive.")
    if args.learning_rate <= 0.0:
        raise ValueError("--learning_rate must be positive.")
    if not args.hidden_sizes or any(size <= 0 for size in args.hidden_sizes):
        raise ValueError("--hidden_sizes must contain positive layer sizes.")

    rollout_size = args.n_steps * args.num_vec_envs * N_AGENTS
    if args.batch_size > rollout_size:
        raise ValueError(
            f"--batch_size ({args.batch_size}) cannot exceed one rollout "
            f"({rollout_size} = n_steps * num_vec_envs * n_agents)."
        )
    if rollout_size % args.batch_size != 0:
        raise ValueError(
            f"--batch_size ({args.batch_size}) must divide the rollout size "
            f"({rollout_size} = n_steps * num_vec_envs * n_agents)."
        )


def save_run_config(
    run_dir: Path,
    args,
    agent_slots: int,
    rollout_size: int,
    metadata: CheckpointMetadata,
) -> dict:
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
        "method": "MAPPO-style shared actor with centralized critic",
        "actor_observation": (
            "Each agent receives only its local MPE2 observation, plus a one-hot "
            "agent ID when use_agent_id is true."
        ),
        "critic_observation": (
            "The critic receives the ordered concatenation of all agent "
            "observations, plus the same one-hot agent ID when enabled."
        ),
        "credit_assignment": (
            "PPO advantages are computed per agent. In team reward mode, each "
            "agent receives the same mean team reward before advantage "
            "estimation; in individual mode, each agent keeps its environment "
            "reward. Both modes use the centralized value baseline."
        ),
        "reward_mode_note": (
            "team: train all agents on the mean per-agent team reward; "
            "individual: train each agent on its own MPE2 reward."
        ),
        "timestep_note": (
            "This script counts one timestep per agent slot, matching the "
            "Simple Spread baseline semantics."
        ),
        "checkpoint_metadata": metadata.__dict__,
    }
    with open(run_dir / "run_config.json", "w", encoding="utf-8") as file_obj:
        json.dump(config, file_obj, indent=2, sort_keys=True)
        file_obj.write("\n")
    return config


def save_eval_logs(
    run_dir: Path,
    timesteps: list[int],
    results: list[list[float]],
    ep_lengths: list[list[int]],
    n_agents: int,
    agent_slots: int,
) -> None:
    timestep_array = np.array(timesteps, dtype=np.int64)
    np.savez(
        run_dir / "eval_logs" / "evaluations.npz",
        timesteps=timestep_array,
        parallel_env_steps=timestep_array // n_agents,
        per_env_steps=timestep_array // agent_slots,
        results=np.array(results, dtype=np.float64),
        ep_lengths=np.array(ep_lengths, dtype=np.int64),
    )


def main():
    args = parse_args()
    validate_args(args)
    set_random_seed(args.seed)
    device = resolve_device(args.device)

    agents, local_obs_dim, action_dim = infer_env_spaces(
        max_cycles=args.max_cycles,
        terminate_on_success=args.terminate_on_success,
    )
    joint_obs_dim = local_obs_dim * len(agents)
    metadata = CheckpointMetadata(
        agents=agents,
        local_obs_dim=local_obs_dim,
        joint_obs_dim=joint_obs_dim,
        action_dim=action_dim,
        n_agents=len(agents),
        hidden_sizes=list(args.hidden_sizes),
        use_agent_id=args.use_agent_id,
    )

    run_dir = RUNS_DIR / f"simple_spread_mappo_seed{args.seed}"
    (run_dir / "best_model").mkdir(parents=True, exist_ok=True)
    (run_dir / "eval_logs").mkdir(parents=True, exist_ok=True)

    rollout_size = args.n_steps * args.num_vec_envs * len(agents)
    effective_timesteps = (
        ((args.timesteps + rollout_size - 1) // rollout_size) * rollout_size
    )
    n_updates = effective_timesteps // rollout_size
    agent_slots = args.num_vec_envs * len(agents)
    config = save_run_config(
        run_dir=run_dir,
        args=args,
        agent_slots=agent_slots,
        rollout_size=rollout_size,
        metadata=metadata,
    )

    model = SharedCentralizedActorCritic(
        local_obs_dim=local_obs_dim,
        joint_obs_dim=joint_obs_dim,
        action_dim=action_dim,
        n_agents=len(agents),
        hidden_sizes=list(args.hidden_sizes),
        use_agent_id=args.use_agent_id,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, eps=1e-5)

    runner = SimpleSpreadVectorRunner(
        seed=args.seed,
        max_cycles=args.max_cycles,
        num_vec_envs=args.num_vec_envs,
        terminate_on_success=args.terminate_on_success,
        agents=agents,
        local_obs_dim=local_obs_dim,
        device=device,
        gamma=args.gamma,
        reward_mode=args.reward_mode,
    )

    eval_timesteps: list[int] = []
    eval_results: list[list[float]] = []
    eval_lengths: list[list[int]] = []
    best_mean_return = -np.inf
    next_eval_timestep = args.eval_freq
    total_timesteps = 0
    training_start_time = time.time()

    print(
        "Training Simple Spread MAPPO-style PPO with "
        f"{agent_slots} agent slots ({args.num_vec_envs} env copies x "
        f"{len(agents)} agents)."
    )
    print(
        "Actor: shared local-observation policy"
        f"{' with agent ID' if args.use_agent_id else ''}."
    )
    print(
        "Critic: centralized value function over ordered joint observations"
        f"{' with agent ID' if args.use_agent_id else ''}."
    )
    print(f"Training reward mode: {args.reward_mode}.")
    print(f"Device: {device}")
    print(
        f"Rollout size: {rollout_size} agent-slot timesteps per PPO update."
    )
    if effective_timesteps != args.timesteps:
        print(
            f"Requested {args.timesteps} timesteps, rounded up to "
            f"{effective_timesteps} for full PPO rollouts."
        )

    try:
        for update_idx in range(1, n_updates + 1):
            model.train()
            rollout = runner.collect_rollout(model=model, n_steps=args.n_steps)
            advantages, returns = compute_gae(
                rollout=rollout,
                gamma=args.gamma,
                gae_lambda=args.gae_lambda,
            )
            batch = flatten_rollout(
                rollout=rollout,
                advantages=advantages,
                returns=returns,
            )
            train_stats = ppo_update(
                model=model,
                optimizer=optimizer,
                batch=batch,
                batch_size=args.batch_size,
                n_epochs=args.n_epochs,
                clip_range=args.clip_range,
                ent_coef=args.ent_coef,
                vf_coef=args.vf_coef,
                max_grad_norm=args.max_grad_norm,
                target_kl=args.target_kl,
                device=device,
            )
            total_timesteps += rollout_size

            recent_returns = runner.completed_returns[-20:]
            recent_lengths = runner.completed_lengths[-20:]
            if recent_returns:
                recent_return = float(np.mean(recent_returns))
                recent_length = float(np.mean(recent_lengths))
            else:
                recent_return = float("nan")
                recent_length = float("nan")

            if (
                update_idx == 1
                or update_idx == n_updates
                or update_idx % args.log_interval == 0
            ):
                print_training_table(
                    update_idx=update_idx,
                    n_updates=n_updates,
                    total_timesteps=total_timesteps,
                    start_time=training_start_time,
                    recent_return=recent_return,
                    recent_length=recent_length,
                    train_stats=train_stats,
                )

            while total_timesteps >= next_eval_timestep:
                model.eval()
                returns_eval, lengths_eval = evaluate_team_policy(
                    model=model,
                    metadata=metadata,
                    seed=args.seed + 1000 + len(eval_timesteps) * 10_000,
                    n_eval_episodes=args.n_eval_episodes,
                    max_cycles=args.max_cycles,
                    terminate_on_success=args.terminate_on_success,
                    device=device,
                    deterministic=True,
                )
                mean_return = float(returns_eval.mean())
                eval_timesteps.append(total_timesteps)
                eval_results.append(returns_eval.tolist())
                eval_lengths.append(lengths_eval.tolist())
                save_eval_logs(
                    run_dir=run_dir,
                    timesteps=eval_timesteps,
                    results=eval_results,
                    ep_lengths=eval_lengths,
                    n_agents=len(agents),
                    agent_slots=agent_slots,
                )

                if mean_return > best_mean_return:
                    best_mean_return = mean_return
                    save_checkpoint(
                        run_dir / "best_model" / "best_model.pt",
                        model=model,
                        metadata=metadata,
                        config=config,
                    )
                    print(
                        f"New best mean per-agent return: {mean_return:.3f} "
                        f"at timestep {total_timesteps}"
                    )
                else:
                    print(
                        f"Eval mean per-agent return: {mean_return:.3f} "
                        f"at timestep {total_timesteps}"
                    )
                next_eval_timestep += args.eval_freq

        if not eval_timesteps or eval_timesteps[-1] != total_timesteps:
            model.eval()
            returns_eval, lengths_eval = evaluate_team_policy(
                model=model,
                metadata=metadata,
                seed=args.seed + 1000 + len(eval_timesteps) * 10_000,
                n_eval_episodes=args.n_eval_episodes,
                max_cycles=args.max_cycles,
                terminate_on_success=args.terminate_on_success,
                device=device,
                deterministic=True,
            )
            mean_return = float(returns_eval.mean())
            eval_timesteps.append(total_timesteps)
            eval_results.append(returns_eval.tolist())
            eval_lengths.append(lengths_eval.tolist())
            save_eval_logs(
                run_dir=run_dir,
                timesteps=eval_timesteps,
                results=eval_results,
                ep_lengths=eval_lengths,
                n_agents=len(agents),
                agent_slots=agent_slots,
            )
            if mean_return > best_mean_return:
                best_mean_return = mean_return
                save_checkpoint(
                    run_dir / "best_model" / "best_model.pt",
                    model=model,
                    metadata=metadata,
                    config=config,
                )

        save_checkpoint(
            run_dir / "final_model.pt",
            model=model,
            metadata=metadata,
            config=config,
        )

        print(f"Saved final model to {run_dir / 'final_model.pt'}")
        print(f"Saved best model to {run_dir / 'best_model' / 'best_model.pt'}")
        print(f"Saved eval logs to {run_dir / 'eval_logs' / 'evaluations.npz'}")
        print(f"Saved run config to {run_dir / 'run_config.json'}")
    finally:
        runner.close()


if __name__ == "__main__":
    main()
