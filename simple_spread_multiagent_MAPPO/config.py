from __future__ import annotations

from dataclasses import asdict, dataclass, field

from simple_spread_baseline.config import (
    ACTION_DIM,
    COVERAGE_THRESHOLD,
    DEFAULT_CONTINUOUS_ACTIONS,
    DEFAULT_LOCAL_RATIO,
    DEFAULT_MAX_CYCLES,
    N_AGENTS,
    OBS_DIM,
    STATE_DIM,
    SUCCESS_WINDOW,
)

RUN_PREFIX = "simple_spread_multiagent"


@dataclass
class MultiAgentConfig:
    seed: int = 0
    timesteps: int = 1_000_000
    eval_freq: int = 10_000
    n_eval_episodes: int = 10
    log_interval: int = 1
    num_envs: int = 64
    rollout_steps: int = 25
    learning_rate: float = 1e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    update_epochs: int = 4
    minibatch_size: int = 1600
    assignment_aux_coef: float = 0.1
    actor_hidden_sizes: tuple[int, int] = field(default_factory=lambda: (128, 128))
    critic_hidden_sizes: tuple[int, int] = field(default_factory=lambda: (128, 128))
    local_ratio: float = DEFAULT_LOCAL_RATIO
    max_cycles: int = DEFAULT_MAX_CYCLES
    continuous_actions: bool = DEFAULT_CONTINUOUS_ACTIONS
    terminate_on_success: bool = True
    curriculum: bool = False
    curriculum_switch_step: int = 0
    # Kept for backward-compatible checkpoint loading; the current joint actor
    # uses fixed-order team features and does not consume agent IDs.
    use_agent_id: bool = False
    expert_warmstart_samples: int = 0
    expert_warmstart_epochs: int = 0
    expert_warmstart_batch_size: int = 512
    expert_warmstart_lr: float = 3e-4
    device: str = "auto"
    render_mode: str | None = None

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["actor_hidden_sizes"] = list(self.actor_hidden_sizes)
        payload["critic_hidden_sizes"] = list(self.critic_hidden_sizes)
        payload["n_agents"] = N_AGENTS
        payload["obs_dim"] = OBS_DIM
        payload["state_dim"] = STATE_DIM
        payload["action_dim"] = ACTION_DIM
        payload["coverage_threshold"] = COVERAGE_THRESHOLD
        payload["success_window"] = SUCCESS_WINDOW
        return payload
