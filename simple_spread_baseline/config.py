from __future__ import annotations

from dataclasses import asdict, dataclass, field

ENV_ID = "simple_spread"
RUN_PREFIX = "simple_spread_baseline"
N_AGENTS = 3
OBS_DIM = 18
STATE_DIM = 54
ACTION_DIM = 5
POLICY_ACTION_DIM = 2
DISCRETE_POLICY_ACTION_DIM = 1
DEFAULT_LOCAL_RATIO = 0.5
DEFAULT_MAX_CYCLES = 25
DEFAULT_CONTINUOUS_ACTIONS = False
COVERAGE_THRESHOLD = 0.1
SUCCESS_WINDOW = 5


@dataclass
class BaselineConfig:
    seed: int = 0
    timesteps: int = 1_000_000
    eval_freq: int = 10_000
    n_eval_episodes: int = 10
    log_interval: int = 1
    num_envs: int = 64
    rollout_steps: int = 25
    learning_rate: float = 5e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    update_epochs: int = 15
    minibatch_size: int = 1600
    hidden_sizes: tuple[int, int] = field(default_factory=lambda: (64, 64))
    local_ratio: float = DEFAULT_LOCAL_RATIO
    max_cycles: int = DEFAULT_MAX_CYCLES
    continuous_actions: bool = DEFAULT_CONTINUOUS_ACTIONS
    terminate_on_success: bool = True
    curriculum: bool = True
    curriculum_switch_step: int = 0
    device: str = "auto"
    render_mode: str | None = None

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["hidden_sizes"] = list(self.hidden_sizes)
        payload["env_id"] = ENV_ID
        payload["n_agents"] = N_AGENTS
        payload["obs_dim"] = OBS_DIM
        payload["state_dim"] = STATE_DIM
        payload["action_dim"] = ACTION_DIM
        payload["policy_action_dim"] = (
            POLICY_ACTION_DIM if self.continuous_actions else DISCRETE_POLICY_ACTION_DIM
        )
        payload["action_space"] = (
            "continuous_box_5" if self.continuous_actions else "discrete_5"
        )
        payload["coverage_threshold"] = COVERAGE_THRESHOLD
        payload["success_window"] = SUCCESS_WINDOW
        return payload
