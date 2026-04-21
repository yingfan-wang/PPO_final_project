"""Shared helpers for the traced Simple Spread SB3 PPO adaptation.

This multi-agent folder now uses the same naive SB3 agent-slot PPO path that
produced the strongest saved Simple Spread behavior in this workspace.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from simple_spread_baseline.simple_spread_baseline_common import (  # noqa: E402
    DEFAULT_MAX_CYCLES,
    DEFAULT_NUM_VEC_ENVS,
    GREEDY_COLLISION_DISTANCE,
    GREEDY_COLLISION_PENALTY,
    LOCAL_RATIO,
    N_AGENTS,
    GreedyLocalRewardWrapper,
    SB3VectorEnvAdapter,
    evaluate_team_policy,
    greedy_local_reward_from_obs,
    make_parallel_env,
    make_vec_env,
    mean_agent_reward,
    predict_parallel_actions,
    resolve_model_path,
    safe_std,
)

BASE_DIR = Path(__file__).resolve().parent
RUNS_DIR = BASE_DIR / "runs"

__all__ = [
    "BASE_DIR",
    "DEFAULT_MAX_CYCLES",
    "DEFAULT_NUM_VEC_ENVS",
    "GREEDY_COLLISION_DISTANCE",
    "GREEDY_COLLISION_PENALTY",
    "LOCAL_RATIO",
    "N_AGENTS",
    "RUNS_DIR",
    "GreedyLocalRewardWrapper",
    "SB3VectorEnvAdapter",
    "evaluate_team_policy",
    "greedy_local_reward_from_obs",
    "make_parallel_env",
    "make_vec_env",
    "mean_agent_reward",
    "predict_parallel_actions",
    "resolve_model_path",
    "safe_std",
]
