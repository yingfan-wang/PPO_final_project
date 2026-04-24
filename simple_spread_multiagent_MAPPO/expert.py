from __future__ import annotations

import itertools

import numpy as np

from simple_spread_baseline.config import DEFAULT_CONTINUOUS_ACTIONS, N_AGENTS
from simple_spread_baseline.env import SimpleSpreadEnv, SimpleSpreadVectorEnv
from simple_spread_multiagent_MAPPO.env import build_actor_inputs

LANDMARK_SLICE_START = 4
LANDMARK_DIM = 2
STOP_THRESHOLD = 0.16
CONTROL_SCALE = 0.5
DISCRETE_DT = 0.1
DISCRETE_DAMPING = 0.25
DISCRETE_ACCEL = 0.5
DISCRETE_COLLISION_DISTANCE = 0.30
DISCRETE_MPC_HORIZON = 2
DISCRETE_SPEED_PENALTY = 0.02
DISCRETE_COLLISION_PENALTY = 1.0
ASSIGNMENTS = np.asarray(list(itertools.permutations(range(N_AGENTS))), dtype=np.int64)
ASSIGNMENT_TO_INDEX = {
    tuple(assignment.tolist()): index for index, assignment in enumerate(ASSIGNMENTS)
}
ASSIGNMENT_ACTION_DIM = int(ASSIGNMENTS.shape[0])
DISCRETE_DIRECTION_VECTORS = np.asarray(
    [
        [0.0, 0.0],
        [-1.0, 0.0],
        [1.0, 0.0],
        [0.0, -1.0],
        [0.0, 1.0],
    ],
    dtype=np.float32,
)
JOINT_DISCRETE_ACTIONS = np.asarray(
    list(itertools.product(range(len(DISCRETE_DIRECTION_VECTORS)), repeat=N_AGENTS)),
    dtype=np.int64,
)


def _simulate_discrete_joint_dynamics(
    positions: np.ndarray,
    velocities: np.ndarray,
    directions: np.ndarray,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    predicted_positions = positions.astype(np.float32).copy()
    predicted_velocities = velocities.astype(np.float32).copy()
    for _ in range(horizon):
        predicted_positions = predicted_positions + DISCRETE_DT * predicted_velocities
        predicted_velocities = (
            (1.0 - DISCRETE_DAMPING) * predicted_velocities
            + DISCRETE_ACCEL * directions
        )
    return predicted_positions, predicted_velocities


def _discrete_target_indices_to_env_actions(
    observations: np.ndarray,
    target_indices: np.ndarray,
    horizon: int = DISCRETE_MPC_HORIZON,
    speed_penalty: float = DISCRETE_SPEED_PENALTY,
    collision_penalty: float = DISCRETE_COLLISION_PENALTY,
) -> np.ndarray:
    batch_size = observations.shape[0]
    positions = observations[..., 2:4].astype(np.float32)
    velocities = observations[..., :2].astype(np.float32)
    landmark_vectors = observations[
        ..., LANDMARK_SLICE_START : LANDMARK_SLICE_START + 6
    ].reshape(batch_size, N_AGENTS, N_AGENTS, LANDMARK_DIM)
    chosen_vectors = np.take_along_axis(
        landmark_vectors,
        target_indices[..., None, None].astype(np.int64),
        axis=2,
    ).squeeze(axis=2)
    target_positions = positions + chosen_vectors

    discrete_actions = np.zeros((batch_size, N_AGENTS, 1), dtype=np.int64)
    for env_idx in range(batch_size):
        best_joint_action = JOINT_DISCRETE_ACTIONS[0]
        best_cost: float | None = None
        for joint_action in JOINT_DISCRETE_ACTIONS:
            directions = DISCRETE_DIRECTION_VECTORS[joint_action]
            predicted_positions, predicted_velocities = _simulate_discrete_joint_dynamics(
                positions[env_idx],
                velocities[env_idx],
                directions,
                horizon=horizon,
            )
            distance_cost = float(
                np.linalg.norm(
                    target_positions[env_idx] - predicted_positions, axis=-1
                ).sum()
            )
            velocity_cost = float(
                speed_penalty * np.linalg.norm(predicted_velocities, axis=-1).sum()
            )
            pairwise_collision_cost = 0.0
            for first_idx in range(N_AGENTS):
                for second_idx in range(first_idx + 1, N_AGENTS):
                    separation = float(
                        np.linalg.norm(
                            predicted_positions[first_idx]
                            - predicted_positions[second_idx]
                        )
                    )
                    pairwise_collision_cost += collision_penalty * max(
                        0.0, DISCRETE_COLLISION_DISTANCE - separation
                    ) / DISCRETE_COLLISION_DISTANCE
            total_cost = distance_cost + velocity_cost + pairwise_collision_cost
            if best_cost is None or total_cost < best_cost:
                best_cost = total_cost
                best_joint_action = joint_action
        discrete_actions[env_idx, :, 0] = best_joint_action
    return discrete_actions


def target_indices_to_env_actions(
    observations: np.ndarray,
    target_indices: np.ndarray,
    continuous_actions: bool = DEFAULT_CONTINUOUS_ACTIONS,
    stop_threshold: float = STOP_THRESHOLD,
    control_scale: float = CONTROL_SCALE,
) -> np.ndarray:
    landmarks = observations[..., LANDMARK_SLICE_START : LANDMARK_SLICE_START + 6].reshape(
        observations.shape[0], N_AGENTS, N_AGENTS, LANDMARK_DIM
    )
    chosen = np.take_along_axis(
        landmarks,
        target_indices[..., None, None].astype(np.int64),
        axis=2,
    ).squeeze(axis=2)
    distances = np.linalg.norm(chosen, axis=-1, keepdims=True)
    controls = np.where(
        distances < stop_threshold,
        0.0,
        chosen / max(control_scale, 1e-6),
    ).astype(np.float32)
    control_norms = np.linalg.norm(controls, axis=-1, keepdims=True)
    controls = np.where(
        control_norms > 1.0,
        controls / np.maximum(control_norms, 1e-6),
        controls,
    ).astype(np.float32)

    if continuous_actions:
        env_actions = np.zeros((*controls.shape[:-1], 5), dtype=np.float32)
        horizontal = controls[..., 0]
        vertical = controls[..., 1]
        env_actions[..., 1] = np.maximum(0.0, -horizontal)
        env_actions[..., 2] = np.maximum(0.0, horizontal)
        env_actions[..., 3] = np.maximum(0.0, -vertical)
        env_actions[..., 4] = np.maximum(0.0, vertical)
        return env_actions

    return _discrete_target_indices_to_env_actions(observations, target_indices)


def assignment_indices_to_target_indices(assignment_indices: np.ndarray) -> np.ndarray:
    assignment_indices = np.asarray(assignment_indices, dtype=np.int64)
    return ASSIGNMENTS[assignment_indices]


def target_indices_to_assignment_indices(target_indices: np.ndarray) -> np.ndarray:
    target_indices = np.asarray(target_indices, dtype=np.int64)
    flat_targets = target_indices.reshape(-1, N_AGENTS)
    assignment_indices = np.asarray(
        [ASSIGNMENT_TO_INDEX[tuple(target.tolist())] for target in flat_targets],
        dtype=np.int64,
    )
    return assignment_indices.reshape(target_indices.shape[:-1])


def assignment_indices_to_env_actions(
    observations: np.ndarray,
    assignment_indices: np.ndarray,
    continuous_actions: bool = DEFAULT_CONTINUOUS_ACTIONS,
    stop_threshold: float = STOP_THRESHOLD,
    control_scale: float = CONTROL_SCALE,
) -> np.ndarray:
    target_indices = assignment_indices_to_target_indices(assignment_indices)
    return target_indices_to_env_actions(
        observations,
        target_indices=target_indices,
        continuous_actions=continuous_actions,
        stop_threshold=stop_threshold,
        control_scale=control_scale,
    )


def assignment_target_indices_from_observations(
    observations: np.ndarray,
) -> np.ndarray:
    observations = np.asarray(observations, dtype=np.float32)
    batch_size = observations.shape[0]
    agent_positions = observations[..., 2:4]
    landmark_rel_positions = observations[
        ..., LANDMARK_SLICE_START : LANDMARK_SLICE_START + 6
    ].reshape(batch_size, N_AGENTS, N_AGENTS, LANDMARK_DIM)
    landmark_positions = (
        agent_positions[:, :1, None, :] + landmark_rel_positions[:, :1, :, :]
    )
    pairwise_costs = np.linalg.norm(
        agent_positions[:, :, None, :] - landmark_positions,
        axis=-1,
    )
    assignment_costs = np.stack(
        [
            pairwise_costs[:, np.arange(N_AGENTS), assignment].sum(axis=1)
            for assignment in ASSIGNMENTS
        ],
        axis=1,
    )
    return assignment_costs.argmin(axis=1).astype(np.int64)


def assignment_target_indices(env: SimpleSpreadEnv) -> np.ndarray:
    world = env.raw_env.unwrapped.world
    agent_positions = np.asarray(
        [agent.state.p_pos for agent in world.agents], dtype=np.float32
    )
    landmark_positions = np.asarray(
        [landmark.state.p_pos for landmark in world.landmarks], dtype=np.float32
    )

    best_assignment: tuple[int, ...] | None = None
    best_cost: float | None = None
    for assignment in itertools.permutations(range(N_AGENTS)):
        cost = 0.0
        for agent_idx, landmark_idx in enumerate(assignment):
            cost += float(
                np.linalg.norm(landmark_positions[landmark_idx] - agent_positions[agent_idx])
            )
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_assignment = assignment
    assert best_assignment is not None
    return np.asarray(best_assignment, dtype=np.int64)


def collect_expert_dataset(
    *,
    samples: int,
    seed: int,
    num_envs: int,
    local_ratio: float,
    max_cycles: int,
    continuous_actions: bool,
    terminate_on_success: bool,
    curriculum: bool,
) -> tuple[np.ndarray, np.ndarray]:
    envs = SimpleSpreadVectorEnv(
        num_envs=num_envs,
        local_ratio=local_ratio,
        max_cycles=max_cycles,
        continuous_actions=continuous_actions,
        terminate_on_success=terminate_on_success,
        curriculum=curriculum,
    )
    if curriculum:
        envs.set_curriculum_stage(0)
    observations, _ = envs.reset(seed=seed)
    actor_observations: list[np.ndarray] = []
    assignment_indices: list[np.ndarray] = []

    try:
        while len(actor_observations) * num_envs < samples:
            actor_inputs = build_actor_inputs(observations).astype(np.float32)
            targets = np.stack(
                [assignment_target_indices(env) for env in envs.envs], axis=0
            ).astype(np.int64)
            assignments = target_indices_to_assignment_indices(targets)
            actor_observations.append(actor_inputs)
            assignment_indices.append(assignments)
            env_actions = target_indices_to_env_actions(
                observations,
                targets,
                continuous_actions=continuous_actions,
            )
            observations, _, _, _, _ = envs.step(env_actions)
    finally:
        envs.close()

    flat_actor_observations = np.concatenate(actor_observations, axis=0)
    flat_assignment_indices = np.concatenate(assignment_indices, axis=0).reshape(-1)
    return flat_actor_observations[:samples], flat_assignment_indices[:samples]
