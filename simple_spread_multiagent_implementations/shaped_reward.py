import numpy as np

# def shape_reward(agent_obs, base_reward):
#     landmark_positions = agent_obs[4:10]
#     distances = [
#         np.sqrt(landmark_positions[2*i]**2 + landmark_positions[2*i+1]**2)
#         for i in range(3)
#     ]
#     nearest_dist = min(distances)
#     other_agent_obs = agent_obs[10:14]
#     collision_penalty = 0.0
#     for i in range(2):
#         dx = other_agent_obs[2*i]
#         dy = other_agent_obs[2*i + 1]
#         dist = np.sqrt(dx**2 + dy**2)
#         if dist < 0.5:
#             collision_penalty -= 0.3 * (0.5 - dist) / 0.5
#     return (
#         base_reward
#         + 3.0 * np.exp(-10.0 * nearest_dist)
#         - 0.5 * nearest_dist
#         + collision_penalty
#     )

# DEFAULT REWARD STRUCTURE
def shape_reward(agent_obs, base_reward):
    return base_reward

TIMESTEPS = 1_500_000

HYPERPARAMS = {
    "learning_rate": 3e-4,
    "n_steps": 512,
    "batch_size": 256,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
}