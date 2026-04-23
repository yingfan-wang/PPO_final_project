from pettingzoo.mpe import simple_spread_v3
from stable_baselines3 import PPO
import supersuit as ss

model = PPO.load("runs/simple_spread/best_model/best_model")

env = simple_spread_v3.parallel_env(
    N=3,
    max_cycles=25,
    continuous_actions=True,
    render_mode="human",
    dynamic_rescaling=False
)
env = ss.pettingzoo_env_to_vec_env_v1(env)
env = ss.concat_vec_envs_v1(env, 1, base_class="stable_baselines3")

obs = env.reset()
total_reward = 0
runs = 0
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    if done.all():
        total_reward += reward
        runs += 1
        print("Average reward", total_reward / runs) 
        obs = env.reset()

env.close()