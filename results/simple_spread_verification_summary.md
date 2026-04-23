# Simple Spread Verification Summary

Date: 2026-04-22
Environment: `mpe2.simple_spread_v3`
Conda env: `rl_final`

## External Checks

- Official MPE2 Simple Spread docs confirm:
  - 3 agents, 3 landmarks
  - default discrete action space `Discrete(5)`
  - optional continuous action space `Box(0.0, 1.0, (5,))`
  - shared global reward from landmark coverage distance
  - local collision penalty weighted by `local_ratio`
  - `terminate_on_success=True` is an officially supported stronger training signal
- Official MAPPO guidance and scripts emphasize:
  - centralized value learning
  - parameter sharing / structured coordination
  - careful control of PPO data reuse
  - the importance of rollout-thread count and PPO update settings in MPE

Sources:

- https://mpe2.farama.org/environments/simple_spread/
- https://bair.berkeley.edu/blog/2021/07/14/mappo/
- https://raw.githubusercontent.com/marlbenchmark/on-policy/main/onpolicy/scripts/train_mpe_scripts/train_mpe_spread.sh

## Final Baseline

Method:

- shared local PPO actor-critic
- actor input: one agent local observation
- critic input: that same local observation
- no centralized team state
- no coordinated assignment action space

Matched training setting:

- `timesteps=16000`
- `eval_freq=4000`
- `n_eval_episodes=20`
- `num_envs=64`
- `rollout_steps=25`
- `minibatch_size=1600`
- `learning_rate=7e-4`
- `update_epochs=10`
- `ent_coef=0.01`
- `continuous_actions=false`
- `terminate_on_success=true`

60-episode deterministic re-evaluation of best checkpoints:

| seed | mean return | collisions | landmarks covered | success near end | sum min dists |
|---|---:|---:|---:|---:|---:|
| 8600 | -22.871 | 6.67 | 0.09 | 0.00 | 1.652 |
| 8602 | -26.688 | 2.77 | 0.06 | 0.00 | 2.061 |
| 8603 | -22.910 | 4.53 | 0.10 | 0.00 | 1.712 |
| mean | -24.156 | 4.657 | 0.083 | 0.00 | 1.808 |

## Final Multi-Agent

Method:

- actor input: 30D task-structured joint geometry
- actor policy: PPO over one categorical team assignment action (`3! = 6`)
- actor prior: exact distance-based assignment scores plus a learned residual
- low-level controller: short-horizon joint discrete MPC over all `5^3 = 125`
  team actions, using the actual MPE2 discrete dynamics and a collision-aware
  cost
- critic: centralized scalar value function over the 54D team state
- no expert warm-start

Matched training setting:

- `timesteps=16000`
- `eval_freq=4000`
- `n_eval_episodes=20`
- `num_envs=64`
- `rollout_steps=25`
- `minibatch_size=1600`
- `learning_rate=1e-4`
- `update_epochs=4`
- `ent_coef=0.0`
- `assignment_aux_coef=0.1`
- `continuous_actions=false`
- `terminate_on_success=true`

60-episode deterministic re-evaluation of best checkpoints:

| seed | mean return | collisions | landmarks covered | success near end | sum min dists |
|---|---:|---:|---:|---:|---:|
| 8501 | -6.298 | 0.52 | 1.12 | 0.9500 | 1.014 |
| 8502 | -6.166 | 0.50 | 1.13 | 0.9500 | 0.995 |
| 8503 | -6.217 | 0.55 | 1.13 | 0.9500 | 0.990 |
| mean | -6.227 | 0.523 | 1.127 | 0.9500 | 1.000 |

## Behavior Check

- The final multi-agent checkpoints run cleanly through the watch path with `render_mode=rgb_array`.
- In watch runs, agents split across different landmarks and usually terminate early when all three are covered.
- The baseline does not show this pattern reliably and usually runs the full 25 steps with low coverage.

## Interpretation

- The naive baseline still exhibits the intended failure mode for the paper: it lacks a team-level mechanism for resolving which agent should take which landmark.
- The final multi-agent method now beats the baseline by a large margin on all 3 seeds.
- The final multi-agent 3-seed mean return is better than `-10`, which matches the expected "simple environment" behavior much more closely.
- The key fix was not just longer training. The decisive changes were:
  - structured joint observation features
  - an assignment action space
  - a distance-based coordination prior
  - a discrete controller that reasons over the actual MPE2 transition model
