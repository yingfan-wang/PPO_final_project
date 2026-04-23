# Simple Spread Fair Comparison Without Warm-Start

Date: 2026-04-22
Environment: `mpe2.simple_spread_v3`
Conda env: `rl_final`

This file records the final fair apples-to-apples comparison where neither
method uses expert warm-start.

## Shared Environment Setting

- `N=3`
- `local_ratio=0.5`
- `max_cycles=25`
- `continuous_actions=false`
- `terminate_on_success=true`

## Baseline

Method:

- shared local-observation PPO actor-critic
- actor input: one agent's local observation
- critic input: that same local observation
- no centralized team state
- no joint assignment prior

Training command pattern:

```bash
python simple_spread_baseline/train_simple_spread.py \
  --seed <seed> \
  --timesteps 16000 \
  --eval_freq 4000 \
  --n_eval_episodes 20 \
  --num_envs 64 \
  --rollout_steps 25 \
  --minibatch_size 1600 \
  --learning_rate 7e-4 \
  --update_epochs 10 \
  --ent_coef 0.01 \
  --continuous_actions false \
  --terminate_on_success true
```

60-episode deterministic checkpoint re-evaluation:

| seed | mean return | collisions | landmarks covered | success near end | sum min dists |
|---|---:|---:|---:|---:|---:|
| 8600 | -22.871 | 6.67 | 0.09 | 0.00 | 1.652 |
| 8602 | -26.688 | 2.77 | 0.06 | 0.00 | 2.061 |
| 8603 | -22.910 | 4.53 | 0.10 | 0.00 | 1.712 |

Average over 3 seeds:

- mean return: `-24.156`
- mean collisions: `4.657`
- mean landmarks covered: `0.083`
- success near end: `0.00`
- mean sum min dists: `1.808`

## Multi-agent

Method:

- PPO with a centralized scalar critic over the 54D team state
- actor input: 30D task-structured joint geometry
- actor head: learned residual pairwise scores on top of an exact distance-based
  assignment prior
- policy distribution: one categorical action over the `3! = 6` landmark
  assignments
- low-level controller: short-horizon joint discrete MPC over all `5^3 = 125`
  team moves, with a collision-aware cost built from the actual MPE2 discrete
  dynamics
- no expert warm-start
- small assignment auxiliary loss to keep the learned actor close to the
  minimum-cost matching target

Training command pattern:

```bash
python simple_spread_multiagent/train_simple_spread.py \
  --seed <seed> \
  --timesteps 16000 \
  --eval_freq 4000 \
  --n_eval_episodes 20 \
  --num_envs 64 \
  --rollout_steps 25 \
  --minibatch_size 1600 \
  --learning_rate 1e-4 \
  --update_epochs 4 \
  --ent_coef 0.0 \
  --assignment_aux_coef 0.1 \
  --continuous_actions false \
  --terminate_on_success true
```

60-episode deterministic checkpoint re-evaluation:

| seed | mean return | collisions | landmarks covered | success near end | sum min dists |
|---|---:|---:|---:|---:|---:|
| 8501 | -6.298 | 0.52 | 1.12 | 0.9500 | 1.014 |
| 8502 | -6.166 | 0.50 | 1.13 | 0.9500 | 0.995 |
| 8503 | -6.217 | 0.55 | 1.13 | 0.9500 | 0.990 |

Average over 3 seeds:

- mean return: `-6.227`
- mean collisions: `0.523`
- mean landmarks covered: `1.127`
- success near end: `0.9500`
- mean sum min dists: `1.000`

## Conclusion

- The final no-warm-start multi-agent method now clearly beats the naive PPO baseline on all 3 seeds.
- The naive baseline still shows the expected coordination failure mode: very low landmark coverage and zero near-end full coverage.
- The multi-agent extension reaches the intended collaborative behavior much more reliably by combining:
  - task-structured joint observation handling
  - a coordinated assignment action space
  - a centralized critic
  - a distance-based assignment prior that breaks the same-landmark symmetry
  - a discrete controller that actually reasons over the MPE2 physics instead of
    treating discrete moves like clipped continuous thrust
- In this final fair setting, the multi-agent method is comfortably better than `-10` on the 3-seed mean.
