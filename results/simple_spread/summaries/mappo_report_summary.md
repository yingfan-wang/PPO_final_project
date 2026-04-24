# MAPPO Report Summary

## Final MAPPO Setup

Source run config:
- `simple_spread_multiagent/runs/simple_spread_multiagent_seed0/run_config.json`

Final fair comparison settings:
- environment: `simple_spread`
- seeds: `0, 1, 2`
- requested environment steps: `16000`
- effective environment steps: `16000`
- effective agent steps: `48000`
- number of agents: `3`
- number of parallel environments: `64`
- rollout steps: `25`
- learning rate: `1e-4`
- PPO clip coefficient: `0.2`
- GAE lambda: `0.95`
- discount factor gamma: `0.99`
- update epochs: `4`
- minibatch size: `1600`
- value loss coefficient: `0.5`
- entropy coefficient: `0.0`
- max grad norm: `0.5`
- actor hidden sizes: `(128, 128)`
- critic hidden sizes: `(128, 128)`
- assignment auxiliary coefficient: `0.1`
- expert warm-start samples: `0`
- terminate on success: `true`
- curriculum: `false`
- action space: `discrete_5`

## What Makes This MAPPO-Style Variant Different

- The actor observes a 30D joint geometry feature vector rather than one agent's local observation alone.
- The policy outputs one categorical decision over the `3! = 6` possible team landmark assignments.
- A distance-based assignment prior is added to learned residual scores before the categorical distribution is formed.
- The selected assignment is converted into low-level actions by a short-horizon joint discrete controller that evaluates all `5^3 = 125` team actions.
- The critic is centralized and uses the full 54D global state.
- PPO is optimized on the mean team reward, so the update target is cooperative rather than per-agent.
- A small assignment auxiliary loss regularizes the actor toward the minimum-cost matching target.

## Final 60-Episode Re-Evaluation Averages Across Seeds

MAPPO:
- mean episode return: `-6.8159`
- return standard deviation: `3.3988`
- mean episode length: `13.1889`
- mean collision count: `0.9778`
- mean landmarks covered: `1.0966`
- success near end rate: `0.9500`
- mean sum of minimum landmark distances: `0.9793`

Baseline:
- mean episode return: `-20.5604`
- return standard deviation: `5.8082`
- mean episode length: `24.8056`
- mean collision count: `6.5222`
- mean landmarks covered: `0.1590`
- success near end rate: `0.0278`
- mean sum of minimum landmark distances: `1.4783`

Relative change from baseline to MAPPO:
- return improvement: `+13.7445`
- return magnitude reduction: `66.85%`
- collision reduction: `85.01%`
- episode length reduction: `46.83%`
- landmark coverage multiplier: `6.90x`
- success rate multiplier: `34.20x`
- mean sum of minimum landmark distances reduction: `33.76%`

## Final Evaluation Means From Training Curves

These are the final evaluation points at `16000` environment steps:
- seed 0: `-5.7026`
- seed 1: `-5.5821`
- seed 2: `-5.9442`
- three-seed final eval mean: `-5.7430`

Baseline final evaluation points at `16000` environment steps:
- seed 0: `-19.0332`
- seed 1: `-19.7296`
- seed 2: `-20.0031`
- three-seed final eval mean: `-19.5887`

## Report-Ready Paragraph

Our final Simple Spread method is a structured MAPPO-style PPO variant designed around the coordination structure of the task. Instead of learning three independent local policies, the actor receives a joint geometry representation of the full team configuration and predicts a categorical distribution over the six possible one-to-one assignments between agents and landmarks. A centralized critic estimates a single team value from the full 54D global state, and PPO is optimized on the mean team reward, which improves cooperative credit assignment. To further stabilize coordination, the actor includes a distance-based assignment prior and a small assignment auxiliary loss, while a short-horizon joint discrete controller maps each chosen assignment into collision-aware low-level actions. In the three-seed fair comparison, this approach improved the 60-episode reevaluation mean return from `-20.5604` for the baseline to `-6.8159`, reduced collisions by `85.01%`, shortened episodes by `46.83%`, and increased near-end full-coverage success from `2.78%` to `95.00%`.

## Useful Figures

- `results/simple_spread/curves/simple_spread_baseline_vs_multiagent_seed012_curve.png`
- `results/simple_spread/curves/simple_spread_multiagent_3seed_curve.png`
- `results/simple_spread/animations/side_by_side/simple_spread_side_by_side_seed0_long.gif`
