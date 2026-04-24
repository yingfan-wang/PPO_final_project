# Pure MAPPO vs Structured Multi-Agent PPO (Seed 0)

This note records the first direct comparison between:

- `simple_spread_pure_mappo/`: decentralized primitive-action actor with a
  centralized critic and no low-level controller
- `simple_spread_multiagent_MAPPO/`: structured assignment-based policy with a
  centralized critic and a fixed low-level controller

## Shared Training Budget

Both runs used:

- seed: `0`
- environment steps: `16000`
- agent steps: `48000`
- number of parallel environments: `64`
- rollout steps: `25`
- learning rate: `1e-4`
- PPO update epochs: `4`
- minibatch size: `1600`
- entropy coefficient: `0.0`
- terminate on success: `true`
- primitive action space: `Discrete(5)`

The pure MAPPO run also used:

- `use_agent_id=true`

## Final 20-Episode Evaluation Point At 16000 Environment Steps

Pure MAPPO:
- mean return: `-22.303`
- success near end: `0.00%`

Structured multi-agent PPO:
- mean return: `-5.703`
- success near end: `95.00%`

## 60-Episode Re-Evaluation Of Final Checkpoints

Pure MAPPO final checkpoint:
- mean episodic return: `-24.336`
- std episodic return: `4.989`
- mean episode length: `25.00`
- mean collisions: `8.13`
- mean landmarks covered: `0.07`
- success near end: `0.00%`
- mean sum min dists: `1.730`

Structured multi-agent final checkpoint:
- mean episodic return: `-6.482`
- std episodic return: `3.046`
- mean episode length: `12.47`
- mean collisions: `0.52`
- mean landmarks covered: `1.10`
- success near end: `100.00%`
- mean sum min dists: `0.988`

## Interpretation

Under the same seed and step budget, the direct primitive-action MAPPO variant
is much weaker than the structured method. This suggests that the main gain in
the structured approach is not just "using PPO with a centralized critic," but
changing the action abstraction itself. The pure MAPPO agent still has to learn
basic motion, collision avoidance, and team-level coordination jointly from the
primitive action space. The structured method removes much of that burden by
letting PPO decide only the team assignment, while a fixed low-level controller
handles target-tracking and collision-aware motion.

## Report-Safe Summary

Compared with the current structured implementation, a direct primitive-action
MAPPO baseline trained under the same step budget performed much worse on seed
0. Its final 60-episode reevaluation mean return was `-24.336`, versus `-6.482`
for the structured assignment-based method, and it never achieved near-end
three-landmark coverage. This supports the claim that the structured action
representation and controller contribute substantially to performance beyond the
centralized-critic PPO update alone.
