# Simple Spread Baseline

This folder contains the revised naive PPO transfer for MPE2 Simple Spread.
The policy is shared across agents, and every PPO sample still uses one
agent's local observation for both the actor and the critic, but the actor now
chooses a landmark target instead of a raw movement action. A simple local
controller converts that target choice into a discrete `Discrete(5)` move.

That keeps the comparison intentionally simple:

- local observation only
- no centralized critic
- no agent IDs
- no communication module
- one PPO update rule applied to flattened per-agent samples
- no joint assignment logic, so agents can still chase the same landmark

In the final 3-seed fair comparison, this baseline reached a mean return of
`-20.560` over 60-episode checkpoint re-evaluations, with a final training-curve
mean of `-19.589`. That makes it a clearer coordination-limited reference point:
it learns to move toward landmarks, but it still collides and under-covers
compared with the structured multi-agent method.
