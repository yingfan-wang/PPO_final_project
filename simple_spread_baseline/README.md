# Simple Spread Baseline

This folder contains the spec-aligned naive PPO transfer for MPE2 Simple
Spread. The policy is shared across agents, but every PPO sample uses one
agent's local observation for both the actor and the critic. The final report
uses the discrete `Discrete(5)` action version of `simple_spread_v3`.

That keeps the comparison intentionally simple:

- local observation only
- no centralized critic
- no agent IDs
- no communication module
- one PPO update rule applied to flattened per-agent samples

In the final 3-seed fair comparison, this baseline reached a mean return of
`-24.156` over 60-episode checkpoint re-evaluations, which makes it a clear
coordination-limited reference point against the structured multi-agent method.
