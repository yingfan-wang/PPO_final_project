# Simple Spread Pure MAPPO

This folder contains a direct primitive-action MAPPO-style baseline for MPE2
Simple Spread.

Compared with the structured multi-agent method:

- the actor outputs one primitive `Discrete(5)` environment action per agent
- there is no hand-written landmark-assignment action abstraction
- there is no low-level controller or MPC-style action decoder
- the critic is still centralized and uses the full global state
- the actor is shared across agents and can optionally use one-hot agent IDs

This makes the implementation much closer to a textbook decentralized-actor,
centralized-critic PPO setup. It also creates a cleaner comparison for the
report: if this direct primitive-action MAPPO underperforms the structured
method, then that gap is strong evidence that the assignment abstraction and
controller are doing important work.
