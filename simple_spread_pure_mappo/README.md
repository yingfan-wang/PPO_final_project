# Simple Spread Pure MAPPO

This folder contains a direct primitive-action MAPPO-style baseline for MPE2
Simple Spread.

## Folder Guide

The source layout mirrors the other final Simple Spread packages:

- `config.py`: MAPPO hyperparameters and environment constants
- `env.py`: actor/critic input builders on top of the shared Simple Spread wrappers
- `networks.py`: primitive-action actor and centralized critic definitions
- `ppo.py`: MAPPO-style update logic
- `train_simple_spread.py`: training loop, checkpoint writing, and periodic evaluation
- `eval_simple_spread.py`: checkpoint evaluation
- `watch_simple_spread.py`: live rendering or saved rollout animations
- `plot_results.py`: multi-seed curve plotting

Generated artifacts to treat separately from the code:

- `runs/`: per-seed checkpoints and evaluation logs
- `animations/`: optional rendered episodes if you choose to save them
- `.mplconfig/`, `__pycache__/`, `.DS_Store`: local cache files

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
