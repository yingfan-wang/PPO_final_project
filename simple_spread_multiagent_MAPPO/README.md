# Simple Spread Multi-Agent PPO

This folder contains the final coordinated Simple Spread method used for the
project report.

## Folder Guide

The folder is easiest to navigate if you separate the core training path from
the reporting helpers:

- `config.py`: structured-method hyperparameters
- `env.py`: joint feature construction and controller-facing environment helpers
- `networks.py`: assignment-policy network pieces
- `ma_ppo.py`: centralized-critic PPO agent for the joint assignment policy
- `expert.py`: exact-assignment targets and optional warm-start dataset generation
- `train_simple_spread.py`: main training loop, evaluation schedule, and checkpoint writing
- `eval_simple_spread.py`: checkpoint evaluation
- `watch_simple_spread.py`: live rendering or saved rollout animations
- `plot_results.py`: multi-seed aggregation and legacy NPZ export
- `generate_side_by_side_gifs.py`: convenience reporting script for visual comparisons
- `report_results_pdf.py`: PDF report generation from saved eval logs

Generated artifacts to treat as outputs rather than source:

- `runs/`: per-seed checkpoints and evaluation logs
- `animations/`: optional rendered episodes
- `.mplconfig/`, `__pycache__/`, `.DS_Store`: local cache files

## Final Design

The final design is a structured multi-agent PPO transfer:

- actor input: a 30D joint geometry feature vector built from all
  agent-to-landmark relative vectors and all other-agent relative vectors
- actor prior: exact distance-based pair scores, so the initial policy already
  prefers the minimum-cost landmark assignment
- actor head: a learned residual over the pair scores, inducing a categorical
  PPO policy over the `3! = 6` possible landmark matchings
- low-level controller: a short-horizon joint discrete MPC controller that
  scores all `5^3 = 125` team moves under the actual MPE2 dynamics and adds a
  collision penalty, which makes the landmark-covering behavior line up with the
  discrete environment physics
- critic: centralized scalar value function over the full 54D team state
- reward target: mean team reward

Compared with the naive baseline, this explicitly tackles:

- coordination: by choosing one team assignment instead of three independent
  local thrust actions
- observation aggregation: by using joint geometry features instead of isolated
  local observations
- credit assignment: by using one centralized team critic and one team reward

## Fair Main Comparison

The current baseline-vs-multi-agent result is with:

- `expert_warmstart_samples=0`
- `expert_warmstart_epochs=0`
- `assignment_aux_coef=0.1`

The latest 3-seed comparison is recorded in:

- [results/simple_spread/summaries/simple_spread_seed012_summary.md](/Users/Andrew/Desktop/CS%204260/Final%20Projects/rl_final_project/results/simple_spread/summaries/simple_spread_seed012_summary.md)
- [results/simple_spread/reports/simple_spread_multiagent_seed_report.pdf](/Users/Andrew/Desktop/CS%204260/Final%20Projects/rl_final_project/results/simple_spread/reports/simple_spread_multiagent_seed_report.pdf)

Headline result:

- baseline final 3-seed eval mean: `-19.589`
- multi-agent final 3-seed eval mean: `-5.743`
- baseline 60-episode re-eval mean: `-20.560`
- multi-agent 60-episode re-eval mean: `-6.816`

## Recommended Training Command

```bash
python simple_spread_multiagent/train_simple_spread.py \
  --seed 8501 \
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

This is the no-warm-start discrete configuration that produced the stable
`>-10` and, in practice, roughly `-6.2` multi-agent results.
