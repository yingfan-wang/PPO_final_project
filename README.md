# PPO Final Project

This repository contains four PPO experiment tracks:

- `mujoco/`: the original Stable-Baselines3 PPO workflow for Gymnasium MuJoCo tasks
- `simple_spread_baseline/`: a naive Simple Spread transfer where each agent independently picks a landmark target and a local controller turns that target into a move
- `simple_spread_pure_mappo/`: a direct primitive-action decentralized-actor centralized-critic PPO baseline without the structured controller
- `simple_spread_multiagent_MAPPO/`: the final structured multi-agent PPO method with coordinated landmark assignment

For Simple Spread, the final code lives in those three folders above. The separate
`simple_spread_multiagent_implementations/` directory is a legacy sandbox with older
experiments and is not part of the final training pipeline.

All Simple Spread variants use the MPE2 `simple_spread_v3` parallel environment with:

- `N=3`
- `local_ratio=0.5`
- `max_cycles=25`
- `continuous_actions=False`

## Repo Layout

```text
rl_final_project/
|- README.md
|- requirements.txt
|- mujoco/                               # SB3 PPO reproduction
|- simple_spread_baseline/              # naive local-observation PPO transfer
|- simple_spread_pure_mappo/            # primitive-action MAPPO baseline
|- simple_spread_multiagent_MAPPO/      # structured coordinated PPO
|- simple_spread_multiagent_implementations/  # older Simple Spread experiments
|- runs/                                # shared/legacy exported run artifacts
`- results/                             # plots, summaries, reports, animations
```

## How To Read The Simple Spread Folders

The three final Simple Spread packages intentionally follow the same shape:

- entry scripts: `train_simple_spread.py`, `eval_simple_spread.py`, `watch_simple_spread.py`, and `plot_results.py`
- environment/config layer: `config.py` and `env.py`
- model layer: `networks.py` plus `ppo.py` or `ma_ppo.py`
- package-specific helpers: `utils.py` in the baseline and `expert.py` in the structured multi-agent method
- local notes: each folder has its own `README.md`

If you are reviewing source code, the fastest path is:

1. `config.py` for the track-level defaults and dimensions.
2. `env.py` for observations, actions, and any controller or feature construction.
3. `ppo.py` or `ma_ppo.py` for the actual policy update logic.
4. `train_simple_spread.py` for the end-to-end experiment flow.

Some Simple Spread folders also contain generated artifacts such as `runs/`,
`animations/`, `.mplconfig/`, `__pycache__/`, and `.DS_Store`. Those are outputs or
machine-local files, not part of the core implementation.

## Setup

Use the course environment if you already have it:

```bash
conda activate rl_final
```

Or create a fresh environment and install the project dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Key dependencies:

- `stable-baselines3` for the MuJoCo reproduction
- `torch` for the custom Simple Spread PPO implementations
- `mpe2` for `simple_spread_v3`
- `matplotlib`, `pandas`, and `numpy` for logging and plots

## MuJoCo Track

The MuJoCo path is unchanged and still uses Stable-Baselines3 PPO.

Train:

```bash
python mujoco/train_mujoco.py --env_id HalfCheetah-v5 --seed 0 --timesteps 1000000
```

Evaluate:

```bash
python mujoco/eval_mujoco.py --model_path mujoco/runs/HalfCheetah-v5_seed0/final_model.zip --env_id HalfCheetah-v5 --episodes 20
```

Watch:

```bash
python mujoco/watch_mujoco.py --model_path mujoco/runs/HalfCheetah-v5_seed0/final_model.zip --env_id HalfCheetah-v5 --episodes 3
```

Plot:

```bash
python mujoco/plot_results.py --env_id HalfCheetah-v5 --seeds 0 1 2
```

## Simple Spread Baseline

The baseline is intentionally naive. It uses one shared local-observation actor-critic network, but each agent only chooses which landmark to chase and a simple local controller turns that choice into a `Discrete(5)` move:

- actor input: one agent's local observation
- critic input: that same local observation
- policy output: one of the three landmarks
- low-level control: a simple local axis-aligned move toward the chosen landmark
- no centralized state in the value function
- no agent IDs by default
- no communication or reward shaping

This is the coordination-limited reference point for the report. It learns landmark-seeking behavior, but it still makes independent target choices and does not explicitly solve team-level credit assignment or assignment conflicts.

The training script also exposes the MPE2 `terminate_on_success` and `curriculum` options so we can run matched ablations without changing the code path.

Train:

```bash
python simple_spread_baseline/train_simple_spread.py \
  --seed 0 \
  --timesteps 16000 \
  --eval_freq 4000 \
  --n_eval_episodes 20 \
  --learning_rate 5e-4 \
  --update_epochs 15 \
  --ent_coef 0.0 \
  --continuous_actions false \
  --terminate_on_success true \
  --curriculum true \
  --device cpu
```

Evaluate:

```bash
python simple_spread_baseline/eval_simple_spread.py --model_path simple_spread_baseline/runs/simple_spread_baseline_seed0/final_model.pt --episodes 20 --device cpu
```

Watch:

```bash
python simple_spread_baseline/watch_simple_spread.py --model_path simple_spread_baseline/runs/simple_spread_baseline_seed0/final_model.pt --episodes 3
```

Plot:

```bash
python simple_spread_baseline/plot_results.py --seeds 0 1 2
```

## Simple Spread Pure MAPPO

This track is the closest thing in the repo to a textbook MAPPO-style
comparison point: the actor outputs one primitive `Discrete(5)` action per
agent directly from local observations, while the critic uses the full 54D
global state.

- actor input: one local observation per agent, optionally with a one-hot
  agent ID
- critic input: the full 54D global state, repeated per agent and optionally
  augmented with a one-hot agent ID
- policy output: one primitive environment action per agent
- no assignment prior
- no low-level controller
- no hand-written action abstraction

This makes it a useful ablation against the structured method: if pure MAPPO
underperforms, then the gap can be attributed to the assignment abstraction and
controller rather than to PPO alone.

Train:

```bash
python simple_spread_pure_mappo/train_simple_spread.py \
  --seed 0 \
  --timesteps 16000 \
  --eval_freq 4000 \
  --n_eval_episodes 20 \
  --num_envs 64 \
  --rollout_steps 25 \
  --minibatch_size 1600 \
  --learning_rate 1e-4 \
  --update_epochs 4 \
  --ent_coef 0.0 \
  --continuous_actions false \
  --terminate_on_success true \
  --use_agent_id true \
  --device cpu
```

Evaluate:

```bash
python simple_spread_pure_mappo/eval_simple_spread.py --model_path simple_spread_pure_mappo/runs/simple_spread_pure_mappo_seed0/final_model.pt --episodes 20 --device cpu
```

Watch:

```bash
python simple_spread_pure_mappo/watch_simple_spread.py --model_path simple_spread_pure_mappo/runs/simple_spread_pure_mappo_seed0/final_model.pt --episodes 3 --device cpu
```

Plot:

```bash
python simple_spread_pure_mappo/plot_results.py --seeds 0 1 2
```

The current seed-0 structured-vs-pure comparison is summarized in
[results/simple_spread/summaries/pure_mappo_vs_structured_seed0_summary.md](/Users/Andrew/Desktop/CS%204260/Final%20Projects/rl_final_project/results/simple_spread/summaries/pure_mappo_vs_structured_seed0_summary.md).

## Simple Spread Multi-Agent Track

The final multi-agent track uses a task-structured PPO design that was tuned specifically to make the intended "three agents split across three landmarks" behavior stable:

- actor input: a 30D joint geometry feature vector containing all agent-to-landmark relative vectors and all other-agent relative vectors
- actor prior: exact distance-based assignment scores, so the initial policy already prefers the minimum-cost landmark matching
- actor head: a learned residual over agent-landmark pair scores, inducing a categorical distribution over the `3! = 6` landmark assignments
- low-level control: a short-horizon joint discrete controller that scores all `5^3 = 125` team moves under the actual MPE2 dynamics and adds a collision cost
- critic input: the fixed-order 54D global state from `env.state()`
- critic output: one scalar team value
- reward target: the mean team reward, so all agents optimize one cooperative PPO objective

This is still a PPO adaptation, but the action abstraction and policy prior are much closer to the real coordination structure of Simple Spread than the naive local baseline.

The training script still supports optional expert warm-start, but the final fair no-warm-start comparison below uses `expert_warmstart_samples=0` and a small assignment auxiliary loss to keep the actor close to the minimum-cost matching target.

Train:

```bash
python simple_spread_multiagent_MAPPO/train_simple_spread.py \
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
  --terminate_on_success true \
  --device cpu
```

Optional ablations:

```bash
python simple_spread_multiagent_MAPPO/train_simple_spread.py --seed 0 --assignment_aux_coef 1.0
python simple_spread_multiagent_MAPPO/train_simple_spread.py --seed 0 --expert_warmstart_samples 32768 --expert_warmstart_epochs 120 --expert_warmstart_batch_size 1024 --expert_warmstart_lr 1e-3
```

Evaluate:

```bash
python simple_spread_multiagent_MAPPO/eval_simple_spread.py --model_path simple_spread_multiagent_MAPPO/runs/simple_spread_multiagent_seed0/final_model.pt --episodes 20 --device cpu
```

Watch:

```bash
python simple_spread_multiagent_MAPPO/watch_simple_spread.py --model_path simple_spread_multiagent_MAPPO/runs/simple_spread_multiagent_seed0/final_model.pt --episodes 3
```

Plot:

```bash
python simple_spread_multiagent_MAPPO/plot_results.py --seeds 0 1 2
```

## Outputs And Metrics

The Simple Spread code is organized so that source files stay in the package folders,
while experiment outputs are easy to recognize separately:

- per-track checkpoints and logs live under each package's `runs/` directory
- shared comparison plots, summaries, reports, and animations live under `results/simple_spread/`
- a few top-level `runs/` files exist for teammate-compatible exports and older shared artifacts

Per-seed training directories live at:

```text
simple_spread_baseline/runs/simple_spread_baseline_seed<seed>/
simple_spread_pure_mappo/runs/simple_spread_pure_mappo_seed<seed>/
simple_spread_multiagent_MAPPO/runs/simple_spread_multiagent_seed<seed>/
```

Each run contains:

- `run_config.json`
- `train_metrics.csv`
- `final_model.pt`
- `best_model/best_model.pt`
- `eval_logs/eval_metrics.csv`
- `eval_logs/evaluations.npz`
- `eval_curve.png`

The evaluation logs track:

- mean episodic return
- episode length
- collision count
- average number of distinct landmarks covered
- fraction of episodes with full three-landmark coverage near the end
- mean sum of min landmark distances

`--timesteps` for the Simple Spread scripts is counted in joint environment steps. Agent-step budget is tracked separately as:

```text
agent_steps = env_steps * 3
```

This keeps baseline and multi-agent comparisons aligned on equal environment-step budgets.

## Final Fair Result

The current 3-seed comparison is summarized in [results/simple_spread/summaries/simple_spread_seed012_summary.md](/Users/Andrew/Desktop/CS%204260/Final%20Projects/rl_final_project/results/simple_spread/summaries/simple_spread_seed012_summary.md).

Using matched `terminate_on_success=true` discrete-action training runs:

- baseline final 3-seed eval mean: `-19.589`
- multi-agent final 3-seed eval mean: `-5.743`
- baseline 60-episode re-eval mean: `-20.560`
- multi-agent 60-episode re-eval mean: `-6.816`

So the final multi-agent implementation is still substantially stronger, but the revised baseline now visibly learns to chase landmarks instead of failing at the raw control layer.

## Smoke Test Commands

The new implementations were smoke-tested in `rl_final` with short runs:

```bash
python simple_spread_baseline/train_simple_spread.py --seed 0 --timesteps 16 --eval_freq 8 --n_eval_episodes 2 --num_envs 2 --rollout_steps 4 --minibatch_size 8 --update_epochs 2 --device cpu
python simple_spread_multiagent_MAPPO/train_simple_spread.py --seed 0 --timesteps 16 --eval_freq 8 --n_eval_episodes 2 --num_envs 2 --rollout_steps 4 --minibatch_size 8 --update_epochs 2 --device cpu
python simple_spread_baseline/eval_simple_spread.py --model_path simple_spread_baseline/runs/simple_spread_baseline_seed0/final_model.pt --episodes 2 --device cpu
python simple_spread_multiagent_MAPPO/eval_simple_spread.py --model_path simple_spread_multiagent_MAPPO/runs/simple_spread_multiagent_seed0/final_model.pt --episodes 2 --device cpu
python simple_spread_baseline/watch_simple_spread.py --model_path simple_spread_baseline/runs/simple_spread_baseline_seed0/final_model.pt --episodes 1 --render_mode rgb_array --device cpu
python simple_spread_multiagent_MAPPO/watch_simple_spread.py --model_path simple_spread_multiagent_MAPPO/runs/simple_spread_multiagent_seed0/final_model.pt --episodes 1 --render_mode rgb_array --device cpu
```
