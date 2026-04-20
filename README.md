# PPO Final Project

This repository contains three PPO experiment tracks:

- a standard single-agent PPO workflow for Gymnasium MuJoCo tasks
- a Simple Spread baseline that adapts PPO to the multi-agent MPE2 setting
- a Simple Spread multi-agent implementation track for the stronger adaptation

The current Simple Spread baseline is intentionally straightforward so it can be
used as the "unmodified PPO" comparison point for the more principled
multi-agent PPO variant.

## Repo Layout

```text
rl_final_project/
|- mujoco/
|  |- train_mujoco.py
|  |- eval_mujoco.py
|  |- watch_mujoco.py
|  `- plot_results.py
|- simple_spread_baseline/
|  |- train_simple_spread_baseline.py
|  |- eval_simple_spread_baseline.py
|  |- watch_simple_spread_baseline.py
|  |- plot_simple_spread_baseline.py
|  `- simple_spread_baseline_common.py
|- simple_spread_multiagent/
|  `- runs/
|- results/
|- requirements.txt
`- README.md
```

## Setup

Create and activate a Python environment, then install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Notes:

- `gymnasium[mujoco]` may need MuJoCo system dependencies depending on your machine.
- The Simple Spread scripts use `mpe2`, `supersuit`, and `stable-baselines3`.

## MuJoCo PPO Track

The MuJoCo scripts live under `mujoco/` and provide a standard SB3 PPO workflow
for tasks like `HalfCheetah-v5` or `Hopper-v5`.

### Train

From the repo root:

```bash
python3 mujoco/train_mujoco.py
```

Example:

```bash
python3 mujoco/train_mujoco.py --env_id HalfCheetah-v5 --seed 1 --timesteps 1000000 --eval_freq 10000 --n_eval_episodes 10
```

Outputs are written to:

```text
mujoco/runs/<env_id>_seed<seed>/
```

Important files:

- `final_model.zip`: final checkpoint
- `best_model/best_model.zip`: best checkpoint seen during evaluation
- `eval_logs/evaluations.npz`: evaluation history
- `tb/`: TensorBoard logs

### Evaluate

```bash
python3 mujoco/eval_mujoco.py --model_path mujoco/runs/HalfCheetah-v5_seed0/final_model.zip --env_id HalfCheetah-v5 --episodes 20
```

### Plot

```bash
python3 mujoco/plot_results.py --env_id HalfCheetah-v5 --seeds 0 1 2
```

Default output:

```text
results/HalfCheetah-v5_3seed_curve.png
```

Useful plot arguments:

- `--runs_dir`: directory containing `<env_id>_seed<seed>/eval_logs/evaluations.npz`
- `--no_show`: save the figure without opening a window
- `--output_path`: custom path for the saved figure

### Watch

```bash
python3 mujoco/watch_mujoco.py --model_path mujoco/runs/HalfCheetah-v5_seed0/final_model.zip --env_id HalfCheetah-v5 --episodes 3
```

## Simple Spread Baseline Track

The Simple Spread baseline scripts live under `simple_spread_baseline/`.

This baseline is intentionally simple:

- one shared PPO policy across all three agents
- each agent acts from its own local observation only
- no communication channel
- no centralized critic

The current training script still uses SB3 PPO, but the multi-agent wrapping
logic is now explicit and documented. It also hard-codes the standard PPO MLP
policy/value network and loss settings instead of relying on library defaults.

### Baseline Training Design

The baseline uses:

- shared parameters for all agents
- discrete actions
- `local_ratio=0.5`
- `max_cycles=100` by default
- separate `64x64` tanh policy and value MLP heads
- standard PPO settings such as `clip_range=0.2`, `gamma=0.99`,
  `gae_lambda=0.95`, `ent_coef=0.0`, and `vf_coef=0.5`

SB3 counts one timestep per agent slot, not per joint world step. For Simple
Spread with 3 agents:

- one joint environment step contributes 3 SB3 timesteps per parallel env copy
- rollout size is `n_steps * num_vec_envs * 3`

The vector wrapper also preserves `TimeLimit.truncated` and
`terminal_observation` information so PPO can bootstrap correctly at max-cycle
cutoffs.

### Train the Baseline

From the repo root:

```bash
python3 simple_spread_baseline/train_simple_spread_baseline.py --seed 0
```

Or from inside the folder:

```bash
cd simple_spread_baseline
python3 train_simple_spread_baseline.py --seed 0
```

Example with explicit rollout settings:

```bash
python3 simple_spread_baseline/train_simple_spread_baseline.py --seed 0 --timesteps 1000000 --eval_freq 20000 --n_eval_episodes 12 --num_vec_envs 4 --n_steps 512 --batch_size 256
```

Outputs are written to:

```text
simple_spread_baseline/runs/simple_spread_baseline_seed<seed>/
```

Important files:

- `final_model.zip`: final checkpoint after training
- `best_model/best_model.zip`: best checkpoint by evaluation mean per-agent return
- `eval_logs/evaluations.npz`: evaluation history
- `tb/`: TensorBoard logs
- `run_config.json`: saved training configuration and timestep semantics

The evaluation file contains mean per-agent episode returns. This keeps the
scale comparable as an average agent outcome rather than multiplying every
return by the fixed number of agents.

The evaluation file contains:

- `timesteps`: raw SB3 agent-slot timesteps
- `parallel_env_steps`: timesteps divided by the number of agents
- `per_env_steps`: timesteps divided by total agent slots
- `results`: mean per-agent returns for each eval batch
- `ep_lengths`: episode lengths for each eval batch

### Evaluate a Saved Baseline Checkpoint

```bash
python3 simple_spread_baseline/eval_simple_spread_baseline.py --model_path simple_spread_baseline/runs/simple_spread_baseline_seed0/best_model/best_model --episodes 20
```

This prints:

- mean return
- standard deviation of return
- min and max return
- mean and standard deviation of episode length

### Plot Multiple Seeds

```bash
python3 simple_spread_baseline/plot_simple_spread_baseline.py --seeds 0 1 2
```

Default output:

```text
results/simple_spread_baseline_3seed_curve.png
```

The plot script can also change the x-axis and metric:

```bash
python3 simple_spread_baseline/plot_simple_spread_baseline.py --seeds 0 1 2 --x_axis parallel_env_steps --metric mean_episode_length
```

Options:

- `--x_axis`: `timesteps`, `parallel_env_steps`, or `per_env_steps`
- `--metric`: `mean_return` or `mean_episode_length`
- `--no_show`: save the figure without opening a window
- `--output_path`: custom path for the saved figure

The plotter can still read older legacy CSV logs, but only for return curves
with raw SB3 timesteps.

### Watch a Trained Baseline

For the usual human viewer:

```bash
python3 simple_spread_baseline/watch_simple_spread_baseline.py --model_path simple_spread_baseline/runs/simple_spread_baseline_seed0/best_model/best_model --episodes 3 --fps 30
```

Notes:

- `--fps` now controls the actual MPE2 render clock in human mode.
- The default watch FPS is `30`.
- Watching `best_model` is usually more informative than watching `final_model`.

If you want frame dumps instead of a live window:

```bash
python3 simple_spread_baseline/watch_simple_spread_baseline.py --model_path simple_spread_baseline/runs/simple_spread_baseline_seed0/best_model/best_model --episodes 1 --render_mode rgb_array --frame_dir /tmp/simple_spread_watch_seed0
```

Useful watch arguments:

- `--fps`: human render speed
- `--deterministic` or `--stochastic`: action selection mode
- `--terminate_on_success` or `--no_terminate_on_success`
- `--render_mode human` or `--render_mode rgb_array`
- `--frame_dir`: directory for saved PNG frames when using `rgb_array`

## Simple Spread Multi-Agent Track

The `simple_spread_multiagent/` directory is for the stronger Simple Spread
implementation that should be compared against the baseline above.

Use this track for the project's main multi-agent adaptation, while keeping the
baseline track unchanged as the shared-policy local-observation comparison.
Recommended conventions:

- keep implementation-specific checkpoints under `simple_spread_multiagent/runs/`
- keep aggregate figures under `results/`
- report the same evaluation metrics as the baseline when possible: mean
  per-agent return, episode length, timestep semantics, and seed-level curves
- compare against `simple_spread_baseline/` with the same seeds, `max_cycles`,
  and evaluation episode counts

This checkout currently contains the multi-agent output directory but no
multi-agent train/eval/watch/plot scripts, so this README does not list commands
for that track yet.

## Suggested Reproduction Flow

For the project deliverables, a reasonable workflow is:

1. Train 3 MuJoCo seeds and plot the standard PPO locomotion curve.
2. Train 3 Simple Spread baseline seeds with the shared-policy local-observation setup.
3. Evaluate and plot the Simple Spread baseline.
4. Watch the best checkpoint from each seed to qualitatively inspect coordination failures.
5. Run the stronger multi-agent Simple Spread implementation in `simple_spread_multiagent/`.
6. Compare the multi-agent results against the baseline using matched seeds and evaluation settings.

## Notes

- If a Simple Spread agent appears to stop or hover, that may be a true
  coordination limitation of the baseline rather than a broken watcher.
- After the truncation-handling fix in the current code, older Simple Spread
  checkpoints should be retrained if you want results that match the updated
  baseline implementation.
- Keep results under `results/` and per-seed checkpoints under the appropriate
  run directory so experiments stay easy to compare.
