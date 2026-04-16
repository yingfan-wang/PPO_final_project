# PPO Final Project

This repository currently contains a simple reinforcement learning workflow for
training, evaluating, and plotting a PPO agent on MuJoCo tasks with
`stable-baselines3`.

Right now the default setup is aimed at `HalfCheetah-v5`, but the scripts accept
an `--env_id` argument so we can extend the repo to other Gymnasium MuJoCo
environments later.

## Current Repo Layout

```text
rl_final_project/
|- train_mujoco.py      # train PPO and save checkpoints/logs
|- eval_mujoco.py       # evaluate one saved model
|- watch_mujoco.py      # render one saved model in a human viewer
|- plot_results.py      # combine multiple seeds into one figure
|- requirments.txt      # Python dependencies (note the current filename)
|- runs/                # per-seed training outputs
`- results/             # generated plots
```

## What Each Script Does

### `train_mujoco.py`

Trains a PPO agent and writes all outputs to:

```text
runs/<env_id>_seed<seed>/
```

For example, the default run with seed `0` writes to:

```text
runs/HalfCheetah-v5_seed0/
```

Important outputs inside each run directory:

- `final_model.zip`: the final checkpoint after training finishes
- `best_model/best_model.zip`: the best checkpoint found during periodic evaluation
- `eval_logs/evaluations.npz`: evaluation returns collected during training
- `tb/`: TensorBoard event files

### `eval_mujoco.py`

Loads a saved `.zip` PPO checkpoint and prints a summary of returns:

- mean return
- standard deviation
- min return
- max return

This script does not create a new results file by default. It prints the numbers
to the terminal.

### `plot_results.py`

Reads `evaluations.npz` files from multiple seeds, then creates:

- one line per seed
- the mean curve across seeds
- a shaded band for `mean +- std`

The figure is saved under:

```text
results/
```

With the current code, the default output file is:

```text
results/HalfCheetah-v5_3seed_curve.png
```

### `watch_mujoco.py`

Loads a saved `.zip` PPO checkpoint and opens a rendered MuJoCo window so you
can visually inspect how the trained policy behaves.

This script:

- runs a small number of episodes with `render_mode="human"`
- prints one return per watched episode
- does not save a new file by default

## Setup

Create and activate a Python environment, then install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirments.txt
```

Notes:

- The dependency file is currently named `requirments.txt`, not `requirements.txt`.
- `gymnasium[mujoco]` may require MuJoCo-related system dependencies depending on
  your machine setup.

## How To Use The Existing Scripts

### 1. Train a model

Default training command:

```bash
python3 train_mujoco.py
```

Example with custom settings:

```bash
python3 train_mujoco.py --env_id HalfCheetah-v5 --seed 1 --timesteps 1000000 --eval_freq 10000 --n_eval_episodes 10
```

Arguments:

- `--env_id`: Gymnasium environment id
- `--seed`: random seed used for training
- `--timesteps`: total PPO training timesteps
- `--eval_freq`: how often to run evaluation during training
- `--n_eval_episodes`: number of episodes per evaluation checkpoint

### 2. Evaluate a saved model

Example:

```bash
python3 eval_mujoco.py --model_path runs/HalfCheetah-v5_seed0/final_model.zip --env_id HalfCheetah-v5 --episodes 20
```

Arguments:

- `--model_path`: path to a saved SB3 PPO `.zip` checkpoint
- `--env_id`: Gymnasium environment id used for evaluation
- `--episodes`: number of evaluation episodes
- `--seed`: base seed for evaluation rollouts

### 3. Plot multiple seeds

Example:

```bash
python3 plot_results.py --env_id HalfCheetah-v5 --seeds 0 1 2
```

This reads:

- `runs/HalfCheetah-v5_seed0/eval_logs/evaluations.npz`
- `runs/HalfCheetah-v5_seed1/eval_logs/evaluations.npz`
- `runs/HalfCheetah-v5_seed2/eval_logs/evaluations.npz`

And saves the combined figure to `results/HalfCheetah-v5_3seed_curve.png`.

### 4. Watch a saved model

Example:

```bash
python3 watch_mujoco.py --model_path runs/HalfCheetah-v5_seed0/final_model.zip --env_id HalfCheetah-v5 --episodes 3
```

Arguments:

- `--model_path`: path to a saved SB3 PPO `.zip` checkpoint
- `--env_id`: Gymnasium environment id to render
- `--episodes`: number of episodes to watch

Notes:

- This uses `render_mode="human"`, so it needs a local graphical environment.
- It is mainly for qualitative inspection, not for formal evaluation.


## Notes For Future Additions

If we add more experiments later, it will stay organized if we keep the same
pattern:

- put training logs and checkpoints under `runs/<env_id>_seed<seed>/`
- put generated figures under `results/`
- document any new scripts and outputs in this README
- keep command-line arguments explicit so experiments are easy to reproduce
