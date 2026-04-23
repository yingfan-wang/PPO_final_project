# PPO Final Project

This repository now contains three PPO experiment tracks:

- `mujoco/`: the original Stable-Baselines3 PPO workflow for Gymnasium MuJoCo tasks
- `simple_spread_baseline/`: a naive Simple Spread transfer where each agent independently picks a landmark target with PPO and a simple local controller converts that target into a move
- `simple_spread_multiagent/`: a structured multi-agent PPO transfer that uses a coordinated assignment prior to address coordination

The Simple Spread implementation follows the MPE2 `simple_spread_v3` parallel environment with:

- `N=3`
- `local_ratio=0.5`
- `max_cycles=25`
- `continuous_actions=False`

## Repo Layout

```text
rl_final_project/
|- mujoco/
|  |- train_mujoco.py
|  |- eval_mujoco.py
|  |- watch_mujoco.py
|  `- plot_results.py
|- simple_spread_baseline/
|  |- train_simple_spread.py
|  |- eval_simple_spread.py
|  |- watch_simple_spread.py
|  |- plot_results.py
|  |- config.py
|  |- env.py
|  |- networks.py
|  |- ppo.py
|  |- utils.py
|  `- README.md
|- simple_spread_multiagent/
|  |- train_simple_spread.py
|  |- eval_simple_spread.py
|  |- watch_simple_spread.py
|  |- plot_results.py
|  |- config.py
|  |- env.py
|  |- networks.py
|  |- expert.py
|  |- ma_ppo.py
|  `- README.md
|- results/
|- requirements.txt
`- README.md
```

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
  --terminate_on_success true \
  --device cpu
```

Optional ablations:

```bash
python simple_spread_multiagent/train_simple_spread.py --seed 0 --assignment_aux_coef 1.0
python simple_spread_multiagent/train_simple_spread.py --seed 0 --expert_warmstart_samples 32768 --expert_warmstart_epochs 120 --expert_warmstart_batch_size 1024 --expert_warmstart_lr 1e-3
```

Evaluate:

```bash
python simple_spread_multiagent/eval_simple_spread.py --model_path simple_spread_multiagent/runs/simple_spread_multiagent_seed0/final_model.pt --episodes 20 --device cpu
```

Watch:

```bash
python simple_spread_multiagent/watch_simple_spread.py --model_path simple_spread_multiagent/runs/simple_spread_multiagent_seed0/final_model.pt --episodes 3
```

Plot:

```bash
python simple_spread_multiagent/plot_results.py --seeds 0 1 2
```

## Outputs And Metrics

Both Simple Spread methods write one run directory per seed:

```text
simple_spread_baseline/runs/simple_spread_baseline_seed<seed>/
simple_spread_multiagent/runs/simple_spread_multiagent_seed<seed>/
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

The current 3-seed comparison is summarized in [results/simple_spread_seed012_summary.md](/Users/Andrew/Desktop/CS%204260/Final%20Projects/rl_final_project/results/simple_spread_seed012_summary.md).

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
python simple_spread_multiagent/train_simple_spread.py --seed 0 --timesteps 16 --eval_freq 8 --n_eval_episodes 2 --num_envs 2 --rollout_steps 4 --minibatch_size 8 --update_epochs 2 --device cpu
python simple_spread_baseline/eval_simple_spread.py --model_path simple_spread_baseline/runs/simple_spread_baseline_seed0/final_model.pt --episodes 2 --device cpu
python simple_spread_multiagent/eval_simple_spread.py --model_path simple_spread_multiagent/runs/simple_spread_multiagent_seed0/final_model.pt --episodes 2 --device cpu
python simple_spread_baseline/watch_simple_spread.py --model_path simple_spread_baseline/runs/simple_spread_baseline_seed0/final_model.pt --episodes 1 --render_mode rgb_array --device cpu
python simple_spread_multiagent/watch_simple_spread.py --model_path simple_spread_multiagent/runs/simple_spread_multiagent_seed0/final_model.pt --episodes 1 --render_mode rgb_array --device cpu
```
