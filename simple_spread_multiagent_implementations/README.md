# README - Simple Spread Multi-Agent PPO Implementations

This project implements and compares five different multi-agent reinforcement learning approaches for the simple_spread environment from PettingZoo MPE, where three agents must cooperate to cover three landmarks.

---

## Implementations

| Implementation | Script | Key Features |
|----------------|--------|-------------|
| Parameter Sharing | train_simple_spread.py | Single shared policy for all 3 agents |
| Round-Robin Multi-Agent | train_simple_spread_multiagent.py | 3 separate policies trained sequentially with frozen peers |
| IPPO | train_shared_pol_param.py | Independent policies with frozen opponents |
| Joint Observation | train_joint_observation.py | Each agent sees full concatenated global observation (54D) |
| MAPPO (basic) | train_mappo.py | Decentralized actors + centralized critic |
| MAPPO (Advanced / TRUE) | train_mappo_true.py | Full MAPPO with GAE + centralized value function |

---

## Reward Shaping

All implementations use a shared reward shaping function (shaped_reward.py):

```python
def shape_reward(agent_obs, base_reward):
    return base_reward + 
           3.0 * exp(-10.0 * nearest_dist) - 
           0.5 * nearest_dist + 
           collision_penalty
```

Uncomment the comment in shaped_reward.py for default rewards

---

### Components

Exponential attraction to nearest landmark → encourages fast convergence  
Linear distance penalty → consistent progress signal  
Soft collision penalty → discourages overlap without forcing separation  

Note: the +0.1 * mean(distances) term was removed because it encouraged spreading instead of coordination.

---

## Training Configuration

- Total timesteps: 1,500,000 per algorithm  
- Seeds: 1, 2, 3  

### Shared Hyperparameters

``` python
HYPERPARAMS = {
    learning_rate: 3e-4,
    n_steps: 512,
    batch_size: 256,
    n_epochs: 10,
    gamma: 0.99,
    gae_lambda: 0.95,
    clip_range: 0.2,
    ent_coef: 0.01,
    vf_coef: 0.5,
    max_grad_norm: 0.5,
}
```

---

## Scripts Reference

Script → Purpose

```train_all.py``` → runs all training pipelines  
```plot_results_2.py``` → compares training curves  
```plot_testing_results.py``` → evaluates final models  
```generate_gifs.py``` → creates animations  
```shaped_reward.py``` → reward + hyperparameters  

Visualization:
```watch_ippo.py```
```watch_joint_observation.py```
```watch_round_robin.py```
```watch_simple_spread.py```
```watch_mappo.py```

---

## Directory Structure

```
PPO_final_project/
├── runs/
│   ├── simple_spread/
│   ├── round_robin/
│   ├── multiagent_spread/
│   ├── joint_obs_spread/
│   ├── mappo_spread/
│   └── mappo_TRUE_baseline/
├── gifs/
├── results.pdf
├── test_results.pdf
└── simple_spread_multiagent_implementations/
```

---

## Key Notes

- SB3 implementations use VecMonitor
- MAPPO uses PyTorch training loops
- Evaluation saved in eval_logs/evaluations.npz
- Reward shaping is centralized in shaped_reward.py

---

## Usage

Train all models:
```
python train_all.py
```

Plot training results:
```
python plot_results_2.py
```

Evaluate models:
```
python plot_testing_results.py
```

Generate GIFs:
```
python generate_gifs.py
```

Run visualization:
``` 
python watch_shared.py
```
