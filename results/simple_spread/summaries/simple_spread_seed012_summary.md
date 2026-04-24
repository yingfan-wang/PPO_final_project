# Simple Spread Seed 0/1/2 Summary

Fresh 3-seed runs with the revised baseline target-selection PPO and the existing structured multi-agent method.

## Baseline

| method | seed | mean_episode_return | std_episode_return | mean_episode_length | mean_collision_count | mean_landmarks_covered | success_near_end_rate | mean_sum_min_dists |
|---|---|---|---|---|---|---|---|---|
| baseline | 0 | -22.2318 | 6.3536 | 24.9000 | 7.1833 | 0.1507 | 0.0167 | 1.5893 |
| baseline | 1 | -18.7322 | 5.3439 | 24.6000 | 5.9667 | 0.1699 | 0.0500 | 1.3573 |
| baseline | 2 | -20.7172 | 5.7271 | 24.9167 | 6.4167 | 0.1563 | 0.0167 | 1.4884 |

Average over 3 seeds:

- seed: `1.0000`
- mean_episode_return: `-20.5604`
- std_episode_return: `5.8082`
- mean_episode_length: `24.8056`
- mean_collision_count: `6.5222`
- mean_landmarks_covered: `0.1590`
- success_near_end_rate: `0.0278`
- mean_sum_min_dists: `1.4783`

## Multiagent

| method | seed | mean_episode_return | std_episode_return | mean_episode_length | mean_collision_count | mean_landmarks_covered | success_near_end_rate | mean_sum_min_dists |
|---|---|---|---|---|---|---|---|---|
| multiagent | 0 | -6.8101 | 3.3765 | 13.2167 | 0.9500 | 1.1017 | 0.9500 | 0.9770 |
| multiagent | 1 | -6.8585 | 3.3986 | 13.2000 | 0.9833 | 1.0958 | 0.9500 | 0.9848 |
| multiagent | 2 | -6.7790 | 3.4215 | 13.1500 | 1.0000 | 1.0923 | 0.9500 | 0.9760 |

Average over 3 seeds:

- seed: `1.0000`
- mean_episode_return: `-6.8159`
- std_episode_return: `3.3988`
- mean_episode_length: `13.1889`
- mean_collision_count: `0.9778`
- mean_landmarks_covered: `1.0966`
- success_near_end_rate: `0.9500`
- mean_sum_min_dists: `0.9793`

## Headline

- Baseline final 3-seed eval mean: `-19.589`
- Multi-agent final 3-seed eval mean: `-5.743`
