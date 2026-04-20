# Final Report — small-to-large ladder v3

## Configuration

- session: `/Data/janis.aiad/abstractrl/src/AbstractBeam/crossbeam/experiment/v3/runs/gcp_trm_scaleup_v3/small_to_large_ladder_1776709278`
- sizes: `[100, 200]`
- per-size: `2`
- budgets: `['12:24']`
- timeout_sec: `40.0`
- profile_every: `4`

## Budget 12:24

| Method | solved_rate | timeout_rate | mean_conf | median_conf | mean_time_s | mean_primitive_calls | win_rate_vs_greedy | delta_mean_vs_greedy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed_tabu_recolor | 0.0000 | 0.0000 | 40.0000 | 38.5000 | 0.8806 | 26.0000 | 1.0000 | -19.2500 |
| greedy_best_of_orders | 0.0000 | 0.0000 | 59.2500 | 58.5000 | 0.0865 | - | 0.5000 | 0.0000 |
| mcts_no_model | 0.0000 | 0.0000 | 53.7500 | 53.0000 | 3.0065 | 4.2500 | 1.0000 | -5.5000 |
| mcts_trained_curriculum | 0.0000 | 0.5000 | 16.0000 | 16.0000 | 27.1956 | 22.0000 | 0.5000 | -12.5000 |
| mcts_trained_small | 0.0000 | 0.5000 | 16.0000 | 16.0000 | 27.0352 | 22.0000 | 0.5000 | -12.5000 |
| mcts_untrained | 0.0000 | 0.0000 | 58.5000 | 58.5000 | 4.4977 | 8.0000 | 0.5000 | -0.7500 |

### Compute-normalized

| Method | gain_per_sec | gain_per_primitive | anytime_improvement_per_sec |
|---|---:|---:|---:|
| fixed_tabu_recolor | 26.0158 | 0.7248 | 23.4503 |
| greedy_best_of_orders | 0.0000 | - | 0.0000 |
| mcts_no_model | 1.9080 | 1.3042 | 1.3108 |
| mcts_trained_curriculum | 0.8781 | 0.5833 | 0.1224 |
| mcts_trained_small | 0.8973 | 0.5833 | 0.1249 |
| mcts_untrained | 0.2850 | 0.7500 | 1.9358 |

### Paired Comparisons

| Pair | n_pairs | win_rate(a>b) | delta_mean(a-b) | delta_ci95(a-b) |
|---|---:|---:|---:|---:|
| mcts_trained_curriculum_vs_mcts_trained_small | 4 | 0.5000 | 0.0000 | [0.0000, 0.0000] |
| mcts_trained_small_vs_greedy_best_of_orders | 4 | 0.5000 | -12.5000 | [-13.0000, -12.0000] |
| mcts_trained_small_vs_mcts_no_model | 4 | 0.5000 | -7.5000 | [-9.0000, -6.0000] |
| mcts_trained_small_vs_mcts_untrained | 4 | 0.5000 | -9.5000 | [-13.0000, -6.0000] |

