# Final Report — small-to-large ladder v3

## Configuration

- session: `/Data/janis.aiad/abstractrl/src/AbstractBeam/crossbeam/experiment/v3/runs/gcp_trm_scaleup_v3/small_to_large_ladder_1776708524`
- sizes: `[100]`
- per-size: `1`
- budgets: `['12:24']`
- timeout_sec: `30.0`
- profile_every: `4`

## Budget 12:24

| Method | solved_rate | timeout_rate | mean_conf | median_conf | mean_time_s | mean_primitive_calls | win_rate_vs_greedy | delta_mean_vs_greedy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed_tabu_recolor | 0.0000 | 0.0000 | 13.0000 | 13.0000 | 0.4259 | 25.0000 | 1.0000 | -15.0000 |
| greedy_best_of_orders | 0.0000 | 0.0000 | 28.0000 | 28.0000 | 0.0483 | - | 0.5000 | 0.0000 |
| mcts_no_model | 0.0000 | 0.0000 | 25.0000 | 25.0000 | 2.3336 | 4.0000 | 1.0000 | -3.0000 |
| mcts_trained_curriculum | 0.0000 | 0.0000 | 16.0000 | 16.0000 | 13.6536 | 18.0000 | 1.0000 | -12.0000 |
| mcts_trained_small | 0.0000 | 0.0000 | 16.0000 | 16.0000 | 13.6165 | 18.0000 | 1.0000 | -12.0000 |
| mcts_untrained | 0.0000 | 0.0000 | 26.0000 | 26.0000 | 3.8083 | 2.0000 | 1.0000 | -2.0000 |

### Compute-normalized

| Method | gain_per_sec | gain_per_primitive | anytime_improvement_per_sec |
|---|---:|---:|---:|
| fixed_tabu_recolor | 35.2155 | 0.6000 | 35.2155 |
| greedy_best_of_orders | 0.0000 | - | 0.0000 |
| mcts_no_model | 1.2856 | 0.7500 | 2.0941 |
| mcts_trained_curriculum | 0.8789 | 0.6667 | 0.0000 |
| mcts_trained_small | 0.8813 | 0.6667 | 0.0000 |
| mcts_untrained | 0.5252 | 1.0000 | 0.0000 |

### Paired Comparisons

| Pair | n_pairs | win_rate(a>b) | delta_mean(a-b) | delta_ci95(a-b) |
|---|---:|---:|---:|---:|
| mcts_trained_curriculum_vs_mcts_trained_small | 1 | 0.5000 | 0.0000 | [0.0000, 0.0000] |
| mcts_trained_small_vs_greedy_best_of_orders | 1 | 1.0000 | -12.0000 | [-12.0000, -12.0000] |
| mcts_trained_small_vs_mcts_no_model | 1 | 1.0000 | -9.0000 | [-9.0000, -9.0000] |
| mcts_trained_small_vs_mcts_untrained | 1 | 1.0000 | -10.0000 | [-10.0000, -10.0000] |

