# Final Report — small-to-large ladder v3

## Configuration

- session: `/Data/janis.aiad/abstractrl/src/AbstractBeam/crossbeam/experiment/v3/runs/gcp_trm_scaleup_v3/small_to_large_ladder_1776706193`
- sizes: `[100, 200]`
- per-size: `2`
- budgets: `['12:24']`
- timeout_sec: `40.0`
- profile_every: `4`

## Budget 12:24

| Method | solved_rate | timeout_rate | mean_conf | median_conf | mean_time_s | mean_primitive_calls | win_rate_vs_greedy | delta_mean_vs_greedy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed_tabu_recolor | 0.0000 | 0.0000 | 40.0000 | 38.5000 | 0.8739 | 26.0000 | 1.0000 | -19.2500 |
| greedy_best_of_orders | 0.0000 | 0.0000 | 59.2500 | 58.5000 | 0.0872 | - | 0.5000 | 0.0000 |
| mcts_no_model | 0.0000 | 0.0000 | 53.7500 | 53.0000 | 3.0080 | 4.2500 | 1.0000 | -5.5000 |
| mcts_trained_curriculum | 0.0000 | 0.5000 | 16.0000 | 16.0000 | 27.2487 | 22.0000 | 0.5000 | -12.5000 |
| mcts_trained_small | 0.0000 | 0.5000 | 16.0000 | 16.0000 | 27.0556 | 22.0000 | 0.5000 | -12.5000 |
| mcts_untrained | 0.0000 | 0.0000 | 58.7500 | 58.0000 | 5.3993 | 5.5000 | 0.7500 | -0.5000 |

### Compute-normalized

| Method | gain_per_sec | gain_per_primitive | anytime_improvement_per_sec |
|---|---:|---:|---:|
| fixed_tabu_recolor | 26.1153 | 0.7248 | 23.5265 |
| greedy_best_of_orders | 0.0000 | - | 0.0000 |
| mcts_no_model | 1.9060 | 1.3042 | 1.3155 |
| mcts_trained_curriculum | 0.8700 | 0.5833 | 0.1221 |
| mcts_trained_small | 0.8955 | 0.5833 | 0.1258 |
| mcts_untrained | 0.0776 | 0.2292 | 0.1922 |

