# Final Report — small-to-large ladder v3

## Configuration

- session: `/Data/janis.aiad/abstractrl/src/AbstractBeam/crossbeam/experiment/v3/runs/gcp_trm_scaleup_v3/small_to_large_ladder_1776705423`
- sizes: `[100, 200]`
- per-size: `2`
- budgets: `['12:24']`
- timeout_sec: `40.0`
- profile_every: `4`

## Budget 12:24

| Method | solved_rate | timeout_rate | mean_conf | median_conf | mean_time_s | mean_primitive_calls | win_rate_vs_greedy | delta_mean_vs_greedy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed_tabu_recolor | 0.0000 | 0.0000 | 40.0000 | 38.5000 | 0.8611 | 26.0000 | 1.0000 | -19.2500 |
| greedy_best_of_orders | 0.0000 | 0.0000 | 59.2500 | 58.5000 | 0.0852 | - | 0.5000 | 0.0000 |
| mcts_no_model | 0.0000 | 0.0000 | 53.7500 | 53.0000 | 3.0397 | 4.2500 | 1.0000 | -5.5000 |
| mcts_trained_small | 0.0000 | 0.5000 | 17.5000 | 17.5000 | 28.5498 | 14.0000 | 0.5000 | -11.0000 |
| mcts_untrained | 0.0000 | 0.0000 | 52.7500 | 51.0000 | 5.1777 | 7.2500 | 1.0000 | -6.5000 |

### Compute-normalized

| Method | gain_per_sec | gain_per_primitive | anytime_improvement_per_sec |
|---|---:|---:|---:|
| fixed_tabu_recolor | 26.5435 | 0.7248 | 23.8932 |
| greedy_best_of_orders | 0.0000 | - | 0.0000 |
| mcts_no_model | 1.8938 | 1.3042 | 1.3071 |
| mcts_trained_small | 0.6522 | 0.8075 | 0.0553 |
| mcts_untrained | 1.3202 | 0.8688 | 0.8400 |

