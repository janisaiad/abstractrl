# Final Report — small-to-large ladder v3

## Configuration

- session: `/Data/janis.aiad/abstractrl/src/AbstractBeam/crossbeam/experiment/v3/runs/gcp_trm_scaleup_v3/small_to_large_ladder_1776709550`
- sizes: `[100, 200]`
- per-size: `10`
- budgets: `['12:24', '32:48']`
- timeout_sec: `60.0`
- profile_every: `4`

## Budget 12:24

| Method | solved_rate | timeout_rate | mean_conf | median_conf | mean_time_s | mean_primitive_calls | win_rate_vs_greedy | delta_mean_vs_greedy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed_tabu_recolor | 0.0000 | 0.0000 | 39.1500 | 37.5000 | 0.6891 | 27.1500 | 1.0000 | -17.9000 |
| greedy_best_of_orders | 0.0000 | 0.0000 | 57.0500 | 55.0000 | 0.0870 | - | 0.5000 | 0.0000 |
| mcts_no_model | 0.0000 | 0.0000 | 53.6500 | 51.0000 | 3.0690 | 4.5000 | 0.8500 | -3.4000 |
| mcts_trained_curriculum | 0.0000 | 0.0000 | 47.8000 | 46.0000 | 28.4518 | 22.5263 | 0.9500 | -9.2500 |
| mcts_trained_small | 0.0000 | 0.0000 | 48.0500 | 46.5000 | 28.4677 | 22.3000 | 0.9500 | -9.0000 |
| mcts_trained_small_macros | 0.0000 | 1.0000 | - | - | 60.1669 | - | 0.0000 | - |
| mcts_untrained | 0.0000 | 0.0000 | 50.8000 | 49.0000 | 5.3925 | 11.9474 | 0.9500 | -6.2500 |

### Compute-normalized

| Method | gain_per_sec | gain_per_primitive | anytime_improvement_per_sec |
|---|---:|---:|---:|
| fixed_tabu_recolor | 30.8983 | 0.7078 | 29.8407 |
| greedy_best_of_orders | 0.0000 | - | 0.0000 |
| mcts_no_model | 1.1983 | 0.7867 | 2.0430 |
| mcts_trained_curriculum | 0.4207 | 0.5071 | 0.0886 |
| mcts_trained_small | 0.4204 | 0.4070 | 0.0592 |
| mcts_trained_small_macros | - | - | - |
| mcts_untrained | 1.2206 | 0.6278 | 0.7032 |

### Paired Comparisons

| Pair | n_pairs | win_rate(a>b) | delta_mean(a-b) | delta_ci95(a-b) |
|---|---:|---:|---:|---:|
| mcts_trained_curriculum_vs_mcts_trained_small | 20 | 0.5500 | -0.2500 | [-0.8000, 0.2500] |
| mcts_trained_small_vs_greedy_best_of_orders | 20 | 0.9500 | -9.0000 | [-11.1500, -6.6500] |
| mcts_trained_small_vs_mcts_no_model | 20 | 0.9500 | -5.6000 | [-7.4000, -3.6500] |
| mcts_trained_small_vs_mcts_untrained | 20 | 0.9000 | -2.7500 | [-3.9500, -1.7000] |

## Budget 32:48

| Method | solved_rate | timeout_rate | mean_conf | median_conf | mean_time_s | mean_primitive_calls | win_rate_vs_greedy | delta_mean_vs_greedy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed_tabu_recolor | 0.0000 | 0.0000 | 39.4000 | 40.0000 | 0.7027 | 27.7000 | 1.0000 | -17.2500 |
| greedy_best_of_orders | 0.0000 | 0.0000 | 56.6500 | 54.5000 | 0.0873 | - | 0.5000 | 0.0000 |
| mcts_no_model | 0.0000 | 0.0000 | 50.4000 | 47.0000 | 12.6600 | 8.5500 | 0.9500 | -6.2500 |
| mcts_trained_curriculum | 0.0000 | 1.0000 | - | - | 60.1711 | - | 0.0000 | - |
| mcts_trained_small | 0.0000 | 1.0000 | - | - | 60.1648 | - | 0.0000 | - |
| mcts_trained_small_macros | 0.0000 | 1.0000 | - | - | 60.1691 | - | 0.0000 | - |
| mcts_untrained | 0.0000 | 0.0000 | 50.8000 | 49.0000 | 15.7120 | 11.9474 | 0.9000 | -5.8500 |

### Compute-normalized

| Method | gain_per_sec | gain_per_primitive | anytime_improvement_per_sec |
|---|---:|---:|---:|
| fixed_tabu_recolor | 26.7689 | 0.6437 | 26.0125 |
| greedy_best_of_orders | 0.0000 | - | 0.0000 |
| mcts_no_model | 0.5780 | 0.6995 | 0.5325 |
| mcts_trained_curriculum | - | - | - |
| mcts_trained_small | - | - | - |
| mcts_trained_small_macros | - | - | - |
| mcts_untrained | 0.4243 | 0.5981 | 0.1100 |

### Paired Comparisons

| Pair | n_pairs | win_rate(a>b) | delta_mean(a-b) | delta_ci95(a-b) |
|---|---:|---:|---:|---:|
| mcts_trained_curriculum_vs_mcts_trained_small | 20 | 0.5000 | - | - |
| mcts_trained_small_vs_greedy_best_of_orders | 20 | 0.0000 | - | - |
| mcts_trained_small_vs_mcts_no_model | 20 | 0.0000 | - | - |
| mcts_trained_small_vs_mcts_untrained | 20 | 0.0000 | - | - |

