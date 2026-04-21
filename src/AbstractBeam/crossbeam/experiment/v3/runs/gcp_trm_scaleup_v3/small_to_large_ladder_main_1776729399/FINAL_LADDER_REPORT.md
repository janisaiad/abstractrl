# Final Report — small-to-large ladder v3

## Configuration

- session: `/Data/janis.aiad/abstractrl/src/AbstractBeam/crossbeam/experiment/v3/runs/gcp_trm_scaleup_v3/small_to_large_ladder_main_1776729399`
- sizes: `[100, 200, 400]`
- per-size: `10`
- budgets: `['12:24', '32:48', '64:96']`
- timeout_sec: `60.0`
- profile_every: `4`

## Budget 12:24

| Method | solved_rate | timeout_rate | mean_conf_finished | penalized_conf_mean | median_conf_finished | mean_time_s | mean_primitive_calls | win_rate_vs_fixed_tabu_recolor | delta_mean_vs_fixed_tabu_recolor |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed_tabu_recolor | 0.0000 | 0.0000 | 99.0667 | 99.0667 | 59.5000 | 0.9223 | 27.0000 | 0.5000 | 0.0000 |
| greedy_best_of_orders | 0.0000 | 0.0000 | 121.6000 | 121.6000 | 85.0000 | 0.1796 | - | 0.0000 | 22.5333 |
| mcts_no_model | 0.0000 | 0.0000 | 117.5000 | 117.5000 | 82.5000 | 4.5046 | 4.4333 | 0.0000 | 18.4333 |
| mcts_trained_curriculum | 0.0000 | 0.3333 | 47.8000 | 111.5667 | 46.0000 | 40.6183 | 22.5263 | 0.1000 | 8.7500 |
| mcts_trained_small | 0.0000 | 0.3333 | 48.0500 | 111.7333 | 46.5000 | 40.8237 | 22.3000 | 0.1000 | 9.0000 |
| mcts_untrained | 0.0000 | 0.3333 | 48.4500 | 112.0000 | 49.0000 | 44.4743 | 20.4737 | 0.1167 | 9.4000 |

### Compute-normalized

| Method | gain_per_sec | gain_per_primitive | anytime_improvement_per_sec |
|---|---:|---:|---:|
| fixed_tabu_recolor | 0.0000 | 0.0000 | 28.1754 |
| greedy_best_of_orders | -184.7860 | - | 0.0000 |
| mcts_no_model | -4.2045 | -9.0585 | 1.7522 |
| mcts_trained_curriculum | -0.2497 | -0.5894 | 0.0879 |
| mcts_trained_small | -0.2516 | -0.5804 | 0.0589 |
| mcts_untrained | -0.2276 | -0.7570 | 0.0609 |

### Paired Comparisons (timeout-aware)

| Pair | n_pairs | win_rate(a>b) | delta_mean(a-b) | delta_ci95(a-b) | penalized_delta_mean(a-b) | penalized_delta_ci95(a-b) |
|---|---:|---:|---:|---:|---:|---:|
| mcts_trained_curriculum_vs_fixed_tabu_recolor | 30 | 0.1000 | 8.7500 | [5.3500, 12.1000] | 12.5000 | [9.4333, 15.3333] |
| mcts_trained_curriculum_vs_mcts_trained_small | 30 | 0.5333 | -0.2500 | [-0.8500, 0.3000] | -0.1667 | [-0.5667, 0.2000] |
| mcts_trained_small_macros_vs_fixed_tabu_recolor | 0 | - | - | - | - | - |
| mcts_trained_small_macros_vs_mcts_trained_small | 0 | - | - | - | - | - |
| mcts_trained_small_vs_fixed_tabu_recolor | 30 | 0.1000 | 9.0000 | [5.7000, 12.3500] | 12.6667 | [9.6667, 15.3667] |
| mcts_trained_small_vs_greedy_best_of_orders | 30 | 0.6333 | -9.3500 | [-11.5500, -7.2000] | -9.8667 | [-12.1333, -7.7000] |
| mcts_trained_small_vs_mcts_no_model | 30 | 0.6333 | -5.6000 | [-7.4500, -3.7000] | -5.7667 | [-8.4333, -3.2333] |
| mcts_trained_small_vs_mcts_untrained | 30 | 0.5833 | -0.4000 | [-1.6000, 0.7500] | -0.2667 | [-1.0667, 0.4667] |

## Budget 32:48

| Method | solved_rate | timeout_rate | mean_conf_finished | penalized_conf_mean | median_conf_finished | mean_time_s | mean_primitive_calls | win_rate_vs_fixed_tabu_recolor | delta_mean_vs_fixed_tabu_recolor |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed_tabu_recolor | 0.0000 | 0.0000 | 99.9667 | 99.9667 | 59.5000 | 0.8737 | 27.0667 | 0.5000 | 0.0000 |
| greedy_best_of_orders | 0.0000 | 0.0000 | 121.1667 | 121.1667 | 84.5000 | 0.1754 | - | 0.0000 | 21.2000 |
| mcts_no_model | 0.0000 | 0.0000 | 114.6667 | 114.6667 | 77.5000 | 24.6835 | 7.5000 | 0.0333 | 14.7000 |
| mcts_trained_curriculum | 0.0000 | 1.0000 | - | 119.9667 | - | 60.1824 | - | 0.0000 | - |
| mcts_trained_small | 0.0000 | 1.0000 | - | 119.9667 | - | 60.1737 | - | 0.0000 | - |
| mcts_untrained | 0.0000 | 1.0000 | - | 119.9667 | - | 60.1725 | - | 0.0000 | - |

### Compute-normalized

| Method | gain_per_sec | gain_per_primitive | anytime_improvement_per_sec |
|---|---:|---:|---:|
| fixed_tabu_recolor | 0.0000 | 0.0000 | 28.6851 |
| greedy_best_of_orders | -178.0942 | - | 0.0000 |
| mcts_no_model | -0.7080 | -4.4758 | 0.3901 |
| mcts_trained_curriculum | - | - | - |
| mcts_trained_small | - | - | - |
| mcts_untrained | - | - | - |

### Paired Comparisons (timeout-aware)

| Pair | n_pairs | win_rate(a>b) | delta_mean(a-b) | delta_ci95(a-b) | penalized_delta_mean(a-b) | penalized_delta_ci95(a-b) |
|---|---:|---:|---:|---:|---:|---:|
| mcts_trained_curriculum_vs_fixed_tabu_recolor | 30 | 0.0000 | - | - | 20.0000 | [20.0000, 20.0000] |
| mcts_trained_curriculum_vs_mcts_trained_small | 30 | 0.5000 | - | - | 0.0000 | [0.0000, 0.0000] |
| mcts_trained_small_macros_vs_fixed_tabu_recolor | 0 | - | - | - | - | - |
| mcts_trained_small_macros_vs_mcts_trained_small | 0 | - | - | - | - | - |
| mcts_trained_small_vs_fixed_tabu_recolor | 30 | 0.0000 | - | - | 20.0000 | [20.0000, 20.0000] |
| mcts_trained_small_vs_greedy_best_of_orders | 30 | 0.0000 | - | - | -1.2000 | [-4.0333, 1.7000] |
| mcts_trained_small_vs_mcts_no_model | 30 | 0.0000 | - | - | 5.3000 | [1.9333, 8.6333] |
| mcts_trained_small_vs_mcts_untrained | 30 | 0.5000 | - | - | 0.0000 | [0.0000, 0.0000] |

## Budget 64:96

| Method | solved_rate | timeout_rate | mean_conf_finished | penalized_conf_mean | median_conf_finished | mean_time_s | mean_primitive_calls | win_rate_vs_fixed_tabu_recolor | delta_mean_vs_fixed_tabu_recolor |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed_tabu_recolor | 0.0000 | 0.0000 | 99.4333 | 99.4333 | 59.5000 | 0.9791 | 27.7333 | 0.5000 | 0.0000 |
| greedy_best_of_orders | 0.0000 | 0.0000 | 120.6667 | 120.6667 | 84.5000 | 0.1757 | - | 0.0000 | 21.2333 |
| mcts_no_model | 0.0000 | 0.6667 | 21.9000 | 114.7333 | 20.5000 | 48.2753 | 13.6000 | 0.0667 | 5.9000 |
| mcts_trained_curriculum | 0.0000 | 1.0000 | - | 119.4333 | - | 60.2092 | - | 0.0000 | - |
| mcts_trained_small | 0.0000 | 1.0000 | - | 119.4333 | - | 60.2415 | - | 0.0000 | - |
| mcts_untrained | 0.0000 | 1.0000 | - | 119.4333 | - | 60.2380 | - | 0.0000 | - |

### Compute-normalized

| Method | gain_per_sec | gain_per_primitive | anytime_improvement_per_sec |
|---|---:|---:|---:|
| fixed_tabu_recolor | 0.0000 | 0.0000 | 26.2369 |
| greedy_best_of_orders | -176.7334 | - | 0.0000 |
| mcts_no_model | -0.2377 | -0.4951 | 0.2499 |
| mcts_trained_curriculum | - | - | - |
| mcts_trained_small | - | - | - |
| mcts_untrained | - | - | - |

### Paired Comparisons (timeout-aware)

| Pair | n_pairs | win_rate(a>b) | delta_mean(a-b) | delta_ci95(a-b) | penalized_delta_mean(a-b) | penalized_delta_ci95(a-b) |
|---|---:|---:|---:|---:|---:|---:|
| mcts_trained_curriculum_vs_fixed_tabu_recolor | 30 | 0.0000 | - | - | 20.0000 | [20.0000, 20.0000] |
| mcts_trained_curriculum_vs_mcts_trained_small | 30 | 0.5000 | - | - | 0.0000 | [0.0000, 0.0000] |
| mcts_trained_small_macros_vs_fixed_tabu_recolor | 0 | - | - | - | - | - |
| mcts_trained_small_macros_vs_mcts_trained_small | 0 | - | - | - | - | - |
| mcts_trained_small_vs_fixed_tabu_recolor | 30 | 0.0000 | - | - | 20.0000 | [20.0000, 20.0000] |
| mcts_trained_small_vs_greedy_best_of_orders | 30 | 0.0000 | - | - | -1.2333 | [-4.7000, 2.0333] |
| mcts_trained_small_vs_mcts_no_model | 30 | 0.3333 | - | - | 4.7000 | [2.0667, 7.4333] |
| mcts_trained_small_vs_mcts_untrained | 30 | 0.5000 | - | - | 0.0000 | [0.0000, 0.0000] |

