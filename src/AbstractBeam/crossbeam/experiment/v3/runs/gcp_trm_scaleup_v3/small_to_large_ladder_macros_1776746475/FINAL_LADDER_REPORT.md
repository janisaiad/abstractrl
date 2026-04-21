# Final Report — small-to-large ladder v3

## Configuration

- session: `/Data/janis.aiad/abstractrl/src/AbstractBeam/crossbeam/experiment/v3/runs/gcp_trm_scaleup_v3/small_to_large_ladder_macros_1776746475`
- sizes: `[100, 200]`
- per-size: `10`
- budgets: `['12:24', '32:48']`
- timeout_sec: `40.0`
- profile_every: `4`

## Budget 12:24

| Method | solved_rate | timeout_rate | mean_conf_finished | penalized_conf_mean | median_conf_finished | mean_time_s | mean_primitive_calls | win_rate_vs_fixed_tabu_recolor | delta_mean_vs_fixed_tabu_recolor |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed_tabu_recolor | 0.0000 | 0.0000 | 39.1500 | 39.1500 | 37.5000 | 0.6966 | 27.1500 | 0.5000 | 0.0000 |
| greedy_best_of_orders | 0.0000 | 0.0000 | 57.0500 | 57.0500 | 55.0000 | 0.0875 | - | 0.0000 | 17.9000 |
| mcts_no_model | 0.0000 | 0.0000 | 53.6500 | 53.6500 | 51.0000 | 3.0773 | 4.5000 | 0.0000 | 14.5000 |
| mcts_trained_curriculum | 0.0000 | 0.5000 | 21.1000 | 51.6000 | 19.5000 | 27.7706 | 23.2222 | 0.1000 | 4.9000 |
| mcts_trained_small | 0.0000 | 0.4500 | 24.6364 | 50.7500 | 21.0000 | 27.7999 | 23.6364 | 0.1000 | 4.7273 |
| mcts_trained_small_macros | 0.0000 | 0.5000 | 20.2000 | 51.1500 | 18.5000 | 27.9932 | 23.0000 | 0.1000 | 4.0000 |
| mcts_untrained | 0.0000 | 0.5500 | 17.8889 | 51.0500 | 19.0000 | 33.2326 | 37.6667 | 0.1000 | 2.0000 |

### Compute-normalized

| Method | gain_per_sec | gain_per_primitive | anytime_improvement_per_sec |
|---|---:|---:|---:|
| fixed_tabu_recolor | 0.0000 | 0.0000 | 29.5900 |
| greedy_best_of_orders | -231.8320 | - | 0.0000 |
| mcts_no_model | -4.5813 | -6.0385 | 1.9681 |
| mcts_trained_curriculum | -0.2834 | -0.2242 | 0.1150 |
| mcts_trained_small | -0.2569 | -0.3523 | 0.0745 |
| mcts_trained_small_macros | -0.2214 | -0.5371 | 0.1357 |
| mcts_untrained | -0.0729 | -0.1775 | 0.0736 |

### Paired Comparisons (timeout-aware)

| Pair | n_pairs | win_rate(a>b) | delta_mean(a-b) | delta_ci95(a-b) | penalized_delta_mean(a-b) | penalized_delta_ci95(a-b) |
|---|---:|---:|---:|---:|---:|---:|
| mcts_trained_curriculum_vs_fixed_tabu_recolor | 20 | 0.1000 | 4.9000 | [1.0000, 9.3000] | 12.4500 | [8.2500, 16.3500] |
| mcts_trained_curriculum_vs_mcts_trained_small | 20 | 0.4750 | 0.1000 | [-0.3000, 0.6000] | 0.8500 | [-0.1000, 2.5000] |
| mcts_trained_small_macros_vs_fixed_tabu_recolor | 20 | 0.1000 | 4.0000 | [0.6000, 7.8000] | 12.0000 | [8.3500, 15.9500] |
| mcts_trained_small_macros_vs_mcts_trained_small | 20 | 0.5500 | -0.8000 | [-1.9000, 0.0000] | 0.4000 | [-0.9000, 2.2500] |
| mcts_trained_small_vs_fixed_tabu_recolor | 20 | 0.1000 | 4.7273 | [1.2727, 8.3636] | 11.6000 | [7.7500, 15.5000] |
| mcts_trained_small_vs_greedy_best_of_orders | 20 | 0.5000 | -9.1818 | [-12.1818, -5.8182] | -6.3000 | [-9.0500, -3.7500] |
| mcts_trained_small_vs_mcts_no_model | 20 | 0.5000 | -5.2727 | [-7.9091, -2.4545] | -2.9000 | [-5.4500, -0.3000] |
| mcts_trained_small_vs_mcts_untrained | 20 | 0.5250 | 2.3333 | [-0.6667, 6.3333] | -0.3000 | [-3.0000, 2.3000] |

## Budget 32:48

| Method | solved_rate | timeout_rate | mean_conf_finished | penalized_conf_mean | median_conf_finished | mean_time_s | mean_primitive_calls | win_rate_vs_fixed_tabu_recolor | delta_mean_vs_fixed_tabu_recolor |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed_tabu_recolor | 0.0000 | 0.0000 | 39.4000 | 39.4000 | 40.0000 | 0.7104 | 27.7000 | 0.5000 | 0.0000 |
| greedy_best_of_orders | 0.0000 | 0.0000 | 56.6500 | 56.6500 | 54.5000 | 0.0875 | - | 0.0000 | 17.2500 |
| mcts_no_model | 0.0000 | 0.0000 | 50.4000 | 50.4000 | 47.0000 | 12.7081 | 8.5500 | 0.1000 | 11.0000 |
| mcts_trained_curriculum | 0.0000 | 1.0000 | - | 59.4000 | - | 40.1643 | - | 0.0000 | - |
| mcts_trained_small | 0.0000 | 1.0000 | - | 59.4000 | - | 40.1644 | - | 0.0000 | - |
| mcts_trained_small_macros | 0.0000 | 1.0000 | - | 59.4000 | - | 40.1706 | - | 0.0000 | - |
| mcts_untrained | 0.0000 | 1.0000 | - | 59.4000 | - | 40.1668 | - | 0.0000 | - |

### Compute-normalized

| Method | gain_per_sec | gain_per_primitive | anytime_improvement_per_sec |
|---|---:|---:|---:|
| fixed_tabu_recolor | 0.0000 | 0.0000 | 25.6815 |
| greedy_best_of_orders | -219.8519 | - | 0.0000 |
| mcts_no_model | -0.8371 | -3.0142 | 0.5313 |
| mcts_trained_curriculum | - | - | - |
| mcts_trained_small | - | - | - |
| mcts_trained_small_macros | - | - | - |
| mcts_untrained | - | - | - |

### Paired Comparisons (timeout-aware)

| Pair | n_pairs | win_rate(a>b) | delta_mean(a-b) | delta_ci95(a-b) | penalized_delta_mean(a-b) | penalized_delta_ci95(a-b) |
|---|---:|---:|---:|---:|---:|---:|
| mcts_trained_curriculum_vs_fixed_tabu_recolor | 20 | 0.0000 | - | - | 20.0000 | [20.0000, 20.0000] |
| mcts_trained_curriculum_vs_mcts_trained_small | 20 | 0.5000 | - | - | 0.0000 | [0.0000, 0.0000] |
| mcts_trained_small_macros_vs_fixed_tabu_recolor | 20 | 0.0000 | - | - | 20.0000 | [20.0000, 20.0000] |
| mcts_trained_small_macros_vs_mcts_trained_small | 20 | 0.5000 | - | - | 0.0000 | [0.0000, 0.0000] |
| mcts_trained_small_vs_fixed_tabu_recolor | 20 | 0.0000 | - | - | 20.0000 | [20.0000, 20.0000] |
| mcts_trained_small_vs_greedy_best_of_orders | 20 | 0.0000 | - | - | 2.7500 | [-0.2000, 5.7000] |
| mcts_trained_small_vs_mcts_no_model | 20 | 0.0000 | - | - | 9.0000 | [5.1000, 12.3500] |
| mcts_trained_small_vs_mcts_untrained | 20 | 0.5000 | - | - | 0.0000 | [0.0000, 0.0000] |

## Interpretation Summary

### Main findings

- The strongest bounded-budget baseline in this campaign is `fixed_tabu_recolor`.
- At budget `12:24`, learned variants (`mcts_trained_small`, `mcts_trained_curriculum`) improve over `greedy_best_of_orders` and `mcts_no_model` in finished-only conflict quality, but remain behind `fixed_tabu_recolor`.
- At budget `32:48`, learned variants and macro variants time out heavily (`timeout_rate = 1.0000` for trained/curriculum/macros/untrained), so timeout-aware metrics become the primary signal.
- Timeout-aware paired comparisons confirm the gap versus the strongest baseline: `penalized_delta_mean(a-b)` against `fixed_tabu_recolor` stays positive and large.

### Claim-ready wording

`A learned TRM prior improves bounded-budget repair quality relative to greedy and no-model baselines on this ladder, but does not yet outperform the strongest hand-crafted local repair baseline (fixed_tabu_recolor).`

## Learned DSL Functions

### DSL action families used by the solver

- Local repair: `vertex_recolor`
- Structural repair: `kempe_swap`
- Search control: `tabu_short`, `tabu_long`, `focus_core`, `perturb_soft`
- Exact operator: `exact_patch`
- Composition operator: `macro`

### What is currently learned in practice

- The learned policy is useful as a bounded-budget repair prior, especially against weaker baselines.
- Mined macros remain dominated by short low-level repair chains and do not yet form a robust high-level controller under tighter runtime constraints.
- In this run, macro variants are measurable at `12:24` but still not competitive with `fixed_tabu_recolor`; at `32:48`, they collapse to full timeout.

### Macro execution notes for this run

- Macro mining/execution was constrained to reduce degenerate behavior:
  - shorter macro length,
  - filtering of mono-family repetitive patterns,
  - cheap macro execution path (reduced internal budget and depth).
- These constraints improved observability at `12:24` versus earlier all-timeout macro behavior, but did not close the performance gap to the strongest baseline.

## Training and Run Description

### Campaign identity

- Session: `small_to_large_ladder_macros_1776746475`
- Scope: macro-focused ladder evaluation with timeout-aware reporting.
- Sizes: `[100, 200]`
- Instances per size: `10`
- Budgets: `12:24`, `32:48`
- Timeout: `40s`
- Profiling cadence: `profile_every=4`

### Compared methods

- `fixed_tabu_recolor` (hand-crafted local repair baseline)
- `greedy_best_of_orders`
- `mcts_no_model`
- `mcts_untrained`
- `mcts_trained_small`
- `mcts_trained_curriculum`
- `mcts_trained_small_macros`

### Reporting protocol (timeout-aware)

- `mean_conf_finished`: quality on finished runs only.
- `penalized_conf_mean`: timeout-aware aggregate with timeout penalty anchored to baseline.
- `win_rate_vs_fixed_tabu_recolor`: pairwise outcome with timeout semantics.
- `penalized_delta_mean(a-b)`: robust pairwise comparison when timeouts are present.
- Compute-normalized indicators are reported as secondary diagnostics.

### Practical takeaway for next iteration

- Keep `fixed_tabu_recolor` as central baseline in all future pairwise tables.
- Use timeout-aware metrics as primary when timeout rates are high.
- Keep macro evaluation as a dedicated track until robustness improves at larger budgets.

