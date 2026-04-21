# V3 Consolidated Supervisor Report

This document consolidates the latest V3 ladder campaigns under `gcp_trm_scaleup_v3`, including:
- run inventory,
- training statistics,
- macro-mining/execution diagnostics,
- timeout-aware evaluation results,
- claim-ready interpretation.

All numbers below are extracted from run artifacts currently present in this folder.

## 1) Campaign inventory

### Completed ladder reports

- `small_to_large_ladder_1776705423`
- `small_to_large_ladder_1776706193`
- `small_to_large_ladder_1776708524`
- `small_to_large_ladder_1776709278`
- `small_to_large_ladder_1776709550`
- `small_to_large_ladder_main_1776729399` (main campaign, timeout-aware reporting)
- `small_to_large_ladder_macros_1776746475` (dedicated macro campaign, timeout-aware reporting)

### Additional training artifacts present

- `big_train_1776710200` (20-epoch dedicated training)
- `small_to_large_ladder_1776705591` (partial run artifact, training log present)
- `small_to_large_ladder_1776706467` (partial run artifact, training log present)

## 2) Training statistics

## 2.1 Big training run

- Run: `big_train_1776710200`
- Epochs: `20` (0..19)
- Best `valid_loss`: `0.8889` (epoch 19)
- Best `valid_ce`: `0.5487` (epoch 19)
- Last epoch `valid_loss`: `0.8889`
- Sum of per-epoch `time_sec` in log: `39.36s`

Latest epochs (from `train_log.jsonl`):

- epoch 17: `valid_loss=0.9246`, `valid_ce=0.5722`
- epoch 18: `valid_loss=0.9023`, `valid_ce=0.5546`
- epoch 19: `valid_loss=0.8889`, `valid_ce=0.5487`

## 2.2 Small model training used inside ladder sessions

For both new large campaigns (`small_to_large_ladder_main_1776729399`, `small_to_large_ladder_macros_1776746475`), `trained_small/train_log.jsonl` contains 2 epochs:

- epoch 0: `valid_loss=1.6943`, `valid_ce=1.0686`
- epoch 1: `valid_loss=1.3761`, `valid_ce=0.8597`

This confirms the per-session small-trace adaptation improved validation metrics before ladder evaluation.

## 3) Macro mining and filtering diagnostics

Both new timeout-aware campaigns include macro filter stats:

- `small_to_large_ladder_main_1776729399/small_macros.filter_stats.json`
- `small_to_large_ladder_macros_1776746475/small_macros.filter_stats.json`

Values (identical in both):

- input macros: `10`
- kept: `6`
- dropped mono-family: `4`
- dropped missing structural family: `0`

Interpretation:
- degenerate repetitive macros were partially removed (`4/10` dropped),
- but enough macro capacity remained for dedicated macro testing.

## 4) DSL learned-function framing

The solver DSL family space remains:

- local: `vertex_recolor`
- structural: `kempe_swap`
- control/search: `tabu_short`, `tabu_long`, `focus_core`, `perturb_soft`
- exact: `exact_patch`
- composition: `macro`

Observed practical behavior from results:
- the learned prior provides value against weaker baselines in bounded budgets,
- but the strongest hand-coded local baseline (`fixed_tabu_recolor`) remains dominant,
- macro variants are still fragile under tighter runtime constraints (timeouts dominate at higher budgets).

## 5) Main campaign (no macro method) — `small_to_large_ladder_main_1776729399`

Configuration:

- sizes: `[100, 200, 400]`
- per-size: `10`
- budgets: `12:24`, `32:48`, `64:96`
- timeout: `60s`
- reporting baseline: `fixed_tabu_recolor`

### 5.1 Budget 12:24 (main)

Key rows:

- `fixed_tabu_recolor`: `penalized_conf_mean=99.0667`, `timeout_rate=0.0000`
- `mcts_trained_small`: `penalized_conf_mean=111.7333`, `timeout_rate=0.3333`
- `mcts_trained_curriculum`: `penalized_conf_mean=111.5667`, `timeout_rate=0.3333`
- `mcts_no_model`: `penalized_conf_mean=117.5000`, `timeout_rate=0.0000`
- `greedy_best_of_orders`: `penalized_conf_mean=121.6000`, `timeout_rate=0.0000`

Paired (timeout-aware):

- `mcts_trained_small_vs_fixed_tabu_recolor`: `win_rate=0.1000`, `penalized_delta_mean=12.6667`
- `mcts_trained_small_vs_greedy_best_of_orders`: `win_rate=0.6333`, `penalized_delta_mean=-9.8667`
- `mcts_trained_small_vs_mcts_no_model`: `win_rate=0.6333`, `penalized_delta_mean=-5.7667`

Interpretation:
- learned prior improves vs greedy and no-model,
- but remains clearly behind `fixed_tabu_recolor`.

### 5.2 Budget 32:48 (main)

Key rows:

- `fixed_tabu_recolor`: `timeout_rate=0.0000`, `penalized_conf_mean=99.9667`
- `mcts_trained_small`: `timeout_rate=1.0000`, `penalized_conf_mean=119.9667`
- `mcts_trained_curriculum`: `timeout_rate=1.0000`, `penalized_conf_mean=119.9667`
- `mcts_untrained`: `timeout_rate=1.0000`, `penalized_conf_mean=119.9667`
- `mcts_no_model`: `timeout_rate=0.0000`, `penalized_conf_mean=114.6667`

Paired (timeout-aware):

- `mcts_trained_small_vs_fixed_tabu_recolor`: `penalized_delta_mean=20.0000`
- `mcts_trained_small_vs_mcts_no_model`: `penalized_delta_mean=5.3000`

Interpretation:
- trained variants collapse to full timeout,
- timeout-aware score strongly favors fixed baseline.

### 5.3 Budget 64:96 (main)

Key rows:

- `fixed_tabu_recolor`: `timeout_rate=0.0000`, `penalized_conf_mean=99.4333`
- `mcts_trained_small`: `timeout_rate=1.0000`, `penalized_conf_mean=119.4333`
- `mcts_trained_curriculum`: `timeout_rate=1.0000`, `penalized_conf_mean=119.4333`
- `mcts_untrained`: `timeout_rate=1.0000`, `penalized_conf_mean=119.4333`
- `mcts_no_model`: `timeout_rate=0.6667`, `penalized_conf_mean=114.7333`

Paired (timeout-aware):

- `mcts_trained_small_vs_fixed_tabu_recolor`: `penalized_delta_mean=20.0000`
- `mcts_trained_small_vs_mcts_no_model`: `penalized_delta_mean=4.7000`

Interpretation:
- high-budget regime remains dominated by timeout behavior for trained variants.

## 6) Dedicated macro campaign — `small_to_large_ladder_macros_1776746475`

Configuration:

- sizes: `[100, 200]`
- per-size: `10`
- budgets: `12:24`, `32:48`
- timeout: `40s`
- methods include `mcts_trained_small_macros`

### 6.1 Budget 12:24 (macros)

Key rows:

- `fixed_tabu_recolor`: `penalized_conf_mean=39.1500`, `timeout_rate=0.0000`
- `mcts_trained_small`: `penalized_conf_mean=50.7500`, `timeout_rate=0.4500`
- `mcts_trained_small_macros`: `penalized_conf_mean=51.1500`, `timeout_rate=0.5000`
- `mcts_trained_curriculum`: `penalized_conf_mean=51.6000`, `timeout_rate=0.5000`

Paired (timeout-aware):

- `mcts_trained_small_macros_vs_mcts_trained_small`: `win_rate=0.5500`, `delta_mean=-0.8000`, `penalized_delta_mean=0.4000`
- `mcts_trained_small_macros_vs_fixed_tabu_recolor`: `win_rate=0.1000`, `penalized_delta_mean=12.0000`
- `mcts_trained_small_vs_fixed_tabu_recolor`: `win_rate=0.1000`, `penalized_delta_mean=11.6000`

Interpretation:
- macros become measurable at 12:24 (no longer universal timeout),
- but remain non-competitive versus fixed baseline.

### 6.2 Budget 32:48 (macros)

Key rows:

- `fixed_tabu_recolor`: `timeout_rate=0.0000`, `penalized_conf_mean=39.4000`
- `mcts_trained_small`: `timeout_rate=1.0000`, `penalized_conf_mean=59.4000`
- `mcts_trained_small_macros`: `timeout_rate=1.0000`, `penalized_conf_mean=59.4000`
- `mcts_trained_curriculum`: `timeout_rate=1.0000`, `penalized_conf_mean=59.4000`
- `mcts_untrained`: `timeout_rate=1.0000`, `penalized_conf_mean=59.4000`

Paired (timeout-aware):

- `mcts_trained_small_macros_vs_fixed_tabu_recolor`: `penalized_delta_mean=20.0000`
- `mcts_trained_small_vs_fixed_tabu_recolor`: `penalized_delta_mean=20.0000`

Interpretation:
- macro and trained variants still collapse at this budget.

## 7) Historical runs snapshot (pre-timeout-aware reports)

These runs used older report schema (baseline often `greedy_best_of_orders`):

- `small_to_large_ladder_1776705423`
- `small_to_large_ladder_1776706193`
- `small_to_large_ladder_1776708524`
- `small_to_large_ladder_1776709278`
- `small_to_large_ladder_1776709550`

Notable point:

- In `1776709550`, macros timed out at `100%` at both `12:24` and `32:48`.
- New macro campaign reduced timeout at `12:24` (`50%`), showing operational improvement but not a quality win over `fixed_tabu_recolor`.

## 8) Claim-ready scientific summary

Suggested main claim:

`A learned TRM prior trained on small-graph repair traces improves bounded-budget conflict reduction relative to greedy and no-model baselines, including on larger target graphs, but does not yet outperform the strongest hand-crafted local repair baseline (fixed_tabu_recolor).`

Suggested limitation statement:

`At larger budgets in the current implementation, learned variants become timeout-prone, so timeout-aware metrics (penalized conflicts and timeout-aware pairwise deltas) are necessary and currently favor the fixed baseline.`

Suggested macro statement:

`Filtered/cheap macros improved short-budget executability versus earlier all-timeout behavior, but do not yet provide a robust improvement under stricter budgets.`

## 9) Concrete next actions for publication-quality results

- Keep `fixed_tabu_recolor` as the central baseline in all pairwise sections.
- Keep timeout-aware metrics as first-class outputs:
  - `timeout_rate`,
  - `penalized_conf_mean`,
  - timeout-aware paired deltas.
- Continue separating campaigns:
  - main no-macro campaign for primary claim,
  - macro-specific campaign for method development.
- For macro track:
  - preserve family-diversity filtering,
  - further reduce internal macro execution cost or improve macro semantics.

## 10) Source artifact index

Primary consolidated artifacts:

- `.../big_train_1776710200/train_log.jsonl`
- `.../small_to_large_ladder_main_1776729399/FINAL_LADDER_REPORT.md`
- `.../small_to_large_ladder_main_1776729399/ladder_results.json`
- `.../small_to_large_ladder_main_1776729399/small_macros.filter_stats.json`
- `.../small_to_large_ladder_macros_1776746475/FINAL_LADDER_REPORT.md`
- `.../small_to_large_ladder_macros_1776746475/ladder_results.json`
- `.../small_to_large_ladder_macros_1776746475/small_macros.filter_stats.json`

Historical reference reports:

- `.../small_to_large_ladder_1776705423/FINAL_LADDER_REPORT.md`
- `.../small_to_large_ladder_1776706193/FINAL_LADDER_REPORT.md`
- `.../small_to_large_ladder_1776708524/FINAL_LADDER_REPORT.md`
- `.../small_to_large_ladder_1776709278/FINAL_LADDER_REPORT.md`
- `.../small_to_large_ladder_1776709550/FINAL_LADDER_REPORT.md`

# Rapport consolidé — ladders V3 (regénéré automatiquement)

Ce fichier est régénéré à la fin de chaque orchestration. Il réunit tous les `FINAL_LADDER_REPORT.md` présents sous `runs/gcp_trm_scaleup_v3/` au moment de la génération, dans l’ordre alphabétique.

## Table des matières

1. [small_to_large_ladder_1776705423](#small-to-large-ladder-1776705423)
2. [small_to_large_ladder_1776706193](#small-to-large-ladder-1776706193)
3. [small_to_large_ladder_1776708524](#small-to-large-ladder-1776708524)
4. [small_to_large_ladder_1776709278](#small-to-large-ladder-1776709278)
5. [small_to_large_ladder_1776709550](#small-to-large-ladder-1776709550)
6. [small_to_large_ladder_macros_1776746475](#small-to-large-ladder-macros-1776746475)
7. [small_to_large_ladder_main_1776729399](#small-to-large-ladder-main-1776729399)


## Big train logs
- `big_train_1776710200`: epochs=20, last epoch=19, last valid_loss=0.8889, best valid_loss=0.8889 (epoch 19), sum time_sec=39.36s

## small_to_large_ladder_1776705423

- Fichier: `/Data/janis.aiad/abstractrl/src/AbstractBeam/crossbeam/experiment/v3/runs/gcp_trm_scaleup_v3/small_to_large_ladder_1776705423/FINAL_LADDER_REPORT.md`

### Contenu du rapport source — small_to_large_ladder_1776705423

#### Configuration

- session: `/Data/janis.aiad/abstractrl/src/AbstractBeam/crossbeam/experiment/v3/runs/gcp_trm_scaleup_v3/small_to_large_ladder_1776705423`
- sizes: `[100, 200]`
- per-size: `2`
- budgets: `['12:24']`
- timeout_sec: `40.0`
- profile_every: `4`

#### Budget 12:24

| Method | solved_rate | timeout_rate | mean_conf | median_conf | mean_time_s | mean_primitive_calls | win_rate_vs_greedy | delta_mean_vs_greedy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed_tabu_recolor | 0.0000 | 0.0000 | 40.0000 | 38.5000 | 0.8611 | 26.0000 | 1.0000 | -19.2500 |
| greedy_best_of_orders | 0.0000 | 0.0000 | 59.2500 | 58.5000 | 0.0852 | - | 0.5000 | 0.0000 |
| mcts_no_model | 0.0000 | 0.0000 | 53.7500 | 53.0000 | 3.0397 | 4.2500 | 1.0000 | -5.5000 |
| mcts_trained_small | 0.0000 | 0.5000 | 17.5000 | 17.5000 | 28.5498 | 14.0000 | 0.5000 | -11.0000 |
| mcts_untrained | 0.0000 | 0.0000 | 52.7500 | 51.0000 | 5.1777 | 7.2500 | 1.0000 | -6.5000 |

##### Compute-normalized

| Method | gain_per_sec | gain_per_primitive | anytime_improvement_per_sec |
|---|---:|---:|---:|
| fixed_tabu_recolor | 26.5435 | 0.7248 | 23.8932 |
| greedy_best_of_orders | 0.0000 | - | 0.0000 |
| mcts_no_model | 1.8938 | 1.3042 | 1.3071 |
| mcts_trained_small | 0.6522 | 0.8075 | 0.0553 |
| mcts_untrained | 1.3202 | 0.8688 | 0.8400 |

## small_to_large_ladder_1776706193

- Fichier: `/Data/janis.aiad/abstractrl/src/AbstractBeam/crossbeam/experiment/v3/runs/gcp_trm_scaleup_v3/small_to_large_ladder_1776706193/FINAL_LADDER_REPORT.md`

### Contenu du rapport source — small_to_large_ladder_1776706193

#### Configuration

- session: `/Data/janis.aiad/abstractrl/src/AbstractBeam/crossbeam/experiment/v3/runs/gcp_trm_scaleup_v3/small_to_large_ladder_1776706193`
- sizes: `[100, 200]`
- per-size: `2`
- budgets: `['12:24']`
- timeout_sec: `40.0`
- profile_every: `4`

#### Budget 12:24

| Method | solved_rate | timeout_rate | mean_conf | median_conf | mean_time_s | mean_primitive_calls | win_rate_vs_greedy | delta_mean_vs_greedy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed_tabu_recolor | 0.0000 | 0.0000 | 40.0000 | 38.5000 | 0.8739 | 26.0000 | 1.0000 | -19.2500 |
| greedy_best_of_orders | 0.0000 | 0.0000 | 59.2500 | 58.5000 | 0.0872 | - | 0.5000 | 0.0000 |
| mcts_no_model | 0.0000 | 0.0000 | 53.7500 | 53.0000 | 3.0080 | 4.2500 | 1.0000 | -5.5000 |
| mcts_trained_curriculum | 0.0000 | 0.5000 | 16.0000 | 16.0000 | 27.2487 | 22.0000 | 0.5000 | -12.5000 |
| mcts_trained_small | 0.0000 | 0.5000 | 16.0000 | 16.0000 | 27.0556 | 22.0000 | 0.5000 | -12.5000 |
| mcts_untrained | 0.0000 | 0.0000 | 58.7500 | 58.0000 | 5.3993 | 5.5000 | 0.7500 | -0.5000 |

##### Compute-normalized

| Method | gain_per_sec | gain_per_primitive | anytime_improvement_per_sec |
|---|---:|---:|---:|
| fixed_tabu_recolor | 26.1153 | 0.7248 | 23.5265 |
| greedy_best_of_orders | 0.0000 | - | 0.0000 |
| mcts_no_model | 1.9060 | 1.3042 | 1.3155 |
| mcts_trained_curriculum | 0.8700 | 0.5833 | 0.1221 |
| mcts_trained_small | 0.8955 | 0.5833 | 0.1258 |
| mcts_untrained | 0.0776 | 0.2292 | 0.1922 |

## small_to_large_ladder_1776708524

- Fichier: `/Data/janis.aiad/abstractrl/src/AbstractBeam/crossbeam/experiment/v3/runs/gcp_trm_scaleup_v3/small_to_large_ladder_1776708524/FINAL_LADDER_REPORT.md`

### Contenu du rapport source — small_to_large_ladder_1776708524

#### Configuration

- session: `/Data/janis.aiad/abstractrl/src/AbstractBeam/crossbeam/experiment/v3/runs/gcp_trm_scaleup_v3/small_to_large_ladder_1776708524`
- sizes: `[100]`
- per-size: `1`
- budgets: `['12:24']`
- timeout_sec: `30.0`
- profile_every: `4`

#### Budget 12:24

| Method | solved_rate | timeout_rate | mean_conf | median_conf | mean_time_s | mean_primitive_calls | win_rate_vs_greedy | delta_mean_vs_greedy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed_tabu_recolor | 0.0000 | 0.0000 | 13.0000 | 13.0000 | 0.4259 | 25.0000 | 1.0000 | -15.0000 |
| greedy_best_of_orders | 0.0000 | 0.0000 | 28.0000 | 28.0000 | 0.0483 | - | 0.5000 | 0.0000 |
| mcts_no_model | 0.0000 | 0.0000 | 25.0000 | 25.0000 | 2.3336 | 4.0000 | 1.0000 | -3.0000 |
| mcts_trained_curriculum | 0.0000 | 0.0000 | 16.0000 | 16.0000 | 13.6536 | 18.0000 | 1.0000 | -12.0000 |
| mcts_trained_small | 0.0000 | 0.0000 | 16.0000 | 16.0000 | 13.6165 | 18.0000 | 1.0000 | -12.0000 |
| mcts_untrained | 0.0000 | 0.0000 | 26.0000 | 26.0000 | 3.8083 | 2.0000 | 1.0000 | -2.0000 |

##### Compute-normalized

| Method | gain_per_sec | gain_per_primitive | anytime_improvement_per_sec |
|---|---:|---:|---:|
| fixed_tabu_recolor | 35.2155 | 0.6000 | 35.2155 |
| greedy_best_of_orders | 0.0000 | - | 0.0000 |
| mcts_no_model | 1.2856 | 0.7500 | 2.0941 |
| mcts_trained_curriculum | 0.8789 | 0.6667 | 0.0000 |
| mcts_trained_small | 0.8813 | 0.6667 | 0.0000 |
| mcts_untrained | 0.5252 | 1.0000 | 0.0000 |

##### Paired Comparisons

| Pair | n_pairs | win_rate(a>b) | delta_mean(a-b) | delta_ci95(a-b) |
|---|---:|---:|---:|---:|
| mcts_trained_curriculum_vs_mcts_trained_small | 1 | 0.5000 | 0.0000 | [0.0000, 0.0000] |
| mcts_trained_small_vs_greedy_best_of_orders | 1 | 1.0000 | -12.0000 | [-12.0000, -12.0000] |
| mcts_trained_small_vs_mcts_no_model | 1 | 1.0000 | -9.0000 | [-9.0000, -9.0000] |
| mcts_trained_small_vs_mcts_untrained | 1 | 1.0000 | -10.0000 | [-10.0000, -10.0000] |

## small_to_large_ladder_1776709278

- Fichier: `/Data/janis.aiad/abstractrl/src/AbstractBeam/crossbeam/experiment/v3/runs/gcp_trm_scaleup_v3/small_to_large_ladder_1776709278/FINAL_LADDER_REPORT.md`

### Contenu du rapport source — small_to_large_ladder_1776709278

#### Configuration

- session: `/Data/janis.aiad/abstractrl/src/AbstractBeam/crossbeam/experiment/v3/runs/gcp_trm_scaleup_v3/small_to_large_ladder_1776709278`
- sizes: `[100, 200]`
- per-size: `2`
- budgets: `['12:24']`
- timeout_sec: `40.0`
- profile_every: `4`

#### Budget 12:24

| Method | solved_rate | timeout_rate | mean_conf | median_conf | mean_time_s | mean_primitive_calls | win_rate_vs_greedy | delta_mean_vs_greedy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed_tabu_recolor | 0.0000 | 0.0000 | 40.0000 | 38.5000 | 0.8806 | 26.0000 | 1.0000 | -19.2500 |
| greedy_best_of_orders | 0.0000 | 0.0000 | 59.2500 | 58.5000 | 0.0865 | - | 0.5000 | 0.0000 |
| mcts_no_model | 0.0000 | 0.0000 | 53.7500 | 53.0000 | 3.0065 | 4.2500 | 1.0000 | -5.5000 |
| mcts_trained_curriculum | 0.0000 | 0.5000 | 16.0000 | 16.0000 | 27.1956 | 22.0000 | 0.5000 | -12.5000 |
| mcts_trained_small | 0.0000 | 0.5000 | 16.0000 | 16.0000 | 27.0352 | 22.0000 | 0.5000 | -12.5000 |
| mcts_untrained | 0.0000 | 0.0000 | 58.5000 | 58.5000 | 4.4977 | 8.0000 | 0.5000 | -0.7500 |

##### Compute-normalized

| Method | gain_per_sec | gain_per_primitive | anytime_improvement_per_sec |
|---|---:|---:|---:|
| fixed_tabu_recolor | 26.0158 | 0.7248 | 23.4503 |
| greedy_best_of_orders | 0.0000 | - | 0.0000 |
| mcts_no_model | 1.9080 | 1.3042 | 1.3108 |
| mcts_trained_curriculum | 0.8781 | 0.5833 | 0.1224 |
| mcts_trained_small | 0.8973 | 0.5833 | 0.1249 |
| mcts_untrained | 0.2850 | 0.7500 | 1.9358 |

##### Paired Comparisons

| Pair | n_pairs | win_rate(a>b) | delta_mean(a-b) | delta_ci95(a-b) |
|---|---:|---:|---:|---:|
| mcts_trained_curriculum_vs_mcts_trained_small | 4 | 0.5000 | 0.0000 | [0.0000, 0.0000] |
| mcts_trained_small_vs_greedy_best_of_orders | 4 | 0.5000 | -12.5000 | [-13.0000, -12.0000] |
| mcts_trained_small_vs_mcts_no_model | 4 | 0.5000 | -7.5000 | [-9.0000, -6.0000] |
| mcts_trained_small_vs_mcts_untrained | 4 | 0.5000 | -9.5000 | [-13.0000, -6.0000] |

## small_to_large_ladder_1776709550

- Fichier: `/Data/janis.aiad/abstractrl/src/AbstractBeam/crossbeam/experiment/v3/runs/gcp_trm_scaleup_v3/small_to_large_ladder_1776709550/FINAL_LADDER_REPORT.md`

### Contenu du rapport source — small_to_large_ladder_1776709550

#### Configuration

- session: `/Data/janis.aiad/abstractrl/src/AbstractBeam/crossbeam/experiment/v3/runs/gcp_trm_scaleup_v3/small_to_large_ladder_1776709550`
- sizes: `[100, 200]`
- per-size: `10`
- budgets: `['12:24', '32:48']`
- timeout_sec: `60.0`
- profile_every: `4`

#### Budget 12:24

| Method | solved_rate | timeout_rate | mean_conf | median_conf | mean_time_s | mean_primitive_calls | win_rate_vs_greedy | delta_mean_vs_greedy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed_tabu_recolor | 0.0000 | 0.0000 | 39.1500 | 37.5000 | 0.6891 | 27.1500 | 1.0000 | -17.9000 |
| greedy_best_of_orders | 0.0000 | 0.0000 | 57.0500 | 55.0000 | 0.0870 | - | 0.5000 | 0.0000 |
| mcts_no_model | 0.0000 | 0.0000 | 53.6500 | 51.0000 | 3.0690 | 4.5000 | 0.8500 | -3.4000 |
| mcts_trained_curriculum | 0.0000 | 0.0000 | 47.8000 | 46.0000 | 28.4518 | 22.5263 | 0.9500 | -9.2500 |
| mcts_trained_small | 0.0000 | 0.0000 | 48.0500 | 46.5000 | 28.4677 | 22.3000 | 0.9500 | -9.0000 |
| mcts_trained_small_macros | 0.0000 | 1.0000 | - | - | 60.1669 | - | 0.0000 | - |
| mcts_untrained | 0.0000 | 0.0000 | 50.8000 | 49.0000 | 5.3925 | 11.9474 | 0.9500 | -6.2500 |

##### Compute-normalized

| Method | gain_per_sec | gain_per_primitive | anytime_improvement_per_sec |
|---|---:|---:|---:|
| fixed_tabu_recolor | 30.8983 | 0.7078 | 29.8407 |
| greedy_best_of_orders | 0.0000 | - | 0.0000 |
| mcts_no_model | 1.1983 | 0.7867 | 2.0430 |
| mcts_trained_curriculum | 0.4207 | 0.5071 | 0.0886 |
| mcts_trained_small | 0.4204 | 0.4070 | 0.0592 |
| mcts_trained_small_macros | - | - | - |
| mcts_untrained | 1.2206 | 0.6278 | 0.7032 |

##### Paired Comparisons

| Pair | n_pairs | win_rate(a>b) | delta_mean(a-b) | delta_ci95(a-b) |
|---|---:|---:|---:|---:|
| mcts_trained_curriculum_vs_mcts_trained_small | 20 | 0.5500 | -0.2500 | [-0.8000, 0.2500] |
| mcts_trained_small_vs_greedy_best_of_orders | 20 | 0.9500 | -9.0000 | [-11.1500, -6.6500] |
| mcts_trained_small_vs_mcts_no_model | 20 | 0.9500 | -5.6000 | [-7.4000, -3.6500] |
| mcts_trained_small_vs_mcts_untrained | 20 | 0.9000 | -2.7500 | [-3.9500, -1.7000] |

#### Budget 32:48

| Method | solved_rate | timeout_rate | mean_conf | median_conf | mean_time_s | mean_primitive_calls | win_rate_vs_greedy | delta_mean_vs_greedy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed_tabu_recolor | 0.0000 | 0.0000 | 39.4000 | 40.0000 | 0.7027 | 27.7000 | 1.0000 | -17.2500 |
| greedy_best_of_orders | 0.0000 | 0.0000 | 56.6500 | 54.5000 | 0.0873 | - | 0.5000 | 0.0000 |
| mcts_no_model | 0.0000 | 0.0000 | 50.4000 | 47.0000 | 12.6600 | 8.5500 | 0.9500 | -6.2500 |
| mcts_trained_curriculum | 0.0000 | 1.0000 | - | - | 60.1711 | - | 0.0000 | - |
| mcts_trained_small | 0.0000 | 1.0000 | - | - | 60.1648 | - | 0.0000 | - |
| mcts_trained_small_macros | 0.0000 | 1.0000 | - | - | 60.1691 | - | 0.0000 | - |
| mcts_untrained | 0.0000 | 0.0000 | 50.8000 | 49.0000 | 15.7120 | 11.9474 | 0.9000 | -5.8500 |

##### Compute-normalized

| Method | gain_per_sec | gain_per_primitive | anytime_improvement_per_sec |
|---|---:|---:|---:|
| fixed_tabu_recolor | 26.7689 | 0.6437 | 26.0125 |
| greedy_best_of_orders | 0.0000 | - | 0.0000 |
| mcts_no_model | 0.5780 | 0.6995 | 0.5325 |
| mcts_trained_curriculum | - | - | - |
| mcts_trained_small | - | - | - |
| mcts_trained_small_macros | - | - | - |
| mcts_untrained | 0.4243 | 0.5981 | 0.1100 |

##### Paired Comparisons

| Pair | n_pairs | win_rate(a>b) | delta_mean(a-b) | delta_ci95(a-b) |
|---|---:|---:|---:|---:|
| mcts_trained_curriculum_vs_mcts_trained_small | 20 | 0.5000 | - | - |
| mcts_trained_small_vs_greedy_best_of_orders | 20 | 0.0000 | - | - |
| mcts_trained_small_vs_mcts_no_model | 20 | 0.0000 | - | - |
| mcts_trained_small_vs_mcts_untrained | 20 | 0.0000 | - | - |

## small_to_large_ladder_macros_1776746475

- Fichier: `/Data/janis.aiad/abstractrl/src/AbstractBeam/crossbeam/experiment/v3/runs/gcp_trm_scaleup_v3/small_to_large_ladder_macros_1776746475/FINAL_LADDER_REPORT.md`

### Contenu du rapport source — small_to_large_ladder_macros_1776746475

#### Configuration

- session: `/Data/janis.aiad/abstractrl/src/AbstractBeam/crossbeam/experiment/v3/runs/gcp_trm_scaleup_v3/small_to_large_ladder_macros_1776746475`
- sizes: `[100, 200]`
- per-size: `10`
- budgets: `['12:24', '32:48']`
- timeout_sec: `40.0`
- profile_every: `4`

#### Budget 12:24

| Method | solved_rate | timeout_rate | mean_conf_finished | penalized_conf_mean | median_conf_finished | mean_time_s | mean_primitive_calls | win_rate_vs_fixed_tabu_recolor | delta_mean_vs_fixed_tabu_recolor |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed_tabu_recolor | 0.0000 | 0.0000 | 39.1500 | 39.1500 | 37.5000 | 0.6966 | 27.1500 | 0.5000 | 0.0000 |
| greedy_best_of_orders | 0.0000 | 0.0000 | 57.0500 | 57.0500 | 55.0000 | 0.0875 | - | 0.0000 | 17.9000 |
| mcts_no_model | 0.0000 | 0.0000 | 53.6500 | 53.6500 | 51.0000 | 3.0773 | 4.5000 | 0.0000 | 14.5000 |
| mcts_trained_curriculum | 0.0000 | 0.5000 | 21.1000 | 51.6000 | 19.5000 | 27.7706 | 23.2222 | 0.1000 | 4.9000 |
| mcts_trained_small | 0.0000 | 0.4500 | 24.6364 | 50.7500 | 21.0000 | 27.7999 | 23.6364 | 0.1000 | 4.7273 |
| mcts_trained_small_macros | 0.0000 | 0.5000 | 20.2000 | 51.1500 | 18.5000 | 27.9932 | 23.0000 | 0.1000 | 4.0000 |
| mcts_untrained | 0.0000 | 0.5500 | 17.8889 | 51.0500 | 19.0000 | 33.2326 | 37.6667 | 0.1000 | 2.0000 |

##### Compute-normalized

| Method | gain_per_sec | gain_per_primitive | anytime_improvement_per_sec |
|---|---:|---:|---:|
| fixed_tabu_recolor | 0.0000 | 0.0000 | 29.5900 |
| greedy_best_of_orders | -231.8320 | - | 0.0000 |
| mcts_no_model | -4.5813 | -6.0385 | 1.9681 |
| mcts_trained_curriculum | -0.2834 | -0.2242 | 0.1150 |
| mcts_trained_small | -0.2569 | -0.3523 | 0.0745 |
| mcts_trained_small_macros | -0.2214 | -0.5371 | 0.1357 |
| mcts_untrained | -0.0729 | -0.1775 | 0.0736 |

##### Paired Comparisons (timeout-aware)

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

#### Budget 32:48

| Method | solved_rate | timeout_rate | mean_conf_finished | penalized_conf_mean | median_conf_finished | mean_time_s | mean_primitive_calls | win_rate_vs_fixed_tabu_recolor | delta_mean_vs_fixed_tabu_recolor |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed_tabu_recolor | 0.0000 | 0.0000 | 39.4000 | 39.4000 | 40.0000 | 0.7104 | 27.7000 | 0.5000 | 0.0000 |
| greedy_best_of_orders | 0.0000 | 0.0000 | 56.6500 | 56.6500 | 54.5000 | 0.0875 | - | 0.0000 | 17.2500 |
| mcts_no_model | 0.0000 | 0.0000 | 50.4000 | 50.4000 | 47.0000 | 12.7081 | 8.5500 | 0.1000 | 11.0000 |
| mcts_trained_curriculum | 0.0000 | 1.0000 | - | 59.4000 | - | 40.1643 | - | 0.0000 | - |
| mcts_trained_small | 0.0000 | 1.0000 | - | 59.4000 | - | 40.1644 | - | 0.0000 | - |
| mcts_trained_small_macros | 0.0000 | 1.0000 | - | 59.4000 | - | 40.1706 | - | 0.0000 | - |
| mcts_untrained | 0.0000 | 1.0000 | - | 59.4000 | - | 40.1668 | - | 0.0000 | - |

##### Compute-normalized

| Method | gain_per_sec | gain_per_primitive | anytime_improvement_per_sec |
|---|---:|---:|---:|
| fixed_tabu_recolor | 0.0000 | 0.0000 | 25.6815 |
| greedy_best_of_orders | -219.8519 | - | 0.0000 |
| mcts_no_model | -0.8371 | -3.0142 | 0.5313 |
| mcts_trained_curriculum | - | - | - |
| mcts_trained_small | - | - | - |
| mcts_trained_small_macros | - | - | - |
| mcts_untrained | - | - | - |

##### Paired Comparisons (timeout-aware)

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

## small_to_large_ladder_main_1776729399

- Fichier: `/Data/janis.aiad/abstractrl/src/AbstractBeam/crossbeam/experiment/v3/runs/gcp_trm_scaleup_v3/small_to_large_ladder_main_1776729399/FINAL_LADDER_REPORT.md`

### Contenu du rapport source — small_to_large_ladder_main_1776729399

#### Configuration

- session: `/Data/janis.aiad/abstractrl/src/AbstractBeam/crossbeam/experiment/v3/runs/gcp_trm_scaleup_v3/small_to_large_ladder_main_1776729399`
- sizes: `[100, 200, 400]`
- per-size: `10`
- budgets: `['12:24', '32:48', '64:96']`
- timeout_sec: `60.0`
- profile_every: `4`

#### Budget 12:24

| Method | solved_rate | timeout_rate | mean_conf_finished | penalized_conf_mean | median_conf_finished | mean_time_s | mean_primitive_calls | win_rate_vs_fixed_tabu_recolor | delta_mean_vs_fixed_tabu_recolor |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed_tabu_recolor | 0.0000 | 0.0000 | 99.0667 | 99.0667 | 59.5000 | 0.9223 | 27.0000 | 0.5000 | 0.0000 |
| greedy_best_of_orders | 0.0000 | 0.0000 | 121.6000 | 121.6000 | 85.0000 | 0.1796 | - | 0.0000 | 22.5333 |
| mcts_no_model | 0.0000 | 0.0000 | 117.5000 | 117.5000 | 82.5000 | 4.5046 | 4.4333 | 0.0000 | 18.4333 |
| mcts_trained_curriculum | 0.0000 | 0.3333 | 47.8000 | 111.5667 | 46.0000 | 40.6183 | 22.5263 | 0.1000 | 8.7500 |
| mcts_trained_small | 0.0000 | 0.3333 | 48.0500 | 111.7333 | 46.5000 | 40.8237 | 22.3000 | 0.1000 | 9.0000 |
| mcts_untrained | 0.0000 | 0.3333 | 48.4500 | 112.0000 | 49.0000 | 44.4743 | 20.4737 | 0.1167 | 9.4000 |

##### Compute-normalized

| Method | gain_per_sec | gain_per_primitive | anytime_improvement_per_sec |
|---|---:|---:|---:|
| fixed_tabu_recolor | 0.0000 | 0.0000 | 28.1754 |
| greedy_best_of_orders | -184.7860 | - | 0.0000 |
| mcts_no_model | -4.2045 | -9.0585 | 1.7522 |
| mcts_trained_curriculum | -0.2497 | -0.5894 | 0.0879 |
| mcts_trained_small | -0.2516 | -0.5804 | 0.0589 |
| mcts_untrained | -0.2276 | -0.7570 | 0.0609 |

##### Paired Comparisons (timeout-aware)

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

#### Budget 32:48

| Method | solved_rate | timeout_rate | mean_conf_finished | penalized_conf_mean | median_conf_finished | mean_time_s | mean_primitive_calls | win_rate_vs_fixed_tabu_recolor | delta_mean_vs_fixed_tabu_recolor |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed_tabu_recolor | 0.0000 | 0.0000 | 99.9667 | 99.9667 | 59.5000 | 0.8737 | 27.0667 | 0.5000 | 0.0000 |
| greedy_best_of_orders | 0.0000 | 0.0000 | 121.1667 | 121.1667 | 84.5000 | 0.1754 | - | 0.0000 | 21.2000 |
| mcts_no_model | 0.0000 | 0.0000 | 114.6667 | 114.6667 | 77.5000 | 24.6835 | 7.5000 | 0.0333 | 14.7000 |
| mcts_trained_curriculum | 0.0000 | 1.0000 | - | 119.9667 | - | 60.1824 | - | 0.0000 | - |
| mcts_trained_small | 0.0000 | 1.0000 | - | 119.9667 | - | 60.1737 | - | 0.0000 | - |
| mcts_untrained | 0.0000 | 1.0000 | - | 119.9667 | - | 60.1725 | - | 0.0000 | - |

##### Compute-normalized

| Method | gain_per_sec | gain_per_primitive | anytime_improvement_per_sec |
|---|---:|---:|---:|
| fixed_tabu_recolor | 0.0000 | 0.0000 | 28.6851 |
| greedy_best_of_orders | -178.0942 | - | 0.0000 |
| mcts_no_model | -0.7080 | -4.4758 | 0.3901 |
| mcts_trained_curriculum | - | - | - |
| mcts_trained_small | - | - | - |
| mcts_untrained | - | - | - |

##### Paired Comparisons (timeout-aware)

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

#### Budget 64:96

| Method | solved_rate | timeout_rate | mean_conf_finished | penalized_conf_mean | median_conf_finished | mean_time_s | mean_primitive_calls | win_rate_vs_fixed_tabu_recolor | delta_mean_vs_fixed_tabu_recolor |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed_tabu_recolor | 0.0000 | 0.0000 | 99.4333 | 99.4333 | 59.5000 | 0.9791 | 27.7333 | 0.5000 | 0.0000 |
| greedy_best_of_orders | 0.0000 | 0.0000 | 120.6667 | 120.6667 | 84.5000 | 0.1757 | - | 0.0000 | 21.2333 |
| mcts_no_model | 0.0000 | 0.6667 | 21.9000 | 114.7333 | 20.5000 | 48.2753 | 13.6000 | 0.0667 | 5.9000 |
| mcts_trained_curriculum | 0.0000 | 1.0000 | - | 119.4333 | - | 60.2092 | - | 0.0000 | - |
| mcts_trained_small | 0.0000 | 1.0000 | - | 119.4333 | - | 60.2415 | - | 0.0000 | - |
| mcts_untrained | 0.0000 | 1.0000 | - | 119.4333 | - | 60.2380 | - | 0.0000 | - |

##### Compute-normalized

| Method | gain_per_sec | gain_per_primitive | anytime_improvement_per_sec |
|---|---:|---:|---:|
| fixed_tabu_recolor | 0.0000 | 0.0000 | 26.2369 |
| greedy_best_of_orders | -176.7334 | - | 0.0000 |
| mcts_no_model | -0.2377 | -0.4951 | 0.2499 |
| mcts_trained_curriculum | - | - | - |
| mcts_trained_small | - | - | - |
| mcts_untrained | - | - | - |

##### Paired Comparisons (timeout-aware)

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

