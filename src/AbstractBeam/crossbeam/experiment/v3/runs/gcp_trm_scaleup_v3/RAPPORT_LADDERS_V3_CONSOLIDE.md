# Rapport consolidé — ladders V3 (`FINAL_LADDER_REPORT`)

Document généré en regroupant tous les rapports finaux présents sous `runs/gcp_trm_scaleup_v3/`, plus un résumé du run d’entraînement `big_train_1776710200`.

## Table des matières

1. [Résumé entraînement `big_train_1776710200`](#résumé-entraînement-big_train_1776710200)
2. [Session `small_to_large_ladder_1776709550`](#session-small_to_large_ladder_1776709550) — principal (2 budgets, macros, `per_size=10`)
3. [Session `small_to_large_ladder_1776709278`](#session-small_to_large_ladder_1776709278)
4. [Session `small_to_large_ladder_1776708524`](#session-small_to_large_ladder_1776708524)
5. [Session `small_to_large_ladder_1776706193`](#session-small_to_large_ladder_1776706193)
6. [Session `small_to_large_ladder_1776705423`](#session-small_to_large_ladder_1776705423)

---

## Résumé entraînement `big_train_1776710200`

- **Chemin** : `gcp_trm_scaleup_v3/big_train_1776710200/train_log.jsonl`
- **Epochs** : 20 (indices 0–19)
- **Meilleure `valid_loss` observée** : ≈ **0,8889** (atteinte à l’epoch 19 ; c’est aussi la dernière ligne du log)
- **`valid_ce` à cette epoch** : ≈ **0,5487**
- **Somme des `time_sec` par epoch dans le log** : ≈ **39,4 s** (hors overhead lancement ; un wall-clock mesuré côté terminal peut différer légèrement)

---

## Session `small_to_large_ladder_1776709550`

### Final Report — small-to-large ladder v3

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

---

## Session `small_to_large_ladder_1776709278`

### Final Report — small-to-large ladder v3

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

---

## Session `small_to_large_ladder_1776708524`

### Final Report — small-to-large ladder v3

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

---

## Session `small_to_large_ladder_1776706193`

### Final Report — small-to-large ladder v3

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

*Remarque : le fichier `FINAL_LADDER_REPORT.md` d’origine pour cette session ne contenait pas de section « Paired Comparisons ».*

---

## Session `small_to_large_ladder_1776705423`

### Final Report — small-to-large ladder v3

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

*Remarque : pas de méthode curriculum ni section « Paired Comparisons » dans le fichier d’origine.*

---

## Sources (chemins relatifs au dépôt)

| Session | Rapport source |
|---------|----------------|
| 1776709550 | `src/AbstractBeam/crossbeam/experiment/v3/runs/gcp_trm_scaleup_v3/small_to_large_ladder_1776709550/FINAL_LADDER_REPORT.md` |
| 1776709278 | `.../small_to_large_ladder_1776709278/FINAL_LADDER_REPORT.md` |
| 1776708524 | `.../small_to_large_ladder_1776708524/FINAL_LADDER_REPORT.md` |
| 1776706193 | `.../small_to_large_ladder_1776706193/FINAL_LADDER_REPORT.md` |
| 1776705423 | `.../small_to_large_ladder_1776705423/FINAL_LADDER_REPORT.md` |

Chaque session dispose aussi d’un `ladder_results.json` au même niveau que le rapport final.
