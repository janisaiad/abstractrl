# Autorecherche — tri quasi trié (AbstractBeam) et TSP avec baseline

Ce document résume les expériences menées pour relier deux pistes: (1) une zone de transition sur des listes « presque triées », (2) le TSP petit format avec une mesure de qualité inspirée d’un baselin nearest-neighbor et du critère « pas plus de 10 % au-dessus de l’optimal ».

## 1. Tri `Sort(x)` — longueurs et distributions

### 1.1 Sweep longueur (listes aléatoires, entiers uniques dans une plage fixe)

Protocole: une suite de runs séparés pour chaque longueur $L \in \{5,\ldots,18\}$, solution cible `Sort(x)` encodée en objet DSL (pas seulement une chaîne).

Résultat: **succès 100 %** sur tous les $L$ testés (40 tâches d’éval par longueur).

Fichier agrégé: `sort_len5_18_gpu/summary_len5_18.json`.

### 1.2 « Phase transition » avec swaps sur une liste triée initiale

Deux familles de générateurs (voir `crossbeam/data/deepcoder/sort_list_generators.py`):

- `nearly_sorted_swap`: liste triée puis $k$ swaps aléatoires de positions (répétition possible, certains swaps peuvent annuler les précédents).
- `nearly_sorted_adjacent`: swaps d’éléments adjacents uniquement.

Constat principal (grille sur $L=5..10$, incluant `swaps=0` dans un premier sweep):

- `swaps = 0`: entrées **déjà triées** → paires I/O souvent **non informatives** pour apprendre `Sort` (dégénérescence); le taux peut tomber à 0 % pour des runs courts — ce n’est **pas** une mesure fiable de « difficulté structurelle ».
- `swaps >= 2` jusqu’à 20 (et dans le run `from2` jusqu’à 32): **100 %** de succès avec le protocole utilisé (train court, beam fixe).

Fichiers:

- `sort_phase_transition_gpu/phase_transition_summary_5_10.json` (grille incluant `swaps=0`)
- `sort_phase_transition_gpu/phase_from2_until_drop_summary.json` (scan à partir de `swaps>=2`, recherche d’une chute $\le 50$ %; **aucune chute observée** sur la grille testée; seuils `first_drop_leq_50pct` restent `null`).

### 1.3 Run « variants » plus strict (protocole différent, plus difficile)

Un ensemble de suites avec moins de steps / autre mise au point a montré des taux plus bas sur certaines configs (notamment quasi trié long), sans contredire le constat ci-dessus: **la « transition » dépend fortement du protocole** (nombre d’exemples, steps, beam, seed, et du fait que le nombre de swaps ne contrôle pas directement la distance de tri réelle).

Fichier: `sort_variants_gpu_quick/variants_summary.json`.

### 1.4 Interprétation pour une vraie transition de phase

Pour un papier, il vaut mieux paramétrer la difficulté par une grandeur monotone, par exemple:

- proportion d’éléments mal placés, nombre d’inversions, ou distance de Kendall–Tau normalisée;
- ou mélange « aléatoire complet » avec un contrôle explicite du niveau de désordre.

Le paramètre brut « nombre de swaps » n’est pas monotone ni reproductible sans contrainte supplémentaire.

## 2. TSP (ATSP, départ/retour ville 0) — AbstractBeam vs nearest-neighbor

### 2.1 Curriculum $n=3,4,5$ (instances aléatoires, optimum par brute force)

Résumé multi-niveaux: `tsp_curriculum_gpu_quick/curriculum_summary.json` (taux faibles sur ce protocole court).

### 2.2 Baseline Zak / nearest-neighbor

- **Nearest-neighbor** depuis la ville 0 (heuristique simple, déterministe sur une matrice fixée).
- **Règle pratique**: qualité « acceptable » si le coût NN est au plus **10 %** au-dessus de l’optimal exact (sinon considéré comme mauvais dans cette convention).

Comparaison sur le jeu d’éval du run `level3_n5`:

- `tsp_curriculum_gpu_quick/nn_vs_abstractbeam_level3_n5.json`

Sweeps multi-seeds (80 instances par seed, même protocole d’entraînement court):

- `tsp_curriculum_gpu_quick/night_nn_vs_ab_summary.json`

Lecture rapide des agrégats multi-seeds:

- `ab_exact_rate` varie fortement selon la seed (0 % à ~27 % sur 4 seeds).
- `nn_within_10pct_rate` est typiquement **30–52 %** selon la seed.
- `nn_mean_gap_pct` reste souvent **> 10 %** (donc « cooked » au sens de la règle stricte sur la moyenne), alors que le NN peut quand même être proche de l’optimal sur une fraction non triviale d’instances.

**Attention**: AbstractBeam est évalué sur **égalité exacte** avec l’optimal (objectif de synthèse symbolique), alors que le NN est jugé sur un **écart relatif**. Les deux métriques ne répondent pas à la même question.

### 2.3 Instances structurées (option exploratoire)

Résumé rapide hiérarchique vs XOR: `tsp_structured_gpu_quick/structured_summary.json`.

## 3. Pipelines « nuit » et reproductibilité

- Orchestrateur: `nightly/launch_night_all.sh` (enchaîne phase `from2`, recherche large de configs tri, sweep TSP AB vs NN).
- Les gros artefacts (checkpoints, pickles, logs tensorboard) **ne doivent pas** être versionnés; voir `neurips/.gitignore`.

## 4. Pistes suivantes (recherche)

- Tri: mesurer **inversions / tau de Kendall** après génération plutôt que « nombre de swaps ».
- TSP: augmenter $n$ ou le budget de recherche; reporter **gap relatif** des programmes trouvés quand l’optimal exact n’est pas atteint; comparer NN à d’autres baselines (2-opt léger sur petit $n$).

## 5. Commandes pour rerun les expériences

Depuis `src/AbstractBeam`, lancer:

```bash
# 1) Sweep TSP difficulté (config actuelle: variante big-train sur niveaux durs)
uv run python neurips/tsp_difficulty_curve/run_tsp_difficulty_curve.py
```

Sortie principale:

- `neurips/tsp_difficulty_curve/difficulty_curve_summary_bigtrain.json`
- un dossier par niveau dans `neurips/tsp_difficulty_curve/lvl*_*/`

Puis générer les plots:

```bash
# 2) Génération des figures (courbes, box/violin, histogrammes + mini-graphes d'instances)
uv run python neurips/tsp_difficulty_curve/plot_tsp_difficulty_curve.py
```

PNGs générés dans `neurips/tsp_difficulty_curve/`:

- `plot_base_success_curves.png`
- `plot_base_nn_gap_distribution.png`
- `plot_base_nn_gap_violin.png`
- `plot_base_nn_gap_histograms.png`
- `plot_bigtrain_success_curves.png` (si le summary big-train existe)
- `plot_bigtrain_nn_gap_distribution.png` (si le summary big-train existe)
- `plot_bigtrain_nn_gap_violin.png` (si le summary big-train existe)
- `plot_bigtrain_nn_gap_histograms.png` (si le summary big-train existe)
