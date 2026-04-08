#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import sys
from itertools import permutations
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
_ABSTRACTBEAM_ROOT = os.path.abspath(os.path.join(ROOT, "..", ".."))
if _ABSTRACTBEAM_ROOT not in sys.path:
  sys.path.insert(0, _ABSTRACTBEAM_ROOT)

from crossbeam.common.configs_all import get_config
from crossbeam.dsl import domains
from crossbeam.experiment.run_crossbeam import init_model

BASE_LEVEL_CONFIG = {
    "lvl1_very_easy_n5": {"num_train": 500, "train_steps": 80},
    "lvl2_easy_n5": {"num_train": 500, "train_steps": 80},
    "lvl3_medium_n6": {"num_train": 600, "train_steps": 100},
    "lvl4_hard_n6": {"num_train": 600, "train_steps": 100},
    "lvl5_very_hard_n7": {"num_train": 700, "train_steps": 120},
}

BIGTRAIN_LEVEL_CONFIG = {
    "lvl1_very_easy_n5_bigtrain": {"num_train": 500, "train_steps": 80},
    "lvl2_easy_n5_bigtrain": {"num_train": 700, "train_steps": 80},
    "lvl3_medium_n6_bigtrain": {"num_train": 2200, "train_steps": 140},
    "lvl4_hard_n6_bigtrain": {"num_train": 3200, "train_steps": 160},
    "lvl5_very_hard_n7_bigtrain": {"num_train": 4200, "train_steps": 180},
}


def _load_json(path: str):
  with open(path, "r") as f:
    return json.load(f)


def _parse_matrix_and_opt(task_str: str):
  start = task_str.find("'m': [[[")
  if start < 0:
    return None, None
  start = task_str.find("[[[", start)
  end = task_str.find("]]]", start) + 3
  matrix = json.loads(task_str[start:end].replace("'", "\""))[0]
  opt = int(task_str.split("outputs=[")[1].split("]")[0])
  return matrix, opt


def _nn_cost(matrix: List[List[int]]) -> int:
  n = len(matrix)
  unvisited = set(range(1, n))
  cur = 0
  cost = 0
  while unvisited:
    nxt = min(unvisited, key=lambda j: matrix[cur][j])
    cost += matrix[cur][nxt]
    unvisited.remove(nxt)
    cur = nxt
  return cost + matrix[cur][0]


def _tour_cost(matrix: List[List[int]], tour: List[int]) -> int:
  return sum(matrix[tour[i]][tour[i + 1]] for i in range(len(tour) - 1))


def _nearest_neighbor_tour(matrix: List[List[int]]) -> List[int]:
  n = len(matrix)
  unvisited = set(range(1, n))
  cur = 0
  tour = [0]
  while unvisited:
    nxt = min(unvisited, key=lambda j: matrix[cur][j])
    unvisited.remove(nxt)
    tour.append(nxt)
    cur = nxt
  tour.append(0)
  return tour


def _optimal_tour_bruteforce(matrix: List[List[int]]) -> List[int]:
  n = len(matrix)
  if n <= 1:
    return [0, 0]
  best_tour = None
  best_cost = None
  for perm in permutations(range(1, n)):
    tour = [0] + list(perm) + [0]
    c = _tour_cost(matrix, tour)
    if best_cost is None or c < best_cost:
      best_cost = c
      best_tour = tour
  return best_tour if best_tour is not None else [0, 0]


def _gap_distribution(results_json: str) -> List[float]:
  rows = _load_json(results_json).get("results", [])
  gaps = []
  for row in rows:
    matrix, opt = _parse_matrix_and_opt(row.get("task", ""))
    if matrix is None or opt <= 0:
      continue
    nn = _nn_cost(matrix)
    gaps.append((nn - opt) * 100.0 / opt)
  return gaps


def _gap_examples(results_json: str) -> Dict[str, Dict]:
  rows = _load_json(results_json).get("results", [])
  items = []
  for row in rows:
    matrix, opt = _parse_matrix_and_opt(row.get("task", ""))
    if matrix is None or opt <= 0:
      continue
    nn = _nn_cost(matrix)
    gap = (nn - opt) * 100.0 / opt
    nn_tour = _nearest_neighbor_tour(matrix)
    opt_tour = _optimal_tour_bruteforce(matrix)
    items.append({
        "gap": gap,
        "matrix": matrix,
        "opt": opt,
        "nn": nn,
        "nn_tour": nn_tour,
        "opt_tour": opt_tour,
    })
  if not items:
    return {"min": {}, "median": {}, "max": {}}
  items = sorted(items, key=lambda x: x["gap"])
  min_item = items[0]
  med_item = items[len(items) // 2]
  max_item = items[-1]

  return {
      "min": min_item,
      "median": med_item,
      "max": max_item,
  }


def _draw_tour_edges(ax, pts, matrix: List[List[int]], tour: List[int], color: str, ls: str, lw: float):
  for i in range(len(tour) - 1):
    a = tour[i]
    b = tour[i + 1]
    x0, y0 = pts[a]
    x1, y1 = pts[b]
    ax.plot([x0, x1], [y0, y1], color=color, linestyle=ls, linewidth=lw, alpha=0.95, zorder=3)
    mx = (x0 + x1) / 2.0
    my = (y0 + y1) / 2.0
    ax.text(mx, my, str(matrix[a][b]), fontsize=6, color=color, ha="center", va="center", zorder=4)


def _draw_instance_graph(ax, matrix: List[List[int]], title: str, opt_tour: List[int], nn_tour: List[int], opt_cost: int, nn_cost: int):
  n = len(matrix)
  if n == 0:
    ax.set_axis_off()
    return
  angles = [2.0 * math.pi * i / n for i in range(n)]
  pts = [(math.cos(a), math.sin(a)) for a in angles]
  vals = []
  for i in range(n):
    for j in range(i + 1, n):
      vals.append(0.5 * (matrix[i][j] + matrix[j][i]))
  vmin = min(vals) if vals else 0.0
  vmax = max(vals) if vals else 1.0
  denom = max(vmax - vmin, 1e-8)
  for i in range(n):
    for j in range(i + 1, n):
      w = 0.5 * (matrix[i][j] + matrix[j][i])
      t = (w - vmin) / denom
      alpha = 0.12 + 0.68 * t
      lw = 0.4 + 1.4 * t
      ax.plot(
          [pts[i][0], pts[j][0]],
          [pts[i][1], pts[j][1]],
          color="#1f77b4",
          alpha=alpha,
          linewidth=lw,
      )
  # Overlay tours:
  # - Optimal tour in green solid.
  # - Found NN tour in red dashed.
  _draw_tour_edges(ax, pts, matrix, opt_tour, color="#2ca02c", ls="-", lw=2.2)
  _draw_tour_edges(ax, pts, matrix, nn_tour, color="#d62728", ls="--", lw=2.0)

  for idx, (x, y) in enumerate(pts):
    ax.scatter([x], [y], s=28, color="black")
    ax.text(x, y, str(idx), fontsize=7, color="white", ha="center", va="center")
  ax.set_title(f"{title}\nopt={opt_cost}, found={nn_cost}", fontsize=7)
  ax.set_xticks([])
  ax.set_yticks([])
  ax.set_xlim(-1.18, 1.18)
  ax.set_ylim(-1.18, 1.18)
  ax.set_aspect("equal", adjustable="box")
  for sp in ax.spines.values():
    sp.set_visible(False)


def _q(xs: List[float], p: float) -> float:
  ys = sorted(xs)
  if not ys:
    return float("nan")
  if len(ys) == 1:
    return ys[0]
  k = (len(ys) - 1) * p
  lo = math.floor(k)
  hi = math.ceil(k)
  if lo == hi:
    return ys[lo]
  return ys[lo] * (hi - k) + ys[hi] * (k - lo)


def _extract_series(summary_path: str) -> Tuple[List[Dict], List[float], List[float], List[str], List[int], List[List[float]]]:
  data = _load_json(summary_path)
  data = sorted(data, key=lambda r: r["difficulty_index"])
  x = [float(r["difficulty_index"]) for r in data]
  ab = [100.0 * r["ab_exact_rate"] for r in data]
  nn10 = [100.0 * r["nn_within_10pct_rate"] for r in data]
  labels = [r["name"] for r in data]
  nvals = [r["n"] for r in data]
  gap_dists = [_gap_distribution(r["results_json"]) for r in data]
  return data, x, ab, nn10, labels, nvals, gap_dists


def _count_model_parameters() -> int:
  cfg = get_config()
  cfg.domain = "tsp"
  cfg.model_type = "deepcoder"
  cfg.io_encoder = "lambda_signature"
  cfg.value_encoder = "lambda_signature"
  cfg.encode_weight = True
  cfg.use_op_specific_lstm = True
  cfg.arg_selector = "lstm"
  cfg.step_score_func = "mlp"
  cfg.score_normed = True
  cfg.embed_dim = 128
  model = init_model(cfg, domains.get_domain(cfg.domain), cfg.model_type)
  return int(sum(p.numel() for p in model.parameters()))


def _footer_config_text(labels: List[str], level_cfg: Dict[str, Dict[str, int]], model_params: int) -> str:
  rows = []
  for lb in labels:
    cfg = level_cfg.get(lb, {})
    num_train = cfg.get("num_train", "n/a")
    train_steps = cfg.get("train_steps", "n/a")
    rows.append(f"{lb}: train={num_train}, steps={train_steps}")
  head = "Config: " + " | ".join(rows)
  tail = f"Temps train: proxy via train_steps (temps mur exact non loggue ici) | Parametres modele: {model_params:,}"
  return head + "\n" + tail


def _plot_main_curve(summary_path: str, out_png: str, title: str, level_cfg: Dict[str, Dict[str, int]], model_params: int):
  _, x, ab, nn10, labels, nvals, _ = _extract_series(summary_path)
  plt.figure(figsize=(9, 5))
  plt.plot(x, ab, marker="o", linewidth=2, label="AB exact (%)")
  plt.plot(x, nn10, marker="s", linewidth=2, label="NN within 10% (%)")
  for xi, ai, ni in zip(x, ab, nvals):
    plt.text(xi, ai + 1.0, f"n={ni}", fontsize=8, ha="center")
  plt.xticks(x, labels, rotation=20, ha="right")
  plt.ylabel("Pourcentage (%)")
  plt.xlabel("Niveau de difficulte")
  plt.title(title)
  plt.grid(alpha=0.25)
  plt.legend()
  footer = _footer_config_text(labels, level_cfg, model_params)
  plt.figtext(0.01, -0.12, footer, ha="left", va="top", fontsize=8)
  plt.tight_layout(rect=[0, 0.08, 1, 1])
  plt.savefig(out_png, dpi=180)
  plt.close()


def _plot_gap_boxplot(summary_path: str, out_png: str, title: str, level_cfg: Dict[str, Dict[str, int]], model_params: int):
  _, x, _, _, labels, _, dists = _extract_series(summary_path)
  plt.figure(figsize=(10, 5))
  plt.boxplot(dists, tick_labels=labels, showfliers=False)
  medians = [_q(d, 0.5) for d in dists]
  p90 = [_q(d, 0.9) for d in dists]
  plt.plot(range(1, len(labels) + 1), medians, marker="o", linewidth=1.5, label="Mediane gap NN (%)")
  plt.plot(range(1, len(labels) + 1), p90, marker="^", linewidth=1.5, label="P90 gap NN (%)")
  plt.ylabel("Gap NN vs optimal (%)")
  plt.xlabel("Niveau de difficulte")
  plt.title(title)
  plt.grid(axis="y", alpha=0.25)
  plt.legend()
  plt.xticks(rotation=20, ha="right")
  footer = _footer_config_text(labels, level_cfg, model_params)
  plt.figtext(0.01, -0.14, footer, ha="left", va="top", fontsize=8)
  plt.tight_layout(rect=[0, 0.08, 1, 1])
  plt.savefig(out_png, dpi=180)
  plt.close()


def _plot_gap_violin(summary_path: str, out_png: str, title: str, level_cfg: Dict[str, Dict[str, int]], model_params: int):
  _, _, _, _, labels, _, dists = _extract_series(summary_path)
  plt.figure(figsize=(10, 5))
  parts = plt.violinplot(dists, showmeans=True, showmedians=True, showextrema=False)
  for body in parts["bodies"]:
    body.set_alpha(0.35)
  plt.xticks(range(1, len(labels) + 1), labels, rotation=20, ha="right")
  plt.ylabel("Gap NN vs optimal (%)")
  plt.xlabel("Niveau de difficulte")
  plt.title(title)
  plt.grid(axis="y", alpha=0.25)
  footer = _footer_config_text(labels, level_cfg, model_params)
  plt.figtext(0.01, -0.14, footer, ha="left", va="top", fontsize=8)
  plt.tight_layout(rect=[0, 0.08, 1, 1])
  plt.savefig(out_png, dpi=180)
  plt.close()


def _plot_gap_hist_panels(summary_path: str, out_png: str, title: str, level_cfg: Dict[str, Dict[str, int]], model_params: int):
  data, _, _, _, labels, _, dists = _extract_series(summary_path)
  n = len(labels)
  cols = min(3, n)
  rows = math.ceil(n / cols)
  fig = plt.figure(figsize=(8.6 * cols, 7.8 * rows))
  outer = fig.add_gridspec(rows, cols, hspace=0.55, wspace=0.28)

  bins = [0, 5, 10, 20, 30, 40, 60, 100]
  for i, (label, dist) in enumerate(zip(labels, dists)):
    r = i // cols
    c = i % cols
    inner = outer[r, c].subgridspec(2, 3, height_ratios=[2.6, 1.7], hspace=0.32, wspace=0.20)

    ax_hist = fig.add_subplot(inner[0, :])
    ax_hist.hist(dist, bins=bins, alpha=0.8, color="#4c78a8", edgecolor="white")
    ax_hist.set_title(label, fontsize=10)
    ax_hist.grid(alpha=0.2)
    ax_hist.set_xlim(0, 100)
    ax_hist.set_xlabel("Gap NN vs optimal (%)", fontsize=8)
    ax_hist.set_ylabel("Count", fontsize=8)
    ax_hist.tick_params(labelsize=8)

    ex = _gap_examples(data[i]["results_json"])
    examples = [("min", ex.get("min", {})), ("median", ex.get("median", {})), ("max", ex.get("max", {}))]
    for j, (k, item) in enumerate(examples):
      iax = fig.add_subplot(inner[1, j])
      if not item:
        iax.axis("off")
        continue
      sub_title = f"{k}: gap={item['gap']:.1f}%"
      _draw_instance_graph(
          iax,
          item["matrix"],
          sub_title,
          item.get("opt_tour", []),
          item.get("nn_tour", []),
          int(item.get("opt", 0)),
          int(item.get("nn", 0)),
      )

  # Hide empty panel slots if grid is not full.
  for k in range(n, rows * cols):
    r = k // cols
    c = k % cols
    ax_empty = fig.add_subplot(outer[r, c])
    ax_empty.axis("off")

  fig.suptitle(title)
  footer = _footer_config_text(labels, level_cfg, model_params)
  fig.text(0.01, 0.01, footer, ha="left", va="bottom", fontsize=8)
  fig.subplots_adjust(left=0.05, right=0.99, top=0.93, bottom=0.12, hspace=0.58, wspace=0.28)
  plt.savefig(out_png, dpi=180)
  plt.close()


def main():
  base_summary = os.path.join(ROOT, "difficulty_curve_summary.json")
  big_summary = os.path.join(ROOT, "difficulty_curve_summary_bigtrain.json")
  model_params = _count_model_parameters()

  if os.path.exists(base_summary):
    _plot_main_curve(
        base_summary,
        os.path.join(ROOT, "plot_base_success_curves.png"),
        "TSP difficulty sweep: AB vs NN (base train size)",
        BASE_LEVEL_CONFIG,
        model_params,
    )
    _plot_gap_boxplot(
        base_summary,
        os.path.join(ROOT, "plot_base_nn_gap_distribution.png"),
        "Distribution des gaps NN (base train size)",
        BASE_LEVEL_CONFIG,
        model_params,
    )
    _plot_gap_violin(
        base_summary,
        os.path.join(ROOT, "plot_base_nn_gap_violin.png"),
        "Violin plot des gaps NN (base train size)",
        BASE_LEVEL_CONFIG,
        model_params,
    )
    _plot_gap_hist_panels(
        base_summary,
        os.path.join(ROOT, "plot_base_nn_gap_histograms.png"),
        "Histogrammes des gaps NN par niveau (base train size)",
        BASE_LEVEL_CONFIG,
        model_params,
    )

  if os.path.exists(big_summary):
    _plot_main_curve(
        big_summary,
        os.path.join(ROOT, "plot_bigtrain_success_curves.png"),
        "TSP difficulty sweep: AB vs NN (big train hard levels)",
        BIGTRAIN_LEVEL_CONFIG,
        model_params,
    )
    _plot_gap_boxplot(
        big_summary,
        os.path.join(ROOT, "plot_bigtrain_nn_gap_distribution.png"),
        "Distribution des gaps NN (big train hard levels)",
        BIGTRAIN_LEVEL_CONFIG,
        model_params,
    )
    _plot_gap_violin(
        big_summary,
        os.path.join(ROOT, "plot_bigtrain_nn_gap_violin.png"),
        "Violin plot des gaps NN (big train hard levels)",
        BIGTRAIN_LEVEL_CONFIG,
        model_params,
    )
    _plot_gap_hist_panels(
        big_summary,
        os.path.join(ROOT, "plot_bigtrain_nn_gap_histograms.png"),
        "Histogrammes des gaps NN par niveau (big train hard levels)",
        BIGTRAIN_LEVEL_CONFIG,
        model_params,
    )

  print("Plots generated in:", ROOT)


if __name__ == "__main__":
  main()
