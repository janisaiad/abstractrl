#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from typing import List


def nearest_neighbor_tour_cost(mat: List[List[int]]) -> int:
  n = len(mat)
  unvisited = set(range(1, n))
  cur = 0
  cost = 0
  while unvisited:
    nxt = min(unvisited, key=lambda j: mat[cur][j])
    cost += mat[cur][nxt]
    unvisited.remove(nxt)
    cur = nxt
  cost += mat[cur][0]
  return cost


def main():
  root = os.path.dirname(os.path.abspath(__file__))
  path = os.path.join(root, "level3_n5", "results.json")
  with open(path, "r") as f:
    data = json.load(f)

  results = data["results"]
  total = len(results)
  ab_solved = sum(1 for r in results if r.get("success"))

  nn_costs = []
  opt_costs = []
  nn_within_10 = 0
  nn_beaten_by_ab = 0

  for r in results:
    task_str = r["task"]
    # task string contains "... 'm': [[[...]]], ... outputs=[OPT]"
    # robust parse by extracting JSON-like matrix region manually
    m_start = task_str.find("'m': [[[")
    if m_start < 0:
      continue
    m_start = task_str.find("[[[", m_start)
    m_end = task_str.find("]]]", m_start) + 3
    matrix = json.loads(task_str[m_start:m_end].replace("'", "\""))[0]
    opt = int(r["task"].split("outputs=[")[1].split("]")[0])
    nn = nearest_neighbor_tour_cost(matrix)
    nn_costs.append(nn)
    opt_costs.append(opt)
    if nn <= 1.1 * opt:
      nn_within_10 += 1
    if r.get("success"):
      # AB exact optimum beats NN whenever NN is strictly suboptimal
      if nn > opt:
        nn_beaten_by_ab += 1

  mean_gap = 100.0 * sum((nn - o) / o for nn, o in zip(nn_costs, opt_costs)) / len(opt_costs)
  summary = {
      "instances": total,
      "abstractbeam_exact_solved": ab_solved,
      "abstractbeam_exact_rate": ab_solved / total if total else 0.0,
      "nn_within_10pct_count": nn_within_10,
      "nn_within_10pct_rate": nn_within_10 / len(opt_costs) if opt_costs else 0.0,
      "nn_mean_gap_pct": mean_gap,
      "ab_exact_beats_nn_count": nn_beaten_by_ab,
      "zak_rule_bad_if_gap_gt_10pct": "true",
  }
  out_path = os.path.join(root, "nn_vs_abstractbeam_level3_n5.json")
  with open(out_path, "w") as f:
    json.dump(summary, f, indent=2)
  print(json.dumps(summary, indent=2))
  print("wrote", out_path)


if __name__ == "__main__":
  main()
