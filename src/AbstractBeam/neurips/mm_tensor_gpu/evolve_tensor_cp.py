"""Recherche évolutionnaire simple sur les facteurs CP (faible dimension 4×r)."""

from __future__ import annotations

import json
import math
import os
import sys
from typing import Any, Dict

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
  sys.path.insert(0, _SCRIPT_DIR)

from tensor_cp_baseline import matrix_multiplication_tensor_222, reconstruct


def _loss(U: np.ndarray, V: np.ndarray, W: np.ndarray, T: np.ndarray) -> float:
  r = reconstruct(U, V, W) - T
  return float(np.linalg.norm(r) / (np.linalg.norm(T) + 1e-12))


def evolve_cp(T: np.ndarray, rank: int, generations: int = 400, pop: int = 48, seed: int = 1) -> Dict[str, Any]:
  rng = np.random.default_rng(seed)
  d = 4

  def rand_indiv():
    return (
        rng.standard_normal((d, rank)),
        rng.standard_normal((d, rank)),
        rng.standard_normal((d, rank)),
    )

  population = [rand_indiv() for _ in range(pop)]
  best_score = math.inf
  best = population[0]
  sigma = 0.35
  for _gen in range(generations):
    scored = []
    for U, V, W in population:
      s = _loss(U, V, W, T)
      scored.append((s, (U, V, W)))
      if s < best_score:
        best_score = s
        best = (U.copy(), V.copy(), W.copy())
    scored.sort(key=lambda x: x[0])
    elites = [x[1] for x in scored[: pop // 4]]
    new_pop = list(elites)
    while len(new_pop) < pop:
      a, b = elites[rng.integers(len(elites))], elites[rng.integers(len(elites))]
      child = tuple((0.5 * (x + y) + rng.normal(0, sigma, x.shape) for x, y in zip(a, b)))
      new_pop.append(child)
    population = new_pop
    sigma *= 0.997
  return {"rank": rank, "best_relative_fro_error": best_score, "generations": generations}


def run_evolution(out_path: str) -> Dict[str, Any]:
  T = matrix_multiplication_tensor_222()
  out = {"rank7_evolution": evolve_cp(T, 7, generations=500, pop=64, seed=42)}
  os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
  with open(out_path, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2)
  return out


if __name__ == "__main__":
  p = os.path.join(os.path.dirname(__file__), "tensor_evolve_report.json")
  run_evolution(p)
  print("wrote", p)
