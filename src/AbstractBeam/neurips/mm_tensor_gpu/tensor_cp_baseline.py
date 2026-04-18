"""Baseline « type AlphaEvolve » : décomposition CP du tenseur ⟨2,2,2⟩ (multiplication 2×2).

Le rang tensoriel (ℝ) optimal est 7 (Strassen). On mesure l’erreur de reconstruction pour r donné.
Référence : https://arxiv.org/abs/2406.xxx AlphaEvolve (Google DeepMind) — ici on n’a pas leur stack,
seulement ALS + recherches aléatoires pour situer Abstract Beam sur l’axe « rang / erreur ».
"""

from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, Tuple

import numpy as np


def matrix_multiplication_tensor_222() -> np.ndarray:
  """T[c,a,b] = 1 ssi (avec A,B 2×2 aplatis lignes) C = A B."""
  t = np.zeros((4, 4, 4), dtype=np.float64)
  for i in range(2):
    for j in range(2):
      for k in range(2):
        c = 2 * i + j
        a = 2 * i + k
        b = 2 * k + j
        t[c, a, b] = 1.0
  return t


def _outer3(u: np.ndarray, v: np.ndarray, w: np.ndarray) -> np.ndarray:
  return (u.reshape(-1, 1, 1) * v.reshape(1, -1, 1) * w.reshape(1, 1, -1))


def reconstruct(U: np.ndarray, V: np.ndarray, W: np.ndarray) -> np.ndarray:
  r = U.shape[1]
  acc = np.zeros((4, 4, 4))
  for ri in range(r):
    acc += _outer3(U[:, ri], V[:, ri], W[:, ri])
  return acc


def cp_als(T: np.ndarray, rank: int, n_iter: int = 80, seed: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
  """ALS pour T[c,a,b] = sum_r U[c,r] V[a,r] W[b,r] (indices modes 0,1,2)."""
  rng = np.random.default_rng(seed)
  U = rng.standard_normal((4, rank))
  V = rng.standard_normal((4, rank))
  W = rng.standard_normal((4, rank))
  best = math.inf
  best_triple = (U.copy(), V.copy(), W.copy())
  for _it in range(n_iter):
    # mode c: X0[c, a+4*b] = T[c,a,b] ; Z0[a+4*b,r] = V[a,r]*W[b,r]
    X0 = np.zeros((4, 16))
    Z0 = np.zeros((16, rank))
    for c in range(4):
      for a in range(4):
        for b in range(4):
          X0[c, a + 4 * b] = T[c, a, b]
    for a in range(4):
      for b in range(4):
        Z0[a + 4 * b, :] = V[a, :] * W[b, :]
    U = X0 @ np.linalg.pinv(Z0.T)
    # mode a: X1[a, c+4*b] = T[c,a,b] ; Z1[c+4*b,r] = U[c,r]*W[b,r]
    X1 = np.zeros((4, 16))
    Z1 = np.zeros((16, rank))
    for c in range(4):
      for a in range(4):
        for b in range(4):
          X1[a, c + 4 * b] = T[c, a, b]
    for c in range(4):
      for b in range(4):
        Z1[c + 4 * b, :] = U[c, :] * W[b, :]
    V = X1 @ np.linalg.pinv(Z1.T)
    # mode b: X2[b, c+4*a] = T[c,a,b] ; Z2[c+4*a,r] = U[c,r]*V[a,r]
    X2 = np.zeros((4, 16))
    Z2 = np.zeros((16, rank))
    for c in range(4):
      for a in range(4):
        for b in range(4):
          X2[b, c + 4 * a] = T[c, a, b]
    for c in range(4):
      for a in range(4):
        Z2[c + 4 * a, :] = U[c, :] * V[a, :]
    W = X2 @ np.linalg.pinv(Z2.T)
    err = np.linalg.norm(T - reconstruct(U, V, W)) / (np.linalg.norm(T) + 1e-12)
    if err < best:
      best = err
      best_triple = (U.copy(), V.copy(), W.copy())
  return best_triple[0], best_triple[1], best_triple[2], float(best)


def multi_restart_cp(T: np.ndarray, rank: int, n_restarts: int = 12, als_iter: int = 120) -> Dict[str, Any]:
  best_err = math.inf
  for s in range(n_restarts):
    _, _, _, err = cp_als(T, rank, n_iter=als_iter, seed=s * 997)
    best_err = min(best_err, err)
  return {"rank": rank, "best_relative_fro_error": best_err}


def run_baselines(out_path: str) -> Dict[str, Any]:
  T = matrix_multiplication_tensor_222()
  fnorm = float(np.linalg.norm(T))
  rows = []
  for r in (4, 5, 6, 7, 8):
    rows.append(multi_restart_cp(T, r))
  known_optimal_rank = 7
  report: Dict[str, Any] = {
    "tensor_shape": list(T.shape),
    "tensor_frobenius_norm": fnorm,
    "known_bilinear_rank_2x2x2_real": known_optimal_rank,
    "cp_als_sweeps": rows,
    "notes": "r=7 erreur ~0 attendue en précision machine; r<7 borne inférieure Strassen.",
  }
  os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
  with open(out_path, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)
  return report


if __name__ == "__main__":
  p = os.path.join(os.path.dirname(__file__), "tensor_cp_report.json")
  run_baselines(p)
  print("wrote", p)
