"""Instances ATSP structurées : clusters (hiérarchique) et distances XOR pondérées (auto-similaire).

Utilisé pour entraîner / évaluer le même pipeline que `tsp_tasks`, avec des matrices
plus structurées que du bruit i.i.d.
"""

from __future__ import annotations

import itertools
import random
from typing import List, Sequence, Tuple

from crossbeam.dsl import deepcoder_operations as ops
from crossbeam.dsl import task as task_module
from crossbeam.dsl import value as value_module

Add = ops.Add()
MinOp = ops.Min()
Access = ops.Access()


def tour_cost_atsp_from_0(mat: List[List[int]]) -> int:
  """Coût minimal d'un tour hamiltonien fixant la ville 0 comme départ et retour."""
  n = len(mat)
  best = None
  for perm in itertools.permutations(range(1, n)):
    route = (0,) + perm + (0,)
    c = sum(mat[route[k]][route[k + 1]] for k in range(len(route) - 1))
    best = c if best is None else min(best, c)
  return best  # type: ignore[return-value]


def hierarchical_clustered_matrix(
    n: int,
    rng: random.Random,
    num_clusters: int | None = None,
    intra_lo: int = 1,
    intra_hi: int = 8,
    inter_lo: int = 40,
    inter_hi: int = 120,
) -> List[List[int]]:
  """ATSP : petites distances intra-cluster, grandes inter-clusters (structure à deux niveaux)."""
  if n < 2:
    raise ValueError("n >= 2")
  if num_clusters is None:
    num_clusters = max(2, min(n, 3))
  # Répartition grossière des villes en groupes contigus [0..k), [k..), ...
  sizes = [n // num_clusters] * num_clusters
  for i in range(n % num_clusters):
    sizes[i] += 1
  cluster_of = []
  c = 0
  for sz in sizes:
    cluster_of.extend([c] * sz)
    c += 1
  m = [[0] * n for _ in range(n)]
  for i in range(n):
    for j in range(n):
      if i == j:
        continue
      if cluster_of[i] == cluster_of[j]:
        m[i][j] = rng.randint(intra_lo, intra_hi)
      else:
        m[i][j] = rng.randint(inter_lo, inter_hi)
  return m


def fractal_xor_weighted_matrix(n: int, rng: random.Random) -> List[List[int]]:
  """ATSP : coût lié aux bits de i^j (échelles 2^k), + petite asymétrie aléatoire.

  Idée : deux indices qui diffèrent sur un bit de poids fort ont tendance à être
  « loin » dans une grille binaire ; bruit faible casse la symétrie (ATSP).
  """
  m = [[0] * n for _ in range(n)]
  maxv = n - 1 if n > 1 else 1
  nbits = maxv.bit_length()
  for i in range(n):
    for j in range(n):
      if i == j:
        continue
      x = i ^ j
      base = 0
      for b in range(nbits):
        if (x >> b) & 1:
          base += 1 << b
      asym = rng.randint(0, 3)
      m[i][j] = max(1, base + asym + (i - j) % 2)
  return m


def _edge_expr(m_var: value_module.InputVariable, i: int, j: int):
  return Access.apply(
      [value_module.ConstantValue(j), Access.apply([value_module.ConstantValue(i), m_var])])


def _sum_expr(exprs: Sequence):
  cur = exprs[0]
  for e in exprs[1:]:
    cur = Add.apply([cur, e])
  return cur


def _min_expr(exprs: Sequence):
  cur = exprs[0]
  for e in exprs[1:]:
    cur = MinOp.apply([cur, e])
  return cur


def tsp_bruteforce_solution_expr(m_var: value_module.InputVariable, n: int):
  """Expression DSL = min sur tous les tours depuis 0 (exact pour n petit)."""
  cities = list(range(1, n))
  costs = []
  for perm in itertools.permutations(cities):
    route = (0,) + perm + (0,)
    edges = [_edge_expr(m_var, route[k], route[k + 1]) for k in range(len(route) - 1)]
    costs.append(_sum_expr(edges))
  return _min_expr(costs)


def make_atsp_tasks(
    n: int,
    num_train: int,
    num_eval: int,
    matrix_factory,
    rng: random.Random,
    train_prefix: str = "tsp_struct_train",
    eval_prefix: str = "tsp_struct_eval",
) -> Tuple[List[task_module.Task], List[task_module.Task]]:
  """Construit des Task avec solution exacte par énumération."""

  def build_one(i: int, prefix: str) -> task_module.Task:
    mat = matrix_factory(rng)
    out = tour_cost_atsp_from_0(mat)
    m_var = value_module.InputVariable([mat], name="m")
    sol = tsp_bruteforce_solution_expr(m_var, n)
    return task_module.Task(
        name=f"{prefix}:n{n}:{i}",
        inputs_dict={"m": [mat]},
        outputs=[out],
        solution=sol,
    )

  train = [build_one(i, train_prefix) for i in range(num_train)]
  eval_tasks = [build_one(i, eval_prefix) for i in range(num_eval)]
  return train, eval_tasks
