"""Tâches ATSP à 3 villes : longueur minimale sur les deux tours hamiltoniens depuis la ville 0.

La matrice m est une liste de listes d'entiers (distances), diagonale nulle.
Formule : min( m[0][1]+m[1][2]+m[2][0], m[0][2]+m[2][1]+m[1][0] ).
"""

from __future__ import annotations

import os
import pickle

from crossbeam.dsl import task as task_module

Task = task_module.Task

# Solution DSL (même primitives que DeepCoder / LambdaBeam).
_TSP3_SOL = (
    "Min(Add(Add(Access(1, Access(0, m)), Access(2, Access(1, m))), Access(0, Access(2, m))), "
    "Add(Add(Access(2, Access(0, m)), Access(1, Access(2, m))), Access(0, Access(1, m))))"
)


def _tour3_cost(m):
  a = m[0][1] + m[1][2] + m[2][0]
  b = m[0][2] + m[2][1] + m[1][0]
  return min(a, b)


_MATRICES = [
    [[0, 2, 9], [3, 0, 1], [4, 5, 0]],
    [[0, 10, 15], [5, 0, 12], [8, 9, 0]],
    [[0, 1, 100], [100, 0, 1], [1, 100, 0]],
    [[0, 4, 4], [2, 0, 6], [3, 1, 0]],
    [[0, 20, 18], [22, 0, 15], [19, 17, 0]],
    [[0, 3, 8], [7, 0, 2], [5, 6, 0]],
    [[0, 11, 14], [16, 0, 9], [13, 10, 0]],
    [[0, 50, 45], [48, 0, 40], [42, 44, 0]],
]

TSP_TASKS = [
    Task(
        name=f"tsp:atsp3_{i}",
        inputs_dict={"m": [mat]},
        outputs=[_tour3_cost(mat)],
        solution=_TSP3_SOL,
    )
    for i, mat in enumerate(_MATRICES)
]


def write_training_pickles(
    data_dir: str,
    weight: int = 3,
    repeats: int = 400,
    shard_index: int = 0,
) -> str:
  """Écrit train-weight-{weight}-{shard:05d}.pkl pour démarrer l'entraînement sans data_gen.

  Note : get_local_weighted_files ne prend que le premier chiffre du poids dans le nom
  (bug amont) ; utiliser un poids à un chiffre (3–9) pour un bucket cohérent.
  """
  os.makedirs(data_dir, exist_ok=True)
  path = os.path.join(data_dir, f"train-weight-{weight}-{shard_index:05d}.pkl")
  payload = (list(TSP_TASKS) * ((repeats + len(TSP_TASKS) - 1) // len(TSP_TASKS)))[:repeats]
  with open(path, "wb") as f:
    pickle.dump(payload, f, pickle.HIGHEST_PROTOCOL)
  return path


if __name__ == "__main__":
  from crossbeam.dsl import deepcoder_utils

  for t in TSP_TASKS:
    got = deepcoder_utils.run_program(t.solution, t.inputs_dict)
    assert got == t.outputs, (t.name, got, t.outputs)
  print("tsp_tasks: OK", len(TSP_TASKS), "tâches")
  p = write_training_pickles("./neurips/tsp/data")
  print("écrit", p)
