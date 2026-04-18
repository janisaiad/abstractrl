"""Tâches multiplication matricielle n×n : une entrée de C = A B par tâche (défaut n=4).

Abstract Beam synthétise des programmes DSL (Add, Multiply, Access). Pour n=4 il y a 16 cellules.
"""

from __future__ import annotations

import random
from typing import List, Tuple

from crossbeam.dsl import deepcoder_operations as dops
from crossbeam.dsl import task as task_module
from crossbeam.dsl import value as value_module

Task = task_module.Task

_ADD = next(op for op in dops.get_operations_() if op.name == "Add")
_MULT = next(op for op in dops.get_operations_() if op.name == "Multiply")
_ACCESS = next(op for op in dops.get_operations_() if op.name == "Access")


def _mat_el(mat_var: value_module.InputVariable, i: int, j: int):
  return _ACCESS.apply(
      [value_module.ConstantValue(j), _ACCESS.apply([value_module.ConstantValue(i), mat_var])])


def mm_cell_solution_expr(
    a_var: value_module.InputVariable,
    b_var: value_module.InputVariable,
    i: int,
    j: int,
    n: int,
):
  """C[i][j] = sum_{k=0}^{n-1} A[i][k]*B[k][j]."""
  terms = []
  for k in range(n):
    terms.append(_MULT.apply([_mat_el(a_var, i, k), _mat_el(b_var, k, j)]))
  cur = terms[0]
  for t in terms[1:]:
    cur = _ADD.apply([cur, t])
  return cur


def reference_scalar(a: List[List[int]], b: List[List[int]], i: int, j: int, n: int) -> int:
  return sum(a[i][k] * b[k][j] for k in range(n))


def random_matrix(rng: random.Random, n: int, lo: int = -4, hi: int = 4) -> List[List[int]]:
  return [[rng.randint(lo, hi) for _ in range(n)] for _ in range(n)]


def make_mm_tasks(
    rng: random.Random,
    n: int,
    num_examples: int,
    num_tasks_per_cell: int,
    lo: int = -4,
    hi: int = 4,
    name_prefix: str = "mm",
) -> List[Task]:
  """Une tâche par (cellule, tidx) avec `num_examples` paires (A,B)."""
  tasks: List[Task] = []
  for i in range(n):
    for j in range(n):
      for tidx in range(num_tasks_per_cell):
        mats_a: List[List[List[int]]] = []
        mats_b: List[List[List[int]]] = []
        outs: List[int] = []
        for _ in range(num_examples):
          a = random_matrix(rng, n, lo, hi)
          b = random_matrix(rng, n, lo, hi)
          mats_a.append(a)
          mats_b.append(b)
          outs.append(reference_scalar(a, b, i, j, n))
        a_var = value_module.InputVariable(mats_a, name="a")
        b_var = value_module.InputVariable(mats_b, name="b")
        sol = mm_cell_solution_expr(a_var, b_var, i, j, n)
        tasks.append(
            Task(
                name=f"{name_prefix}:n{n}:cell_{i}_{j}:{tidx}",
                inputs_dict={"a": mats_a, "b": mats_b},
                outputs=outs,
                solution=sol,
            )
        )
  return tasks


def make_mm_train_eval_split(
    rng: random.Random,
    n: int,
    num_train_pairs: int,
    num_eval_pairs: int,
    tasks_per_split: int,
    lo: int = -4,
    hi: int = 4,
    train_prefix: str = "mm_train",
    eval_prefix: str = "mm_eval",
) -> Tuple[List[Task], List[Task]]:
  train = make_mm_tasks(
      rng, n, num_examples=num_train_pairs, num_tasks_per_cell=tasks_per_split, lo=lo, hi=hi, name_prefix=train_prefix
  )
  eval_tasks = make_mm_tasks(
      rng, n, num_examples=num_eval_pairs, num_tasks_per_cell=tasks_per_split, lo=lo, hi=hi, name_prefix=eval_prefix
  )
  return train, eval_tasks


def make_mm2_tasks(
    rng: random.Random,
    num_examples: int,
    num_tasks_per_cell: int,
    lo: int = -4,
    hi: int = 4,
    name_prefix: str = "mm2",
) -> List[Task]:
  return make_mm_tasks(rng, 2, num_examples, num_tasks_per_cell, lo, hi, name_prefix)


def make_mm2_train_eval_split(
    rng: random.Random,
    num_train_pairs: int,
    num_eval_pairs: int,
    tasks_per_split: int,
    lo: int = -4,
    hi: int = 4,
    train_prefix: str = "mm2_train",
    eval_prefix: str = "mm2_eval",
) -> Tuple[List[Task], List[Task]]:
  return make_mm_train_eval_split(
      rng, 2, num_train_pairs, num_eval_pairs, tasks_per_split, lo, hi, train_prefix, eval_prefix
)


if __name__ == "__main__":
  from crossbeam.dsl import deepcoder_utils

  rng = random.Random(0)
  for dim in (2, 4):
    lo, hi = (-2, 2) if dim >= 4 else (-4, 4)
    ts = make_mm_tasks(rng, dim, num_examples=3, num_tasks_per_cell=1, lo=lo, hi=hi, name_prefix=f"test{dim}")
    for t in ts:
      got = deepcoder_utils.run_program(t.solution.expression(), t.inputs_dict, dreamcoder_prims=False)
      assert got == t.outputs, (t.name, got, t.outputs)
    print("mm_tasks OK n=", dim, "num_tasks", len(ts), "ref_weight", ts[0].solution.get_weight())
