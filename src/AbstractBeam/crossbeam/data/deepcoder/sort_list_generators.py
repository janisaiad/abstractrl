"""Génération de tâches DeepCoder « Sort(x) » : listes longues et listes presque triées."""

from __future__ import annotations

import random
from typing import List, Literal

from crossbeam.dsl import deepcoder_operations as ops
from crossbeam.dsl import task as task_module
from crossbeam.dsl import value as value_module

Task = task_module.Task
Sort = ops.Sort()

Mode = Literal["random", "nearly_sorted_adjacent", "nearly_sorted_swap"]


def _random_int_list(
    rng: random.Random,
    length: int,
    lo: int = -25,
    hi: int = 25,
    unique: bool = True,
) -> List[int]:
  """Liste d'entiers dans [lo, hi], pour rester dans le filtre DeepCoder (petites valeurs)."""
  span = hi - lo + 1
  if unique and span >= length:
    return rng.sample(range(lo, hi + 1), k=length)
  return [rng.randint(lo, hi) for _ in range(length)]


def _nearly_sorted_adjacent(
    rng: random.Random,
    length: int,
    num_swaps: int,
    lo: int = -25,
    hi: int = 25,
    unique: bool = True,
) -> List[int]:
  """Tri croissant puis `num_swaps` échanges d'éléments adjacents (quasi tri bulle)."""
  lst = _random_int_list(rng, length, lo=lo, hi=hi, unique=unique)
  lst.sort()
  for _ in range(num_swaps):
    i = rng.randint(0, length - 2)
    lst[i], lst[i + 1] = lst[i + 1], lst[i]
  return lst


def _nearly_sorted_random_swaps(
    rng: random.Random,
    length: int,
    num_swaps: int,
    lo: int = -25,
    hi: int = 25,
    unique: bool = True,
) -> List[int]:
  """Tri croissant puis `num_swaps` échanges aléatoires de deux positions (peu de désordre)."""
  lst = _random_int_list(rng, length, lo=lo, hi=hi, unique=unique)
  lst.sort()
  for _ in range(num_swaps):
    i = rng.randrange(length)
    j = rng.randrange(length)
    if i != j:
      lst[i], lst[j] = lst[j], lst[i]
  return lst


def sample_list_for_mode(
    rng: random.Random,
    length: int,
    mode: Mode,
    num_swaps: int,
    lo: int = -25,
    hi: int = 25,
    unique: bool = True,
) -> List[int]:
  if mode == "random":
    return _random_int_list(rng, length, lo=lo, hi=hi, unique=unique)
  if mode == "nearly_sorted_adjacent":
    return _nearly_sorted_adjacent(rng, length, num_swaps, lo=lo, hi=hi, unique=unique)
  if mode == "nearly_sorted_swap":
    return _nearly_sorted_random_swaps(rng, length, num_swaps, lo=lo, hi=hi, unique=unique)
  raise ValueError(mode)


def make_sort_tasks(
    rng: random.Random,
    list_len: int,
    num_tasks: int,
    num_examples: int,
    mode: Mode,
    num_swaps: int = 0,
    name_prefix: str = "sortgen",
    lo: int = -25,
    hi: int = 25,
    unique: bool = True,
) -> List[Task]:
  """Une Task = plusieurs exemples (listes) ; solution toujours `Sort(x)`."""
  tasks: List[Task] = []
  for t in range(num_tasks):
    xs = [
        sample_list_for_mode(rng, list_len, mode, num_swaps, lo=lo, hi=hi, unique=unique)
        for _ in range(num_examples)
    ]
    x_var = value_module.InputVariable(xs, name="x")
    tasks.append(
        Task(
            name=f"{name_prefix}:L{list_len}:{mode}:{t}",
            inputs_dict={"x": xs},
            outputs=[sorted(row) for row in xs],
            solution=Sort.apply([x_var]),
        )
    )
  return tasks
