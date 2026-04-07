#!/usr/bin/env python3
"""Runs GPU rapides : tri sur listes plus longues + listes presque triées (peu de swaps)."""

from __future__ import annotations

import json
import os
import pickle
import random
import sys

import torch.multiprocessing as mp

_ABSTRACTBEAM_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ABSTRACTBEAM_ROOT not in sys.path:
  sys.path.insert(0, _ABSTRACTBEAM_ROOT)

from crossbeam.common.configs_all import get_config
from crossbeam.data.deepcoder import sort_list_generators as slg
from crossbeam.datasets import data_gen
from crossbeam.dsl import deepcoder_utils
from crossbeam.dsl import domains
from crossbeam.experiment.exp_common import set_global_seed
from crossbeam.experiment.run_crossbeam import init_model
from crossbeam.experiment.train_eval import main_train_eval

ROOT = os.path.dirname(os.path.abspath(__file__))


def _verify_one_task():
  rng = random.Random(0)
  tasks = slg.make_sort_tasks(
      rng, list_len=6, num_tasks=1, num_examples=3, mode="nearly_sorted_adjacent", num_swaps=2
  )
  t = tasks[0]
  got = deepcoder_utils.run_program(t.solution, t.inputs_dict)
  assert got == t.outputs, (got, t.outputs)


def _run_suite(
    suite_name: str,
    list_len: int,
    mode: slg.Mode,
    num_swaps: int,
    seed: int,
    train_n: int,
    eval_n: int,
    steps: int,
    port: int,
):
  assert list_len <= 20, "deepcoder_small_value_filter limite len(liste) à 20"
  rng = random.Random(seed)
  set_global_seed(seed)
  suite_dir = os.path.join(ROOT, suite_name)
  data_dir = os.path.join(suite_dir, "data")
  os.makedirs(data_dir, exist_ok=True)
  for fn in os.listdir(data_dir):
    if fn.startswith("train-") and fn.endswith(".pkl"):
      os.remove(os.path.join(data_dir, fn))

  train_tasks = slg.make_sort_tasks(
      rng,
      list_len=list_len,
      num_tasks=train_n,
      num_examples=5,
      mode=mode,
      num_swaps=num_swaps,
      name_prefix=f"{suite_name}_train",
  )
  eval_rng = random.Random(seed + 99991)
  eval_tasks = slg.make_sort_tasks(
      eval_rng,
      list_len=list_len,
      num_tasks=eval_n,
      num_examples=5,
      mode=mode,
      num_swaps=num_swaps,
      name_prefix=f"{suite_name}_eval",
  )

  with open(os.path.join(data_dir, "train-weight-3-00000.pkl"), "wb") as f:
    pickle.dump(train_tasks, f)

  cfg = get_config()
  cfg.domain = "deepcoder"
  cfg.model_type = "deepcoder"
  cfg.io_encoder = "lambda_signature"
  cfg.value_encoder = "lambda_signature"
  cfg.encode_weight = True
  cfg.use_op_specific_lstm = True
  cfg.arg_selector = "lstm"
  cfg.step_score_func = "mlp"
  cfg.score_normed = True
  cfg.embed_dim = 128
  cfg.save_dir = suite_dir
  cfg.data_save_dir = data_dir
  cfg.json_results_file = os.path.join(suite_dir, "results.json")
  cfg.do_test = False
  cfg.dynamic_tasks = False
  cfg.abstraction = False
  cfg.train_steps = steps
  cfg.eval_every = max(10, steps // 2)
  cfg.grad_accumulate = 4
  cfg.num_proc = 1
  cfg.gpu_list = "0"
  cfg.gpu = 0
  cfg.use_ur = False
  cfg.use_ur_in_valid = False
  cfg.timeout = 2.0
  cfg.max_search_weight = 12
  cfg.beam_size = 10
  cfg.used_invs = None
  cfg.schedule_type = "uniform"
  cfg.steps_per_curr_stage = 5000
  cfg.random_beam = False
  cfg.stochastic_beam = False
  cfg.static_weight = False
  cfg.restarts_timeout = None
  cfg.temperature = 1.0
  cfg.type_masking = True
  cfg.grad_clip = 5.0
  cfg.lr = 5e-4
  cfg.port = str(port)

  domain = domains.get_domain(cfg.domain)
  model = init_model(cfg, domain, cfg.model_type)
  main_train_eval(cfg, model, trace_gen=data_gen.trace_gen, checkpoint=None, original_tasks=eval_tasks)

  with open(cfg.json_results_file, "r") as f:
    res = json.load(f)
  solved = res.get("num_tasks_solved", 0)
  total = res.get("num_tasks", len(eval_tasks))
  return {
      "suite": suite_name,
      "list_len": list_len,
      "mode": mode,
      "num_swaps": num_swaps,
      "solved": solved,
      "total": total,
      "rate": solved / total if total else 0.0,
  }


def main():
  mp.set_start_method("spawn", force=True)
  _verify_one_task()
  out = []
  port = 31601
  # Listes longues (aléatoires)
  out.append(
      _run_suite(
          "sort_L12_random",
          list_len=12,
          mode="random",
          num_swaps=0,
          seed=12001,
          train_n=180,
          eval_n=50,
          steps=60,
          port=port,
      )
  )
  port += 1
  out.append(
      _run_suite(
          "sort_L18_random",
          list_len=18,
          mode="random",
          num_swaps=0,
          seed=12002,
          train_n=200,
          eval_n=50,
          steps=60,
          port=port,
      )
  )
  port += 1
  # Quasi triées : peu d’échanges adjacents
  out.append(
      _run_suite(
          "sort_L14_nearly_adj4",
          list_len=14,
          mode="nearly_sorted_adjacent",
          num_swaps=4,
          seed=12003,
          train_n=200,
          eval_n=50,
          steps=60,
          port=port,
      )
  )
  port += 1
  # Quasi triées : quelques swaps arbitraires
  out.append(
      _run_suite(
          "sort_L16_nearly_swap6",
          list_len=16,
          mode="nearly_sorted_swap",
          num_swaps=6,
          seed=12004,
          train_n=200,
          eval_n=50,
          steps=60,
          port=port,
      )
  )

  summary_path = os.path.join(ROOT, "variants_summary.json")
  with open(summary_path, "w") as f:
    json.dump(out, f, indent=2)
  print(json.dumps(out, indent=2))
  print("wrote", summary_path)


if __name__ == "__main__":
  main()
