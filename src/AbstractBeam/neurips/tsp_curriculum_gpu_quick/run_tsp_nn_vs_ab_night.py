#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import pickle
import random
import sys
from typing import List

import torch.multiprocessing as mp

_ABSTRACTBEAM_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ABSTRACTBEAM_ROOT not in sys.path:
  sys.path.insert(0, _ABSTRACTBEAM_ROOT)

from crossbeam.common.configs_all import get_config
from crossbeam.data.tsp import tsp_structured_instances as tsi
from crossbeam.datasets import data_gen
from crossbeam.dsl import domains
from crossbeam.experiment.exp_common import set_global_seed
from crossbeam.experiment.run_crossbeam import init_model
from crossbeam.experiment.train_eval import main_train_eval

ROOT = os.path.dirname(os.path.abspath(__file__))


def _random_atsp5(rng: random.Random) -> List[List[int]]:
  n = 5
  m = [[0] * n for _ in range(n)]
  for i in range(n):
    for j in range(n):
      if i != j:
        m[i][j] = rng.randint(1, 50)
  return m


def _nn_cost(mat: List[List[int]]) -> int:
  n = len(mat)
  cur = 0
  rem = set(range(1, n))
  cost = 0
  while rem:
    nxt = min(rem, key=lambda j: mat[cur][j])
    cost += mat[cur][nxt]
    rem.remove(nxt)
    cur = nxt
  return cost + mat[cur][0]


def _extract_matrix_and_opt(task_str: str):
  m_start = task_str.find("'m': [[[")
  if m_start < 0:
    return None, None
  m_start = task_str.find("[[[", m_start)
  m_end = task_str.find("]]]", m_start) + 3
  mat = json.loads(task_str[m_start:m_end].replace("'", "\""))[0]
  opt = int(task_str.split("outputs=[")[1].split("]")[0])
  return mat, opt


def _run_one(seed: int, port: int):
  rng = random.Random(seed)
  set_global_seed(seed)
  run_dir = os.path.join(ROOT, "night_nn_vs_ab", f"seed_{seed}")
  data_dir = os.path.join(run_dir, "data")
  os.makedirs(data_dir, exist_ok=True)
  for fn in os.listdir(data_dir):
    if fn.startswith("train-") and fn.endswith(".pkl"):
      os.remove(os.path.join(data_dir, fn))

  train_tasks, eval_tasks = tsi.make_atsp_tasks(
      5, 220, 80, _random_atsp5, rng, train_prefix=f"seed{seed}_train", eval_prefix=f"seed{seed}_eval")
  with open(os.path.join(data_dir, "train-weight-3-00000.pkl"), "wb") as f:
    pickle.dump(train_tasks, f)

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
  cfg.save_dir = run_dir
  cfg.data_save_dir = data_dir
  cfg.json_results_file = os.path.join(run_dir, "results.json")
  cfg.do_test = False
  cfg.dynamic_tasks = False
  cfg.abstraction = False
  cfg.train_steps = 50
  cfg.eval_every = 25
  cfg.grad_accumulate = 4
  cfg.num_proc = 1
  cfg.gpu_list = "0"
  cfg.gpu = 0
  cfg.use_ur = False
  cfg.use_ur_in_valid = False
  cfg.timeout = 1.0
  cfg.max_search_weight = 15
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

  model = init_model(cfg, domains.get_domain(cfg.domain), cfg.model_type)
  main_train_eval(cfg, model, trace_gen=data_gen.trace_gen, checkpoint=None, original_tasks=eval_tasks)

  with open(cfg.json_results_file, "r") as f:
    results = json.load(f)
  rows = results["results"]
  total = len(rows)
  solved = sum(1 for r in rows if r.get("success"))
  nn_within_10 = 0
  mean_gap = 0.0
  cnt = 0
  for r in rows:
    mat, opt = _extract_matrix_and_opt(r["task"])
    if mat is None:
      continue
    nn = _nn_cost(mat)
    gap = (nn - opt) / opt
    mean_gap += gap
    cnt += 1
    if gap <= 0.10:
      nn_within_10 += 1
  mean_gap = 100.0 * mean_gap / cnt if cnt else 0.0
  return {
      "seed": seed,
      "instances": total,
      "ab_exact_rate": solved / total if total else 0.0,
      "nn_within_10pct_rate": nn_within_10 / cnt if cnt else 0.0,
      "nn_mean_gap_pct": mean_gap,
      "zak_bad_if_gap_gt_10pct": True,
      "results": cfg.json_results_file,
  }


def main():
  mp.set_start_method("spawn", force=True)
  out = []
  port = 32100
  for seed in [41, 42, 43, 44]:
    print("=== seed", seed, "===")
    row = _run_one(seed, port)
    out.append(row)
    port += 1
    print(row)
  path = os.path.join(ROOT, "night_nn_vs_ab_summary.json")
  with open(path, "w") as f:
    json.dump(out, f, indent=2)
  print(json.dumps(out, indent=2))
  print("wrote", path)


if __name__ == "__main__":
  main()
