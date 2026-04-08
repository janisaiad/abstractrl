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


def extract_matrix_and_opt_from_task_str(task_str: str):
  m_start = task_str.find("'m': [[[")
  if m_start < 0:
    return None, None
  m_start = task_str.find("[[[", m_start)
  m_end = task_str.find("]]]", m_start) + 3
  mat = json.loads(task_str[m_start:m_end].replace("'", "\""))[0]
  opt = int(task_str.split("outputs=[")[1].split("]")[0])
  return mat, opt


def main():
  mp.set_start_method("spawn", force=True)
  out_dir = os.path.join(ROOT, "easy_n5_run")
  data_dir = os.path.join(out_dir, "data")
  os.makedirs(data_dir, exist_ok=True)
  for fn in os.listdir(data_dir):
    if fn.startswith("train-") and fn.endswith(".pkl"):
      os.remove(os.path.join(data_dir, fn))

  rng = random.Random(4242)
  set_global_seed(4242)

  # "Easy" TSP: strong cluster structure (intra << inter).
  train_tasks, eval_tasks = tsi.make_atsp_tasks(
      n=5,
      num_train=600,
      num_eval=300,
      matrix_factory=lambda r: tsi.hierarchical_clustered_matrix(
          5, r, num_clusters=2, intra_lo=1, intra_hi=6, inter_lo=60, inter_hi=140
      ),
      rng=rng,
      train_prefix="easy_tsp_train",
      eval_prefix="easy_tsp_eval",
  )

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
  cfg.save_dir = out_dir
  cfg.data_save_dir = data_dir
  cfg.json_results_file = os.path.join(out_dir, "results.json")
  cfg.do_test = False
  cfg.dynamic_tasks = False
  cfg.abstraction = False
  cfg.train_steps = 80
  cfg.eval_every = 40
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
  cfg.port = "35222"

  model = init_model(cfg, domains.get_domain(cfg.domain), cfg.model_type)
  main_train_eval(cfg, model, trace_gen=data_gen.trace_gen, checkpoint=None, original_tasks=eval_tasks)

  with open(cfg.json_results_file, "r") as f:
    result = json.load(f)
  rows = result["results"]

  ab_solved = sum(1 for r in rows if r.get("success"))
  nn_within_10 = 0
  nn_mean_gap = 0.0
  nn_beaten_by_ab = 0
  count = 0
  for r in rows:
    mat, opt = extract_matrix_and_opt_from_task_str(r["task"])
    if mat is None:
      continue
    nn = nearest_neighbor_tour_cost(mat)
    gap = (nn - opt) / opt
    nn_mean_gap += gap
    if gap <= 0.10:
      nn_within_10 += 1
    if r.get("success") and nn > opt:
      nn_beaten_by_ab += 1
    count += 1

  summary = {
      "instances": count,
      "ab_exact_solved": ab_solved,
      "ab_exact_rate": ab_solved / count if count else 0.0,
      "nn_within_10pct_count": nn_within_10,
      "nn_within_10pct_rate": nn_within_10 / count if count else 0.0,
      "nn_mean_gap_pct": 100.0 * nn_mean_gap / count if count else 0.0,
      "ab_exact_beats_nn_count": nn_beaten_by_ab,
      "zak_rule_bad_if_gap_gt_10pct": True,
      "config": {
          "n": 5,
          "instances_eval": 300,
          "easy_structure": "hierarchical clusters (2) with strong inter-cluster penalty",
          "intra_range": [1, 6],
          "inter_range": [60, 140],
          "train_steps": 80,
      },
      "results_json": cfg.json_results_file,
  }

  out_path = os.path.join(ROOT, "easy_n5_nn_vs_ab_summary.json")
  with open(out_path, "w") as f:
    json.dump(summary, f, indent=2)
  print(json.dumps(summary, indent=2))
  print("wrote", out_path)


if __name__ == "__main__":
  main()
