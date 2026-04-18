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
RUN_TAG = "abstrue"


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
  return cost + mat[cur][0]


def extract_matrix_and_opt(task_str: str):
  m_start = task_str.find("'m': [[[")
  if m_start < 0:
    return None, None
  m_start = task_str.find("[[[", m_start)
  m_end = task_str.find("]]]", m_start) + 3
  mat = json.loads(task_str[m_start:m_end].replace("'", "\""))[0]
  opt = int(task_str.split("outputs=[")[1].split("]")[0])
  return mat, opt


def make_cluster_factory(n: int, intra_lo: int, intra_hi: int, inter_lo: int, inter_hi: int):
  return lambda r: tsi.hierarchical_clustered_matrix(
      n, r, num_clusters=2, intra_lo=intra_lo, intra_hi=intra_hi, inter_lo=inter_lo, inter_hi=inter_hi
  )


def run_level(level: dict, port: int):
  name = level["name"]
  n = level["n"]
  seed = level["seed"]
  set_global_seed(seed)
  rng = random.Random(seed)

  out_dir = os.path.join(ROOT, f"{name}_{RUN_TAG}")
  data_dir = os.path.join(out_dir, "data")
  os.makedirs(data_dir, exist_ok=True)
  for fn in os.listdir(data_dir):
    if fn.startswith("train-") and fn.endswith(".pkl"):
      os.remove(os.path.join(data_dir, fn))

  train_tasks, eval_tasks = tsi.make_atsp_tasks(
      n=n,
      num_train=level["num_train"],
      num_eval=level["num_eval"],
      matrix_factory=make_cluster_factory(
          n, level["intra_lo"], level["intra_hi"], level["inter_lo"], level["inter_hi"]
      ),
      rng=rng,
      train_prefix=f"{name}_train",
      eval_prefix=f"{name}_eval",
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
  cfg.abstraction = True
  cfg.num_starting_ops = 28
  cfg.initialization_method = "top"
  cfg.abstraction_pruning = True
  cfg.abstraction_refinement = True
  cfg.top_k = 2
  cfg.num_inventions_per_iter = 20
  cfg.invention_arity = 3
  cfg.max_invention = 20
  cfg.castrate_macros = False
  cfg.train_steps = level["train_steps"]
  cfg.eval_every = max(10, level["train_steps"] // 2)
  cfg.grad_accumulate = 4
  cfg.num_proc = 1
  cfg.gpu_list = "0"
  cfg.gpu = 0
  cfg.use_ur = False
  cfg.use_ur_in_valid = True
  cfg.timeout = 1.0
  cfg.max_search_weight = max(15, n * 4)
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
    result = json.load(f)
  rows = result["results"]
  total = len(rows)
  ab_solved = sum(1 for r in rows if r.get("success"))

  nn_within_10 = 0
  nn_mean_gap = 0.0
  count = 0
  for r in rows:
    mat, opt = extract_matrix_and_opt(r["task"])
    if mat is None:
      continue
    nn = nearest_neighbor_tour_cost(mat)
    gap = (nn - opt) / opt
    if gap <= 0.10:
      nn_within_10 += 1
    nn_mean_gap += gap
    count += 1

  return {
      "name": name,
      "n": n,
      "intra": [level["intra_lo"], level["intra_hi"]],
      "inter": [level["inter_lo"], level["inter_hi"]],
      "difficulty_index": level["difficulty_index"],
      "ab_exact_rate": ab_solved / total if total else 0.0,
      "nn_within_10pct_rate": nn_within_10 / count if count else 0.0,
      "nn_mean_gap_pct": 100.0 * nn_mean_gap / count if count else 0.0,
      "zak_bad_mean_gap_gt_10pct": (100.0 * nn_mean_gap / count > 10.0) if count else True,
      "instances_eval": total,
      "results_json": cfg.json_results_file,
  }


def main():
  mp.set_start_method("spawn", force=True)
  os.makedirs(ROOT, exist_ok=True)

  levels = [
      # very easy -> hard: reduce cluster separation + increase n.
      # For harder levels we substantially increase training set size.
      {"name": "lvl1_very_easy_n5_bigtrain", "n": 5, "intra_lo": 1, "intra_hi": 4, "inter_lo": 90, "inter_hi": 160, "difficulty_index": 1, "train_steps": 80, "num_train": 500, "num_eval": 220, "seed": 7101},
      {"name": "lvl2_easy_n5_bigtrain", "n": 5, "intra_lo": 1, "intra_hi": 8, "inter_lo": 60, "inter_hi": 140, "difficulty_index": 2, "train_steps": 80, "num_train": 700, "num_eval": 220, "seed": 7102},
      {"name": "lvl3_medium_n6_bigtrain", "n": 6, "intra_lo": 1, "intra_hi": 12, "inter_lo": 35, "inter_hi": 95, "difficulty_index": 3, "train_steps": 140, "num_train": 2200, "num_eval": 260, "seed": 7103},
      {"name": "lvl4_hard_n6_bigtrain", "n": 6, "intra_lo": 5, "intra_hi": 22, "inter_lo": 20, "inter_hi": 55, "difficulty_index": 4, "train_steps": 160, "num_train": 3200, "num_eval": 260, "seed": 7104},
      {"name": "lvl5_very_hard_n7_bigtrain", "n": 7, "intra_lo": 8, "intra_hi": 28, "inter_lo": 14, "inter_hi": 40, "difficulty_index": 5, "train_steps": 180, "num_train": 4200, "num_eval": 300, "seed": 7105},
  ]

  out = []
  port = 36001
  for lv in levels:
    print(f"=== {lv['name']} ===")
    row = run_level(lv, port)
    out.append(row)
    port += 1
    print(row)
    # Keep running all levels to verify whether larger train sets help hard regimes.

  out_path = os.path.join(ROOT, "difficulty_curve_summary_bigtrain_abstrue.json")
  with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
  print(json.dumps(out, indent=2))
  print("wrote", out_path)


if __name__ == "__main__":
  main()
