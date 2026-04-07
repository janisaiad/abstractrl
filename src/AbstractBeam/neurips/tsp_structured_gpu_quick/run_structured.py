#!/usr/bin/env python3
"""Court entraînement GPU : TSP ATSP sur instances hiérarchiques (clusters) et fractales (XOR)."""

from __future__ import annotations

import json
import os
import pickle
import random
import sys

_ABSTRACTBEAM_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ABSTRACTBEAM_ROOT not in sys.path:
  sys.path.insert(0, _ABSTRACTBEAM_ROOT)

import torch.multiprocessing as mp

from crossbeam.common.configs_all import get_config
from crossbeam.data.tsp import tsp_structured_instances as tsi
from crossbeam.datasets import data_gen
from crossbeam.dsl import domains
from crossbeam.experiment.exp_common import set_global_seed
from crossbeam.experiment.run_crossbeam import init_model
from crossbeam.experiment.train_eval import main_train_eval

ROOT = os.path.dirname(os.path.abspath(__file__))


def _run_suite(
    name: str,
    n: int,
    factory,
    seed: int,
    train_n: int,
    eval_n: int,
    steps: int,
    port: int,
):
  rng = random.Random(seed)
  set_global_seed(seed)
  suite_dir = os.path.join(ROOT, name)
  data_dir = os.path.join(suite_dir, "data")
  os.makedirs(data_dir, exist_ok=True)
  for fn in os.listdir(data_dir):
    if fn.startswith("train-") and fn.endswith(".pkl"):
      os.remove(os.path.join(data_dir, fn))

  train_tasks, eval_tasks = tsi.make_atsp_tasks(
      n,
      train_n,
      eval_n,
      lambda r: factory(n, r),
      rng,
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
  cfg.timeout = 1.0
  cfg.max_search_weight = max(8, n * 3)
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
  return {"name": name, "n": n, "solved": solved, "total": total, "rate": solved / total if total else 0.0}


def main():
  mp.set_start_method("spawn", force=True)
  out = []
  # Hiérarchique : n=4 (6 permutations), matrices contrastées
  out.append(
      _run_suite(
          "hierarchical_n4",
          4,
          tsi.hierarchical_clustered_matrix,
          seed=4242,
          train_n=120,
          eval_n=30,
          steps=30,
          port=31501,
      )
  )
  # Fractale (XOR) : même taille pour comparer
  out.append(
      _run_suite(
          "fractal_xor_n4",
          4,
          tsi.fractal_xor_weighted_matrix,
          seed=4343,
          train_n=120,
          eval_n=30,
          steps=30,
          port=31502,
      )
  )
  summary_path = os.path.join(ROOT, "structured_summary.json")
  with open(summary_path, "w") as f:
    json.dump(out, f, indent=2)
  print(json.dumps(out, indent=2))
  print("wrote", summary_path)


if __name__ == "__main__":
  main()
