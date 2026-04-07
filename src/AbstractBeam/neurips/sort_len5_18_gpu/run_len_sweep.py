#!/usr/bin/env python3
"""Run GPU sweep for Sort(x) with list lengths 5..18."""

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
from crossbeam.dsl import domains
from crossbeam.experiment.exp_common import set_global_seed
from crossbeam.experiment.run_crossbeam import init_model
from crossbeam.experiment.train_eval import main_train_eval

ROOT = os.path.dirname(os.path.abspath(__file__))


def run_length(length: int, seed: int, port: int):
  set_global_seed(seed)
  rng = random.Random(seed)
  out_dir = os.path.join(ROOT, f"len_{length}")
  data_dir = os.path.join(out_dir, "data")
  os.makedirs(data_dir, exist_ok=True)
  for fn in os.listdir(data_dir):
    if fn.startswith("train-") and fn.endswith(".pkl"):
      os.remove(os.path.join(data_dir, fn))

  train_tasks = slg.make_sort_tasks(
      rng, list_len=length, num_tasks=140, num_examples=5,
      mode="random", num_swaps=0, name_prefix=f"len{length}_train")
  eval_tasks = slg.make_sort_tasks(
      random.Random(seed + 12345), list_len=length, num_tasks=40, num_examples=5,
      mode="random", num_swaps=0, name_prefix=f"len{length}_eval")

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
  cfg.save_dir = out_dir
  cfg.data_save_dir = data_dir
  cfg.json_results_file = os.path.join(out_dir, "results.json")
  cfg.do_test = False
  cfg.dynamic_tasks = False
  cfg.abstraction = False
  cfg.train_steps = 40
  cfg.eval_every = 20
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
    result = json.load(f)
  solved = result.get("num_tasks_solved", 0)
  total = result.get("num_tasks", len(eval_tasks))
  return {
      "length": length,
      "solved": solved,
      "total": total,
      "rate": solved / total if total else 0.0,
      "results": cfg.json_results_file,
  }


def main():
  mp.set_start_method("spawn", force=True)
  summary = []
  port = 31705
  for length in range(5, 19):
    print(f"=== length {length} ===")
    summary.append(run_length(length, seed=9000 + length, port=port))
    port += 1
    print(f"DONE length {length}: {summary[-1]['solved']}/{summary[-1]['total']} ({summary[-1]['rate']*100:.1f}%)")
  out_path = os.path.join(ROOT, "summary_len5_18.json")
  with open(out_path, "w") as f:
    json.dump(summary, f, indent=2)
  print(json.dumps(summary, indent=2))
  print("wrote", out_path)


if __name__ == "__main__":
  main()
