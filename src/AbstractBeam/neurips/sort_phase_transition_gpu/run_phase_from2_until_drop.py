#!/usr/bin/env python3
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


def run_suite(length: int, swaps: int, seed: int, port: int):
  set_global_seed(seed)
  rng = random.Random(seed)
  name = f"from2_len{length}_sw{swaps}"
  out_dir = os.path.join(ROOT, name)
  data_dir = os.path.join(out_dir, "data")
  os.makedirs(data_dir, exist_ok=True)
  for fn in os.listdir(data_dir):
    if fn.startswith("train-") and fn.endswith(".pkl"):
      os.remove(os.path.join(data_dir, fn))

  train_tasks = slg.make_sort_tasks(
      rng, list_len=length, num_tasks=90, num_examples=5,
      mode="nearly_sorted_swap", num_swaps=swaps, name_prefix=f"{name}_train")
  eval_tasks = slg.make_sort_tasks(
      random.Random(seed + 999), list_len=length, num_tasks=30, num_examples=5,
      mode="nearly_sorted_swap", num_swaps=swaps, name_prefix=f"{name}_eval")

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
  cfg.train_steps = 20
  cfg.eval_every = 10
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

  model = init_model(cfg, domains.get_domain(cfg.domain), cfg.model_type)
  main_train_eval(cfg, model, trace_gen=data_gen.trace_gen, checkpoint=None, original_tasks=eval_tasks)

  with open(cfg.json_results_file, "r") as f:
    r = json.load(f)
  solved = r.get("num_tasks_solved", 0)
  total = r.get("num_tasks", len(eval_tasks))
  return {
      "length": length,
      "swaps": swaps,
      "solved": solved,
      "total": total,
      "rate": solved / total if total else 0.0,
      "results": cfg.json_results_file,
  }


def main():
  mp.set_start_method("spawn", force=True)
  os.makedirs(ROOT, exist_ok=True)

  lengths = range(5, 11)
  swaps_grid = [2, 4, 6, 8, 10, 12, 16, 20, 24, 28, 32]
  stop_rate = 0.5
  all_rows = []
  threshold = {}
  port = 32000
  for length in lengths:
    dropped = None
    for swaps in swaps_grid:
      print(f"=== len={length}, swaps={swaps} ===")
      row = run_suite(length, swaps, seed=70000 + length * 100 + swaps, port=port)
      port += 1
      all_rows.append(row)
      print(f"DONE len={length}, swaps={swaps}: {row['solved']}/{row['total']} ({row['rate']*100:.1f}%)")
      if row["rate"] <= stop_rate:
        dropped = swaps
        break
    threshold[str(length)] = {
        "first_drop_leq_50pct": dropped
    }

  out = {
      "rows": all_rows,
      "thresholds": threshold,
      "stop_condition": "rate <= 0.5",
  }
  path = os.path.join(ROOT, "phase_from2_until_drop_summary.json")
  with open(path, "w") as f:
    json.dump(out, f, indent=2)
  print(json.dumps(out, indent=2))
  print("wrote", path)


if __name__ == "__main__":
  main()
