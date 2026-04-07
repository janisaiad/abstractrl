#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import pickle
import random
import sys
from datetime import datetime

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
OUT_DIR = os.path.join(ROOT, "results_sort_config_search")


def run_one(cfg_item: dict, port: int) -> dict:
  seed = cfg_item["seed"]
  set_global_seed(seed)
  rng = random.Random(seed)

  tag = (
      f"L{cfg_item['length']}_m{cfg_item['mode']}_sw{cfg_item['swaps']}"
      f"_st{cfg_item['steps']}_s{seed}"
  )
  run_dir = os.path.join(OUT_DIR, tag)
  data_dir = os.path.join(run_dir, "data")
  os.makedirs(data_dir, exist_ok=True)
  for fn in os.listdir(data_dir):
    if fn.startswith("train-") and fn.endswith(".pkl"):
      os.remove(os.path.join(data_dir, fn))

  train_tasks = slg.make_sort_tasks(
      rng,
      list_len=cfg_item["length"],
      num_tasks=cfg_item["train_tasks"],
      num_examples=5,
      mode=cfg_item["mode"],
      num_swaps=cfg_item["swaps"],
      name_prefix=f"{tag}_train",
  )
  eval_tasks = slg.make_sort_tasks(
      random.Random(seed + 12345),
      list_len=cfg_item["length"],
      num_tasks=cfg_item["eval_tasks"],
      num_examples=5,
      mode=cfg_item["mode"],
      num_swaps=cfg_item["swaps"],
      name_prefix=f"{tag}_eval",
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
  cfg.save_dir = run_dir
  cfg.data_save_dir = data_dir
  cfg.json_results_file = os.path.join(run_dir, "results.json")
  cfg.do_test = False
  cfg.dynamic_tasks = False
  cfg.abstraction = False
  cfg.train_steps = cfg_item["steps"]
  cfg.eval_every = max(10, cfg_item["steps"] // 2)
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
      **cfg_item,
      "solved": solved,
      "total": total,
      "rate": solved / total if total else 0.0,
      "results": cfg.json_results_file,
      "finished_at": datetime.utcnow().isoformat() + "Z",
  }


def main():
  mp.set_start_method("spawn", force=True)
  os.makedirs(OUT_DIR, exist_ok=True)

  lengths = list(range(5, 19))
  modes = [
      ("random", 0),
      ("nearly_sorted_adjacent", 2),
      ("nearly_sorted_adjacent", 6),
      ("nearly_sorted_swap", 2),
      ("nearly_sorted_swap", 8),
      ("nearly_sorted_swap", 16),
  ]
  steps_list = [20, 40]
  seeds = [101, 202]

  grid = []
  for length in lengths:
    for mode, swaps in modes:
      for steps in steps_list:
        for seed in seeds:
          grid.append({
              "length": length,
              "mode": mode,
              "swaps": swaps,
              "steps": steps,
              "seed": seed,
              "train_tasks": 100,
              "eval_tasks": 30,
          })

  results = []
  port = 33000
  progress_path = os.path.join(OUT_DIR, "progress.json")
  for i, item in enumerate(grid, start=1):
    print(f"[{i}/{len(grid)}] start {item}")
    row = run_one(item, port=port)
    port += 1
    results.append(row)
    with open(progress_path, "w") as f:
      json.dump(results, f, indent=2)
    print(f"[{i}/{len(grid)}] done rate={row['rate']:.3f}")

  out_path = os.path.join(OUT_DIR, "final_summary.json")
  with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
  print("wrote", out_path)


if __name__ == "__main__":
  main()
