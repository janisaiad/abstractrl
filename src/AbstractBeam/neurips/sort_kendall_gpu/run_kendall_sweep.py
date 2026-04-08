#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import sys
from dataclasses import dataclass
from typing import List

import torch.multiprocessing as mp

_ABSTRACTBEAM_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ABSTRACTBEAM_ROOT not in sys.path:
  sys.path.insert(0, _ABSTRACTBEAM_ROOT)

from crossbeam.common.configs_all import get_config
from crossbeam.datasets import data_gen
from crossbeam.dsl import deepcoder_operations as ops
from crossbeam.dsl import domains
from crossbeam.dsl import task as task_module
from crossbeam.dsl import value as value_module
from crossbeam.experiment.exp_common import set_global_seed
from crossbeam.experiment.run_crossbeam import init_model
from crossbeam.experiment.train_eval import main_train_eval
from crossbeam.experiment.train_eval import do_eval

Task = task_module.Task
Sort = ops.Sort()

ROOT = os.path.dirname(os.path.abspath(__file__))


def inversions_count(xs: List[int]) -> int:
  inv = 0
  n = len(xs)
  for i in range(n):
    xi = xs[i]
    for j in range(i + 1, n):
      if xi > xs[j]:
        inv += 1
  return inv


def kendall_tau_norm(xs: List[int]) -> float:
  n = len(xs)
  denom = n * (n - 1) // 2
  if denom == 0:
    return 0.0
  return inversions_count(xs) / denom


def sample_with_target_tau(rng: random.Random, length: int, tau_target: float, tol: float = 0.05) -> List[int]:
  base = list(range(-length, 0)) + list(range(1, length + 1))
  vals = rng.sample(base, k=length)
  vals.sort()

  # Guided random walk in permutation space to approximate target tau.
  best = vals[:]
  best_gap = abs(kendall_tau_norm(best) - tau_target)
  cur = vals[:]
  for _ in range(2000):
    i = rng.randrange(length)
    j = rng.randrange(length)
    if i == j:
      continue
    cur[i], cur[j] = cur[j], cur[i]
    tau = kendall_tau_norm(cur)
    gap = abs(tau - tau_target)
    if gap < best_gap:
      best = cur[:]
      best_gap = gap
      if gap <= tol:
        return best
    # occasionally revert to avoid random drift
    if rng.random() < 0.15:
      cur = best[:]
  return best


def make_kendall_sort_tasks(
    rng: random.Random,
    length: int,
    tau_target: float,
    num_tasks: int,
    num_examples: int,
    name_prefix: str,
) -> List[Task]:
  tasks: List[Task] = []
  for t in range(num_tasks):
    xs = [sample_with_target_tau(rng, length, tau_target) for _ in range(num_examples)]
    x_var = value_module.InputVariable(xs, name="x")
    tasks.append(Task(
        name=f"{name_prefix}:L{length}:tau{tau_target:.2f}:{t}",
        inputs_dict={"x": xs},
        outputs=[sorted(x) for x in xs],
        solution=Sort.apply([x_var]),
    ))
  return tasks


@dataclass
class RunCfg:
  length: int
  tau: float
  seed: int
  steps: int
  train_tasks: int
  eval_tasks: int
  out_root: str
  port: int


def run_one(cfg_item: RunCfg) -> dict:
  set_global_seed(cfg_item.seed)
  rng = random.Random(cfg_item.seed)
  tag = f"L{cfg_item.length}_tau{cfg_item.tau:.2f}_s{cfg_item.seed}_st{cfg_item.steps}"
  run_dir = os.path.join(cfg_item.out_root, tag)
  data_dir = os.path.join(run_dir, "data")
  os.makedirs(data_dir, exist_ok=True)
  for fn in os.listdir(data_dir):
    if fn.startswith("train-") and fn.endswith(".pkl"):
      os.remove(os.path.join(data_dir, fn))

  train_tasks = make_kendall_sort_tasks(
      rng, cfg_item.length, cfg_item.tau, cfg_item.train_tasks, 5, f"{tag}_train")
  eval_tasks = make_kendall_sort_tasks(
      random.Random(cfg_item.seed + 9999), cfg_item.length, cfg_item.tau, cfg_item.eval_tasks, 5, f"{tag}_eval")
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
  cfg.train_steps = cfg_item.steps
  cfg.eval_every = max(10, cfg_item.steps // 2)
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
  cfg.port = str(cfg_item.port)

  model = init_model(cfg, domains.get_domain(cfg.domain), cfg.model_type)
  main_train_eval(cfg, model, trace_gen=data_gen.trace_gen, checkpoint=None, original_tasks=eval_tasks)

  # Robust metric collection: evaluate directly with the trained model.
  solved_rate, json_dict = do_eval(
      eval_tasks, domains.get_domain(cfg.domain), model,
      max_search_weight=cfg.max_search_weight,
      beam_size=cfg.beam_size,
      device="cuda",
      timeout=cfg.timeout,
      restarts_timeout=cfg.restarts_timeout,
      max_values_explored=cfg.max_values_explored,
      is_stochastic=cfg.stochastic_beam,
      temperature=cfg.temperature,
      use_ur=cfg.use_ur,
      use_type_masking=cfg.type_masking,
      static_weight=cfg.static_weight,
  )
  solved = int(round(solved_rate * len(eval_tasks)))
  total = len(eval_tasks)
  with open(cfg.json_results_file, "w") as f:
    json.dump(json_dict, f, indent=2)
  return {
      "length": cfg_item.length,
      "tau_target": cfg_item.tau,
      "seed": cfg_item.seed,
      "steps": cfg_item.steps,
      "solved": solved,
      "total": total,
      "rate": solved / total if total else 0.0,
      "results": cfg.json_results_file,
  }


def parse_args():
  p = argparse.ArgumentParser()
  p.add_argument("--lengths", type=str, default="5,10,20")
  p.add_argument("--taus", type=str, default="0.10,0.30,0.50,0.70,0.90")
  p.add_argument("--seeds", type=str, default="101")
  p.add_argument("--steps", type=int, default=12)
  p.add_argument("--train_tasks", type=int, default=70)
  p.add_argument("--eval_tasks", type=int, default=30)
  p.add_argument("--out_dir", type=str, default=os.path.join(ROOT, "kendall_sweep"))
  return p.parse_args()


def main():
  args = parse_args()
  mp.set_start_method("spawn", force=True)
  os.makedirs(args.out_dir, exist_ok=True)

  lengths = [int(x) for x in args.lengths.split(",") if x]
  taus = [float(x) for x in args.taus.split(",") if x]
  seeds = [int(x) for x in args.seeds.split(",") if x]

  rows = []
  port = 34000
  total_runs = len(lengths) * len(taus) * len(seeds)
  idx = 0
  for L in lengths:
    for tau in taus:
      for seed in seeds:
        idx += 1
        print(f"[{idx}/{total_runs}] L={L} tau={tau:.2f} seed={seed}")
        row = run_one(RunCfg(
            length=L, tau=tau, seed=seed, steps=args.steps,
            train_tasks=args.train_tasks, eval_tasks=args.eval_tasks,
            out_root=args.out_dir, port=port))
        port += 1
        rows.append(row)
        with open(os.path.join(args.out_dir, "progress.json"), "w") as f:
          json.dump(rows, f, indent=2)
        print(f"rate={row['rate']:.3f}")

  out = {
      "meta": {
          "lengths": lengths,
          "taus": taus,
          "seeds": seeds,
          "steps": args.steps,
          "train_tasks": args.train_tasks,
          "eval_tasks": args.eval_tasks,
      },
      "rows": rows,
  }
  out_path = os.path.join(args.out_dir, "summary.json")
  with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
  print("wrote", out_path)


if __name__ == "__main__":
  main()
