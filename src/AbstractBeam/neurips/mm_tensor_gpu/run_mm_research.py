#!/usr/bin/env python3
"""Lance baselines tensor CP + entraînement Abstract Beam (abstraction=True) sur MM n×n (défaut 4×4)."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import subprocess
import sys
import time
import shlex

import torch.multiprocessing as mp

_ABSTRACTBEAM_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ABSTRACTBEAM_ROOT not in sys.path:
  sys.path.insert(0, _ABSTRACTBEAM_ROOT)

from crossbeam.common.configs_all import get_config
from crossbeam.data.mm_tensor import mm_tasks as mm_tasks_module
from crossbeam.datasets import data_gen
from crossbeam.dsl import domains
from crossbeam.experiment.exp_common import set_global_seed
import torch

from crossbeam.experiment.run_crossbeam import init_model
from crossbeam.experiment.train_eval import main_train_eval

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_ROOT = os.path.join(SCRIPT_DIR, "mm_abstrue_runs")
MM_N = 4
EVAL_TASKS_NAME = "eval_tasks.pkl"
PHASE_META_NAME = "phase_meta.json"


def _run_tensor_baselines():
  import importlib.util

  def load_run(mod_name: str, fname: str):
    path = os.path.join(SCRIPT_DIR, fname)
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod

  tcp = load_run("tcp", "tensor_cp_baseline.py")
  evo = load_run("evo", "evolve_tensor_cp.py")
  r1 = tcp.run_baselines(os.path.join(SCRIPT_DIR, "tensor_cp_report.json"))
  r2 = evo.run_evolution(os.path.join(SCRIPT_DIR, "tensor_evolve_report.json"))
  return {"cp_als": r1, "evolution_rank7": r2}


def run_training_phase(
    *,
    seed: int,
    port: str,
    n: int,
    phase_out_dir: str,
    train_steps_absolute: int,
    eval_every: int,
    resume_same_phase: bool,
    timeout: float,
    max_search_weight: int,
    beam_size: int,
    num_train_pairs: int,
    num_eval_pairs: int,
    tasks_per_split: int,
    smoke: bool,
    lr: float | None = None,
) -> dict:
  """Une portion d'entraînement ; si resume_same_phase et checkpoint présent, reprend les poids/inventions."""
  set_global_seed(seed)
  rng = random.Random(seed)
  out_dir = phase_out_dir
  data_dir = os.path.join(out_dir, "data")
  os.makedirs(data_dir, exist_ok=True)
  eval_path = os.path.join(out_dir, EVAL_TASKS_NAME)
  ckpt_path = os.path.join(out_dir, "model-latest.ckpt")

  if resume_same_phase and os.path.isfile(ckpt_path) and os.path.isfile(eval_path):
    with open(eval_path, "rb") as f:
      eval_tasks = pickle.load(f)
    train_files = [fn for fn in os.listdir(data_dir) if fn.startswith("train-") and fn.endswith(".pkl")]
    if not train_files:
      raise FileNotFoundError("resume sans shard train dans data/")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
  else:
    ckpt = None
    for fn in os.listdir(data_dir) if os.path.isdir(data_dir) else []:
      if fn.startswith("train-") and fn.endswith(".pkl"):
        os.remove(os.path.join(data_dir, fn))
    train, eval_tasks = mm_tasks_module.make_mm_train_eval_split(
        rng,
        n,
        num_train_pairs=num_train_pairs,
        num_eval_pairs=num_eval_pairs,
        tasks_per_split=tasks_per_split,
        lo=-2,
        hi=2,
        train_prefix="mm_train",
        eval_prefix="mm_eval",
    )
    with open(eval_path, "wb") as f:
      pickle.dump(eval_tasks, f, pickle.HIGHEST_PROTOCOL)
    pkl_path = os.path.join(data_dir, "train-weight-3-00000.pkl")
    with open(pkl_path, "wb") as f:
      pickle.dump(train, f, pickle.HIGHEST_PROTOCOL)

  cfg = get_config()
  cfg.domain = "mm_tensor"
  cfg.seed = seed
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
  cfg.abstraction_refinement = True
  cfg.num_starting_ops = 28
  cfg.initialization_method = "top"
  cfg.abstraction_pruning = True
  cfg.top_k = 3
  cfg.num_inventions_per_iter = 24
  cfg.invention_arity = 3
  cfg.max_invention = 24
  cfg.train_steps = 121 if smoke else int(train_steps_absolute)
  cfg.eval_every = 120 if smoke else int(eval_every)
  cfg.grad_accumulate = 4
  cfg.num_proc = 1
  cfg.gpu_list = "0"
  cfg.gpu = 0
  cfg.use_ur = False
  cfg.use_ur_in_valid = True
  cfg.timeout = timeout
  cfg.max_search_weight = max_search_weight
  cfg.beam_size = beam_size
  cfg.used_invs = None
  cfg.lr = 5e-4 if lr is None else lr
  cfg.grad_clip = 5.0
  cfg.port = port
  cfg.type_masking = True
  cfg.random_beam = False
  cfg.stochastic_beam = False
  cfg.log_every = 5

  if ckpt is not None:
    domain = ckpt["domain"]
    model = init_model(cfg, domain, cfg.model_type, ckpt["inventions"])
  else:
    model = init_model(cfg, domains.get_domain(cfg.domain), cfg.model_type)

  t0 = time.time()
  main_train_eval(cfg, model, trace_gen=data_gen.trace_gen, checkpoint=ckpt, original_tasks=eval_tasks)
  elapsed = time.time() - t0

  summary = {
      "mm_n": n,
      "out_dir": out_dir,
      "elapsed_sec": elapsed,
      "eval_tasks": len(eval_tasks),
      "train_steps_target": train_steps_absolute,
      "resume": resume_same_phase,
  }
  if os.path.isfile(cfg.json_results_file):
    with open(cfg.json_results_file, "r", encoding="utf-8") as f:
      jd = json.load(f)
    rows = jd.get("results", [])
    summary["ab_solved"] = sum(1 for r in rows if r.get("success"))
    summary["ab_total"] = len(rows)
    summary["ab_rate"] = summary["ab_solved"] / max(1, summary["ab_total"])
  with open(os.path.join(out_dir, PHASE_META_NAME), "w", encoding="utf-8") as f:
    json.dump({**summary, "hyper": {"timeout": timeout, "max_search_weight": max_search_weight, "beam_size": beam_size, "eval_every": eval_every, "lr": cfg.lr}}, f, indent=2)
  return summary


def run_abstract_beam(smoke: bool, seed: int, port: str, n: int = MM_N) -> dict:
  set_global_seed(seed)
  tag = "smoke" if smoke else f"long_s{seed}_n{n}"
  out_dir = os.path.join(OUT_ROOT, tag)
  return run_training_phase(
      seed=seed,
      port=port,
      n=n,
      phase_out_dir=out_dir,
      train_steps_absolute=121 if smoke else 50_000_000,
      eval_every=120 if smoke else 3500,
      resume_same_phase=False,
      timeout=5.0 if smoke else 14.0,
      max_search_weight=100 if n >= 4 else 36,
      beam_size=18 if n >= 4 else 14,
      num_train_pairs=3 if smoke else 12,
      num_eval_pairs=3 if smoke else 8,
      tasks_per_split=1 if smoke else 5,
      smoke=smoke,
  )


def main():
  mp.set_start_method("spawn", force=True)
  parser = argparse.ArgumentParser()
  parser.add_argument("--smoke", action="store_true")
  parser.add_argument("--seed", type=int, default=901)
  parser.add_argument("--port", type=str, default="58291")
  parser.add_argument("--skip_baselines", action="store_true")
  parser.add_argument("--nohup_long", action="store_true", help="Soumettre run long via nohup et quitter")
  parser.add_argument("--n", type=int, default=MM_N, help="Taille des matrices n×n")
  args = parser.parse_args()

  if not args.skip_baselines:
    base = _run_tensor_baselines()
    with open(os.path.join(SCRIPT_DIR, "combined_tensor_baselines.json"), "w", encoding="utf-8") as f:
      json.dump(base, f, indent=2)

  if args.nohup_long and not args.smoke:
    this_py = os.path.abspath(__file__)
    logp = os.path.join(OUT_ROOT, f"nohup_mm_s{args.seed}_n{args.n}.log")
    os.makedirs(OUT_ROOT, exist_ok=True)
    cmd = ["uv", "run", "python", this_py, "--seed", str(args.seed), "--port", args.port, "--n", str(args.n), "--skip_baselines"]
    out = open(logp, "ab", buffering=0)
    subprocess.Popen(["nohup"] + cmd, cwd=_ABSTRACTBEAM_ROOT, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
    print("nohup lancé:", shlex.join(cmd), "log:", logp)
    return

  run_abstract_beam(smoke=args.smoke, seed=args.seed, port=args.port, n=args.n)


if __name__ == "__main__":
  main()
