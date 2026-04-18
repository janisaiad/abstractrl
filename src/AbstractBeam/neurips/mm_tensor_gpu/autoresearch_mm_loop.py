#!/usr/bin/env python3
"""Autoresearch ~10h: phase n=2 (signal + réglage) puis n=4, chunks avec reprise + raffinement."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass

import torch
import torch.multiprocessing as mp

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ABSTRACTBEAM_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
if _ABSTRACTBEAM_ROOT not in sys.path:
  sys.path.insert(0, _ABSTRACTBEAM_ROOT)
if _SCRIPT_DIR not in sys.path:
  sys.path.insert(0, _SCRIPT_DIR)

import run_mm_research as rmm


@dataclass
class Hyper:
  timeout: float
  max_search_weight: int
  beam_size: int
  eval_every: int
  chunk_steps: int
  lr: float


def refine(h: Hyper, rate: float, n: int) -> Hyper:
  if rate <= 0.001:
    return Hyper(
        timeout=min(48.0, h.timeout * 1.28),
        max_search_weight=min(175, h.max_search_weight + 14),
        beam_size=min(32, h.beam_size + 4),
        eval_every=max(1200, int(h.eval_every * 0.9)),
        chunk_steps=min(14000, int(h.chunk_steps * 1.08)),
        lr=h.lr,
    )
  if rate < 0.08:
    return Hyper(
        timeout=min(40.0, h.timeout * 1.15),
        max_search_weight=min(155, h.max_search_weight + 10),
        beam_size=min(28, h.beam_size + 2),
        eval_every=max(1400, int(h.eval_every * 0.95)),
        chunk_steps=h.chunk_steps,
        lr=h.lr * 0.96,
    )
  return Hyper(
      timeout=h.timeout,
      max_search_weight=h.max_search_weight,
      beam_size=h.beam_size,
      eval_every=h.eval_every,
      chunk_steps=h.chunk_steps,
      lr=h.lr * 0.94,
  )


def step_target(phase_dir: str, chunk_steps: int) -> int:
  ckpt_path = os.path.join(phase_dir, "model-latest.ckpt")
  if os.path.isfile(ckpt_path):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return int(ck["step"]) + int(chunk_steps)
  return int(chunk_steps)


def run_phase_adaptive(
    *,
    n: int,
    phase_dir: str,
    seed: int,
    port_base: int,
    deadline: float,
    success_rate_stop: float,
    hyper: Hyper,
    num_train_pairs: int,
    num_eval_pairs: int,
    tasks_per_split: int,
    log_path: str,
    phase_label: str,
) -> tuple[Hyper, float]:
  port_off = 0
  h = hyper
  last_rate = 0.0
  chunk_idx = 0
  while time.time() < deadline:
    tgt = step_target(phase_dir, h.chunk_steps)
    resume = os.path.isfile(os.path.join(phase_dir, "model-latest.ckpt")) and os.path.isfile(
        os.path.join(phase_dir, rmm.EVAL_TASKS_NAME)
    )
    rec = {
        "ts": time.time(),
        "phase": phase_label,
        "chunk": chunk_idx,
        "n": n,
        "target_step": tgt,
        "resume": resume,
        "hyper": asdict(h),
    }
    with open(log_path, "a", encoding="utf-8") as lf:
      lf.write(json.dumps(rec) + "\n")
    summary = rmm.run_training_phase(
        seed=seed,
        port=str(port_base + port_off),
        n=n,
        phase_out_dir=phase_dir,
        train_steps_absolute=tgt,
        eval_every=h.eval_every,
        resume_same_phase=resume,
        timeout=h.timeout,
        max_search_weight=h.max_search_weight,
        beam_size=h.beam_size,
        num_train_pairs=num_train_pairs,
        num_eval_pairs=num_eval_pairs,
        tasks_per_split=tasks_per_split,
        smoke=False,
        lr=h.lr,
    )
    last_rate = float(summary.get("ab_rate", 0.0))
    out_rec = {"ts": time.time(), "phase": phase_label, "chunk": chunk_idx, "summary": summary}
    with open(log_path, "a", encoding="utf-8") as lf:
      lf.write(json.dumps(out_rec) + "\n")
    chunk_idx += 1
    port_off = (port_off + 1) % 80
    if last_rate >= success_rate_stop:
      break
    h = refine(h, last_rate, n)
  return h, last_rate


def main():
  mp.set_start_method("spawn", force=True)
  parser = argparse.ArgumentParser()
  parser.add_argument("--budget_hours", type=float, default=10.0)
  parser.add_argument("--seed", type=int, default=17042)
  parser.add_argument("--port_base", type=int, default=59000)
  args = parser.parse_args()

  t0 = time.time()
  deadline = t0 + float(args.budget_hours) * 3600.0
  session = os.path.join(rmm.OUT_ROOT, f"autoresearch_{int(t0)}")
  os.makedirs(session, exist_ok=True)
  log_path = os.path.join(session, "autoresearch.jsonl")

  h0 = Hyper(
      timeout=13.0,
      max_search_weight=88,
      beam_size=17,
      eval_every=2100,
      chunk_steps=8400,
      lr=5e-4,
  )

  phase2_budget = t0 + min(args.budget_hours * 3600 * 0.35, 3.5 * 3600)
  phase2_dir = os.path.join(session, "phase_n2")
  h_after_n2, rate_n2 = run_phase_adaptive(
      n=2,
      phase_dir=phase2_dir,
      seed=args.seed,
      port_base=args.port_base,
      deadline=min(deadline, phase2_budget),
      success_rate_stop=0.38,
      hyper=h0,
      num_train_pairs=14,
      num_eval_pairs=10,
      tasks_per_split=6,
      log_path=log_path,
      phase_label="n2",
  )

  remaining = deadline - time.time()
  if remaining < 600:
    with open(os.path.join(session, "final_status.json"), "w", encoding="utf-8") as f:
      json.dump(
          {
              "note": "budget épuisé après n=2",
              "rate_n2": rate_n2,
              "session": session,
          },
          f,
          indent=2,
      )
    return

  h_start_n4 = Hyper(
      timeout=max(18.0, h_after_n2.timeout * 1.05),
      max_search_weight=max(100, h_after_n2.max_search_weight + 4),
      beam_size=max(18, h_after_n2.beam_size + 1),
      eval_every=max(1600, int(h_after_n2.eval_every * 0.92)),
      chunk_steps=max(7200, int(h_after_n2.chunk_steps * 0.95)),
      lr=h_after_n2.lr,
  )

  phase4_dir = os.path.join(session, "phase_n4")
  h_after_n4, rate_n4 = run_phase_adaptive(
      n=4,
      phase_dir=phase4_dir,
      seed=args.seed + 7,
      port_base=args.port_base + 40,
      deadline=deadline,
      success_rate_stop=0.2,
      hyper=h_start_n4,
      num_train_pairs=12,
      num_eval_pairs=8,
      tasks_per_split=5,
      log_path=log_path,
      phase_label="n4",
  )

  with open(os.path.join(session, "final_status.json"), "w", encoding="utf-8") as f:
    json.dump(
        {
            "session": session,
            "budget_hours": args.budget_hours,
            "rate_n2": rate_n2,
            "rate_n4": rate_n4,
            "hyper_after_n2": asdict(h_after_n2),
            "hyper_after_n4": asdict(h_after_n4),
            "elapsed_sec": time.time() - t0,
        },
        f,
        indent=2,
    )


if __name__ == "__main__":
  main()