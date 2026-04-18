#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import shutil
import subprocess
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
TRACE_DIR = DATA / "gcp_traces"
RUNS_DIR = ROOT / "runs" / "gcp_trm_auto"
TRAIN_JSONL = DATA / "gcp_train.jsonl"
SMALL_JSONL = DATA / "gcp_small.jsonl"


@dataclass
class Hyper:
    batch_size: int = 64
    steps_per_epoch: int = 2000
    valid_steps: int = 250
    epochs: int = 4
    lr: float = 3e-4
    grad_accum: int = 1
    d_model: int = 256
    refine_steps: int = 3
    dropout: float = 0.1


def run_cmd(cmd: list[str], cwd: Path) -> None:
    subprocess.run(cmd, cwd=str(cwd), check=True)


def ensure_small_jsonl() -> None:
    if SMALL_JSONL.exists():
        return
    if not TRAIN_JSONL.exists():
        raise FileNotFoundError(f"Missing {TRAIN_JSONL}")
    lines = [ln for ln in TRAIN_JSONL.read_text().splitlines() if ln.strip()]
    if len(lines) < 2:
        raise RuntimeError("Need at least 2 records in gcp_train.jsonl to build gcp_small.jsonl")
    SMALL_JSONL.write_text("\n".join(lines[:2]) + "\n")


def count_shards(pattern: str) -> int:
    return len(glob.glob(pattern))


def ensure_valid_split(seed: int) -> str:
    valid_dir = TRACE_DIR / "valid_auto"
    valid_dir.mkdir(parents=True, exist_ok=True)
    pat = str(valid_dir / "gcp-valid-*.pt")
    if count_shards(pat) > 0:
        return pat
    run_cmd(
        [
            "python",
            "gcp_trace_abstractbeam.py",
            "build-traces",
            "--input",
            str(TRAIN_JSONL),
            "--out-dir",
            str(valid_dir),
            "--prefix",
            "gcp-valid",
            "--samples-per-shard",
            "4000",
            "--corruptions-per-graph",
            "2500",
            "--max-steps",
            "24",
            "--seed",
            str(seed + 17),
            "--log-every-records",
            "1",
        ],
        ROOT,
    )
    return pat


def ensure_train_data(seed: int, stage: str, min_shards: int) -> str:
    if stage == "small":
        out_dir = TRACE_DIR / "train_small_auto"
        out_dir.mkdir(parents=True, exist_ok=True)
        pat = str(out_dir / "gcp-small-*.pt")
        if count_shards(pat) >= min_shards:
            return pat
        ensure_small_jsonl()
        run_cmd(
            [
                "python",
                "gcp_trace_abstractbeam.py",
                "build-traces",
                "--input",
                str(SMALL_JSONL),
                "--out-dir",
                str(out_dir),
                "--prefix",
                "gcp-small",
                "--samples-per-shard",
                "5000",
                "--corruptions-per-graph",
                "12000",
                "--max-steps",
                "20",
                "--seed",
                str(seed + 31),
                "--log-every-records",
                "1",
            ],
            ROOT,
        )
        return pat

    out_dir = TRACE_DIR / "train_huge"
    out_dir.mkdir(parents=True, exist_ok=True)
    pat = str(out_dir / "gcp-trace*.pt")
    if count_shards(pat) >= min_shards:
        return pat
    # Top-up additional data into the same directory with a new prefix.
    extra_prefix = f"gcp-trace-extra-{int(time.time())}"
    run_cmd(
        [
            "python",
            "gcp_trace_abstractbeam.py",
            "build-traces",
            "--input",
            str(TRAIN_JSONL),
            "--out-dir",
            str(out_dir),
            "--prefix",
            extra_prefix,
            "--samples-per-shard",
            "5000",
            "--corruptions-per-graph",
            "20000",
            "--max-steps",
            "32",
            "--seed",
            str(seed + 53),
            "--log-every-records",
            "1",
        ],
        ROOT,
    )
    return pat


def parse_train_log(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"best_valid_loss": None, "last_valid_loss": None, "improvement": None}
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                rows.append(json.loads(ln))
    if not rows:
        return {"best_valid_loss": None, "last_valid_loss": None, "improvement": None}
    valid_losses = [r["valid_loss"] for r in rows if "valid_loss" in r]
    if not valid_losses:
        train_losses = [r.get("train_loss") for r in rows if r.get("train_loss") is not None]
        last = train_losses[-1] if train_losses else None
        best = min(train_losses) if train_losses else None
    else:
        last = valid_losses[-1]
        best = min(valid_losses)
    improvement = None
    if best is not None and last is not None and best != 0:
        improvement = (last - best) / abs(best)
    return {"best_valid_loss": best, "last_valid_loss": last, "improvement": improvement, "epochs_logged": len(rows)}


def train_once(
    out_dir: Path,
    nproc: int,
    train_glob: str,
    valid_glob: str,
    h: Hyper,
    seed: int,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "torchrun",
        f"--nproc_per_node={nproc}",
        "gcp_trace_abstractbeam.py",
        "train",
        "--train",
        train_glob,
        "--valid",
        valid_glob,
        "--out-dir",
        str(out_dir),
        "--batch-size",
        str(h.batch_size),
        "--epochs",
        str(h.epochs),
        "--steps-per-epoch",
        str(h.steps_per_epoch),
        "--valid-steps",
        str(h.valid_steps),
        "--lr",
        str(h.lr),
        "--grad-accum",
        str(h.grad_accum),
        "--d-model",
        str(h.d_model),
        "--refine-steps",
        str(h.refine_steps),
        "--dropout",
        str(h.dropout),
        "--seed",
        str(seed),
        "--amp",
    ]
    run_cmd(cmd, ROOT)
    metrics = parse_train_log(out_dir / "train_log.jsonl")
    metrics["run_dir"] = str(out_dir)
    return metrics


def adjust_hyper(h: Hyper, metrics: dict[str, Any]) -> Hyper:
    best = metrics.get("best_valid_loss")
    last = metrics.get("last_valid_loss")
    if best is None or last is None:
        h.steps_per_epoch = min(3000, h.steps_per_epoch + 400)
        h.valid_steps = min(400, h.valid_steps + 50)
        return h
    # Plateau if gap between last and best is tiny.
    if abs(last - best) / max(abs(best), 1e-8) < 0.01:
        h.lr = max(8e-5, h.lr * 0.7)
        h.steps_per_epoch = min(3600, h.steps_per_epoch + 500)
        h.valid_steps = min(500, h.valid_steps + 60)
        h.batch_size = max(32, h.batch_size // 2)
        h.grad_accum = min(4, h.grad_accum + 1)
    else:
        h.lr = max(1e-4, h.lr * 0.9)
    return h


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--nproc", type=int, default=8)
    args = parser.parse_args()

    random.seed(args.seed)
    nproc = min(max(1, int(args.nproc)), max(1, torch.cuda.device_count() or 1))
    deadline = time.time() + args.hours * 3600.0
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    session = RUNS_DIR / f"session_{int(time.time())}"
    session.mkdir(parents=True, exist_ok=True)
    history_file = session / "autoresearch_history.jsonl"

    valid_glob = ensure_valid_split(args.seed)
    stage = "small"
    h = Hyper()
    iteration = 0
    best_seen = float("inf")

    while time.time() < deadline:
        train_glob = ensure_train_data(args.seed + iteration, stage=stage, min_shards=3 if stage == "small" else 8)
        run_dir = session / f"{stage}_iter_{iteration:03d}"
        try:
            metrics = train_once(run_dir, nproc, train_glob, valid_glob, h, seed=args.seed + 1000 + iteration)
        except subprocess.CalledProcessError as e:
            record = {"iteration": iteration, "stage": stage, "error": f"train_failed:{e.returncode}", "hyper": asdict(h)}
            with history_file.open("a") as f:
                f.write(json.dumps(record) + "\n")
            # Recovery: lower batch and nproc pressure.
            h.batch_size = max(16, h.batch_size // 2)
            h.grad_accum = min(6, h.grad_accum + 1)
            nproc = max(1, nproc // 2)
            iteration += 1
            continue

        best_loss = metrics.get("best_valid_loss")
        if best_loss is not None:
            best_seen = min(best_seen, float(best_loss))

        record = {
            "iteration": iteration,
            "stage": stage,
            "hyper": asdict(h),
            "metrics": metrics,
            "nproc": nproc,
            "train_glob": train_glob,
            "valid_glob": valid_glob,
            "best_seen": best_seen,
        }
        with history_file.open("a") as f:
            f.write(json.dumps(record) + "\n")

        # Curriculum:
        # - start small to break plateau
        # - move to full once model has signal or after two rounds
        if stage == "small" and (iteration >= 1 or (best_loss is not None and best_loss < 1.5)):
            stage = "full"

        # If full stage stalls, add more data and briefly go back small for stabilization.
        if stage == "full" and best_loss is not None and metrics.get("improvement") is not None:
            if abs(float(metrics["improvement"])) < 0.01:
                ensure_train_data(args.seed + 5000 + iteration, stage="full", min_shards=12)
                stage = "small"

        h = adjust_hyper(h, metrics)
        iteration += 1

    final = {
        "session": str(session),
        "history_file": str(history_file),
        "iterations": iteration,
        "best_seen_valid_loss": best_seen,
        "final_stage": stage,
        "final_hyper": asdict(h),
        "nproc_final": nproc,
    }
    with (session / "final_status.json").open("w") as f:
        json.dump(final, f, indent=2)
    print(json.dumps(final, indent=2), flush=True)


if __name__ == "__main__":
    main()
