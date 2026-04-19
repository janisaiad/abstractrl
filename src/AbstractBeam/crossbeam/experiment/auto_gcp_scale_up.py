#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import random
import subprocess
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs" / "gcp_trm_auto"
SCALE_ROOT = ROOT / "runs" / "gcp_trm_scaleup"
DATA_ROOT = ROOT / "data" / "gcp_scaleup"


@dataclass
class StageCfg:
    n: int
    k: int
    train_graphs: int
    valid_graphs: int
    eval_graphs: int
    p: float
    epochs: int
    steps_per_epoch: int
    valid_steps: int
    batch_size: int
    corruptions_train: int
    corruptions_valid: int


def run_cmd(cmd: list[str], cwd: Path) -> None:
    subprocess.run(cmd, cwd=str(cwd), check=True)


def process_running(pattern: str) -> bool:
    res = subprocess.run(["pgrep", "-af", pattern], capture_output=True, text=True)
    if res.returncode != 0:
        return False
    # Filter out the pgrep command itself in some shells.
    lines = [ln for ln in res.stdout.splitlines() if "pgrep -af" not in ln]
    return len(lines) > 0


def wait_small_autoresearch_done(poll_sec: int = 60) -> None:
    pattern = "auto_gcp_research.py --hours 10 --nproc 8 --seed 2027"
    while process_running(pattern):
        print("Waiting small-graph autoresearch to finish...", flush=True)
        time.sleep(poll_sec)
    print("Small-graph autoresearch finished.", flush=True)


def latest_session_dir() -> Path:
    sessions = sorted([p for p in RUNS.glob("session_*") if p.is_dir()], key=lambda p: p.name)
    if not sessions:
        raise RuntimeError(f"No session_* found under {RUNS}")
    return sessions[-1]


def best_ckpt_from_session(session: Path) -> Path:
    hist = session / "autoresearch_history.jsonl"
    if not hist.exists():
        raise RuntimeError(f"Missing history file: {hist}")
    best = None
    best_run_dir = None
    for ln in hist.read_text().splitlines():
        if not ln.strip():
            continue
        row = json.loads(ln)
        b = row.get("metrics", {}).get("best_valid_loss")
        r = row.get("metrics", {}).get("run_dir")
        if isinstance(b, (int, float)) and r:
            if best is None or b < best:
                best = float(b)
                best_run_dir = Path(r)
    if best_run_dir is None:
        raise RuntimeError("Could not find a valid best_run_dir from history")
    ckpt = best_run_dir / "model-best.pt"
    if not ckpt.exists():
        ckpt = best_run_dir / "model-last.pt"
    if not ckpt.exists():
        raise RuntimeError(f"No model checkpoint found in {best_run_dir}")
    print(f"Using base checkpoint: {ckpt}", flush=True)
    return ckpt


def _dsatur_ub_on_rec(rec: dict[str, Any]) -> int:
    import numpy as np

    import gcp_trace_abstractbeam as gtr

    edges = rec.get("edges") or []
    arr = np.asarray(edges, dtype=np.int64)
    if arr.size == 0:
        arr = arr.reshape(0, 2)
    sol = rec.get("solution")
    gr = gtr.GraphRecord(
        name=str(rec.get("name", "tmp")),
        n=int(rec["n"]),
        edges=arr,
        solution=np.asarray(sol, dtype=np.int16) if sol is not None else None,
    ).to_runtime()
    return int(gr.dsatur_ub)


def harden_planted_until_dsatur_k(rec: dict[str, Any], k: int, rng: random.Random, max_extra: int) -> None:
    """Add cross-class edges until DSATUR needs at least k colors (graph stays k-colorable by the planted coloring)."""
    colors: list[int] = [int(c) for c in rec["solution"]]
    n = int(rec["n"])
    edge_set: set[tuple[int, int]] = {tuple(sorted((int(u), int(v)))) for u, v in rec["edges"]}
    for _ in range(max_extra):
        rec["edges"] = [[a, b] for a, b in sorted(edge_set)]
        if _dsatur_ub_on_rec(rec) >= k:
            break
        u = rng.randrange(n)
        v = rng.randrange(n)
        if u == v or colors[u] == colors[v]:
            continue
        a, b = (u, v) if u < v else (v, u)
        if (a, b) in edge_set:
            continue
        edge_set.add((a, b))
    rec["edges"] = [[a, b] for a, b in sorted(edge_set)]


def make_colored_graph(n: int, k: int, p: float, rng: random.Random, name: str) -> dict[str, Any]:
    # Build a k-colorable graph by only adding edges across color classes.
    colors = [rng.randrange(k) for _ in range(n)]
    # Ensure all colors appear at least once when possible.
    for c in range(min(k, n)):
        colors[c] = c
    rng.shuffle(colors)
    edges: list[list[int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            if colors[i] != colors[j] and rng.random() < p:
                edges.append([i, j])
    rec: dict[str, Any] = {"name": name, "n": n, "edges": edges, "solution": colors}
    budget = min(500_000, max(1, n * n * 4))
    harden_planted_until_dsatur_k(rec, k, rng, max_extra=budget)
    return rec


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def build_dataset_files(stage: StageCfg, seed: int) -> tuple[Path, Path, Path]:
    rng = random.Random(seed + stage.n * 101)
    stage_dir = DATA_ROOT / f"n{stage.n}"
    train_file = stage_dir / "train.jsonl"
    valid_file = stage_dir / "valid.jsonl"
    eval_file = stage_dir / "eval.jsonl"
    train_rows = [
        make_colored_graph(stage.n, stage.k, stage.p, rng, f"n{stage.n}_train_{i:04d}")
        for i in range(stage.train_graphs)
    ]
    valid_rows = [
        make_colored_graph(stage.n, stage.k, stage.p, rng, f"n{stage.n}_valid_{i:04d}")
        for i in range(stage.valid_graphs)
    ]
    eval_rows = [
        make_colored_graph(stage.n, stage.k, stage.p, rng, f"n{stage.n}_eval_{i:04d}")
        for i in range(stage.eval_graphs)
    ]
    write_jsonl(train_file, train_rows)
    write_jsonl(valid_file, valid_rows)
    write_jsonl(eval_file, eval_rows)
    return train_file, valid_file, eval_file


def build_traces(input_jsonl: Path, out_dir: Path, prefix: str, corruptions: int, max_steps: int, seed: int) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    run_cmd(
        [
            "python",
            "gcp_trace_abstractbeam.py",
            "build-traces",
            "--input",
            str(input_jsonl),
            "--out-dir",
            str(out_dir),
            "--prefix",
            prefix,
            "--samples-per-shard",
            "5000",
            "--corruptions-per-graph",
            str(corruptions),
            "--max-steps",
            str(max_steps),
            "--seed",
            str(seed),
            "--log-every-records",
            "50",
        ],
        ROOT,
    )
    return str(out_dir / f"{prefix}-*.pt")


def train_stage(stage: StageCfg, train_glob: str, valid_glob: str, out_dir: Path, seed: int, nproc: int, init_ckpt: Path | None) -> Path:
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
        str(stage.batch_size),
        "--epochs",
        str(stage.epochs),
        "--steps-per-epoch",
        str(stage.steps_per_epoch),
        "--valid-steps",
        str(stage.valid_steps),
        "--amp",
        "--seed",
        str(seed),
    ]
    if init_ckpt is not None and init_ckpt.exists():
        cmd.extend(["--load", str(init_ckpt)])
    run_cmd(cmd, ROOT)
    ckpt = out_dir / "model-best.pt"
    if not ckpt.exists():
        ckpt = out_dir / "model-last.pt"
    return ckpt


def independent_conflicts(n: int, edges: list[list[int]], colors: list[int]) -> int:
    c = 0
    for u, v in edges:
        if colors[u] == colors[v]:
            c += 1
    return c


def eval_stage(stage: StageCfg, eval_jsonl: Path, ckpt: Path, out_dir: Path) -> dict[str, Any]:
    rows = [json.loads(x) for x in eval_jsonl.read_text().splitlines() if x.strip()]
    audits: list[dict[str, Any]] = []
    solved = 0
    for rec in rows:
        tmp = out_dir / f"{rec['name']}.json"
        tmp.write_text(json.dumps(rec))
        out = subprocess.check_output(
            [
                "python",
                "gcp_trace_abstractbeam.py",
                "solve",
                "--input",
                str(tmp),
                "--ckpt",
                str(ckpt),
                "--k",
                str(stage.k),
            ],
            cwd=str(ROOT),
            text=True,
        )
        sol = json.loads(out)
        indep = independent_conflicts(rec["n"], rec["edges"], sol["colors"])
        ok = indep == 0
        solved += int(ok)
        audits.append(
            {
                "name": rec["name"],
                "n": rec["n"],
                "k_used": sol["k"],
                "conflicts_reported": sol["conflicts"],
                "conflicts_independent": indep,
                "rigorous_solved": ok,
            }
        )
        tmp.unlink(missing_ok=True)
    result = {
        "n": stage.n,
        "eval_graphs": len(rows),
        "solved_rigorous": solved,
        "solved_rate": solved / max(1, len(rows)),
        "audits": audits,
    }
    (out_dir / "eval_audit.json").write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-start-now", action="store_true")
    parser.add_argument("--wait-poll-sec", type=int, default=60)
    args = parser.parse_args()
    SCALE_ROOT.mkdir(parents=True, exist_ok=True)
    if not args.force_start_now:
        wait_small_autoresearch_done(poll_sec=max(5, int(args.wait_poll_sec)))
    else:
        print("Force-start enabled: skipping wait for small-graph autoresearch.", flush=True)
    session_small = latest_session_dir()
    base_ckpt = best_ckpt_from_session(session_small)

    # Curriculum sizes requested by user.
    stages = [
        StageCfg(n=10, k=4, train_graphs=200, valid_graphs=60, eval_graphs=40, p=0.28, epochs=6, steps_per_epoch=1200, valid_steps=200, batch_size=64, corruptions_train=120, corruptions_valid=60),
        StageCfg(n=30, k=6, train_graphs=240, valid_graphs=80, eval_graphs=50, p=0.18, epochs=8, steps_per_epoch=1500, valid_steps=250, batch_size=48, corruptions_train=80, corruptions_valid=40),
        StageCfg(n=50, k=8, train_graphs=260, valid_graphs=90, eval_graphs=60, p=0.14, epochs=10, steps_per_epoch=1800, valid_steps=280, batch_size=40, corruptions_train=60, corruptions_valid=30),
        StageCfg(n=100, k=12, train_graphs=320, valid_graphs=100, eval_graphs=70, p=0.08, epochs=12, steps_per_epoch=2200, valid_steps=320, batch_size=24, corruptions_train=40, corruptions_valid=20),
    ]

    session = SCALE_ROOT / f"session_{int(time.time())}"
    session.mkdir(parents=True, exist_ok=True)
    history_path = session / "scaleup_history.jsonl"
    nproc = max(1, min(8, torch.cuda.device_count() or 1))

    current_ckpt = base_ckpt
    for i, st in enumerate(stages):
        stage_dir = session / f"n{st.n}"
        stage_dir.mkdir(parents=True, exist_ok=True)
        train_jsonl, valid_jsonl, eval_jsonl = build_dataset_files(st, seed=1000 + i)

        train_glob = build_traces(train_jsonl, stage_dir / "traces_train", "train-trace", st.corruptions_train, max_steps=32, seed=2000 + i)
        valid_glob = build_traces(valid_jsonl, stage_dir / "traces_valid", "valid-trace", st.corruptions_valid, max_steps=24, seed=3000 + i)

        try:
            current_ckpt = train_stage(st, train_glob, valid_glob, stage_dir / "train_run", seed=4000 + i, nproc=nproc, init_ckpt=current_ckpt)
        except subprocess.CalledProcessError:
            # Fallback to fewer processes and batch if resources are tight.
            nproc = max(1, nproc // 2)
            st.batch_size = max(16, st.batch_size // 2)
            current_ckpt = train_stage(st, train_glob, valid_glob, stage_dir / "train_run_retry", seed=5000 + i, nproc=nproc, init_ckpt=current_ckpt)

        eval_result = eval_stage(st, eval_jsonl, current_ckpt, stage_dir)
        row = {
            "stage": asdict(st),
            "checkpoint": str(current_ckpt),
            "eval_result": eval_result,
            "nproc": nproc,
        }
        with history_path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps({"stage_n": st.n, "solved_rate": eval_result["solved_rate"], "solved_rigorous": eval_result["solved_rigorous"]}), flush=True)

    # Final summary
    lines = [json.loads(x) for x in history_path.read_text().splitlines() if x.strip()]
    summary = {
        "session": str(session),
        "stages_done": [ln["stage"]["n"] for ln in lines],
        "final_checkpoint": str(current_ckpt),
        "results_by_n": {str(ln["stage"]["n"]): ln["eval_result"] for ln in lines},
    }
    (session / "final_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
