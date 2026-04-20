#!/usr/bin/env python3
from __future__ import annotations

import json
import random
import subprocess
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parent
SCALE_ROOT = ROOT / "runs" / "gcp_trm_scaleup_v2"
DATA_ROOT = ROOT / "data" / "gcp_scaleup_v2"


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
    teacher_train: int
    teacher_valid: int
    solve_train: int
    solve_valid: int
    simulations_trace: int
    simulations_eval: int


def run_cmd(cmd: list[str], cwd: Path) -> None:
    subprocess.run(cmd, cwd=str(cwd), check=True)


def make_forced_k_graph(n: int, k: int, p: float, rng: random.Random, name: str) -> dict[str, Any]:
    colors = [rng.randrange(k) for _ in range(n)]
    for c in range(min(k, n)):
        colors[c] = c
    rng.shuffle(colors)
    class_members: list[list[int]] = [[] for _ in range(k)]
    for i, c in enumerate(colors):
        class_members[c].append(i)
    edges: set[tuple[int, int]] = set()
    reps = [members[0] for members in class_members if members]
    for i in range(len(reps)):
        for j in range(i + 1, len(reps)):
            u, v = reps[i], reps[j]
            edges.add((min(u, v), max(u, v)))
    for i in range(n):
        ci = colors[i]
        for j in range(i + 1, n):
            if ci != colors[j] and rng.random() < p:
                edges.add((i, j))
    return {"name": name, "n": n, "edges": [[u, v] for (u, v) in sorted(edges)], "solution": colors, "tight_k": k}


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
    train_rows = [make_forced_k_graph(stage.n, stage.k, stage.p, rng, f"n{stage.n}_train_{i:04d}") for i in range(stage.train_graphs)]
    valid_rows = [make_forced_k_graph(stage.n, stage.k, stage.p, rng, f"n{stage.n}_valid_{i:04d}") for i in range(stage.valid_graphs)]
    eval_rows = [make_forced_k_graph(stage.n, stage.k, stage.p, rng, f"n{stage.n}_eval_{i:04d}") for i in range(stage.eval_graphs)]
    write_jsonl(train_file, train_rows)
    write_jsonl(valid_file, valid_rows)
    write_jsonl(eval_file, eval_rows)
    return train_file, valid_file, eval_file


def build_teacher_traces(input_jsonl: Path, out_dir: Path, prefix: str, corruptions: int, max_steps: int, seed: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    run_cmd(
        [
            "python", "gcp_trace_abstractbeam_v3.py", "build-traces",
            "--input", str(input_jsonl),
            "--out-dir", str(out_dir),
            "--prefix", prefix,
            "--samples-per-shard", "5000",
            "--corruptions-per-graph", str(corruptions),
            "--max-steps", str(max_steps),
            "--seed", str(seed),
            "--log-every-records", "50",
        ],
        ROOT,
    )


def build_solve_traces(input_jsonl: Path, out_dir: Path, prefix: str, ckpt: Path | None, k: int, episodes: int, max_steps: int, simulations: int, seed: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python", "gcp_trace_abstractbeam_v3.py", "build-solve-traces",
        "--input", str(input_jsonl),
        "--out-dir", str(out_dir),
        "--prefix", prefix,
        "--samples-per-shard", "5000",
        "--episodes-per-graph", str(episodes),
        "--max-steps", str(max_steps),
        "--k", str(k),
        "--simulations", str(simulations),
        "--seed", str(seed),
        "--log-every-records", "50",
    ]
    if ckpt is not None and ckpt.exists():
        cmd.extend(["--ckpt", str(ckpt)])
    run_cmd(cmd, ROOT)


def train_stage(stage: StageCfg, train_glob: str, valid_glob: str, out_dir: Path, seed: int, nproc: int, init_ckpt: Path | None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "torchrun", f"--nproc_per_node={nproc}", "gcp_trace_abstractbeam_v3.py", "train",
        "--train", train_glob,
        "--valid", valid_glob,
        "--out-dir", str(out_dir),
        "--batch-size", str(stage.batch_size),
        "--epochs", str(stage.epochs),
        "--steps-per-epoch", str(stage.steps_per_epoch),
        "--valid-steps", str(stage.valid_steps),
        "--amp",
        "--seed", str(seed),
    ]
    if init_ckpt is not None and init_ckpt.exists():
        cmd.extend(["--init-ckpt", str(init_ckpt)])
    run_cmd(cmd, ROOT)
    ckpt = out_dir / "model-best.pt"
    if not ckpt.exists():
        ckpt = out_dir / "model-last.pt"
    return ckpt


def eval_stage(stage: StageCfg, eval_jsonl: Path, ckpt: Path, out_dir: Path) -> dict[str, Any]:
    rows = [json.loads(x) for x in eval_jsonl.read_text().splitlines() if x.strip()]
    audits: list[dict[str, Any]] = []
    solved = 0
    for rec in rows:
        tmp = out_dir / f"{rec['name']}.json"
        tmp.write_text(json.dumps({"name": rec["name"], "n": rec["n"], "edges": rec["edges"]}))
        out = subprocess.check_output(
            [
                "python", "gcp_trace_abstractbeam_v3.py", "solve",
                "--input", str(tmp),
                "--ckpt", str(ckpt),
                "--k", str(stage.k),
                "--simulations", str(stage.simulations_eval),
                "--max-depth", "128",
            ],
            cwd=str(ROOT), text=True,
        )
        sol = json.loads(out)
        indep = sum(1 for u, v in rec["edges"] if sol["colors"][u] == sol["colors"][v])
        ok = indep == 0
        solved += int(ok)
        audits.append({
            "name": rec["name"],
            "n": rec["n"],
            "tight_k": stage.k,
            "conflicts_independent": indep,
            "rigorous_solved": ok,
            "reported_conflicts": sol["conflicts"],
        })
        tmp.unlink(missing_ok=True)
    result = {"n": stage.n, "k": stage.k, "eval_graphs": len(rows), "solved_rigorous": solved, "solved_rate": solved / max(1, len(rows)), "audits": audits}
    (out_dir / "eval_audit.json").write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--nproc", type=int, default=max(1, min(8, torch.cuda.device_count() or 1)))
    args = parser.parse_args()

    SCALE_ROOT.mkdir(parents=True, exist_ok=True)
    session = SCALE_ROOT / f"session_{int(time.time())}"
    session.mkdir(parents=True, exist_ok=True)
    history_path = session / "scaleup_history.jsonl"

    stages = [
        StageCfg(n=20, k=5, train_graphs=200, valid_graphs=60, eval_graphs=40, p=0.28, epochs=3, steps_per_epoch=600, valid_steps=100, batch_size=64, teacher_train=12, teacher_valid=6, solve_train=2, solve_valid=1, simulations_trace=64, simulations_eval=96),
        StageCfg(n=40, k=7, train_graphs=220, valid_graphs=70, eval_graphs=45, p=0.18, epochs=4, steps_per_epoch=800, valid_steps=120, batch_size=48, teacher_train=8, teacher_valid=4, solve_train=2, solve_valid=1, simulations_trace=96, simulations_eval=128),
        StageCfg(n=80, k=10, train_graphs=240, valid_graphs=80, eval_graphs=50, p=0.12, epochs=5, steps_per_epoch=1000, valid_steps=150, batch_size=32, teacher_train=6, teacher_valid=3, solve_train=2, solve_valid=1, simulations_trace=128, simulations_eval=160),
    ]

    current_ckpt: Path | None = None
    for i, st in enumerate(stages):
        stage_dir = session / f"n{st.n}"
        stage_dir.mkdir(parents=True, exist_ok=True)
        train_jsonl, valid_jsonl, eval_jsonl = build_dataset_files(st, seed=args.seed + i)

        train_trace_dir = stage_dir / "traces_train"
        valid_trace_dir = stage_dir / "traces_valid"
        build_teacher_traces(train_jsonl, train_trace_dir, "teacher", st.teacher_train, max_steps=24, seed=args.seed + 100 + i)
        build_teacher_traces(valid_jsonl, valid_trace_dir, "teacher", st.teacher_valid, max_steps=20, seed=args.seed + 200 + i)
        build_solve_traces(train_jsonl, train_trace_dir, "solve", current_ckpt, k=st.k, episodes=st.solve_train, max_steps=16, simulations=st.simulations_trace, seed=args.seed + 300 + i)
        build_solve_traces(valid_jsonl, valid_trace_dir, "solve", current_ckpt, k=st.k, episodes=st.solve_valid, max_steps=12, simulations=max(32, st.simulations_trace // 2), seed=args.seed + 400 + i)

        current_ckpt = train_stage(
            st,
            train_glob=str(train_trace_dir / "*.pt"),
            valid_glob=str(valid_trace_dir / "*.pt"),
            out_dir=stage_dir / "train_run",
            seed=args.seed + 500 + i,
            nproc=args.nproc,
            init_ckpt=current_ckpt,
        )

        eval_result = eval_stage(st, eval_jsonl, current_ckpt, stage_dir)
        row = {"stage": asdict(st), "checkpoint": str(current_ckpt), "eval_result": eval_result}
        with history_path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps({"stage_n": st.n, "k": st.k, "solved_rate": eval_result["solved_rate"], "solved_rigorous": eval_result["solved_rigorous"]}), flush=True)

    lines = [json.loads(x) for x in history_path.read_text().splitlines() if x.strip()]
    summary = {"session": str(session), "stages_done": [ln["stage"]["n"] for ln in lines], "final_checkpoint": str(current_ckpt) if current_ckpt else None, "results_by_n": {str(ln["stage"]["n"]): ln["eval_result"] for ln in lines}}
    (session / "final_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
