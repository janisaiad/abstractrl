#!/usr/bin/env python3
from __future__ import annotations

"""
Mixed curriculum orchestration for GCP small-to-large training.

Goal
----
Push the learned TRM curriculum beyond small graphs, but in a staged and
controlled way:
  Stage 0: small traces / optional pretrained checkpoint
  Stage 1: mix small traces with n=100/200 teacher + solve traces
  Stage 2: mix Stage 1 data with n=400 teacher + solve traces

This runner is designed to work with the current v3 stack:
  * gcp_trace_abstractbeam_v3.py
  * run_small_to_large_ladder_v3.py
  * hard-k controlled graph generation

It does NOT change the GCP solver itself. It only orchestrates datasets,
trace generation, fine-tuning, in-distribution audits, and optional ladder eval.
"""

import argparse
import glob
import json
import random
import subprocess
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

ROOT = Path(__file__).resolve().parent
V3_ROOT = ROOT.parent / "v3"
RUNS_ROOT = ROOT / "runs" / "gcp_trm_mixed_curriculum_v5"
DATA_ROOT = ROOT / "data" / "gcp_mixed_curriculum_v5"


@dataclass
class SizeCfg:
    n: int
    k: int
    p: float
    train_graphs: int
    valid_graphs: int
    eval_graphs: int


@dataclass
class TrainCfg:
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


DEFAULT_STAGE1_SIZES: Tuple[SizeCfg, ...] = (
    SizeCfg(n=100, k=6, p=0.22, train_graphs=120, valid_graphs=24, eval_graphs=20),
    SizeCfg(n=200, k=8, p=0.18, train_graphs=120, valid_graphs=24, eval_graphs=20),
)
DEFAULT_STAGE2_SIZES: Tuple[SizeCfg, ...] = (
    SizeCfg(n=400, k=10, p=0.14, train_graphs=100, valid_graphs=20, eval_graphs=20),
)

DEFAULT_STAGE1_TRAIN = TrainCfg(
    epochs=4,
    steps_per_epoch=1200,
    valid_steps=150,
    batch_size=48,
    teacher_train=6,
    teacher_valid=3,
    solve_train=2,
    solve_valid=1,
    simulations_trace=48,
    simulations_eval=96,
)
DEFAULT_STAGE2_TRAIN = TrainCfg(
    epochs=4,
    steps_per_epoch=1400,
    valid_steps=180,
    batch_size=32,
    teacher_train=4,
    teacher_valid=2,
    solve_train=2,
    solve_valid=1,
    simulations_trace=64,
    simulations_eval=128,
)


def run_cmd(cmd: List[str], cwd: Path) -> None:
    subprocess.run(cmd, cwd=str(cwd), check=True)


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def make_forced_k_graph(n: int, k: int, p: float, rng: random.Random, name: str) -> Dict[str, Any]:
    colors = [rng.randrange(k) for _ in range(n)]
    for c in range(min(k, n)):
        colors[c] = c
    rng.shuffle(colors)
    class_members: List[List[int]] = [[] for _ in range(k)]
    for i, c in enumerate(colors):
        class_members[c].append(i)
    edges: set[Tuple[int, int]] = set()
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
    return {
        "name": name,
        "n": n,
        "edges": [[u, v] for (u, v) in sorted(edges)],
        "solution": colors,
        "tight_k": k,
    }


def build_dataset_files(base_dir: Path, size_cfgs: Sequence[SizeCfg], seed: int, prefix: str) -> Tuple[List[Path], List[Path], List[Path]]:
    rng = random.Random(seed)
    train_files: List[Path] = []
    valid_files: List[Path] = []
    eval_files: List[Path] = []
    for sc in size_cfgs:
        stage_dir = base_dir / f"{prefix}_n{sc.n}"
        train_file = stage_dir / "train.jsonl"
        valid_file = stage_dir / "valid.jsonl"
        eval_file = stage_dir / "eval.jsonl"
        train_rows = [make_forced_k_graph(sc.n, sc.k, sc.p, rng, f"{prefix}_n{sc.n}_train_{i:04d}") for i in range(sc.train_graphs)]
        valid_rows = [make_forced_k_graph(sc.n, sc.k, sc.p, rng, f"{prefix}_n{sc.n}_valid_{i:04d}") for i in range(sc.valid_graphs)]
        eval_rows = [make_forced_k_graph(sc.n, sc.k, sc.p, rng, f"{prefix}_n{sc.n}_eval_{i:04d}") for i in range(sc.eval_graphs)]
        write_jsonl(train_file, train_rows)
        write_jsonl(valid_file, valid_rows)
        write_jsonl(eval_file, eval_rows)
        train_files.append(train_file)
        valid_files.append(valid_file)
        eval_files.append(eval_file)
    return train_files, valid_files, eval_files


def _merge_shards(patterns: Sequence[str], out_path: Path) -> int:
    rows: List[dict[str, Any]] = []
    for pat in patterns:
        for path in sorted(glob.glob(pat)):
            shard = torch.load(path, map_location="cpu", weights_only=False)
            if not isinstance(shard, list):
                raise TypeError(f"Unexpected shard type in {path}: {type(shard)}")
            rows.extend(shard)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(rows, out_path)
    return len(rows)


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
        V3_ROOT,
    )


def build_solve_traces(
    input_jsonl: Path,
    out_dir: Path,
    prefix: str,
    ckpt: Optional[Path],
    k: int,
    episodes: int,
    max_steps: int,
    simulations: int,
    seed: int,
    mcts_max_depth: int = 48,
    prune_every: int = 32,
    prune_keep_topk: int = 4,
    worker_count: Optional[int] = None,
    mcts_sim_trace: str = "aggregates",
    mcts_sim_trace_cap: int = 50000,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    tree_dir = out_dir / "mcts_trees"
    tree_dir.mkdir(parents=True, exist_ok=True)
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
        "--max-depth", str(int(mcts_max_depth)),
        "--prune-every", str(int(prune_every)),
        "--prune-keep-topk", str(int(prune_keep_topk)),
        "--search-alpha-mean", "0.75",
        "--search-beta-max", "0.25",
        "--novelty-coef", "0.05",
        "--search-mode", "collect",
        "--collect-prior-mix", "0.35",
        "--collect-prior-temp", "1.9",
        "--worker-count",
        str(
            int(worker_count)
            if worker_count is not None
            else max(1, torch.cuda.device_count() * 4 if torch.cuda.is_available() else 4)
        ),
        "--virtual-loss", "0.25",
        "--alloc-lambda-best", "1.0",
        "--alloc-lambda-uncertainty", "0.7",
        "--alloc-lambda-novelty", "0.35",
        "--train-policy-target-mode", "best_through",
        "--train-topk-k", "3",
        "--train-lse-beta", "4.0",
        "--track-distinct-terminals",
        "--mcts-tree-dump-dir", str(tree_dir),
        "--mcts-sim-trace",
        str(mcts_sim_trace),
        "--mcts-sim-trace-cap",
        str(int(mcts_sim_trace_cap)),
        "--seed", str(seed),
        "--log-every-records", "50",
    ]
    if ckpt is not None and ckpt.exists():
        cmd.extend(["--ckpt", str(ckpt)])
    run_cmd(cmd, V3_ROOT)


def train_mixed(
    train_pt: Path,
    valid_pt: Path,
    out_dir: Path,
    cfg: TrainCfg,
    seed: int,
    init_ckpt: Optional[Path] = None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python", "gcp_trace_abstractbeam_v3.py", "train",
        "--train", str(train_pt),
        "--valid", str(valid_pt),
        "--out-dir", str(out_dir),
        "--epochs", str(cfg.epochs),
        "--steps-per-epoch", str(cfg.steps_per_epoch),
        "--valid-steps", str(cfg.valid_steps),
        "--batch-size", str(cfg.batch_size),
        "--num-workers", str(max(1, min(4, torch.cuda.device_count() or 1))),
        "--seed", str(seed),
        "--amp",
    ]
    if init_ckpt is not None and init_ckpt.exists():
        cmd.extend(["--init-ckpt", str(init_ckpt)])
    run_cmd(cmd, V3_ROOT)
    ckpt = out_dir / "model-best.pt"
    if not ckpt.exists():
        ckpt = out_dir / "model-last.pt"
    if not ckpt.exists():
        raise RuntimeError(f"No checkpoint found in {out_dir}")
    return ckpt


def eval_indistribution(
    eval_files: Sequence[Path],
    ckpt: Path,
    size_cfgs: Sequence[SizeCfg],
    out_dir: Path,
    mcts_sim_trace: str = "aggregates",
    mcts_sim_trace_cap: int = 50000,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    results: List[Dict[str, Any]] = []
    for path, sc in zip(eval_files, size_cfgs):
        rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
        solved = 0
        audits: List[Dict[str, Any]] = []
        for rec in rows:
            tmp = out_dir / f"{rec['name']}.json"
            tree_dir = out_dir / "mcts_trees" / f"n{sc.n}"
            tree_dir.mkdir(parents=True, exist_ok=True)
            tree_dump = tree_dir / f"{rec['name']}.mcts_tree.json"
            tmp.write_text(json.dumps({"name": rec["name"], "n": rec["n"], "edges": rec["edges"]}))
            out = subprocess.check_output(
                [
                    "python", "gcp_trace_abstractbeam_v3.py", "solve",
                    "--input", str(tmp),
                    "--ckpt", str(ckpt),
                    "--k", str(sc.k),
                    "--simulations", str(DEFAULT_STAGE1_TRAIN.simulations_eval if sc.n < 400 else DEFAULT_STAGE2_TRAIN.simulations_eval),
                    "--max-depth", "128",
                    "--search-alpha-mean", "0.75",
                    "--search-beta-max", "0.25",
                    "--search-mode", "infer",
                    "--worker-count", str(max(1, torch.cuda.device_count() * 4 if torch.cuda.is_available() else 4)),
                    "--mcts-tree-dump", str(tree_dump),
                    "--mcts-sim-trace",
                    str(mcts_sim_trace),
                    "--mcts-sim-trace-cap",
                    str(int(mcts_sim_trace_cap)),
                ],
                cwd=str(V3_ROOT), text=True,
            )
            sol = json.loads(out)
            indep = sum(1 for u, v in rec["edges"] if sol["colors"][u] == sol["colors"][v])
            ok = indep == 0
            solved += int(ok)
            audits.append(
                {
                    "name": rec["name"],
                    "n": rec["n"],
                    "tight_k": sc.k,
                    "conflicts_independent": indep,
                    "rigorous_solved": ok,
                    "reported_conflicts": sol["conflicts"],
                }
            )
            tmp.unlink(missing_ok=True)
        block = {
            "n": sc.n,
            "k": sc.k,
            "eval_graphs": len(rows),
            "solved_rigorous": solved,
            "solved_rate": solved / max(1, len(rows)),
            "audits": audits,
        }
        (out_dir / f"eval_n{sc.n}.json").write_text(json.dumps(block, indent=2))
        results.append(block)
    summary = {"blocks": results}
    (out_dir / "eval_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def run_ladder_eval(
    session_dir: Path,
    small_train_globs: Sequence[str],
    small_valid_globs: Sequence[str],
    small_ckpt: Path,
    curriculum_ckpt: Path,
    timeout_sec: float,
    profile_every: int,
) -> Path:
    out_session_name = f"ladder_eval_{int(time.time())}"
    cmd = [
        "python", "run_small_to_large_ladder_v3.py",
        "--session", out_session_name,
        "--small-train-globs", ",".join(small_train_globs),
        "--small-valid-globs", ",".join(small_valid_globs),
        "--curriculum-ckpt", str(curriculum_ckpt),
        "--small-pretrained-ckpt", str(small_ckpt),
        "--skip-small-train",
        "--sizes", "100,200,400",
        "--per-size", "10",
        "--budget-ladder", "12:24,32:48,64:96",
        "--timeout-sec", str(timeout_sec),
        "--profile-every", str(profile_every),
        "--baseline-method", "fixed_tabu_recolor",
        "--penalty-mode", "last_profile_plus_const",
        "--solver-mode", "inprocess",
        "--store-mcts-trees",
    ]
    run_cmd(cmd, V3_ROOT)
    return V3_ROOT / "runs" / "gcp_trm_scaleup_v3" / out_session_name


def _cfgs_to_json(cfgs: Sequence[SizeCfg]) -> List[Dict[str, Any]]:
    return [asdict(c) for c in cfgs]


def main() -> None:
    ap = argparse.ArgumentParser(description="Mixed curriculum orchestration: small pretrain -> n100/200 finetune -> n400 finetune")
    ap.add_argument("--session", type=str, default="")
    ap.add_argument("--seed", type=int, default=20260422)
    ap.add_argument("--small-train-globs", type=str, required=True, help="Comma-separated V2/V3 small-trace train globs (e.g. n20+n40)")
    ap.add_argument("--small-valid-globs", type=str, required=True, help="Comma-separated V2/V3 small-trace valid globs")
    ap.add_argument("--small-pretrained-ckpt", type=str, default="", help="Optional pretrained small checkpoint (e.g. big_train_1776710200/model-best.pt)")
    ap.add_argument("--train-small-if-missing", action="store_true", help="If no small ckpt is given, train a small model from merged small traces")
    ap.add_argument("--run-ladder-eval", action="store_true", help="Run the v3 ladder at the end with final curriculum ckpt")
    ap.add_argument("--timeout-sec", type=float, default=60.0)
    ap.add_argument("--profile-every", type=int, default=4)
    # Stage overrides
    ap.add_argument("--stage1-epochs", type=int, default=DEFAULT_STAGE1_TRAIN.epochs)
    ap.add_argument("--stage1-steps", type=int, default=DEFAULT_STAGE1_TRAIN.steps_per_epoch)
    ap.add_argument("--stage1-valid-steps", type=int, default=DEFAULT_STAGE1_TRAIN.valid_steps)
    ap.add_argument("--stage1-batch", type=int, default=DEFAULT_STAGE1_TRAIN.batch_size)
    ap.add_argument("--stage2-epochs", type=int, default=DEFAULT_STAGE2_TRAIN.epochs)
    ap.add_argument("--stage2-steps", type=int, default=DEFAULT_STAGE2_TRAIN.steps_per_epoch)
    ap.add_argument("--stage2-valid-steps", type=int, default=DEFAULT_STAGE2_TRAIN.valid_steps)
    ap.add_argument("--stage2-batch", type=int, default=DEFAULT_STAGE2_TRAIN.batch_size)
    ap.add_argument(
        "--deep-mcts-traces",
        action="store_true",
        help="Use deeper/wider MCTS for solve-trace collection (more simulations, higher max-depth, gentler pruning, fewer parallel workers for stability).",
    )
    ap.add_argument(
        "--trace-mcts-simulations",
        type=int,
        default=0,
        help="Override MCTS simulations for solve-trace collection when --deep-mcts-traces is set (0 => 320 stage1, 384 stage2).",
    )
    ap.add_argument("--trace-mcts-max-depth", type=int, default=0, help="With --deep-mcts-traces: max MCTS depth (0 => 96).")
    ap.add_argument("--trace-mcts-prune-every", type=int, default=0, help="With --deep-mcts-traces: prune period (0 => 128).")
    ap.add_argument("--trace-mcts-prune-keep-topk", type=int, default=0, help="With --deep-mcts-traces: keep-topk when pruning (0 => 8).")
    ap.add_argument("--trace-mcts-workers", type=int, default=0, help="With --deep-mcts-traces: worker count (0 => 2).")
    ap.add_argument(
        "--mcts-sim-trace",
        type=str,
        default="aggregates",
        choices=["off", "aggregates", "full"],
        help="Per-simulation MCTS instrumentation in dumped trees (solve-traces + in-dist eval).",
    )
    ap.add_argument("--mcts-sim-trace-cap", type=int, default=50000, help="Cap on per-tree simulation summaries / full paths.")
    args = ap.parse_args()

    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    session_name = str(args.session).strip() or f"mixed_curriculum_{int(time.time())}"
    session_dir = RUNS_ROOT / session_name
    session_dir.mkdir(parents=True, exist_ok=True)

    small_train_globs = [x.strip() for x in str(args.small_train_globs).split(",") if x.strip()]
    small_valid_globs = [x.strip() for x in str(args.small_valid_globs).split(",") if x.strip()]
    if not small_train_globs or not small_valid_globs:
        raise ValueError("Both --small-train-globs and --small-valid-globs are required")

    # Stage-0: merge small traces
    base_dir = session_dir / "small_base"
    small_train_pt = base_dir / "small_train_merged.pt"
    small_valid_pt = base_dir / "small_valid_merged.pt"
    small_train_rows = _merge_shards(small_train_globs, small_train_pt)
    small_valid_rows = _merge_shards(small_valid_globs, small_valid_pt)

    base_ckpt: Optional[Path] = Path(str(args.small_pretrained_ckpt)).expanduser() if str(args.small_pretrained_ckpt).strip() else None
    if base_ckpt is not None and not base_ckpt.exists():
        raise FileNotFoundError(base_ckpt)
    if base_ckpt is None and args.train_small_if_missing:
        small_cfg = TrainCfg(
            epochs=4,
            steps_per_epoch=1200,
            valid_steps=150,
            batch_size=64,
            teacher_train=0,
            teacher_valid=0,
            solve_train=0,
            solve_valid=0,
            simulations_trace=0,
            simulations_eval=0,
        )
        base_ckpt = train_mixed(
            small_train_pt,
            small_valid_pt,
            session_dir / "small_pretrain",
            small_cfg,
            seed=int(args.seed) + 1,
            init_ckpt=None,
        )
    if base_ckpt is None:
        raise ValueError("Provide --small-pretrained-ckpt or use --train-small-if-missing")

    stage1_train_cfg = TrainCfg(
        epochs=int(args.stage1_epochs),
        steps_per_epoch=int(args.stage1_steps),
        valid_steps=int(args.stage1_valid_steps),
        batch_size=int(args.stage1_batch),
        teacher_train=DEFAULT_STAGE1_TRAIN.teacher_train,
        teacher_valid=DEFAULT_STAGE1_TRAIN.teacher_valid,
        solve_train=DEFAULT_STAGE1_TRAIN.solve_train,
        solve_valid=DEFAULT_STAGE1_TRAIN.solve_valid,
        simulations_trace=DEFAULT_STAGE1_TRAIN.simulations_trace,
        simulations_eval=DEFAULT_STAGE1_TRAIN.simulations_eval,
    )
    stage2_train_cfg = TrainCfg(
        epochs=int(args.stage2_epochs),
        steps_per_epoch=int(args.stage2_steps),
        valid_steps=int(args.stage2_valid_steps),
        batch_size=int(args.stage2_batch),
        teacher_train=DEFAULT_STAGE2_TRAIN.teacher_train,
        teacher_valid=DEFAULT_STAGE2_TRAIN.teacher_valid,
        solve_train=DEFAULT_STAGE2_TRAIN.solve_train,
        solve_valid=DEFAULT_STAGE2_TRAIN.solve_valid,
        simulations_trace=DEFAULT_STAGE2_TRAIN.simulations_trace,
        simulations_eval=DEFAULT_STAGE2_TRAIN.simulations_eval,
    )

    trace_tm: Dict[str, Any]
    trace_s1_simulations: Optional[int]
    trace_s2_simulations: Optional[int]
    if bool(args.deep_mcts_traces):
        trace_tm = {
            "mcts_max_depth": int(args.trace_mcts_max_depth or 96),
            "prune_every": int(args.trace_mcts_prune_every or 128),
            "prune_keep_topk": int(args.trace_mcts_prune_keep_topk or 8),
            "worker_count": int(args.trace_mcts_workers or 2),
        }
        trace_s1_simulations = int(args.trace_mcts_simulations) if int(args.trace_mcts_simulations) > 0 else 320
        trace_s2_simulations = int(args.trace_mcts_simulations) if int(args.trace_mcts_simulations) > 0 else 384
    else:
        trace_tm = {"mcts_max_depth": 48, "prune_every": 32, "prune_keep_topk": 4, "worker_count": None}
        trace_s1_simulations = None
        trace_s2_simulations = None

    # Stage-1: n=100/200
    stage1_dir = session_dir / "stage1_100_200"
    s1_train_jsonl, s1_valid_jsonl, s1_eval_jsonl = build_dataset_files(stage1_dir / "data", DEFAULT_STAGE1_SIZES, seed=int(args.seed) + 11, prefix="stage1")
    s1_train_trace_patterns: List[str] = []
    s1_valid_trace_patterns: List[str] = []
    for idx, sc in enumerate(DEFAULT_STAGE1_SIZES):
        tr_dir = stage1_dir / f"n{sc.n}" / "traces_train"
        va_dir = stage1_dir / f"n{sc.n}" / "traces_valid"
        build_teacher_traces(s1_train_jsonl[idx], tr_dir, "teacher", stage1_train_cfg.teacher_train, max_steps=24, seed=int(args.seed) + 100 + idx)
        build_teacher_traces(s1_valid_jsonl[idx], va_dir, "teacher", stage1_train_cfg.teacher_valid, max_steps=20, seed=int(args.seed) + 200 + idx)
        build_solve_traces(
            s1_train_jsonl[idx],
            tr_dir,
            "solve",
            base_ckpt,
            k=sc.k,
            episodes=stage1_train_cfg.solve_train,
            max_steps=16,
            simulations=int(trace_s1_simulations) if trace_s1_simulations is not None else int(stage1_train_cfg.simulations_trace),
            seed=int(args.seed) + 300 + idx,
            mcts_sim_trace=str(args.mcts_sim_trace),
            mcts_sim_trace_cap=int(args.mcts_sim_trace_cap),
            **trace_tm,
        )
        build_solve_traces(
            s1_valid_jsonl[idx],
            va_dir,
            "solve",
            base_ckpt,
            k=sc.k,
            episodes=stage1_train_cfg.solve_valid,
            max_steps=12,
            simulations=max(32, (int(trace_s1_simulations) if trace_s1_simulations is not None else int(stage1_train_cfg.simulations_trace)) // 2),
            seed=int(args.seed) + 400 + idx,
            mcts_sim_trace=str(args.mcts_sim_trace),
            mcts_sim_trace_cap=int(args.mcts_sim_trace_cap),
            **trace_tm,
        )
        s1_train_trace_patterns.append(str(tr_dir / "*.pt"))
        s1_valid_trace_patterns.append(str(va_dir / "*.pt"))

    s1_train_mix = session_dir / "merged" / "stage1_train_mixed.pt"
    s1_valid_mix = session_dir / "merged" / "stage1_valid_mixed.pt"
    _merge_shards([str(small_train_pt)] + s1_train_trace_patterns, s1_train_mix)
    _merge_shards([str(small_valid_pt)] + s1_valid_trace_patterns, s1_valid_mix)
    stage1_ckpt = train_mixed(s1_train_mix, s1_valid_mix, stage1_dir / "train_run", stage1_train_cfg, seed=int(args.seed) + 500, init_ckpt=base_ckpt)
    stage1_eval = eval_indistribution(
        s1_eval_jsonl,
        stage1_ckpt,
        DEFAULT_STAGE1_SIZES,
        stage1_dir / "eval",
        mcts_sim_trace=str(args.mcts_sim_trace),
        mcts_sim_trace_cap=int(args.mcts_sim_trace_cap),
    )

    # Stage-2: n=400
    stage2_dir = session_dir / "stage2_400"
    s2_train_jsonl, s2_valid_jsonl, s2_eval_jsonl = build_dataset_files(stage2_dir / "data", DEFAULT_STAGE2_SIZES, seed=int(args.seed) + 21, prefix="stage2")
    s2_train_trace_patterns: List[str] = []
    s2_valid_trace_patterns: List[str] = []
    for idx, sc in enumerate(DEFAULT_STAGE2_SIZES):
        tr_dir = stage2_dir / f"n{sc.n}" / "traces_train"
        va_dir = stage2_dir / f"n{sc.n}" / "traces_valid"
        build_teacher_traces(s2_train_jsonl[idx], tr_dir, "teacher", stage2_train_cfg.teacher_train, max_steps=24, seed=int(args.seed) + 700 + idx)
        build_teacher_traces(s2_valid_jsonl[idx], va_dir, "teacher", stage2_train_cfg.teacher_valid, max_steps=20, seed=int(args.seed) + 800 + idx)
        build_solve_traces(
            s2_train_jsonl[idx],
            tr_dir,
            "solve",
            stage1_ckpt,
            k=sc.k,
            episodes=stage2_train_cfg.solve_train,
            max_steps=16,
            simulations=int(trace_s2_simulations) if trace_s2_simulations is not None else int(stage2_train_cfg.simulations_trace),
            seed=int(args.seed) + 900 + idx,
            mcts_sim_trace=str(args.mcts_sim_trace),
            mcts_sim_trace_cap=int(args.mcts_sim_trace_cap),
            **trace_tm,
        )
        build_solve_traces(
            s2_valid_jsonl[idx],
            va_dir,
            "solve",
            stage1_ckpt,
            k=sc.k,
            episodes=stage2_train_cfg.solve_valid,
            max_steps=12,
            simulations=max(32, (int(trace_s2_simulations) if trace_s2_simulations is not None else int(stage2_train_cfg.simulations_trace)) // 2),
            seed=int(args.seed) + 1000 + idx,
            mcts_sim_trace=str(args.mcts_sim_trace),
            mcts_sim_trace_cap=int(args.mcts_sim_trace_cap),
            **trace_tm,
        )
        s2_train_trace_patterns.append(str(tr_dir / "*.pt"))
        s2_valid_trace_patterns.append(str(va_dir / "*.pt"))

    s2_train_mix = session_dir / "merged" / "stage2_train_mixed.pt"
    s2_valid_mix = session_dir / "merged" / "stage2_valid_mixed.pt"
    _merge_shards([str(small_train_pt)] + s1_train_trace_patterns + s2_train_trace_patterns, s2_train_mix)
    _merge_shards([str(small_valid_pt)] + s1_valid_trace_patterns + s2_valid_trace_patterns, s2_valid_mix)
    final_ckpt = train_mixed(s2_train_mix, s2_valid_mix, stage2_dir / "train_run", stage2_train_cfg, seed=int(args.seed) + 1100, init_ckpt=stage1_ckpt)
    stage2_eval = eval_indistribution(
        s2_eval_jsonl,
        final_ckpt,
        DEFAULT_STAGE2_SIZES,
        stage2_dir / "eval",
        mcts_sim_trace=str(args.mcts_sim_trace),
        mcts_sim_trace_cap=int(args.mcts_sim_trace_cap),
    )

    summary: Dict[str, Any] = {
        "session_dir": str(session_dir),
        "seed": int(args.seed),
        "small_train_rows": int(small_train_rows),
        "small_valid_rows": int(small_valid_rows),
        "stage0": {"base_ckpt": str(base_ckpt)},
        "stage1": {
            "sizes": _cfgs_to_json(DEFAULT_STAGE1_SIZES),
            "checkpoint": str(stage1_ckpt),
            "eval": stage1_eval,
            "train_cfg": asdict(stage1_train_cfg),
        },
        "stage2": {
            "sizes": _cfgs_to_json(DEFAULT_STAGE2_SIZES),
            "checkpoint": str(final_ckpt),
            "eval": stage2_eval,
            "train_cfg": asdict(stage2_train_cfg),
        },
        "ladder_eval_session": None,
    }

    if args.run_ladder_eval:
        ladder_session = run_ladder_eval(
            session_dir,
            small_train_globs,
            small_valid_globs,
            base_ckpt,
            final_ckpt,
            timeout_sec=float(args.timeout_sec),
            profile_every=int(args.profile_every),
        )
        summary["ladder_eval_session"] = str(ladder_session)

    (session_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
