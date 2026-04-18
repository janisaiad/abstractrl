#!/usr/bin/env python3
"""
Run GCP TRM + MCTS pipeline for ~8h with the revised protocol:

  * Train/valid: k-colorable random graphs (with solution in JSON for trace building only).
  * Traces: bounded corruptions / max-steps so trace generation usually finishes well inside budget.
  * Train: GPU + AMP, d_model=512, time-shaped epochs/steps from remaining budget.
  * OOD eval: tight k, JSON without solution; instances in a **medium band** (greedy-k conflicts in [min,max])
    so MCTS can plausibly reach zero; model loaded once (no per-graph subprocess reload).
  * Report: final JSON + progress.jsonl (flushed).
  * mine-macros on train trace shards at the end.

Launch (from this directory):

  mkdir -p runs/gcp_trm_scaleup
  nohup env PYTHONUNBUFFERED=1 python -u gcp_run8h_perfect.py \\
    --budget-hours 8 > runs/gcp_trm_scaleup/nohup_8h_perfect.log 2>&1 &
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

import gcp_trace_abstractbeam as gcp
from gcp_hard_benchmark import best_greedy_k_conflicts, gen_k_colorable_graph, sample_hard_instance, to_public_json

ROOT = Path(__file__).resolve().parent


def log(progress_path: Path, t0: float, event: str, **kwargs) -> None:
    rec = {"wall_s": round(time.time() - t0, 3), "event": event, **kwargs}
    line = json.dumps(rec, ensure_ascii=False) + "\n"
    print(line, end="", flush=True)
    with progress_path.open("a") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def indep_conflicts(edges: list[list[int]], colors: list[int]) -> int:
    return sum(1 for u, v in edges if colors[int(u)] == colors[int(v)])


def make_solve_ns(simulations: int, max_depth: int) -> argparse.Namespace:
    return argparse.Namespace(
        cpuct=1.25,
        gamma=1.0,
        simulations=int(simulations),
        max_depth=int(max_depth),
        prune_every=24,
        prune_min_visits=4,
        prune_keep_topk=6,
        confidence_beta=1.5,
        action_budget=128,
        exact_patch_limit=12,
        cpu=False,
    )


def run_py(args: list[str], cwd: Path, t0: float, progress_path: Path, label: str) -> None:
    cmd = ["python", "-u", *args]
    log(progress_path, t0, "subprocess_start", label=label, cmd=" ".join(cmd))
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    subprocess.check_call(cmd, cwd=str(cwd), env=env)
    log(progress_path, t0, "subprocess_done", label=label)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget-hours", type=float, default=8.0)
    ap.add_argument(
        "--session",
        type=str,
        default="",
        help="Optional session dir name under runs/gcp_trm_scaleup/ (default: session_8h_perfect_<unix>)",
    )
    ap.add_argument(
        "--no-phase2",
        action="store_true",
        help="Skip second training stage (d_model=512 from scratch after phase 256)",
    )
    args = ap.parse_args()

    t0 = time.time()
    budget_s = float(args.budget_hours) * 3600.0
    reserve_end_s = 25 * 60.0
    reserve_eval_floor_s = 35 * 60.0

    def elapsed() -> float:
        return time.time() - t0

    def remaining() -> float:
        return max(0.0, budget_s - elapsed())

    base = ROOT / "runs" / "gcp_trm_scaleup"
    base.mkdir(parents=True, exist_ok=True)
    session_name = str(args.session).strip() or f"session_8h_perfect_{int(time.time())}"
    session = base / session_name
    data = session / "data"
    traces = session / "traces"
    train_phase1 = session / "train_phase1"
    train_phase2 = session / "train_phase2"
    report_path = session / "report_8h_perfect.json"
    progress_path = session / "progress.jsonl"
    for p in (data, traces, train_phase1):
        p.mkdir(parents=True, exist_ok=True)
    if progress_path.exists():
        progress_path.unlink()

    rng = random.Random(20260418)
    log(progress_path, t0, "start", session=str(session), budget_hours=float(args.budget_hours))

    train: list[dict] = []
    valid: list[dict] = []
    for i in range(140):
        train.append(gen_k_colorable_graph(rng, 100, 16, 0.088, f"train_n100_{i:04d}"))
    for i in range(140):
        train.append(gen_k_colorable_graph(rng, 200, 24, 0.052, f"train_n200_{i:04d}"))
    for i in range(100):
        train.append(gen_k_colorable_graph(rng, 400, 32, 0.028, f"train_n400_{i:04d}"))
    for i in range(28):
        valid.append(gen_k_colorable_graph(rng, 100, 16, 0.088, f"valid_n100_{i:04d}"))
    for i in range(28):
        valid.append(gen_k_colorable_graph(rng, 200, 24, 0.052, f"valid_n200_{i:04d}"))
    for i in range(20):
        valid.append(gen_k_colorable_graph(rng, 400, 32, 0.028, f"valid_n400_{i:04d}"))

    train_jsonl = data / "train.jsonl"
    valid_jsonl = data / "valid.jsonl"
    write_jsonl(train_jsonl, train)
    write_jsonl(valid_jsonl, valid)
    log(
        progress_path,
        t0,
        "datasets_written",
        train=len(train),
        valid=len(valid),
        remaining_s=round(remaining(), 1),
    )

    run_py(
        [
            "gcp_trace_abstractbeam.py",
            "build-traces",
            "--input",
            str(train_jsonl),
            "--out-dir",
            str(traces / "train"),
            "--prefix",
            "trace",
            "--samples-per-shard",
            "8192",
            "--corruptions-per-graph",
            "10",
            "--max-steps",
            "72",
            "--seed",
            "101",
            "--log-every-records",
            "40",
        ],
        ROOT,
        t0,
        progress_path,
        "build_traces_train",
    )
    run_py(
        [
            "gcp_trace_abstractbeam.py",
            "build-traces",
            "--input",
            str(valid_jsonl),
            "--out-dir",
            str(traces / "valid"),
            "--prefix",
            "trace",
            "--samples-per-shard",
            "8192",
            "--corruptions-per-graph",
            "6",
            "--max-steps",
            "72",
            "--seed",
            "103",
            "--log-every-records",
            "14",
        ],
        ROOT,
        t0,
        progress_path,
        "build_traces_valid",
    )

    rem_train = remaining() - reserve_end_s - reserve_eval_floor_s
    rem_train = max(rem_train, 45 * 60.0)
    min_spe = 800
    max_spe = 1000
    epochs = min(26, max(14, int(rem_train / 1350.0)))
    spe = min(max_spe, max(min_spe, int(rem_train / (epochs * 2.05))))
    vsteps = min(200, max(120, min(spe // 5, 200)))
    nw = 6 if torch.cuda.is_available() else 2
    log(
        progress_path,
        t0,
        "train_shape_phase1",
        d_model=256,
        epochs=epochs,
        steps_per_epoch=spe,
        valid_steps=vsteps,
        batch_size=96,
        min_steps_per_epoch=min_spe,
        num_workers=nw,
        cuda=torch.cuda.is_available(),
        remaining_before_train_s=round(remaining(), 1),
    )

    run_py(
        [
            "gcp_trace_abstractbeam.py",
            "train",
            "--train",
            str(traces / "train" / "trace-*.pt"),
            "--valid",
            str(traces / "valid" / "trace-*.pt"),
            "--out-dir",
            str(train_phase1),
            "--epochs",
            str(epochs),
            "--batch-size",
            "96",
            "--steps-per-epoch",
            str(spe),
            "--valid-steps",
            str(vsteps),
            "--d-model",
            "256",
            "--lr",
            "2e-4",
            "--lr-scheduler",
            "cosine",
            "--lr-min",
            "1e-6",
            "--num-workers",
            str(nw),
            "--amp",
        ],
        ROOT,
        t0,
        progress_path,
        "train_phase1_d256",
    )

    ckpt_p1 = train_phase1 / "model-best.pt"
    log(progress_path, t0, "train_phase1_done", checkpoint=str(ckpt_p1), exists=ckpt_p1.is_file())

    ckpt = ckpt_p1
    train_hparams_report: dict = {
        "phase1": {
            "d_model": 256,
            "epochs": epochs,
            "steps_per_epoch": spe,
            "valid_steps": vsteps,
            "lr": 2e-4,
            "lr_scheduler": "cosine",
            "lr_min": 1e-6,
            "batch_size": 96,
        },
        "phase2": None,
    }

    if (not args.no_phase2) and remaining() > 55 * 60 and ckpt_p1.is_file():
        train_phase2.mkdir(parents=True, exist_ok=True)
        epochs2 = min(12, max(8, epochs // 2))
        rem2 = remaining() - reserve_end_s - 20 * 60
        spe2 = min(max_spe, max(min_spe, int(max(rem2, 3600) / max(epochs2 * 1.8, 1))))
        vsteps2 = min(200, max(100, spe2 // 5))
        log(
            progress_path,
            t0,
            "train_shape_phase2",
            d_model=512,
            epochs=epochs2,
            steps_per_epoch=spe2,
            remaining_s=round(remaining(), 1),
        )
        run_py(
            [
                "gcp_trace_abstractbeam.py",
                "train",
                "--train",
                str(traces / "train" / "trace-*.pt"),
                "--valid",
                str(traces / "valid" / "trace-*.pt"),
                "--out-dir",
                str(train_phase2),
                "--epochs",
                str(epochs2),
                "--batch-size",
                "96",
                "--steps-per-epoch",
                str(spe2),
                "--valid-steps",
                str(vsteps2),
                "--d-model",
                "512",
                "--lr",
                "5e-5",
                "--lr-scheduler",
                "cosine",
                "--lr-min",
                "1e-6",
                "--num-workers",
                str(nw),
                "--amp",
            ],
            ROOT,
            t0,
            progress_path,
            "train_phase2_d512",
        )
        ckpt_p2 = train_phase2 / "model-best.pt"
        if ckpt_p2.is_file():
            ckpt = ckpt_p2
        train_hparams_report["phase2"] = {
            "d_model": 512,
            "epochs": epochs2,
            "steps_per_epoch": spe2,
            "valid_steps": vsteps2,
            "lr": 5e-5,
            "lr_scheduler": "cosine",
            "lr_min": 1e-6,
            "batch_size": 96,
        }
        log(progress_path, t0, "train_phase2_done", checkpoint=str(ckpt), used_phase2=ckpt_p2.is_file())
    else:
        log(
            progress_path,
            t0,
            "train_phase2_skipped",
            reason="--no-phase2" if args.no_phase2 else "time or missing_phase1",
        )

    log(progress_path, t0, "train_all_done", eval_checkpoint=str(ckpt), exists=ckpt.is_file())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = gcp.load_model_checkpoint(str(ckpt), device)
    model.eval()
    log(progress_path, t0, "ood_model_loaded", device=str(device))

    # Medium band: greedy-k conflicts not trivial (>= min_c) but not thousands (<= max_c).
    # Lower p_cross than before; higher sim/depth; in-process solve (one model load).
    cfgs_template = [
        {"n": 220, "k": 8, "p": 0.085, "min_c": 4, "max_c": 180, "m": 8, "sim": 128, "depth": 180, "max_tries": 550},
        {"n": 420, "k": 9, "p": 0.048, "min_c": 8, "max_c": 300, "m": 6, "sim": 112, "depth": 200, "max_tries": 650},
        {"n": 750, "k": 10, "p": 0.032, "min_c": 12, "max_c": 450, "m": 5, "sim": 96, "depth": 220, "max_tries": 750},
        {"n": 1200, "k": 11, "p": 0.022, "min_c": 18, "max_c": 650, "m": 4, "sim": 88, "depth": 240, "max_tries": 850},
    ]
    all_results: list[dict] = []

    for cfg in cfgs_template:
        if remaining() < reserve_end_s + 120:
            log(progress_path, t0, "skip_ood_block_low_time", n=cfg["n"], remaining_s=round(remaining(), 1))
            break
        model_solved = 0
        model_wins = 0
        rows: list[dict] = []
        t_block = time.time()
        m_cap = int(cfg["m"])
        if remaining() < 55 * 60:
            m_cap = max(2, m_cap - 2)
        if remaining() < 40 * 60:
            m_cap = max(2, m_cap - 1)

        for i in range(m_cap):
            if remaining() < reserve_end_s + 90:
                log(progress_path, t0, "ood_early_stop_time", n=cfg["n"], done=i)
                break
            sim = int(cfg["sim"])
            depth = int(cfg["depth"])
            if remaining() < 50 * 60:
                sim = max(64, sim - 16)
                depth = max(160, depth - 32)

            hid = sample_hard_instance(
                rng,
                int(cfg["n"]),
                int(cfg["k"]),
                float(cfg["p"]),
                f"hard_ood_n{cfg['n']}_{i:03d}",
                max_tries=min(int(cfg["max_tries"]), 200 + 50 * i),
                min_greedy_conflicts=int(cfg["min_c"]),
                max_greedy_conflicts=int(cfg["max_c"]),
                num_random_orders=8,
            )
            if hid is None:
                rows.append({"error": "no_hard_instance_found", "n": cfg["n"], "k": cfg["k"], "p": cfg["p"]})
                continue
            pub = to_public_json(hid)
            kk = int(hid["tight_k"])
            rec = gcp.GraphRecord(
                name=str(pub["name"]),
                n=int(pub["n"]),
                edges=np.asarray(pub["edges"], dtype=np.int64),
                solution=None,
                metadata={},
            )
            graph = rec.to_runtime()
            bl_conf, _ = best_greedy_k_conflicts(graph, kk, rng, num_random_orders=8)

            solve_ns = make_solve_ns(sim, depth)
            pred = gcp.solve_instance(graph, kk, model, device, {}, solve_ns)
            conf_m = int(pred["conflicts"])
            ok = conf_m == 0
            model_solved += int(ok)
            model_wins += int(ok and bl_conf > 0)
            rows.append(
                {
                    "name": hid["name"],
                    "n": cfg["n"],
                    "tight_k": kk,
                    "m": len(hid["edges"]),
                    "greedy_band": [int(cfg["min_c"]), int(cfg["max_c"])],
                    "baseline_greedy_k_conflicts": int(bl_conf),
                    "model_conflicts": int(conf_m),
                    "model_solved_rigorous": bool(ok),
                    "simulations": sim,
                    "max_depth": depth,
                }
            )

        good = len([r for r in rows if "error" not in r])
        denom = max(1, good)
        mean_bl = sum(int(r["baseline_greedy_k_conflicts"]) for r in rows if "error" not in r) / denom
        block = {
            "n": cfg["n"],
            "k_gen": cfg["k"],
            "p_cross": cfg["p"],
            "greedy_conflict_band": [int(cfg["min_c"]), int(cfg["max_c"])],
            "graphs_target": m_cap,
            "graphs_hard_ok": good,
            "elapsed_sec": time.time() - t_block,
            "model_solved": model_solved,
            "model_rate": model_solved / denom,
            "mean_baseline_greedy_k_conflicts": mean_bl,
            "model_wins_vs_hard_greedy_k": model_wins,
            "rows": rows,
        }
        all_results.append(block)
        log(
            progress_path,
            t0,
            "ood_block_done",
            n=cfg["n"],
            model=f"{model_solved}/{denom}",
            wins_vs_greedy_k=f"{model_wins}/{denom}",
            remaining_s=round(remaining(), 1),
        )

    macros_out = session / "mined_macros_train.json"
    try:
        run_py(
            [
                "gcp_trace_abstractbeam.py",
                "mine-macros",
                "--trace",
                str(traces / "train" / "trace-*.pt"),
                "--out",
                str(macros_out),
                "--min-support",
                "250",
                "--max-len",
                "3",
                "--top-k",
                "16",
            ],
            ROOT,
            t0,
            progress_path,
            "mine_macros",
        )
    except subprocess.CalledProcessError as e:
        log(progress_path, t0, "mine_macros_failed", error=str(e))

    report = {
        "session": str(session),
        "budget_hours": float(args.budget_hours),
        "protocol": {
            "train_k_colorable_with_solution_in_jsonl": True,
            "solve_uses_explicit_tight_k": True,
            "eval_json_omits_solution": True,
            "ood_greedy_k_conflict_band": "min/max on multi-order greedy_k (reachable repair)",
            "ood_rejection_until_greedy_k_has_conflicts": True,
            "baseline_is_multi_order_greedy_k_assignment": True,
            "ood_eval_in_process_single_model_load": True,
            "train_amp_cuda_when_available": True,
            "cudnn_benchmark_on_cuda": True,
            "monitor_ce_valid_minus_train_in_train_log": True,
            "phase1_d_model_256_then_optional_phase2_d512": True,
        },
        "wall_clock_sec": elapsed(),
        "checkpoint_eval": str(ckpt),
        "checkpoint_phase1": str(ckpt_p1),
        "checkpoint_phase2": str(train_phase2 / "model-best.pt") if train_phase2.exists() else "",
        "train_hparams": train_hparams_report,
        "macros": str(macros_out),
        "results_ood_hard": all_results,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log(progress_path, t0, "done", report=str(report_path), wall_clock_sec=round(elapsed(), 1))


if __name__ == "__main__":
    main()
