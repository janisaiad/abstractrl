#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

import gcp_trace_abstractbeam_v3 as gcp
import gcp_hard_benchmark_v3 as hard

ROOT = Path(__file__).resolve().parent
RUNS_ROOT = ROOT / "runs" / "gcp_trm_scaleup_v3"


@dataclass
class BudgetCfg:
    simulations: int
    max_depth: int

    @property
    def tag(self) -> str:
        return f"{self.simulations}:{self.max_depth}"


def _parse_csv_ints(s: str) -> List[int]:
    return [int(x.strip()) for x in str(s).split(",") if x.strip()]


def _parse_budget_ladder(s: str) -> List[BudgetCfg]:
    out: List[BudgetCfg] = []
    for chunk in str(s).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        a, b = chunk.split(":", 1)
        out.append(BudgetCfg(int(a), int(b)))
    if not out:
        raise ValueError("empty budget ladder")
    return out


def _load_shards(patterns: Sequence[str]) -> List[dict[str, Any]]:
    rows: List[dict[str, Any]] = []
    for pat in patterns:
        for path in sorted(glob.glob(pat)):
            shard = torch.load(path, map_location="cpu", weights_only=False)
            if isinstance(shard, list):
                rows.extend(shard)
            else:
                raise TypeError(f"Unexpected shard payload in {path}: {type(shard)}")
    return rows


def _merge_shards(patterns: Sequence[str], out_path: Path) -> int:
    rows = _load_shards(patterns)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(rows, out_path)
    return len(rows)


def _save_untrained_ckpt(out_path: Path, d_model: int, refine_steps: int, dropout: float) -> None:
    model = gcp.TRMPolicyValue(d_model=d_model, refine_steps=refine_steps, dropout=dropout)
    payload = {
        "model": model.state_dict(),
        "optimizer": {},
        "epoch": 0,
        "args": {
            "d_model": int(d_model),
            "refine_steps": int(refine_steps),
            "dropout": float(dropout),
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)


def _train_small_model(
    train_pt: Path,
    valid_pt: Path,
    out_dir: Path,
    epochs: int,
    steps_per_epoch: int,
    valid_steps: int,
    batch_size: int,
    num_workers: int,
    seed: int,
    d_model: int,
    refine_steps: int,
    dropout: float,
    lr: float,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python",
        "gcp_trace_abstractbeam_v3.py",
        "train",
        "--train",
        str(train_pt),
        "--valid",
        str(valid_pt),
        "--out-dir",
        str(out_dir),
        "--epochs",
        str(int(epochs)),
        "--steps-per-epoch",
        str(int(steps_per_epoch)),
        "--valid-steps",
        str(int(valid_steps)),
        "--batch-size",
        str(int(batch_size)),
        "--num-workers",
        str(int(num_workers)),
        "--seed",
        str(int(seed)),
        "--d-model",
        str(int(d_model)),
        "--refine-steps",
        str(int(refine_steps)),
        "--dropout",
        str(float(dropout)),
        "--lr",
        str(float(lr)),
        "--amp",
    ]
    subprocess.run(cmd, cwd=str(ROOT), check=True)
    ckpt = out_dir / "model-best.pt"
    if not ckpt.exists():
        ckpt = out_dir / "model-last.pt"
    if not ckpt.exists():
        raise RuntimeError(f"No checkpoint produced in {out_dir}")
    return ckpt


def _mine_macros(trace_pt: Path, out_path: Path, min_support: int, max_len: int, top_k: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python",
        "gcp_trace_abstractbeam_v3.py",
        "mine-macros",
        "--trace",
        str(trace_pt),
        "--out",
        str(out_path),
        "--min-support",
        str(int(min_support)),
        "--max-len",
        str(int(max_len)),
        "--top-k",
        str(int(top_k)),
    ]
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def _best_greedy_solution(graph: gcp.GCGraph, k: int, rng: random.Random, num_random_orders: int) -> Tuple[np.ndarray, int]:
    n = graph.n
    degs = graph.degrees
    orders: List[np.ndarray] = [
        np.argsort(-degs),
        np.argsort(degs),
        np.arange(n, dtype=np.int64),
        np.arange(n - 1, -1, -1, dtype=np.int64),
    ]
    for _ in range(int(num_random_orders)):
        lst = list(range(n))
        rng.shuffle(lst)
        orders.append(np.asarray(lst, dtype=np.int64))
    best_cols = None
    best_conf = 10 ** 18
    for order in orders:
        cols = gcp.greedy_k_assignment(graph, k, order=order)
        st = gcp.RepairState(cols, k=k, plateau=0, step=0)
        met = gcp.compute_state_metrics(graph, st)
        if int(met.conflicts) < best_conf:
            best_conf = int(met.conflicts)
            best_cols = cols.copy()
    assert best_cols is not None
    return best_cols, int(best_conf)


def _trace_row(simulation: int, primitive_calls: int, best_conflicts: int, elapsed_sec: float) -> Dict[str, float]:
    return {
        "simulation": int(simulation),
        "primitive_calls": int(primitive_calls),
        "best_conflicts": int(best_conflicts),
        "elapsed_sec": float(elapsed_sec),
    }


def _run_greedy_best_of_orders(graph: gcp.GCGraph, k: int, rng: random.Random, num_random_orders: int) -> Dict[str, Any]:
    t0 = time.time()
    cols, conf = _best_greedy_solution(graph, k, rng, num_random_orders)
    elapsed = time.time() - t0
    return {
        "colors": cols.tolist(),
        "conflicts": int(conf),
        "solved": bool(conf == 0),
        "primitive_calls": 0,
        "time_sec": float(elapsed),
        "anytime_trace": [_trace_row(0, 0, int(conf), float(elapsed))],
        "timeout": False,
        "method": "greedy_best_of_orders",
    }


def _run_fixed_tabu_recolor(
    graph: gcp.GCGraph,
    k: int,
    rng: random.Random,
    max_steps: int,
    num_random_orders: int,
    exact_patch_limit: int,
    profile_every: int,
) -> Dict[str, Any]:
    t0 = time.time()
    cols, _ = _best_greedy_solution(graph, k, rng, num_random_orders)
    state = gcp.RepairState(cols, k=k, plateau=0, step=0)
    metrics = gcp.compute_state_metrics(graph, state)
    best_state = state.copy()
    best_metrics = metrics
    anytime: List[Dict[str, float]] = [_trace_row(0, 0, int(best_metrics.conflicts), 0.0)]
    for step_idx in range(int(max_steps)):
        if best_metrics.conflicts == 0:
            break
        cands = gcp.generate_candidate_actions(
            graph,
            state,
            metrics,
            macros=None,
            action_budget=64,
            restrict_families={
                gcp.PrimitiveFamily.VERTEX_RECOLOR.value,
                gcp.PrimitiveFamily.TABU_SHORT.value,
                gcp.PrimitiveFamily.TABU_LONG.value,
                gcp.PrimitiveFamily.EXACT_PATCH.value,
            },
            exact_patch_limit=int(exact_patch_limit),
        )
        if not cands:
            break
        # Hand-made fixed scheduler: prefer local recolor improvements; every 3rd step allow tabu burst; exact patch if tiny core.
        def score(c: gcp.CandidateAction) -> Tuple[float, float, float]:
            bias = 0.0
            if step_idx % 3 == 2 and c.family == gcp.PrimitiveFamily.TABU_SHORT.value:
                bias += 0.15
            if metrics.core_size <= exact_patch_limit and c.family == gcp.PrimitiveFamily.EXACT_PATCH.value:
                bias += 0.10
            return (float(c.est_delta_conflicts) + bias, float(c.est_delta_vertices), -float(c.est_cost))
        cands.sort(key=score, reverse=True)
        action = cands[0]
        state, metrics, _ = gcp.transition_state(graph, state, metrics, action, macros=None, exact_patch_limit=int(exact_patch_limit))
        if (metrics.conflicts < best_metrics.conflicts) or (
            metrics.conflicts == best_metrics.conflicts and metrics.conflict_vertices < best_metrics.conflict_vertices
        ):
            best_state = state.copy()
            best_metrics = metrics
        if int(profile_every) > 0 and (((step_idx + 1) % int(profile_every)) == 0 or best_metrics.conflicts == 0):
            anytime.append(_trace_row(step_idx + 1, int(state.step), int(best_metrics.conflicts), time.time() - t0))
    elapsed = time.time() - t0
    if not anytime or anytime[-1]["elapsed_sec"] < elapsed:
        anytime.append(_trace_row(int(max_steps), int(best_state.step), int(best_metrics.conflicts), elapsed))
    return {
        "colors": best_state.colors.tolist(),
        "conflicts": int(best_metrics.conflicts),
        "solved": bool(best_metrics.conflicts == 0),
        "primitive_calls": int(best_state.step),
        "time_sec": float(elapsed),
        "anytime_trace": anytime,
        "timeout": False,
        "method": "fixed_tabu_recolor",
    }


def _run_mcts_solve(
    method: str,
    graph_json: Path,
    k: int,
    simulations: int,
    max_depth: int,
    timeout_sec: float,
    ckpt: Optional[Path] = None,
    macros: Optional[Path] = None,
    profile_every: int = 4,
) -> Dict[str, Any]:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fprof:
        profile_path = Path(fprof.name)
    cmd = [
        "python",
        "gcp_trace_abstractbeam_v3.py",
        "solve",
        "--input",
        str(graph_json),
        "--k",
        str(int(k)),
        "--simulations",
        str(int(simulations)),
        "--max-depth",
        str(int(max_depth)),
        "--profile-every",
        str(int(profile_every)),
        "--profile-out",
        str(profile_path),
    ]
    if ckpt is not None:
        cmd.extend(["--ckpt", str(ckpt)])
    if macros is not None:
        cmd.extend(["--macros", str(macros)])
    t0 = time.time()
    try:
        out = subprocess.check_output(cmd, cwd=str(ROOT), text=True, timeout=float(timeout_sec))
        payload = json.loads(out)
        payload["time_sec"] = float(time.time() - t0)
        payload["timeout"] = False
        payload["method"] = method
        return payload
    except subprocess.TimeoutExpired:
        payload: Dict[str, Any] = {
            "colors": None,
            "conflicts": None,
            "solved": False,
            "primitive_calls": None,
            "anytime_trace": [],
            "time_sec": float(time.time() - t0),
            "timeout": True,
            "method": method,
        }
        if profile_path.exists():
            try:
                prof = json.loads(profile_path.read_text())
                payload["primitive_calls"] = prof.get("primitive_calls")
                payload["anytime_trace"] = prof.get("anytime_trace", [])
                payload["conflicts"] = prof.get("final_conflicts")
            except Exception:
                pass
        return payload
    finally:
        profile_path.unlink(missing_ok=True)


def _bootstrap_ci(values: Sequence[float], seed: int, n_boot: int = 2000) -> Tuple[Optional[float], Optional[float]]:
    vals = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if not vals:
        return None, None
    rng = random.Random(int(seed))
    means: List[float] = []
    m = len(vals)
    for _ in range(int(n_boot)):
        sample = [vals[rng.randrange(m)] for _ in range(m)]
        means.append(float(sum(sample) / max(len(sample), 1)))
    means.sort()
    lo = means[int(0.025 * (len(means) - 1))]
    hi = means[int(0.975 * (len(means) - 1))]
    return float(lo), float(hi)


def _paired_outcome(a_conf: Optional[float], a_timeout: bool, b_conf: Optional[float], b_timeout: bool) -> float:
    if a_timeout and b_timeout:
        return 0.5
    if a_timeout and not b_timeout:
        return 0.0
    if b_timeout and not a_timeout:
        return 1.0
    if a_conf is None or b_conf is None:
        return 0.5
    if a_conf < b_conf:
        return 1.0
    if a_conf > b_conf:
        return 0.0
    return 0.5


def _anytime_improvement_per_sec(trace: Sequence[Dict[str, Any]]) -> Optional[float]:
    if not trace:
        return None
    first = trace[0]
    last = trace[-1]
    t0 = float(first.get("elapsed_sec", 0.0))
    t1 = float(last.get("elapsed_sec", 0.0))
    c0 = float(first.get("best_conflicts", 0.0))
    c1 = float(last.get("best_conflicts", 0.0))
    dt = max(t1 - t0, 1e-9)
    return float((c0 - c1) / dt)


def _aggregate_budget_results(rows: List[Dict[str, Any]], baseline_method: str, seed: int) -> Dict[str, Any]:
    by_method: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_method.setdefault(str(r["method"]), []).append(r)
    baseline_rows = {str(r["instance_id"]): r for r in by_method.get(baseline_method, [])}

    summary: Dict[str, Any] = {}
    for method, mrows in by_method.items():
        finished = [r for r in mrows if not bool(r.get("timeout", False)) and r.get("conflicts") is not None]
        confs = [float(r["conflicts"]) for r in finished]
        times = [float(r.get("time_sec", 0.0)) for r in mrows]
        prims = [float(r["primitive_calls"]) for r in finished if r.get("primitive_calls") not in (None, 0)]
        pair_deltas: List[float] = []
        win_scores: List[float] = []
        gain_per_sec: List[float] = []
        gain_per_prim: List[float] = []
        anytime_rates: List[float] = []
        for r in mrows:
            bid = str(r["instance_id"])
            b = baseline_rows.get(bid)
            if b is not None:
                win_scores.append(_paired_outcome(r.get("conflicts"), bool(r.get("timeout", False)), b.get("conflicts"), bool(b.get("timeout", False))))
                if (not bool(r.get("timeout", False))) and (not bool(b.get("timeout", False))) and r.get("conflicts") is not None and b.get("conflicts") is not None:
                    delta = float(r["conflicts"]) - float(b["conflicts"])
                    pair_deltas.append(delta)
                    t = max(float(r.get("time_sec", 0.0)), 1e-9)
                    gain_per_sec.append(float((float(b["conflicts"]) - float(r["conflicts"])) / t))
                    pc = r.get("primitive_calls")
                    if pc not in (None, 0):
                        gain_per_prim.append(float((float(b["conflicts"]) - float(r["conflicts"])) / max(float(pc), 1e-9)))
            at = _anytime_improvement_per_sec(r.get("anytime_trace", []))
            if at is not None:
                anytime_rates.append(float(at))
        ci_lo, ci_hi = _bootstrap_ci(pair_deltas, seed=seed + abs(hash(method)) % 100000)
        summary[method] = {
            "n": len(mrows),
            "n_finished": len(finished),
            "solved_rate": float(sum(1 for r in mrows if bool(r.get("solved", False))) / max(len(mrows), 1)),
            "timeout_rate": float(sum(1 for r in mrows if bool(r.get("timeout", False))) / max(len(mrows), 1)),
            "mean_conflicts": float(np.mean(confs)) if confs else None,
            "median_conflicts": float(np.median(confs)) if confs else None,
            "mean_time_sec": float(np.mean(times)) if times else None,
            "mean_primitive_calls": float(np.mean(prims)) if prims else None,
            "delta_mean_vs_baseline": float(np.mean(pair_deltas)) if pair_deltas else None,
            "delta_mean_vs_baseline_ci95": [ci_lo, ci_hi],
            "win_rate_vs_baseline": float(np.mean(win_scores)) if win_scores else None,
            "compute_normalized_gain_per_sec": float(np.mean(gain_per_sec)) if gain_per_sec else None,
            "compute_normalized_gain_per_primitive": float(np.mean(gain_per_prim)) if gain_per_prim else None,
            "anytime_improvement_per_sec": float(np.mean(anytime_rates)) if anytime_rates else None,
        }

    # Explicit paired comparisons requested for scientific reporting.
    pair_specs = [
        ("mcts_trained_small", "greedy_best_of_orders"),
        ("mcts_trained_small", "mcts_untrained"),
        ("mcts_trained_small", "mcts_no_model"),
        ("mcts_trained_curriculum", "mcts_trained_small"),
    ]
    pairwise: Dict[str, Any] = {}
    rows_by_method_inst: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for r in rows:
        rows_by_method_inst[(str(r["method"]), str(r["instance_id"]))] = r
    for a, b in pair_specs:
        outcomes: List[float] = []
        deltas: List[float] = []
        insts = sorted({iid for (_, iid) in rows_by_method_inst.keys()})
        for iid in insts:
            ra = rows_by_method_inst.get((a, iid))
            rb = rows_by_method_inst.get((b, iid))
            if ra is None or rb is None:
                continue
            outcomes.append(_paired_outcome(ra.get("conflicts"), bool(ra.get("timeout", False)), rb.get("conflicts"), bool(rb.get("timeout", False))))
            if (not bool(ra.get("timeout", False))) and (not bool(rb.get("timeout", False))) and ra.get("conflicts") is not None and rb.get("conflicts") is not None:
                deltas.append(float(ra["conflicts"]) - float(rb["conflicts"]))
        lo, hi = _bootstrap_ci(deltas, seed=seed + abs(hash((a, b))) % 100000)
        key = f"{a}_vs_{b}"
        pairwise[key] = {
            "a": a,
            "b": b,
            "n_pairs": len(outcomes),
            "win_rate_a_over_b": float(np.mean(outcomes)) if outcomes else None,
            "delta_mean_a_minus_b": float(np.mean(deltas)) if deltas else None,
            "delta_ci95_a_minus_b": [lo, hi],
        }
    return {"summary_by_method": summary, "pairwise": pairwise}


def _markdown_report(payload: Dict[str, Any]) -> str:
    lines: List[str] = []
    cfg = payload["config"]
    lines.append("# Final Report — small-to-large ladder v3")
    lines.append("")
    lines.append("## Configuration")
    lines.append("")
    lines.append(f"- session: `{cfg['session_dir']}`")
    lines.append(f"- sizes: `{cfg['sizes']}`")
    lines.append(f"- per-size: `{cfg['per_size']}`")
    lines.append(f"- budgets: `{cfg['budget_ladder']}`")
    lines.append(f"- timeout_sec: `{cfg['timeout_sec']}`")
    lines.append(f"- profile_every: `{cfg['profile_every']}`")
    lines.append("")
    for budget_tag, block in payload["budgets"].items():
        lines.append(f"## Budget {budget_tag}")
        lines.append("")
        lines.append("| Method | solved_rate | timeout_rate | mean_conf | median_conf | mean_time_s | mean_primitive_calls | win_rate_vs_greedy | delta_mean_vs_greedy |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for method, s in sorted(block["summary_by_method"].items()):
            def fmt(v: Any) -> str:
                if v is None:
                    return "-"
                if isinstance(v, float):
                    return f"{v:.4f}"
                return str(v)
            lines.append(
                "| " + " | ".join([
                    method,
                    fmt(s.get("solved_rate")),
                    fmt(s.get("timeout_rate")),
                    fmt(s.get("mean_conflicts")),
                    fmt(s.get("median_conflicts")),
                    fmt(s.get("mean_time_sec")),
                    fmt(s.get("mean_primitive_calls")),
                    fmt(s.get("win_rate_vs_baseline")),
                    fmt(s.get("delta_mean_vs_baseline")),
                ]) + " |"
            )
        lines.append("")
        lines.append("### Compute-normalized")
        lines.append("")
        lines.append("| Method | gain_per_sec | gain_per_primitive | anytime_improvement_per_sec |")
        lines.append("|---|---:|---:|---:|")
        for method, s in sorted(block["summary_by_method"].items()):
            def fmt(v: Any) -> str:
                if v is None:
                    return "-"
                if isinstance(v, float):
                    return f"{v:.4f}"
                return str(v)
            lines.append(
                "| " + " | ".join([
                    method,
                    fmt(s.get("compute_normalized_gain_per_sec")),
                    fmt(s.get("compute_normalized_gain_per_primitive")),
                    fmt(s.get("anytime_improvement_per_sec")),
                ]) + " |"
            )
        lines.append("")
        lines.append("### Paired Comparisons")
        lines.append("")
        lines.append("| Pair | n_pairs | win_rate(a>b) | delta_mean(a-b) | delta_ci95(a-b) |")
        lines.append("|---|---:|---:|---:|---:|")
        for key, pair in sorted(block.get("pairwise", {}).items()):
            def fmtp(v: Any) -> str:
                if v is None:
                    return "-"
                if isinstance(v, float):
                    return f"{v:.4f}"
                return str(v)
            ci = pair.get("delta_ci95_a_minus_b", [None, None])
            ci_txt = "-"
            if isinstance(ci, list) and len(ci) == 2 and ci[0] is not None and ci[1] is not None:
                ci_txt = f"[{float(ci[0]):.4f}, {float(ci[1]):.4f}]"
            lines.append(
                "| " + " | ".join([
                    key,
                    fmtp(pair.get("n_pairs")),
                    fmtp(pair.get("win_rate_a_over_b")),
                    fmtp(pair.get("delta_mean_a_minus_b")),
                    ci_txt,
                ]) + " |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Small-to-large ladder v3 with anytime + compute-normalized metrics")
    ap.add_argument("--session", type=str, default="")
    ap.add_argument("--seed", type=int, default=20260420)
    ap.add_argument("--small-train-globs", type=str, default="")
    ap.add_argument("--small-valid-globs", type=str, default="")
    ap.add_argument("--curriculum-ckpt", type=str, default="")
    ap.add_argument("--sizes", type=str, default="100,200,400")
    ap.add_argument("--per-size", type=int, default=10)
    ap.add_argument("--budget-ladder", type=str, default="12:24,32:48,64:96")
    ap.add_argument("--timeout-sec", type=float, default=60.0)
    ap.add_argument("--profile-every", type=int, default=4)
    ap.add_argument("--random-orders", type=int, default=6)
    ap.add_argument("--min-greedy-conflicts", type=int, default=1)
    ap.add_argument("--max-greedy-conflicts", type=int, default=0)
    ap.add_argument("--max-tries", type=int, default=400)
    ap.add_argument("--fixed-max-steps", type=int, default=24)
    ap.add_argument("--include-macro-method", action="store_true")
    ap.add_argument("--train-epochs", type=int, default=2)
    ap.add_argument("--train-steps-per-epoch", type=int, default=600)
    ap.add_argument("--train-valid-steps", type=int, default=100)
    ap.add_argument("--train-batch-size", type=int, default=64)
    ap.add_argument("--train-num-workers", type=int, default=2)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--refine-steps", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--macro-min-support", type=int, default=8)
    ap.add_argument("--macro-max-len", type=int, default=5)
    ap.add_argument("--macro-top-k", type=int, default=64)
    args = ap.parse_args()

    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    session_name = str(args.session).strip() or f"small_to_large_ladder_{int(time.time())}"
    session_dir = RUNS_ROOT / session_name
    session_dir.mkdir(parents=True, exist_ok=True)

    small_train_globs = [x.strip() for x in str(args.small_train_globs).split(",") if x.strip()]
    small_valid_globs = [x.strip() for x in str(args.small_valid_globs).split(",") if x.strip()]
    if not small_train_globs:
        raise ValueError("--small-train-globs is required")
    if not small_valid_globs:
        raise ValueError("--small-valid-globs is required")

    merged_dir = session_dir / "merged_traces"
    train_pt = merged_dir / "train_merged.pt"
    valid_pt = merged_dir / "valid_merged.pt"
    train_rows = _merge_shards(small_train_globs, train_pt)
    valid_rows = _merge_shards(small_valid_globs, valid_pt)

    train_out = session_dir / "trained_small"
    trained_small_ckpt = _train_small_model(
        train_pt,
        valid_pt,
        out_dir=train_out,
        epochs=int(args.train_epochs),
        steps_per_epoch=int(args.train_steps_per_epoch),
        valid_steps=int(args.train_valid_steps),
        batch_size=int(args.train_batch_size),
        num_workers=int(args.train_num_workers),
        seed=int(args.seed),
        d_model=int(args.d_model),
        refine_steps=int(args.refine_steps),
        dropout=float(args.dropout),
        lr=float(args.lr),
    )

    macros_path = session_dir / "small_macros.json"
    _mine_macros(train_pt, macros_path, min_support=int(args.macro_min_support), max_len=int(args.macro_max_len), top_k=int(args.macro_top_k))

    untrained_ckpt = session_dir / "untrained_trm.pt"
    _save_untrained_ckpt(untrained_ckpt, int(args.d_model), int(args.refine_steps), float(args.dropout))

    rng = random.Random(int(args.seed))
    sizes = _parse_csv_ints(args.sizes)
    budget_ladder = _parse_budget_ladder(args.budget_ladder)

    instances: List[Dict[str, Any]] = []
    schedule: List[Tuple[int, int, float]] = []
    for n in sizes:
        if n <= 120:
            schedule.append((n, 6, 0.22))
        elif n <= 250:
            schedule.append((n, 8, 0.18))
        else:
            schedule.append((n, 10, 0.14))
    for n, k, p_cross in schedule:
        for i in range(int(args.per_size)):
            hid = hard.sample_hard_instance(
                rng,
                n,
                k,
                p_cross,
                name=f"hard_n{n}_{i:03d}",
                max_tries=int(args.max_tries),
                min_greedy_conflicts=int(args.min_greedy_conflicts),
                num_random_orders=int(args.random_orders),
                max_greedy_conflicts=int(args.max_greedy_conflicts) if int(args.max_greedy_conflicts) > 0 else None,
            )
            if hid is None:
                continue
            instances.append(hid)

    methods: List[str] = [
        "greedy_best_of_orders",
        "fixed_tabu_recolor",
        "mcts_no_model",
        "mcts_untrained",
        "mcts_trained_small",
    ]
    curriculum_ckpt = Path(str(args.curriculum_ckpt)) if str(args.curriculum_ckpt).strip() else None
    if curriculum_ckpt is not None and curriculum_ckpt.exists():
        methods.append("mcts_trained_curriculum")
    if args.include_macro_method:
        methods.append("mcts_trained_small_macros")

    results_by_budget: Dict[str, Dict[str, Any]] = {}
    for budget in budget_ladder:
        rows: List[Dict[str, Any]] = []
        for inst in instances:
            rec = gcp.GraphRecord(name=inst["name"], n=int(inst["n"]), edges=np.asarray(inst["edges"], dtype=np.int64), solution=None, metadata={})
            graph = rec.to_runtime()
            instance_id = str(inst["name"])
            k = int(inst["tight_k"])
            for method in methods:
                if method == "greedy_best_of_orders":
                    pred = _run_greedy_best_of_orders(graph, k, rng, int(args.random_orders))
                elif method == "fixed_tabu_recolor":
                    pred = _run_fixed_tabu_recolor(graph, k, rng, max_steps=int(args.fixed_max_steps), num_random_orders=int(args.random_orders), exact_patch_limit=12, profile_every=int(args.profile_every))
                else:
                    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
                        json.dump(hard.to_public_json(inst), f)
                        graph_json = Path(f.name)
                    if method == "mcts_no_model":
                        pred = _run_mcts_solve(method, graph_json, k, budget.simulations, budget.max_depth, timeout_sec=float(args.timeout_sec), ckpt=None, macros=None, profile_every=int(args.profile_every))
                    elif method == "mcts_untrained":
                        pred = _run_mcts_solve(method, graph_json, k, budget.simulations, budget.max_depth, timeout_sec=float(args.timeout_sec), ckpt=untrained_ckpt, macros=None, profile_every=int(args.profile_every))
                    elif method == "mcts_trained_small":
                        pred = _run_mcts_solve(method, graph_json, k, budget.simulations, budget.max_depth, timeout_sec=float(args.timeout_sec), ckpt=trained_small_ckpt, macros=None, profile_every=int(args.profile_every))
                    elif method == "mcts_trained_small_macros":
                        pred = _run_mcts_solve(method, graph_json, k, budget.simulations, budget.max_depth, timeout_sec=float(args.timeout_sec), ckpt=trained_small_ckpt, macros=macros_path, profile_every=int(args.profile_every))
                    elif method == "mcts_trained_curriculum":
                        assert curriculum_ckpt is not None
                        pred = _run_mcts_solve(method, graph_json, k, budget.simulations, budget.max_depth, timeout_sec=float(args.timeout_sec), ckpt=curriculum_ckpt, macros=None, profile_every=int(args.profile_every))
                    else:
                        raise ValueError(method)
                    graph_json.unlink(missing_ok=True)
                conf = pred.get("conflicts")
                if conf is None:
                    rigorous_conf = None
                    rigorous_solved = False
                else:
                    rigorous_conf = hard.count_conflicts(inst["edges"], pred["colors"]) if pred.get("colors") is not None else int(conf)
                    rigorous_solved = int(rigorous_conf) == 0
                rows.append(
                    {
                        "instance_id": instance_id,
                        "n": int(inst["n"]),
                        "tight_k": k,
                        "greedy_k_best_conflicts": int(inst.get("greedy_k_best_conflicts", -1)),
                        "method": method,
                        "budget": budget.tag,
                        "conflicts": int(rigorous_conf) if rigorous_conf is not None else None,
                        "reported_conflicts": pred.get("conflicts"),
                        "solved": bool(rigorous_solved),
                        "time_sec": float(pred.get("time_sec", 0.0)),
                        "primitive_calls": pred.get("primitive_calls"),
                        "timeout": bool(pred.get("timeout", False)),
                        "anytime_trace": pred.get("anytime_trace", []),
                    }
                )
        results_by_budget[budget.tag] = {
            "rows": rows,
            **_aggregate_budget_results(rows, baseline_method="greedy_best_of_orders", seed=int(args.seed) + abs(hash(budget.tag)) % 10000),
        }

    payload: Dict[str, Any] = {
        "config": {
            "session_dir": str(session_dir),
            "seed": int(args.seed),
            "small_train_globs": small_train_globs,
            "small_valid_globs": small_valid_globs,
            "curriculum_ckpt": str(curriculum_ckpt) if curriculum_ckpt else None,
            "sizes": sizes,
            "per_size": int(args.per_size),
            "budget_ladder": [b.tag for b in budget_ladder],
            "timeout_sec": float(args.timeout_sec),
            "profile_every": int(args.profile_every),
            "include_macro_method": bool(args.include_macro_method),
            "methods": methods,
            "train_rows": int(train_rows),
            "valid_rows": int(valid_rows),
            "trained_small_ckpt": str(trained_small_ckpt),
            "small_macros": str(macros_path),
            "untrained_ckpt": str(untrained_ckpt),
        },
        "budgets": results_by_budget,
    }
    json_path = session_dir / "ladder_results.json"
    json_path.write_text(json.dumps(payload, indent=2))
    md_path = session_dir / "FINAL_LADDER_REPORT.md"
    md_path.write_text(_markdown_report(payload))
    print(json.dumps({"wrote": str(json_path), "report": str(md_path), "session": str(session_dir)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
