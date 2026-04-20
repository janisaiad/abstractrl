#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import subprocess
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

import gcp_hard_benchmark_v2 as hb
import gcp_trace_abstractbeam_v2 as gcp


@dataclass
class MethodCfg:
    name: str
    kind: str
    ckpt: str = ""
    macros: str = ""
    enabled_sizes: Optional[List[int]] = None


def run_cmd(command: List[str], cwd: Path) -> None:
    subprocess.run(command, cwd=str(cwd), check=True)


def parse_sizes(spec: str) -> List[int]:
    return [int(x.strip()) for x in spec.split(",") if x.strip()]


def parse_budgets(spec: str) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for token in spec.split(","):
        t = token.strip()
        if not t:
            continue
        sim_s, dep_s = t.split(":")
        out.append((int(sim_s), int(dep_s)))
    return out


def concat_shards(shard_paths: Iterable[Path], out_path: Path) -> int:
    rows: List[dict] = []
    for shard in shard_paths:
        payload = torch.load(shard, map_location="cpu", weights_only=False)
        if isinstance(payload, list):
            rows.extend(payload)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(rows, out_path)
    return len(rows)


def make_untrained_checkpoint(path: Path, d_model: int = 256, refine_steps: int = 3, dropout: float = 0.1) -> None:
    model = gcp.TRMPolicyValue(d_model=d_model, refine_steps=refine_steps, dropout=dropout)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": -1,
        "args": {
            "d_model": d_model,
            "refine_steps": refine_steps,
            "dropout": dropout,
            "batch_size": 0,
            "epochs": 0,
            "steps_per_epoch": 0,
            "valid_steps": 0,
            "lr": 3e-4,
        },
    }
    torch.save(ckpt, path)


def fixed_tabu_recolor_solve(graph: gcp.GCGraph, k: int, max_steps: int, action_budget: int) -> Dict[str, object]:
    seed = gcp.greedy_k_assignment(graph, int(k))
    state = gcp.RepairState(seed, k=int(k), plateau=0, step=0)
    fam_counts: Counter[str] = Counter()
    primitive_calls = 0
    families = {
        gcp.PrimitiveFamily.VERTEX_RECOLOR.value,
        gcp.PrimitiveFamily.TABU_SHORT.value,
        gcp.PrimitiveFamily.TABU_LONG.value,
    }
    for _ in range(int(max_steps)):
        metrics = gcp.compute_state_metrics(graph, state)
        if metrics.conflicts == 0:
            break
        cands = gcp.generate_candidate_actions(
            graph=graph,
            state=state,
            metrics=metrics,
            macros=None,
            action_budget=int(action_budget),
            restrict_families=families,
            exact_patch_limit=12,
        )
        if not cands:
            break
        scored = gcp.evaluate_candidates(graph, state, metrics, cands, macros=None, exact_patch_limit=12)
        idx = int(np.argmax(scored["teacher_scores"]))
        chosen = cands[idx]
        fam_counts[chosen.family] += 1
        primitive_calls += 1
        state, _, _ = gcp.transition_state(graph, state, metrics, chosen, macros=None, exact_patch_limit=12)
    final = gcp.compute_state_metrics(graph, state)
    return {
        "conflicts": int(final.conflicts),
        "solved": bool(final.conflicts == 0),
        "primitive_calls": int(primitive_calls),
        "family_usage": dict(fam_counts),
        "core_size": int(final.core_size),
        "conflict_vertices": int(final.conflict_vertices),
    }


def solve_with_cli(
    graph_dict: Dict[str, object],
    ckpt: Optional[Path],
    macros: Optional[Path],
    k: int,
    simulations: int,
    max_depth: int,
    timeout_sec: int,
    cwd: Path,
) -> Dict[str, object]:
    public = {"name": graph_dict["name"], "n": graph_dict["n"], "edges": graph_dict["edges"]}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(public, f)
        json_path = Path(f.name)
    with tempfile.NamedTemporaryFile("w", suffix=".profile.json", delete=False) as pf:
        profile_path = Path(pf.name)
    args = [
        "python3",
        "gcp_trace_abstractbeam_v2.py",
        "solve",
        "--input",
        str(json_path),
        "--k",
        str(int(k)),
        "--simulations",
        str(int(simulations)),
        "--max-depth",
        str(int(max_depth)),
        "--profile-every",
        "4",
        "--profile-out",
        str(profile_path),
    ]
    if ckpt is not None and str(ckpt):
        args.extend(["--ckpt", str(ckpt)])
    if macros is not None and str(macros):
        args.extend(["--macros", str(macros)])
    try:
        out = subprocess.check_output(args, cwd=str(cwd), text=True, timeout=int(timeout_sec))
        result = json.loads(out)
        if profile_path.exists():
            try:
                prof = json.loads(profile_path.read_text())
                result["profile"] = prof
            except Exception:
                pass
        return result
    finally:
        json_path.unlink(missing_ok=True)
        profile_path.unlink(missing_ok=True)


def schedule_for_n(n: int) -> Tuple[int, float]:
    if n <= 120:
        return 6, 0.22
    if n <= 250:
        return 8, 0.18
    return 10, 0.14


def generate_instances(
    sizes: List[int],
    per_size: int,
    max_tries: int,
    min_greedy_conflicts: int,
    max_greedy_conflicts: int,
    random_orders: int,
    seed: int,
) -> List[Dict[str, object]]:
    rng = random.Random(seed)
    rows: List[Dict[str, object]] = []
    for n in sizes:
        k, p_cross = schedule_for_n(int(n))
        for i in range(int(per_size)):
            hid = hb.sample_hard_instance(
                rng=rng,
                n=int(n),
                k=int(k),
                p_cross=float(p_cross),
                name=f"hard_n{n}_{i:03d}",
                max_tries=int(max_tries),
                min_greedy_conflicts=int(min_greedy_conflicts),
                num_random_orders=int(random_orders),
                max_greedy_conflicts=int(max_greedy_conflicts) if int(max_greedy_conflicts) > 0 else None,
            )
            if hid is None:
                rows.append({"name": f"hard_n{n}_{i:03d}", "n": int(n), "k": int(k), "error": "no_hard_instance_found"})
                continue
            rows.append(
                {
                    "name": str(hid["name"]),
                    "n": int(hid["n"]),
                    "k": int(hid["tight_k"]),
                    "p_cross": float(p_cross),
                    "edges": hid["edges"],
                    "m": int(len(hid["edges"])),
                    "greedy_k_best_conflicts": int(hid.get("greedy_k_best_conflicts", -1)),
                    "hard_sample_tries": int(hid.get("hard_sample_tries", -1)),
                }
            )
    return rows


def bootstrap_ci(values: List[float], n_boot: int, seed: int) -> Tuple[float, float]:
    if not values:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=np.float64)
    boots = []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, arr.shape[0], size=arr.shape[0])
        boots.append(float(np.mean(arr[idx])))
    boots_arr = np.asarray(boots)
    lo = float(np.quantile(boots_arr, 0.025))
    hi = float(np.quantile(boots_arr, 0.975))
    return (lo, hi)


def summarize(records: List[Dict[str, object]], methods: List[MethodCfg], baseline: str, n_boot: int, seed: int) -> Dict[str, object]:
    by_method: Dict[str, Dict[str, object]] = {}
    by_key: Dict[Tuple[str, int, int], Dict[str, Dict[str, object]]] = defaultdict(dict)
    for r in records:
        if "error" in r:
            continue
        key = (str(r["instance"]), int(r["simulations"]), int(r["max_depth"]))
        by_key[key][str(r["method"])] = r
    for method in methods:
        all_rows = [r for r in records if r.get("method") == method.name]
        method_rows = [r for r in all_rows if "error" not in r]
        timeout_rows = [r for r in all_rows if r.get("error") == "timeout"]
        solved = sum(1 for r in method_rows if bool(r.get("solved", False)))
        total = len(method_rows)
        conflicts = [float(r.get("model_conflicts", 0.0)) for r in method_rows]
        times = [float(r.get("elapsed_sec", 0.0)) for r in method_rows]
        primitive_calls = [float(r.get("primitive_calls", 0.0)) for r in method_rows if "primitive_calls" in r]
        usage = defaultdict(int)
        for row in method_rows:
            fam = row.get("family_usage", {})
            if isinstance(fam, dict):
                for k, v in fam.items():
                    usage[str(k)] += int(v)
        deltas = []
        wins = 0
        ties = 0
        den = 0
        for key, per_method in by_key.items():
            if baseline not in per_method or method.name not in per_method:
                continue
            b = float(per_method[baseline]["model_conflicts"])
            m = float(per_method[method.name]["model_conflicts"])
            deltas.append(m - b)
            den += 1
            if m < b:
                wins += 1
            elif m == b:
                ties += 1
        lo, hi = bootstrap_ci(deltas, n_boot=n_boot, seed=seed + hash(method.name) % 9973)
        gain_per_sec = []
        gain_per_primitive = []
        anytime_impr_per_sec = []
        for key, per_method in by_key.items():
            if baseline not in per_method or method.name not in per_method:
                continue
            b = float(per_method[baseline]["model_conflicts"])
            mrow = per_method[method.name]
            m = float(mrow["model_conflicts"])
            elapsed = max(float(mrow.get("elapsed_sec", 0.0)), 1e-9)
            gain_per_sec.append((b - m) / elapsed)
            if "primitive_calls" in mrow and float(mrow["primitive_calls"]) > 0:
                gain_per_primitive.append((b - m) / float(mrow["primitive_calls"]))
            trace = mrow.get("anytime_trace", [])
            if isinstance(trace, list) and trace:
                first = trace[0]
                last = trace[-1]
                if isinstance(first, dict) and isinstance(last, dict):
                    c0 = float(first.get("best_conflicts", m))
                    c1 = float(last.get("best_conflicts", m))
                    t1 = max(float(last.get("elapsed_sec", elapsed)), 1e-9)
                    anytime_impr_per_sec.append((c0 - c1) / t1)
        by_method[method.name] = {
            "num_attempts": len(all_rows),
            "num_instances": total,
            "num_timeouts": len(timeout_rows),
            "timeout_rate": (len(timeout_rows) / max(len(all_rows), 1)),
            "solved_rigorous": solved,
            "solved_rate": (solved / total) if total else None,
            "mean_model_conflicts": (float(np.mean(conflicts)) if conflicts else None),
            "median_model_conflicts": (float(np.median(conflicts)) if conflicts else None),
            "mean_elapsed_sec": (float(np.mean(times)) if times else None),
            "mean_primitive_calls": (float(np.mean(primitive_calls)) if primitive_calls else None),
            "family_usage": dict(sorted(usage.items(), key=lambda kv: kv[1], reverse=True)),
            "delta_vs_baseline_mean": (float(np.mean(deltas)) if deltas else None),
            "delta_vs_baseline_ci95": [lo, hi] if deltas else None,
            "win_rate_vs_baseline": (wins / den) if den else None,
            "tie_rate_vs_baseline": (ties / den) if den else None,
            "compute_normalized_gain_per_sec": (float(np.mean(gain_per_sec)) if gain_per_sec else None),
            "compute_normalized_gain_per_primitive": (float(np.mean(gain_per_primitive)) if gain_per_primitive else None),
            "anytime_improvement_per_sec": (float(np.mean(anytime_impr_per_sec)) if anytime_impr_per_sec else None),
        }
    return {"baseline": baseline, "by_method": by_method}


def write_markdown_report(path: Path, payload: Dict[str, object]) -> None:
    summary = payload["summary"]["by_method"]  # type: ignore[index]
    methods = payload["methods"]  # type: ignore[index]
    baseline = payload["summary"]["baseline"]  # type: ignore[index]
    cfg = payload["config"]  # type: ignore[index]
    lines: List[str] = []
    lines.append("# Rapport final — Small-to-large budget ladder v2")
    lines.append("")
    lines.append("## 1. Protocole")
    lines.append("")
    lines.append(f"- Tailles: `{cfg['sizes']}`")
    lines.append(f"- Instances par taille: `{cfg['per_size']}`")
    lines.append(f"- Budgets (sim:depth): `{cfg['budgets']}`")
    lines.append(f"- Baseline de comparaison: `{baseline}`")
    lines.append("")
    lines.append("## 2. Résultats agrégés")
    lines.append("")
    lines.append("| Méthode | solved_rate | mean_conf | median_conf | win_rate_vs_greedy | delta_mean_vs_greedy | CI95 delta | mean_time_sec | timeout_rate |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for m in methods:
        name = m["name"]
        row = summary.get(name, {})
        ci = row.get("delta_vs_baseline_ci95")
        ci_txt = "n/a"
        if isinstance(ci, list) and len(ci) == 2 and ci[0] is not None and ci[1] is not None:
            ci_txt = f"[{float(ci[0]):.3f}, {float(ci[1]):.3f}]"
        solved_rate_txt = "n/a" if row.get("solved_rate") is None else f"{float(row['solved_rate']):.3f}"
        mean_conf_txt = "n/a" if row.get("mean_model_conflicts") is None else f"{float(row['mean_model_conflicts']):.3f}"
        median_conf_txt = "n/a" if row.get("median_model_conflicts") is None else f"{float(row['median_model_conflicts']):.3f}"
        win_rate_txt = "n/a" if row.get("win_rate_vs_baseline") is None else f"{float(row['win_rate_vs_baseline']):.3f}"
        delta_txt = "n/a" if row.get("delta_vs_baseline_mean") is None else f"{float(row['delta_vs_baseline_mean']):.3f}"
        mean_time_txt = "n/a" if row.get("mean_elapsed_sec") is None else f"{float(row['mean_elapsed_sec']):.3f}"
        timeout_txt = "n/a" if row.get("timeout_rate") is None else f"{float(row['timeout_rate']):.3f}"
        lines.append(f"| `{name}` | {solved_rate_txt} | {mean_conf_txt} | {median_conf_txt} | {win_rate_txt} | {delta_txt} | {ci_txt} | {mean_time_txt} | {timeout_txt} |")
    lines.append("")
    lines.append("## 3. Compute-normalized")
    lines.append("")
    lines.append("| Méthode | gain_per_sec | gain_per_primitive | anytime_improvement_per_sec |")
    lines.append("|---|---:|---:|---:|")
    for m in methods:
        name = m["name"]
        row = summary.get(name, {})
        gps = "n/a" if row.get("compute_normalized_gain_per_sec") is None else f"{float(row['compute_normalized_gain_per_sec']):.4f}"
        gpp = "n/a" if row.get("compute_normalized_gain_per_primitive") is None else f"{float(row['compute_normalized_gain_per_primitive']):.4f}"
        ais = "n/a" if row.get("anytime_improvement_per_sec") is None else f"{float(row['anytime_improvement_per_sec']):.4f}"
        lines.append(f"| `{name}` | {gps} | {gpp} | {ais} |")
    lines.append("")
    lines.append("## 4. Utilisation DSL")
    lines.append("")
    for m in methods:
        name = m["name"]
        usage = summary.get(name, {}).get("family_usage", {})
        if usage:
            lines.append(f"- `{name}`: `{json.dumps(usage, ensure_ascii=True)}`")
    lines.append("")
    lines.append("## 5. Artefacts")
    lines.append("")
    lines.append(f"- JSON brut: `{path.parent / 'ladder_results.json'}`")
    lines.append("")
    path.write_text("\n".join(lines) + "\n")


def build_methods(small_ckpt: Path, untrained_ckpt: Path, curriculum_ckpt: Path, macros_path: Path, include_macros: bool, macro_sizes: List[int]) -> List[MethodCfg]:
    methods = [
        MethodCfg(name="greedy_best_of_orders", kind="greedy"),
        MethodCfg(name="fixed_tabu_recolor", kind="fixed_tabu"),
        MethodCfg(name="mcts_no_model", kind="mcts"),
        MethodCfg(name="mcts_untrained", kind="mcts", ckpt=str(untrained_ckpt)),
        MethodCfg(name="mcts_trained_small", kind="mcts", ckpt=str(small_ckpt)),
        MethodCfg(name="mcts_trained_curriculum", kind="mcts", ckpt=str(curriculum_ckpt)),
    ]
    if include_macros:
        methods.append(
            MethodCfg(
                name="mcts_trained_small_macros",
                kind="mcts",
                ckpt=str(small_ckpt),
                macros=str(macros_path),
                enabled_sizes=macro_sizes,
            )
        )
    return methods


def main() -> None:
    parser = argparse.ArgumentParser(description="Small-to-large ladder runner (v2)")
    parser.add_argument("--root", type=str, default=str(Path(__file__).resolve().parent))
    parser.add_argument("--base-session", type=str, default="runs/gcp_trm_scaleup_v2/session_1776550919")
    parser.add_argument("--curriculum-ckpt", type=str, default="runs/gcp_trm_scaleup_v2/session_1776550919/n80/train_run/model-best.pt")
    parser.add_argument("--sizes", type=str, default="100,200,400")
    parser.add_argument("--per-size", type=int, default=10)
    parser.add_argument("--budgets", type=str, default="12:24,32:48,64:96")
    parser.add_argument("--seed", type=int, default=20260420)
    parser.add_argument("--max-tries", type=int, default=220)
    parser.add_argument("--min-greedy-conflicts", type=int, default=2)
    parser.add_argument("--max-greedy-conflicts", type=int, default=4000)
    parser.add_argument("--random-orders", type=int, default=8)
    parser.add_argument("--solve-timeout-sec", type=int, default=180)
    parser.add_argument("--action-budget", type=int, default=128)
    parser.add_argument("--fixed-max-steps", type=int, default=96)
    parser.add_argument("--train-epochs", type=int, default=4)
    parser.add_argument("--train-steps-per-epoch", type=int, default=1200)
    parser.add_argument("--train-valid-steps", type=int, default=200)
    parser.add_argument("--train-batch-size", type=int, default=64)
    parser.add_argument("--include-macro-method", action="store_true")
    parser.add_argument("--macro-sizes", type=str, default="100")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    base_session = (root / args.base_session).resolve()
    curriculum_ckpt = (root / args.curriculum_ckpt).resolve()
    sizes = parse_sizes(args.sizes)
    budgets = parse_budgets(args.budgets)
    macro_sizes = parse_sizes(args.macro_sizes)

    ts = int(time.time())
    out_dir = root / "runs" / "gcp_trm_scaleup_v2" / f"small_to_large_ladder_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    small_dir = out_dir / "small_train"
    train_shard = small_dir / "train_merged.pt"
    valid_shard = small_dir / "valid_merged.pt"
    train_count = concat_shards(
        sorted((base_session / "n20" / "traces_train").glob("*.pt")) + sorted((base_session / "n40" / "traces_train").glob("*.pt")),
        train_shard,
    )
    valid_count = concat_shards(
        sorted((base_session / "n20" / "traces_valid").glob("*.pt")) + sorted((base_session / "n40" / "traces_valid").glob("*.pt")),
        valid_shard,
    )

    model_dir = out_dir / "trained_small"
    run_cmd(
        [
            "python3",
            "gcp_trace_abstractbeam_v2.py",
            "train",
            "--train",
            str(train_shard),
            "--valid",
            str(valid_shard),
            "--out-dir",
            str(model_dir),
            "--batch-size",
            str(int(args.train_batch_size)),
            "--epochs",
            str(int(args.train_epochs)),
            "--steps-per-epoch",
            str(int(args.train_steps_per_epoch)),
            "--valid-steps",
            str(int(args.train_valid_steps)),
            "--seed",
            str(int(args.seed) + 11),
            "--amp",
        ],
        root,
    )
    small_ckpt = model_dir / "model-best.pt"
    if not small_ckpt.exists():
        small_ckpt = model_dir / "model-last.pt"

    macros_path = out_dir / "small_macros.json"
    run_cmd(
        [
            "python3",
            "gcp_trace_abstractbeam_v2.py",
            "mine-macros",
            "--trace",
            str(train_shard),
            "--out",
            str(macros_path),
            "--min-support",
            "12",
            "--max-len",
            "5",
            "--top-k",
            "64",
        ],
        root,
    )

    untrained_ckpt = out_dir / "untrained_trm.pt"
    make_untrained_checkpoint(untrained_ckpt)

    instances = generate_instances(
        sizes=sizes,
        per_size=int(args.per_size),
        max_tries=int(args.max_tries),
        min_greedy_conflicts=int(args.min_greedy_conflicts),
        max_greedy_conflicts=int(args.max_greedy_conflicts),
        random_orders=int(args.random_orders),
        seed=int(args.seed) + 37,
    )

    methods = build_methods(
        small_ckpt=small_ckpt,
        untrained_ckpt=untrained_ckpt,
        curriculum_ckpt=curriculum_ckpt,
        macros_path=macros_path,
        include_macros=bool(args.include_macro_method),
        macro_sizes=macro_sizes,
    )

    results: List[Dict[str, object]] = []
    rng = random.Random(int(args.seed) + 73)
    for inst in instances:
        if "error" in inst:
            for sim, dep in budgets:
                for method in methods:
                    results.append(
                        {
                            "instance": inst.get("name"),
                            "n": int(inst.get("n", -1)),
                            "method": method.name,
                            "simulations": int(sim),
                            "max_depth": int(dep),
                            "error": str(inst["error"]),
                        }
                    )
            continue
        edges = np.asarray(inst["edges"], dtype=np.int64)
        graph = gcp.GraphRecord(name=str(inst["name"]), n=int(inst["n"]), edges=edges, solution=None, metadata={}).to_runtime()
        k = int(inst["k"])
        greedy_best = int(inst.get("greedy_k_best_conflicts", -1))
        if greedy_best < 0:
            greedy_best, _ = hb.best_greedy_k_conflicts(graph, k, rng, num_random_orders=int(args.random_orders))
            greedy_best = int(greedy_best)
        for sim, dep in budgets:
            for method in methods:
                if method.enabled_sizes is not None and int(inst["n"]) not in method.enabled_sizes:
                    continue
                row: Dict[str, object] = {
                    "instance": str(inst["name"]),
                    "n": int(inst["n"]),
                    "k": int(k),
                    "m": int(inst.get("m", graph.m)),
                    "method": method.name,
                    "simulations": int(sim),
                    "max_depth": int(dep),
                    "baseline_greedy_k_conflicts": int(greedy_best),
                }
                t0 = time.time()
                try:
                    if method.kind == "greedy":
                        conf = int(greedy_best)
                        row.update({"model_conflicts": conf, "solved": bool(conf == 0)})
                    elif method.kind == "fixed_tabu":
                        solved = fixed_tabu_recolor_solve(
                            graph=graph,
                            k=k,
                            max_steps=int(args.fixed_max_steps),
                            action_budget=int(args.action_budget),
                        )
                        row.update(
                            {
                                "model_conflicts": int(solved["conflicts"]),
                                "solved": bool(solved["solved"]),
                                "primitive_calls": int(solved["primitive_calls"]),
                                "family_usage": solved["family_usage"],
                            }
                        )
                    else:
                        pred = solve_with_cli(
                            graph_dict=inst,
                            ckpt=Path(method.ckpt) if method.ckpt else None,
                            macros=Path(method.macros) if method.macros else None,
                            k=k,
                            simulations=int(sim),
                            max_depth=int(dep),
                            timeout_sec=int(args.solve_timeout_sec),
                            cwd=root,
                        )
                        conf = hb.count_conflicts(inst["edges"], pred["colors"])  # type: ignore[index]
                        row.update(
                            {
                                "model_conflicts": int(conf),
                                "solved": bool(conf == 0),
                                "model_reported_conflicts": int(pred.get("conflicts", -1)),
                                "core_size": int(pred.get("core_size", -1)),
                                "conflict_vertices": int(pred.get("conflict_vertices", -1)),
                                "primitive_calls": int(pred.get("primitive_calls", -1)),
                                "anytime_trace": pred.get("anytime_trace", []),
                            }
                        )
                except subprocess.TimeoutExpired:
                    row["error"] = "timeout"
                except Exception as exc:
                    row["error"] = str(exc)
                row["elapsed_sec"] = float(time.time() - t0)
                results.append(row)

    payload: Dict[str, object] = {
        "out_dir": str(out_dir),
        "config": vars(args),
        "artifacts": {
            "merged_train": str(train_shard),
            "merged_valid": str(valid_shard),
            "trained_small_ckpt": str(small_ckpt),
            "untrained_ckpt": str(untrained_ckpt),
            "macros": str(macros_path),
            "curriculum_ckpt": str(curriculum_ckpt),
            "train_rows": int(train_count),
            "valid_rows": int(valid_count),
        },
        "instances": instances,
        "methods": [m.__dict__ for m in methods],
        "results": results,
    }
    payload["summary"] = summarize(
        records=results,
        methods=methods,
        baseline="greedy_best_of_orders",
        n_boot=int(args.bootstrap_samples),
        seed=int(args.seed) + 101,
    )

    raw_path = out_dir / "ladder_results.json"
    raw_path.write_text(json.dumps(payload, indent=2))
    report_path = out_dir / "FINAL_LADDER_REPORT.md"
    write_markdown_report(report_path, payload)
    print(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "raw_results": str(raw_path),
                "report": str(report_path),
                "num_instances": len(instances),
                "num_results": len(results),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

