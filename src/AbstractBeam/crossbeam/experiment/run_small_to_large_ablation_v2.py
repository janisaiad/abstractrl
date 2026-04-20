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
from typing import Dict, Iterable, List, Optional

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


def run_cmd(command: List[str], cwd: Path) -> None:
    subprocess.run(command, cwd=str(cwd), check=True)


def parse_sizes(spec: str) -> List[int]:
    return [int(x.strip()) for x in spec.split(",") if x.strip()]


def concat_shards(shard_paths: Iterable[Path], out_path: Path) -> int:
    rows: List[dict] = []
    for shard in shard_paths:
        payload = torch.load(shard, map_location="cpu", weights_only=False)
        if not isinstance(payload, list):
            continue
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


def greedy_conflicts(graph: gcp.GCGraph, k: int, rng: random.Random, random_orders: int) -> int:
    best, _ = hb.best_greedy_k_conflicts(graph, k, rng, num_random_orders=random_orders)
    return int(best)


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
            graph,
            state,
            metrics,
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
    cwd: Path,
) -> Dict[str, object]:
    public = {"name": graph_dict["name"], "n": graph_dict["n"], "edges": graph_dict["edges"]}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(public, f)
        json_path = Path(f.name)
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
    ]
    if ckpt is not None and str(ckpt):
        args.extend(["--ckpt", str(ckpt)])
    if macros is not None and str(macros):
        args.extend(["--macros", str(macros)])
    try:
        out = subprocess.check_output(args, cwd=str(cwd), text=True)
        return json.loads(out)
    finally:
        json_path.unlink(missing_ok=True)


def schedule_for_n(n: int) -> tuple[int, float]:
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
                rows.append(
                    {
                        "name": f"hard_n{n}_{i:03d}",
                        "n": int(n),
                        "k": int(k),
                        "p_cross": float(p_cross),
                        "error": "no_hard_instance_found",
                    }
                )
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


def method_table(
    out_dir: Path,
    small_best_ckpt: Path,
    untrained_ckpt: Path,
    macros_path: Path,
    curriculum_ckpt: Path,
    include_macro_method: bool,
) -> List[MethodCfg]:
    methods = [
        MethodCfg(name="greedy_best_of_orders", kind="greedy"),
        MethodCfg(name="fixed_tabu_recolor", kind="fixed_tabu"),
        MethodCfg(name="mcts_no_model", kind="mcts", ckpt=""),
        MethodCfg(name="mcts_untrained", kind="mcts", ckpt=str(untrained_ckpt)),
        MethodCfg(name="mcts_trained_small", kind="mcts", ckpt=str(small_best_ckpt)),
        MethodCfg(name="mcts_trained_curriculum", kind="mcts", ckpt=str(curriculum_ckpt)),
    ]
    if include_macro_method:
        methods.append(MethodCfg(name="mcts_trained_small_macros", kind="mcts", ckpt=str(small_best_ckpt), macros=str(macros_path)))
    return methods


def summarize(records: List[Dict[str, object]], methods: List[MethodCfg]) -> Dict[str, object]:
    by_method: Dict[str, Dict[str, object]] = {}
    for method in methods:
        method_rows = [r for r in records if r.get("method") == method.name and "error" not in r]
        solved = sum(1 for r in method_rows if bool(r.get("solved", False)))
        total = len(method_rows)
        mean_conf = float(np.mean([float(r.get("model_conflicts", 0.0)) for r in method_rows])) if total else float("nan")
        mean_time = float(np.mean([float(r.get("elapsed_sec", 0.0)) for r in method_rows])) if total else float("nan")
        prim_calls_rows = [float(r["primitive_calls"]) for r in method_rows if "primitive_calls" in r]
        usage = defaultdict(int)
        for row in method_rows:
            fam = row.get("family_usage", {})
            if isinstance(fam, dict):
                for k, v in fam.items():
                    usage[str(k)] += int(v)
        by_method[method.name] = {
            "num_instances": total,
            "solved_rigorous": solved,
            "solved_rate": (solved / total) if total else None,
            "mean_model_conflicts": mean_conf if total else None,
            "mean_elapsed_sec": mean_time if total else None,
            "mean_primitive_calls": (sum(prim_calls_rows) / len(prim_calls_rows)) if prim_calls_rows else None,
            "family_usage": dict(sorted(usage.items(), key=lambda kv: kv[1], reverse=True)),
        }
    return {"by_method": by_method}


def write_markdown_report(path: Path, payload: Dict[str, object]) -> None:
    summary = payload["summary"]["by_method"]  # type: ignore[index]
    instances = payload["instances"]  # type: ignore[index]
    methods = payload["methods"]  # type: ignore[index]

    lines: List[str] = []
    lines.append("# Rapport final — Small-to-Large ablation v2")
    lines.append("")
    lines.append("## 1. Protocole exécuté")
    lines.append("")
    lines.append(f"- Nombre d'instances hard générées: **{len(instances)}**")
    lines.append(f"- Méthodes évaluées: **{len(methods)}**")
    lines.append("- Évaluation rigoureuse: conflits recalculés sur arêtes (`model_conflicts`).")
    lines.append("")
    lines.append("## 2. Résultats agrégés")
    lines.append("")
    lines.append("| Méthode | solved_rate | mean_conflicts | mean_time_sec | mean_primitive_calls |")
    lines.append("|---|---:|---:|---:|---:|")
    for m in methods:
        name = m["name"]
        row = summary.get(name, {})
        solved_rate = row.get("solved_rate")
        mean_conf = row.get("mean_model_conflicts")
        mean_time = row.get("mean_elapsed_sec")
        mean_calls = row.get("mean_primitive_calls")
        lines.append(
            f"| `{name}` | "
            f"{'n/a' if solved_rate is None else f'{float(solved_rate):.3f}'} | "
            f"{'n/a' if mean_conf is None else f'{float(mean_conf):.3f}'} | "
            f"{'n/a' if mean_time is None else f'{float(mean_time):.3f}'} | "
            f"{'n/a' if mean_calls is None else f'{float(mean_calls):.3f}'} |"
        )
    lines.append("")
    lines.append("## 3. Utilisation DSL (familles)")
    lines.append("")
    for m in methods:
        name = m["name"]
        row = summary.get(name, {})
        usage = row.get("family_usage", {})
        if not usage:
            continue
        lines.append(f"- `{name}`: `{json.dumps(usage, ensure_ascii=True)}`")
    lines.append("")
    lines.append("## 4. Notes")
    lines.append("")
    lines.append("- `greedy_best_of_orders` correspond au meilleur glouton multi-ordres.")
    lines.append("- `fixed_tabu_recolor` applique un scheduler déterministe (recolor/tabu) sans réseau.")
    lines.append("- Les variantes `mcts_*` partagent les mêmes budgets de recherche.")
    lines.append("- Les détails instance-par-instance sont dans le JSON brut du même dossier.")
    lines.append("")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Small-to-large ablation runner for GCP v2")
    parser.add_argument("--root", type=str, default=str(Path(__file__).resolve().parent))
    parser.add_argument("--base-session", type=str, default="runs/gcp_trm_scaleup_v2/session_1776550919")
    parser.add_argument("--curriculum-ckpt", type=str, default="runs/gcp_trm_scaleup_v2/session_1776550919/n80/train_run/model-best.pt")
    parser.add_argument("--sizes", type=str, default="100,200,400")
    parser.add_argument("--per-size", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260419)
    parser.add_argument("--max-tries", type=int, default=160)
    parser.add_argument("--min-greedy-conflicts", type=int, default=2)
    parser.add_argument("--max-greedy-conflicts", type=int, default=3000)
    parser.add_argument("--random-orders", type=int, default=6)
    parser.add_argument("--simulations", type=int, default=48)
    parser.add_argument("--max-depth", type=int, default=72)
    parser.add_argument("--action-budget", type=int, default=128)
    parser.add_argument("--fixed-max-steps", type=int, default=72)
    parser.add_argument("--train-epochs", type=int, default=3)
    parser.add_argument("--train-steps-per-epoch", type=int, default=900)
    parser.add_argument("--train-valid-steps", type=int, default=140)
    parser.add_argument("--train-batch-size", type=int, default=64)
    parser.add_argument("--include-macro-method", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    base_session = (root / args.base_session).resolve()
    curriculum_ckpt = (root / args.curriculum_ckpt).resolve()
    ts = int(time.time())
    out_dir = root / "runs" / "gcp_trm_scaleup_v2" / f"small_to_large_ablation_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    small_dir = out_dir / "small_train"
    train_shard = small_dir / "train_merged.pt"
    valid_shard = small_dir / "valid_merged.pt"
    train_count = concat_shards(sorted((base_session / "n20" / "traces_train").glob("*.pt")) + sorted((base_session / "n40" / "traces_train").glob("*.pt")), train_shard)
    valid_count = concat_shards(sorted((base_session / "n20" / "traces_valid").glob("*.pt")) + sorted((base_session / "n40" / "traces_valid").glob("*.pt")), valid_shard)

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
    small_best_ckpt = model_dir / "model-best.pt"
    if not small_best_ckpt.exists():
        small_best_ckpt = model_dir / "model-last.pt"

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
    make_untrained_checkpoint(untrained_ckpt, d_model=256, refine_steps=3, dropout=0.1)

    instances = generate_instances(
        sizes=parse_sizes(args.sizes),
        per_size=int(args.per_size),
        max_tries=int(args.max_tries),
        min_greedy_conflicts=int(args.min_greedy_conflicts),
        max_greedy_conflicts=int(args.max_greedy_conflicts),
        random_orders=int(args.random_orders),
        seed=int(args.seed) + 37,
    )

    methods = method_table(
        out_dir=out_dir,
        small_best_ckpt=small_best_ckpt,
        untrained_ckpt=untrained_ckpt,
        macros_path=macros_path,
        curriculum_ckpt=curriculum_ckpt,
        include_macro_method=bool(args.include_macro_method),
    )
    results: List[Dict[str, object]] = []
    rng = random.Random(int(args.seed) + 73)
    for inst in instances:
        if "error" in inst:
            for method in methods:
                results.append(
                    {
                        "instance": inst.get("name"),
                        "method": method.name,
                        "error": str(inst["error"]),
                    }
                )
            continue
        edges = np.asarray(inst["edges"], dtype=np.int64)
        graph = gcp.GraphRecord(
            name=str(inst["name"]),
            n=int(inst["n"]),
            edges=edges,
            solution=None,
            metadata={},
        ).to_runtime()
        k = int(inst["k"])
        greedy_best = int(inst.get("greedy_k_best_conflicts", -1))
        if greedy_best < 0:
            greedy_best = greedy_conflicts(graph, k, rng, random_orders=int(args.random_orders))
        for method in methods:
            t0 = time.time()
            row: Dict[str, object] = {
                "instance": str(inst["name"]),
                "n": int(inst["n"]),
                "k": int(k),
                "m": int(inst.get("m", graph.m)),
                "method": method.name,
                "baseline_greedy_k_conflicts": int(greedy_best),
            }
            try:
                if method.kind == "greedy":
                    conf = int(greedy_best)
                    row.update(
                        {
                            "model_conflicts": conf,
                            "solved": bool(conf == 0),
                            "core_size": None,
                            "conflict_vertices": None,
                        }
                    )
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
                            "core_size": solved["core_size"],
                            "conflict_vertices": solved["conflict_vertices"],
                            "primitive_calls": solved["primitive_calls"],
                            "family_usage": solved["family_usage"],
                        }
                    )
                else:
                    pred = solve_with_cli(
                        graph_dict=inst,
                        ckpt=Path(method.ckpt) if method.ckpt else None,
                        macros=Path(method.macros) if method.macros else None,
                        k=k,
                        simulations=int(args.simulations),
                        max_depth=int(args.max_depth),
                        cwd=root,
                    )
                    conf = hb.count_conflicts(inst["edges"], pred["colors"])  # type: ignore[index]
                    row.update(
                        {
                            "model_conflicts": int(conf),
                            "solved": bool(conf == 0),
                            "core_size": int(pred.get("core_size", -1)),
                            "conflict_vertices": int(pred.get("conflict_vertices", -1)),
                            "model_reported_conflicts": int(pred.get("conflicts", -1)),
                        }
                    )
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
            "trained_small_ckpt": str(small_best_ckpt),
            "untrained_ckpt": str(untrained_ckpt),
            "macros": str(macros_path),
            "curriculum_ckpt": str(curriculum_ckpt),
            "train_rows": train_count,
            "valid_rows": valid_count,
        },
        "instances": instances,
        "methods": [m.__dict__ for m in methods],
        "results": results,
    }
    payload["summary"] = summarize(results, methods)

    raw_path = out_dir / "ablation_results.json"
    raw_path.write_text(json.dumps(payload, indent=2))
    report_path = out_dir / "FINAL_SMALL_TO_LARGE_REPORT.md"
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
