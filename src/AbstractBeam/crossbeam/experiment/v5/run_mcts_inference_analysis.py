#!/usr/bin/env python3
"""
Inférence MCTS + analyse (sans réentraînement).

Charge un checkpoint TRM déjà entraîné, résout un petit benchmark (JSONL de graphes),
écrit les dumps d'arbre MCTS, produit un rapport JSON lisible (primitives par arête,
stats N/Q/prior, action la plus visitée, lien parent→enfant), puis lance plot_mcts_trees.

Usage typique (petit benchmark, checkpoint "small" fort) :

  PYTHONPATH=/Data/janis.aiad/abstractrl/src/AbstractBeam/crossbeam/experiment \\
    python3 v5/run_mcts_inference_analysis.py \\
      --ckpt /Data/janis.aiad/abstractrl/src/AbstractBeam/crossbeam/experiment/v3/runs/gcp_trm_scaleup_v3/big_train_1776710200/model-best.pt \\
      --input /chemin/vers/eval.jsonl \\
      --out-dir /chemin/vers/run_mcts_infer_001 \\
      --max-graphs 8
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

EXPERIMENT_ROOT = Path(__file__).resolve().parent.parent
V3_ROOT = EXPERIMENT_ROOT / "v3"
V4_ROOT = EXPERIMENT_ROOT / "v4"


def _candidate_label(c: Dict[str, Any]) -> str:
    fam = str(c.get("family", "?"))
    sel = str(c.get("selector", ""))
    args = c.get("args") or []
    params = c.get("params") or []
    if sel:
        return f"{sel} | {fam} | args={args} | params={params}"
    return f"{fam} | args={args} | params={params}"


def _depths(nodes: List[Dict[str, Any]]) -> List[int]:
    n = len(nodes)
    parent = [int(x.get("parent", -1)) for x in nodes]
    depth = [-1] * n
    for i in range(n):
        if depth[i] >= 0:
            continue
        cur = i
        stack: List[int] = []
        while 0 <= cur < n and depth[cur] < 0:
            stack.append(cur)
            p = parent[cur]
            if p == cur:
                p = -1
            cur = p
        base = 0 if cur < 0 or cur >= n else depth[cur] + 1
        for nid in reversed(stack):
            depth[nid] = base
            base += 1
    for i in range(n):
        if depth[i] < 0:
            depth[i] = 0
    return depth


def _approx_selection_score(
    br: Dict[str, Any],
    total_visits_root: int,
    cpuct: float,
    search_alpha_mean: float,
    search_beta_max: float,
    novelty_coef: float,
) -> float:
    n = int(br.get("edge_visits_n", 0))
    q_mean = float(br.get("edge_mean_q", 0.0))
    q_max = float(br.get("edge_max_q", 0.0))
    if q_max <= -1e17:
        q_max = q_mean
    prior = float(br.get("prior", 0.0))
    q_dist = float(br.get("edge_q_distinct", 0.0))
    total_n = max(total_visits_root, 1)
    bonus = cpuct * prior * math.sqrt(total_n) / (1.0 + n)
    novelty = novelty_coef / math.sqrt(1.0 + float(n))
    return search_alpha_mean * q_mean + search_beta_max * q_max + bonus + novelty + 0.05 * q_dist


def build_narrative_report(
    tree: Dict[str, Any],
    *,
    cpuct: float,
    search_alpha_mean: float,
    search_beta_max: float,
    novelty_coef: float,
) -> Dict[str, Any]:
    nodes = tree.get("nodes") or []
    depths = _depths(nodes)
    out_nodes: List[Dict[str, Any]] = []
    for nid, node in enumerate(nodes):
        branches = node.get("branches") or []
        total_v = int(node.get("visit_count_V", 0))
        br_enriched: List[Dict[str, Any]] = []
        best_vis_a = -1
        best_vis_n = -1
        for br in branches:
            na = int(br.get("edge_visits_n", 0))
            if na > best_vis_n:
                best_vis_n = na
                best_vis_a = int(br.get("action_index", -1))
            cand = br.get("candidate") or {}
            br_enriched.append(
                {
                    "action_index": int(br.get("action_index", -1)),
                    "primitive_label": _candidate_label(cand),
                    "candidate": cand,
                    "edge_visits_n": na,
                    "edge_mean_q": float(br.get("edge_mean_q", 0.0)),
                    "edge_max_q": float(br.get("edge_max_q", 0.0)),
                    "edge_q_distinct": float(br.get("edge_q_distinct", 0.0)),
                    "prior": float(br.get("prior", 0.0)),
                    "alive": bool(br.get("alive", False)),
                    "child_node_id": int(br.get("child_node_id", -1)),
                    "approx_selection_score": _approx_selection_score(
                        br, total_v, cpuct, search_alpha_mean, search_beta_max, novelty_coef
                    ),
                }
            )
        br_sorted = sorted(br_enriched, key=lambda x: (-x["edge_visits_n"], -x["approx_selection_score"]))
        parent_action = int(node.get("parent_action", -1))
        parent_id = int(node.get("parent", -1))
        incoming = None
        if 0 <= parent_id < len(nodes):
            pbrs = nodes[parent_id].get("branches") or []
            for pb in pbrs:
                if int(pb.get("action_index", -2)) == parent_action:
                    incoming = _candidate_label((pb.get("candidate") or {}))
                    break
        out_nodes.append(
            {
                "node_id": nid,
                "depth": int(depths[nid]) if nid < len(depths) else 0,
                "parent": parent_id,
                "parent_action": parent_action,
                "reached_via_primitive": incoming,
                "visit_count_V": total_v,
                "value_mean_V": float(node.get("value_mean_V", 0.0)),
                "terminal": bool(node.get("terminal", False)),
                "solved": bool(node.get("solved", False)),
                "frozen": bool(node.get("frozen", False)),
                "dead": bool(node.get("dead", False)),
                "best_terminal_conflicts": int(node.get("best_terminal_conflicts", 10**9)),
                "best_through": float(node.get("best_through", 0.0)),
                "most_visited_action_index": best_vis_a,
                "branches_ranked_by_visits_then_score": br_sorted,
            }
        )
    return {
        "format": "mcts_inference_narrative_v1",
        "source_tree_format": tree.get("format"),
        "graph_name": tree.get("graph_name"),
        "graph_n": tree.get("graph_n"),
        "k": tree.get("k"),
        "num_nodes": len(nodes),
        "selection_proxy": {
            "note": "approx_selection_score reproduit la forme search (mean/max + UCB-like bonus + novelty + petit terme distinct) avec les hyperparamètres passés au solve; ce n'est pas log ligne par ligne des choix effectifs pendant les threads.",
            "cpuct": cpuct,
            "search_alpha_mean": search_alpha_mean,
            "search_beta_max": search_beta_max,
            "novelty_coef": novelty_coef,
        },
        "nodes": out_nodes,
    }


def run_cmd(cmd: List[str], cwd: Path, extra_env: Optional[Dict[str, str]] = None) -> None:
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    subprocess.run(cmd, cwd=str(cwd), check=True, env=env)


def main() -> None:
    ap = argparse.ArgumentParser(description="MCTS inference-only analysis on a benchmark JSONL (no training).")
    ap.add_argument("--ckpt", type=str, required=True, help="Path to model-best.pt (or model-last.pt)")
    ap.add_argument("--input", type=str, required=True, help="JSONL of graph records (name, n, edges, optional solution)")
    ap.add_argument("--out-dir", type=str, required=True, help="Output directory for trees, reports, plots")
    ap.add_argument("--max-graphs", type=int, default=10, help="Max lines to process from JSONL")
    ap.add_argument("--k", type=int, default=0, help="Fixed k for solve (0 => use clique/dsatur heuristic inside solver)")
    ap.add_argument("--simulations", type=int, default=320)
    ap.add_argument("--max-depth", type=int, default=96)
    ap.add_argument("--search-mode", type=str, default="infer", choices=["infer", "collect", "noprior"])
    ap.add_argument("--search-alpha-mean", type=float, default=0.75)
    ap.add_argument("--search-beta-max", type=float, default=0.25)
    ap.add_argument("--novelty-coef", type=float, default=0.05)
    ap.add_argument("--cpuct", type=float, default=1.25)
    ap.add_argument("--worker-count", type=int, default=2)
    ap.add_argument(
        "--mcts-sim-trace",
        type=str,
        default="aggregates",
        choices=["off", "aggregates", "full"],
        help="Per-simulation instrumentation in tree JSON (full = step paths, larger files)",
    )
    ap.add_argument("--mcts-sim-trace-cap", type=int, default=50000)
    ap.add_argument("--skip-plots", action="store_true", help="Do not run plot_mcts_trees.py")
    args = ap.parse_args()

    ckpt = Path(args.ckpt).expanduser()
    if not ckpt.is_file():
        raise FileNotFoundError(str(ckpt))
    bench = Path(args.input).expanduser()
    if not bench.is_file():
        raise FileNotFoundError(str(bench))
    out = Path(args.out_dir).expanduser()
    trees_dir = out / "mcts_trees"
    reports_dir = out / "reports"
    plots_dir = out / "plots_plot_mcts_trees"
    trees_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    with bench.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if len(rows) >= int(args.max_graphs):
                break

    index: List[Dict[str, Any]] = []
    for rec in rows:
        name = str(rec.get("name", "graph"))
        tmp = out / "_tmp_input" / f"{name}.json"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps({"name": name, "n": rec.get("n"), "edges": rec.get("edges", [])}))
        tree_path = trees_dir / f"{name}.mcts_tree.json"
        solve_out = reports_dir / f"{name}_solve.json"
        cmd = [
            "python",
            str(V3_ROOT / "gcp_trace_abstractbeam_v3.py"),
            "solve",
            "--input",
            str(tmp),
            "--ckpt",
            str(ckpt),
            "--simulations",
            str(int(args.simulations)),
            "--max-depth",
            str(int(args.max_depth)),
            "--search-mode",
            str(args.search_mode),
            "--search-alpha-mean",
            str(float(args.search_alpha_mean)),
            "--search-beta-max",
            str(float(args.search_beta_max)),
            "--novelty-coef",
            str(float(args.novelty_coef)),
            "--cpuct",
            str(float(args.cpuct)),
            "--worker-count",
            str(int(args.worker_count)),
            "--mcts-sim-trace",
            str(args.mcts_sim_trace),
            "--mcts-sim-trace-cap",
            str(int(args.mcts_sim_trace_cap)),
            "--mcts-tree-dump",
            str(tree_path),
            "--out",
            str(solve_out),
        ]
        if int(args.k) > 0:
            cmd.extend(["--k", str(int(args.k))])
        run_cmd(cmd, V3_ROOT, extra_env={"PYTHONPATH": str(EXPERIMENT_ROOT)})
        if not tree_path.is_file():
            raise RuntimeError(f"Missing tree dump: {tree_path}")
        tree_payload = json.loads(tree_path.read_text())
        narrative = build_narrative_report(
            tree_payload,
            cpuct=float(args.cpuct),
            search_alpha_mean=float(args.search_alpha_mean),
            search_beta_max=float(args.search_beta_max),
            novelty_coef=float(args.novelty_coef),
        )
        rep_path = reports_dir / f"{name}_narrative.json"
        rep_path.write_text(json.dumps(narrative, indent=2), encoding="utf-8")
        root0 = (tree_payload.get("nodes") or [{}])[0]
        metrics = root0.get("metrics") or {}
        solve_conflicts = None
        if solve_out.is_file():
            try:
                sol = json.loads(solve_out.read_text())
                solve_conflicts = sol.get("conflicts")
            except json.JSONDecodeError:
                solve_conflicts = None
        index.append(
            {
                "name": name,
                "tree": str(tree_path),
                "narrative": str(rep_path),
                "solve_out": str(solve_out),
                "root_conflicts_from_tree_metrics": metrics.get("conflicts"),
                "solve_conflicts": solve_conflicts,
                "num_nodes": tree_payload.get("num_nodes"),
            }
        )

    (out / "inference_index.json").write_text(json.dumps({"graphs": index}, indent=2), encoding="utf-8")

    if not args.skip_plots:
        run_cmd(
            [
                "python3",
                str(V4_ROOT / "plot_mcts_trees.py"),
                "--inputs",
                str(trees_dir),
                "--out-dir",
                str(plots_dir),
            ],
            cwd=str(V4_ROOT),
            extra_env={"PYTHONPATH": str(EXPERIMENT_ROOT)},
        )

    print(json.dumps({"out_dir": str(out), "n_graphs": len(index), "trees": str(trees_dir), "reports": str(reports_dir), "plots": str(plots_dir)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
