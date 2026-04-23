#!/usr/bin/env python3
"""
Agrège des dumps d'arbres MCTS (JSON `gcp_mcts_tree_v2`) pour reverse-engineer un run sanity.

Parcourt récursivement des répertoires ou des globs, fusionne les histogrammes d'instrumentation
(`mcts_instrumentation` si présent), résume la structure des arbres (profondeur, familles de
primitives à la racine, visites racine), et écrit un JSON unique de synthèse.

Exemple :

  python3 v5/aggregate_mcts_sanity_stats.py \\
    --inputs /path/to/session/mcts_trees /path/to/other/**/*.mcts_tree.json \\
    --out-json /path/to/session/mcts_sanity_complete_stats.json
"""

from __future__ import annotations

import argparse
import glob as globmod
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


def _expand_inputs(raw: Sequence[str]) -> List[Path]:
    out: List[Path] = []
    for item in raw:
        p = Path(item).expanduser()
        if p.is_file() and p.suffix == ".json":
            out.append(p)
            continue
        if any(ch in str(p) for ch in "*?["):
            for x in sorted(globmod.glob(str(p), recursive=True)):
                if str(x).endswith(".json"):
                    out.append(Path(x))
            continue
        if p.is_dir():
            out.extend(sorted(p.rglob("*.json")))
    seen: set[str] = set()
    uniq: List[Path] = []
    for q in out:
        key = str(q.resolve())
        if key in seen:
            continue
        seen.add(key)
        uniq.append(q)
    return uniq


def _merge_int_hist(dst: Dict[int, int], src: Dict[str, Any]) -> None:
    for k, v in src.items():
        try:
            dst[int(k)] += int(v)
        except (TypeError, ValueError):
            continue


def _merge_str_hist(dst: Counter[str], src: Dict[str, Any]) -> None:
    for k, v in src.items():
        dst[str(k)] += int(v)


def _percentile_nearest(xs: List[float], q: float) -> float:
    if not xs:
        return float("nan")
    ys = sorted(xs)
    if len(ys) == 1:
        return float(ys[0])
    idx = int(round(q * (len(ys) - 1)))
    idx = max(0, min(len(ys) - 1, idx))
    return float(ys[idx])


def _root_family_hist(nodes: List[Dict[str, Any]]) -> Counter[str]:
    c: Counter[str] = Counter()
    if not nodes:
        return c
    root = nodes[0]
    for br in root.get("branches") or []:
        cand = br.get("candidate") or {}
        fam = str(cand.get("family", "?"))
        c[fam] += 1
    return c


def _root_visit_stats(nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not nodes:
        return {"n_actions": 0, "max_edge_n": 0, "entropy_prior": float("nan")}
    root = nodes[0]
    branches = root.get("branches") or []
    ns = [int(b.get("edge_visits_n", 0)) for b in branches]
    pr = [float(b.get("prior", 0.0)) for b in branches]
    tot_n = sum(ns)
    max_n = max(ns) if ns else 0
    ent = float("nan")
    if pr and sum(pr) > 0:
        s = float(sum(pr))
        ent = 0.0
        for p in pr:
            x = max(float(p) / s, 1e-12)
            ent -= x * math.log(x)
    return {
        "n_actions": int(len(branches)),
        "total_edge_visits_at_root": int(tot_n),
        "max_edge_visits_at_root": int(max_n),
        "prior_entropy": float(ent),
    }


def _walk_one_tree(path: Path) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        return {"path": str(path), "error": str(e)}, None
    fmt = str(payload.get("format", ""))
    if not fmt.startswith("gcp_mcts_tree"):
        return {"path": str(path), "error": f"unexpected format {fmt!r}"}, None
    nodes = payload.get("nodes") or []
    inst = payload.get("mcts_instrumentation")
    row: Dict[str, Any] = {
        "path": str(path),
        "graph_name": payload.get("graph_name"),
        "graph_n": payload.get("graph_n"),
        "graph_m": payload.get("graph_m"),
        "k": payload.get("k"),
        "num_nodes": int(payload.get("num_nodes", len(nodes))),
        "root_conflicts": None,
        "instrumentation_mode": None,
        "legacy_no_instrumentation": inst is None,
    }
    if nodes:
        m0 = (nodes[0].get("metrics") or {}) if isinstance(nodes[0], dict) else {}
        if isinstance(m0, dict) and "conflicts" in m0:
            row["root_conflicts"] = int(m0["conflicts"])
    if isinstance(inst, dict):
        row["instrumentation_mode"] = inst.get("mode")
        run = inst.get("run") or {}
        row["wall_seconds"] = float(run.get("wall_seconds", 0.0) or 0.0)
        row["simulations_done"] = int(run.get("simulations_done", 0) or 0)
        row["early_stop_solved"] = bool(run.get("early_stop_solved", False))
        row["expand_tt_reuse"] = int(run.get("expand_tt_reuse", 0) or 0)
        row["expand_model_eval"] = int(run.get("expand_model_eval", 0) or 0)
        row["expand_trivial_terminal"] = int(run.get("expand_trivial_terminal", 0) or 0)
    row["root_families"] = dict(_root_family_hist(nodes))
    row["root_visit_stats"] = _root_visit_stats(nodes)
    return row, inst if isinstance(inst, dict) else None


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate MCTS tree JSON dumps into one reverse-engineering report.")
    ap.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Directories (recursive *.json) and/or globs pointing to gcp_mcts_tree JSON files",
    )
    ap.add_argument("--out-json", type=str, required=True, help="Output path for the consolidated stats JSON")
    args = ap.parse_args()

    paths = _expand_inputs(args.inputs)
    per_tree: List[Dict[str, Any]] = []
    errors = 0
    depth_hist: Dict[int, int] = defaultdict(int)
    outcome_hist: Counter[str] = Counter()
    root_action_hist: Dict[int, int] = defaultdict(int)
    forced_hist: Counter[str] = Counter()
    tt_join_rollout_hist: Dict[int, int] = defaultdict(int)
    edge_reuse_rollout_hist: Dict[int, int] = defaultdict(int)
    new_child_rollout_hist: Dict[int, int] = defaultdict(int)
    family_total: Counter[str] = Counter()
    tot_expand_tt = 0
    tot_expand_model = 0
    tot_expand_trivial = 0
    tot_sims_done = 0
    tot_wall = 0.0
    early_solves = 0
    legacy = 0
    num_nodes_list: List[int] = []

    for p in paths:
        row, inst = _walk_one_tree(p)
        per_tree.append(row)
        if "error" in row:
            errors += 1
            continue
        num_nodes_list.append(int(row.get("num_nodes", 0)))
        for fam, c in (row.get("root_families") or {}).items():
            family_total[str(fam)] += int(c)
        if row.get("legacy_no_instrumentation"):
            legacy += 1
            continue
        tot_wall += float(row.get("wall_seconds", 0.0))
        tot_sims_done += int(row.get("simulations_done", 0))
        if row.get("early_stop_solved"):
            early_solves += 1
        tot_expand_tt += int(row.get("expand_tt_reuse", 0))
        tot_expand_model += int(row.get("expand_model_eval", 0))
        tot_expand_trivial += int(row.get("expand_trivial_terminal", 0))
        if inst is None or str(inst.get("mode", "off")) == "off":
            continue
        _merge_int_hist(depth_hist, inst.get("sim_depth_hist") or {})
        _merge_str_hist(outcome_hist, inst.get("sim_outcome_hist") or {})
        _merge_int_hist(root_action_hist, inst.get("sim_root_action_hist") or {})
        _merge_str_hist(forced_hist, inst.get("sim_forced_root_hist") or {})
        _merge_int_hist(tt_join_rollout_hist, inst.get("sim_tt_joins_per_rollout_hist") or {})
        _merge_int_hist(edge_reuse_rollout_hist, inst.get("sim_edge_reuses_per_rollout_hist") or {})
        _merge_int_hist(new_child_rollout_hist, inst.get("sim_new_children_per_rollout_hist") or {})

    n_ok = len(per_tree) - errors
    summary_num_nodes: Dict[str, float] = {}
    if num_nodes_list:
        summary_num_nodes = {
            "mean": float(statistics.mean(num_nodes_list)),
            "median": float(statistics.median(num_nodes_list)),
            "p90": _percentile_nearest([float(x) for x in num_nodes_list], 0.9),
            "min": float(min(num_nodes_list)),
            "max": float(max(num_nodes_list)),
        }

    out_payload: Dict[str, Any] = {
        "n_files_seen": len(paths),
        "n_trees_parsed_ok": n_ok,
        "n_parse_errors": errors,
        "n_legacy_trees_without_mcts_instrumentation": legacy,
        "num_nodes_summary": summary_num_nodes,
        "rollups": {
            "sum_wall_seconds_trees_with_inst": float(tot_wall),
            "sum_simulations_done_trees_with_inst": int(tot_sims_done),
            "n_trees_early_stop_solved_flag": int(early_solves),
            "sum_expand_tt_reuse": int(tot_expand_tt),
            "sum_expand_model_eval": int(tot_expand_model),
            "sum_expand_trivial_terminal": int(tot_expand_trivial),
        },
        "merged_histograms": {
            "sim_depth": {str(k): int(v) for k, v in sorted(depth_hist.items(), key=lambda kv: kv[0])},
            "sim_outcome": dict(outcome_hist.most_common()),
            "sim_root_action": {str(k): int(v) for k, v in sorted(root_action_hist.items(), key=lambda kv: kv[0])},
            "sim_forced_root": dict(forced_hist.most_common()),
            "sim_tt_joins_per_rollout": {str(k): int(v) for k, v in sorted(tt_join_rollout_hist.items(), key=lambda kv: kv[0])},
            "sim_edge_reuses_per_rollout": {str(k): int(v) for k, v in sorted(edge_reuse_rollout_hist.items(), key=lambda kv: kv[0])},
            "sim_new_children_per_rollout": {str(k): int(v) for k, v in sorted(new_child_rollout_hist.items(), key=lambda kv: kv[0])},
        },
        "root_candidate_family_counts_across_trees": dict(family_total.most_common()),
        "per_tree": per_tree,
    }

    out_path = Path(str(args.out_json)).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_payload, indent=2), encoding="utf-8")
    print(f"wrote {out_path} ({len(per_tree)} rows)", flush=True)


if __name__ == "__main__":
    main()
