#!/usr/bin/env python3
from __future__ import annotations

"""
ConstructiveMDP prototype for k-coloring with optional DeltaBelief-style reward.

Purpose
-------
This file opens the constructive branch alongside the current repair pipeline.
It supports:
  * partial-coloring states with explicit domains
  * color-label symmetry canonicalization for partial states
  * DSATUR-style constructive candidate generation
  * dense constructive reward and optional DeltaBelief reward
  * teacher-trace generation from solved examples
  * simple beam-search constructive rollouts

This is an experimental prototype. It is intentionally standalone and does not
replace the current repair-based GCP pipeline.
"""

import argparse
import dataclasses
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

import numpy as np
import torch

import gcp_trace_abstractbeam_v3 as repair

EPS = 1e-9
DEFAULT_VERTEX_BUDGET = 32
DEFAULT_ACTION_BUDGET = 128


@dataclasses.dataclass
class ConstructiveState:
    colors: np.ndarray        # int16 [n], -1 = uncolored
    domains_mask: np.ndarray  # bool [n, k]
    k: int
    open_colors: int
    step: int = 0

    def copy(self) -> "ConstructiveState":
        return ConstructiveState(
            colors=self.colors.copy(),
            domains_mask=self.domains_mask.copy(),
            k=int(self.k),
            open_colors=int(self.open_colors),
            step=int(self.step),
        )


@dataclasses.dataclass
class ConstructiveMetrics:
    colored_count: int
    uncolored_count: int
    zero_domain_vertices: int
    singleton_domain_vertices: int
    mean_domain_size: float
    max_domain_size: int
    max_saturation: int
    frontier_size: int
    class_sizes: np.ndarray
    legal_action_count: int
    entropy: float
    dead: bool
    solved: bool
    belief: float


@dataclasses.dataclass(frozen=True)
class ConstructiveAction:
    vertex: int
    color: int
    opens_new_color: bool
    est_priority: float
    token: Tuple[Any, ...]


class BeliefModel(Protocol):
    def predict(self, graph: repair.GCGraph, state: ConstructiveState, metrics: ConstructiveMetrics) -> float:
        ...


class HeuristicBeliefModel:
    """Cheap solvability/extensibility estimator for DeltaBelief-style rewards."""

    def predict(self, graph: repair.GCGraph, state: ConstructiveState, metrics: ConstructiveMetrics) -> float:
        if metrics.dead:
            return 0.0
        if metrics.solved:
            return 1.0
        x = 0.0
        x -= 4.0 * (metrics.zero_domain_vertices / max(graph.n, 1))
        x -= 0.8 * (metrics.singleton_domain_vertices / max(graph.n, 1))
        x += 1.4 * (metrics.colored_count / max(graph.n, 1))
        x += 1.2 * (metrics.mean_domain_size / max(state.k, 1))
        x -= 0.8 * (state.open_colors / max(state.k, 1))
        x += 0.2 * (metrics.entropy / max(math.log(max(state.k, 2)), 1.0))
        return float(1.0 / (1.0 + math.exp(-x)))


def canonicalize_partial_colors(colors: np.ndarray) -> np.ndarray:
    colors = np.asarray(colors, dtype=np.int16)
    out = np.full_like(colors, -1)
    remap: Dict[int, int] = {}
    nxt = 0
    for i, c in enumerate(colors.tolist()):
        c = int(c)
        if c < 0:
            continue
        if c not in remap:
            remap[c] = nxt
            nxt += 1
        out[i] = remap[c]
    return out


def _reorder_domain_columns(domains_mask: np.ndarray, mapping: Dict[int, int], k: int) -> np.ndarray:
    out = np.zeros_like(domains_mask, dtype=np.bool_)
    for old_c in range(k):
        if old_c in mapping:
            out[:, mapping[old_c]] = domains_mask[:, old_c]
    return out


def canonicalize_constructive_state(state: ConstructiveState) -> ConstructiveState:
    colors = state.colors.copy()
    domains = state.domains_mask.copy()
    remap: Dict[int, int] = {}
    nxt = 0
    for c in colors.tolist():
        c = int(c)
        if c < 0:
            continue
        if c not in remap:
            remap[c] = nxt
            nxt += 1
    colors = canonicalize_partial_colors(colors)
    if remap:
        domains = _reorder_domain_columns(domains, remap, state.k)
    return ConstructiveState(colors=colors, domains_mask=domains, k=int(state.k), open_colors=int(nxt), step=int(state.step))


def initial_constructive_state(graph: repair.GCGraph, k: int) -> ConstructiveState:
    return ConstructiveState(
        colors=np.full(graph.n, -1, dtype=np.int16),
        domains_mask=np.ones((graph.n, k), dtype=np.bool_),
        k=int(k),
        open_colors=0,
        step=0,
    )


def _count_conflicts(graph: repair.GCGraph, colors: np.ndarray) -> int:
    bad = 0
    for u, v in graph.edges:
        if int(colors[u]) >= 0 and int(colors[u]) == int(colors[v]):
            bad += 1
    return int(bad)


def compute_domains(graph: repair.GCGraph, state: ConstructiveState) -> np.ndarray:
    n, k = graph.n, state.k
    domains = np.zeros((n, k), dtype=np.bool_)
    colors = state.colors
    for v in range(n):
        if int(colors[v]) >= 0:
            continue
        forbidden = {int(colors[u]) for u in graph.adj_list[v] if int(colors[u]) >= 0}
        for c in range(state.open_colors):
            if c not in forbidden:
                domains[v, c] = True
        if state.open_colors < k and state.open_colors not in forbidden:
            domains[v, state.open_colors] = True
    return domains


def compute_constructive_metrics(
    graph: repair.GCGraph,
    state: ConstructiveState,
    belief_model: Optional[BeliefModel] = None,
) -> ConstructiveMetrics:
    colors = state.colors
    domains = compute_domains(graph, state)
    state.domains_mask = domains
    uncolored = colors < 0
    colored_count = int((~uncolored).sum())
    uncolored_count = int(uncolored.sum())
    domain_sizes = domains.sum(axis=1).astype(np.int32)
    domain_sizes_colored = domain_sizes.copy()
    domain_sizes_colored[~uncolored] = 0
    zero_domain_vertices = int(((domain_sizes_colored == 0) & uncolored).sum())
    singleton_domain_vertices = int(((domain_sizes_colored == 1) & uncolored).sum())
    mean_domain_size = float(domain_sizes_colored[uncolored].mean()) if uncolored_count > 0 else 0.0
    max_domain_size = int(domain_sizes_colored.max()) if uncolored_count > 0 else 0
    saturation = np.zeros(graph.n, dtype=np.int32)
    for v in np.where(uncolored)[0].tolist():
        neigh_cols = {int(colors[u]) for u in graph.adj_list[v] if int(colors[u]) >= 0}
        saturation[v] = len(neigh_cols)
    max_saturation = int(saturation.max()) if uncolored_count > 0 else 0
    frontier_size = int((saturation > 0).sum())
    class_sizes = np.bincount(np.clip(colors.astype(np.int64), 0, max(state.k - 1, 0)), minlength=state.k).astype(np.int32)
    legal_action_count = int(domains[uncolored].sum()) if uncolored_count > 0 else 0
    probs = class_sizes.astype(np.float64) / max(int(class_sizes.sum()), 1)
    entropy = float(-(probs[probs > 0] * np.log(probs[probs > 0] + EPS)).sum())
    dead = bool(zero_domain_vertices > 0)
    solved = bool(uncolored_count == 0 and _count_conflicts(graph, colors) == 0)
    tmp = ConstructiveMetrics(
        colored_count=colored_count,
        uncolored_count=uncolored_count,
        zero_domain_vertices=zero_domain_vertices,
        singleton_domain_vertices=singleton_domain_vertices,
        mean_domain_size=mean_domain_size,
        max_domain_size=max_domain_size,
        max_saturation=max_saturation,
        frontier_size=frontier_size,
        class_sizes=class_sizes,
        legal_action_count=legal_action_count,
        entropy=entropy,
        dead=dead,
        solved=solved,
        belief=0.0,
    )
    if belief_model is not None:
        tmp.belief = float(belief_model.predict(graph, state, tmp))
    return tmp


def generate_constructive_actions(
    graph: repair.GCGraph,
    state: ConstructiveState,
    metrics: ConstructiveMetrics,
    vertex_budget: int = DEFAULT_VERTEX_BUDGET,
    action_budget: int = DEFAULT_ACTION_BUDGET,
) -> List[ConstructiveAction]:
    colors = state.colors
    domains = state.domains_mask
    uncolored_idx = np.where(colors < 0)[0]
    if uncolored_idx.size == 0:
        return []
    sat = np.zeros(graph.n, dtype=np.int32)
    dom_sizes = domains.sum(axis=1).astype(np.int32)
    for v in uncolored_idx.tolist():
        neigh_cols = {int(colors[u]) for u in graph.adj_list[v] if int(colors[u]) >= 0}
        sat[v] = len(neigh_cols)
    ranked = sorted(
        uncolored_idx.tolist(),
        key=lambda v: (-sat[v], -int(graph.degrees[v]), int(dom_sizes[v]), int(v)),
    )[: min(vertex_budget, int(uncolored_idx.size))]
    actions: List[ConstructiveAction] = []
    for v in ranked:
        available = np.where(domains[v])[0].tolist()
        for c in available:
            opens_new = int(c) == int(state.open_colors)
            est_priority = 0.0
            est_priority += 2.0 * float(sat[v])
            est_priority += 1.0 * float(graph.degrees[v]) / max(graph.max_degree, 1)
            est_priority -= 0.75 * float(opens_new)
            est_priority -= 0.10 * float(dom_sizes[v])
            actions.append(
                ConstructiveAction(
                    vertex=int(v),
                    color=int(c),
                    opens_new_color=bool(opens_new),
                    est_priority=float(est_priority),
                    token=(int(v), int(c), bool(opens_new)),
                )
            )
    actions.sort(key=lambda a: a.est_priority, reverse=True)
    dedup: List[ConstructiveAction] = []
    seen = set()
    for a in actions:
        if a.token in seen:
            continue
        seen.add(a.token)
        dedup.append(a)
    return dedup[:action_budget]


def apply_constructive_action(graph: repair.GCGraph, state: ConstructiveState, action: ConstructiveAction) -> ConstructiveState:
    nxt = state.copy()
    if nxt.colors[action.vertex] >= 0:
        return nxt
    nxt.colors[action.vertex] = int(action.color)
    nxt.step += 1
    nxt = canonicalize_constructive_state(nxt)
    return nxt


def constructive_reward_dense(
    graph: repair.GCGraph,
    before: ConstructiveMetrics,
    after: ConstructiveMetrics,
    opens_new_color: bool,
) -> float:
    r = 0.0
    r += 1.00 * (after.colored_count - before.colored_count) / max(graph.n, 1)
    r += 0.35 * (before.zero_domain_vertices - after.zero_domain_vertices) / max(graph.n, 1)
    r += 0.20 * (before.singleton_domain_vertices - after.singleton_domain_vertices) / max(graph.n, 1)
    r += 0.10 * (after.mean_domain_size - before.mean_domain_size) / max(after.class_sizes.sum(), 1)
    r += 0.05 * (after.entropy - before.entropy) / max(math.log(max(after.class_sizes.sum(), 2)), 1.0)
    r -= 0.08 * float(opens_new_color)
    if after.dead:
        r -= 1.0
    if after.solved:
        r += 1.0
    return float(r)


def constructive_reward_delta_belief(before_belief: float, after_belief: float) -> float:
    return float(after_belief - before_belief)


def constructive_reward_hybrid(
    graph: repair.GCGraph,
    before: ConstructiveMetrics,
    after: ConstructiveMetrics,
    opens_new_color: bool,
) -> float:
    dense = constructive_reward_dense(graph, before, after, opens_new_color)
    delta_b = constructive_reward_delta_belief(before.belief, after.belief)
    return float(0.5 * dense + 0.5 * delta_b)


def transition_constructive(
    graph: repair.GCGraph,
    state: ConstructiveState,
    metrics: ConstructiveMetrics,
    action: ConstructiveAction,
    belief_model: Optional[BeliefModel],
    reward_mode: str,
) -> Tuple[ConstructiveState, ConstructiveMetrics, float]:
    nxt = apply_constructive_action(graph, state, action)
    nxt_metrics = compute_constructive_metrics(graph, nxt, belief_model=belief_model)
    if reward_mode == "dense":
        reward = constructive_reward_dense(graph, metrics, nxt_metrics, action.opens_new_color)
    elif reward_mode == "delta_belief":
        reward = constructive_reward_delta_belief(metrics.belief, nxt_metrics.belief)
    elif reward_mode == "hybrid":
        reward = constructive_reward_hybrid(graph, metrics, nxt_metrics, action.opens_new_color)
    else:
        raise ValueError(f"unknown reward_mode={reward_mode}")
    return nxt, nxt_metrics, reward


def state_key_constructive(state: ConstructiveState) -> bytes:
    state = canonicalize_constructive_state(state)
    header = np.asarray([state.k, state.open_colors], dtype=np.int16).tobytes()
    return header + state.colors.tobytes() + np.packbits(state.domains_mask.astype(np.uint8), axis=None).tobytes()


def build_teacher_traces_for_graph(
    graph: repair.GCGraph,
    oracle_solution: np.ndarray,
    k: int,
    reward_mode: str,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    belief = HeuristicBeliefModel()
    state = initial_constructive_state(graph, k)
    metrics = compute_constructive_metrics(graph, state, belief_model=belief)
    traces: List[Dict[str, Any]] = []
    remaining = set(range(graph.n))
    while remaining:
        actions = generate_constructive_actions(graph, state, metrics)
        if not actions:
            break
        best_idx = None
        best_score = -1e18
        for i, a in enumerate(actions):
            if int(oracle_solution[a.vertex]) != int(a.color):
                continue
            score = float(a.est_priority)
            if not a.opens_new_color:
                score += 0.5
            if score > best_score:
                best_score = score
                best_idx = i
        if best_idx is None:
            best_idx = 0
        chosen = actions[best_idx]
        nxt, nxt_metrics, reward = transition_constructive(graph, state, metrics, chosen, belief_model=belief, reward_mode=reward_mode)
        traces.append(
            {
                "graph_name": graph.name,
                "step": int(state.step),
                "state_colors": state.colors.tolist(),
                "domains_mask": state.domains_mask.astype(np.int8).tolist(),
                "target_action": {"vertex": chosen.vertex, "color": chosen.color},
                "chosen_family": "constructive_assign",
                "reward": float(reward),
                "belief_before": float(metrics.belief),
                "belief_after": float(nxt_metrics.belief),
                "solved": bool(nxt_metrics.solved),
            }
        )
        state, metrics = nxt, nxt_metrics
        remaining = set(np.where(state.colors < 0)[0].tolist())
        if metrics.dead or metrics.solved:
            break
    return traces


def load_graph_records(path: Path) -> List[repair.GraphRecord]:
    rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    out: List[repair.GraphRecord] = []
    for row in rows:
        edges = np.asarray(row.get("edges", []), dtype=np.int64)
        if edges.size == 0:
            edges = edges.reshape(0, 2)
        sol = row.get("solution")
        out.append(
            repair.GraphRecord(
                name=str(row.get("name", "")),
                n=int(row["n"]),
                edges=edges,
                solution=np.asarray(sol, dtype=np.int16) if sol is not None else None,
                metadata=row.get("metadata", {}) or {},
            )
        )
    return out


def beam_constructive_search(
    graph: repair.GCGraph,
    k: int,
    beam_width: int,
    max_steps: Optional[int],
    reward_mode: str,
    belief_model: Optional[BeliefModel],
) -> Dict[str, Any]:
    if max_steps is None:
        max_steps = graph.n
    init = initial_constructive_state(graph, k)
    init_metrics = compute_constructive_metrics(graph, init, belief_model=belief_model)
    BeamEntry = Tuple[float, ConstructiveState, ConstructiveMetrics, List[Dict[str, Any]]]
    beam: List[BeamEntry] = [(0.0, init, init_metrics, [])]
    best = beam[0]
    for _ in range(int(max_steps)):
        expanded: List[BeamEntry] = []
        for score, state, metrics, hist in beam:
            if metrics.dead or metrics.solved:
                expanded.append((score, state, metrics, hist))
                continue
            actions = generate_constructive_actions(graph, state, metrics)
            for a in actions[: min(beam_width, len(actions))]:
                nxt, nxt_metrics, reward = transition_constructive(graph, state, metrics, a, belief_model=belief_model, reward_mode=reward_mode)
                nh = hist + [{"vertex": a.vertex, "color": a.color, "reward": reward, "belief": nxt_metrics.belief}]
                expanded.append((score + reward, nxt, nxt_metrics, nh))
        expanded.sort(key=lambda x: (x[2].solved, -x[2].zero_domain_vertices, x[0]), reverse=True)
        beam = expanded[: max(1, int(beam_width))]
        best = beam[0]
        if best[2].solved:
            break
    score, state, metrics, hist = best
    return {
        "score": float(score),
        "colors": state.colors.tolist(),
        "open_colors": int(state.open_colors),
        "solved": bool(metrics.solved),
        "dead": bool(metrics.dead),
        "colored_count": int(metrics.colored_count),
        "uncolored_count": int(metrics.uncolored_count),
        "zero_domain_vertices": int(metrics.zero_domain_vertices),
        "belief": float(metrics.belief),
        "history": hist,
    }


def cmd_build_traces(args: argparse.Namespace) -> None:
    rng = random.Random(int(args.seed))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = load_graph_records(Path(args.input))
    if not records:
        raise RuntimeError("No records found")
    all_traces: List[Dict[str, Any]] = []
    for rec in records:
        if rec.solution is None:
            continue
        graph = rec.to_runtime()
        traces = build_teacher_traces_for_graph(graph, np.asarray(rec.solution, dtype=np.int16), int(args.k) if args.k > 0 else int(graph.solution.max()) + 1, str(args.reward_mode), rng)
        all_traces.extend(traces)
    out_path = out_dir / f"{args.prefix}.jsonl"
    with out_path.open("w") as f:
        for row in all_traces:
            f.write(json.dumps(row) + "\n")
    print(json.dumps({"wrote": str(out_path), "n_traces": len(all_traces)}), flush=True)


def cmd_rollout(args: argparse.Namespace) -> None:
    recs = load_graph_records(Path(args.input))
    if len(recs) != 1:
        raise RuntimeError("rollout expects exactly one graph record")
    graph = recs[0].to_runtime()
    belief = HeuristicBeliefModel() if str(args.reward_mode) in {"delta_belief", "hybrid"} else None
    result = beam_constructive_search(
        graph=graph,
        k=int(args.k),
        beam_width=int(args.beam_width),
        max_steps=int(args.max_steps) if int(args.max_steps) > 0 else None,
        reward_mode=str(args.reward_mode),
        belief_model=belief,
    )
    print(json.dumps(result, indent=2), flush=True)


def make_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="ConstructiveMDP prototype for k-coloring")
    sp = ap.add_subparsers(dest="cmd", required=True)

    p_bt = sp.add_parser("build-traces", help="Build constructive teacher traces from solved examples")
    p_bt.add_argument("--input", type=str, required=True)
    p_bt.add_argument("--out-dir", type=str, required=True)
    p_bt.add_argument("--prefix", type=str, default="constructive_traces")
    p_bt.add_argument("--k", type=int, default=0, help="If 0, infer from solution max+1")
    p_bt.add_argument("--reward-mode", type=str, default="hybrid", choices=["dense", "delta_belief", "hybrid"])
    p_bt.add_argument("--seed", type=int, default=20260422)
    p_bt.set_defaults(func=cmd_build_traces)

    p_ro = sp.add_parser("rollout", help="Run a constructive beam-search rollout on one graph")
    p_ro.add_argument("--input", type=str, required=True)
    p_ro.add_argument("--k", type=int, required=True)
    p_ro.add_argument("--beam-width", type=int, default=16)
    p_ro.add_argument("--max-steps", type=int, default=0, help="0 = n")
    p_ro.add_argument("--reward-mode", type=str, default="hybrid", choices=["dense", "delta_belief", "hybrid"])
    p_ro.set_defaults(func=cmd_rollout)

    return ap


def main() -> None:
    parser = make_argparser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
