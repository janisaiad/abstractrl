#!/usr/bin/env python3
from __future__ import annotations

"""
Constructive MDP prototype for k-coloring.

Purpose
-------
This file provides a standalone constructive environment that complements the
existing repair-based pipeline. It supports:
  * partial-coloring states with explicit domains
  * canonicalization under color-label symmetry
  * DSATUR-style candidate generation
  * dense constructive rewards
  * optional DeltaBelief-style intrinsic reward via a separate belief model
  * teacher-trace generation from solved examples
  * heuristic beam-search constructive rollouts

This is an experimental branch meant for future integration with the TRM/MCTS
stack. It does not replace the current repair pipeline.
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
GLOBAL_FEAT_DIM = 24
VERTEX_FEAT_DIM = 16
ACTION_FEAT_DIM = 28
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
            self.colors.copy(),
            self.domains_mask.copy(),
            int(self.k),
            int(self.open_colors),
            int(self.step),
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
    """Cheap extensibility belief estimator for delta-belief rewards."""

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
        legal_cols = np.where(domains[v])[0].tolist()
        legal_cols.sort(key=lambda c: (int(c) >= state.open_colors, int(c)))
        for c in legal_cols:
            opens_new = bool(int(c) == state.open_colors and state.open_colors < state.k)
            priority = 2.0 * float(sat[v]) + 0.5 * float(graph.degrees[v]) - (1.0 if opens_new else 0.0)
            actions.append(ConstructiveAction(int(v), int(c), opens_new, float(priority), (int(v), int(c), int(opens_new))))
    seen = set()
    dedup: List[ConstructiveAction] = []
    for a in sorted(actions, key=lambda x: x.est_priority, reverse=True):
        if a.token in seen:
            continue
        seen.add(a.token)
        dedup.append(a)
    return dedup[:action_budget]


def transition_constructive(graph: repair.GCGraph, state: ConstructiveState, action: ConstructiveAction) -> ConstructiveState:
    nxt = state.copy()
    if int(nxt.colors[action.vertex]) >= 0:
        return nxt
    nxt.colors[int(action.vertex)] = int(action.color)
    nxt.step += 1
    if action.opens_new_color:
        nxt.open_colors = max(int(nxt.open_colors), int(action.color) + 1)
    nxt = canonicalize_constructive_state(nxt)
    nxt.domains_mask = compute_domains(graph, nxt)
    return nxt


def dense_constructive_reward(before: ConstructiveMetrics, after: ConstructiveMetrics, opens_new_color: bool) -> float:
    n_tot = max(before.colored_count + before.uncolored_count, 1)
    r = 0.0
    r += 1.00 * (after.colored_count - before.colored_count) / n_tot
    r += 0.60 * (before.zero_domain_vertices - after.zero_domain_vertices) / n_tot
    r += 0.25 * (before.singleton_domain_vertices - after.singleton_domain_vertices) / n_tot
    r += 0.15 * (after.mean_domain_size - before.mean_domain_size) / max(before.max_domain_size, 1)
    r -= 0.10 * float(opens_new_color)
    if after.dead:
        r -= 1.0
    if after.solved:
        r += 1.5
    return float(r)


def delta_belief_reward(before: ConstructiveMetrics, after: ConstructiveMetrics, opens_new_color: bool, alpha_dense: float = 0.25) -> float:
    return float(after.belief - before.belief + alpha_dense * dense_constructive_reward(before, after, opens_new_color))


def constructive_reward(before: ConstructiveMetrics, after: ConstructiveMetrics, opens_new_color: bool, reward_mode: str) -> float:
    if reward_mode == "dense":
        return dense_constructive_reward(before, after, opens_new_color)
    if reward_mode == "delta_belief":
        return delta_belief_reward(before, after, opens_new_color, alpha_dense=0.0)
    if reward_mode == "hybrid":
        return delta_belief_reward(before, after, opens_new_color, alpha_dense=0.25)
    raise ValueError(f"unknown reward_mode: {reward_mode}")


def build_constructive_observation(
    graph: repair.GCGraph,
    state: ConstructiveState,
    metrics: ConstructiveMetrics,
    actions: Sequence[ConstructiveAction],
    vertex_budget: int = DEFAULT_VERTEX_BUDGET,
    action_budget: int = DEFAULT_ACTION_BUDGET,
) -> Dict[str, np.ndarray]:
    colors = state.colors
    domains = state.domains_mask
    uncolored_idx = np.where(colors < 0)[0]
    vtoks = np.zeros((vertex_budget, VERTEX_FEAT_DIM), dtype=np.float32)
    vmask = np.zeros(vertex_budget, dtype=np.bool_)
    if uncolored_idx.size > 0:
        sat = np.zeros(graph.n, dtype=np.int32)
        dom_sizes = domains.sum(axis=1).astype(np.int32)
        for v in uncolored_idx.tolist():
            neigh_cols = {int(colors[u]) for u in graph.adj_list[v] if int(colors[u]) >= 0}
            sat[v] = len(neigh_cols)
        ranked = sorted(uncolored_idx.tolist(), key=lambda v: (-sat[v], -int(graph.degrees[v]), int(dom_sizes[v]), int(v)))[: min(vertex_budget, int(uncolored_idx.size))]
        for i, v in enumerate(ranked):
            vmask[i] = True
            vtoks[i, 0] = float(graph.degrees[v]) / max(graph.max_degree, 1)
            vtoks[i, 1] = float(sat[v]) / max(state.open_colors, 1)
            vtoks[i, 2] = float(dom_sizes[v]) / max(state.k, 1)
            vtoks[i, 3] = float(dom_sizes[v] == 0)
            vtoks[i, 4] = float(dom_sizes[v] == 1)
            vtoks[i, 5] = float(v) / max(graph.n - 1, 1)
            vtoks[i, 6] = float(metrics.colored_count) / max(graph.n, 1)
            vtoks[i, 7] = float(metrics.uncolored_count) / max(graph.n, 1)
            vtoks[i, 8] = float(metrics.zero_domain_vertices) / max(graph.n, 1)
            vtoks[i, 9] = float(metrics.singleton_domain_vertices) / max(graph.n, 1)
            vtoks[i,10] = float(state.open_colors) / max(state.k, 1)
            vtoks[i,11] = float(graph.clique_lb) / max(state.k, 1)
            vtoks[i,12] = float(graph.degeneracy_hint) / max(state.k, 1)
            vtoks[i,13] = float(metrics.entropy) / max(math.log(max(state.k, 2)), 1.0)
            vtoks[i,14] = float(metrics.belief)
            vtoks[i,15] = float(metrics.legal_action_count) / max(graph.n * state.k, 1)
    atoks = np.zeros((action_budget, ACTION_FEAT_DIM), dtype=np.float32)
    amask = np.zeros(action_budget, dtype=np.bool_)
    for i, a in enumerate(actions[:action_budget]):
        amask[i] = True
        atoks[i, 0] = float(a.vertex) / max(graph.n - 1, 1)
        atoks[i, 1] = float(a.color) / max(state.k - 1, 1)
        atoks[i, 2] = float(a.opens_new_color)
        atoks[i, 3] = float(a.est_priority) / max(graph.max_degree + state.k, 1)
        atoks[i, 4] = float(metrics.colored_count) / max(graph.n, 1)
        atoks[i, 5] = float(metrics.uncolored_count) / max(graph.n, 1)
        atoks[i, 6] = float(metrics.zero_domain_vertices) / max(graph.n, 1)
        atoks[i, 7] = float(metrics.singleton_domain_vertices) / max(graph.n, 1)
        atoks[i, 8] = float(metrics.mean_domain_size) / max(state.k, 1)
        atoks[i, 9] = float(metrics.max_saturation) / max(state.open_colors, 1)
        atoks[i,10] = float(state.open_colors) / max(state.k, 1)
        atoks[i,11] = float(metrics.belief)
    gfeat = np.zeros(GLOBAL_FEAT_DIM, dtype=np.float32)
    gfeat[0] = float(graph.n) / 2048.0
    gfeat[1] = float(graph.density)
    gfeat[2] = float(state.k) / max(graph.n, 1)
    gfeat[3] = float(state.open_colors) / max(state.k, 1)
    gfeat[4] = float(graph.clique_lb) / max(state.k, 1)
    gfeat[5] = float(graph.degeneracy_hint) / max(state.k, 1)
    gfeat[6] = float(metrics.colored_count) / max(graph.n, 1)
    gfeat[7] = float(metrics.uncolored_count) / max(graph.n, 1)
    gfeat[8] = float(metrics.zero_domain_vertices) / max(graph.n, 1)
    gfeat[9] = float(metrics.singleton_domain_vertices) / max(graph.n, 1)
    gfeat[10] = float(metrics.mean_domain_size) / max(state.k, 1)
    gfeat[11] = float(metrics.max_domain_size) / max(state.k, 1)
    gfeat[12] = float(metrics.max_saturation) / max(state.open_colors, 1)
    gfeat[13] = float(metrics.frontier_size) / max(graph.n, 1)
    gfeat[14] = float(metrics.entropy) / max(math.log(max(state.k, 2)), 1.0)
    gfeat[15] = float(metrics.legal_action_count) / max(graph.n * state.k, 1)
    gfeat[16] = float(metrics.dead)
    gfeat[17] = float(metrics.solved)
    gfeat[18] = float(metrics.belief)
    gfeat[19] = float(min(len(actions), action_budget)) / max(state.k * DEFAULT_VERTEX_BUDGET, 1)
    return {
        "global_feats": gfeat,
        "vertex_tokens": vtoks,
        "vertex_mask": vmask,
        "action_tokens": atoks,
        "action_mask": amask,
    }


def teacher_constructive_traces_for_record(
    graph: repair.GCGraph,
    reward_mode: str,
    belief_model: Optional[BeliefModel],
    max_steps: int,
    seed: int,
) -> List[Dict[str, Any]]:
    if graph.solution is None:
        raise ValueError("constructive teacher traces require solved graphs")
    solution = canonicalize_partial_colors(graph.solution.astype(np.int16))
    k = int(solution.max()) + 1 if solution.size else 0
    state = initial_constructive_state(graph, k)
    samples: List[Dict[str, Any]] = []
    episode_id = f"{graph.name}|constructive|seed={seed}"
    for step in range(int(max_steps)):
        metrics = compute_constructive_metrics(graph, state, belief_model=belief_model)
        if metrics.dead or metrics.solved:
            break
        actions = generate_constructive_actions(graph, state, metrics)
        if not actions:
            break
        target_idx = None
        for i, a in enumerate(actions):
            if int(solution[a.vertex]) == int(a.color):
                target_idx = i
                break
        if target_idx is None:
            target_idx = 0
        obs = build_constructive_observation(graph, state, metrics, actions)
        next_state = transition_constructive(graph, state, actions[target_idx])
        next_metrics = compute_constructive_metrics(graph, next_state, belief_model=belief_model)
        reward = constructive_reward(metrics, next_metrics, actions[target_idx].opens_new_color, reward_mode=reward_mode)
        samples.append({
            "global_feats": obs["global_feats"],
            "vertex_tokens": obs["vertex_tokens"],
            "vertex_mask": obs["vertex_mask"],
            "class_tokens": np.zeros((1, 1), dtype=np.float32),
            "class_mask": np.ones((1,), dtype=np.bool_),
            "action_tokens": obs["action_tokens"],
            "action_mask": obs["action_mask"],
            "target_action": int(target_idx),
            "target_policy": None,
            "value_target": float(next_metrics.belief if reward_mode != "dense" else (1.0 if next_metrics.solved else 0.0)),
            "reward": float(reward),
            "chosen_family": "constructive_assign",
            "episode_id": episode_id,
            "step": int(step),
            "reward_mode": reward_mode,
        })
        state = next_state
    return samples


def beam_constructive_search(
    graph: repair.GCGraph,
    k: int,
    beam_width: int,
    max_steps: int,
    reward_mode: str,
    belief_model: Optional[BeliefModel],
) -> Dict[str, Any]:
    t0 = time.time()
    init = initial_constructive_state(graph, k)
    init_metrics = compute_constructive_metrics(graph, init, belief_model=belief_model)
    beam: List[Tuple[float, ConstructiveState, ConstructiveMetrics]] = [(0.0, init, init_metrics)]
    best_state = init.copy()
    best_metrics = init_metrics
    for _ in range(int(max_steps)):
        new_items: List[Tuple[float, ConstructiveState, ConstructiveMetrics]] = []
        for score, state, metrics in beam:
            if metrics.solved:
                best_state, best_metrics = state, metrics
                beam = [(score, state, metrics)]
                break
            if metrics.dead:
                continue
            actions = generate_constructive_actions(graph, state, metrics)
            for a in actions[: min(beam_width, len(actions))]:
                nxt = transition_constructive(graph, state, a)
                nxt_metrics = compute_constructive_metrics(graph, nxt, belief_model=belief_model)
                r = constructive_reward(metrics, nxt_metrics, a.opens_new_color, reward_mode)
                total = score + r
                new_items.append((total, nxt, nxt_metrics))
                if nxt_metrics.solved:
                    best_state, best_metrics = nxt, nxt_metrics
        if best_metrics.solved or not new_items:
            break
        new_items.sort(key=lambda x: x[0], reverse=True)
        beam = new_items[: int(beam_width)]
        _, cand_state, cand_metrics = beam[0]
        if (cand_metrics.colored_count > best_metrics.colored_count) or (
            cand_metrics.colored_count == best_metrics.colored_count and cand_metrics.zero_domain_vertices < best_metrics.zero_domain_vertices
        ):
            best_state, best_metrics = cand_state, cand_metrics
    elapsed = time.time() - t0
    return {
        "k": int(k),
        "colors": best_state.colors.tolist(),
        "solved": bool(best_metrics.solved),
        "dead": bool(best_metrics.dead),
        "colored_count": int(best_metrics.colored_count),
        "uncolored_count": int(best_metrics.uncolored_count),
        "zero_domain_vertices": int(best_metrics.zero_domain_vertices),
        "belief": float(best_metrics.belief),
        "time_sec": float(elapsed),
    }


def load_records(path: str) -> List[repair.GraphRecord]:
    return repair.load_records(path)


def command_build_traces(args: argparse.Namespace) -> None:
    belief_model: Optional[BeliefModel] = HeuristicBeliefModel() if args.reward_mode in ("delta_belief", "hybrid") else None
    records = load_records(args.input)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    total = 0
    shard: List[Dict[str, Any]] = []
    shard_id = 0
    for ridx, rec in enumerate(records):
        graph = rec.to_runtime()
        if graph.solution is None:
            if args.skip_unsolved:
                continue
            raise ValueError(f"{graph.name} has no solution")
        samples = teacher_constructive_traces_for_record(graph, args.reward_mode, belief_model, int(args.max_steps), int(args.seed) + ridx)
        for s in samples:
            shard.append(s)
            total += 1
            if len(shard) >= int(args.samples_per_shard):
                out = Path(args.out_dir) / f"{args.prefix}-{shard_id:05d}.pt"
                torch.save(shard, out)
                shard_id += 1
                shard = []
        if (ridx + 1) % max(int(args.log_every_records), 1) == 0:
            print(f"[{ridx + 1}/{len(records)}] wrote {total} constructive samples", flush=True)
    if shard:
        out = Path(args.out_dir) / f"{args.prefix}-{shard_id:05d}.pt"
        torch.save(shard, out)
    print(f"done: wrote {total} constructive samples to {args.out_dir}", flush=True)


def command_rollout(args: argparse.Namespace) -> None:
    belief_model: Optional[BeliefModel] = HeuristicBeliefModel() if args.reward_mode in ("delta_belief", "hybrid") else None
    graph = load_records(args.input)[0].to_runtime()
    result = beam_constructive_search(graph, int(args.k), int(args.beam_width), int(args.max_steps), args.reward_mode, belief_model)
    print(json.dumps(result, indent=2), flush=True)


def command_debug(args: argparse.Namespace) -> None:
    belief_model: Optional[BeliefModel] = HeuristicBeliefModel() if args.reward_mode in ("delta_belief", "hybrid") else None
    graph = load_records(args.input)[0].to_runtime()
    state = initial_constructive_state(graph, int(args.k))
    metrics = compute_constructive_metrics(graph, state, belief_model=belief_model)
    actions = generate_constructive_actions(graph, state, metrics)
    obs = build_constructive_observation(graph, state, metrics, actions)
    payload = {
        "graph": {"name": graph.name, "n": graph.n, "m": graph.m},
        "metrics": {
            **dataclasses.asdict(metrics),
            "class_sizes": metrics.class_sizes.tolist(),
        },
        "num_actions": len(actions),
        "first_actions": [dataclasses.asdict(a) for a in actions[:10]],
        "obs_shapes": {
            "global_feats": list(obs["global_feats"].shape),
            "vertex_tokens": list(obs["vertex_tokens"].shape),
            "action_tokens": list(obs["action_tokens"].shape),
        },
    }
    print(json.dumps(payload, indent=2), flush=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Constructive MDP prototype for k-coloring with optional DeltaBelief-style reward")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build-traces", help="build constructive traces from solved examples")
    b.add_argument("--input", required=True)
    b.add_argument("--out-dir", required=True)
    b.add_argument("--prefix", default="gcp-constructive")
    b.add_argument("--samples-per-shard", type=int, default=5000)
    b.add_argument("--max-steps", type=int, default=128)
    b.add_argument("--reward-mode", choices=["dense", "delta_belief", "hybrid"], default="dense")
    b.add_argument("--skip-unsolved", action="store_true")
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--log-every-records", type=int, default=25)
    b.set_defaults(func=command_build_traces)

    r = sub.add_parser("rollout", help="run constructive beam-search baseline")
    r.add_argument("--input", required=True)
    r.add_argument("--k", type=int, required=True)
    r.add_argument("--beam-width", type=int, default=16)
    r.add_argument("--max-steps", type=int, default=256)
    r.add_argument("--reward-mode", choices=["dense", "delta_belief", "hybrid"], default="dense")
    r.set_defaults(func=command_rollout)

    d = sub.add_parser("debug", help="inspect initial constructive state / actions")
    d.add_argument("--input", required=True)
    d.add_argument("--k", type=int, required=True)
    d.add_argument("--reward-mode", choices=["dense", "delta_belief", "hybrid"], default="dense")
    d.set_defaults(func=command_debug)
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
