#!/usr/bin/env python3
"""
Graph Coloring trace training + repair search in one file (v2).

Designed to live inside janisaiad/abstractrl, e.g.:
  src/AbstractBeam/crossbeam/experiment/gcp_trace_abstractbeam.py

What this file does:
  * parses solved GCP datasets (JSON/JSONL/PKL/PT/DIMACS)
  * builds teacher traces from solved examples and solver-generated traces from self-solving episodes
  * trains a TRM-style policy/value prior with torchrun/DDP
  * mines lightweight parameterized macros from traces
  * solves new instances with a deterministic single-player MCTS over GCP primitives

This is intentionally self-contained: it does not modify CrossBeam internals.
It is meant to work in the abstractrl environment described in the repo README:
PyTorch + optional stitch-core + torchrun/DDP.
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import dataclasses
import enum
import glob
import hashlib
import json
import math
import os
import pickle
import random
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, IterableDataset

try:
    from scipy.optimize import linear_sum_assignment  # type: ignore
except Exception:  # pragma: no cover
    linear_sum_assignment = None

try:
    import stitch_core  # type: ignore
except Exception:  # pragma: no cover
    stitch_core = None


EPS = 1e-9
GLOBAL_FEAT_DIM = 24
VERTEX_FEAT_DIM = 16
CLASS_FEAT_DIM = 12
ACTION_FEAT_DIM = 28
DEFAULT_VERTEX_BUDGET = 32
DEFAULT_CLASS_BUDGET = 16
DEFAULT_ACTION_BUDGET = 128


class PrimitiveFamily(str, enum.Enum):
    VERTEX_RECOLOR = "vertex_recolor"
    KEMPE_SWAP = "kempe_swap"
    TABU_SHORT = "tabu_short"
    TABU_LONG = "tabu_long"
    FOCUS_CORE = "focus_core"
    PERTURB_SOFT = "perturb_soft"
    EXACT_PATCH = "exact_patch"
    MACRO = "macro"


FAMILY_TO_ID: Dict[str, int] = {name.value: i for i, name in enumerate(PrimitiveFamily)}


@dataclasses.dataclass(frozen=True)
class MacroProgram:
    name: str
    families: Tuple[str, ...]
    support: int
    score: float
    min_conflict_ratio: float = 0.0
    max_conflict_ratio: float = 1.0
    min_core_ratio: float = 0.0
    max_core_ratio: float = 1.0


@dataclasses.dataclass
class GraphRecord:
    name: str
    n: int
    edges: np.ndarray  # shape [m, 2], int64
    solution: Optional[np.ndarray] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_runtime(self) -> "GCGraph":
        return GCGraph(self.n, self.edges, self.name, self.solution, self.metadata or {})


class GCGraph:
    def __init__(
        self,
        n: int,
        edges: np.ndarray,
        name: str = "",
        solution: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.name = name
        self.n = int(n)
        edges = np.asarray(edges, dtype=np.int64)
        if edges.size == 0:
            edges = edges.reshape(0, 2)
        if edges.ndim != 2 or edges.shape[1] != 2:
            raise ValueError(f"edges must have shape [m,2], got {edges.shape}")
        mask = edges[:, 0] != edges[:, 1]
        edges = edges[mask]
        if edges.shape[0]:
            edges = np.sort(edges, axis=1)
            edges = np.unique(edges, axis=0)
        self.edges = edges
        self.m = int(edges.shape[0])
        self.solution = None if solution is None else canonicalize_colors(np.asarray(solution, dtype=np.int16))
        self.metadata = metadata or {}
        self.adj_list: List[np.ndarray] = [np.empty(0, dtype=np.int32) for _ in range(self.n)]
        neighs: List[List[int]] = [[] for _ in range(self.n)]
        for u, v in self.edges:
            neighs[int(u)].append(int(v))
            neighs[int(v)].append(int(u))
        self.adj_list = [np.asarray(sorted(xs), dtype=np.int32) for xs in neighs]
        self.degrees = np.asarray([len(xs) for xs in neighs], dtype=np.int32)
        self.max_degree = int(self.degrees.max()) if self.n else 0
        self.density = 0.0 if self.n <= 1 else (2.0 * self.m) / (self.n * (self.n - 1))
        self.degeneracy_hint = degeneracy_plus_one(self.adj_list)
        self.clique_lb = greedy_clique_lower_bound(self.adj_list, self.degrees)
        self.lower_bound = self.clique_lb
        self.dsatur_ub = dsatur_upper_bound(self)


@dataclasses.dataclass
class RepairState:
    colors: np.ndarray  # int16 [n], canonicalized into [0, k-1]
    k: int
    plateau: int = 0
    step: int = 0

    def copy(self) -> "RepairState":
        return RepairState(self.colors.copy(), int(self.k), int(self.plateau), int(self.step))


@dataclasses.dataclass
class StateMetrics:
    conflicts: int
    conflict_vertices: int
    core_size: int
    class_sizes: np.ndarray
    class_conflicts: np.ndarray
    local_conflicts: np.ndarray
    conflict_mask: np.ndarray
    core_mask: np.ndarray
    same_color_neighbors: np.ndarray
    distinct_neighbor_colors: np.ndarray
    legal_color_counts: np.ndarray
    zero_slack_vertices: int
    one_slack_vertices: int
    mean_legal_colors: float
    patchable_score: float
    entropy: float


@dataclasses.dataclass(frozen=True)
class CandidateAction:
    selector: str
    family: str
    args: Tuple[int, ...]
    params: Tuple[Any, ...]
    est_delta_conflicts: float
    est_delta_vertices: float
    est_cost: float
    token: Tuple[Any, ...]
    meta: Tuple[Any, ...] = ()


@dataclasses.dataclass
class TraceSample:
    global_feats: np.ndarray
    vertex_tokens: np.ndarray
    vertex_mask: np.ndarray
    class_tokens: np.ndarray
    class_mask: np.ndarray
    action_tokens: np.ndarray
    action_mask: np.ndarray
    target_action: int
    target_policy: Optional[np.ndarray] = None
    value_target: float = 0.0
    reward: float = 0.0
    chosen_family: str = PrimitiveFamily.VERTEX_RECOLOR.value
    episode_id: str = ''
    step: int = 0


@dataclasses.dataclass
class SearchConfig:
    cpuct: float = 1.25
    gamma: float = 1.0
    simulations: int = 256
    max_depth: int = 48
    prune_every: int = 32
    prune_min_visits: int = 4
    prune_keep_topk: int = 4
    confidence_beta: float = 1.5
    action_budget: int = DEFAULT_ACTION_BUDGET
    exact_patch_limit: int = 12
    profile_every: int = 0
    # Mean-for-search / max-for-train controls.
    search_alpha_mean: float = 1.0
    search_beta_max: float = 0.0
    novelty_coef: float = 0.0
    # Parallel search controls.
    worker_count: int = 1
    worker_rounds: int = 1
    virtual_loss: float = 0.0
    # Distinct-terminal tracking controls.
    track_distinct_terminals: bool = False
    # Explicit mode split: collect (weak prior, more diversity) vs infer (strong prior).
    search_mode: str = "collect"
    collect_prior_mix: float = 0.30
    collect_prior_temp: float = 1.8
    infer_prior_mix: float = 0.0
    infer_prior_temp: float = 1.0
    # Root worker allocator weights (best-through, uncertainty, novelty).
    alloc_lambda_best: float = 1.0
    alloc_lambda_uncertainty: float = 0.5
    alloc_lambda_novelty: float = 0.25
    # Learning target shaping from search statistics.
    train_policy_target_mode: str = "best_through"  # qmax|best_through|topk_mean|logsumexp
    train_topk_k: int = 3
    train_lse_beta: float = 4.0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ddp_is_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if ddp_is_initialized() else 0


def get_world_size() -> int:
    return dist.get_world_size() if ddp_is_initialized() else 1


def is_main_process() -> bool:
    return get_rank() == 0


def init_distributed(backend: Optional[str] = None) -> Tuple[torch.device, bool]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return device, False
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    backend = backend or ("nccl" if torch.cuda.is_available() else "gloo")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    dist.init_process_group(backend=backend, init_method="env://")
    return device, True


def cleanup_distributed() -> None:
    if ddp_is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def canonicalize_colors(colors: np.ndarray) -> np.ndarray:
    colors = np.asarray(colors, dtype=np.int16)
    remap: Dict[int, int] = {}
    nxt = 0
    out = np.empty_like(colors)
    for i, c in enumerate(colors.tolist()):
        if c not in remap:
            remap[c] = nxt
            nxt += 1
        out[i] = remap[c]
    return out


def degeneracy_plus_one(adj_list: List[np.ndarray]) -> int:
    if not adj_list:
        return 0
    n = len(adj_list)
    deg = np.asarray([len(xs) for xs in adj_list], dtype=np.int32)
    remaining = deg.copy()
    removed = np.zeros(n, dtype=np.bool_)
    import heapq
    heap: List[Tuple[int, int]] = [(int(remaining[v]), int(v)) for v in range(n)]
    heapq.heapify(heap)
    core = 0
    while heap:
        d, v = heapq.heappop(heap)
        if removed[v] or d != int(remaining[v]):
            continue
        removed[v] = True
        core = max(core, int(d))
        for u in adj_list[v]:
            u = int(u)
            if not removed[u]:
                remaining[u] -= 1
                heapq.heappush(heap, (int(remaining[u]), u))
    return int(core + 1)


def greedy_clique_lower_bound(adj_list: List[np.ndarray], degrees: np.ndarray) -> int:
    n = len(adj_list)
    if n == 0:
        return 0
    order = np.argsort(-degrees)
    clique: List[int] = []
    neigh_sets = [set(xs.tolist()) for xs in adj_list]
    for v in order.tolist():
        if all(v in neigh_sets[u] for u in clique):
            clique.append(v)
    return max(1, len(clique))


def dsatur_upper_bound(graph: GCGraph) -> int:
    n = graph.n
    if n == 0:
        return 0
    colors = np.full(n, -1, dtype=np.int32)
    sat = np.zeros(n, dtype=np.int32)
    uncolored = set(range(n))
    neigh_colors: List[set[int]] = [set() for _ in range(n)]
    while uncolored:
        v = max(uncolored, key=lambda x: (sat[x], graph.degrees[x], -x))
        used = neigh_colors[v]
        c = 0
        while c in used:
            c += 1
        colors[v] = c
        uncolored.remove(v)
        for u in graph.adj_list[v]:
            if colors[u] < 0 and c not in neigh_colors[u]:
                neigh_colors[u].add(int(c))
                sat[u] = len(neigh_colors[u])
    return int(colors.max()) + 1


def greedy_k_assignment(graph: GCGraph, k: int, order: Optional[np.ndarray] = None) -> np.ndarray:
    if k <= 0:
        raise ValueError("k must be positive")
    n = graph.n
    if order is None:
        order = np.argsort(-graph.degrees)
    colors = np.full(n, -1, dtype=np.int16)
    for v in order.tolist():
        penalties = np.zeros(k, dtype=np.int32)
        for u in graph.adj_list[v]:
            cu = int(colors[int(u)])
            if cu >= 0:
                penalties[cu] += 1
        best_pen = int(penalties.min())
        best_colors = np.where(penalties == best_pen)[0]
        if best_colors.size > 1:
            class_sizes = np.bincount(np.clip(colors.astype(np.int64), 0, max(k - 1, 0)), minlength=k).astype(np.int32)
            best_color = int(best_colors[np.argmin(class_sizes[best_colors])])
        else:
            best_color = int(best_colors[0])
        colors[v] = best_color
    return canonicalize_colors(colors)


def corrupt_solution(solution: np.ndarray, k: int, rate: float, rng: np.random.Generator) -> np.ndarray:
    colors = canonicalize_colors(np.asarray(solution, dtype=np.int16)).copy()
    n = colors.shape[0]
    rate = float(np.clip(rate, 0.0, 1.0))
    num = max(1, int(round(rate * n)))
    idx = rng.choice(n, size=num, replace=False)
    perm = np.arange(k, dtype=np.int16)
    rng.shuffle(perm)
    colors = perm[colors % k]
    new_cols = rng.integers(0, k, size=num, dtype=np.int16)
    colors[idx] = new_cols
    return canonicalize_colors(colors)


def deterministic_rng_seed(colors: np.ndarray, salt: int = 0) -> int:
    h = hashlib.blake2b(colors.tobytes(), digest_size=8)
    return int.from_bytes(h.digest(), "little") ^ int(salt)


def compute_state_metrics(graph: GCGraph, state: RepairState, core_radius: int = 1) -> StateMetrics:
    colors = state.colors
    n = graph.n
    same_color = np.zeros(n, dtype=np.int32)
    local_conflicts = np.zeros(n, dtype=np.int32)
    conflict_mask = np.zeros(n, dtype=np.bool_)
    class_sizes = np.bincount(colors.astype(np.int64), minlength=state.k).astype(np.int32)
    class_conflicts = np.zeros(state.k, dtype=np.int32)
    distinct_sets: List[set[int]] = [set() for _ in range(n)]
    for u, v in graph.edges:
        cu = int(colors[u])
        cv = int(colors[v])
        distinct_sets[int(u)].add(cv)
        distinct_sets[int(v)].add(cu)
        if cu == cv:
            same_color[int(u)] += 1
            same_color[int(v)] += 1
            local_conflicts[int(u)] += 1
            local_conflicts[int(v)] += 1
            conflict_mask[int(u)] = True
            conflict_mask[int(v)] = True
            class_conflicts[cu] += 1
    distinct_neighbor_colors = np.asarray([len(s) for s in distinct_sets], dtype=np.int32)
    legal_color_counts = np.maximum(state.k - distinct_neighbor_colors, 0).astype(np.int32)
    conflicts = int(local_conflicts.sum() // 2)
    conflict_vertices = int(conflict_mask.sum())
    core_mask = conflict_mask.copy()
    if core_radius > 0 and conflict_vertices > 0:
        frontier = np.where(conflict_mask)[0].tolist()
        for _ in range(core_radius):
            nxt: List[int] = []
            for v in frontier:
                for u in graph.adj_list[v]:
                    if not core_mask[u]:
                        core_mask[u] = True
                        nxt.append(int(u))
            frontier = nxt
            if not frontier:
                break
    core_size = int(core_mask.sum())
    zero_slack_vertices = int((legal_color_counts == 0).sum())
    one_slack_vertices = int((legal_color_counts == 1).sum())
    mean_legal_colors = float(legal_color_counts.mean()) if legal_color_counts.size else 0.0
    patchable_score = 1.0 if core_size == 0 else min(1.0, 12.0 / max(core_size, 1))
    probs = class_sizes.astype(np.float64) / max(int(class_sizes.sum()), 1)
    entropy = float(-(probs[probs > 0] * np.log(probs[probs > 0] + EPS)).sum())
    return StateMetrics(
        conflicts=conflicts,
        conflict_vertices=conflict_vertices,
        core_size=core_size,
        class_sizes=class_sizes,
        class_conflicts=class_conflicts,
        local_conflicts=local_conflicts,
        conflict_mask=conflict_mask,
        core_mask=core_mask,
        same_color_neighbors=same_color,
        distinct_neighbor_colors=distinct_neighbor_colors,
        legal_color_counts=legal_color_counts,
        zero_slack_vertices=zero_slack_vertices,
        one_slack_vertices=one_slack_vertices,
        mean_legal_colors=mean_legal_colors,
        patchable_score=patchable_score,
        entropy=entropy,
    )


def state_key(state: RepairState) -> bytes:
    colors = canonicalize_colors(state.colors)
    header = np.asarray([state.k], dtype=np.int16).tobytes()
    return header + colors.tobytes()


def local_delta_for_recolor(graph: GCGraph, state: RepairState, metrics: StateMetrics, v: int, new_color: int) -> Tuple[int, int]:
    old_color = int(state.colors[v])
    if old_color == new_color:
        return 0, 0
    old_conflicts = 0
    new_conflicts = 0
    old_v_conflict = metrics.local_conflicts[v] > 0
    new_v_conflict = False
    changed_vertices = {v}
    for u in graph.adj_list[v]:
        if int(state.colors[u]) == old_color:
            old_conflicts += 1
        if int(state.colors[u]) == new_color:
            new_conflicts += 1
            new_v_conflict = True
        if int(state.colors[u]) in (old_color, new_color):
            changed_vertices.add(int(u))
    delta_edges = old_conflicts - new_conflicts
    # Approximate delta conflicting-vertices locally.
    delta_vertices = 0
    if old_v_conflict and not new_v_conflict:
        delta_vertices += 1
    elif (not old_v_conflict) and new_v_conflict:
        delta_vertices -= 1
    for u in changed_vertices:
        if u == v:
            continue
        cu = int(state.colors[u])
        before = metrics.local_conflicts[u] > 0
        after_count = metrics.local_conflicts[u]
        if cu == old_color:
            after_count -= 1
        if cu == new_color:
            after_count += 1
        after = after_count > 0
        if before and not after:
            delta_vertices += 1
        elif (not before) and after:
            delta_vertices -= 1
    return int(delta_edges), int(delta_vertices)


def kempe_component(graph: GCGraph, colors: np.ndarray, start_v: int, c1: int, c2: int) -> np.ndarray:
    keep = np.zeros(graph.n, dtype=np.bool_)
    if int(colors[start_v]) not in (c1, c2):
        return keep
    stack = [int(start_v)]
    keep[start_v] = True
    while stack:
        v = stack.pop()
        for u in graph.adj_list[v]:
            if not keep[u] and int(colors[u]) in (c1, c2):
                keep[u] = True
                stack.append(int(u))
    return keep


def greedy_tabu_burst(
    graph: GCGraph,
    state: RepairState,
    steps: int,
    tenure: int,
    focus_mask: Optional[np.ndarray] = None,
) -> RepairState:
    cur = state.copy()
    tabu = np.zeros((graph.n, cur.k), dtype=np.int32)
    best = cur.copy()
    best_metrics = compute_state_metrics(graph, best)
    cur_metrics = best_metrics
    for t in range(steps):
        if cur_metrics.conflicts == 0:
            return cur
        candidates: List[Tuple[int, int, int, int]] = []
        vertices = np.where(cur_metrics.conflict_mask)[0] if focus_mask is None else np.where(cur_metrics.conflict_mask & focus_mask)[0]
        if vertices.size == 0:
            vertices = np.where(cur_metrics.conflict_mask)[0]
        for v in vertices[:64].tolist():
            old = int(cur.colors[v])
            for c in range(cur.k):
                if c == old:
                    continue
                delta_e, _ = local_delta_for_recolor(graph, cur, cur_metrics, int(v), c)
                aspiration = cur_metrics.conflicts - delta_e < best_metrics.conflicts
                is_tabu = tabu[int(v), c] > t
                if is_tabu and not aspiration:
                    continue
                candidates.append((delta_e, -graph.degrees[v], int(v), int(c)))
        if not candidates:
            break
        candidates.sort(reverse=True)
        _, _, v, c = candidates[0]
        old = int(cur.colors[v])
        cur.colors[v] = int(c)
        cur.colors = canonicalize_colors(cur.colors)
        cur.step += 1
        cur_metrics = compute_state_metrics(graph, cur)
        tabu[v, old] = t + tenure
        if cur_metrics.conflicts < best_metrics.conflicts:
            best = cur.copy()
            best_metrics = cur_metrics
            cur.plateau = 0
        else:
            cur.plateau += 1
    return best


def exact_patch_backtracking(
    graph: GCGraph,
    state: RepairState,
    metrics: StateMetrics,
    limit: int,
) -> Optional[RepairState]:
    core_vertices = np.where(metrics.core_mask)[0]
    if core_vertices.size == 0 or core_vertices.size > limit:
        return None
    core_set = set(core_vertices.tolist())
    order = sorted(core_vertices.tolist(), key=lambda v: (-metrics.local_conflicts[v], -graph.degrees[v]))
    fixed_forbidden: Dict[int, set[int]] = {}
    for v in order:
        forb: set[int] = set()
        for u in graph.adj_list[v]:
            if int(u) not in core_set:
                forb.add(int(state.colors[u]))
        fixed_forbidden[int(v)] = forb
    colors = state.colors.copy()
    domains: Dict[int, List[int]] = {v: [c for c in range(state.k) if c not in fixed_forbidden[v]] for v in order}
    neigh_core = {v: [int(u) for u in graph.adj_list[v] if int(u) in core_set] for v in order}

    def dfs(idx: int) -> bool:
        if idx == len(order):
            return True
        v = order[idx]
        used = {int(colors[u]) for u in neigh_core[v] if u in order[:idx]}
        dom = [c for c in domains[v] if c not in used]
        if not dom:
            return False
        dom.sort(key=lambda c: 0 if c == int(state.colors[v]) else 1)
        for c in dom:
            colors[v] = int(c)
            if dfs(idx + 1):
                return True
        return False

    if dfs(0):
        out = RepairState(canonicalize_colors(colors), state.k, 0, state.step + 1)
        out_metrics = compute_state_metrics(graph, out)
        if out_metrics.conflicts <= metrics.conflicts:
            return out
    return None


def apply_candidate_action(
    graph: GCGraph,
    state: RepairState,
    metrics: StateMetrics,
    action: CandidateAction,
    macros: Optional[Dict[str, MacroProgram]] = None,
    exact_patch_limit: int = 12,
) -> RepairState:
    fam = action.family
    if fam == PrimitiveFamily.VERTEX_RECOLOR.value:
        v, c = action.args[:2]
        nxt = state.copy()
        nxt.colors[int(v)] = int(c)
        nxt.colors = canonicalize_colors(nxt.colors)
        nxt.step += 1
        return nxt
    if fam == PrimitiveFamily.KEMPE_SWAP.value:
        v, c_new = action.args[:2]
        old = int(state.colors[int(v)])
        keep = kempe_component(graph, state.colors, int(v), old, int(c_new))
        nxt = state.copy()
        idx = np.where(keep)[0]
        if idx.size:
            old_mask = nxt.colors[idx] == old
            nxt.colors[idx[old_mask]] = int(c_new)
            nxt.colors[idx[~old_mask]] = old
            nxt.colors = canonicalize_colors(nxt.colors)
        nxt.step += 1
        return nxt
    if fam == PrimitiveFamily.TABU_SHORT.value:
        return greedy_tabu_burst(graph, state, steps=max(4, int(action.args[0])), tenure=max(3, int(action.args[1])))
    if fam == PrimitiveFamily.TABU_LONG.value:
        return greedy_tabu_burst(graph, state, steps=max(8, int(action.args[0])), tenure=max(5, int(action.args[1])))
    if fam == PrimitiveFamily.FOCUS_CORE.value:
        return greedy_tabu_burst(graph, state, steps=max(4, int(action.args[0])), tenure=max(3, int(action.args[1])), focus_mask=metrics.core_mask)
    if fam == PrimitiveFamily.PERTURB_SOFT.value:
        nxt = state.copy()
        seed = deterministic_rng_seed(state.colors, salt=int(action.args[0]) + 17 * state.step)
        rng = np.random.default_rng(seed)
        frac = float(action.meta[0]) if action.meta else 0.08
        num = max(1, int(round(frac * graph.n)))
        conflict_vertices = np.where(metrics.conflict_mask)[0]
        if conflict_vertices.size == 0:
            conflict_vertices = np.argsort(-graph.degrees)[: max(1, num)]
        if conflict_vertices.size > num:
            chosen = rng.choice(conflict_vertices, size=num, replace=False)
        else:
            chosen = conflict_vertices
        nxt.colors[chosen] = rng.integers(0, state.k, size=chosen.shape[0], dtype=np.int16)
        nxt.colors = canonicalize_colors(nxt.colors)
        nxt.plateau += 1
        nxt.step += 1
        return nxt
    if fam == PrimitiveFamily.EXACT_PATCH.value:
        patched = exact_patch_backtracking(graph, state, metrics, limit=min(exact_patch_limit, int(action.args[0])))
        if patched is not None:
            return patched
        return state.copy()
    if fam == PrimitiveFamily.MACRO.value:
        if not macros:
            return state.copy()
        macro_name = str(action.meta[0]) if action.meta else str(action.args[0])
        macro = macros.get(macro_name)
        if macro is None:
            return state.copy()
        # Cheap macro execution knobs (controlled by the caller via env vars, to keep v2 API stable):
        #   GCP_MACRO_ACTION_BUDGET (int, default 32): internal candidate generation budget per family step
        #   GCP_MACRO_MAX_STEPS     (int, default 0 = len(families)): cap on executed macro steps
        #   GCP_MACRO_CHEAP         ("1"/"true" enables): skip full evaluate_candidates, pick candidate by greedy delta
        try:
            macro_budget = int(os.environ.get("GCP_MACRO_ACTION_BUDGET", "32"))
        except ValueError:
            macro_budget = 32
        try:
            macro_max_steps = int(os.environ.get("GCP_MACRO_MAX_STEPS", "0"))
        except ValueError:
            macro_max_steps = 0
        if macro_max_steps <= 0:
            macro_max_steps = len(macro.families)
        cheap = str(os.environ.get("GCP_MACRO_CHEAP", "0")).lower() in ("1", "true", "yes")
        cur = state.copy()
        for fam_i, family in enumerate(macro.families[: int(macro_max_steps)]):
            cur_metrics = compute_state_metrics(graph, cur)
            cands = generate_candidate_actions(
                graph,
                cur,
                cur_metrics,
                list(macros.values()),
                action_budget=int(macro_budget),
                restrict_families={family},
                exact_patch_limit=exact_patch_limit,
            )
            if not cands:
                continue
            if cheap:
                # Cheap path: prefer biggest expected drop in conflicts (no network re-eval, no teacher scoring).
                best_idx = int(
                    max(
                        range(len(cands)),
                        key=lambda j: (float(cands[j].est_delta_conflicts), float(cands[j].est_delta_vertices), -float(cands[j].est_cost)),
                    )
                )
            else:
                scored = evaluate_candidates(graph, cur, cur_metrics, cands, macros=macros, exact_patch_limit=exact_patch_limit)
                best_idx = int(np.argmax(scored["teacher_scores"]))
            cur = apply_candidate_action(graph, cur, cur_metrics, cands[best_idx], macros=macros, exact_patch_limit=exact_patch_limit)
        return cur
    raise ValueError(f"unknown family: {fam}")


def transition_state(
    graph: GCGraph,
    state: RepairState,
    metrics: StateMetrics,
    action: CandidateAction,
    macros: Optional[Dict[str, MacroProgram]] = None,
    exact_patch_limit: int = 12,
) -> Tuple[RepairState, StateMetrics, float]:
    nxt = apply_candidate_action(graph, state, metrics, action, macros=macros, exact_patch_limit=exact_patch_limit)
    nxt_metrics = compute_state_metrics(graph, nxt)
    improved = (nxt_metrics.conflicts < metrics.conflicts) or (
        nxt_metrics.conflicts == metrics.conflicts and nxt_metrics.conflict_vertices < metrics.conflict_vertices
    ) or (
        nxt_metrics.conflicts == metrics.conflicts and nxt_metrics.conflict_vertices == metrics.conflict_vertices and nxt_metrics.core_size < metrics.core_size
    )
    nxt.plateau = 0 if improved else min(int(state.plateau) + 1, 255)
    reward = dense_reward(graph, metrics, nxt_metrics, action_cost=action.est_cost)
    return nxt, nxt_metrics, reward


def dense_reward(
    graph: GCGraph,
    before: StateMetrics,
    after: StateMetrics,
    action_cost: float,
    terminal_bonus: float = 1.0,
) -> float:
    r = 0.0
    r += 1.00 * (before.conflicts - after.conflicts) / max(graph.m, 1)
    r += 0.35 * (before.conflict_vertices - after.conflict_vertices) / max(graph.n, 1)
    r += 0.20 * (before.core_size - after.core_size) / max(graph.n, 1)
    r += 0.05 * (after.entropy - before.entropy) / max(math.log(max(after.class_sizes.sum(), 2)), 1.0)
    r += 0.08 * (after.mean_legal_colors - before.mean_legal_colors) / max(after.class_sizes.sum(), 1)
    r += 0.08 * (before.zero_slack_vertices - after.zero_slack_vertices) / max(graph.n, 1)
    r += 0.04 * (after.patchable_score - before.patchable_score)
    r -= 0.02 * action_cost
    if after.conflicts == 0:
        r += terminal_bonus
    return float(r)


def color_alignment_score(cur: np.ndarray, target: np.ndarray, k: int) -> float:
    cur = canonicalize_colors(cur)
    target = canonicalize_colors(target)
    if linear_sum_assignment is None:
        eq = cur[:, None] == target[None, :]
        return float(np.trace(eq.astype(np.float64)[: min(eq.shape[0], eq.shape[1]), : min(eq.shape[0], eq.shape[1])])) / max(cur.shape[0], 1)
    K = max(k, int(cur.max()) + 1 if cur.size else 0, int(target.max()) + 1 if target.size else 0)
    conf = np.zeros((K, K), dtype=np.int32)
    for a, b in zip(cur.tolist(), target.tolist()):
        conf[int(a), int(b)] += 1
    row, col = linear_sum_assignment(-conf)
    matched = conf[row, col].sum()
    return float(matched) / max(cur.shape[0], 1)


def family_token(action: CandidateAction) -> str:
    return action.family


def macro_applicable(macro: MacroProgram, metrics: StateMetrics, graph: GCGraph) -> bool:
    cr = metrics.conflicts / max(graph.m, 1)
    hr = metrics.core_size / max(graph.n, 1)
    return macro.min_conflict_ratio <= cr <= macro.max_conflict_ratio and macro.min_core_ratio <= hr <= macro.max_core_ratio


def color_signature(metrics: StateMetrics, c: int) -> Tuple[int, int]:
    return int(metrics.class_sizes[c]), int(metrics.class_conflicts[c])


def generate_candidate_actions(
    graph: GCGraph,
    state: RepairState,
    metrics: StateMetrics,
    macros: Optional[Sequence[MacroProgram]] = None,
    action_budget: int = DEFAULT_ACTION_BUDGET,
    restrict_families: Optional[set[str]] = None,
    exact_patch_limit: int = 12,
) -> List[CandidateAction]:
    colors = state.colors
    k = state.k
    n = graph.n
    candidates: List[CandidateAction] = []
    top_vertices = np.argsort(-metrics.local_conflicts)
    top_vertices = top_vertices[metrics.local_conflicts[top_vertices] > 0]
    if top_vertices.size == 0:
        top_vertices = np.argsort(-graph.degrees)[: min(DEFAULT_VERTEX_BUDGET, n)]
    top_vertices = top_vertices[: min(DEFAULT_VERTEX_BUDGET, top_vertices.size)]
    class_sizes = metrics.class_sizes

    def accept(family: str) -> bool:
        return restrict_families is None or family in restrict_families

    # 1) local recolors
    if accept(PrimitiveFamily.VERTEX_RECOLOR.value):
        for v in top_vertices.tolist():
            current = int(colors[v])
            color_order = np.argsort(metrics.class_conflicts + 0.2 * class_sizes)
            for c in color_order.tolist()[: min(k, 4)]:
                c = int(c)
                if c == current:
                    continue
                de, dv = local_delta_for_recolor(graph, state, metrics, int(v), c)
                candidates.append(
                    CandidateAction(
                        selector="vertex",
                        family=PrimitiveFamily.VERTEX_RECOLOR.value,
                        args=(int(v), int(c)),
                        params=(int(v), int(c)),
                        est_delta_conflicts=float(de),
                        est_delta_vertices=float(dv),
                        est_cost=1.0,
                        token=(PrimitiveFamily.VERTEX_RECOLOR.value, int(v), color_signature(metrics, int(c))),
                    )
                )

    # 2) kempe
    if accept(PrimitiveFamily.KEMPE_SWAP.value):
        for v in top_vertices.tolist()[: min(12, top_vertices.size)]:
            current = int(colors[v])
            for c in range(k):
                if c == current:
                    continue
                de, dv = local_delta_for_recolor(graph, state, metrics, int(v), c)
                candidates.append(
                    CandidateAction(
                        selector="vertex",
                        family=PrimitiveFamily.KEMPE_SWAP.value,
                        args=(int(v), int(c)),
                        params=(int(v), int(c)),
                        est_delta_conflicts=float(max(0, de)),
                        est_delta_vertices=float(dv),
                        est_cost=2.0,
                        token=(PrimitiveFamily.KEMPE_SWAP.value, int(v), color_signature(metrics, int(c))),
                    )
                )

    # 3) tabu and focus
    if accept(PrimitiveFamily.TABU_SHORT.value):
        candidates.append(CandidateAction("core", PrimitiveFamily.TABU_SHORT.value, (8, 4), (8, 4), 0.0, 0.0, 3.0, (PrimitiveFamily.TABU_SHORT.value, 8, 4)))
    if accept(PrimitiveFamily.TABU_LONG.value):
        candidates.append(CandidateAction("core", PrimitiveFamily.TABU_LONG.value, (16, 6), (16, 6), 0.0, 0.0, 5.0, (PrimitiveFamily.TABU_LONG.value, 16, 6)))
    if accept(PrimitiveFamily.FOCUS_CORE.value):
        candidates.append(CandidateAction("core", PrimitiveFamily.FOCUS_CORE.value, (8, 4), (8, 4), 0.0, 0.0, 4.0, (PrimitiveFamily.FOCUS_CORE.value, 8, 4)))

    # 4) exact patch
    if accept(PrimitiveFamily.EXACT_PATCH.value) and metrics.core_size > 0 and metrics.core_size <= exact_patch_limit:
        candidates.append(CandidateAction("exact", PrimitiveFamily.EXACT_PATCH.value, (exact_patch_limit,), (exact_patch_limit,), 0.0, 0.0, 6.0, (PrimitiveFamily.EXACT_PATCH.value, exact_patch_limit)))

    # 5) perturb only if plateau or hard conflict
    if accept(PrimitiveFamily.PERTURB_SOFT.value) and (state.plateau >= 2 or metrics.conflicts > max(2, graph.n // 8)):
        candidates.append(CandidateAction("diversify", PrimitiveFamily.PERTURB_SOFT.value, (1,), (1,), 0.0, 0.0, 6.0, (PrimitiveFamily.PERTURB_SOFT.value, 1), meta=(0.08,)))

    # 6) macros
    if accept(PrimitiveFamily.MACRO.value) and macros:
        for macro in macros:
            if macro_applicable(macro, metrics, graph):
                candidates.append(
                    CandidateAction(
                        selector="macro",
                        family=PrimitiveFamily.MACRO.value,
                        args=(hash(macro.name) & 0xFFFF,),
                        params=(macro.name,),
                        est_delta_conflicts=0.0,
                        est_delta_vertices=0.0,
                        est_cost=float(len(macro.families)),
                        token=(PrimitiveFamily.MACRO.value, macro.name),
                        meta=(macro.name,),
                    )
                )

    # Deduplicate.
    seen = set()
    dedup: List[CandidateAction] = []
    for cand in candidates:
        if cand.token in seen:
            continue
        seen.add(cand.token)
        dedup.append(cand)
    dedup.sort(key=lambda c: (c.est_delta_conflicts, c.est_delta_vertices, -c.est_cost), reverse=True)
    return dedup[:action_budget]


def candidate_feature_vector(
    graph: GCGraph,
    state: RepairState,
    metrics: StateMetrics,
    action: CandidateAction,
    macros: Optional[Dict[str, MacroProgram]] = None,
) -> np.ndarray:
    x = np.zeros(ACTION_FEAT_DIM, dtype=np.float32)
    family_id = FAMILY_TO_ID[action.family]
    x[family_id] = 1.0
    base = len(FAMILY_TO_ID)
    x[base + 0] = float(action.est_delta_conflicts) / max(graph.m, 1)
    x[base + 1] = float(action.est_delta_vertices) / max(graph.n, 1)
    x[base + 2] = float(action.est_cost) / 16.0
    x[base + 3] = float(metrics.conflicts) / max(graph.m, 1)
    x[base + 4] = float(metrics.conflict_vertices) / max(graph.n, 1)
    x[base + 5] = float(metrics.core_size) / max(graph.n, 1)
    x[base + 6] = float(state.plateau) / 16.0
    if action.family in (PrimitiveFamily.VERTEX_RECOLOR.value, PrimitiveFamily.KEMPE_SWAP.value):
        v = int(action.args[0])
        c = int(action.args[1])
        x[base + 7] = float(graph.degrees[v]) / max(graph.max_degree, 1)
        x[base + 8] = float(metrics.local_conflicts[v]) / max(graph.degrees[v], 1)
        x[base + 9] = float(metrics.class_sizes[c]) / max(graph.n, 1)
        x[base + 10] = float(metrics.class_conflicts[c]) / max(graph.m, 1)
        x[base + 11] = float(metrics.core_mask[v])
        x[base + 12] = float(state.colors[v]) / max(state.k - 1, 1)
        x[base + 13] = float(c) / max(state.k - 1, 1)
        x[base + 14] = float(metrics.legal_color_counts[v]) / max(state.k, 1)
        x[base + 15] = float(metrics.zero_slack_vertices) / max(graph.n, 1)
        x[base + 16] = float(metrics.patchable_score)
    elif action.family == PrimitiveFamily.MACRO.value and action.meta:
        macro = macros.get(str(action.meta[0])) if macros else None
        if macro is not None:
            x[base + 14] = float(len(macro.families)) / 8.0
            x[base + 15] = float(macro.support) / 64.0
            x[base + 16] = float(macro.score) / 64.0
    return x


def vertex_feature_matrix(graph: GCGraph, state: RepairState, metrics: StateMetrics, budget: int) -> Tuple[np.ndarray, np.ndarray]:
    idx = np.argsort(-metrics.local_conflicts)
    idx = idx[: min(budget, graph.n)]
    feats = np.zeros((budget, VERTEX_FEAT_DIM), dtype=np.float32)
    mask = np.zeros(budget, dtype=np.bool_)
    for i, v in enumerate(idx.tolist()):
        mask[i] = True
        c = int(state.colors[v])
        feats[i, 0] = float(metrics.conflict_mask[v])
        feats[i, 1] = float(metrics.core_mask[v])
        feats[i, 2] = float(graph.degrees[v]) / max(graph.max_degree, 1)
        feats[i, 3] = float(metrics.local_conflicts[v]) / max(graph.degrees[v], 1)
        feats[i, 4] = float(metrics.same_color_neighbors[v]) / max(graph.degrees[v], 1)
        feats[i, 5] = float(metrics.distinct_neighbor_colors[v]) / max(state.k, 1)
        feats[i, 6] = float(metrics.class_sizes[c]) / max(graph.n, 1)
        feats[i, 7] = float(metrics.class_conflicts[c]) / max(graph.m, 1)
        feats[i, 8] = float(c) / max(state.k - 1, 1)
        feats[i, 9] = float(v) / max(graph.n - 1, 1)
        feats[i, 10] = float(graph.lower_bound) / max(state.k, 1)
        feats[i, 11] = float(metrics.entropy) / max(math.log(max(state.k, 2)), 1.0)
        feats[i, 12] = float(state.plateau) / 16.0
        feats[i, 13] = float(metrics.conflicts) / max(graph.m, 1)
        feats[i, 14] = float(metrics.legal_color_counts[v]) / max(state.k, 1)
        feats[i, 15] = float(metrics.legal_color_counts[v] <= 1)
    return feats, mask


def class_feature_matrix(graph: GCGraph, state: RepairState, metrics: StateMetrics, budget: int) -> Tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-(metrics.class_conflicts * 4 + metrics.class_sizes))
    order = order[: min(budget, state.k)]
    feats = np.zeros((budget, CLASS_FEAT_DIM), dtype=np.float32)
    mask = np.zeros(budget, dtype=np.bool_)
    max_size = max(int(metrics.class_sizes.max()) if metrics.class_sizes.size else 1, 1)
    max_conf = max(int(metrics.class_conflicts.max()) if metrics.class_conflicts.size else 1, 1)
    for i, c in enumerate(order.tolist()):
        mask[i] = True
        member_mask = state.colors == int(c)
        member_legal = float(metrics.legal_color_counts[member_mask].mean()) if member_mask.any() else 0.0
        feats[i, 0] = float(metrics.class_sizes[c]) / max(graph.n, 1)
        feats[i, 1] = float(metrics.class_conflicts[c]) / max(graph.m, 1)
        feats[i, 2] = float(metrics.class_sizes[c]) / max_size
        feats[i, 3] = float(metrics.class_conflicts[c]) / max_conf
        feats[i, 4] = float(c) / max(state.k - 1, 1)
        feats[i, 5] = float(metrics.class_sizes[c] == metrics.class_sizes.max())
        feats[i, 6] = float(metrics.class_sizes[c] == metrics.class_sizes.min())
        feats[i, 7] = float(metrics.class_conflicts[c] > 0)
        feats[i, 8] = float(metrics.entropy) / max(math.log(max(state.k, 2)), 1.0)
        feats[i, 9] = float(graph.lower_bound) / max(state.k, 1)
        feats[i, 10] = member_legal / max(state.k, 1)
        feats[i, 11] = float(metrics.patchable_score)
    return feats, mask


def regime_bucket(graph: GCGraph) -> int:
    if graph.n == 0:
        return 0
    if graph.density >= 0.35:
        return 2
    if graph.density <= 0.08:
        return 0
    return 1


def global_feature_vector(graph: GCGraph, state: RepairState, metrics: StateMetrics, candidate_count: int) -> np.ndarray:
    x = np.zeros(GLOBAL_FEAT_DIM, dtype=np.float32)
    x[0] = float(graph.n) / 2048.0
    x[1] = float(graph.m) / max(graph.n * max(graph.n - 1, 1) // 2, 1)
    x[2] = float(graph.density)
    x[3] = float(state.k) / max(graph.n, 1)
    x[4] = float(graph.lower_bound) / max(state.k, 1)
    x[5] = float(graph.dsatur_ub) / max(state.k, 1)
    x[6] = float(metrics.conflicts) / max(graph.m, 1)
    x[7] = float(metrics.conflict_vertices) / max(graph.n, 1)
    x[8] = float(metrics.core_size) / max(graph.n, 1)
    x[9] = float(metrics.entropy) / max(math.log(max(state.k, 2)), 1.0)
    x[10] = float(metrics.class_sizes.max()) / max(graph.n, 1)
    x[11] = float(metrics.class_sizes.min()) / max(graph.n, 1)
    x[12] = float(metrics.class_conflicts.max()) / max(graph.m, 1)
    x[13] = float(graph.max_degree) / max(graph.n - 1, 1)
    x[14] = float(metrics.local_conflicts.max()) / max(graph.max_degree, 1)
    x[15] = float(state.plateau) / 16.0
    x[16] = float(state.step) / 128.0
    x[17] = float(candidate_count) / DEFAULT_ACTION_BUDGET
    x[18] = float(metrics.conflicts == 0)
    x[19] = float(metrics.core_size <= 12)
    x[20] = float(graph.clique_lb) / max(state.k, 1)
    x[21] = float(graph.degeneracy_hint) / max(state.k, 1)
    x[22] = float(metrics.zero_slack_vertices) / max(graph.n, 1)
    x[23] = float(regime_bucket(graph)) / 2.0
    return x


def build_observation(
    graph: GCGraph,
    state: RepairState,
    metrics: StateMetrics,
    candidates: List[CandidateAction],
    macros: Optional[Dict[str, MacroProgram]] = None,
    vertex_budget: int = DEFAULT_VERTEX_BUDGET,
    class_budget: int = DEFAULT_CLASS_BUDGET,
    action_budget: int = DEFAULT_ACTION_BUDGET,
) -> Dict[str, np.ndarray]:
    vertex_tokens, vertex_mask = vertex_feature_matrix(graph, state, metrics, vertex_budget)
    class_tokens, class_mask = class_feature_matrix(graph, state, metrics, class_budget)
    action_tokens = np.zeros((action_budget, ACTION_FEAT_DIM), dtype=np.float32)
    action_mask = np.zeros(action_budget, dtype=np.bool_)
    for i, cand in enumerate(candidates[:action_budget]):
        action_mask[i] = True
        action_tokens[i] = candidate_feature_vector(graph, state, metrics, cand, macros=macros)
    global_feats = global_feature_vector(graph, state, metrics, candidate_count=min(action_budget, len(candidates)))
    return {
        "global_feats": global_feats,
        "vertex_tokens": vertex_tokens,
        "vertex_mask": vertex_mask,
        "class_tokens": class_tokens,
        "class_mask": class_mask,
        "action_tokens": action_tokens,
        "action_mask": action_mask,
    }


def evaluate_candidates(
    graph: GCGraph,
    state: RepairState,
    metrics: StateMetrics,
    candidates: Sequence[CandidateAction],
    oracle_solution: Optional[np.ndarray] = None,
    macros: Optional[Dict[str, MacroProgram]] = None,
    exact_patch_limit: int = 12,
) -> Dict[str, np.ndarray]:
    rewards = np.zeros(len(candidates), dtype=np.float32)
    teacher = np.zeros(len(candidates), dtype=np.float32)
    align_before = None
    if oracle_solution is not None:
        align_before = color_alignment_score(state.colors, oracle_solution, state.k)
    for i, cand in enumerate(candidates):
        nxt, nxt_metrics, reward = transition_state(graph, state, metrics, cand, macros=macros, exact_patch_limit=exact_patch_limit)
        rewards[i] = reward
        score = float(reward)
        if align_before is not None:
            align_after = color_alignment_score(nxt.colors, oracle_solution, nxt.k)
            score += 0.15 * (align_after - align_before)
        if nxt_metrics.conflicts == 0:
            score += 0.25
        teacher[i] = score
    return {"rewards": rewards, "teacher_scores": teacher}


def collate_trace_samples(batch: List[TraceSample]) -> Dict[str, torch.Tensor]:
    out: Dict[str, Any] = {}
    out["global_feats"] = torch.from_numpy(np.stack([x.global_feats for x in batch])).float()
    out["vertex_tokens"] = torch.from_numpy(np.stack([x.vertex_tokens for x in batch])).float()
    out["vertex_mask"] = torch.from_numpy(np.stack([x.vertex_mask for x in batch])).bool()
    out["class_tokens"] = torch.from_numpy(np.stack([x.class_tokens for x in batch])).float()
    out["class_mask"] = torch.from_numpy(np.stack([x.class_mask for x in batch])).bool()
    out["action_tokens"] = torch.from_numpy(np.stack([x.action_tokens for x in batch])).float()
    out["action_mask"] = torch.from_numpy(np.stack([x.action_mask for x in batch])).bool()
    out["target_action"] = torch.tensor([x.target_action for x in batch], dtype=torch.long)
    policies: List[np.ndarray] = []
    for x in batch:
        if x.target_policy is None:
            p = np.zeros(x.action_mask.shape[0], dtype=np.float32)
            if 0 <= int(x.target_action) < p.shape[0]:
                p[int(x.target_action)] = 1.0
        else:
            p = np.asarray(x.target_policy, dtype=np.float32)
        policies.append(p)
    out["target_policy"] = torch.from_numpy(np.stack(policies)).float()
    out["value_target"] = torch.tensor([x.value_target for x in batch], dtype=torch.float32)
    out["reward"] = torch.tensor([x.reward for x in batch], dtype=torch.float32)
    out["step"] = torch.tensor([x.step for x in batch], dtype=torch.long)
    out["chosen_family"] = [x.chosen_family for x in batch]
    out["episode_id"] = [x.episode_id for x in batch]
    return out


class ShardedTraceDataset(IterableDataset):
    def __init__(self, shard_pattern: str, shuffle_shards: bool = True, seed: int = 0):
        super().__init__()
        self.files = sorted(glob.glob(shard_pattern))
        if not self.files:
            raise FileNotFoundError(f"No shards match: {shard_pattern}")
        self.shuffle_shards = shuffle_shards
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[TraceSample]:
        files = list(self.files)
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1
        rank = get_rank()
        world_size = get_world_size()
        full_mod = max(world_size * num_workers, 1)
        full_idx = rank * num_workers + worker_id
        rng = random.Random(self.seed + 997 * self.epoch + 131 * rank + worker_id)
        if self.shuffle_shards:
            rng.shuffle(files)
        files = files[full_idx::full_mod]
        for path in files:
            shard = torch.load(path, map_location="cpu", weights_only=False)
            if self.shuffle_shards:
                rng.shuffle(shard)
            for item in shard:
                yield TraceSample(**item)


class MLP(nn.Module):
    def __init__(self, dims: Sequence[int], dropout: float = 0.0):
        super().__init__()
        layers: List[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.GELU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GatedPool(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        score = self.gate(x).squeeze(-1)
        # FP16-safe masking: -1e9 overflows to -inf in half on some PyTorch builds.
        score = score.masked_fill(~mask, torch.finfo(score.dtype).min)
        attn = torch.softmax(score, dim=-1)
        attn = attn * mask.float()
        denom = attn.sum(dim=-1, keepdim=True).clamp_min(EPS)
        attn = attn / denom
        return torch.einsum("bn,bnd->bd", attn, x)


class TRMPolicyValue(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        refine_steps: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.refine_steps = int(refine_steps)
        self.global_proj = MLP([GLOBAL_FEAT_DIM, d_model, d_model], dropout=dropout)
        self.vertex_proj = MLP([VERTEX_FEAT_DIM, d_model, d_model], dropout=dropout)
        self.class_proj = MLP([CLASS_FEAT_DIM, d_model, d_model], dropout=dropout)
        self.action_proj = MLP([ACTION_FEAT_DIM, d_model, d_model], dropout=dropout)
        self.vertex_pool = GatedPool(d_model)
        self.class_pool = GatedPool(d_model)
        self.action_pool = GatedPool(d_model)
        self.init_mlp = MLP([4 * d_model, d_model, d_model], dropout=dropout)
        self.cell = nn.GRUCell(4 * d_model, d_model)
        self.policy_head = MLP([2 * d_model, d_model, 1], dropout=dropout)
        self.value_head = MLP([d_model, d_model, 1], dropout=dropout)
        self.family_head = MLP([d_model, d_model, len(FAMILY_TO_ID)], dropout=dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        g = self.global_proj(batch["global_feats"])
        v = self.vertex_proj(batch["vertex_tokens"])
        c = self.class_proj(batch["class_tokens"])
        a = self.action_proj(batch["action_tokens"])
        vm = batch["vertex_mask"]
        cm = batch["class_mask"]
        am = batch["action_mask"]

        v_pool = self.vertex_pool(v, vm)
        c_pool = self.class_pool(c, cm)
        a_pool = self.action_pool(a, am)
        z = self.init_mlp(torch.cat([g, v_pool, c_pool, a_pool], dim=-1))
        y = a_pool
        aux_values: List[torch.Tensor] = []
        aux_logits: List[torch.Tensor] = []
        for _ in range(self.refine_steps):
            inp = torch.cat([g, v_pool, c_pool, y], dim=-1)
            z = self.norm(self.cell(inp, z))
            z_expand = z[:, None, :].expand(-1, a.shape[1], -1)
            logits = self.policy_head(torch.cat([z_expand, a], dim=-1)).squeeze(-1)
            logits = logits.masked_fill(~am, torch.finfo(logits.dtype).min)
            probs = torch.softmax(logits, dim=-1)
            probs = probs * am.float()
            denom = probs.sum(dim=-1, keepdim=True).clamp_min(EPS)
            probs = probs / denom
            y = torch.einsum("bn,bnd->bd", probs, a)
            value = self.value_head(z).squeeze(-1)
            aux_values.append(value)
            aux_logits.append(logits)
        family_logits = self.family_head(z)
        return {
            "logits": aux_logits[-1],
            "value": aux_values[-1],
            "aux_logits": aux_logits,
            "aux_values": aux_values,
            "family_logits": family_logits,
        }


def compute_loss(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    family_coef: float = 0.1,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    target_action = batch["target_action"]
    target_policy = batch["target_policy"]
    logits = outputs["logits"]
    value = outputs["value"]
    log_probs = F.log_softmax(logits, dim=-1)
    ce = -(target_policy * log_probs).sum(dim=-1).mean()
    value_loss = F.mse_loss(value, batch["value_target"])
    probs = torch.softmax(logits, dim=-1)
    entropy = -(probs * torch.log(probs.clamp_min(EPS))).sum(dim=-1).mean()
    chosen_families = torch.tensor([FAMILY_TO_ID.get(name, 0) for name in batch["chosen_family"]], device=logits.device)
    family_loss = F.cross_entropy(outputs["family_logits"], chosen_families)
    aux = 0.0
    for aux_logits, aux_value in zip(outputs["aux_logits"][:-1], outputs["aux_values"][:-1]):
        aux_log_probs = F.log_softmax(aux_logits, dim=-1)
        aux = aux + 0.5 * (-(target_policy * aux_log_probs).sum(dim=-1).mean()) + 0.25 * F.mse_loss(aux_value, batch["value_target"])
    loss = ce + value_coef * value_loss - entropy_coef * entropy + family_coef * family_loss + 0.5 * aux
    metrics = {
        "loss": float(loss.detach().cpu()),
        "ce": float(ce.detach().cpu()),
        "value": float(value_loss.detach().cpu()),
        "entropy": float(entropy.detach().cpu()),
        "family": float(family_loss.detach().cpu()),
    }
    return loss, metrics


def move_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def reduce_metrics(metrics: Dict[str, float]) -> Dict[str, float]:
    if not ddp_is_initialized():
        return metrics
    keys = sorted(metrics)
    vals = torch.tensor([metrics[k] for k in keys], dtype=torch.float64, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    dist.all_reduce(vals, op=dist.ReduceOp.SUM)
    vals /= get_world_size()
    return {k: float(v) for k, v in zip(keys, vals.cpu().tolist())}


class TraceShardWriter:
    def __init__(self, out_dir: str, prefix: str, samples_per_shard: int = 5000):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.samples_per_shard = int(samples_per_shard)
        self.buffer: List[Dict[str, Any]] = []
        self.shard_id = 0

    def add(self, sample: TraceSample) -> None:
        self.buffer.append(dataclasses.asdict(sample))
        if len(self.buffer) >= self.samples_per_shard:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        path = self.out_dir / f"{self.prefix}-{self.shard_id:05d}.pt"
        torch.save(self.buffer, path)
        self.buffer = []
        self.shard_id += 1


def build_teacher_traces_for_record(
    graph: GCGraph,
    corruptions_per_graph: int,
    max_steps: int,
    macros: Optional[Dict[str, MacroProgram]],
    vertex_budget: int,
    class_budget: int,
    action_budget: int,
    seed: int,
    exact_patch_limit: int = 12,
) -> List[TraceSample]:
    if graph.solution is None:
        raise ValueError(f"Graph {graph.name} has no solution; trace generation from solved examples needs a solution")
    k = int(graph.solution.max()) + 1 if graph.solution.size else 0
    rng = np.random.default_rng(seed)
    out: List[TraceSample] = []
    macro_list = list(macros.values()) if macros else None
    for epi in range(corruptions_per_graph):
        rate = float(rng.uniform(0.08, 0.35))
        cur = RepairState(corrupt_solution(graph.solution, k, rate, rng), k=k, plateau=0, step=0)
        rewards_ep: List[float] = []
        samples_ep: List[TraceSample] = []
        episode_id = f"{graph.name}::corr{epi}"
        for step in range(max_steps):
            metrics = compute_state_metrics(graph, cur)
            if metrics.conflicts == 0:
                break
            candidates = generate_candidate_actions(graph, cur, metrics, macro_list, action_budget=action_budget, exact_patch_limit=exact_patch_limit)
            if not candidates:
                break
            obs = build_observation(graph, cur, metrics, candidates, macros=macros, vertex_budget=vertex_budget, class_budget=class_budget, action_budget=action_budget)
            scored = evaluate_candidates(graph, cur, metrics, candidates, oracle_solution=graph.solution, macros=macros, exact_patch_limit=exact_patch_limit)
            target = int(np.argmax(scored["teacher_scores"]))
            chosen = candidates[target]
            nxt, nxt_metrics, reward = transition_state(graph, cur, metrics, chosen, macros=macros, exact_patch_limit=exact_patch_limit)
            rewards_ep.append(float(reward))
            target_policy = np.zeros(action_budget, dtype=np.float32)
            target_policy[target] = 1.0
            samples_ep.append(
                TraceSample(
                    global_feats=obs["global_feats"],
                    vertex_tokens=obs["vertex_tokens"],
                    vertex_mask=obs["vertex_mask"],
                    class_tokens=obs["class_tokens"],
                    class_mask=obs["class_mask"],
                    action_tokens=obs["action_tokens"],
                    action_mask=obs["action_mask"],
                    target_action=target,
                    target_policy=target_policy,
                    value_target=0.0,
                    reward=float(reward),
                    chosen_family=chosen.family,
                    episode_id=episode_id,
                    step=step,
                )
            )
            cur = nxt
            if nxt_metrics.conflicts == 0:
                break
        rtg = 0.0
        for sample, reward in zip(reversed(samples_ep), reversed(rewards_ep)):
            rtg = float(reward) + rtg
            sample.value_target = float(rtg)
        out.extend(samples_ep)
    return out


def choose_k_for_graph(graph: GCGraph, solution: Optional[np.ndarray], policy: str, explicit_k: Optional[int], offset: int = 1) -> int:
    if explicit_k is not None:
        return int(explicit_k)
    policy = str(policy)
    if policy == "solution" and solution is not None:
        return max(graph.clique_lb, int(solution.max()) + 1)
    if policy == "solution_minus_one" and solution is not None:
        return max(graph.clique_lb, int(solution.max()) + 1 - max(offset, 0))
    if policy == "dsatur":
        return max(graph.clique_lb, graph.dsatur_ub)
    if policy == "dsatur_minus_one":
        return max(graph.clique_lb, graph.dsatur_ub - max(offset, 0))
    return max(graph.clique_lb, graph.dsatur_ub)


def build_solve_traces_for_record(
    graph: GCGraph,
    episodes_per_graph: int,
    max_steps: int,
    model: Optional[TRMPolicyValue],
    device: torch.device,
    macros: Optional[Dict[str, MacroProgram]],
    cfg: SearchConfig,
    vertex_budget: int,
    class_budget: int,
    action_budget: int,
    k_policy: str,
    explicit_k: Optional[int],
    k_offset: int,
    seed: int,
    tree_dump_dir: Optional[Path] = None,
) -> List[TraceSample]:
    out: List[TraceSample] = []
    rng = np.random.default_rng(seed)
    for epi in range(episodes_per_graph):
        k = choose_k_for_graph(graph, graph.solution, k_policy, explicit_k, offset=k_offset)
        order = np.argsort(-(graph.degrees + rng.random(graph.n) * 1e-3))
        seed_colors = greedy_k_assignment(graph, int(k), order=order)
        cur = RepairState(seed_colors, k=int(k), plateau=0, step=0)
        seed_metrics = compute_state_metrics(graph, cur)
        if seed_metrics.conflicts == 0:
            # Force a non-trivial repair episode by perturbing a few vertices.
            num = max(1, int(round(0.08 * graph.n)))
            idx = rng.choice(graph.n, size=num, replace=False)
            cur.colors[idx] = rng.integers(0, int(k), size=num, dtype=np.int16)
            cur.colors = canonicalize_colors(cur.colors)
        rewards_ep: List[float] = []
        samples_ep: List[TraceSample] = []
        episode_id = f"{graph.name}::solve{epi}"
        for step in range(max_steps):
            metrics = compute_state_metrics(graph, cur)
            if metrics.conflicts == 0:
                break
            mcts = GCPMCTS(graph, model, device, macros, cfg)
            root = mcts.run_search(cur)
            if tree_dump_dir is not None:
                tree_dump_dir.mkdir(parents=True, exist_ok=True)
                dump_path = tree_dump_dir / f"{graph.name}__solve{epi:03d}__step{step:03d}.mcts_tree.json"
                with dump_path.open("w") as f:
                    json.dump(serialize_gcp_mcts_tree(mcts, graph, int(k)), f, indent=2)
            if root.metrics is None:
                break
            if not root.candidates:
                break
            obs = build_observation(graph, cur, root.metrics, root.candidates, macros=macros, vertex_budget=vertex_budget, class_budget=class_budget, action_budget=action_budget)
            raw_scores = np.asarray([mcts._edge_train_score(root, a) for a in range(len(root.candidates))], dtype=np.float64)
            if raw_scores.size == 0:
                break
            if not np.isfinite(raw_scores).any():
                raw_scores = root.nsa.astype(np.float64)
            logits = raw_scores - float(np.max(raw_scores))
            probs = np.exp(logits)
            probs = probs / max(float(probs.sum()), EPS)
            target = int(np.argmax(probs[: len(root.candidates)]))
            chosen = root.candidates[target]
            nxt, nxt_metrics, reward = transition_state(graph, cur, root.metrics, chosen, macros=macros, exact_patch_limit=cfg.exact_patch_limit)
            target_policy = np.zeros(action_budget, dtype=np.float32)
            target_policy[: len(root.candidates)] = probs[: len(root.candidates)].astype(np.float32)
            samples_ep.append(
                TraceSample(
                    global_feats=obs["global_feats"],
                    vertex_tokens=obs["vertex_tokens"],
                    vertex_mask=obs["vertex_mask"],
                    class_tokens=obs["class_tokens"],
                    class_mask=obs["class_mask"],
                    action_tokens=obs["action_tokens"],
                    action_mask=obs["action_mask"],
                    target_action=target,
                    target_policy=target_policy,
                    value_target=0.0,
                    reward=float(reward),
                    chosen_family=chosen.family,
                    episode_id=episode_id,
                    step=step,
                )
            )
            rewards_ep.append(float(reward))
            cur = nxt
            if nxt_metrics.conflicts == 0:
                break
        rtg = 0.0
        best_through = -1e18
        for sample, reward in zip(reversed(samples_ep), reversed(rewards_ep)):
            rtg = float(reward) + rtg
            best_through = max(best_through, rtg)
            sample.value_target = float(best_through)
        out.extend(samples_ep)
    return out


def load_records(path: str) -> List[GraphRecord]:
    p = Path(path)
    if p.is_dir():
        out: List[GraphRecord] = []
        for child in sorted(p.iterdir()):
            if child.suffix.lower() in {".jsonl", ".json", ".pkl", ".pt", ".col"}:
                out.extend(load_records(str(child)))
        return out
    suffix = p.suffix.lower()
    if suffix == ".jsonl":
        out = []
        with p.open() as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                out.append(record_from_obj(obj, default_name=p.stem))
        return out
    if suffix == ".json":
        with p.open() as f:
            obj = json.load(f)
        if isinstance(obj, list):
            return [record_from_obj(x, default_name=f"{p.stem}-{i}") for i, x in enumerate(obj)]
        return [record_from_obj(obj, default_name=p.stem)]
    if suffix in {".pkl", ".pickle", ".pt"}:
        obj = torch.load(p, map_location="cpu", weights_only=False) if suffix == ".pt" else pickle.load(p.open("rb"))
        if isinstance(obj, list):
            return [record_from_obj(x, default_name=f"{p.stem}-{i}") for i, x in enumerate(obj)]
        return [record_from_obj(obj, default_name=p.stem)]
    if suffix == ".col":
        return [parse_dimacs_col(str(p))]
    raise ValueError(f"Unsupported input format: {path}")


def record_from_obj(obj: Any, default_name: str = "graph") -> GraphRecord:
    if isinstance(obj, GraphRecord):
        return obj
    if dataclasses.is_dataclass(obj):
        obj = dataclasses.asdict(obj)
    if not isinstance(obj, dict):
        raise ValueError(f"Cannot convert object to GraphRecord: {type(obj)}")
    name = str(obj.get("name", default_name))
    n = int(obj.get("n", obj.get("num_vertices", 0)))
    edges = np.asarray(obj.get("edges", []), dtype=np.int64)
    if n <= 0 and edges.size:
        n = int(edges.max()) + 1
    sol = obj.get("solution")
    solution = None if sol is None else np.asarray(sol, dtype=np.int16)
    meta = {k: v for k, v in obj.items() if k not in {"name", "n", "num_vertices", "edges", "solution"}}
    return GraphRecord(name=name, n=n, edges=edges, solution=solution, metadata=meta)


def parse_dimacs_col(path: str) -> GraphRecord:
    name = Path(path).stem
    n = 0
    edges: List[Tuple[int, int]] = []
    solution: Optional[np.ndarray] = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("c"):
                continue
            if line.startswith("p"):
                parts = line.split()
                if len(parts) >= 4:
                    n = int(parts[2])
            elif line.startswith("e"):
                _, u, v = line.split()[:3]
                edges.append((int(u) - 1, int(v) - 1))
            elif line.startswith("s"):
                arr = [int(x) - 1 for x in line.split()[1:]]
                solution = np.asarray(arr, dtype=np.int16)
    return GraphRecord(name=name, n=n, edges=np.asarray(edges, dtype=np.int64), solution=solution)


def _sanitize_checkpoint_args(args: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in args.items():
        if callable(v):
            continue
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[k] = v
        elif isinstance(v, Path):
            out[k] = str(v)
        else:
            try:
                json.dumps(v)
                out[k] = v
            except Exception:
                out[k] = str(v)
    return out


def save_checkpoint(path: str, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, args: Dict[str, Any]) -> None:
    model_state = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
    torch.save({"model": model_state, "optimizer": optimizer.state_dict(), "epoch": epoch, "args": _sanitize_checkpoint_args(args)}, path)


def load_model_checkpoint(path: str, device: torch.device) -> Tuple[TRMPolicyValue, Dict[str, Any]]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    args = ckpt.get("args", {})
    model = TRMPolicyValue(d_model=int(args.get("d_model", 256)), refine_steps=int(args.get("refine_steps", 3)), dropout=float(args.get("dropout", 0.1)))
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model, ckpt


def train_loop(args: argparse.Namespace) -> None:
    device, distributed = init_distributed(args.backend)
    set_seed(args.seed + get_rank())
    train_ds = ShardedTraceDataset(args.train, shuffle_shards=True, seed=args.seed)
    valid_ds = ShardedTraceDataset(args.valid, shuffle_shards=False, seed=args.seed + 1) if args.valid else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
        collate_fn=collate_trace_samples,
        prefetch_factor=4 if args.num_workers > 0 else None,
    )
    valid_loader = None
    if valid_ds is not None:
        valid_loader = DataLoader(
            valid_ds,
            batch_size=args.batch_size,
            num_workers=max(1, args.num_workers // 2),
            pin_memory=torch.cuda.is_available(),
            persistent_workers=args.num_workers > 0,
            collate_fn=collate_trace_samples,
            prefetch_factor=4 if args.num_workers > 0 else None,
        )
    model = TRMPolicyValue(d_model=args.d_model, refine_steps=args.refine_steps, dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    start_epoch = 0
    if args.init_ckpt:
        ckpt = torch.load(args.init_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"], strict=True)
        if args.resume_optimizer and "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
            start_epoch = int(ckpt.get("epoch", -1)) + 1
    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model)  # type: ignore
    if distributed:
        model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None, find_unused_parameters=False)
    amp_enabled = bool(device.type == "cuda" and args.amp)
    # AMP API compatibility across PyTorch versions.
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler") and hasattr(torch.amp, "autocast"):
        grad_scaler_ctor = lambda: torch.amp.GradScaler("cuda", enabled=amp_enabled)
        autocast_ctx = lambda: torch.amp.autocast("cuda", enabled=amp_enabled)
    elif hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "GradScaler") and hasattr(torch.cuda.amp, "autocast"):
        grad_scaler_ctor = lambda: torch.cuda.amp.GradScaler(enabled=amp_enabled)
        autocast_ctx = lambda: torch.cuda.amp.autocast(enabled=amp_enabled)
    else:
        grad_scaler_ctor = lambda: None
        autocast_ctx = lambda: contextlib.nullcontext()
        amp_enabled = False
    scaler = grad_scaler_ctor()
    out_dir = Path(args.out_dir)
    if is_main_process():
        out_dir.mkdir(parents=True, exist_ok=True)
    best_valid = float("inf")
    history_path = out_dir / "train_log.jsonl"

    for epoch in range(start_epoch, start_epoch + args.epochs):
        if hasattr(train_ds, "set_epoch"):
            train_ds.set_epoch(epoch)
        model.train()
        running = collections.defaultdict(float)
        count = 0
        t0 = time.time()
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader):
            batch = move_to_device(batch, device)
            with autocast_ctx():
                outputs = model(batch)
                loss, metrics = compute_loss(outputs, batch, value_coef=args.value_coef, entropy_coef=args.entropy_coef, family_coef=args.family_coef)
                loss = loss / max(args.grad_accum, 1)
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            if (step + 1) % max(args.grad_accum, 1) == 0:
                if args.grad_clip > 0:
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for k, v in metrics.items():
                running[k] += float(v)
            count += 1
            if is_main_process() and args.log_every > 0 and (step + 1) % args.log_every == 0:
                avg = {k: running[k] / max(count, 1) for k in running}
                print(f"[epoch {epoch:03d} step {step + 1:06d}] " + " ".join(f"{k}={v:.4f}" for k, v in sorted(avg.items())), flush=True)
            if args.steps_per_epoch > 0 and (step + 1) >= args.steps_per_epoch:
                break
        train_metrics = {k: running[k] / max(count, 1) for k in running}
        train_metrics = reduce_metrics(train_metrics)
        valid_metrics: Dict[str, float] = {}
        if valid_loader is not None:
            model.eval()
            vrunning = collections.defaultdict(float)
            vcount = 0
            with torch.no_grad():
                for vstep, batch in enumerate(valid_loader):
                    batch = move_to_device(batch, device)
                    outputs = model(batch)
                    _, metrics = compute_loss(outputs, batch, value_coef=args.value_coef, entropy_coef=args.entropy_coef, family_coef=args.family_coef)
                    for k, v in metrics.items():
                        vrunning[k] += float(v)
                    vcount += 1
                    if args.valid_steps > 0 and (vstep + 1) >= args.valid_steps:
                        break
            valid_metrics = {f"valid_{k}": vrunning[k] / max(vcount, 1) for k in vrunning}
            valid_metrics = reduce_metrics(valid_metrics)
        metrics = {**{f"train_{k}": v for k, v in train_metrics.items()}, **valid_metrics, "epoch": epoch, "time_sec": time.time() - t0}
        if is_main_process():
            with history_path.open("a") as f:
                f.write(json.dumps(metrics) + "\n")
            print(json.dumps(metrics), flush=True)
            save_checkpoint(str(out_dir / "model-last.pt"), model, optimizer, epoch, vars(args))
            current_valid = valid_metrics.get("valid_loss", train_metrics.get("loss", float("inf")))
            if current_valid < best_valid:
                best_valid = current_valid
                save_checkpoint(str(out_dir / "model-best.pt"), model, optimizer, epoch, vars(args))
    cleanup_distributed()


@dataclasses.dataclass
class CacheEntry:
    priors: np.ndarray
    value: float
    solved: bool
    metrics: StateMetrics
    candidates: List[CandidateAction]


class MCTSNode:
    __slots__ = (
        "state",
        "parent",
        "parent_action",
        "children",
        "reward_from_parent",
        "visit",
        "value_sum",
        "expanded",
        "terminal",
        "candidates",
        "priors",
        "nsa",
        "wsa",
        "qsa",
        "qmaxsa",
        "distinct_sumsa",
        "distinct_countsa",
        "topk_returnssa",
        "lse_sum_expsa",
        "lse_countsa",
        "alive",
        "metrics",
        "frozen",
        "dead",
        "solved",
        "best_terminal_conflicts",
        "best_through",
        "tt_key",
    )

    def __init__(self, state: RepairState, parent: int = -1, parent_action: int = -1, reward_from_parent: float = 0.0):
        self.state = state
        self.parent = int(parent)
        self.parent_action = int(parent_action)
        self.children: Dict[int, int] = {}
        self.reward_from_parent = float(reward_from_parent)
        self.visit = 0
        self.value_sum = 0.0
        self.expanded = False
        self.terminal = False
        self.candidates: List[CandidateAction] = []
        self.priors = np.zeros(0, dtype=np.float32)
        self.nsa = np.zeros(0, dtype=np.int32)
        self.wsa = np.zeros(0, dtype=np.float64)
        self.qsa = np.zeros(0, dtype=np.float64)
        self.qmaxsa = np.zeros(0, dtype=np.float64)
        self.distinct_sumsa = np.zeros(0, dtype=np.float64)
        self.distinct_countsa = np.zeros(0, dtype=np.int32)
        self.topk_returnssa: List[List[float]] = []
        self.lse_sum_expsa = np.zeros(0, dtype=np.float64)
        self.lse_countsa = np.zeros(0, dtype=np.int32)
        self.alive = np.zeros(0, dtype=np.bool_)
        self.metrics: Optional[StateMetrics] = None
        self.frozen = False
        self.dead = False
        self.solved = False
        self.best_terminal_conflicts = int(10**9)
        self.best_through = -1e18
        self.tt_key = b""

    @property
    def value(self) -> float:
        return 0.0 if self.visit == 0 else self.value_sum / self.visit


class GCPMCTS:
    def __init__(
        self,
        graph: GCGraph,
        model: Optional[TRMPolicyValue],
        device: torch.device,
        macros: Optional[Dict[str, MacroProgram]],
        cfg: SearchConfig,
    ):
        self.graph = graph
        self.model = model
        self.device = device
        self.macros = macros or {}
        self.cfg = cfg
        self.nodes: List[MCTSNode] = []
        self.cache: Dict[bytes, CacheEntry] = {}
        self.best_state: Optional[RepairState] = None
        self.best_metrics: Optional[StateMetrics] = None
        self.anytime_trace: List[Dict[str, float]] = []
        self.tree_lock = threading.RLock()
        self.distinct_terminals: Set[bytes] = set()
        self.state_to_node: Dict[bytes, int] = {}
        self.root_action_alloc: np.ndarray = np.zeros(0, dtype=np.int32)

    def evaluate_with_model(self, state: RepairState, metrics: StateMetrics, candidates: List[CandidateAction]) -> Tuple[np.ndarray, float]:
        if str(getattr(self.cfg, "search_mode", "collect")) == "noprior":
            priors = np.full(len(candidates), 1.0 / max(len(candidates), 1), dtype=np.float32)
            value = self._rollout_value(state, metrics)
            return priors, float(value)
        if self.model is None:
            scores = np.asarray([
                c.est_delta_conflicts + 0.25 * c.est_delta_vertices - 0.05 * c.est_cost for c in candidates
            ], dtype=np.float32)
            priors = np.exp(scores - scores.max())
            priors = priors / max(priors.sum(), EPS)
            value = float(1.0 - metrics.conflicts / max(self.graph.m, 1) + 0.1 * metrics.mean_legal_colors / max(state.k, 1) + 0.05 * metrics.patchable_score)
            return self._apply_prior_mode(priors), value
        obs = build_observation(self.graph, state, metrics, candidates, macros=self.macros, action_budget=self.cfg.action_budget)
        batch = {
            "global_feats": torch.from_numpy(obs["global_feats"]).unsqueeze(0).to(self.device),
            "vertex_tokens": torch.from_numpy(obs["vertex_tokens"]).unsqueeze(0).to(self.device),
            "vertex_mask": torch.from_numpy(obs["vertex_mask"]).unsqueeze(0).to(self.device),
            "class_tokens": torch.from_numpy(obs["class_tokens"]).unsqueeze(0).to(self.device),
            "class_mask": torch.from_numpy(obs["class_mask"]).unsqueeze(0).to(self.device),
            "action_tokens": torch.from_numpy(obs["action_tokens"]).unsqueeze(0).to(self.device),
            "action_mask": torch.from_numpy(obs["action_mask"]).unsqueeze(0).to(self.device),
        }
        with torch.no_grad():
            out = self.model(batch)
            logits = out["logits"][0].detach().float().cpu().numpy()[: len(candidates)]
            priors = np.exp(logits - logits.max())
            priors = priors / max(priors.sum(), EPS)
            value = float(out["value"][0].detach().cpu())
        priors = self._apply_prior_mode(priors)
        return priors, value

    def _rollout_value(self, state: RepairState, metrics: StateMetrics) -> float:
        cur = state.copy()
        cur_metrics = metrics
        horizon = min(12, max(4, int(self.cfg.max_depth // 4)))
        gamma = float(self.cfg.gamma)
        ret = 0.0
        disc = 1.0
        for _ in range(horizon):
            if int(cur_metrics.conflicts) == 0:
                ret += disc * 1.0
                break
            cands = generate_candidate_actions(
                self.graph,
                cur,
                cur_metrics,
                list(self.macros.values()),
                action_budget=min(self.cfg.action_budget, 24),
                exact_patch_limit=self.cfg.exact_patch_limit,
            )
            if not cands:
                break
            # Pure rollout policy: choose best local-improvement action without model prior.
            cand = max(cands, key=lambda x: (x.est_delta_conflicts, x.est_delta_vertices, -x.est_cost))
            nxt, nxt_metrics, reward = transition_state(self.graph, cur, cur_metrics, cand, macros=self.macros, exact_patch_limit=self.cfg.exact_patch_limit)
            ret += disc * float(reward)
            disc *= gamma
            cur = nxt
            cur_metrics = nxt_metrics
        return float(ret)

    def _apply_prior_mode(self, priors: np.ndarray) -> np.ndarray:
        p = np.asarray(priors, dtype=np.float64)
        if p.size == 0:
            return priors
        mode = str(getattr(self.cfg, "search_mode", "collect"))
        if mode == "noprior":
            uni = np.full_like(p, 1.0 / max(int(p.size), 1), dtype=np.float64)
            return uni.astype(np.float32)
        if mode == "collect":
            temp = max(float(getattr(self.cfg, "collect_prior_temp", 1.8)), 1e-6)
            mix = float(getattr(self.cfg, "collect_prior_mix", 0.30))
        else:
            temp = max(float(getattr(self.cfg, "infer_prior_temp", 1.0)), 1e-6)
            mix = float(getattr(self.cfg, "infer_prior_mix", 0.0))
        p = np.power(np.maximum(p, EPS), 1.0 / temp)
        p = p / max(float(p.sum()), EPS)
        if mix > 0.0:
            uni = np.full_like(p, 1.0 / max(int(p.size), 1), dtype=np.float64)
            p = (1.0 - mix) * p + mix * uni
            p = p / max(float(p.sum()), EPS)
        return p.astype(np.float32)

    def _edge_train_score(self, node: MCTSNode, a: int) -> float:
        mode = str(getattr(self.cfg, "train_policy_target_mode", "best_through"))
        qmax = float(node.qmaxsa[a]) if a < len(node.qmaxsa) and float(node.qmaxsa[a]) > -1e17 else float(node.qsa[a] if a < len(node.qsa) else 0.0)
        if mode == "qmax":
            return qmax
        if mode == "topk_mean":
            vals = node.topk_returnssa[a] if a < len(node.topk_returnssa) else []
            return float(sum(vals) / max(len(vals), 1)) if vals else qmax
        if mode == "logsumexp":
            beta = max(float(getattr(self.cfg, "train_lse_beta", 4.0)), 1e-6)
            s = float(node.lse_sum_expsa[a]) if a < len(node.lse_sum_expsa) else 0.0
            c = int(node.lse_countsa[a]) if a < len(node.lse_countsa) else 0
            if s <= 0.0 or c <= 0:
                return qmax
            return float(math.log(s / max(c, 1)) / beta)
        # best_through default
        if a in node.children:
            cid = int(node.children[a])
            if 0 <= cid < len(self.nodes):
                return float(self.nodes[cid].best_through)
        return qmax

    def _compute_root_allocation(self, batch: int) -> List[int]:
        if not self.nodes:
            return [-1 for _ in range(batch)]
        root = self.nodes[0]
        if not root.expanded or len(root.candidates) == 0:
            return [-1 for _ in range(batch)]
        if self.root_action_alloc.shape[0] != len(root.candidates):
            self.root_action_alloc = np.zeros(len(root.candidates), dtype=np.int32)
        else:
            self.root_action_alloc.fill(0)
        alloc_scores = np.full(len(root.candidates), -1e18, dtype=np.float64)
        total_n = max(root.visit, 1)
        for a in range(len(root.candidates)):
            if a >= len(root.alive) or not bool(root.alive[a]):
                continue
            qmax = root.qmaxsa[a] if a < len(root.qmaxsa) and root.qmaxsa[a] > -1e17 else (root.qsa[a] if a < len(root.qsa) else 0.0)
            uncertainty = math.sqrt(math.log(total_n + 2.0) / (1.0 + float(root.nsa[a] if a < len(root.nsa) else 0.0)))
            novelty = 1.0 / math.sqrt(1.0 + float(root.nsa[a] if a < len(root.nsa) else 0.0) + float(self.root_action_alloc[a]))
            alloc_scores[a] = (
                float(getattr(self.cfg, "alloc_lambda_best", 1.0)) * float(qmax)
                + float(getattr(self.cfg, "alloc_lambda_uncertainty", 0.5)) * float(uncertainty)
                + float(getattr(self.cfg, "alloc_lambda_novelty", 0.25)) * float(novelty)
            )
        if not np.isfinite(alloc_scores).any():
            return [-1 for _ in range(batch)]
        order = np.argsort(-alloc_scores)
        out: List[int] = []
        for i in range(batch):
            a = int(order[i % len(order)])
            if not np.isfinite(alloc_scores[a]):
                out.append(-1)
                continue
            self.root_action_alloc[a] += 1
            out.append(a)
        return out

    def _propagate_status_bottom_up(self, visited_nodes: List[int]) -> None:
        seen: Set[int] = set()
        for nid in reversed(visited_nodes):
            if nid in seen or nid < 0 or nid >= len(self.nodes):
                continue
            seen.add(nid)
            node = self.nodes[nid]
            child_ids = [int(cid) for cid in node.children.values() if 0 <= int(cid) < len(self.nodes)]
            if child_ids:
                child_best = min([self.nodes[cid].best_terminal_conflicts for cid in child_ids] + [node.best_terminal_conflicts])
                node.best_terminal_conflicts = int(child_best)
                node.best_through = max([self.nodes[cid].best_through for cid in child_ids] + [node.best_through])
            node.solved = bool(node.best_terminal_conflicts == 0)
            if node.terminal and node.solved:
                node.frozen = True
            if child_ids:
                all_children_frozen = all(self.nodes[cid].frozen for cid in child_ids)
                all_children_dead = all(self.nodes[cid].dead or self.nodes[cid].frozen for cid in child_ids)
                if all_children_frozen:
                    node.frozen = True
                if all_children_dead and node.expanded:
                    node.dead = True

    def maybe_expand(self, node_id: int) -> float:
        node = self.nodes[node_id]
        if node.expanded:
            return 0.0
        metrics = compute_state_metrics(self.graph, node.state)
        key = state_key(node.state)
        node.tt_key = key
        if key not in self.state_to_node:
            self.state_to_node[key] = int(node_id)
        node.metrics = metrics
        if self.best_metrics is None or metrics.conflicts < self.best_metrics.conflicts:
            self.best_state = node.state.copy()
            self.best_metrics = metrics
        if metrics.conflicts == 0:
            node.expanded = True
            node.terminal = True
            return 1.0
        if key in self.cache:
            cached = self.cache[key]
            node.candidates = cached.candidates
            node.priors = cached.priors.copy()
            node.nsa = np.zeros(len(node.candidates), dtype=np.int32)
            node.wsa = np.zeros(len(node.candidates), dtype=np.float64)
            node.qsa = np.zeros(len(node.candidates), dtype=np.float64)
            node.qmaxsa = np.full(len(node.candidates), -1e18, dtype=np.float64)
            node.distinct_sumsa = np.zeros(len(node.candidates), dtype=np.float64)
            node.distinct_countsa = np.zeros(len(node.candidates), dtype=np.int32)
            node.topk_returnssa = [[] for _ in range(len(node.candidates))]
            node.lse_sum_expsa = np.zeros(len(node.candidates), dtype=np.float64)
            node.lse_countsa = np.zeros(len(node.candidates), dtype=np.int32)
            node.alive = np.ones(len(node.candidates), dtype=np.bool_)
            node.expanded = True
            node.terminal = cached.solved
            return cached.value
        cands = generate_candidate_actions(self.graph, node.state, metrics, list(self.macros.values()), action_budget=self.cfg.action_budget, exact_patch_limit=self.cfg.exact_patch_limit)
        if not cands:
            node.expanded = True
            node.terminal = True
            return float(1.0 - metrics.conflicts / max(self.graph.m, 1))
        priors, value = self.evaluate_with_model(node.state, metrics, cands)
        node.candidates = cands
        node.priors = priors.astype(np.float32)
        node.nsa = np.zeros(len(cands), dtype=np.int32)
        node.wsa = np.zeros(len(cands), dtype=np.float64)
        node.qsa = np.zeros(len(cands), dtype=np.float64)
        node.qmaxsa = np.full(len(cands), -1e18, dtype=np.float64)
        node.distinct_sumsa = np.zeros(len(cands), dtype=np.float64)
        node.distinct_countsa = np.zeros(len(cands), dtype=np.int32)
        node.topk_returnssa = [[] for _ in range(len(cands))]
        node.lse_sum_expsa = np.zeros(len(cands), dtype=np.float64)
        node.lse_countsa = np.zeros(len(cands), dtype=np.int32)
        node.alive = np.ones(len(cands), dtype=np.bool_)
        node.expanded = True
        self.cache[key] = CacheEntry(priors=node.priors.copy(), value=float(value), solved=False, metrics=metrics, candidates=cands)
        return value

    def select_action(self, node_id: int) -> int:
        node = self.nodes[node_id]
        total_n = max(node.visit, 1)
        best_score = -1e18
        best_a = -1
        lcb = np.full(len(node.candidates), -np.inf, dtype=np.float64)
        ucb = np.full(len(node.candidates), -np.inf, dtype=np.float64)
        for a in range(len(node.candidates)):
            if not node.alive[a]:
                continue
            n = node.nsa[a]
            q_mean = node.qsa[a]
            q_max = node.qmaxsa[a] if node.qmaxsa[a] > -1e17 else q_mean
            q_distinct = 0.0 if node.distinct_countsa[a] <= 0 else float(node.distinct_sumsa[a] / max(node.distinct_countsa[a], 1))
            bonus = self.cfg.cpuct * float(node.priors[a]) * math.sqrt(total_n) / (1 + n)
            novelty = self.cfg.novelty_coef / math.sqrt(1.0 + float(n))
            score = (
                self.cfg.search_alpha_mean * q_mean
                + self.cfg.search_beta_max * q_max
                + bonus
                + novelty
                + 0.05 * q_distinct
            )
            if score > best_score:
                best_score = score
                best_a = a
            conf = self.cfg.confidence_beta * math.sqrt(math.log(total_n + 2.0) / (n + 1.0))
            core_q = self.cfg.search_alpha_mean * q_mean + self.cfg.search_beta_max * q_max
            lcb[a] = core_q - conf
            ucb[a] = core_q + conf
        if best_a < 0:
            return 0
        if node.visit > 0 and node.visit % self.cfg.prune_every == 0:
            best_lcb = np.max(lcb[node.alive])
            keep = 0
            order = np.argsort(-ucb)
            alive = np.zeros_like(node.alive)
            for idx in order.tolist():
                if not node.alive[idx]:
                    continue
                if node.nsa[idx] < self.cfg.prune_min_visits or ucb[idx] >= best_lcb - 1e-6 or keep < self.cfg.prune_keep_topk:
                    alive[idx] = True
                    keep += 1
            node.alive = alive
            if not node.alive[best_a]:
                valid = np.where(node.alive)[0]
                best_a = int(valid[0]) if valid.size else 0
        return int(best_a)

    def run_search(self, root_state: RepairState) -> MCTSNode:
        self.nodes = [MCTSNode(root_state.copy())]
        self.nodes[0].tt_key = state_key(root_state)
        self.state_to_node = {self.nodes[0].tt_key: 0}
        self.best_state = root_state.copy()
        self.best_metrics = compute_state_metrics(self.graph, root_state)
        self.anytime_trace = []
        self.distinct_terminals = set()
        t0 = time.time()
        profile_every = max(1, int(getattr(self.cfg, "profile_every", 0) or 0))
        worker_count = max(1, int(getattr(self.cfg, "worker_count", 1)))
        sims_done = 0
        round_width = max(1, worker_count * max(1, int(getattr(self.cfg, "worker_rounds", 1))))
        while sims_done < self.cfg.simulations:
            batch = min(round_width, self.cfg.simulations - sims_done)
            with self.tree_lock:
                forced_roots = self._compute_root_allocation(batch)
            if batch <= 1:
                self._simulate_once(forced_root_action=forced_roots[0] if forced_roots else -1)
            else:
                with concurrent.futures.ThreadPoolExecutor(max_workers=batch) as ex:
                    futs = [ex.submit(self._simulate_once, forced_roots[i] if i < len(forced_roots) else -1) for i in range(batch)]
                    for fut in futs:
                        fut.result()
            sims_done += batch
            if profile_every > 0 and (sims_done % profile_every == 0 or (self.best_metrics is not None and self.best_metrics.conflicts == 0)):
                best_conf = int(self.best_metrics.conflicts) if self.best_metrics is not None else int(10**9)
                self.anytime_trace.append({"simulation": int(sims_done), "best_conflicts": int(best_conf), "elapsed_sec": float(time.time() - t0)})
            if self.best_metrics is not None and self.best_metrics.conflicts == 0:
                break
        return self.nodes[0]

    def _simulate_once(self, forced_root_action: int = -1) -> None:
        path: List[Tuple[int, int, float]] = []
        visited_nodes: List[int] = []
        node_id = 0
        depth = 0
        seen_on_path: Set[int] = set()
        while True:
            with self.tree_lock:
                node = self.nodes[node_id]
                visited_nodes.append(int(node_id))
                leaf_value = self.maybe_expand(node_id)
                node.visit += 1
                if node.terminal or depth >= self.cfg.max_depth or node.dead:
                    backup = float(leaf_value)
                    terminal_conf = int(node.metrics.conflicts) if node.metrics is not None else int(10**9)
                    node.best_terminal_conflicts = min(node.best_terminal_conflicts, terminal_conf)
                    node.best_through = max(float(node.best_through), float(backup))
                    if terminal_conf == 0:
                        node.solved = True
                        node.frozen = True
                        if self.cfg.track_distinct_terminals:
                            self.distinct_terminals.add(state_key(node.state))
                    break
                if depth == 0 and forced_root_action >= 0 and forced_root_action < len(node.candidates) and bool(node.alive[forced_root_action]):
                    a = int(forced_root_action)
                else:
                    a = self.select_action(node_id)
                if a in node.children:
                    child_id = node.children[a]
                    reward = self.nodes[child_id].reward_from_parent
                else:
                    metrics = node.metrics if node.metrics is not None else compute_state_metrics(self.graph, node.state)
                    nxt_state, nxt_metrics, reward = transition_state(self.graph, node.state, metrics, node.candidates[a], macros=self.macros, exact_patch_limit=self.cfg.exact_patch_limit)
                    nxt_key = state_key(nxt_state)
                    if nxt_key in self.state_to_node:
                        child_id = int(self.state_to_node[nxt_key])
                        node.children[a] = child_id
                    else:
                        child_id = len(self.nodes)
                        child_node = MCTSNode(nxt_state, parent=node_id, parent_action=a, reward_from_parent=reward)
                        child_node.tt_key = nxt_key
                        self.nodes.append(child_node)
                        self.state_to_node[nxt_key] = child_id
                        node.children[a] = child_id
                    if self.best_metrics is None or nxt_metrics.conflicts < self.best_metrics.conflicts:
                        self.best_metrics = nxt_metrics
                        self.best_state = nxt_state.copy()
                if child_id in seen_on_path:
                    node.dead = True
                    backup = float(node.qsa[a] if a < len(node.qsa) else 0.0)
                    break
                seen_on_path.add(int(child_id))
                if self.cfg.virtual_loss > 0:
                    node.wsa[a] -= float(self.cfg.virtual_loss)
                path.append((node_id, a, reward))
                node_id = child_id
            depth += 1
        distinct_bonus = 0.0
        if self.cfg.track_distinct_terminals and self.distinct_terminals:
            distinct_bonus = 1.0 / float(len(self.distinct_terminals))
        with self.tree_lock:
            for nid, a, r in reversed(path):
                backup = r + self.cfg.gamma * backup
                node = self.nodes[nid]
                node.nsa[a] += 1
                node.wsa[a] += backup + distinct_bonus
                node.qsa[a] = node.wsa[a] / max(node.nsa[a], 1)
                node.qmaxsa[a] = max(float(node.qmaxsa[a]), float(backup))
                node.best_through = max(float(node.best_through), float(backup))
                if a < len(node.topk_returnssa):
                    vals = node.topk_returnssa[a]
                    vals.append(float(backup))
                    vals.sort(reverse=True)
                    k = max(1, int(getattr(self.cfg, "train_topk_k", 3)))
                    if len(vals) > k:
                        del vals[k:]
                if a < len(node.lse_sum_expsa):
                    beta = max(float(getattr(self.cfg, "train_lse_beta", 4.0)), 1e-6)
                    node.lse_sum_expsa[a] += float(math.exp(beta * float(backup)))
                    node.lse_countsa[a] += 1
                if self.cfg.track_distinct_terminals:
                    node.distinct_sumsa[a] += distinct_bonus
                    node.distinct_countsa[a] += 1
                node.value_sum += backup
                if self.cfg.virtual_loss > 0:
                    node.wsa[a] += float(self.cfg.virtual_loss)
            self._propagate_status_bottom_up(visited_nodes)

    def search(self, root_state: RepairState) -> RepairState:
        self.run_search(root_state)
        return self.best_state if self.best_state is not None else root_state


def _jsonish(x: Any) -> Any:
    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (list, tuple)):
        return [_jsonish(v) for v in x]
    if isinstance(x, dict):
        return {str(k): _jsonish(v) for k, v in x.items()}
    return repr(x)


def _mean_max_float(vals: List[float]) -> Optional[Dict[str, float]]:
    if not vals:
        return None
    return {"mean": float(sum(vals) / len(vals)), "max": float(max(vals)), "min": float(min(vals))}


def _mean_max_int(vals: List[int]) -> Optional[Dict[str, float]]:
    if not vals:
        return None
    return {"mean": float(sum(vals) / len(vals)), "max": float(max(vals)), "min": float(min(vals))}


def _serialize_candidate_action(c: CandidateAction) -> Dict[str, Any]:
    return {
        "selector": str(c.selector),
        "family": str(c.family),
        "args": [_jsonish(a) for a in c.args],
        "params": [_jsonish(a) for a in c.params],
        "est_delta_conflicts": float(c.est_delta_conflicts),
        "est_delta_vertices": float(c.est_delta_vertices),
        "est_cost": float(c.est_cost),
        "token": _jsonish(c.token),
        "meta": _jsonish(c.meta),
    }


def _serialize_state_metrics_full(m: StateMetrics) -> Dict[str, Any]:
    return {
        "conflicts": int(m.conflicts),
        "conflict_vertices": int(m.conflict_vertices),
        "core_size": int(m.core_size),
        "zero_slack_vertices": int(m.zero_slack_vertices),
        "one_slack_vertices": int(m.one_slack_vertices),
        "mean_legal_colors": float(m.mean_legal_colors),
        "patchable_score": float(m.patchable_score),
        "entropy": float(m.entropy),
        "class_sizes": m.class_sizes.tolist(),
        "class_conflicts": m.class_conflicts.tolist(),
        "local_conflicts": m.local_conflicts.tolist(),
        "conflict_mask": m.conflict_mask.astype(int).tolist(),
        "core_mask": m.core_mask.astype(int).tolist(),
        "same_color_neighbors": m.same_color_neighbors.tolist(),
        "distinct_neighbor_colors": m.distinct_neighbor_colors.tolist(),
        "legal_color_counts": m.legal_color_counts.tolist(),
    }


def serialize_gcp_mcts_tree(mcts: GCPMCTS, graph: GCGraph, k: int) -> Dict[str, Any]:
    nodes_out: List[Dict[str, Any]] = []
    for nid, node in enumerate(mcts.nodes):
        n_cand = len(node.candidates)
        branches: List[Dict[str, Any]] = []
        for a in range(n_cand):
            child_id = int(node.children[a]) if a in node.children else -1
            branches.append(
                {
                    "action_index": int(a),
                    "edge_visits_n": int(node.nsa[a]) if a < len(node.nsa) else 0,
                    "edge_value_sum_w": float(node.wsa[a]) if a < len(node.wsa) else 0.0,
                    "edge_mean_q": float(node.qsa[a]) if a < len(node.qsa) else 0.0,
                    "edge_max_q": float(node.qmaxsa[a]) if a < len(node.qmaxsa) else 0.0,
                    "edge_q_distinct": float(node.distinct_sumsa[a] / max(node.distinct_countsa[a], 1)) if a < len(node.distinct_sumsa) and a < len(node.distinct_countsa) and int(node.distinct_countsa[a]) > 0 else 0.0,
                    "prior": float(node.priors[a]) if a < len(node.priors) else 0.0,
                    "alive": bool(node.alive[a]) if a < len(node.alive) else False,
                    "child_node_id": child_id,
                    "best_through": float(mcts.nodes[child_id].best_through) if child_id >= 0 and child_id < len(mcts.nodes) else 0.0,
                    "child_solved": bool(mcts.nodes[child_id].solved) if child_id >= 0 and child_id < len(mcts.nodes) else False,
                    "candidate": _serialize_candidate_action(node.candidates[a]) if a < len(node.candidates) else {},
                }
            )
        alive_idx = [a for a in range(n_cand) if a < len(node.alive) and bool(node.alive[a])]
        visited_idx = [a for a in alive_idx if a < len(node.nsa) and int(node.nsa[a]) > 0]
        q_alive = [float(node.qsa[a]) for a in alive_idx if a < len(node.qsa)]
        q_vis = [float(node.qsa[a]) for a in visited_idx if a < len(node.qsa)]
        n_alive = [int(node.nsa[a]) for a in alive_idx if a < len(node.nsa)]
        n_vis = [int(node.nsa[a]) for a in visited_idx if a < len(node.nsa)]
        branch_stats = {
            "per_action_q_mean_max": {
                "among_alive_actions": _mean_max_float(q_alive),
                "among_visited_actions": _mean_max_float(q_vis),
            },
            "per_action_n_mean_max": {
                "among_alive_actions": _mean_max_int(n_alive),
                "among_visited_actions": _mean_max_int(n_vis),
            },
        }
        metrics_blob: Optional[Dict[str, Any]] = None
        if node.metrics is not None:
            metrics_blob = _serialize_state_metrics_full(node.metrics)
        nodes_out.append(
            {
                "node_id": int(nid),
                "parent": int(node.parent),
                "parent_action": int(node.parent_action),
                "reward_from_parent": float(node.reward_from_parent),
                "visit_count_V": int(node.visit),
                "value_sum_V": float(node.value_sum),
                "value_mean_V": float(0.0 if node.visit == 0 else node.value_sum / max(node.visit, 1)),
                "expanded": bool(node.expanded),
                "terminal": bool(node.terminal),
                "solved": bool(node.solved),
                "frozen": bool(node.frozen),
                "dead": bool(node.dead),
                "best_terminal_conflicts": int(node.best_terminal_conflicts),
                "best_through": float(node.best_through),
                "state": {
                    "colors": [int(x) for x in node.state.colors.tolist()],
                    "k": int(node.state.k),
                    "plateau": int(node.state.plateau),
                    "step": int(node.state.step),
                },
                "children_action_to_child_id": {str(int(a)): int(cid) for a, cid in node.children.items()},
                "branch_stats": branch_stats,
                "branches": branches,
                "metrics": metrics_blob,
            }
        )
    return {
        "format": "gcp_mcts_tree_v2",
        "graph_name": str(graph.name),
        "graph_n": int(graph.n),
        "graph_m": int(graph.m),
        "k": int(k),
        "num_nodes": int(len(mcts.nodes)),
        "nodes": nodes_out,
    }


def solve_instance(
    graph: GCGraph,
    k: Optional[int],
    model: Optional[TRMPolicyValue],
    device: torch.device,
    macros: Optional[Dict[str, MacroProgram]],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    if k is None:
        k = max(graph.clique_lb, graph.dsatur_ub)
    seed_colors = greedy_k_assignment(graph, int(k))
    state = RepairState(seed_colors, k=int(k), plateau=0, step=0)
    mcts = GCPMCTS(graph, model, device, macros, SearchConfig(
        cpuct=args.cpuct,
        gamma=args.gamma,
        simulations=args.simulations,
        max_depth=args.max_depth,
        prune_every=args.prune_every,
        prune_min_visits=args.prune_min_visits,
        prune_keep_topk=args.prune_keep_topk,
        confidence_beta=args.confidence_beta,
        action_budget=args.action_budget,
        exact_patch_limit=args.exact_patch_limit,
        profile_every=int(getattr(args, "profile_every", 0)),
        search_alpha_mean=float(getattr(args, "search_alpha_mean", 1.0)),
        search_beta_max=float(getattr(args, "search_beta_max", 0.0)),
        novelty_coef=float(getattr(args, "novelty_coef", 0.0)),
        worker_count=int(getattr(args, "worker_count", 1)),
        worker_rounds=int(getattr(args, "worker_rounds", 1)),
        virtual_loss=float(getattr(args, "virtual_loss", 0.0)),
        track_distinct_terminals=bool(getattr(args, "track_distinct_terminals", False)),
        search_mode=str(getattr(args, "search_mode", "infer")),
        collect_prior_mix=float(getattr(args, "collect_prior_mix", 0.30)),
        collect_prior_temp=float(getattr(args, "collect_prior_temp", 1.8)),
        infer_prior_mix=float(getattr(args, "infer_prior_mix", 0.0)),
        infer_prior_temp=float(getattr(args, "infer_prior_temp", 1.0)),
        alloc_lambda_best=float(getattr(args, "alloc_lambda_best", 1.0)),
        alloc_lambda_uncertainty=float(getattr(args, "alloc_lambda_uncertainty", 0.5)),
        alloc_lambda_novelty=float(getattr(args, "alloc_lambda_novelty", 0.25)),
        train_policy_target_mode=str(getattr(args, "train_policy_target_mode", "best_through")),
        train_topk_k=int(getattr(args, "train_topk_k", 3)),
        train_lse_beta=float(getattr(args, "train_lse_beta", 4.0)),
    ))
    best = mcts.search(state)
    metrics = compute_state_metrics(graph, best)
    result = {
        "name": graph.name,
        "k": int(k),
        "colors": best.colors.tolist(),
        "conflicts": metrics.conflicts,
        "conflict_vertices": metrics.conflict_vertices,
        "core_size": metrics.core_size,
        "solved": bool(metrics.conflicts == 0),
        "primitive_calls": int(best.step),
        "anytime_trace": mcts.anytime_trace,
    }
    profile_out = str(getattr(args, "profile_out", "") or "")
    if profile_out:
        with open(profile_out, "w") as f:
            json.dump(
                {
                    "name": graph.name,
                    "k": int(k),
                    "solved": bool(metrics.conflicts == 0),
                    "final_conflicts": int(metrics.conflicts),
                    "primitive_calls": int(best.step),
                    "anytime_trace": mcts.anytime_trace,
                },
                f,
                indent=2,
            )
    tree_dump = str(getattr(args, "mcts_tree_dump", "") or "").strip()
    if tree_dump:
        tree_path = Path(tree_dump)
        tree_path.parent.mkdir(parents=True, exist_ok=True)
        tree_payload = serialize_gcp_mcts_tree(mcts, graph, int(k))
        with tree_path.open("w", encoding="utf-8") as tf:
            json.dump(tree_payload, tf, indent=2, default=str)
        result["mcts_tree_dump"] = str(tree_path.resolve())
    return result


def mine_macros_from_samples(samples: Iterable[TraceSample], min_support: int, max_len: int, top_k: int) -> List[MacroProgram]:
    episodes: Dict[str, List[str]] = collections.defaultdict(list)
    for s in samples:
        episodes[s.episode_id].append(s.chosen_family)
    counter: collections.Counter[Tuple[str, ...]] = collections.Counter()
    for seq in episodes.values():
        L = len(seq)
        for n in range(2, max_len + 1):
            for i in range(L - n + 1):
                counter[tuple(seq[i : i + n])] += 1
    items: List[Tuple[Tuple[str, ...], int, float]] = []
    for ng, support in counter.items():
        if support < min_support:
            continue
        score = float((len(ng) - 1) * support)
        items.append((ng, support, score))
    items.sort(key=lambda x: x[2], reverse=True)
    macros: List[MacroProgram] = []
    used = set()
    for ng, support, score in items:
        if ng in used:
            continue
        macros.append(MacroProgram(name=f"macro_{len(macros):04d}", families=tuple(ng), support=int(support), score=float(score)))
        used.add(ng)
        if len(macros) >= top_k:
            break
    return macros


def load_trace_samples_from_pattern(pattern: str) -> List[TraceSample]:
    files = sorted(glob.glob(pattern))
    out: List[TraceSample] = []
    for path in files:
        shard = torch.load(path, map_location="cpu", weights_only=False)
        out.extend(TraceSample(**item) for item in shard)
    return out


def command_build_traces(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    macros: Dict[str, MacroProgram] = {}
    if args.macros:
        macros = load_macros(args.macros)
    records = load_records(args.input)
    writer = TraceShardWriter(args.out_dir, prefix=args.prefix, samples_per_shard=args.samples_per_shard)
    total = 0
    for ridx, rec in enumerate(records):
        graph = rec.to_runtime()
        if graph.solution is None:
            if args.skip_unsolved:
                continue
            raise ValueError(f"{graph.name} has no solution; traces-from-solutions mode requires solution labels")
        samples = build_teacher_traces_for_record(
            graph,
            corruptions_per_graph=args.corruptions_per_graph,
            max_steps=args.max_steps,
            macros=macros,
            vertex_budget=args.vertex_budget,
            class_budget=args.class_budget,
            action_budget=args.action_budget,
            seed=args.seed + ridx,
            exact_patch_limit=args.exact_patch_limit,
        )
        for s in samples:
            writer.add(s)
        total += len(samples)
        if (ridx + 1) % max(args.log_every_records, 1) == 0:
            print(f"[{ridx + 1}/{len(records)}] wrote {total} samples", flush=True)
    writer.flush()
    print(f"done: wrote {total} samples to {args.out_dir}", flush=True)


def command_build_solve_traces(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = None
    if args.ckpt:
        model, _ = load_model_checkpoint(args.ckpt, device)
    macros: Dict[str, MacroProgram] = load_macros(args.macros) if args.macros else {}
    cfg = SearchConfig(
        cpuct=args.cpuct,
        gamma=args.gamma,
        simulations=args.simulations,
        max_depth=args.max_depth,
        prune_every=args.prune_every,
        prune_min_visits=args.prune_min_visits,
        prune_keep_topk=args.prune_keep_topk,
        confidence_beta=args.confidence_beta,
        action_budget=args.action_budget,
        exact_patch_limit=args.exact_patch_limit,
        profile_every=0,
        search_alpha_mean=args.search_alpha_mean,
        search_beta_max=args.search_beta_max,
        novelty_coef=args.novelty_coef,
        worker_count=args.worker_count,
        worker_rounds=max(1, args.worker_rounds),
        virtual_loss=args.virtual_loss,
        track_distinct_terminals=args.track_distinct_terminals,
        search_mode=str(args.search_mode),
        collect_prior_mix=float(args.collect_prior_mix),
        collect_prior_temp=float(args.collect_prior_temp),
        infer_prior_mix=float(args.infer_prior_mix),
        infer_prior_temp=float(args.infer_prior_temp),
        alloc_lambda_best=float(args.alloc_lambda_best),
        alloc_lambda_uncertainty=float(args.alloc_lambda_uncertainty),
        alloc_lambda_novelty=float(args.alloc_lambda_novelty),
        train_policy_target_mode=str(args.train_policy_target_mode),
        train_topk_k=int(args.train_topk_k),
        train_lse_beta=float(args.train_lse_beta),
    )
    records = load_records(args.input)
    writer = TraceShardWriter(args.out_dir, prefix=args.prefix, samples_per_shard=args.samples_per_shard)
    tree_dump_base = Path(str(args.mcts_tree_dump_dir)).expanduser() if str(getattr(args, "mcts_tree_dump_dir", "")).strip() else None
    total = 0
    for ridx, rec in enumerate(records):
        graph = rec.to_runtime()
        record_tree_dir = (tree_dump_base / graph.name) if tree_dump_base is not None else None
        samples = build_solve_traces_for_record(
            graph,
            episodes_per_graph=args.episodes_per_graph,
            max_steps=args.max_steps,
            model=model,
            device=device,
            macros=macros,
            cfg=cfg,
            vertex_budget=args.vertex_budget,
            class_budget=args.class_budget,
            action_budget=args.action_budget,
            k_policy=args.k_policy,
            explicit_k=args.k,
            k_offset=args.k_offset,
            seed=args.seed + ridx,
            tree_dump_dir=record_tree_dir,
        )
        for s in samples:
            writer.add(s)
        total += len(samples)
        if (ridx + 1) % max(args.log_every_records, 1) == 0:
            print(f"[{ridx + 1}/{len(records)}] wrote {total} solve-trace samples", flush=True)
    writer.flush()
    print(f"done: wrote {total} solve-trace samples to {args.out_dir}", flush=True)


def command_train(args: argparse.Namespace) -> None:
    train_loop(args)


def command_mine_macros(args: argparse.Namespace) -> None:
    samples = load_trace_samples_from_pattern(args.trace)
    macros = mine_macros_from_samples(samples, min_support=args.min_support, max_len=args.max_len, top_k=args.top_k)
    payload = [dataclasses.asdict(m) for m in macros]
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {len(macros)} macros to {args.out}", flush=True)
    if stitch_core is not None:
        print("note: stitch-core is installed, but this single-file implementation uses a stable frequent-ngram macro miner instead of depending on stitch-core's evolving API.", flush=True)


def load_macros(path: str) -> Dict[str, MacroProgram]:
    with open(path) as f:
        payload = json.load(f)
    out: Dict[str, MacroProgram] = {}
    for item in payload:
        m = MacroProgram(
            name=str(item["name"]),
            families=tuple(item["families"]),
            support=int(item.get("support", 0)),
            score=float(item.get("score", 0.0)),
            min_conflict_ratio=float(item.get("min_conflict_ratio", 0.0)),
            max_conflict_ratio=float(item.get("max_conflict_ratio", 1.0)),
            min_core_ratio=float(item.get("min_core_ratio", 0.0)),
            max_core_ratio=float(item.get("max_core_ratio", 1.0)),
        )
        out[m.name] = m
    return out


def command_solve(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    graph = load_records(args.input)[0].to_runtime()
    model = None
    if args.ckpt:
        model, _ = load_model_checkpoint(args.ckpt, device)
    macros = load_macros(args.macros) if args.macros else {}
    result = solve_instance(graph, k=args.k, model=model, device=device, macros=macros, args=args)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2), flush=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="GCP traces + TRM prior + MCTS single-file trainer for abstractrl/AbstractBeam")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build-traces", help="build dense-reward repair traces from solved examples")
    b.add_argument("--input", required=True, help="graph dataset: directory, jsonl/json/pkl/pt/col")
    b.add_argument("--out-dir", required=True)
    b.add_argument("--prefix", default="gcp-trace")
    b.add_argument("--samples-per-shard", type=int, default=5000)
    b.add_argument("--corruptions-per-graph", type=int, default=8)
    b.add_argument("--max-steps", type=int, default=32)
    b.add_argument("--vertex-budget", type=int, default=DEFAULT_VERTEX_BUDGET)
    b.add_argument("--class-budget", type=int, default=DEFAULT_CLASS_BUDGET)
    b.add_argument("--action-budget", type=int, default=DEFAULT_ACTION_BUDGET)
    b.add_argument("--exact-patch-limit", type=int, default=12)
    b.add_argument("--macros", default="")
    b.add_argument("--skip-unsolved", action="store_true")
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--log-every-records", type=int, default=25)
    b.set_defaults(func=command_build_traces)

    bs = sub.add_parser("build-solve-traces", help="build solve traces from solver-generated root policies")
    bs.add_argument("--input", required=True, help="graph dataset: directory, jsonl/json/pkl/pt/col")
    bs.add_argument("--out-dir", required=True)
    bs.add_argument("--prefix", default="gcp-solve-trace")
    bs.add_argument("--samples-per-shard", type=int, default=5000)
    bs.add_argument("--episodes-per-graph", type=int, default=2)
    bs.add_argument("--max-steps", type=int, default=24)
    bs.add_argument("--vertex-budget", type=int, default=DEFAULT_VERTEX_BUDGET)
    bs.add_argument("--class-budget", type=int, default=DEFAULT_CLASS_BUDGET)
    bs.add_argument("--action-budget", type=int, default=DEFAULT_ACTION_BUDGET)
    bs.add_argument("--exact-patch-limit", type=int, default=12)
    bs.add_argument("--macros", default="")
    bs.add_argument("--ckpt", default="")
    bs.add_argument("--k", type=int, default=None)
    bs.add_argument("--k-policy", choices=["solution","solution_minus_one","dsatur","dsatur_minus_one"], default="dsatur_minus_one")
    bs.add_argument("--k-offset", type=int, default=1)
    bs.add_argument("--cpu", action="store_true")
    bs.add_argument("--cpuct", type=float, default=1.25)
    bs.add_argument("--gamma", type=float, default=1.0)
    bs.add_argument("--simulations", type=int, default=128)
    bs.add_argument("--max-depth", type=int, default=48)
    bs.add_argument("--prune-every", type=int, default=32)
    bs.add_argument("--prune-min-visits", type=int, default=4)
    bs.add_argument("--prune-keep-topk", type=int, default=4)
    bs.add_argument("--confidence-beta", type=float, default=1.5)
    bs.add_argument("--search-alpha-mean", type=float, default=0.8)
    bs.add_argument("--search-beta-max", type=float, default=0.2)
    bs.add_argument("--novelty-coef", type=float, default=0.05)
    bs.add_argument("--worker-count", type=int, default=max(1, os.cpu_count() // 2 if os.cpu_count() else 2))
    bs.add_argument("--worker-rounds", type=int, default=1)
    bs.add_argument("--virtual-loss", type=float, default=0.25)
    bs.add_argument("--track-distinct-terminals", action="store_true")
    bs.add_argument("--search-mode", choices=["collect", "infer", "noprior"], default="collect")
    bs.add_argument("--collect-prior-mix", type=float, default=0.30)
    bs.add_argument("--collect-prior-temp", type=float, default=1.8)
    bs.add_argument("--infer-prior-mix", type=float, default=0.0)
    bs.add_argument("--infer-prior-temp", type=float, default=1.0)
    bs.add_argument("--alloc-lambda-best", type=float, default=1.0)
    bs.add_argument("--alloc-lambda-uncertainty", type=float, default=0.5)
    bs.add_argument("--alloc-lambda-novelty", type=float, default=0.25)
    bs.add_argument("--train-policy-target-mode", choices=["qmax", "best_through", "topk_mean", "logsumexp"], default="best_through")
    bs.add_argument("--train-topk-k", type=int, default=3)
    bs.add_argument("--train-lse-beta", type=float, default=4.0)
    bs.add_argument("--seed", type=int, default=0)
    bs.add_argument("--log-every-records", type=int, default=25)
    bs.add_argument("--mcts-tree-dump-dir", default="", help="if set, dump one MCTS tree JSON per solve step during solve-trace collection")
    bs.set_defaults(func=command_build_solve_traces)

    t = sub.add_parser("train", help="ddp training over trace shards")
    t.add_argument("--train", required=True, help="glob pattern for train shards, e.g. data/train/gcp-trace-*.pt")
    t.add_argument("--valid", default="", help="glob pattern for valid shards")
    t.add_argument("--out-dir", required=True)
    t.add_argument("--epochs", type=int, default=20)
    t.add_argument("--batch-size", type=int, default=64)
    t.add_argument("--num-workers", type=int, default=4)
    t.add_argument("--steps-per-epoch", type=int, default=2000)
    t.add_argument("--valid-steps", type=int, default=250)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--weight-decay", type=float, default=1e-4)
    t.add_argument("--grad-accum", type=int, default=1)
    t.add_argument("--grad-clip", type=float, default=1.0)
    t.add_argument("--d-model", type=int, default=256)
    t.add_argument("--refine-steps", type=int, default=3)
    t.add_argument("--dropout", type=float, default=0.1)
    t.add_argument("--value-coef", type=float, default=0.5)
    t.add_argument("--entropy-coef", type=float, default=0.01)
    t.add_argument("--family-coef", type=float, default=0.1)
    t.add_argument("--amp", action="store_true")
    t.add_argument("--compile", action="store_true")
    t.add_argument("--backend", default="")
    t.add_argument("--init-ckpt", default="")
    t.add_argument("--resume-optimizer", action="store_true")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--log-every", type=int, default=100)
    t.set_defaults(func=command_train)

    m = sub.add_parser("mine-macros", help="mine frequent primitive-family macros from traces")
    m.add_argument("--trace", required=True, help="glob pattern over trace shards")
    m.add_argument("--out", required=True)
    m.add_argument("--min-support", type=int, default=8)
    m.add_argument("--max-len", type=int, default=4)
    m.add_argument("--top-k", type=int, default=64)
    m.set_defaults(func=command_mine_macros)

    s = sub.add_parser("solve", help="solve one graph with MCTS + trained prior")
    s.add_argument("--input", required=True)
    s.add_argument("--ckpt", default="")
    s.add_argument("--macros", default="")
    s.add_argument("--out", default="")
    s.add_argument("--k", type=int, default=None)
    s.add_argument("--cpu", action="store_true")
    s.add_argument("--cpuct", type=float, default=1.25)
    s.add_argument("--gamma", type=float, default=1.0)
    s.add_argument("--simulations", type=int, default=256)
    s.add_argument("--max-depth", type=int, default=48)
    s.add_argument("--prune-every", type=int, default=32)
    s.add_argument("--prune-min-visits", type=int, default=4)
    s.add_argument("--prune-keep-topk", type=int, default=4)
    s.add_argument("--confidence-beta", type=float, default=1.5)
    s.add_argument("--search-alpha-mean", type=float, default=0.8)
    s.add_argument("--search-beta-max", type=float, default=0.2)
    s.add_argument("--novelty-coef", type=float, default=0.05)
    s.add_argument("--worker-count", type=int, default=max(1, os.cpu_count() // 2 if os.cpu_count() else 2))
    s.add_argument("--worker-rounds", type=int, default=1)
    s.add_argument("--virtual-loss", type=float, default=0.25)
    s.add_argument("--track-distinct-terminals", action="store_true")
    s.add_argument("--search-mode", choices=["collect", "infer", "noprior"], default="infer")
    s.add_argument("--collect-prior-mix", type=float, default=0.30)
    s.add_argument("--collect-prior-temp", type=float, default=1.8)
    s.add_argument("--infer-prior-mix", type=float, default=0.0)
    s.add_argument("--infer-prior-temp", type=float, default=1.0)
    s.add_argument("--alloc-lambda-best", type=float, default=1.0)
    s.add_argument("--alloc-lambda-uncertainty", type=float, default=0.5)
    s.add_argument("--alloc-lambda-novelty", type=float, default=0.25)
    s.add_argument("--train-policy-target-mode", choices=["qmax", "best_through", "topk_mean", "logsumexp"], default="best_through")
    s.add_argument("--train-topk-k", type=int, default=3)
    s.add_argument("--train-lse-beta", type=float, default=4.0)
    s.add_argument("--action-budget", type=int, default=DEFAULT_ACTION_BUDGET)
    s.add_argument("--exact-patch-limit", type=int, default=12)
    s.add_argument("--profile-every", type=int, default=8)
    s.add_argument("--profile-out", default="")
    s.add_argument(
        "--mcts-tree-dump",
        default="",
        help="if set, write the full MCTS search tree (nodes, states, per-action N/W/Q, branch mean/max) as JSON after search",
    )
    s.set_defaults(func=command_solve)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
