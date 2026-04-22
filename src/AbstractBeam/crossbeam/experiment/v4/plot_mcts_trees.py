#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def _expand_inputs(inputs: Sequence[str]) -> List[Path]:
    paths: List[Path] = []
    for raw in inputs:
        p = Path(raw).expanduser()
        if p.is_file():
            paths.append(p)
            continue
        if p.is_dir():
            paths.extend(sorted(p.rglob("*.json")))
            continue
        paths.extend(sorted(Path(".").glob(raw)))
    out: List[Path] = []
    seen: set[Path] = set()
    for p in paths:
        rp = p.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        out.append(rp)
    return out


def _is_mcts_tree_payload(payload: Dict[str, Any]) -> bool:
    return (
        isinstance(payload, dict)
        and str(payload.get("format", "")).startswith("gcp_mcts_tree_")
        and isinstance(payload.get("nodes"), list)
    )


def _load_tree(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text())
    if not _is_mcts_tree_payload(payload):
        raise ValueError(f"{path} is not a gcp_mcts_tree payload")
    return payload


def _compute_depths(nodes: Sequence[Dict[str, Any]]) -> List[int]:
    n = len(nodes)
    parent = [int(node.get("parent", -1)) for node in nodes]
    depth = [-1] * n
    for i in range(n):
        if depth[i] >= 0:
            continue
        cur = i
        stack: List[int] = []
        while cur >= 0 and cur < n and depth[cur] < 0:
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


def _safe_mean(values: Iterable[float]) -> Optional[float]:
    vals = [float(v) for v in values]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _safe_max(values: Iterable[float]) -> Optional[float]:
    vals = [float(v) for v in values]
    if not vals:
        return None
    return float(max(vals))


def _plot_depth_value(nodes: Sequence[Dict[str, Any]], depths: Sequence[int], out_path: Path, title: str) -> None:
    xs = list(depths)
    ys = [float(node.get("value_mean_V", 0.0)) for node in nodes]
    cs = ["tab:green" if bool(node.get("terminal", False)) else "tab:blue" for node in nodes]
    plt.figure(figsize=(9, 5))
    plt.scatter(xs, ys, s=12, c=cs, alpha=0.7)
    plt.xlabel("Node depth")
    plt.ylabel("Node value_mean_V")
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def _plot_visits_by_depth(nodes: Sequence[Dict[str, Any]], depths: Sequence[int], out_path: Path, title: str) -> None:
    bucket: Dict[int, List[int]] = {}
    for node, d in zip(nodes, depths):
        bucket.setdefault(int(d), []).append(int(node.get("visit_count_V", 0)))
    ds = sorted(bucket)
    mean_v = [_safe_mean(bucket[d]) or 0.0 for d in ds]
    max_v = [_safe_max(bucket[d]) or 0.0 for d in ds]
    plt.figure(figsize=(9, 5))
    plt.plot(ds, mean_v, marker="o", linewidth=1.8, label="mean visit_count_V")
    plt.plot(ds, max_v, marker="x", linewidth=1.5, label="max visit_count_V")
    plt.xlabel("Node depth")
    plt.ylabel("Visits")
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def _collect_edge_stats(nodes: Sequence[Dict[str, Any]], depths: Sequence[int]) -> Tuple[List[int], List[float], List[float]]:
    dvals: List[int] = []
    qvals: List[float] = []
    nvals: List[float] = []
    for node, d in zip(nodes, depths):
        branches = node.get("branches", [])
        if not isinstance(branches, list):
            continue
        for br in branches:
            if not isinstance(br, dict):
                continue
            dvals.append(int(d))
            qvals.append(float(br.get("edge_mean_q", 0.0)))
            nvals.append(float(br.get("edge_visits_n", 0.0)))
    return dvals, qvals, nvals


def _plot_edge_q_by_depth(nodes: Sequence[Dict[str, Any]], depths: Sequence[int], out_path: Path, title: str) -> None:
    dvals, qvals, _ = _collect_edge_stats(nodes, depths)
    bucket: Dict[int, List[float]] = {}
    for d, q in zip(dvals, qvals):
        bucket.setdefault(int(d), []).append(float(q))
    ds = sorted(bucket)
    mean_q = [_safe_mean(bucket[d]) or 0.0 for d in ds]
    max_q = [_safe_max(bucket[d]) or 0.0 for d in ds]
    plt.figure(figsize=(9, 5))
    plt.plot(ds, mean_q, marker="o", linewidth=1.8, label="mean edge_mean_q")
    plt.plot(ds, max_q, marker="x", linewidth=1.5, label="max edge_mean_q")
    plt.xlabel("Parent depth")
    plt.ylabel("Q")
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def _plot_edge_visits_hist(nodes: Sequence[Dict[str, Any]], depths: Sequence[int], out_path: Path, title: str) -> None:
    _, _, nvals = _collect_edge_stats(nodes, depths)
    nz = [v for v in nvals if v > 0]
    plt.figure(figsize=(9, 5))
    if nz:
        uniq = sorted(set(nz))
        near_int = all(abs(v - round(v)) < 1e-9 for v in nz)
        if near_int and len(uniq) <= 12:
            ctr = Counter(int(round(v)) for v in nz)
            xs = sorted(ctr)
            ys = [ctr[x] for x in xs]
            plt.bar(xs, ys, width=0.8, color="tab:purple", alpha=0.85)
            plt.xticks(xs)
            plt.xlim(min(xs) - 0.8, max(xs) + 0.8)
        else:
            vmin = min(nz)
            vmax = max(nz)
            bins = min(60, max(10, int(math.sqrt(len(nz)))))
            plt.hist(nz, bins=bins, color="tab:purple", alpha=0.8)
            if abs(vmax - vmin) < 1e-12:
                plt.xlim(vmin - 0.5, vmax + 0.5)
            else:
                pad = 0.02 * (vmax - vmin)
                plt.xlim(vmin - pad, vmax + pad)
    plt.xlabel("edge_visits_n (non-zero)")
    plt.ylabel("count")
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def _tree_layout(nodes: Sequence[Dict[str, Any]], depths: Sequence[int]) -> Tuple[List[float], List[float], List[Tuple[int, int]]]:
    n = len(nodes)
    children: List[List[int]] = [[] for _ in range(n)]
    for i, node in enumerate(nodes):
        p = int(node.get("parent", -1))
        if 0 <= p < n and p != i:
            children[p].append(i)
    for i in range(n):
        children[i].sort()

    x = [0.0] * n
    y = [0.0] * n
    next_x = 0

    def dfs(u: int) -> None:
        nonlocal next_x
        if not children[u]:
            x[u] = float(next_x)
            next_x += 1
        else:
            for v in children[u]:
                dfs(v)
            lo = x[children[u][0]]
            hi = x[children[u][-1]]
            x[u] = 0.5 * (lo + hi)
        y[u] = -float(depths[u])

    roots = [i for i, node in enumerate(nodes) if int(node.get("parent", -1)) < 0]
    if not roots and n > 0:
        roots = [0]
    for r in roots:
        dfs(r)
        next_x += 1

    edges: List[Tuple[int, int]] = []
    for v, node in enumerate(nodes):
        p = int(node.get("parent", -1))
        if 0 <= p < n and p != v:
            edges.append((p, v))
    return x, y, edges


def _subsample_nodes(nodes: Sequence[Dict[str, Any]], max_nodes: int, seed: int = 0) -> List[int]:
    n = len(nodes)
    if n <= max_nodes:
        return list(range(n))
    root = 0
    keep: set[int] = {root}
    by_visit = sorted(range(n), key=lambda i: float(nodes[i].get("visit_count_V", 0)), reverse=True)
    for i in by_visit:
        if len(keep) >= max_nodes:
            break
        keep.add(i)
    if len(keep) < max_nodes:
        rng = random.Random(seed)
        rest = [i for i in range(n) if i not in keep]
        rng.shuffle(rest)
        for i in rest:
            if len(keep) >= max_nodes:
                break
            keep.add(i)
    return sorted(keep)


def _plot_tree_structure(
    nodes: Sequence[Dict[str, Any]],
    depths: Sequence[int],
    out_path: Path,
    title: str,
    max_nodes_draw: int = 1200,
) -> None:
    keep_idx = _subsample_nodes(nodes, max_nodes_draw, seed=7)
    idx_map = {old: new for new, old in enumerate(keep_idx)}
    sub_nodes = [nodes[i] for i in keep_idx]
    sub_depths = [depths[i] for i in keep_idx]
    x, y, edges = _tree_layout(sub_nodes, sub_depths)

    visit = [float(n.get("visit_count_V", 0)) for n in sub_nodes]
    vmax = max(visit) if visit else 1.0
    size = [10.0 + 40.0 * (v / max(vmax, 1.0)) for v in visit]
    color = [float(n.get("value_mean_V", 0.0)) for n in sub_nodes]

    plt.figure(figsize=(14, 8))
    for u, v in edges:
        plt.plot([x[u], x[v]], [y[u], y[v]], color="gray", alpha=0.20, linewidth=0.6)
    sc = plt.scatter(x, y, s=size, c=color, cmap="viridis", alpha=0.9, linewidths=0.0)
    cbar = plt.colorbar(sc)
    cbar.set_label("value_mean_V")
    plt.xlabel("Tree horizontal layout")
    plt.ylabel("Depth (negative)")
    plt.title(title)
    plt.grid(alpha=0.15)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def _plot_tree_animation(
    nodes: Sequence[Dict[str, Any]],
    depths: Sequence[int],
    out_path: Path,
    title: str,
    max_nodes_draw: int = 700,
    frames: int = 60,
) -> Optional[str]:
    keep_idx = _subsample_nodes(nodes, max_nodes_draw, seed=13)
    sub_nodes = [nodes[i] for i in keep_idx]
    sub_depths = [depths[i] for i in keep_idx]
    x, y, edges = _tree_layout(sub_nodes, sub_depths)
    visit = [float(n.get("visit_count_V", 0)) for n in sub_nodes]
    vmax = max(visit) if visit else 1.0
    normalized = [v / max(vmax, 1.0) for v in visit]
    order = sorted(range(len(sub_nodes)), key=lambda i: visit[i], reverse=True)

    fig, ax = plt.subplots(figsize=(12, 7))
    for u, v in edges:
        ax.plot([x[u], x[v]], [y[u], y[v]], color="gray", alpha=0.15, linewidth=0.6)
    ax.set_title(title)
    ax.set_xlabel("Tree horizontal layout")
    ax.set_ylabel("Depth (negative)")
    ax.grid(alpha=0.12)

    scat = ax.scatter([], [], s=[], c=[], cmap="plasma", vmin=0.0, vmax=1.0, alpha=0.92)

    def _frame(k: int) -> Any:
        cut = max(1, int((k + 1) * len(order) / max(frames, 1)))
        idx = order[:cut]
        xx = [x[i] for i in idx]
        yy = [y[i] for i in idx]
        ss = [12.0 + 35.0 * normalized[i] for i in idx]
        cc = [normalized[i] for i in idx]
        scat.set_offsets(list(zip(xx, yy)))
        scat.set_sizes(ss)
        scat.set_array(cc)
        ax.set_title(f"{title} | revealed nodes: {cut}/{len(order)}")
        return scat

    ani = FuncAnimation(fig, _frame, frames=max(frames, 2), interval=120, blit=False)
    try:
        ani.save(str(out_path), writer="pillow", fps=8)
        plt.close(fig)
        return None
    except Exception as exc:
        plt.close(fig)
        return f"{type(exc).__name__}: {exc}"


def _slug(s: str) -> str:
    keep = []
    for ch in s:
        if ch.isalnum() or ch in ("-", "_", "."):
            keep.append(ch)
        else:
            keep.append("_")
    out = "".join(keep).strip("_")
    return out or "tree"


def _plot_single_tree(path: Path, out_dir: Path) -> Dict[str, Any]:
    payload = _load_tree(path)
    nodes = payload["nodes"]
    depths = _compute_depths(nodes)
    graph_name = str(payload.get("graph_name", path.stem))
    method_name = path.parent.name
    budget_name = path.parent.parent.name if path.parent.parent else "budget"
    stem = _slug(f"{graph_name}__{budget_name}__{method_name}")
    target = out_dir / stem
    target.mkdir(parents=True, exist_ok=True)

    _plot_depth_value(
        nodes,
        depths,
        target / "depth_vs_value_mean.png",
        f"{graph_name} | {budget_name} | {method_name} | value_mean_V vs depth",
    )
    _plot_visits_by_depth(
        nodes,
        depths,
        target / "depth_vs_visits.png",
        f"{graph_name} | {budget_name} | {method_name} | visits by depth",
    )
    _plot_edge_q_by_depth(
        nodes,
        depths,
        target / "depth_vs_edge_q.png",
        f"{graph_name} | {budget_name} | {method_name} | edge Q by depth",
    )
    _plot_edge_visits_hist(
        nodes,
        depths,
        target / "edge_visits_hist.png",
        f"{graph_name} | {budget_name} | {method_name} | edge visit histogram",
    )
    _plot_tree_structure(
        nodes,
        depths,
        target / "tree_structure.png",
        f"{graph_name} | {budget_name} | {method_name} | tree structure",
    )
    anim_error = _plot_tree_animation(
        nodes,
        depths,
        target / "tree_exploration.gif",
        f"{graph_name} | {budget_name} | {method_name}",
    )

    summary = {
        "source": str(path),
        "out_dir": str(target),
        "graph_name": graph_name,
        "budget": budget_name,
        "method": method_name,
        "num_nodes": int(payload.get("num_nodes", len(nodes))),
        "max_depth": int(max(depths) if depths else 0),
        "tree_animation_error": anim_error,
    }
    (target / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="General plotter for dumped MCTS tree JSON files")
    ap.add_argument(
        "--inputs",
        type=str,
        required=True,
        help="Comma-separated files/dirs/globs of MCTS tree JSON (e.g. runs/.../mcts_trees or *.json)",
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default="mcts_tree_plots",
        help="Directory where plots are written",
    )
    args = ap.parse_args()

    inputs = [x.strip() for x in str(args.inputs).split(",") if x.strip()]
    files = _expand_inputs(inputs)
    trees: List[Path] = []
    for p in files:
        try:
            payload = json.loads(p.read_text())
        except Exception:
            continue
        if _is_mcts_tree_payload(payload):
            trees.append(p)

    out_dir = Path(str(args.out_dir)).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    reports: List[Dict[str, Any]] = []
    for p in trees:
        try:
            reports.append(_plot_single_tree(p, out_dir))
        except Exception as exc:
            reports.append({"source": str(p), "error": f"{type(exc).__name__}: {exc}"})

    global_report = {
        "n_input_candidates": len(files),
        "n_mcts_trees": len(trees),
        "reports": reports,
    }
    (out_dir / "index.json").write_text(json.dumps(global_report, indent=2))
    print(json.dumps(global_report, indent=2), flush=True)


if __name__ == "__main__":
    main()
