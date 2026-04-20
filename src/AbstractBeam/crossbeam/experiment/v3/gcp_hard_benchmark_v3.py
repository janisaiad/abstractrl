#!/usr/bin/env python3
"""
Hard GCP benchmark v2: explicit k, stronger planted generator, independent auditing.
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

import gcp_trace_abstractbeam_v3 as gcp


def gen_forced_k_graph(rng: random.Random, n: int, k: int, p_cross: float, p_intra_noise: float, name: str) -> dict[str, Any]:
    colors = [rng.randrange(k) for _ in range(n)]
    for c in range(min(k, n)):
        colors[c] = c
    rng.shuffle(colors)
    class_members: list[list[int]] = [[] for _ in range(k)]
    for i, c in enumerate(colors):
        class_members[c].append(i)
    edges: set[tuple[int, int]] = set()
    # Force a clique of size k to make chi >= k.
    reps = [members[0] for members in class_members if members]
    for i in range(len(reps)):
        for j in range(i + 1, len(reps)):
            u, v = reps[i], reps[j]
            edges.add((min(u, v), max(u, v)))
    # Cross edges preserving planted k-coloring.
    for i in range(n):
        ci = colors[i]
        for j in range(i + 1, n):
            cj = colors[j]
            if ci != cj and rng.random() < p_cross:
                edges.add((i, j))
    # Small adversarial within-class noise is omitted to preserve guaranteed k-colorability.
    return {
        "name": name,
        "n": n,
        "edges": [[u, v] for (u, v) in sorted(edges)],
        "solution": list(map(int, colors)),
        "tight_k": k,
        "generator": {"p_cross": p_cross, "p_intra_noise": p_intra_noise},
    }


def to_public_json(g: dict[str, Any]) -> dict[str, Any]:
    return {"name": g["name"], "n": g["n"], "edges": g["edges"]}


def count_conflicts(edges: list[list[int]], colors: list[int]) -> int:
    return sum(1 for u, v in edges if colors[int(u)] == colors[int(v)])


def best_greedy_k_conflicts(graph: gcp.GCGraph, k: int, rng: random.Random, num_random_orders: int) -> tuple[int, int]:
    n = graph.n
    degs = graph.degrees
    orders: list[np.ndarray] = [
        np.argsort(-degs),
        np.argsort(degs),
        np.arange(n, dtype=np.int64),
        np.arange(n - 1, -1, -1, dtype=np.int64),
    ]
    for _ in range(num_random_orders):
        lst = list(range(n))
        rng.shuffle(lst)
        orders.append(np.asarray(lst, dtype=np.int64))
    best = 10**18
    for order in orders:
        cols = gcp.greedy_k_assignment(graph, k, order=order)
        st = gcp.RepairState(cols, k=k, plateau=0, step=0)
        met = gcp.compute_state_metrics(graph, st)
        best = min(best, int(met.conflicts))
    return best, len(orders)


def sample_hard_instance(
    rng: random.Random,
    n: int,
    k: int,
    p_cross: float,
    name: str,
    max_tries: int,
    min_greedy_conflicts: int,
    num_random_orders: int,
    max_greedy_conflicts: int | None = None,
) -> dict[str, Any] | None:
    cap = 10**9 if max_greedy_conflicts is None else int(max_greedy_conflicts)
    for t in range(max_tries):
        g = gen_forced_k_graph(rng, n, k, p_cross, p_intra_noise=0.0, name=f"{name}_try{t}")
        rec = gcp.GraphRecord(name=g["name"], n=g["n"], edges=np.asarray(g["edges"], dtype=np.int64), solution=None, metadata={})
        graph = rec.to_runtime()
        best_c, _ = best_greedy_k_conflicts(graph, int(g["tight_k"]), rng, num_random_orders=num_random_orders)
        if int(min_greedy_conflicts) <= best_c <= cap:
            g["greedy_k_best_conflicts"] = best_c
            g["hard_sample_tries"] = t + 1
            return g
    return None


def run_solve(ckpt: Path | None, graph_json: Path, k: int, simulations: int, max_depth: int, cwd: Path) -> dict[str, Any]:
    try:
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
        ]
        if ckpt is not None:
            cmd.extend(["--ckpt", str(ckpt)])
        out = subprocess.check_output(cmd, cwd=str(cwd), text=True)
        return json.loads(out)
    finally:
        graph_json.unlink(missing_ok=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Hard k-GCP benchmark v2")
    p.add_argument("--ckpt", type=str, default="")
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--cwd", type=str, default=str(Path(__file__).resolve().parent))
    p.add_argument("--seed", type=int, default=20260418)
    p.add_argument("--per-size", type=int, default=6)
    p.add_argument("--max-tries", type=int, default=400)
    p.add_argument("--min-greedy-conflicts", type=int, default=1)
    p.add_argument("--max-greedy-conflicts", type=int, default=0)
    p.add_argument("--random-orders", type=int, default=6)
    p.add_argument("--sizes", type=str, default="100,200,400")
    p.add_argument("--simulations", type=int, default=96)
    p.add_argument("--max-depth", type=int, default=96)
    args = p.parse_args()

    rng = random.Random(int(args.seed))
    cwd = Path(args.cwd)
    ckpt = Path(args.ckpt) if args.ckpt else None
    sizes = [int(x.strip()) for x in str(args.sizes).split(",") if x.strip()]

    schedule: list[tuple[int, int, float]] = []
    for n in sizes:
        if n <= 120:
            schedule.append((n, 6, 0.22))
        elif n <= 250:
            schedule.append((n, 8, 0.18))
        else:
            schedule.append((n, 10, 0.14))

    report: dict[str, Any] = {
        "ckpt": str(ckpt) if ckpt else None,
        "schedule": [{"n": n, "k": k, "p_cross": p} for n, k, p in schedule],
        "per_size": int(args.per_size),
        "blocks": [],
    }

    for n, k, p_cross in schedule:
        rows: list[dict[str, Any]] = []
        solved = 0
        t0 = time.time()
        for i in range(int(args.per_size)):
            hid = sample_hard_instance(
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
                rows.append({"error": "no_hard_instance_found", "n": n, "k": k, "p_cross": p_cross})
                continue
            pub = to_public_json(hid)
            with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
                json.dump(pub, f)
                path = Path(f.name)
            pred = run_solve(ckpt, path, int(hid["tight_k"]), int(args.simulations), int(args.max_depth), cwd)
            conf = count_conflicts(hid["edges"], pred["colors"])
            ok = conf == 0
            solved += int(ok)
            rec = gcp.GraphRecord(name=hid["name"], n=hid["n"], edges=np.asarray(hid["edges"], dtype=np.int64), solution=None, metadata={})
            graph = rec.to_runtime()
            best_g, _ = best_greedy_k_conflicts(graph, int(hid["tight_k"]), rng, num_random_orders=int(args.random_orders))
            rows.append(
                {
                    "instance_id": f"{hid['name']}|n={n}|k={int(hid['tight_k'])}|seed={int(args.seed)}|idx={i}",
                    "name": hid["name"],
                    "n": n,
                    "tight_k": int(hid["tight_k"]),
                    "m": len(hid["edges"]),
                    "greedy_k_best_conflicts": int(hid.get("greedy_k_best_conflicts", best_g)),
                    "baseline_greedy_k_conflicts": int(best_g),
                    "model_conflicts": int(conf),
                    "model_solved_rigorous": bool(ok),
                    "model_reported_conflicts": int(pred.get("conflicts", -1)),
                    "model_k_used": int(pred.get("k", -1)),
                }
            )
        denom = max(1, len([r for r in rows if "error" not in r]))
        report["blocks"].append(
            {
                "n": n,
                "k": k,
                "p_cross": p_cross,
                "elapsed_sec": time.time() - t0,
                "solved_rigorous": solved,
                "solved_rate": solved / denom,
                "rows": rows,
            }
        )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps({"wrote": args.out, "summary": [(b["n"], b["solved_rigorous"], b["solved_rate"]) for b in report["blocks"]]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
