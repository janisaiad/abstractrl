#!/usr/bin/env python3
"""
Benchmark + reverse-engineering MCTS sur les checkpoints « petits » déjà présents dans le repo.

Checkpoints reconnus (chemins relatifs à ce fichier = dossier v5/) :
  * ladder_trained_small : session ladder macros 2026-04-21, TRM entraîné sur traces n20/n40
    (fichier ladder_results.json : trained_small_ckpt).
  * big_train : run dédié 20 epochs `big_train_1776710200` (souvent utilisé comme init curriculum v5).

Note factuelle (RAPPORT_LADDERS_V3_CONSOLIDE.md) : en métrique timeout-aware, le petit MCTS bat
surtout greedy / mcts_no_model ; contre `fixed_tabu_recolor` le win rate reste modeste (~0.1 sur
certaines grilles). Ce script sert quand même à rejouer des evals avec dumps d’arbres + stats
d’instrumentation pour analyse fine.

Enchaîne :
  1) `run_mcts_inference_analysis.py` (solve + arbres + récits + plots optionnels)
  2) `aggregate_mcts_sanity_stats.py` sur le dossier `mcts_trees/` produit.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List

V5 = Path(__file__).resolve().parent
EXPERIMENT_ROOT = V5.parent

DEFAULT_LADDER_SMALL = (
    EXPERIMENT_ROOT
    / "v3"
    / "runs"
    / "gcp_trm_scaleup_v3"
    / "small_to_large_ladder_macros_1776746475"
    / "trained_small"
    / "model-best.pt"
)
DEFAULT_BIG_TRAIN = (
    EXPERIMENT_ROOT
    / "v3"
    / "runs"
    / "gcp_trm_scaleup_v3"
    / "big_train_1776710200"
    / "model-best.pt"
)


def _resolve_ckpt(preset: str, ckpt_override: str) -> Path:
    if str(ckpt_override).strip():
        p = Path(ckpt_override).expanduser()
        if not p.is_file():
            raise FileNotFoundError(str(p))
        return p
    if preset == "ladder_trained_small":
        p = DEFAULT_LADDER_SMALL
    elif preset == "big_train":
        p = DEFAULT_BIG_TRAIN
    else:
        raise ValueError(preset)
    if not p.is_file():
        raise FileNotFoundError(f"Checkpoint manquant: {p}")
    return p


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark JSONL + reverse-engineering MCTS (petit ckpt connu).")
    ap.add_argument(
        "--ckpt-preset",
        choices=["ladder_trained_small", "big_train"],
        default="ladder_trained_small",
        help="Checkpoint par défaut si --ckpt est vide",
    )
    ap.add_argument("--ckpt", type=str, default="", help="Surcharge absolue du chemin .pt")
    ap.add_argument("--input", type=str, required=True, help="JSONL de graphes (ex. stage1_n100/eval.jsonl)")
    ap.add_argument("--out-dir", type=str, required=True, help="Répertoire de sortie (arbres, rapports, plots)")
    ap.add_argument("--max-graphs", type=int, default=20)
    ap.add_argument("--k", type=int, default=0, help="k fixe (0 = heuristique solve)")
    ap.add_argument("--simulations", type=int, default=320)
    ap.add_argument("--max-depth", type=int, default=96)
    ap.add_argument("--worker-count", type=int, default=2)
    ap.add_argument(
        "--mcts-sim-trace",
        type=str,
        default="aggregates",
        choices=["off", "aggregates", "full"],
        help="Instrumentation par simulation (full = très verbeux)",
    )
    ap.add_argument("--mcts-sim-trace-cap", type=int, default=50000)
    ap.add_argument("--skip-plots", action="store_true")
    ap.add_argument("--skip-aggregate", action="store_true", help="Ne pas lancer aggregate_mcts_sanity_stats")
    args = ap.parse_args()

    ckpt = _resolve_ckpt(str(args.ckpt_preset), str(args.ckpt))
    out_dir = Path(str(args.out_dir)).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    infer_script = V5 / "run_mcts_inference_analysis.py"
    agg_script = V5 / "aggregate_mcts_sanity_stats.py"
    trees_dir = out_dir / "mcts_trees"
    agg_out = out_dir / "mcts_reverse_eng_batch_stats.json"
    manifest = {
        "ckpt_preset": str(args.ckpt_preset),
        "ckpt_resolved": str(ckpt.resolve()),
        "input": str(Path(args.input).expanduser().resolve()),
        "out_dir": str(out_dir.resolve()),
    }
    (out_dir / "reverse_eng_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    cmd_inf: List[str] = [
        sys.executable,
        str(infer_script),
        "--ckpt",
        str(ckpt),
        "--input",
        str(Path(args.input).expanduser()),
        "--out-dir",
        str(out_dir),
        "--max-graphs",
        str(int(args.max_graphs)),
        "--k",
        str(int(args.k)),
        "--simulations",
        str(int(args.simulations)),
        "--max-depth",
        str(int(args.max_depth)),
        "--worker-count",
        str(int(args.worker_count)),
        "--mcts-sim-trace",
        str(args.mcts_sim_trace),
        "--mcts-sim-trace-cap",
        str(int(args.mcts_sim_trace_cap)),
    ]
    if args.skip_plots:
        cmd_inf.append("--skip-plots")
    subprocess.run(cmd_inf, cwd=str(EXPERIMENT_ROOT), check=True)

    if not args.skip_aggregate:
        subprocess.run(
            [
                sys.executable,
                str(agg_script),
                "--inputs",
                str(trees_dir),
                "--out-json",
                str(agg_out),
            ],
            cwd=str(EXPERIMENT_ROOT),
            check=True,
        )

    print(json.dumps({"ok": True, "ckpt": str(ckpt), "out_dir": str(out_dir), "aggregate": str(agg_out)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
