#!/usr/bin/env python3
"""Fusionne rapports tensoriels + results.json Abstract Beam pour synthèse compare."""

from __future__ import annotations

import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
  out = {}
  for name in ("tensor_cp_report.json", "tensor_evolve_report.json", "combined_tensor_baselines.json"):
    p = os.path.join(SCRIPT_DIR, name)
    if os.path.isfile(p):
      with open(p, "r", encoding="utf-8") as f:
        out[name.replace(".json", "")] = json.load(f)

  run_root = os.path.join(SCRIPT_DIR, "mm_abstrue_runs")
  ab_summaries = []
  if os.path.isdir(run_root):
    for ent in sorted(os.listdir(run_root)):
      sp = os.path.join(run_root, ent, "run_summary.json")
      if os.path.isfile(sp):
        with open(sp, "r", encoding="utf-8") as f:
          ab_summaries.append(json.load(f))
      rp = os.path.join(run_root, ent, "results.json")
      if os.path.isfile(rp):
        with open(rp, "r", encoding="utf-8") as f:
          jd = json.load(f)
        n_ok = sum(1 for r in jd.get("results", []) if r.get("success"))
        ab_summaries.append({"tag": ent, "from_results_json": True, "solved": n_ok, "total": len(jd.get("results", []))})

  out["abstract_beam_summaries"] = ab_summaries
  out["interpretation"] = (
      "Baselines CP: erreur quasi nulle à partir du rang 7 = borne Strassen ⟨2,2,2⟩. "
      "Abstract Beam optimise la probabilité de programmes DSL exacts sur des I/O (cellules C_ij), "
      "pas la décomposition CP du tenseur; comparer le taux de succès beam au seuil d’erreur relative CP par rang."
  )
  dst = os.path.join(SCRIPT_DIR, "MM_RESEARCH_SYNTHESIS.json")
  with open(dst, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2)
  print("Écrit", dst)
  print(json.dumps(out.get("interpretation"), indent=0))


if __name__ == "__main__":
  main()
