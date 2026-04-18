#!/usr/bin/env python3
"""Analyse: les macros fn_* sont-elles employées de façon stable pour du TSP (chaînes Add d'arêtes m) ?

Lit des results.json (éval) et compte:
- références fn_k
- motifs Access(j, Access(i, m)) = lecture d'une arête dirigée dans la matrice ATSP
- heuristiques liste: Sum/Reverse/Maximum sur Access(., m) sans chaîne tour complète

Sortie: tableaux succès vs échec, et verdict qualitatif.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# Arête ATSP telle que dans les solutions canoniques: Access(dest, Access(src, m))
EDGE_RE = re.compile(r"Access\(\s*(\d+)\s*,\s*Access\(\s*(\d+)\s*,\s*m\s*\)\s*\)")
MACRO_RE = re.compile(r"fn_(\d+)")
# Accès "ligne seule" souvent vu dans les faux positifs: Access(k, m) (pas nested)
ACCESS_ROW_RE = re.compile(r"Access\(\s*(\d+)\s*,\s*m\s*\)")


@dataclass
class RowStats:
  success: bool
  n_cities: Optional[int]
  n_macros: int
  macro_ids: Tuple[str, ...]
  n_tsp_edges: int
  edge_pairs: Tuple[Tuple[int, int], ...]
  n_access_row_m: int
  gold_tsp_edges: int
  has_min: bool
  has_add: bool
  solution_len: int
  task_solution_len: int


def parse_n_from_task(task_str: str) -> Optional[int]:
  if "inputs_dict=" not in task_str:
    return None
  # première matrice: compter les sous-listes au niveau le plus interne simplifié via json
  i = task_str.find("'m': [[[")
  if i < 0:
    return None
  a = task_str.find("[[[", i)
  b = task_str.find("]]]", a) + 3
  try:
    mat = json.loads(task_str[a:b].replace("'", '"'))[0]
    return len(mat)
  except (json.JSONDecodeError, IndexError, TypeError):
    return None


def analyze_solution(
    solution: Optional[str],
    task_str: str,
    success: bool,
    task_solution: Optional[str],
) -> RowStats:
  s = solution or ""
  ts = task_solution or ""
  n = parse_n_from_task(task_str)
  macros = MACRO_RE.findall(s)
  edges = [(int(b), int(a)) for a, b in EDGE_RE.findall(s)]
  access_rows = [int(x) for x in ACCESS_ROW_RE.findall(s)]
  gold_edges = len(EDGE_RE.findall(ts))
  return RowStats(
      success=success,
      n_cities=n,
      n_macros=len(macros),
      macro_ids=tuple(sorted(set(macros))),
      n_tsp_edges=len(edges),
      edge_pairs=tuple(edges),
      n_access_row_m=len(access_rows),
      gold_tsp_edges=gold_edges,
      has_min="Min(" in s,
      has_add="Add(" in s,
      solution_len=len(s),
      task_solution_len=len(ts),
  )


def aggregate(rows: List[RowStats], label: str) -> Dict:
  if not rows:
    return {"label": label, "n": 0}
  succ = [r for r in rows if r.success]
  fail = [r for r in rows if not r.success]
  def pack(sub: List[RowStats], name: str):
    if not sub:
      return {f"{name}_n": 0}
    with_macro = sum(1 for r in sub if r.n_macros > 0)
    with_edges = sum(1 for r in sub if r.n_tsp_edges > 0)
    with_row_m = sum(1 for r in sub if r.n_access_row_m > 0)
    # "tour-shaped": au moins n arêtes typiques pour un chemin fermé (approximation)
    tourish = sum(
        1 for r in sub
        if r.n_cities and r.n_tsp_edges >= max(3, r.n_cities - 1)
    )
    return {
        f"{name}_n": len(sub),
        f"{name}_pct_macro": round(100.0 * with_macro / len(sub), 2),
        f"{name}_pct_any_tsp_edge": round(100.0 * with_edges / len(sub), 2),
        f"{name}_pct_Access_row_m": round(100.0 * with_row_m / len(sub), 2),
        f"{name}_pct_tourish_edges": round(100.0 * tourish / len(sub), 2),
        f"{name}_mean_tsp_edges": round(sum(r.n_tsp_edges for r in sub) / len(sub), 3),
        f"{name}_mean_Access_row_m": round(sum(r.n_access_row_m for r in sub) / len(sub), 3),
        f"{name}_mean_macros": round(sum(r.n_macros for r in sub) / len(sub), 3),
        f"{name}_pct_min": round(100.0 * sum(1 for r in sub if r.has_min) / len(sub), 2),
    }
  out = {"label": label, "n": len(rows)}
  out.update(pack(rows, "all"))
  out.update(pack(succ, "success"))
  out.update(pack(fail, "fail"))
  return out


def macro_cooccurrence(rows: List[RowStats]) -> Dict:
  """Parmi les lignes avec macros: combien ont aussi des arêtes TSP typiques ?"""
  with_m = [r for r in rows if r.n_macros > 0]
  if not with_m:
    return {"rows_with_macros": 0}
  also_edge = sum(1 for r in with_m if r.n_tsp_edges > 0)
  also_row = sum(1 for r in with_m if r.n_access_row_m > 0)
  also_tourish = sum(
      1 for r in with_m
      if r.n_cities and r.n_tsp_edges >= max(3, r.n_cities - 1)
  )
  return {
      "rows_with_macros": len(with_m),
      "macro_rows_also_tsp_edge_pct": round(100.0 * also_edge / len(with_m), 2),
      "macro_rows_also_Access_row_m_pct": round(100.0 * also_row / len(with_m), 2),
      "macro_rows_tourish_pct": round(100.0 * also_tourish / len(with_m), 2),
  }


def stable_macro_ids(rows: List[RowStats], min_frac: float = 0.5) -> List[str]:
  """Macros qui apparaissent dans une fraction min des succès (stabilité grossière)."""
  succ = [r for r in rows if r.success]
  if not succ:
    return []
  c = Counter()
  for r in succ:
    for mid in r.macro_ids:
      c[mid] += 1
  thr = max(1, int(min_frac * len(succ)))
  return sorted([k for k, v in c.items() if v >= thr], key=int)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument(
      "paths",
      nargs="*",
      help="Fichiers results.json ou globs; défaut: tsp_difficulty_curve/**/results.json",
  )
  ap.add_argument(
      "-o",
      "--output",
      default="",
      help="Écrire le JSON complet dans ce fichier (optionnel).",
  )
  args = ap.parse_args()
  if args.paths:
    files = []
    for p in args.paths:
      files.extend(glob.glob(p, recursive=True))
  else:
    root = os.path.dirname(os.path.abspath(__file__))
    files = glob.glob(os.path.join(root, "**", "results.json"), recursive=True)
  files = sorted(set(f for f in files if os.path.isfile(f)))

  by_file = {}
  all_rows: List[RowStats] = []

  for path in files:
    try:
      with open(path, "r") as f:
        data = json.load(f)
    except (OSError, json.JSONDecodeError):
      continue
    results = data.get("results")
    if not isinstance(results, list):
      continue
    stats_list = []
    for row in results:
      st = analyze_solution(
          row.get("solution"),
          row.get("task", ""),
          bool(row.get("success")),
          row.get("task_solution"),
      )
      stats_list.append(st)
      all_rows.append(st)
    by_file[path] = stats_list

  gold_edges = [r.gold_tsp_edges for r in all_rows if r.gold_tsp_edges]
  report = {
      "files": len(by_file),
      "total_rows": len(all_rows),
      "note": (
          "Les results.json ne contiennent que la solution finale (pas le beam). "
          "Les macros fn_* peuvent être absentes si aucun programme final ne les appelle "
          "(souvent des raccourcis Add/Sum/Access(k,m)). "
          "Référence or: task_solution = Min/Add/Access(j,Access(i,m))."
      ),
      "task_solution_Access_edge_term_counts": {
          "note": "Nombre total de motifs Access(j,Access(i,m)) dans tout le task_solution (toutes branches Min), pas la longueur d'un seul tour.",
          "mean": round(sum(gold_edges) / len(gold_edges), 1) if gold_edges else 0,
          "min": min(gold_edges) if gold_edges else 0,
          "max": max(gold_edges) if gold_edges else 0,
      },
      "global": aggregate(all_rows, "global"),
      "macro_cooccurrence_global": macro_cooccurrence(all_rows),
      "stable_macro_ids_in_success_ge_50pct": stable_macro_ids(all_rows, 0.5),
      "per_file": {os.path.basename(os.path.dirname(k)): aggregate(v, k) for k, v in by_file.items()},
  }

  # Verdict texte
  g = report["global"]
  mc = report["macro_cooccurrence_global"]
  verdict = []
  if g["success_n"] == 0:
    verdict.append("Aucun succès: impossible de mesurer stabilité des macros sur programmes exacts.")
  else:
    verdict.append(
        f"Succès: {g['success_n']} lignes — fn_* dans {g['success_pct_macro']}% des succès; "
        f"Access(j,Access(i,m)) dans {g['success_pct_any_tsp_edge']}% des succès; "
        f"Access(k,m) (ligne) dans {g.get('success_pct_Access_row_m', 0)}% des succès."
    )
    if mc["rows_with_macros"] == 0:
      verdict.append(
          "Aucune solution enregistrée ne contient le texte 'fn_*': pas d'emploi stable observable "
          "des macros dans les programmes finaux (elles peuvent exister dans le domaine mais ne sont pas "
          "dans l'expression() du meilleur candidat sauvegardé)."
      )
    else:
      verdict.append(
          f"Parmi les lignes avec macros ({mc['rows_with_macros']}), "
          f"{mc['macro_rows_also_tsp_edge_pct']}% ont Access(j,Access(i,m)), "
          f"{mc['macro_rows_tourish_pct']}% tourish."
      )
  sme = g.get("success_mean_tsp_edges")
  smrow = g.get("success_mean_Access_row_m")
  if sme is not None and smrow is not None and g.get("success_n", 0):
    if sme < 1 and smrow > 0:
      verdict.append(
          "Les succès alignent plutôt sur des lectures Access(k,m) / agrégats (Sum, Reverse…), "
          "pas sur des chaînes d'arêtes dirigées comme la cible Min(Add(Access(...)))."
      )
  report["verdict_lines"] = verdict

  text = json.dumps(report, indent=2)
  print(text)
  if args.output:
    with open(args.output, "w") as f:
      f.write(text)
    print("\nWrote", args.output)
  print("\n--- Verdict ---")
  for line in verdict:
    print(line)


if __name__ == "__main__":
  main()
