#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt


def parse_args():
  p = argparse.ArgumentParser()
  p.add_argument("--summary", required=True)
  p.add_argument("--out_dir", required=True)
  return p.parse_args()


def main():
  args = parse_args()
  os.makedirs(args.out_dir, exist_ok=True)
  with open(args.summary, "r") as f:
    data = json.load(f)
  rows = data["rows"]
  meta = data.get("meta", {})

  grouped = defaultdict(list)
  for r in rows:
    grouped[int(r["length"])].append(r)

  # Figure 1: success rate vs tau for each length.
  plt.figure(figsize=(10, 6))
  for L in sorted(grouped.keys()):
    pts = sorted(grouped[L], key=lambda x: x["tau_target"])
    taus = [p["tau_target"] for p in pts]
    rates = [p["rate"] for p in pts]
    plt.plot(taus, rates, marker="o", linewidth=1.5, label=f"L={L}")
  plt.ylim(-0.02, 1.02)
  plt.xlim(0.0, 1.0)
  plt.xlabel("Kendall tau cible")
  plt.ylabel("Taux de succes")
  plt.title("AbstractBeam Sort(x): taux de succes vs Kendall tau")
  plt.grid(alpha=0.25)
  plt.legend(ncol=3, fontsize=8)
  cfg_txt = (
      f"Config: steps={meta.get('steps')} | train_tasks={meta.get('train_tasks')} | "
      f"eval_tasks={meta.get('eval_tasks')} | seeds={meta.get('seeds')}"
  )
  plt.figtext(0.5, 0.01, cfg_txt, ha="center", fontsize=9)
  out1 = os.path.join(args.out_dir, "kendall_success_by_length.png")
  plt.tight_layout(rect=[0, 0.04, 1, 1])
  plt.savefig(out1, dpi=160)
  plt.close()

  # Figure 2: heatmap (length x tau) with mean rate over seeds.
  taus = sorted({float(r["tau_target"]) for r in rows})
  lengths = sorted(grouped.keys())
  tau_to_idx = {t: i for i, t in enumerate(taus)}
  mat = [[0.0 for _ in taus] for _ in lengths]
  cnt = [[0 for _ in taus] for _ in lengths]
  len_to_idx = {L: i for i, L in enumerate(lengths)}
  for r in rows:
    i = len_to_idx[int(r["length"])]
    j = tau_to_idx[float(r["tau_target"])]
    mat[i][j] += float(r["rate"])
    cnt[i][j] += 1
  for i in range(len(lengths)):
    for j in range(len(taus)):
      if cnt[i][j]:
        mat[i][j] /= cnt[i][j]

  plt.figure(figsize=(10, 6))
  im = plt.imshow(mat, aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
  plt.colorbar(im, label="Taux de succes moyen")
  plt.yticks(range(len(lengths)), [str(L) for L in lengths])
  plt.xticks(range(len(taus)), [f"{t:.2f}" for t in taus], rotation=45)
  plt.xlabel("Kendall tau cible")
  plt.ylabel("Longueur L")
  plt.title("Heatmap succes: L x tau")
  plt.figtext(0.5, 0.01, cfg_txt, ha="center", fontsize=9)
  out2 = os.path.join(args.out_dir, "kendall_success_heatmap.png")
  plt.tight_layout(rect=[0, 0.04, 1, 1])
  plt.savefig(out2, dpi=160)
  plt.close()

  print("wrote", out1)
  print("wrote", out2)


if __name__ == "__main__":
  main()
