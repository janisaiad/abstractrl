#!/usr/bin/env bash
set -u
set -o pipefail

MAIN_SESSION="$1"
V3_DIR="/Data/janis.aiad/abstractrl/src/AbstractBeam/crossbeam/experiment/v3"
RUNS_DIR="${V3_DIR}/runs/gcp_trm_scaleup_v3"
LOGS_DIR="${RUNS_DIR}/_logs"
MAIN_REPORT="${RUNS_DIR}/${MAIN_SESSION}/FINAL_LADDER_REPORT.md"

V2_TRACES_BASE="/Data/janis.aiad/abstractrl/src/AbstractBeam/crossbeam/experiment/runs/gcp_trm_scaleup_v2/session_1776550919"
TRAIN_GLOBS="${V2_TRACES_BASE}/n20/traces_train/*.pt,${V2_TRACES_BASE}/n40/traces_train/*.pt"
VALID_GLOBS="${V2_TRACES_BASE}/n20/traces_valid/*.pt,${V2_TRACES_BASE}/n40/traces_valid/*.pt"
CURRICULUM_CKPT="${V2_TRACES_BASE}/n80/train_run/model-best.pt"

echo "[orchestrator] waiting for main report: ${MAIN_REPORT}"
while [[ ! -f "${MAIN_REPORT}" ]]; do
  sleep 30
done
echo "[orchestrator] main report ready at $(date -Is), launching macros campaign"

TS=$(date +%s)
MACRO_SESSION="small_to_large_ladder_macros_${TS}"
MACRO_LOG="${LOGS_DIR}/${MACRO_SESSION}.log"

cd "${V3_DIR}"
PYTHONPATH="/Data/janis.aiad/abstractrl/src/AbstractBeam/crossbeam/experiment" python3 -u run_small_to_large_ladder_v3.py \
  --session "${MACRO_SESSION}" \
  --small-train-globs "${TRAIN_GLOBS}" \
  --small-valid-globs "${VALID_GLOBS}" \
  --curriculum-ckpt "${CURRICULUM_CKPT}" \
  --sizes "100,200" \
  --per-size 10 \
  --budget-ladder "12:24,32:48" \
  --timeout-sec 40 \
  --profile-every 4 \
  --baseline-method fixed_tabu_recolor \
  --penalty-lambda 20.0 \
  --include-macro-method \
  --macro-min-support 8 \
  --macro-max-len 3 \
  --macro-top-k 32 \
  --macro-min-distinct-families 2 \
  --macro-require-structural \
  --macro-action-budget 8 \
  --macro-max-steps 2 \
  --macro-cheap \
  >> "${MACRO_LOG}" 2>&1
MACRO_STATUS=$?
echo "[orchestrator] macros campaign exited with status=${MACRO_STATUS} at $(date -Is)"

echo "[orchestrator] regenerating consolidated report"
python3 - <<'PY'
import json, os
from pathlib import Path
runs = Path("/Data/janis.aiad/abstractrl/src/AbstractBeam/crossbeam/experiment/v3/runs/gcp_trm_scaleup_v3")
reports = sorted([p for p in runs.glob("small_to_large_ladder_*/FINAL_LADDER_REPORT.md")])
out = runs / "RAPPORT_LADDERS_V3_CONSOLIDE.md"
header = [
    "# Rapport consolidé — ladders V3 (regénéré automatiquement)",
    "",
    f"Ce fichier est régénéré à la fin de chaque orchestration. Il réunit tous les `FINAL_LADDER_REPORT.md` présents sous `runs/gcp_trm_scaleup_v3/` au moment de la génération, dans l’ordre alphabétique.",
    "",
    "## Table des matières",
    "",
]
toc = []
sections = []
for i, p in enumerate(reports, start=1):
    slug = p.parent.name
    anchor = slug.lower().replace("_", "-")
    toc.append(f"{i}. [{slug}](#{anchor})")
    body = p.read_text()
    # re-home top-level heading and down-shift all # by two levels
    adjusted_lines = []
    for line in body.splitlines():
        if line.startswith("# "):
            adjusted_lines.append(f"### Contenu du rapport source — {slug}")
        elif line.startswith("#"):
            adjusted_lines.append("##" + line)
        else:
            adjusted_lines.append(line)
    sections.append(f"## {slug}\n\n- Fichier: `{p}`\n\n" + "\n".join(adjusted_lines))
big_train_logs = sorted(runs.glob("big_train_*/train_log.jsonl"))
big_section = ["## Big train logs"]
for bt in big_train_logs:
    rows = [json.loads(x) for x in bt.read_text().splitlines() if x.strip()]
    if not rows:
        continue
    last = rows[-1]
    best = min(rows, key=lambda r: r.get("valid_loss", 1e9))
    big_section.append(
        f"- `{bt.parent.name}`: epochs={len(rows)}, last epoch={last.get('epoch')}, last valid_loss={last.get('valid_loss'):.4f}, best valid_loss={best.get('valid_loss'):.4f} (epoch {best.get('epoch')}), sum time_sec={sum(float(r.get('time_sec', 0.0)) for r in rows):.2f}s"
    )
out.write_text("\n".join(header + toc + ["", ""] + big_section + [""] + sections) + "\n")
print("WROTE", out)
PY

echo "[orchestrator] done at $(date -Is)"
