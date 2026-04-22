#!/usr/bin/env bash
set -euo pipefail
MAIN_SESSION="$1"
V3_DIR="/Data/janis.aiad/abstractrl/src/AbstractBeam/crossbeam/experiment/v3"
RUNS_DIR="${V3_DIR}/runs/gcp_trm_scaleup_v3"
LOGS_DIR="${RUNS_DIR}/_logs"
MAIN_REPORT="${RUNS_DIR}/${MAIN_SESSION}/FINAL_LADDER_REPORT.md"
while [[ ! -f "${MAIN_REPORT}" ]]; do sleep 30; done
TS=$(date +%s)
SESSION="small_to_large_ladder_macros_v3refactor_${TS}"
LOGFILE="${LOGS_DIR}/${SESSION}.log"
PYTHONPATH=/Data/janis.aiad/abstractrl/src/AbstractBeam/crossbeam/experiment python3 -u "${V3_DIR}/run_small_to_large_ladder_v3.py" \
  --session "${SESSION}" \
  --solver-mode inprocess \
  --penalty-mode last_profile_plus_const \
  --penalty-lambda 20 \
  --small-train-globs "/Data/janis.aiad/abstractrl/src/AbstractBeam/crossbeam/experiment/runs/gcp_trm_scaleup_v2/session_1776550919/n20/traces_train/*.pt,/Data/janis.aiad/abstractrl/src/AbstractBeam/crossbeam/experiment/runs/gcp_trm_scaleup_v2/session_1776550919/n40/traces_train/*.pt" \
  --small-valid-globs "/Data/janis.aiad/abstractrl/src/AbstractBeam/crossbeam/experiment/runs/gcp_trm_scaleup_v2/session_1776550919/n20/traces_valid/*.pt,/Data/janis.aiad/abstractrl/src/AbstractBeam/crossbeam/experiment/runs/gcp_trm_scaleup_v2/session_1776550919/n40/traces_valid/*.pt" \
  --curriculum-ckpt /Data/janis.aiad/abstractrl/src/AbstractBeam/crossbeam/experiment/runs/gcp_trm_scaleup_v2/session_1776550919/n80/train_run/model-best.pt \
  --small-pretrained-ckpt /Data/janis.aiad/abstractrl/src/AbstractBeam/crossbeam/experiment/v3/runs/gcp_trm_scaleup_v3/big_train_1776710200/model-best.pt \
  --skip-small-train \
  --sizes "100,200" \
  --per-size 10 \
  --budget-ladder "12:24,32:48" \
  --timeout-sec 40 \
  --profile-every 4 \
  --baseline-method fixed_tabu_recolor \
  --include-macro-method \
  --macro-max-len 3 \
  --macro-top-k 32 \
  --macro-min-distinct-families 2 \
  --macro-require-structural \
  --macro-action-budget 8 \
  --macro-max-steps 2 \
  --macro-cheap > "${LOGFILE}" 2>&1
