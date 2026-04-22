#!/usr/bin/env bash
# Do not use "set -e": if run 1 fails we still want run 2 overnight. Non-zero exit if either run failed.
set -uo pipefail

V3_DIR="/Data/janis.aiad/abstractrl/src/AbstractBeam/crossbeam/experiment/v3"
RUNS_DIR="${V3_DIR}/runs/gcp_trm_scaleup_v3"
LOG_DIR="${RUNS_DIR}/_logs"
mkdir -p "${LOG_DIR}"

TRAIN_GLOBS="/Data/janis.aiad/abstractrl/src/AbstractBeam/crossbeam/experiment/runs/gcp_trm_scaleup_v2/session_1776550919/n20/traces_train/*.pt,/Data/janis.aiad/abstractrl/src/AbstractBeam/crossbeam/experiment/runs/gcp_trm_scaleup_v2/session_1776550919/n40/traces_train/*.pt"
VALID_GLOBS="/Data/janis.aiad/abstractrl/src/AbstractBeam/crossbeam/experiment/runs/gcp_trm_scaleup_v2/session_1776550919/n20/traces_valid/*.pt,/Data/janis.aiad/abstractrl/src/AbstractBeam/crossbeam/experiment/runs/gcp_trm_scaleup_v2/session_1776550919/n40/traces_valid/*.pt"
CURRICULUM_CKPT="/Data/janis.aiad/abstractrl/src/AbstractBeam/crossbeam/experiment/runs/gcp_trm_scaleup_v2/session_1776550919/n80/train_run/model-best.pt"
BIGTRAIN_CKPT="/Data/janis.aiad/abstractrl/src/AbstractBeam/crossbeam/experiment/v3/runs/gcp_trm_scaleup_v3/big_train_1776710200/model-best.pt"

COMMON_ARGS=(
  --solver-mode inprocess
  --penalty-mode last_profile_plus_const
  --penalty-lambda 20
  --small-train-globs "${TRAIN_GLOBS}"
  --small-valid-globs "${VALID_GLOBS}"
  --curriculum-ckpt "${CURRICULUM_CKPT}"
  --small-pretrained-ckpt "${BIGTRAIN_CKPT}"
  --sizes "100,200,400"
  --per-size 10
  --budget-ladder "12:24,32:48,64:96"
  --timeout-sec 60
  --profile-every 4
  --baseline-method fixed_tabu_recolor
)

# Optional full MCTS tree JSON (huge). Default off for stable overnight runs.
EXTRA_V3=()
if [[ "${STORE_MCTS_TREES:-0}" == "1" ]]; then
  EXTRA_V3+=(--store-mcts-trees)
fi

cd "${V3_DIR}"

TS1=$(date +%s)
SESSION1="small_to_large_ladder_main_bigtrain_skip_${TS1}"
LOG1="${LOG_DIR}/${SESSION1}.log"
echo "[start] ${SESSION1} $(date -Is)"
PYTHONPATH=/Data/janis.aiad/abstractrl/src/AbstractBeam/crossbeam/experiment python3 -u run_small_to_large_ladder_v3.py \
  --session "${SESSION1}" \
  "${COMMON_ARGS[@]}" \
  "${EXTRA_V3[@]}" \
  --skip-small-train \
  > "${LOG1}" 2>&1
STATUS1=$?
echo "[done] ${SESSION1} status=${STATUS1} $(date -Is)"

TS2=$(date +%s)
SESSION2="small_to_large_ladder_main_bigtrain_finetune_${TS2}"
LOG2="${LOG_DIR}/${SESSION2}.log"
echo "[start] ${SESSION2} $(date -Is)"
PYTHONPATH=/Data/janis.aiad/abstractrl/src/AbstractBeam/crossbeam/experiment python3 -u run_small_to_large_ladder_v3.py \
  --session "${SESSION2}" \
  "${COMMON_ARGS[@]}" \
  "${EXTRA_V3[@]}" \
  > "${LOG2}" 2>&1
STATUS2=$?
echo "[done] ${SESSION2} status=${STATUS2} $(date -Is)"

echo "SESSION1=${SESSION1}"
echo "LOG1=${LOG1}"
echo "SESSION2=${SESSION2}"
echo "LOG2=${LOG2}"
echo "STATUS1=${STATUS1} STATUS2=${STATUS2}"

if [[ "${STATUS1}" -ne 0 ]] || [[ "${STATUS2}" -ne 0 ]]; then
  exit 1
fi
exit 0
