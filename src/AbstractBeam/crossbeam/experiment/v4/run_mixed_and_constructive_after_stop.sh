#!/usr/bin/env bash
set -uo pipefail

V4_DIR="/Data/janis.aiad/abstractrl/src/AbstractBeam/crossbeam/experiment/v4"
RUNS_V4="${V4_DIR}/runs/gcp_trm_mixed_curriculum_v4"
LOG_DIR="${RUNS_V4}/_logs"
mkdir -p "${LOG_DIR}"

SMALL_TRAIN_GLOBS="/Data/janis.aiad/abstractrl/src/AbstractBeam/crossbeam/experiment/runs/gcp_trm_scaleup_v2/session_1776550919/n20/traces_train/*.pt,/Data/janis.aiad/abstractrl/src/AbstractBeam/crossbeam/experiment/runs/gcp_trm_scaleup_v2/session_1776550919/n40/traces_train/*.pt"
SMALL_VALID_GLOBS="/Data/janis.aiad/abstractrl/src/AbstractBeam/crossbeam/experiment/runs/gcp_trm_scaleup_v2/session_1776550919/n20/traces_valid/*.pt,/Data/janis.aiad/abstractrl/src/AbstractBeam/crossbeam/experiment/runs/gcp_trm_scaleup_v2/session_1776550919/n40/traces_valid/*.pt"
SMALL_PRETRAINED="/Data/janis.aiad/abstractrl/src/AbstractBeam/crossbeam/experiment/v3/runs/gcp_trm_scaleup_v3/big_train_1776710200/model-best.pt"

TS="$(date +%s)"
MIXED_SESSION="mixed_curriculum_${TS}"
MIXED_LOG="${LOG_DIR}/${MIXED_SESSION}.log"

cd "${V4_DIR}"

echo "[start] mixed ${MIXED_SESSION} $(date -Is)"
PYTHONPATH=/Data/janis.aiad/abstractrl/src/AbstractBeam/crossbeam/experiment python3 -u run_mixed_curriculum_gcp_v4.py \
  --session "${MIXED_SESSION}" \
  --small-train-globs "${SMALL_TRAIN_GLOBS}" \
  --small-valid-globs "${SMALL_VALID_GLOBS}" \
  --small-pretrained-ckpt "${SMALL_PRETRAINED}" \
  --run-ladder-eval \
  > "${MIXED_LOG}" 2>&1
MIXED_STATUS=$?
echo "[done] mixed ${MIXED_SESSION} status=${MIXED_STATUS} $(date -Is)"

CONSTR_LOG="${LOG_DIR}/${MIXED_SESSION}_constructive.log"
CONSTR_INPUT="${RUNS_V4}/${MIXED_SESSION}/stage2_400/data/stage2_n400/eval.jsonl"
CONSTR_OUT="${RUNS_V4}/${MIXED_SESSION}/constructive_traces"

if [[ "${MIXED_STATUS}" -eq 0 ]] && [[ -f "${CONSTR_INPUT}" ]]; then
  echo "[start] constructive_build_traces ${MIXED_SESSION} $(date -Is)"
  PYTHONPATH=/Data/janis.aiad/abstractrl/src/AbstractBeam/crossbeam/experiment python3 -u gcp_constructive_mdp_v0.py build-traces \
    --input "${CONSTR_INPUT}" \
    --out-dir "${CONSTR_OUT}" \
    --reward-mode hybrid \
    > "${CONSTR_LOG}" 2>&1
  CB_STATUS=$?
  echo "[done] constructive_build_traces ${MIXED_SESSION} status=${CB_STATUS} $(date -Is)"

  echo "[start] constructive_rollout ${MIXED_SESSION} $(date -Is)"
  PYTHONPATH=/Data/janis.aiad/abstractrl/src/AbstractBeam/crossbeam/experiment python3 -u gcp_constructive_mdp_v0.py rollout \
    --input "${CONSTR_INPUT}" \
    --k 10 \
    --beam-width 16 \
    --reward-mode delta_belief \
    >> "${CONSTR_LOG}" 2>&1
  CR_STATUS=$?
  echo "[done] constructive_rollout ${MIXED_SESSION} status=${CR_STATUS} $(date -Is)"
else
  CB_STATUS=99
  CR_STATUS=99
  echo "[skip] constructive steps because mixed failed or input missing"
fi

echo "MIXED_SESSION=${MIXED_SESSION}"
echo "MIXED_LOG=${MIXED_LOG}"
echo "CONSTR_LOG=${CONSTR_LOG}"
echo "MIXED_STATUS=${MIXED_STATUS} CB_STATUS=${CB_STATUS} CR_STATUS=${CR_STATUS}"

if [[ "${MIXED_STATUS}" -ne 0 ]] || [[ "${CB_STATUS}" -ne 0 ]] || [[ "${CR_STATUS}" -ne 0 ]]; then
  exit 1
fi
exit 0
