#!/usr/bin/env bash
set -uo pipefail

BASE="/Data/janis.aiad/abstractrl/src/AbstractBeam/crossbeam/experiment/v4/runs/gcp_trm_mixed_curriculum_v4"
LOG_DIR="${BASE}/_logs"
mkdir -p "${LOG_DIR}"

OUT="${LOG_DIR}/auto_status_points.log"
INTERVAL_SEC="${INTERVAL_SEC:-600}"

echo "[watcher_start] $(date -Is) interval_sec=${INTERVAL_SEC}" >> "${OUT}"

while true; do
  NOW="$(date -Is)"
  SESSION="$(ls -dt "${BASE}"/mixed_curriculum_* 2>/dev/null | head -1 || true)"
  PROC="$(pgrep -af 'run_mixed_curriculum_gcp_v4.py|gcp_constructive_mdp_v0.py|run_mixed_and_constructive_after_stop.sh' | grep -v grep || true)"

  {
    echo ""
    echo "===== ${NOW} ====="
    if [[ -n "${SESSION}" ]]; then
      echo "session=$(basename "${SESSION}")"
      if [[ -f "${SESSION}/summary.json" ]]; then
        echo "summary=present"
      else
        echo "summary=missing"
      fi
      if [[ -d "${SESSION}/stage1_100_200" ]]; then
        echo "stage1_dir=present"
      fi
      if [[ -d "${SESSION}/stage2_400" ]]; then
        echo "stage2_dir=present"
      fi
      if [[ -d "${SESSION}/ladder_eval" ]]; then
        echo "ladder_eval_dir=present"
      fi
      if [[ -d "${SESSION}/constructive_traces" ]]; then
        echo "constructive_traces_dir=present"
      fi
    else
      echo "session=none"
    fi

    if [[ -n "${PROC}" ]]; then
      echo "proc=running"
      echo "${PROC}"
    else
      echo "proc=none"
    fi

    MIXED_LOG="$(ls -t "${LOG_DIR}"/mixed_curriculum_*.log 2>/dev/null | head -1 || true)"
    if [[ -n "${MIXED_LOG}" ]]; then
      echo "mixed_log=$(basename "${MIXED_LOG}")"
      tail -n 8 "${MIXED_LOG}" 2>/dev/null | sed 's/^/mixed_tail: /'
    fi

    CONSTR_LOG="$(ls -t "${LOG_DIR}"/*_constructive.log 2>/dev/null | head -1 || true)"
    if [[ -n "${CONSTR_LOG}" ]]; then
      echo "constructive_log=$(basename "${CONSTR_LOG}")"
      tail -n 8 "${CONSTR_LOG}" 2>/dev/null | sed 's/^/constructive_tail: /'
    fi
  } >> "${OUT}"

  sleep "${INTERVAL_SEC}"
done
