#!/usr/bin/env bash
# Launches the sequential two-main-campaign script under nohup (survives SSH logout).
# Usage:
#   bash launch_two_main_overnight.sh
# Optional before calling:
#   export STORE_MCTS_TREES=1   # enables --store-mcts-trees (very large JSON per MCTS solve)

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAIN="${HERE}/run_two_main_v3refactor.sh"
LOG_DIR="${HERE}"
STAMP="$(date +%s)"
META="${LOG_DIR}/overnight_two_main_${STAMP}.log"
PID_FILE="${LOG_DIR}/overnight_two_main.pid"

if [[ ! -x "${MAIN}" ]] && [[ -f "${MAIN}" ]]; then
  chmod +x "${MAIN}" || true
fi

nohup bash "${MAIN}" >>"${META}" 2>&1 &
echo "${!}" >"${PID_FILE}"
echo "nohup_pid=${!}"
echo "meta_log=${META}"
echo "pid_file=${PID_FILE}"
echo "tail: tail -f ${META}"
