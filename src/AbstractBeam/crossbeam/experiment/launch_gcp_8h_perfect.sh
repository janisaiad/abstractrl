#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p runs/gcp_trm_scaleup
LOG="runs/gcp_trm_scaleup/nohup_8h_perfect_$(date +%s).log"
echo "Logging to $LOG"
nohup env PYTHONUNBUFFERED=1 python -u gcp_run8h_perfect.py --budget-hours 8 >"$LOG" 2>&1 &
echo "PID=$!"
