#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
LOGDIR="$ROOT/neurips/mm_tensor_gpu/mm_abstrue_runs"
mkdir -p "$LOGDIR"
LOG="$LOGDIR/autoresearch_10h_$(date +%Y%m%d_%H%M%S).log"
cd "$ROOT"
# Ancien job 2×2 seed 902 : éviter la contention GPU (optionnel)
pkill -f "run_mm_research.py --seed 902" 2>/dev/null || true
exec nohup uv run python "$ROOT/neurips/mm_tensor_gpu/autoresearch_mm_loop.py" \
  --budget_hours 10 --seed 17042 --port_base 59000 >>"$LOG" 2>&1 &
echo "PID $! log $LOG"
