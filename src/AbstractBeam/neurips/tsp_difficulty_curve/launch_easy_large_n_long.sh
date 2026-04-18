#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p neurips/tsp_difficulty_curve/easy_large_n_long_abstrue
exec nohup uv run python neurips/tsp_difficulty_curve/run_easy_large_n_long_abstrue.py \
  > neurips/tsp_difficulty_curve/easy_large_n_long_abstrue/nohup.log 2>&1
