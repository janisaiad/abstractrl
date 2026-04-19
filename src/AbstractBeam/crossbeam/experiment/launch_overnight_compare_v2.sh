#!/usr/bin/env bash
set -euo pipefail

ROOT="/Data/janis.aiad/abstractrl/src/AbstractBeam/crossbeam/experiment"
RUN_ROOT="$ROOT/runs/gcp_trm_scaleup_v2"
BASE_SESSION="$RUN_ROOT/session_1776550324/n20"
TRAIN_GLOB="$BASE_SESSION/traces_train/*.pt"
VALID_GLOB="$BASE_SESSION/traces_valid/*.pt"
OUT_DIR="$RUN_ROOT/overnight_compare"
mkdir -p "$OUT_DIR"

TS="$(date +%s)"

OUT_A="$OUT_DIR/modelA_${TS}"
LOG_A="$OUT_A.log"
REP_A="$OUT_A.hard.json"
mkdir -p "$OUT_A"

OUT_B="$OUT_DIR/modelB_${TS}"
LOG_B="$OUT_B.log"
REP_B="$OUT_B.hard.json"
mkdir -p "$OUT_B"

CMD_A=$(cat <<EOF
set -euo pipefail
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
cd "$ROOT"
torchrun --nproc_per_node=1 gcp_trace_abstractbeam_v2.py train \\
  --train "$TRAIN_GLOB" \\
  --valid "$VALID_GLOB" \\
  --out-dir "$OUT_A" \\
  --batch-size 96 \\
  --epochs 36 \\
  --steps-per-epoch 4500 \\
  --valid-steps 300 \\
  --d-model 384 \\
  --refine-steps 4 \\
  --dropout 0.10 \\
  --lr 2e-4 \\
  --amp \\
  --seed 3101
python3 -u gcp_hard_benchmark_v2.py \\
  --ckpt "$OUT_A/model-best.pt" \\
  --out "$REP_A" \\
  --sizes "100,200,400,800" \\
  --per-size 6 \\
  --simulations 128 \\
  --max-depth 180 \\
  --max-tries 600 \\
  --random-orders 8 \\
  --min-greedy-conflicts 2 \\
  --max-greedy-conflicts 2000
EOF
)

CMD_B=$(cat <<EOF
set -euo pipefail
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
cd "$ROOT"
torchrun --nproc_per_node=1 gcp_trace_abstractbeam_v2.py train \\
  --train "$TRAIN_GLOB" \\
  --valid "$VALID_GLOB" \\
  --out-dir "$OUT_B" \\
  --batch-size 64 \\
  --epochs 44 \\
  --steps-per-epoch 5200 \\
  --valid-steps 320 \\
  --d-model 256 \\
  --refine-steps 5 \\
  --dropout 0.08 \\
  --lr 1.5e-4 \\
  --amp \\
  --seed 3201
python3 -u gcp_hard_benchmark_v2.py \\
  --ckpt "$OUT_B/model-best.pt" \\
  --out "$REP_B" \\
  --sizes "100,200,400,800" \\
  --per-size 6 \\
  --simulations 128 \\
  --max-depth 180 \\
  --max-tries 600 \\
  --random-orders 8 \\
  --min-greedy-conflicts 2 \\
  --max-greedy-conflicts 2000
EOF
)

nohup bash -lc "$CMD_A" > "$LOG_A" 2>&1 &
PID_A=$!
nohup bash -lc "$CMD_B" > "$LOG_B" 2>&1 &
PID_B=$!

echo "OUT_A=$OUT_A"
echo "LOG_A=$LOG_A"
echo "REP_A=$REP_A"
echo "PID_A=$PID_A"
echo "OUT_B=$OUT_B"
echo "LOG_B=$LOG_B"
echo "REP_B=$REP_B"
echo "PID_B=$PID_B"
