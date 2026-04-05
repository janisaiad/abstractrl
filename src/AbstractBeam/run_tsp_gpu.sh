#!/usr/bin/env bash
# Entraînement AbstractBeam / crossbeam sur le domaine TSP (ATSP 3 villes), GPU 0 par défaut.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

mkdir -p "${ROOT}/neurips/tsp/data" "${ROOT}/neurips/tsp/models" "${ROOT}/neurips/tsp/results"

echo "Préparation des pickles d'entraînement TSP..."
python3 -c "from crossbeam.data.tsp.tsp_tasks import write_training_pickles; write_training_pickles('${ROOT}/neurips/tsp/data')"

CONFIG="${ABSTRACTBEAM_CONFIG:-${ROOT}/crossbeam/experiment/deepcoder/configs/train/tsp_gpu.py}"
echo "ROOT=${ROOT}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "CONFIG=${CONFIG}"

python3 -m crossbeam.experiment.run_crossbeam --config="${CONFIG}"
