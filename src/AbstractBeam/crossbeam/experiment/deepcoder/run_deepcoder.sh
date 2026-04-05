#!/usr/bin/env bash
# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

CONFIG_PATH="${ABSTRACTBEAM_CONFIG:-${ROOT}/crossbeam/experiment/deepcoder/configs/testing_config.py}"
mkdir -p "${ROOT}/neurips/testing/data" "${ROOT}/neurips/testing" \
  "${ROOT}/neurips/abstractbeam/data" "${ROOT}/neurips/abstractbeam/models" "${ROOT}/neurips/abstractbeam/results" \
  "${ROOT}/neurips/lambdabeam/data" "${ROOT}/neurips/lambdabeam/models" "${ROOT}/neurips/lambdabeam/results"

echo "ROOT=${ROOT}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "CONFIG_PATH=${CONFIG_PATH}"

python3 -m crossbeam.experiment.run_crossbeam \
  --config="${CONFIG_PATH}" \
  --domain=deepcoder

