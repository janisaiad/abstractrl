#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

mkdir -p neurips/nightly/logs

stamp="$(date +%Y%m%d_%H%M%S)"
runner="neurips/nightly/logs/night_runner_${stamp}.sh"
logfile="neurips/nightly/logs/night_runner_${stamp}.log"
pidfile="neurips/nightly/logs/night_runner_${stamp}.pid"

cat > "$runner" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
cd "/Data/janis.aiad/abstractrl/src/AbstractBeam"

echo "[night] start $(date -Is)"

# 1) Sort phase scan from swaps >=2 until drop <=50%
uv run python neurips/sort_phase_transition_gpu/run_phase_from2_until_drop.py

# 2) Large sort configuration search (many lengths/modes/swaps/seeds/steps)
uv run python neurips/nightly/run_sort_config_search.py

# 3) TSP AB vs NN multi-seed benchmark (Zak rule: <=10% gap)
uv run python neurips/tsp_curriculum_gpu_quick/run_tsp_nn_vs_ab_night.py

echo "[night] done $(date -Is)"
EOF

chmod +x "$runner"
nohup bash "$runner" > "$logfile" 2>&1 &
echo $! > "$pidfile"

echo "runner=$runner"
echo "log=$logfile"
echo "pid=$(cat "$pidfile")"
