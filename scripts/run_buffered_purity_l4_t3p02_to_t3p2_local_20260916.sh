#!/bin/bash

# Continue the paired Y+ L=4 trajectories from t=3.020 to t=3.200.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON=/Users/phys2259/opt/anaconda3/envs/qnexus-env/bin/python
PILOT="$ROOT/runs/buffered_purity_l4_t3_to_t3p02_localpilot_20260916"
OUTPUT="$ROOT/runs/buffered_purity_l4_t3p02_to_t3p2_local_20260916"
CONTROL_PREVIOUS="$PILOT/control/states/step_000019_t3.019.npy"
CONTROL_ANCHOR="$PILOT/control/states/step_000020_t3.020.npy"
PURITY_PREVIOUS="$PILOT/purity64/states/step_000019_t3.019.npy"
PURITY_ANCHOR="$PILOT/purity64/states/step_000020_t3.020.npy"

mkdir -p "$OUTPUT"
exec > >(tee -a "$OUTPUT/controller.log") 2>&1
printf '%s\n' "$$" > "$OUTPUT/controller.pid"
trap 'status=$?; printf "finished=%s exit_status=%s\n" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$status" > "$OUTPUT/last_exit.txt"' EXIT

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROOT/src"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "$ROOT"
test "$(shasum -a 256 "$CONTROL_PREVIOUS" | awk '{print $1}')" = 23e6d0dc8b51b4c22bd5fce2188ec1a1659890c7dd55be4651ca577b7454fa1b
test "$(shasum -a 256 "$CONTROL_ANCHOR" | awk '{print $1}')" = 6a58feeb299330ebfba32d3db800564b978da7f5f028146590ec99c8a16c2668
test "$(shasum -a 256 "$PURITY_PREVIOUS" | awk '{print $1}')" = fd15f83cf390030fe11cd8834f93c5f2dc288d8005320a01629645f2a4259439
test "$(shasum -a 256 "$PURITY_ANCHOR" | awk '{print $1}')" = 60d3c5013d53f9ee584c56b64c167b876bafdf0f4b124fa3d35416f2acba2004

COMMON=(
  --start-time 3.02
  --steps 180
  --accept-cost 3e-16
  --fit-cost-target 1e-17
  --fibre-cost-target 1e-22
  --purity-step 50
  --purity-minimum-step 1e-8
  --purity-max-iterations 2000
  --purity-tracking-iterations 64
  --purity-full-every 1000000
  --purity-gradient-tolerance 2e-8
  --purity-relative-tolerance 1e-12
  --purity-relative-patience 25
  --purity-minimum-full-iterations 200
  --purity-numerical-gradient-ceiling 1e-5
  --jacobian-workers 4
  --pause-after-seconds 14400
  --quiet-purity
)

printf 'started=%s stage=control\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
"$PYTHON" scripts/run_buffered_purity_l4_production.py \
  --output-dir "$OUTPUT/control" \
  --previous-A "$CONTROL_PREVIOUS" \
  --anchor-A "$CONTROL_ANCHOR" \
  --mode control \
  "${COMMON[@]}"

printf 'started=%s stage=purity64\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
"$PYTHON" scripts/run_buffered_purity_l4_production.py \
  --output-dir "$OUTPUT/purity64" \
  --previous-A "$PURITY_PREVIOUS" \
  --anchor-A "$PURITY_ANCHOR" \
  --mode purity \
  --skip-anchor-purity \
  "${COMMON[@]}"

printf 'completed=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
