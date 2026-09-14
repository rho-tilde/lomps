#!/bin/bash -l

# Fixed-iteration scaling check for the t=1.5 L=4,D=12 purity anchor.
# Every task owns four cores; only the explicit J4 construction is varied.

#$ -N lomps-p8-scale
#$ -l h_rt=0:30:0
#$ -l mem=3G
#$ -l tmpfs=3G
#$ -pe smp 4
#$ -ac allow=D
#$ -t 1-3
#$ -tc 3
#$ -cwd
#$ -j y
#$ -o /home/ucancwi/Scratch/lomps-jobs/logs
#$ -m ea
#$ -M ucancwi@ucl.ac.uk

set -euo pipefail

readonly LOMPS_ROOT=/home/ucancwi/Scratch/lomps-buffered-purity-20260914
readonly PYTHON=/home/ucancwi/Scratch/lomps-acfc096/.conda-py312-exact/bin/python
readonly INPUT_ROOT="$LOMPS_ROOT/data/production/yplus_l4_d12_t1p500_20260914"
readonly OUTPUT_ROOT=/home/ucancwi/Scratch/lomps-jobs/benchmarks/buffered_purity_l4_anchor_scaling_20260914
readonly EXPECTED_PREVIOUS_SHA256=657504f714e775ee8ed06df30d43c40fb2e232ca3798b8252a59fac46e8af1ca
readonly EXPECTED_ANCHOR_SHA256=127a87da9aee3200b8de88001a6f25ea2208abe5df8829572ab2b938d85ad6b7

case "${SGE_TASK_ID:?}" in
  1) readonly JACOBIAN_WORKERS=1 ;;
  2) readonly JACOBIAN_WORKERS=2 ;;
  3) readonly JACOBIAN_WORKERS=4 ;;
  *) printf 'Unexpected SGE_TASK_ID=%s\n' "$SGE_TASK_ID" >&2; exit 2 ;;
esac

readonly RUN_DIR="$OUTPUT_ROOT/jacobian_workers_${JACOBIAN_WORKERS}_job_${JOB_ID}"

module purge
export LANG=C
export LC_ALL=C
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$LOMPS_ROOT/src"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p "$OUTPUT_ROOT"
cd "$LOMPS_ROOT"

test "$(sha256sum "$INPUT_ROOT/previous_t1p499_D12.npy" | awk '{print $1}')" = "$EXPECTED_PREVIOUS_SHA256"
test "$(sha256sum "$INPUT_ROOT/anchor_t1p500_D12.npy" | awk '{print $1}')" = "$EXPECTED_ANCHOR_SHA256"

printf 'commit=%s\n' "$(cat DEPLOYED_COMMIT)"
printf 'host=%s slots=%s jacobian_workers=%s\n' \
  "$(hostname)" "${NSLOTS:-unset}" "$JACOBIAN_WORKERS"

set +e
"$PYTHON" scripts/run_buffered_purity_l4_production.py \
  --output-dir "$RUN_DIR" \
  --previous-A "$INPUT_ROOT/previous_t1p499_D12.npy" \
  --anchor-A "$INPUT_ROOT/anchor_t1p500_D12.npy" \
  --mode purity \
  --start-time 1.5 \
  --steps 1 \
  --accept-cost 3e-16 \
  --fit-cost-target 1e-18 \
  --purity-step 50 \
  --purity-max-iterations 50 \
  --purity-gradient-tolerance 0 \
  --purity-relative-tolerance 0 \
  --purity-relative-patience 1000 \
  --jacobian-workers "$JACOBIAN_WORKERS" \
  --quiet-purity
run_status=$?
set -e

# The deliberately capped search should stop at maximum_iterations.  Retain
# its complete compact history and turn any other outcome into a failed job.
"$PYTHON" - "$RUN_DIR" "$run_status" <<'PY'
import json
from pathlib import Path
import sys
import numpy as np

root = Path(sys.argv[1])
run_status = int(sys.argv[2])
manifest = json.loads((root / "manifest.json").read_text())
history = np.load(root / "purity_history" / "step_000000.npz")
status = str(history["status"])
if status != "maximum_iterations" or run_status == 0:
    raise SystemExit(
        f"unexpected capped-search outcome: process={run_status}, "
        f"purity={status}, manifest={manifest['status']}"
    )
elapsed = np.asarray(history["elapsed_seconds"], dtype=float)
print(
    json.dumps(
        {
            "purity_status": status,
            "iterations": int(history["accepted_steps"]),
            "initial_purity": float(history["initial_purity"]),
            "final_purity": float(history["final_purity"]),
            "final_primary_cost": float(history["final_primary_cost"]),
            "final_null_gradient_norm": float(
                history["final_null_gradient_norm"]
            ),
            "sum_iteration_seconds": float(elapsed.sum()),
            "median_iteration_seconds": float(np.median(elapsed)),
        },
        sort_keys=True,
    )
)
PY
