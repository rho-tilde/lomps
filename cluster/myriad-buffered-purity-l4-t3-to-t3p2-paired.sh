#!/bin/bash -l

# Paired Y+ L=4 trajectories from t=3 to 3.2:
# task 1 is ordinary LOMPS; task 2 performs 64 fixed-rho4 P8 updates per step.

#$ -N lomps-p8-traj
#$ -l h_rt=5:00:0
#$ -l mem=3G
#$ -l tmpfs=3G
#$ -pe smp 4
#$ -ac allow=D
#$ -t 1-2
#$ -tc 2
#$ -cwd
#$ -j y
#$ -o /home/ucancwi/Scratch/lomps-jobs/logs
#$ -m ea
#$ -M ucancwi@ucl.ac.uk

set -euo pipefail

readonly LOMPS_ROOT=/home/ucancwi/Scratch/lomps-buffered-purity-20260914
readonly PYTHON=/home/ucancwi/Scratch/lomps-acfc096/.conda-py312-exact/bin/python
readonly INPUT_ROOT="$LOMPS_ROOT/data/production/yplus_l4_d12_t3p000_20260914"
readonly FIBRE_ROOT="$LOMPS_ROOT/data/production/yplus_l4_d12_t3p000_fibre_job339072_20260914"
readonly OUTPUT_ROOT=/home/ucancwi/Scratch/lomps-jobs/runs/buffered_purity_l4_t3_to_t3p2_20260916
readonly TASK_ID=${SGE_TASK_ID:?SGE_TASK_ID is required}

case "$TASK_ID" in
  1)
    readonly MODE=control
    readonly LABEL=control
    readonly ANCHOR="$INPUT_ROOT/anchor_t3p000_D12.npy"
    ;;
  2)
    readonly MODE=purity
    readonly LABEL=purity64
    readonly ANCHOR="$FIBRE_ROOT/anchor_fibre_t3p000_D12.npy"
    ;;
  *)
    printf 'Unexpected task id: %s\n' "$TASK_ID" >&2
    exit 2
    ;;
esac

readonly RUN_DIR="$OUTPUT_ROOT/$LABEL"
readonly ENVIRONMENT_RECORD="${RUN_DIR}.cluster_environment_job_${JOB_ID}.txt"

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

cd "$LOMPS_ROOT"
test "$(sha256sum "$INPUT_ROOT/previous_t2p999_D12.npy" | awk '{print $1}')" = d829feaa1af19ef2338a019d20fb48f5713eeea91a033f14c7c51ea0baf5b52f
test "$(sha256sum "$INPUT_ROOT/anchor_t3p000_D12.npy" | awk '{print $1}')" = fa3ca25dec93ca94e1b5735941f5432923e6f1bcc63ac1a6b2a7ad8d36be79fc
test "$(sha256sum "$FIBRE_ROOT/anchor_fibre_t3p000_D12.npy" | awk '{print $1}')" = ad72284516cdf7ed84b9b8a0ad89325007192f1a9050e36c3d1411a2ea4fb5d1

mkdir -p "$OUTPUT_ROOT"
if [[ -e "$RUN_DIR" ]]; then
  printf 'Output path already exists; refusing to overwrite: %s\n' "$RUN_DIR" >&2
  exit 2
fi

{
  printf 'job_id=%s task_id=%s mode=%s\n' "$JOB_ID" "$TASK_ID" "$MODE"
  printf 'host=%s\n' "$(hostname)"
  printf 'slots=%s\n' "${NSLOTS:-unset}"
  printf 'commit=%s\n' "$(cat DEPLOYED_COMMIT)"
  printf 'jacobian_workers=4 blas_threads=1\n'
  printf 'started=%s\n' "$(date --iso-8601=seconds)"
  "$PYTHON" -c 'import platform,sys,numpy,scipy; print("platform="+platform.platform()); print("python="+sys.version.replace("\n"," ")); print("numpy="+numpy.__version__); print("scipy="+scipy.__version__)'
} | tee "$ENVIRONMENT_RECORD"

ARGS=(
  --output-dir "$RUN_DIR"
  --previous-A "$INPUT_ROOT/previous_t2p999_D12.npy"
  --anchor-A "$ANCHOR"
  --mode "$MODE"
  --start-time 3.0
  --steps 200
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
  --pause-after-seconds 16200
  --quiet-purity
)
if [[ "$MODE" == purity ]]; then
  ARGS+=(--skip-anchor-purity)
fi

/usr/bin/time -v "$PYTHON" scripts/run_buffered_purity_l4_production.py "${ARGS[@]}"
