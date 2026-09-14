#!/bin/bash -l

# Replay the strict t=3 fixed-rho4 fibre search with terminal tensor saving.

#$ -N lomps-p8-t3r
#$ -l h_rt=1:00:0
#$ -l mem=3G
#$ -l tmpfs=3G
#$ -pe smp 4
#$ -ac allow=D
#$ -cwd
#$ -j y
#$ -o /home/ucancwi/Scratch/lomps-jobs/logs
#$ -m ea
#$ -M ucancwi@ucl.ac.uk

set -euo pipefail

readonly LOMPS_ROOT=/home/ucancwi/Scratch/lomps-buffered-purity-20260914
readonly PYTHON=/home/ucancwi/Scratch/lomps-acfc096/.conda-py312-exact/bin/python
readonly INPUT_ROOT="$LOMPS_ROOT/data/production/yplus_l4_d12_t3p000_20260914"
readonly RUN_DIR=/home/ucancwi/Scratch/lomps-jobs/runs/buffered_purity_l4_t3_terminal_tensor_20260914
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

if [[ -e "$RUN_DIR" ]]; then
  printf 'Output path already exists; refusing to overwrite: %s\n' "$RUN_DIR" >&2
  exit 2
fi

{
  printf 'job_id=%s\n' "$JOB_ID"
  printf 'host=%s\n' "$(hostname)"
  printf 'slots=%s\n' "${NSLOTS:-unset}"
  printf 'commit=%s\n' "$(cat DEPLOYED_COMMIT)"
  printf 'jacobian_workers=4 blas_threads=1\n'
  printf 'started=%s\n' "$(date --iso-8601=seconds)"
  "$PYTHON" -c 'import platform,sys,numpy,scipy; print("platform="+platform.platform()); print("python="+sys.version.replace("\n"," ")); print("numpy="+numpy.__version__); print("scipy="+scipy.__version__)'
} | tee "$ENVIRONMENT_RECORD"

/usr/bin/time -v "$PYTHON" scripts/run_buffered_purity_l4_production.py \
  --output-dir "$RUN_DIR" \
  --previous-A "$INPUT_ROOT/previous_t2p999_D12.npy" \
  --anchor-A "$INPUT_ROOT/anchor_t3p000_D12.npy" \
  --mode purity \
  --start-time 3.0 \
  --steps 0 \
  --accept-cost 3e-16 \
  --fit-cost-target 1e-17 \
  --fibre-cost-target 1e-22 \
  --purity-step 50 \
  --purity-minimum-step 1e-8 \
  --purity-max-iterations 2000 \
  --purity-tracking-iterations 0 \
  --purity-full-every 1 \
  --purity-gradient-tolerance 2e-8 \
  --purity-relative-tolerance 1e-12 \
  --purity-relative-patience 25 \
  --purity-minimum-full-iterations 200 \
  --purity-numerical-gradient-ceiling 1e-5 \
  --jacobian-workers 4 \
  --quiet-purity
