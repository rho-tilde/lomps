#!/bin/bash -l

# Matched strict fixed-rho4 anchor searches at t=1.5 and t=3.0.

#$ -N lomps-p8-a13
#$ -l h_rt=4:00:0
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
readonly OUTPUT_ROOT=/home/ucancwi/Scratch/lomps-jobs/runs/buffered_purity_l4_strict_anchors_t1p5_t3_20260914

case "${SGE_TASK_ID:?}" in
  1)
    readonly LABEL=t1p500
    readonly START_TIME=1.5
    readonly INPUT_ROOT="$LOMPS_ROOT/data/production/yplus_l4_d12_t1p500_20260914"
    readonly PREVIOUS_NAME=previous_t1p499_D12.npy
    readonly ANCHOR_NAME=anchor_t1p500_D12.npy
    readonly EXPECTED_PREVIOUS_SHA256=657504f714e775ee8ed06df30d43c40fb2e232ca3798b8252a59fac46e8af1ca
    readonly EXPECTED_ANCHOR_SHA256=127a87da9aee3200b8de88001a6f25ea2208abe5df8829572ab2b938d85ad6b7
    ;;
  2)
    readonly LABEL=t3p000
    readonly START_TIME=3.0
    readonly INPUT_ROOT="$LOMPS_ROOT/data/production/yplus_l4_d12_t3p000_20260914"
    readonly PREVIOUS_NAME=previous_t2p999_D12.npy
    readonly ANCHOR_NAME=anchor_t3p000_D12.npy
    readonly EXPECTED_PREVIOUS_SHA256=d829feaa1af19ef2338a019d20fb48f5713eeea91a033f14c7c51ea0baf5b52f
    readonly EXPECTED_ANCHOR_SHA256=fa3ca25dec93ca94e1b5735941f5432923e6f1bcc63ac1a6b2a7ad8d36be79fc
    ;;
  *) printf 'Unexpected SGE_TASK_ID=%s\n' "$SGE_TASK_ID" >&2; exit 2 ;;
esac

readonly RUN_DIR="$OUTPUT_ROOT/strict_fibre_${LABEL}_L4_D12"
readonly ENVIRONMENT_RECORD="${RUN_DIR}.cluster_environment_job_${JOB_ID}_task_${SGE_TASK_ID}.txt"

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

test "$(sha256sum "$INPUT_ROOT/$PREVIOUS_NAME" | awk '{print $1}')" = "$EXPECTED_PREVIOUS_SHA256"
test "$(sha256sum "$INPUT_ROOT/$ANCHOR_NAME" | awk '{print $1}')" = "$EXPECTED_ANCHOR_SHA256"

if [[ -e "$RUN_DIR" ]]; then
  printf 'Output path already exists; refusing to overwrite: %s\n' "$RUN_DIR" >&2
  exit 2
fi

{
  printf 'job_id=%s\n' "$JOB_ID"
  printf 'task_id=%s\n' "$SGE_TASK_ID"
  printf 'anchor=%s\n' "$LABEL"
  printf 'host=%s\n' "$(hostname)"
  printf 'slots=%s\n' "${NSLOTS:-unset}"
  printf 'commit=%s\n' "$(cat DEPLOYED_COMMIT)"
  printf 'jacobian_workers=4 blas_threads=1\n'
  printf 'started=%s\n' "$(date --iso-8601=seconds)"
  "$PYTHON" -c 'import platform,sys,numpy,scipy; print("platform="+platform.platform()); print("python="+sys.version.replace("\n"," ")); print("numpy="+numpy.__version__); print("scipy="+scipy.__version__)'
} | tee "$ENVIRONMENT_RECORD"

/usr/bin/time -v "$PYTHON" scripts/run_buffered_purity_l4_production.py \
  --output-dir "$RUN_DIR" \
  --previous-A "$INPUT_ROOT/$PREVIOUS_NAME" \
  --anchor-A "$INPUT_ROOT/$ANCHOR_NAME" \
  --mode purity \
  --start-time "$START_TIME" \
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

cp "$ENVIRONMENT_RECORD" "$RUN_DIR/cluster_environment_job_${JOB_ID}_task_${SGE_TASK_ID}.txt"
printf 'finished=%s\n' "$(date --iso-8601=seconds)"
