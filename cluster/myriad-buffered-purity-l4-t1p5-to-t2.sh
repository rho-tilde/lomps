#!/bin/bash -l

# Paired 10-hour comparison from the same high-accuracy Y+ L=4,D=12.
# anchor: task 1 is the primary-only control, task 2 adds substantial P8 search.

#$ -N lomps-p8-l4
#$ -l h_rt=11:00:0
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
readonly INPUT_ROOT="$LOMPS_ROOT/data/production/yplus_l4_d12_t1p500_20260914"
readonly OUTPUT_ROOT=/home/ucancwi/Scratch/lomps-jobs/runs/buffered_purity_l4_t1p5_to_t2_20260914
readonly PAUSE_AFTER_SECONDS=36000
readonly EXPECTED_PREVIOUS_SHA256=657504f714e775ee8ed06df30d43c40fb2e232ca3798b8252a59fac46e8af1ca
readonly EXPECTED_ANCHOR_SHA256=127a87da9aee3200b8de88001a6f25ea2208abe5df8829572ab2b938d85ad6b7

case "${SGE_TASK_ID:?}" in
  1) readonly MODE=control ;;
  2) readonly MODE=purity ;;
  *) printf 'Unexpected SGE_TASK_ID=%s\n' "$SGE_TASK_ID" >&2; exit 2 ;;
esac

readonly RUN_DIR="$OUTPUT_ROOT/${MODE}_L4_D12_t1p500_to_t2p000"
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

test "$(sha256sum "$INPUT_ROOT/previous_t1p499_D12.npy" | awk '{print $1}')" = "$EXPECTED_PREVIOUS_SHA256"
test "$(sha256sum "$INPUT_ROOT/anchor_t1p500_D12.npy" | awk '{print $1}')" = "$EXPECTED_ANCHOR_SHA256"

{
  printf 'job_id=%s\n' "$JOB_ID"
  printf 'task_id=%s\n' "$SGE_TASK_ID"
  printf 'mode=%s\n' "$MODE"
  printf 'host=%s\n' "$(hostname)"
  printf 'slots=%s\n' "${NSLOTS:-unset}"
  printf 'commit=%s\n' "$(cat DEPLOYED_COMMIT)"
  printf 'jacobian_workers=4 blas_threads=1\n'
  printf 'started=%s\n' "$(date --iso-8601=seconds)"
  "$PYTHON" -c 'import platform,sys,numpy,scipy; print("platform="+platform.platform()); print("python="+sys.version.replace("\n"," ")); print("numpy="+numpy.__version__); print("scipy="+scipy.__version__)'
} | tee "$ENVIRONMENT_RECORD"

args=(
  --output-dir "$RUN_DIR"
  --previous-A "$INPUT_ROOT/previous_t1p499_D12.npy"
  --anchor-A "$INPUT_ROOT/anchor_t1p500_D12.npy"
  --mode "$MODE"
  --start-time 1.5
  --steps 500
  --accept-cost 3e-16
  --fit-cost-target 1e-17
  --purity-step 50
  --purity-max-iterations 2000
  --purity-tracking-iterations 64
  --purity-full-every 250
  --purity-gradient-tolerance 2e-8
  --purity-relative-tolerance 1e-12
  --purity-relative-patience 25
  --purity-minimum-full-iterations 200
  --purity-numerical-gradient-ceiling 1e-5
  --jacobian-workers 4
  --pause-after-seconds "$PAUSE_AFTER_SECONDS"
  --quiet-purity
)

if [[ -f "$RUN_DIR/manifest.json" ]]; then
  status="$($PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$RUN_DIR/manifest.json")"
  case "$status" in
    completed) printf '%s is already complete\n' "$RUN_DIR"; exit 0 ;;
    failed) printf '%s is failed; refusing automatic retry\n' "$RUN_DIR" >&2; exit 2 ;;
    *) args=(--resume "${args[@]}") ;;
  esac
elif [[ -e "$RUN_DIR" ]]; then
  printf 'Fresh output path exists without manifest: %s\n' "$RUN_DIR" >&2
  exit 2
fi

/usr/bin/time -v "$PYTHON" scripts/run_buffered_purity_l4_production.py "${args[@]}"

cp "$ENVIRONMENT_RECORD" "$RUN_DIR/cluster_environment_job_${JOB_ID}_task_${SGE_TASK_ID}.txt"
printf 'finished=%s\n' "$(date --iso-8601=seconds)"
