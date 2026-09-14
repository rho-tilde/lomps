#!/bin/zsh
set -euo pipefail

readonly ROOT="/Users/phys2259/Documents/Geometry of MPS RDM/mps_rdm_tangent/LOMPS"
readonly PYTHON="/Users/phys2259/opt/anaconda3/envs/qnexus-env/bin/python"
readonly RUN_DIR="${ROOT}/runs/nonintegrable_l4_d12_buffered_purity_converged_t5p000_to_t5p010_20260911"

cd "${ROOT}"

if [[ -f "${RUN_DIR}/manifest.json" ]]; then
  stored_pid="$(${PYTHON} -c 'import json,sys; print(json.load(open(sys.argv[1])).get("pid", ""))' "${RUN_DIR}/manifest.json")"
  if [[ -n "${stored_pid}" ]] && /bin/kill -0 "${stored_pid}" 2>/dev/null; then
    print -u2 -r -- "Run is already active with PID ${stored_pid}."
    exit 2
  fi
  resume_args=(--resume)
else
  resume_args=()
fi

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${ROOT}/src"

exec /usr/bin/caffeinate -dimsu "${PYTHON}" -B \
  scripts/run_buffered_purity_l4_converged_trajectory.py \
  --output-dir "${RUN_DIR}" \
  --start-time 5.0 \
  --steps 10 \
  --accept-cost 3e-16 \
  --purity-step 50.0 \
  --purity-max-iterations 2000 \
  --purity-gradient-tolerance 1e-8 \
  --purity-relative-tolerance 1e-12 \
  --jacobian-workers 1 \
  "${resume_args[@]}"
