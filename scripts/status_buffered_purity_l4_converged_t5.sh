#!/bin/zsh
set -euo pipefail

readonly ROOT="/Users/phys2259/Documents/Geometry of MPS RDM/mps_rdm_tangent/LOMPS"
readonly RUN_DIR="${ROOT}/runs/nonintegrable_l4_d12_buffered_purity_converged_t5p000_to_t5p010_20260911"
readonly LOG="${RUN_DIR}.log"
readonly PYTHON="/Users/phys2259/opt/anaconda3/envs/qnexus-env/bin/python"

if [[ ! -f "${RUN_DIR}/manifest.json" ]]; then
  print -r -- "No manifest at ${RUN_DIR}."
  exit 1
fi

"${PYTHON}" -c '
import json, os, sys
d = json.load(open(sys.argv[1]))
pid = d.get("pid")
active = bool(pid) and os.path.exists(f"/proc/{pid}")
anchor = d.get("anchor") or {}
latest = d.get("latest_step") or {}
print(
    "status=%s stage=%s pid=%s completed=%s/%s time=%.3f"
    % (
        d.get("status"), d.get("current_stage"), pid,
        d.get("completed_steps", 0), d["config"]["target_steps"],
        d.get("current_time", d["config"]["start_time"]),
    )
)
if anchor:
    print(
        "anchor: status=%s iterations=%s P8=%.12g g_null=%.3e seconds=%.1f"
        % (
            anchor.get("status"), anchor.get("accepted_iterations"),
            anchor.get("final_purity"),
            anchor.get("terminal_null_gradient_norm"), anchor.get("seconds"),
        )
    )
if latest:
    print(
        "latest: status=%s iterations=%s P8=%.12g cost=%.3e seconds=%.1f"
        % (
            latest.get("purity_status"), latest.get("purity_iterations"),
            latest.get("purity_after"), latest.get("ordinary_fit_cost"),
            latest.get("step_seconds"),
        )
    )
' "${RUN_DIR}/manifest.json"

if [[ -f "${LOG}" ]]; then
  /usr/bin/tail -n 8 "${LOG}"
fi
