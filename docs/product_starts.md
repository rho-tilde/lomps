# Product-State Starts and Student Runbook

This is the practical launch guide for product-state LOMPS trajectories. The
goal is reproducible production runs plus fast local audits before sending jobs
to a cluster.

## Default launch protocol

For an input product vector or any lower-bond input tensor, LOMPS separates the
physical source from the optimizer seed:

1. Load `--initial-A`.
   - Shape `(d,)` is treated as a product vector.
   - Shape `(d,D,D)` is treated as a left-canonical uMPS tensor.
   - Shape `(1,d,D,D)` is accepted as a singleton saved batch.
2. Build the first physical target from the original source tensor:
   `target_L(A_source)`.
3. Build a trajectory-D optimizer seed.
   - Default: `--initial-seed-mode embedding`.
   - The source tensor is embedded into the upper-left virtual block.
   - Deterministic complex noise is added only outside the old virtual block.
   - The result is QR-projected back to the left-canonical manifold.
4. Fit the first target with `--first-step-optimizer cg-lm`.
   - The CG stage is the historical fixed-target QDMT launch.
   - The LM stage polishes the same frozen first target.
5. After the first accepted tensor, later targets are built from the accepted
   trajectory tensor and its cached right fixed point.

The product-circuit seed is still available with `--initial-seed-mode circuit`
or `--initial-seed-mode auto`, but it is not the production default. It was
useful diagnostically but less robust in the product-start audits.

## Recommended product-state command

Create the historical `+y` quench vector:

```bash
python examples/product_state_tensor.py --state y --output runs/initial_y.npy
```

Run a short, healthy L=2 local audit trajectory:

```bash
lomps-run \
  --initial-A runs/initial_y.npy \
  --output-dir runs/l2_d4_y_dt1e-3 \
  --block-length 2 \
  --bond-dimension 4 \
  --steps 1000 \
  --base-time 0.0 \
  --delta-t 1e-3 \
  --embedding-noise-amplitude 1e-6 \
  --first-step-accept-cost 3e-16 \
  --accept-cost 1e-15 \
  --first-step-cg-seconds 100 \
  --checkpoint-every 25
```

For L=5/D=20 or larger cluster runs, use the same structure but give the first
step and restart protocol more time:

```bash
lomps-run \
  --initial-A runs/initial_y.npy \
  --output-dir runs/l5_d20_y_dt1e-3 \
  --block-length 5 \
  --bond-dimension 20 \
  --steps 1000 \
  --base-time 0.0 \
  --delta-t 1e-3 \
  --embedding-noise-amplitude 1e-6 \
  --first-step-accept-cost 3e-16 \
  --accept-cost 1e-14 \
  --first-step-cg-seconds 900 \
  --perturb-amplitudes 0.1,0.3,0.6,1.0 \
  --perturbations-per-amplitude 3 \
  --random-restarts 0 \
  --checkpoint-every 10 \
  --run-time-limit 43200
```

Resume with the same command plus `--resume`. Create `runs/.../PAUSE` to stop
cleanly between timesteps.

## Parameter defaults and recommendations

| Parameter | Code default | Production recommendation |
| --- | ---: | --- |
| `--block-length` | `4` | Choose the intended QDMT window L. Odd L uses parity-averaged targets. |
| `--bond-dimension` | input D | Set explicitly for product starts. |
| `--delta-t` | `1e-3` | Use `1e-3` for reference runs; halve it for Trotter checks. |
| `--initial-seed-mode` | `embedding` | Keep default unless deliberately testing circuit seeds. |
| `--embedding-noise-amplitude` | `1e-6` | Good product-start compromise from audits. |
| `--embedding-seed` | `104729` | Keep fixed for reproducibility; vary only in audits. |
| `--first-step-optimizer` | `cg-lm` | Keep default for product or lower-D starts. |
| `--first-step-accept-cost` | `3e-16` | Keep strict for A1 fits unless a large run needs a manual relaxation. |
| `--accept-cost` | `1e-14` | Use `1e-15` for stricter continuation; `3e-16` can chase the floor. |
| `--fixed-point-solver` | `dense` | Use `dense` for reproducibility; `fast` for exploratory speed checks. |
| `--target-contraction` | `tensor` | Keep default; `dense` is a regression path. |
| `--strict-retry` | enabled | Useful with sensible tolerances; can waste time near machine floor. |
| `--checkpoint-every` | `25` | Use smaller values for long cluster runs. |

Cost convention:

```text
C = 1/2 ||rho_L(A_fit) - rho_target||_F^2
```

So the RDM Frobenius residual is `sqrt(2*C)` up to small numerical differences
from later diagnostic fixed-point solves.

## Audit scripts

First-step product-start grid:

```bash
python examples/product_start_audit.py \
  --mode first-step \
  --states x,y,z \
  --block-lengths 2,3,4 \
  --bond-dimensions 4,6,8,12 \
  --delta-ts 1e-3 \
  --embedding-noises 1e-8,1e-6,1e-4 \
  --cg-seconds 50 \
  --output-csv runs/audits/product_first_step_grid.csv
```

Step-2 tolerance check:

```bash
python examples/product_start_audit.py \
  --mode step2 \
  --states y \
  --block-lengths 2 \
  --bond-dimensions 4,6,8 \
  --delta-ts 1e-3,5e-4 \
  --embedding-noises 1e-8,1e-6,1e-4 \
  --first-step-accept-cost 3e-16 \
  --accept-cost 1e-15 \
  --relaxed-accept-cost 1e-14 \
  --cg-seconds 100 \
  --output-csv runs/audits/product_step2_grid.csv
```

The audit CSV records seed diagnostics, CG/LM status, costs, RDM Frobenius
errors, final transfer gap, final right-fixed-point minimum eigenvalue, and
wall time.

## Audit snapshot, 2026-07-15

The broad first-step audit over 90 cases passed with `cg-lm`.

- Maximum polish cost: `5.27e-15`.
- Median polish cost: `2.55e-16`.
- Maximum final RDM Frobenius error: `1.03e-7`.
- Worst final transfer gap: `4.16e-2`.
- Median final transfer gap: `2.72e-1`.

Noise tradeoff:

- `1e-8` is accurate but can leave a nearly rank-deficient right fixed point.
- `1e-6` is the best current default compromise.
- `1e-4` often improves seed conditioning but tends to slow the first solve.

The L=2/D=4 long checks were healthy:

- `dt=1e-3`, 1000 steps to `t=1`: completed with no restarts; max cost
  `9.99e-16`; final transfer gap `0.421`.
- `dt=5e-4`, 2000 steps to `t=1`: completed with no restarts; max cost
  `1.00e-15`; final transfer gap `0.415`.

The main warning from the audits is the continuation tolerance. A strict
`--accept-cost 3e-16` can trigger expensive retries at step 2 for L=2/D=6 and
L=2/D=8, where the optimizer often stops by gradient tolerance around
`1e-15`. This is not a failed physical launch; it is mostly a numerical-floor
issue. The CLI therefore defaults to `--first-step-accept-cost 3e-16` and
`--accept-cost 1e-14`. Use `--accept-cost 1e-15` for stricter continuation.

## Observables and energy density

Use `lomps.observables.local_expectations` for local observables and
`lomps.gates.two_site_hamiltonian_tfim` for the quench energy density:

```python
import numpy as np

from lomps.gates import two_site_hamiltonian_tfim
from lomps.observables import PAULI_X, PAULI_Y, PAULI_Z, local_expectations

A = np.load("runs/l2_d4_y_dt1e-3/states.npy", mmap_mode="r")[100]
H2 = two_site_hamiltonian_tfim(g=1.05, h=-0.5, J=-1.0)
values = local_expectations(
    A,
    {"sx": PAULI_X, "sy": PAULI_Y, "sz": PAULI_Z, "energy": H2},
    solver="dense",
)
print(values)
```

The energy value is `Tr(rho_2 H2)` for the two-site Hamiltonian convention used
by the LOMPS quench protocol.
