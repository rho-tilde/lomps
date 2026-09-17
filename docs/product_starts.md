# Product-state and lower-bond-dimension starts

This guide covers trajectories whose physical initial state has a smaller bond
dimension than the requested LOMPS trajectory.

## Source and optimizer seed

LOMPS treats the physical source and optimizer seed separately:

1. Load the state supplied through `--initial-A`.
2. Build the first evolved target from that original state.
3. Construct a left-canonical optimizer seed at the requested bond dimension.
4. Fit the fixed first target with conjugate gradient followed by an LM polish.
5. Use the first accepted high-dimensional tensor as the source for later time
   steps.

This procedure preserves the exact product or low-dimensional first target.
The lifted seed only initializes the variational fit.

The default `embedding` seed places the source tensor in the upper-left virtual
block, adds deterministic noise in the new virtual subspace, and restores left
canonical form by QR projection. The product-circuit seed remains available
through `--initial-seed-mode circuit`, but it is intended for controlled
comparisons rather than routine production.

## Basic product-state run

Create the `+y` product vector:

```bash
python examples/product_state_tensor.py --state y --output runs/initial_y.npy
```

Run a short `L=2,D=4` trajectory:

```bash
lomps-run \
  --initial-A runs/initial_y.npy \
  --output-dir runs/l2_d4_y_dt1e-3 \
  --block-length 2 \
  --bond-dimension 4 \
  --steps 1000 \
  --base-time 0 \
  --delta-t 1e-3 \
  --embedding-noise-amplitude 1e-6 \
  --first-step-accept-cost 3e-16 \
  --accept-cost 1e-15 \
  --first-step-cg-seconds 100 \
  --checkpoint-every 25
```

For a larger cluster run, keep the same structure and increase the first-step
budget and restart coverage. For example:

```bash
lomps-run \
  --initial-A runs/initial_y.npy \
  --output-dir runs/l5_d20_y_dt1e-3 \
  --block-length 5 \
  --bond-dimension 20 \
  --steps 1000 \
  --base-time 0 \
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

Create `runs/.../PAUSE` to stop between completed time steps. Remove it and
repeat the same command with `--resume` to continue.

## Main controls

| Parameter | Typical choice | Meaning |
| --- | --- | --- |
| `--block-length` | chosen window `L` | Size of the matched local RDM. |
| `--bond-dimension` | set explicitly | Bond dimension of the fitted trajectory. |
| `--delta-t` | `1e-3` | Reference time step; halve it for Trotter checks. |
| `--initial-seed-mode` | `embedding` | Construction of the high-dimensional first seed. |
| `--embedding-noise-amplitude` | `1e-6` | Deterministic noise added in the new virtual subspace. |
| `--first-step-optimizer` | `cg-lm` | Fixed-target CG followed by LM polishing. |
| `--first-step-accept-cost` | `3e-16` | Acceptance threshold for the first fit. |
| `--accept-cost` | `1e-14` or `1e-15` | Threshold for recurrent trajectory steps. |
| `--fixed-point-solver` | `dense` | Reproducible ansatz fixed-point calculation. |
| `--target-contraction` | `tensor` | Local tensor-axis target evolution. |

The cost convention is

```text
C = 1/2 ||rho_L(A_fit) - rho_target||_F^2,
```

so the Frobenius residual is `sqrt(2*C)`.

## External seeds and seed screening

Pass an existing high-dimensional optimizer seed with `--initial-seed-A`. The
first physical target is still built from `--initial-A`. If the supplied seed
must be lifted further, `--initial-seed-lift-noise-amplitude` controls the new
virtual subspace.

For less structured low-dimensional inputs, `--embedding-candidate-seeds`
accepts a comma-separated list of deterministic lift seeds. LOMPS screens each
candidate against the same first target and records the selected seed and the
complete screening table in the run metadata.

## Validation

The product-start tests cover product-vector loading, deterministic lifting,
first-target preservation, CG/LM fitting, checkpointing, and resume behavior:

```bash
python -m unittest discover -s tests -p 'test_evolution_protocol.py'
python -m unittest discover -s tests -p 'test_embedding.py'
```

For a broader parameter study, `examples/product_start_audit.py` can scan
initial states, window sizes, bond dimensions, time steps, and embedding-noise
levels. Its CSV output includes first-fit costs, RDM errors, transfer gaps, and
wall times.

## Observables

Local Pauli observables and the quench energy density can be evaluated from a
saved tensor with `lomps.observables.local_expectations`:

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

Here `energy` is `Tr(rho_2 H2)` for the two-site Hamiltonian convention used by
the nonintegrable Ising protocol.
