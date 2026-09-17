# LOMPS

LOMPS (local optimization of matrix-product states) approximates infinite-system
quantum dynamics by matching a finite reduced density matrix after each time
step. The closure is a uniform, left-canonical matrix-product state (uMPS), so
the calculation works directly in the thermodynamic limit.

The package provides:

- finite-block reduced density matrices and analytic derivatives;
- second-order Ising-quench protocols for even and odd matching windows;
- gauge-aware Levenberg--Marquardt optimization on the MPS manifold;
- dense and matrix-free optimization backends;
- product-state and lower-bond-dimension initialization;
- checkpointed, resumable trajectories with recorded optimizer diagnostics;
- compact reference data for integrable and nonintegrable benchmarks.

## Installation

LOMPS requires Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

The dependency versions used for regression testing are listed in
`requirements-tested.txt`.

## Quick start

Run the compact ten-step example:

```bash
python examples/ten_step_quench.py
```

For a checkpointed trajectory, use `lomps-run`:

```bash
lomps-run \
  --initial-A data/nonintegrable_d10_t3p235.npy \
  --output-dir runs/d10 \
  --bond-dimension 10 \
  --block-length 4 \
  --fixed-point-solver dense \
  --accept-cost 3e-16 \
  --steps 100 \
  --base-time 3.235
```

Create `runs/d10/PAUSE` to stop cleanly between time steps. Remove the file and
repeat the command with `--resume` to continue.

## Algorithm

For a current uMPS tensor `A`, LOMPS constructs the locally evolved target
state on an `L`-site window and finds the next tensor `B` by minimizing

```text
C(B) = 1/2 ||rho_L(B) - target_L(A)||_F^2.
```

The optimizer removes infinitesimal MPS gauge directions, evaluates the RDM
derivative on the remaining tangent space, and retracts accepted steps to the
left-canonical manifold. Even windows use a symmetric `L+4`-site light cone.
Odd windows use an `L+5`-site light cone and average the two parity-related
reductions.

See [docs/algorithm.md](docs/algorithm.md) for the mathematical outline.

## Initial states

`--initial-A` accepts

- a product vector with shape `(d,)`;
- an MPS tensor with shape `(d,D,D)`; or
- a singleton tensor batch with shape `(1,d,D,D)`.

MPS tensors use `(physical,left,right)` order and must be left canonical. Keyed
`.npz` files and historical `(left,physical,right)` arrays can be loaded with
`--initial-key` and `--initial-layout`. `lomps-convert-tensor` converts legacy
inputs and writes a provenance sidecar.

If the input bond dimension is smaller than the requested trajectory bond
dimension, LOMPS builds the first physical target from the original input and
uses a separate lifted tensor only as the optimizer seed. The default first
step combines fixed-target conjugate gradient with an LM polish. Later time
steps use the accepted high-dimensional trajectory tensor.

See [docs/product_starts.md](docs/product_starts.md) for commands and parameter
recommendations.

## Large-window calculations

Dense LM is the reference backend for the established `L=4,D=12` and
`L=5,D=20/21` regimes. For larger problems, select the matrix-free backend:

```bash
lomps-run \
  --optimizer matrix-free-lm \
  --matrix-free-krylov-preconditioner right-fixed-point-stiefel \
  --initial-A data/your_tensor.npy \
  --output-dir runs/matrix_free \
  --bond-dimension 40 \
  --block-length 6 \
  --steps 100
```

This backend applies the RDM Jacobian and its adjoint through local
contractions rather than storing the full Jacobian. Krylov tolerances, budgets,
and contraction counts are written to the run directory. Dense and
matrix-free runs use the same checkpoint and resume format.

For trajectories that increase the bond dimension only when the current
manifold cannot accept the next update, see
[docs/adaptive_bond_runs.md](docs/adaptive_bond_runs.md). The safeguards used
for large dense Jacobians are described in
[docs/large_jacobian_blas_safety.md](docs/large_jacobian_blas_safety.md).

## Integrable TFIM reference

The repository contains VUMPS ground-state tensors for the quench

```text
H0 = -sum_j Z_j Z_(j+1) - 1.5 sum_j X_j
H1 = -sum_j Z_j Z_(j+1) - 0.2 sum_j X_j.
```

The named `integrable-tfim` protocol fixes the historical Hamiltonian and
Trotter conventions. Exact production commands, tensor provenance, and resume
instructions are collected in
[docs/integrable_benchmarks.md](docs/integrable_benchmarks.md).

Generate the independent free-fermion reference with

```bash
python examples/generate_free_fermion_reference.py
```

The output contains the transverse magnetization and local Hilbert--Schmidt
Loschmidt signature on the production time grid.

## Output and conventions

Trajectory tensors are stored in `states.npy`; aligned right transfer fixed
points are stored in `right_fixed_points.npy`. Metadata and step ledgers record
the resolved protocol, optimizer settings, costs, restart decisions, and wall
times. The target remains fixed during every restart of a physical time step.

The main conventions are

```text
A.shape = (physical, left, right)
sum_s A[s]^dagger A[s] = I
C = 1/2 ||rho_L(A_fit) - rho_target||_F^2.
```

## Tests

Run the test suite with

```bash
python -m unittest discover -s tests
```

The integrable production path has a separate three-step regression:

```bash
python -m unittest discover -s tests -p 'test_integrable.py'
```

## Documentation

- [Algorithm](docs/algorithm.md)
- [Product-state starts](docs/product_starts.md)
- [Adaptive bond-dimension trajectories](docs/adaptive_bond_runs.md)
- [Integrable TFIM benchmark](docs/integrable_benchmarks.md)
- [Large-Jacobian BLAS safety](docs/large_jacobian_blas_safety.md)
