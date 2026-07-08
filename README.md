# LOMPS

**LOMPS** (local optimization of matrix-product states) is a compact research
implementation of finite-window quantum dynamics closed by a uniform MPS.
The name is deliberately reminiscent of VUMPS; both methods work directly in
the thermodynamic limit while respecting MPS geometry.

The core implementation provides:

- left-canonical uniform-MPS tensors;
- finite-block reduced density matrices and analytic directional derivatives;
- the true MPS gauge tangent and its orthogonal complement;
- gauge-orthogonal Levenberg--Marquardt updates with polar retraction;
- reproducible dense fixed-point solves by default, with an explicit fast
  ARPACK option for exploratory runs;
- configurable second-order non-integrable Ising quench protocols, with
  symmetric even-`L` and parity-averaged odd-`L` local windows;
- historical-style analytic fixed-target CG for the first lifted product-state
  update, followed by an LM polish;
- resumable evolution with fixed-target multistart rescue and detailed logs.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

The initial release was regression-tested with the versions recorded in
`requirements-tested.txt`; newer compatible NumPy/SciPy versions may also be
used.

Run the compact D=12 example:

```bash
python examples/ten_step_quench.py
```

Run the checkpointed D=10 fixed-target restart benchmark:

```bash
lomps-run \
  --initial-A data/nonintegrable_d10_t3p235.npy \
  --output-dir runs/d10 \
  --bond-dimension 10 \
  --block-length 4 \
  --fixed-point-solver dense \
  --steps 100 \
  --base-time 3.235
```

Create `runs/d10/PAUSE` to pause between timesteps. Remove it and append
`--resume` to the same command to continue.

## Conventions

Tensors have shape `(physical, left, right)` and are left canonical:

```text
sum_s A[s]^dagger A[s] = I.
```

The optimizer cost is the half-squared Frobenius residual

```text
C = 1/2 ||rho_L(A_fit) - rho_target||_F^2.
```

During rescue, `rho_target` is computed once from the current physical state
and remains fixed across all restart seeds.

Initial states may be supplied either as a product vector with shape `(d,)`,
an MPS tensor with shape `(d,D,D)`, or a singleton batch `(1,d,D,D)`. Tensor
inputs must already be left-canonical. Use `--bond-dimension` to choose the
trajectory bond dimension. If the input has smaller bond dimension, LOMPS uses
the original low-D tensor to build the first physical target and separately
constructs a deterministic lifted left-canonical seed at the requested
trajectory bond dimension. This avoids treating a product state as if it were
already a healthy injective high-D tensor.

For product starts, the default `--initial-seed-mode embedding` follows the
historical QDMT start-point convention: the product tensor is embedded into
the upper-left virtual block, tiny deterministic noise is added only in the
new virtual subspace, and the result is QR-projected back to the canonical
manifold. This lifted tensor is only the optimizer seed; the first target RDM
is still built from the original low-D product source. The default lift noise
is `--embedding-noise-amplitude 1e-8`.

When the input source has lower bond dimension than the trajectory tensor,
the default `--first-step-optimizer cg-lm` fits this first frozen target with
the historical-style analytic fixed-target Grassmann CG, then polishes the
same target with the gauge-orthogonal LM optimizer. This is the recommended
product-state launch path. Use `--first-step-cg-verbose` to print the CG
iteration trace for long first-step solves.

For less structured low-D starts, `--embedding-candidate-seeds` accepts a
comma-separated list of lift seeds, screens them against the exact first target
with a bounded optimizer pass, and uses the seed with the lowest screening
cost.

The product-circuit seed is still available as an explicit opt-in with
`--initial-seed-mode circuit`, or through `--initial-seed-mode auto` when
available. In the qubit L=4 protocol this gives a D=12 seed: the
two-site-periodic circuit MPS has alternating bond dimensions 4 and 8, and
LOMPS embeds the pair into one off-diagonal one-site tensor. A small
deterministic Stiefel mixing, controlled by
`--circuit-lift-mixing-amplitude`, breaks the exact period-two transfer
degeneracy before the first optimizer polish.

For a dimension-count estimate of the smallest bond dimension needed to fit
generic translation-invariant local data, use
`minimum_bond_dimension_for_ti_rdm(d, L)`. It compares the quotient tangent
dimension `2(d - 1)D^2` with the TI-compatible local-RDM dimension
`d^(2L) - d^(2L-2)`. For qubits this gives `D_min=5` for `L=3` and `D_min=10`
for `L=4`.

The optimizer evaluates ansatz fixed points with `--fixed-point-solver dense`
by default. This uses the dense eigensolver and is intended for reproducible
production/reference trajectories. `--fixed-point-solver fast` uses ARPACK
first and may be useful for exploratory runs, but tiny non-bitwise differences
can appear between repeated runs.

See [docs/algorithm.md](docs/algorithm.md) for the mathematical outline and
the TOML files in [configs](configs/) for the reference parameters.

## Reference data

The repository includes two compact checkpoint tensors:

- `nonintegrable_d12_t0p001.npy`: the first saved tensor of the historical
  non-integrable D=12 trajectory, used for reproducible comparisons;
- `nonintegrable_d10_t3p235.npy`: the last valid D=10 tensor immediately before
  a warm-start plateau, used to exercise fixed-target rescue.

It also includes the completed new D=12 trajectory under
`data/nonintegrable_d12_trajectory/`.  The standalone checkpoint above is the
tensor at `t=0.001`; `trajectory_states.npy` and `trajectory_times.npy` then
contain the 19,999 aligned samples from `t=0.002` through `t=20`.  See the
README in that directory for shapes, hashes, and a loading example.

The adjacent `ten_step_d12_reference.json` records approximate regression
costs for the compact example; floating-point details may vary across
BLAS/LAPACK implementations.
