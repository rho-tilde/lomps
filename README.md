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
- an optional basis-free, Jacobian-free horizontal LM backend whose damped
  Gauss--Newton equations are solved by adaptive inner CG or rectangular LSMR;
- reproducible dense fixed-point solves by default, with an explicit fast
  ARPACK option for exploratory runs;
- named integrable and non-integrable second-order Ising quench protocols, with
  symmetric even-`L` and parity-averaged odd-`L` local windows;
- tensor-axis target evolution that applies the Trotter gates locally instead
  of materializing the full light-cone brickwall unitary;
- historical-style analytic fixed-target CG for the first lifted product-state
  update, followed by an LM polish;
- optional segmented trajectories that promote the bond dimension only after
  the current manifold cannot accept its next update;
- resumable evolution with fixed-target multistart rescue and detailed logs.
- optional post-fit minimum-purity selection on the four-site Trotter buffer,
  using an adjoint ``rho_{L+4}`` gradient without constructing its Jacobian.

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
  --accept-cost 3e-16 \
  --steps 100 \
  --base-time 3.235
```

Create `runs/d10/PAUSE` to pause between timesteps. Remove it and append
`--resume` to the same command to continue.

For `L=6,D=40` and larger bond dimensions, where constructing the horizontal
basis and dense RDM Jacobian becomes the bottleneck, select the matrix-free
backend explicitly:

```bash
lomps-run \
  --optimizer matrix-free-lm \
  --matrix-free-krylov-initial-iterations 64 \
  --matrix-free-krylov-max-iterations 256 \
  --matrix-free-krylov-preconditioner right-fixed-point-stiefel \
  --initial-A data/your_l6_tensor.npy \
  --output-dir runs/l6_d40_matrix_free \
  --bond-dimension 40 \
  --block-length 6 \
  --steps 100
```

This backend applies `J` and `J^dagger` through direct small-window
prefix/suffix contractions. It constructs neither the Jacobian nor a doubled
MPS ring. Product trees are cached for the lifetime of each outer LM
evaluation and reused by every JVP/VJP. At `D=40`, its default `auto` response
policy factors the stabilized
fixed-point response operator once per outer LM iteration and reuses that
factor across all JVPs/VJPs. The default 128 MB storage ceiling automatically
returns to matrix-free GMRES at larger `D`; it can be changed with
`--matrix-free-fixed-point-response-max-mb`. Krylov budgets, tolerances, and
operator counts are saved in
`matrix_free_optimizer_iterations.csv`; ordinary checkpoints and resume
semantics are unchanged. Add `--matrix-free-verbose` to stream the same outer
iteration diagnostics to the run log. Dense LM remains the default and is
preferable for the established `L=4,D=12` and current `L=5,D=20/21` regimes.
Its large-`D` path also supports a Grassmann tangent slice, Hermitian residual
packing, one stabilized dense-LU fixed-point response factorization per
Jacobian evaluation, and deterministic parallel tangent batches through
`--lm-tangent-slice grassmann --lm-rdm-vectorization hermitian
--lm-jacobian-response-solver dense-lu --lm-jacobian-workers N`.

CG remains the recommended matrix-free linear solver for the physical
`L=6,D=40` quench. The optional `right-fixed-point-stiefel` preconditioner
applies a positive right-fixed-point metric preconditioner on the Stiefel
tangent, leaves harmless gauge components inside the damped inner solve, and
exactly gauge-projects the completed LM step. It reduced the frozen physical
benchmark from 3,584 to 1,531 normal products; it is not enabled globally
because it provides no benefit in the small `L=4,D=12` regime. Rectangular
LSMR is available with
`--matrix-free-krylov-solver lsmr`; it avoids normal equations but has not yet
reduced the physical-target contraction count. Naive previous-solution
recycling is also available for controlled experiments but remains disabled
by default because it did not improve the benchmark.

For weakly entangled starts at large target bond dimension, see
[docs/adaptive_bond_runs.md](docs/adaptive_bond_runs.md). The optional
`lomps-run-adaptive` command runs ordinary checkpointed LOMPS segments and
promotes through a user-supplied bond-dimension ladder only after a failed next
update.

The dense RDM Jacobian uses bounded tangent and physical-output batches. See
[docs/large_jacobian_blas_safety.md](docs/large_jacobian_blas_safety.md) for
the native BLAS failure this avoids and the large-`D` regression coverage.

## Buffered maximum-entropy selection

The experimental buffered-purity optimizer keeps the ordinary LOMPS fit

```text
1/2 ||rho_L(A) - rho_target||_F^2 <= epsilon
```

as a hard constraint and lowers ``Tr(rho_{L+4}(A)^2)`` inside that feasible
set.  It uses the explicit ``rho_L`` Jacobian only to project away visible
directions.  The larger-window purity gradient is a direct adjoint
contraction, so no ``rho_{L+4}`` Jacobian is formed.  Each secondary step is
LM-reprojected and then checked against the exact nonlinear primary cost.
There is deliberately no tensor-continuity penalty.

Run the small mechanism check and the saved ``L=4,D=12`` trajectory benchmark:

```bash
python scripts/benchmark_buffered_purity.py l2-smoke \
  --output-dir benchmarks/buffered_purity_l2

python scripts/benchmark_buffered_purity.py l4-trajectory \
  --output-dir benchmarks/buffered_purity_l4

python scripts/benchmark_buffered_purity_l4_short_trajectory.py \
  --output-dir benchmarks/buffered_purity_l4_short \
  --start-time 5.0 --steps 10
```

For a genuine consecutive branch, including converged refinement of the
starting anchor and every subsequent physical step, use:

```bash
scripts/run_buffered_purity_l4_converged_t5.sh
scripts/status_buffered_purity_l4_converged_t5.sh
```

The production-style runner checkpoints after every completed physical step
and accepts `--resume`.  The cluster-portable runner
`scripts/run_buffered_purity_l4_production.py` first obtains the best ordinary
rho4 fit, freezes that attained rho4, and performs the secondary optimization
on its fixed-rho4 fibre.  It distinguishes formal projected-gradient
convergence, a machine-precision line-search floor, and exhaustion of a
tracking budget; a tracking budget is a maximum rather than a quota.  The
warm-started line search reuses the previous accepted scale, and every trial is
nonlinearly reprojected to the frozen rho4 within `--fibre-cost-target`.  The
physical target cost is still checked independently against `--accept-cost`.
Full searches can be spaced with `--purity-full-every`, while
`--purity-tracking-iterations` caps the work on intervening physical steps.

See [docs/buffered_purity_benchmark_20260911.md](docs/buffered_purity_benchmark_20260911.md)
for the initial numerical audit.

Run the `L=4,D=12` integrable TFIM reference quench with the committed VUMPS
ground-state tensor:

```bash
lomps-run \
  --protocol integrable-tfim \
  --initial-A data/integrable_tfim_reference/tfim_g0_1p5_D12.npy \
  --initial-layout physical-left-right \
  --output-dir runs/integrable_tfim_l4_d12 \
  --bond-dimension 12 \
  --block-length 4 \
  --delta-t 1e-3 \
  --accept-cost 1e-15 \
  --steps 20000 \
  --base-time 0
```

For `L=5,D=21`, use
`data/integrable_tfim_reference/tfim_g0_1p5_D21.npy`, set
`--bond-dimension 21 --block-length 5`, and choose a new output directory.
The physical quench is
`H0=-sum(ZZ)-1.5*sum(X)` to `H1=-sum(ZZ)-0.2*sum(X)`. The preset internally
uses `g=-0.2` because CLI `--g` is the signed coefficient of `X tensor I`, not
the positive physical field magnitude. Prefer omitting `--g` when using the
preset; an explicit override must be written `--g=-0.2`.

See [docs/integrable_benchmarks.md](docs/integrable_benchmarks.md) for the
Hamiltonian sign convention, production restart settings, static legacy-data
preflight, exact-solution audit, and resume procedure.

Generate the independent free-fermion reference data and overview plot with:

```bash
python examples/generate_free_fermion_reference.py
```

The script includes a short Majorana-covariance derivation and a finite-ring
convergence check. The generated NPZ provides exact `<X>(t)` and four-site
local Loschmidt values on the full production time grid.

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

Native tensor files are `.npy` arrays in `(physical,left,right)` order. Keyed
`.npz` inputs and historical `(left,physical,right)` arrays are supported with
`--initial-key` and `--initial-layout`. The `lomps-convert-tensor` command writes
a native `.npy` plus a provenance sidecar containing hashes and canonical
diagnostics.

For product starts, the default `--initial-seed-mode embedding` follows the
production QDMT start-point convention: the product tensor is embedded into
the upper-left virtual block, tiny deterministic noise is added only in the
new virtual subspace, and the result is QR-projected back to the canonical
manifold. This lifted tensor is only the optimizer seed; the first target RDM
is still built from the original low-D product source. The default lift noise
is `--embedding-noise-amplitude 1e-6`. The historical `1e-8` noise can still
be requested explicitly for audits.

When the input source has lower bond dimension than the trajectory tensor,
the default `--first-step-optimizer cg-lm` fits this first frozen target with
the historical-style analytic fixed-target Grassmann CG, then polishes the
same target with the gauge-orthogonal LM optimizer. This is the recommended
product-state launch path. Use `--first-step-cg-verbose` to print the CG
iteration trace for long first-step solves.

The default stopping criteria are deliberately asymmetric for product starts:
`--first-step-accept-cost 3e-16` keeps the A1 fit strict, while recurrent
trajectory steps default to `--accept-cost 1e-14` to avoid expensive retries at
the floating-point floor. Pass `--accept-cost 3e-16` explicitly for historical
strict-continuation checks.

For less structured low-D starts, `--embedding-candidate-seeds` accepts a
comma-separated list of lift seeds, screens them against the exact first target
with a bounded optimizer pass, and uses the seed with the lowest screening
cost.

When a separate high-D optimizer seed is already available, pass it with
`--initial-seed-A`. The physical first target is still built from
`--initial-A`; `--initial-seed-A` only initializes the trajectory manifold and
is lifted to `--bond-dimension` if needed. The external seed lift uses
`--initial-seed-lift-noise-amplitude` when supplied. If that option is omitted,
LOMPS uses `1e-4` for trajectory bond dimensions `D>=20` and otherwise reuses
`--embedding-noise-amplitude`.

The product-circuit seed is still available as an explicit opt-in with
`--initial-seed-mode circuit`, or through `--initial-seed-mode auto` when
available. In the qubit L=4 protocol this gives a D=12 seed: the
two-site-periodic circuit MPS has alternating bond dimensions 4 and 8, and
LOMPS embeds the pair into one off-diagonal one-site tensor. A small
deterministic Stiefel mixing, controlled by
`--circuit-lift-mixing-amplitude`, breaks the exact period-two transfer
degeneracy before the first optimizer polish. This construction is retained as
an experimental diagnostic path, not as the default product launch, because it
has not been robust in the production product-start tests so far.

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

Each saved trajectory tensor also stores its right transfer fixed point in
`right_fixed_points.npy`, aligned with `states.npy`. The next timestep reuses
this cached boundary when constructing the light-cone target, so accepted
trajectory states are checkpointed together with the fixed point needed to
evolve them. The first physical source has its own
`initial_source_right_fixed_point.npy`, because product-state launches build
the first target from `--initial-A`, not from the lifted trajectory seed.

Target construction uses `--target-contraction tensor` by default. This applies
the local Trotter gates directly to density-tensor axes and avoids building the
full light-cone unitary; `--target-contraction dense` keeps the old
`U rho U^\dagger` implementation as a regression path. The fixed point used
when filling or falling back from the target-source cache is controlled
separately by `--target-source-fixed-point-solver`.

See [docs/algorithm.md](docs/algorithm.md) for the mathematical outline,
[docs/product_starts.md](docs/product_starts.md) for the product-state launch
runbook and audits, [docs/integrable_benchmarks.md](docs/integrable_benchmarks.md)
for the integrable-quench runbook, and the TOML files in [configs](configs/)
for the reference parameters.

## Reference data

The repository includes compact checkpoint and reference tensors:

- `nonintegrable_d12_t0p001.npy`: the first saved tensor of the historical
  non-integrable D=12 trajectory, used for reproducible comparisons;
- `nonintegrable_d10_t3p235.npy`: the last valid D=10 tensor immediately before
  a warm-start plateau, used to exercise fixed-target rescue.
- `integrable_tfim_reference/tfim_g0_1p5_D12.npy`,
  `tfim_g0_1p5_D21.npy`, `tfim_g0_1p5_D36.npy`, and
  `tfim_g0_1p5_D42.npy`: left-canonical VUMPS ground-state tensors at physical
  field `g0=1.5`, with hashes, diagnostics, and provenance stored beside them.

It also includes the completed new D=12 trajectory under
`data/nonintegrable_d12_trajectory/`.  The standalone checkpoint above is the
tensor at `t=0.001`; `trajectory_states.npy` and `trajectory_times.npy` then
contain the 19,999 aligned samples from `t=0.002` through `t=20`.  See the
README in that directory for shapes, hashes, and a loading example.

The adjacent `ten_step_d12_reference.json` records approximate regression
costs for the compact example; floating-point details may vary across
BLAS/LAPACK implementations.
