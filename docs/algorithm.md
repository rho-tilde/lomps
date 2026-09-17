# LOMPS algorithm

LOMPS advances local quantum dynamics and closes each time step with a uniform
left-canonical matrix-product state.

## Local update

Let `A` be the current uMPS tensor. The quench protocol builds an evolved local
target from a finite light cone. For an even matching window `L`,

```text
target_L(A) = Tr_{2 left, 2 right}[
    U_Strang rho_{L+4}(A) U_Strang^dagger
].
```

For odd `L`, the light cone contains `L+5` sites. LOMPS averages the two
parity-related reductions,

```text
target_L(A) = 1/2 target_L^{2|3}(A) + 1/2 target_L^{3|2}(A).
```

The next tensor `B` minimizes

```text
C(B) = 1/2 ||rho_L(B) - target_L(A)||_F^2.
```

The target is computed once per physical time step. Optimizer restarts change
the initial guess for `B`, not the target being fitted.

## MPS geometry and optimization

The tensor is constrained to the left-canonical manifold. LOMPS constructs its
real Stiefel tangent, removes the infinitesimal MPS gauge tangent, and evaluates
the analytic RDM derivative on the remaining horizontal directions. The
default dense optimizer takes a damped Gauss--Newton/Levenberg--Marquardt step
and returns accepted steps to the manifold by polar retraction.

The matrix-free backend applies the RDM Jacobian and its adjoint through local
contractions. It solves the damped linear problem with CG or LSMR without
storing the full Jacobian. Both backends minimize the same fixed-target cost.

The ansatz transfer fixed point is computed with a dense eigensolver by
default. The optional `fast` policy tries ARPACK and falls back to the dense
solver if validation fails. Dense fixed points are preferred for reference
trajectories because they are reproducible across repeated runs.

## Lower-bond-dimension initial states

When the initial state has a smaller bond dimension than the requested
trajectory, LOMPS keeps the physical source separate from the optimizer seed.
The first target is built from the original source tensor. A deterministic
left-canonical lift initializes the higher-dimensional optimizer but does not
alter that target.

For product states, the default lift embeds the product tensor into the
upper-left virtual block, adds deterministic noise in the new virtual
subspace, and restores left canonical form by QR projection. The first fit uses
fixed-target conjugate gradient followed by an LM polish. After the first
accepted update, the high-dimensional tensor becomes the source of subsequent
targets.

Alternative seed construction and screening options are described in
[product_starts.md](product_starts.md).

## Dimension estimate

A trace-fixed `L`-site density matrix has `d^(2L)-1` real parameters. Translation
invariance imposes equality of its two `(L-1)`-site marginals, leaving

```text
dim rho_L^TI = d^(2L) - d^(2L-2).
```

The real uMPS quotient tangent has dimension `2(d-1)D^2`. The helper
`minimum_bond_dimension_for_ti_rdm(d,L)` therefore returns the smallest `D`
satisfying

```text
2(d-1)D^2 >= d^(2L) - d^(2L-2).
```

For qubits this count gives `D_min=5` at `L=3` and `D_min=10` at `L=4`. It is a
parameter-counting estimate, not a guarantee that every target is reachable or
that the optimizer is well conditioned.

## Checkpointing and recovery

Every accepted tensor is stored with its right transfer fixed point. If a warm
start fails, the runner retries the same fixed target from configured
left-canonical perturbations and records the cost, distance, evaluation count,
and wall time of each attempt. Failed candidates are never used as physical
sources for later time steps.
