# LOMPS algorithm

LOMPS evolves a finite reduced density matrix while closing the dynamics with
a uniform left-canonical matrix-product state.

For a left-canonical tensor `A`, the non-integrable reference protocol builds
a finite-lightcone density matrix, applies an even-half / odd-full / even-half
Strang circuit, and traces buffer sites to recover the target local block. For
even `L`, the light cone has `L + 4` sites and the reduction is symmetric:

```text
target_L(A) = Tr_{2 left, 2 right}[U_Strang rho_{L+4}(A) U_Strang^dagger].
```

For odd `L`, LOMPS keeps the Strang circuit on an even number of sites by using
an `L + 5` site light cone and averaging the two parity-related reductions:

```text
target_L(A) = 1/2 target_L^{2|3}(A) + 1/2 target_L^{3|2}(A).
```

The next tensor minimizes

```text
C(B) = 1/2 ||rho_L(B) - target_L(A)||_F^2.
```

The runner warns if the two odd-`L` reductions have a trace distance larger
than the configured parity warning threshold.

The optimizer constructs the real Stiefel tangent of the left-canonical
tensor, removes the true infinitesimal MPS gauge tangent, evaluates the
analytic RDM Jacobian on the remaining horizontal slice, and takes a damped
Gauss--Newton/Levenberg--Marquardt step. Each accepted step is returned to the
left-canonical manifold by polar retraction.

Inside optimizer evaluations, LOMPS computes the ansatz transfer fixed point
with a selectable policy. The default `dense` policy uses the dense eigensolver
and is chosen for bitwise reproducibility in production/reference trajectories.
The optional `fast` policy tries ARPACK first and falls back to dense if the
iterative fixed point fails validation; this is faster, but repeated runs can
diverge at the last few floating-point digits because the iterative fixed point
is not bitwise deterministic.

If the warm-started solve plateaus, LOMPS keeps `target(A)` fixed and restarts
the same minimization from more distant left-canonical tensors. A restart never
changes the target; it only changes the optimizer's initial point. The runner
records every trial, its distance, cost, residual, evaluation count, and wall
time.

Stationarity searches and fixed-rho finite-window fibre experiments are
separate research questions and are intentionally absent from the core
evolution package.
