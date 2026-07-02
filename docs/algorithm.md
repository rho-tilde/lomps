# LOMPS algorithm

LOMPS evolves a finite reduced density matrix while closing the dynamics with
a uniform left-canonical matrix-product state.

For a left-canonical tensor `A`, the non-integrable reference protocol builds
the eight-site density matrix, applies an even-half / odd-full / even-half
Strang circuit, and traces the two outer sites on each side:

```text
target(A) = Tr_outer[U_Strang rho_8(A) U_Strang^dagger].
```

The next tensor minimizes

```text
C(B) = 1/2 ||rho_4(B) - target(A)||_F^2.
```

The optimizer constructs the real Stiefel tangent of the left-canonical
tensor, removes the true infinitesimal MPS gauge tangent, evaluates the
analytic RDM Jacobian on the remaining horizontal slice, and takes a damped
Gauss--Newton/Levenberg--Marquardt step. Each accepted step is returned to the
left-canonical manifold by polar retraction.

If the warm-started solve plateaus, LOMPS keeps `target(A)` fixed and restarts
the same minimization from more distant left-canonical tensors. A restart never
changes the target; it only changes the optimizer's initial point. The runner
records every trial, its distance, cost, residual, evaluation count, and wall
time.

Stationarity searches and fixed-rho finite-window fibre experiments are
separate research questions and are intentionally absent from the core
evolution package.
