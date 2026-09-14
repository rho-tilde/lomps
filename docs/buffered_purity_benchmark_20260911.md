# Buffered-purity prototype and benchmark (2026-09-11)

## Objective

For a fitted LOMPS window of length `L`, minimize

```text
P_{L+4}(A) = Tr[rho_{L+4}(A)^2]
```

subject to the unchanged primary constraint

```text
C_L(A) = 1/2 ||rho_L(A) - rho_target||_F^2 <= epsilon.
```

The four added sites are the current second-order Trotter buffer: two sites on
the left and two on the right.  No continuity penalty is present.

## Method

1. Evaluate the ordinary Grassmann-slice Jacobian `J_L`.
2. Evaluate the purity gradient as an RDM vector-Jacobian product with weight
   `2 rho_{L+4}`.  The full `J_{L+4}` is never constructed.
3. Remove the numerical row space of `J_L` from the purity-gradient
   coordinates.
4. Retract a descent step to the canonical manifold.
5. LM-reproject onto the primary fit and accept only if the exact nonlinear
   primary cost remains below `epsilon` and purity decreases.

## L=2 mechanism check

For a reproducible random `D=3` tensor, five secondary steps gave:

- `P_6`: `0.272040656984` -> `0.234309121387` (13.87% reduction);
- final `C_2 = 2.82e-20`, against `epsilon = 1e-12`;
- numerical visible rank `12/18` Grassmann coordinates;
- final left-canonical error `2.57e-15`.

The analytic purity directional derivative agrees with a central finite
difference to relative error `4.6e-7` in the independent derivative check.

## L=4, D=12 physical-trajectory benchmark

Each saved physical state was refined for five purity steps with
`epsilon = 3e-16`.  The secondary RDM is `rho_8`.

| time | baseline C4 | refined C4 | baseline P8 | refined P8 | relative P8 drop | seconds |
|---:|---:|---:|---:|---:|---:|---:|
| 0.003 | 2.70e-16 | 2.70e-16 | 0.9999637714 | 0.9999637708 | 6.20e-10 | 3.47 |
| 0.1 | 5.85e-17 | 1.65e-16 | 0.9620342028 | 0.9620341258 | 8.01e-8 | 3.36 |
| 1.0 | 2.94e-16 | 8.16e-17 | 0.4742207843 | 0.4742188625 | 4.05e-6 | 3.56 |
| 5.0 | 2.83e-19 | 4.64e-18 | 0.0212719141 | 0.0209483490 | 1.521% | 4.57 |
| 10.0 | 1.85e-16 | 1.61e-18 | 0.0203115433 | 0.0199967604 | 1.550% | 4.54 |
| 20.0 | 4.03e-18 | 1.30e-18 | 0.0187027073 | 0.0184373590 | 1.419% | 4.63 |

All six refined tensors pass the exact `C_4 <= 3e-16` criterion.  The
Grassmann tangent has dimension 288 and `J_4` has numerical visible rank 192,
leaving a 96-dimensional local fibre at every sample.

At late times, the five-step change inside this fibre is dynamically relevant:
the next Trotter target built from the refined tensor differs from the baseline
next target by trace distance `7.1e-6` to `8.3e-6` (`9e-12` to `1.2e-11` in
half-squared Frobenius cost).  This is expected because the next update sees
the buffered correlations that `rho_4` does not fix.  It also means that a
short sequential trajectory, rather than agreement with the old `L=4`
trajectory, is the appropriate next physical validation.

The first historical candidate at `t=0.002` was excluded because its current
dense recomputation gives `C_4 = 1.74e-15`, already above the benchmark's
fixed primary threshold before purity refinement.

## Longer t=10 check

Thirty stable secondary steps at `t=10` took 23.49 seconds and gave:

- `P_8`: `0.02031154332` -> `0.01862007602` (8.328% reduction);
- final `C_4 = 2.89e-19`;
- `rho_8` trace distance between baseline and refined tensors: `0.1540`;
- all 30 steps accepted at step length `0.01`.

The projected gradient was still nonzero after 30 steps, so this is a bounded
runtime benchmark rather than a converged minimum-purity solution.

## Short sequential L=4 trajectory

A ten-step branch was evolved from `t=5.000` through `t=5.010`.  Every
physical step used the ordinary accurate `rho_4` fit followed by five
constrained `rho_8` purity steps.  Results:

- all ten physical fits and all fifty purity steps were accepted;
- the largest final primary cost was `4.46e-18`, versus `epsilon=3e-16`;
- ordinary fits required two to four LM evaluations;
- complete physical steps took about 5.8--6.7 seconds, of which roughly
  4.2--4.7 seconds was purity refinement;
- the branch separated from the saved baseline by `6.70e-4` in `rho_4` trace
  distance after ten steps;
- its final distance to the higher-L `L=5` reference was `0.1515231`, versus
  `0.1515294` for the baseline L=4 branch.

The final improvement relative to `L=5` is only `6.3e-6`, far too short a run
to establish that the closure is physically better.  It does establish that
the sequential algorithm is stable and that its dynamical effect accumulates
smoothly rather than producing an immediate branch jump.

## Artifacts

- `benchmarks/buffered_purity_l2_smoke_20260911/`
- `benchmarks/buffered_purity_l4_d12_20260911_v3/`
- `benchmarks/buffered_purity_l4_d12_t10_long_20260911/`
- `benchmarks/buffered_purity_l4_d12_short_t5_20260911/`

The prototype is deliberately not yet enabled in production evolution.  The
benchmark establishes derivative correctness, feasibility preservation, and
a sizable late-time fibre effect before testing dynamical consequences.

## Production hardening for the Y+ L=4 test (2026-09-14)

The original converged-trajectory experiment allowed a tiny relative purity
drop to count as convergence.  At the `t=5` anchor this stopped after 1,535
accepted iterations with `P8 = 0.01352433214` (a 36.42% reduction), but the
projected fibre-gradient norm was still `1.50e-6`.  The exact `C4` value was
also essentially the `3e-16` acceptance ceiling.  It was therefore a useful
large fibre displacement, but not a clean stationarity certificate.

The production runner separates the internal fit/reprojection target from the
acceptance ceiling.  Its Y+ test uses `C4 <= 1e-17` internally while retaining
the unchanged hard ceiling `C4 <= 3e-16`.  A persistent relative-improvement
plateau is not relabelled as projected-gradient convergence.  The output
instead distinguishes a projected-gradient certificate, a numerical
line-search floor after at least 200 accepted updates, and an explicitly
budgeted tracking pass.  Every inner-iteration history is saved in a
compressed per-step ledger.

The first deep `t=1.5` anchor attempt accepted 832 fibre updates before the
line search reached a numerical floor: the last relative purity decrease was
`1.74e-15`, while the projected fibre-gradient norm remained `9.26e-6` rather
than meeting the aspirational `2e-8` threshold.  This is recorded as numerical
stationarity, not formal convergence.  To fit a ten-hour production window,
the paired `t=1.5` to `t=3` run therefore uses 20 accepted constrained purity
updates at every physical step, with deep searches at the anchor, midpoint,
and endpoint.

The paired test starts from the common high-accuracy `L=4,D=12` tensors at
`t=1.499` and `t=1.500`.  It compares a primary-only control with a branch
that strongly minimizes `P8` at the anchor and tracks that low-purity fibre
branch through `t=3`.
The start precedes the rapid TEBD departure: the baseline rho4 trace distance
is `5.36e-4` at `t=1.5`, `6.04e-3` at `t=2`, and `5.53e-2` at `t=3`.
