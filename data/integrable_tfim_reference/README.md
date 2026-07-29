# Integrable TFIM D=12 initial state

`tfim_g0_1p5_D12.npy` is the left-canonical uniform-MPS tensor for the VUMPS
ground state of

```text
H(g0) = -sum_j Z_j Z_(j+1) - 1.5 sum_j X_j.
```

It has native LOMPS shape `(physical, left, right) = (2, 12, 12)`. The tensor
was converted from the historical QDMT archive
`data/ground_state/tfim_AL_D12_g1.5.npz`, key `A`, whose shape was
`(left, physical, right) = (12, 2, 12)`. The adjacent JSON file records both
file hashes and the conversion diagnostics.

The benchmark quench evolves this state with the `integrable-tfim` preset,
which represents the physical post-quench Hamiltonian

```text
H(g1) = -sum_j Z_j Z_(j+1) - 0.2 sum_j X_j.
```

The preset coefficient is `g=-0.2` because LOMPS reproduces the historical
asymmetric bond convention `H_bond = J ZZ + g (X tensor I) + h (Z tensor I)`.
See `docs/integrable_benchmarks.md` before comparing to a legacy archive.

`legacy_first_state_preflight.json` records a static comparison with the first
saved state of the old high-accuracy D=12 archive. Its half-squared Frobenius
cost is `1.8793e-15` for the negative-field integrable target, compared with
`1.8517e-7` for the wrong field sign and `3.2035e-6` for the nonintegrable
target. The asymmetric and symmetric negative-field splittings differ in this
test by only `1.8e-19` in cost, below what that old fit can resolve; the report
therefore validates the quench Hamiltonian but not the historical gate split
by itself.

## Independent free-fermion reference

`free_fermion_g0_1p5_g1_0p2_t0_t20.npz` is independent of both the LOMPS
optimizer and its Trotter circuit. It contains exact free-fermion values on the
same `dt=1e-3` grid from `t=0` through `t=20`:

```text
times
sigma_x
sigma_y
sigma_z
local_hs_rate
local_hs_overlap
```

The local quantities use a four-site patch. The nominally vanishing
magnetizations are included so a trajectory audit can compare every Pauli
component without special cases. The adjacent JSON file records the precise
parameters, formulas, file hashes, and the finite-ring convergence check.

Regenerate the data and its overview plot from the repository root with:

```bash
python examples/generate_free_fermion_reference.py
```

The script is intended to be readable. The underlying Jordan-Wigner/Majorana
construction is in `src/lomps/integrable.py`.
