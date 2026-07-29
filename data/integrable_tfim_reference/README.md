# Integrable TFIM initial states

The committed tensors

- `tfim_g0_1p5_D12.npy`, with shape `(2, 12, 12)`, and
- `tfim_g0_1p5_D21.npy`, with shape `(2, 21, 21)`,

are left-canonical uniform-MPS approximations to the VUMPS ground state of

```text
H0 = -sum_j Z_j Z_(j+1) - 1.5 sum_j X_j.
```

Both use native LOMPS order `(physical, left, right)`. The D=12 tensor was
converted from the historical QDMT archive
`data/ground_state/tfim_AL_D12_g1.5.npz`, key `A`, whose shape was
`(left, physical, right) = (12, 2, 12)`. The adjacent JSON file records both
file hashes and the conversion diagnostics.

The D=21 tensor was generated independently with uniform-MPS VUMPS at the same
physical field `g0=+1.5`. Its JSON sidecar records the seed, package versions,
VUMPS convergence, hashes, canonical diagnostics, exact ground-state checks,
and a strict three-step `L=5,D=21` LOMPS smoke test.

The benchmark quench evolves either state with the `integrable-tfim` preset,
which represents the physical post-quench Hamiltonian

```text
H1 = -sum_j Z_j Z_(j+1) - 0.2 sum_j X_j.
```

There are two different quantities called `g` in the surrounding code and
literature:

```text
physical notation:       H(g_phys) = -sum(ZZ) - g_phys sum(X)
LOMPS CLI coefficient:   H_bond    = J ZZ + g_cli (X tensor I) + h (Z tensor I)
```

Therefore `g_phys=+0.2` means `g_cli=-0.2`. The safest production command is
to pass `--protocol integrable-tfim` and omit `--g`; the preset already stores
the correct value. If an explicit override is necessary, pass `--g=-0.2`,
never `--g=+0.2`. The initial field `g0=+1.5` is already encoded in the
ground-state tensor and is not passed to `lomps-run`.

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
