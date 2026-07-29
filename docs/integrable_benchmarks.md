# Integrable TFIM benchmark runbook

## Physical quench and the two meanings of `g`

The reference quench is

```text
H0 = -sum_j Z_j Z_(j+1) - 1.5 sum_j X_j
             |
             | quench at t=0
             v
H1 = -sum_j Z_j Z_(j+1) - 0.2 sum_j X_j.
```

In the usual physical notation,

```text
H(g_phys) = -sum_j Z_j Z_(j+1) - g_phys sum_j X_j,
```

so this is `g0_phys=+1.5 -> g1_phys=+0.2`.

The argument named `--g` in `lomps-run` is not `g_phys`. It is the signed
coefficient `g_cli` in the historical QDMT bond Hamiltonian

```text
H_bond = J ZZ + g_cli (X tensor I) + h (Z tensor I).
```

Consequently,

```text
J=-1, h=0, g_cli=-g1_phys=-0.2.
```

Use `--protocol integrable-tfim` and normally omit `--g`; the preset already
contains `g_cli=-0.2`. If the field must be overridden explicitly, the correct
spelling is `--g=-0.2`. Passing `--g=+0.2` evolves with the wrong physical
field sign. The initial value `g0_phys=+1.5` is not a `lomps-run` argument:
it was used to generate the initial ground-state tensor.

The field is asymmetric inside one bond gate, and the complete second-order
even-half / odd-full / even-half circuit restores the intended bulk field.
Do not change to `--symmetric-transverse` when reproducing the historical
benchmark. The named preset and all resolved values are saved in
`metadata.json`.

## Reference tensors and conversion

The committed inputs are:

```text
data/integrable_tfim_reference/tfim_g0_1p5_D12.npy
data/integrable_tfim_reference/tfim_g0_1p5_D21.npy
data/integrable_tfim_reference/tfim_g0_1p5_D36.npy
data/integrable_tfim_reference/tfim_g0_1p5_D42.npy
```

All are left-canonical VUMPS ground states of `H0` in native
`(physical,left,right)` order. Use D=12 for `L=4,D=12`, D=21 for
`L=5,D=21`, and the D=36 or D=42 files for matching higher-D runs. The initial
tensor does not impose a block length; set `--block-length` independently.
No lift is required when `--bond-dimension` matches the tensor filename.

The D=12 tensor was produced with the public converter:

```bash
lomps-convert-tensor \
  --input /path/to/tfim_AL_D12_g1.5.npz \
  --key A \
  --source-layout legacy-left-physical-right \
  --output data/integrable_tfim_reference/tfim_g0_1p5_D12.npy
```

The converter writes native `(physical,left,right)` order and a JSON sidecar
containing source/output SHA-256 hashes and left-canonical residuals. `lomps-run`
can also read keyed archives directly through `--initial-key` and
`--initial-layout`, but the converted input is preferred for production.

The D=21, D=36, and D=42 tensors were generated directly with uniform-MPS
VUMPS at physical `g0_phys=+1.5`, then converted through the same
layout-validation path. Their JSON sidecars additionally record VUMPS
convergence, exact thermodynamic-limit energy and magnetization checks,
transfer diagnostics, and evolution-target smoke tests.

To adapt a production command to one of the larger references, change these
three arguments together:

```text
--initial-A data/integrable_tfim_reference/tfim_g0_1p5_D36.npy
--bond-dimension 36
--output-dir runs/integrable_tfim_<chosen-L>_d36
```

or replace both occurrences of `36` by `42`. Choose `--block-length`
separately. The Hamiltonian arguments do not change.

## Static preflight

Before a long run, check the input, tensor-vs-dense target contraction, and,
when available, which Hamiltonian best matches the first state in the old
trajectory:

```bash
python examples/integrable_benchmark_audit.py \
  --legacy-trajectory /path/to/benchmark_dt=0.001_steps=19999_tol=1e-09_it=10000_D=12_cut=172800.npz
```

This performs no variational evolution. The legacy file lacks Hamiltonian and
Trotter metadata, so the candidate-cost report is important provenance rather
than ceremony.

The committed preflight report strongly selects the negative-field integrable
target over the wrong sign and nonintegrable alternatives. At `dt=1e-3`, its
first-state residual cannot distinguish the asymmetric and symmetric
second-order splittings at the accuracy of the old optimizer. LOMPS uses the
asymmetric preset because that is the historical production convention pinned
down from the old implementation; the new run records this choice explicitly.

## Production commands

### L=4, D=12

```bash
lomps-run \
  --protocol integrable-tfim \
  --initial-A data/integrable_tfim_reference/tfim_g0_1p5_D12.npy \
  --initial-layout physical-left-right \
  --output-dir runs/integrable_tfim_l4_d12 \
  --bond-dimension 12 \
  --block-length 4 \
  --delta-t 1e-3 \
  --no-symmetric-transverse \
  --fixed-point-solver dense \
  --target-source-fixed-point-solver dense \
  --target-contraction tensor \
  --accept-cost 1e-15 \
  --perturb-amplitudes 0.1,0.3,0.6,1.0 \
  --perturbations-per-amplitude 3 \
  --random-restarts 0 \
  --no-strict-retry \
  --checkpoint-every 25 \
  --steps 20000 \
  --base-time 0
```

### L=5, D=21

```bash
lomps-run \
  --protocol integrable-tfim \
  --initial-A data/integrable_tfim_reference/tfim_g0_1p5_D21.npy \
  --initial-layout physical-left-right \
  --output-dir runs/integrable_tfim_l5_d21 \
  --bond-dimension 21 \
  --block-length 5 \
  --delta-t 1e-3 \
  --no-symmetric-transverse \
  --fixed-point-solver dense \
  --target-source-fixed-point-solver dense \
  --target-contraction tensor \
  --accept-cost 1e-15 \
  --perturb-amplitudes 0.1,0.3,0.6,1.0 \
  --perturbations-per-amplitude 3 \
  --random-restarts 0 \
  --no-strict-retry \
  --checkpoint-every 25 \
  --steps 20000 \
  --base-time 0
```

Neither command needs `--g`: `--protocol integrable-tfim` resolves it to
`-0.2`. Adding `--g=-0.2` is equivalent but redundant.

This saves 20,001 aligned samples including the initial state at `t=0` and the
final state at `t=20`. Since source and trajectory dimensions match, the first
step uses the standard LM continuation; the low-D product-state CG launch is
not involved. Pause only between completed timesteps by creating `PAUSE` in
the selected run directory; remove it and repeat the command with `--resume`.

For a short cluster smoke test, change only `--steps` and the output directory.
Use a new directory for the full run because resume intentionally rejects a
different requested trajectory length.

The same production path is exercised by a three-step regression test:

```bash
python -m unittest discover -s tests -p 'test_integrable.py'
```

Besides the tensor and contraction checks, this launches `lomps-run` through
its Python module entry point with the strict settings above. It requires all
three costs to be at most `1e-15`, requires zero restarts, and compares the
saved four-site local Loschmidt signature with the independent free-fermion
result. No test data are written into the repository.

## Learning and regenerating the free-fermion reference

The exact comparison is not an old LOMPS trajectory. It follows from the
Jordan-Wigner solution of

```text
H(g) = -sum_j Z_j Z_(j+1) - g sum_j X_j.
```

After mapping spins to Majoranas, the Hamiltonian has the quadratic form
`H=(i/4) w.T A w`. Its ground state is specified by a covariance matrix
`Gamma`, and time evolution under the post-quench generator is

```text
Gamma(t) = exp(A1 t) Gamma(0) exp(A1 t).T.
```

Restricting this covariance to the `2L` Majoranas in a block gives the local
Hilbert--Schmidt overlap directly:

```text
Tr[rho_L(t) rho_L(0)]
    = 2^(-L) sqrt(det(I - Gamma_L(t) Gamma_L(0))).
```

The implementation is deliberately kept in two readable layers:

- `src/lomps/integrable.py` contains the free-fermion mathematics.
- `examples/generate_free_fermion_reference.py` chooses the quench and grid,
  checks finite-ring convergence, and writes the data and plot.

Run:

```bash
python examples/generate_free_fermion_reference.py
```

This reproduces
`data/integrable_tfim_reference/free_fermion_g0_1p5_g1_0p2_t0_t20.npz`
and `docs/figures/integrable_tfim_free_fermion_reference.png`. The NPZ contains
the exact transverse magnetization and four-site local Loschmidt signature at
every LOMPS production time. It is therefore a compact, optimizer-independent
golden reference for cluster runs.

## Result audit

```bash
python examples/integrable_benchmark_audit.py \
  --run-dir runs/integrable_tfim_l4_d12 \
  --output-prefix runs/integrable_tfim_l4_d12/integrable_audit
```

The audit reuses saved right fixed points, reports Pauli observables and energy
density, and compares both `<X>(t)` and the four-site local Hilbert--Schmidt
Loschmidt signature with independent free-fermion calculations for
`g0=1.5 -> g1=0.2`. It writes a compact JSON summary and an NPZ with the aligned
diagnostic arrays. Use `--local-patch-size` and `--ring-sites` to change the
local benchmark. For a different TFIM ground-state input, pass its physical
initial field explicitly with `--g0`.

The historical QDMT archive starts at `t=0.001` and stores tensors in
`(left,physical,right)` order. Compare local RDMs and observables at aligned
times; tensor-by-tensor equality is neither expected nor a meaningful test of
the MPS fiber chosen by two different optimizers.
