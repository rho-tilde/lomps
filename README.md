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
- the `L=4`, second-order non-integrable Ising quench protocol;
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
