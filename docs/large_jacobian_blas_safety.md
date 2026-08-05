# Large-Jacobian BLAS safety

## Failure

On Apple Silicon with NumPy 2.4.6 linked to Accelerate, the optimized complex
contraction

```python
np.einsum("sab,pbc,tac->pst", products, delta_r, products.conj(), optimize=True)
```

segfaulted at `L=6,D=23`, where the shapes are `(64,23,23)`,
`(1058,23,23)`, and `(64,23,23)`. The crash is reproducible with random NumPy
arrays and is therefore not caused by an invalid MPS, fixed point, or optimizer
state. Restricting BLAS to one thread does not prevent it.

The original optimized path formed a single `(1058,64,64)` complex output,
about 66 MiB, and entered NumPy's `bmm_einsum`/Accelerate complex matrix
multiplication. The process terminated with `SIGSEGV`; Python could not catch
or checkpoint the active optimizer evaluation.

## Affected code

`batched_rdm_jacobian` is shared by production `GaugeOrthogonalLM` and the
dense optimizer variants. Consequently, the failure affected ordinary LM,
the LM polish after first-step CG, and adaptive bond-dimension handoffs. The
historical fixed-target CG derivative itself uses a different implementation.

Single-direction RDM derivatives, target construction, observables, and the
matrix-free VJP/JVP contractions do not create this particular rank-three
output. The experimental matrix-free VJP code still batches all tangent
products and has a separate large-`D` memory-scaling limitation.

## Implementation

The dense Jacobian remains mathematically unchanged, but its construction is
now bounded in two directions:

1. Tangent directions are processed end to end in batches chosen to keep the
   directional-product and ket/bra temporaries near 64 MiB.
2. The fixed-point derivative is tiled over physical output columns so both
   the environment and each complex matrix-multiplication output stay below
   32 MiB.
3. Each batch is written directly into the final real Jacobian. The full
   complex derivative tensor is never materialized.

For `L=6`, the tangent batch is 49 at `D=23` and 17 at `D=42`. This also avoids
the multi-gigabyte directional-product allocation that the original all-at-once
implementation would require at larger bond dimension.

## Verification

- The batched Jacobian agrees with the analytic columnwise oracle.
- Matrix-free and dense Jacobian products remain mutually consistent.
- A subprocess regression evaluates the exact `L=6,D=23,count=1058` failure
  shape. A native regression is reported as a test failure instead of killing
  the main test process.
- The production `D=22 -> D=23` handoff reaches `t=2.230` with cost
  `9.78e-15`, matching the pre-hardening result.
- The full test suite passes.
