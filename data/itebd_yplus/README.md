# Finite-bond-dimension iTEBD reference for the Y+ quench

This directory contains the finite-bond-dimension two-site iTEBD data used in
the paper's comparison with the `L=6`, `D=41` LOMPS trajectory (Extended Data
Fig. 3 in the Nature Physics submission).

## Physical and numerical conventions

- Hamiltonian bond term: `-ZZ + 1.05 XI - 0.5 ZI`.
- Initial state: the `Y+` product state.
- Product formula: second order, with A--B half step, B--A full step and A--B
  half step.
- Time step: `dt = 0.001`.
- Absolute Schmidt-value cutoff: `1e-8`.
- The stated iTEBD cap `chi` is imposed separately on each alternating bond.
- Reduced density matrices are the arithmetic mean of windows beginning at
  the two unit-cell origins, matching the manuscript convention.

The independent implementation in
[`../../scripts/run_capped_itebd_yplus.py`](../../scripts/run_capped_itebd_yplus.py)
was checked against the archived growing-bond-dimension trajectory before the
capped runs were produced. With a nonbinding cap, 99.6% of the first 1,000
recorded bond dimensions agree exactly, all growth events occur within one
time step, and the translation-averaged one- through four-site RDMs agree
through `t = 1` to maximum Frobenius errors between `7.8e-7` and `1.4e-6`.

## Files

- `yplus_itebd_chi{41,82,164}_dt1e-3_t7.npz`: capped trajectories through
  `t = 7`, including origin-resolved and averaged RDMs, bond-dimension
  histories, discarded weights, local observables and energy diagnostics.
- Matching `.json` files: human-readable run metadata.
- `capped_itebd_lomps_comparison.npz`: plot-ready trace distances to the
  growing-bond-dimension reference for one- through four-site RDMs.
- `capped_itebd_lomps_comparison.json`: cap times, deviation times and endpoint
  errors used to audit the comparison.

The 459 MB growing-bond-dimension reference and the associated production
LOMPS data are intentionally not duplicated in Git. They are archived in the
UCL Research Data Repository at
[doi:10.5522/04/33898246](https://doi.org/10.5522/04/33898246). Within that
archive, the growing reference is
`data/classical/tebd_reference/yplus/non_integrable_job_11218/all_rdms_L_rdm-6_11218.npy`.

## Regenerating a capped trajectory

From the repository root, run for example:

```bash
python scripts/run_capped_itebd_yplus.py \
  --max-chi 41 --dt 0.001 --t-final 7 --sample-dt 0.01 \
  --max-rdm-length 4 --svd-cutoff 1e-8 \
  --output data/itebd_yplus/yplus_itebd_chi41_dt1e-3_t7.npz
```

The command writes the numerical trajectory and a matching JSON metadata
file. Use a different output path when checking the archived files so the
reference data remain unchanged.
