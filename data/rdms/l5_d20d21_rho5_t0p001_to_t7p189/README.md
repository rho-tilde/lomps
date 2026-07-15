# L=5 D=20/D=21 stitched five-site RDMs

WARNING: This is a stitched L=5 trajectory, not a single homogeneous production run. It uses D=20 tensors from t=0.001 through t=4.481. The D=20 attempted step to t=4.482 failed; the trajectory then switches to a manually found D=21 tensor that fits the D=20-derived t=4.482 target with cost 8.0568347154143779e-15. The D=21 branch continues from t=4.482 to the current paused endpoint t=7.189. The D=20 segment also contains a manual recovery at t=4.437 with accepted cost 1.1617915215754042e-15.

All arrays are NumPy `.npy` files. The D=20 and D=21 tensors/RDMs are stored separately because their bond dimensions differ.

Files:
- `times.npy`
- `bond_dimensions.npy`
- `part_indices.npy`
- `source_segment_indices.npy`
- `source_rows.npy`
- `rho5_D20.npy`
- `times_D20.npy`
- `rho5_D21.npy`
- `times_D21.npy`
- `combined_index.csv`
- `manifest.json`

`combined_index.csv` maps each physical time point to its source segment, source row, bond dimension, and row within the corresponding D=20 or D=21 array.
