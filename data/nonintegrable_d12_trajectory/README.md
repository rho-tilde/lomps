# New non-integrable D=12 trajectory

This directory contains the completed gauge-orthogonal LM evolution used as
the D=12 reference trajectory and as the source for the finite-window branch
experiments.

## Files and alignment

- `../nonintegrable_d12_t0p001.npy`: tensor at `t=0.001`, shape `(2, 12, 12)`;
- `trajectory_states.npy`: 19,999 tensors, shape `(19999, 2, 12, 12)`;
- `trajectory_times.npy`: aligned times from `0.002` through `20.0`, shape
  `(19999,)`;
- `metadata.json`: protocol, optimizer, provenance, and completion metadata.

The tensor convention is `(time, physical, left, right)`.  To obtain one
continuous series beginning at `t=0.001`:

```python
from pathlib import Path
import numpy as np

data = Path("data")
initial = np.load(data / "nonintegrable_d12_t0p001.npy")
states = np.load(
    data / "nonintegrable_d12_trajectory" / "trajectory_states.npy",
    mmap_mode="r",
)
times = np.load(
    data / "nonintegrable_d12_trajectory" / "trajectory_times.npy",
    mmap_mode="r",
)

# Avoid copying the 92 MB trajectory unless a contiguous in-memory array is
# actually required.  The complete logical series is initial, then states[:].
assert times[0] == 0.002
assert states.shape == (19999, 2, 12, 12)
```

## Integrity

SHA-256 checksums:

```text
334dfd0a007438424f0b9faac0299a955aef9a0e32a2036fba8f0149888a7b71  trajectory_states.npy
3e2ce72265d0261e4a2a7eaaaf7bffd0c059051a6700db697b9ca56ccba7ec24  trajectory_times.npy
5d4ac9da105d8e471debf3de60901f5c5f59420999ae7c9ca27999669ec2aed2  ../nonintegrable_d12_t0p001.npy
```
