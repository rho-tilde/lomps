"""Minimal ten-step non-integrable L=4 evolution example."""

from pathlib import Path

import numpy as np

from lomps import LMOptions, NONINTEGRABLE_ISING, optimize_tensor


root = Path(__file__).resolve().parents[1]
A = np.load(root / "data" / "nonintegrable_d12_t0p001.npy")
options = LMOptions(
    cost_tolerance=3e-16,
    gradient_tolerance=1e-11,
    max_iterations=40_000,
    verbose=False,
)

for step in range(1, 11):
    target = NONINTEGRABLE_ISING.target_rdm(A)
    A, result = optimize_tensor(A, target, NONINTEGRABLE_ISING.block_length, options)
    print(
        f"step={step:02d} time={0.001 * (step + 1):.3f} "
        f"cost={result.cost:.6e} residual={result.residual_norm:.6e}"
    )
