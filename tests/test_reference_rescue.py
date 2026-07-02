from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from lomps.canonical import canonical_errors
from lomps.evolution import distant_seeds, fit_fixed_target, options
from lomps.protocol import NONINTEGRABLE_ISING


class ReferenceRescueTests(unittest.TestCase):
    def test_first_distant_seed_rescues_fixed_d10_target(self) -> None:
        root = Path(__file__).resolve().parents[1]
        A = np.load(root / "data" / "nonintegrable_d10_t3p235.npy")[0]
        target = NONINTEGRABLE_ISING.target_rdm(A)
        seed_info = next(
            distant_seeds(
                A,
                step=1,
                amplitudes=(0.3,),
                per_amplitude=1,
                random_restarts=0,
                random_seed=20260702,
            )
        )
        _, amplitude, _, seed, distance = seed_info
        self.assertEqual(amplitude, 0.3)
        self.assertGreater(distance, 0.5)
        self.assertLess(canonical_errors(seed)["left_canonical_error"], 1e-12)
        primary, strict = options(3e-16, 1e-12)
        _, result, _, _ = fit_fixed_target(seed, target, primary, strict)
        self.assertLessEqual(result.cost, 3e-16)


if __name__ == "__main__":
    unittest.main()
