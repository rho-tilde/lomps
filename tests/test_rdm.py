from __future__ import annotations

import unittest

import numpy as np

from lomps.canonical import random_left_canonical, unstack_tensor
from lomps.differential import derivative_difference_norm
from lomps.rdm import block_rdm, rdm_diagnostics
from lomps.tangent import tangent_bases
from lomps.transfer import apply_channel, right_fixed_point


class RDMTests(unittest.TestCase):
    def test_fixed_point_and_rdm_properties(self) -> None:
        A, _ = random_left_canonical(d=2, D=2, seed=3)
        r, info = right_fixed_point(A)
        self.assertLess(info["residual"], 1e-12)
        self.assertLess(np.linalg.norm(apply_channel(A, r) - r), 1e-12)
        self.assertAlmostEqual(np.trace(r).real, 1.0, places=12)
        self.assertGreaterEqual(info["min_eigenvalue"], -1e-12)

        rho = block_rdm(A, L=4, r=r)
        diag = rdm_diagnostics(rho)
        self.assertEqual(diag["shape"], (16, 16))
        self.assertLess(diag["trace_error"], 1e-12)
        self.assertLess(diag["hermiticity_error"], 1e-12)
        self.assertGreaterEqual(diag["min_eigenvalue"], -1e-12)
        self.assertLessEqual(diag["rank"], 4)

    def test_analytic_derivative_matches_finite_difference(self) -> None:
        A, W = random_left_canonical(d=2, D=2, seed=4)
        bases = tangent_bases(W, d=2, D=2)
        delta_A = unstack_tensor(bases.perp[0], d=2, D=2)
        diff = derivative_difference_norm(A, delta_A, L=4, epsilon=1e-6)
        self.assertLess(diff["relative"], 2e-5)


if __name__ == "__main__":
    unittest.main()
