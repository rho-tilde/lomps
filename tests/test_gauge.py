from __future__ import annotations

import unittest

import numpy as np

from lomps.differential import build_jacobian
from lomps.gauge import (
    finite_gauge_transform,
    gauge_basis,
)
from lomps.rdm import block_rdm
from lomps.tangent import antihermitian_basis, singular_values_and_rank
from lomps.canonical import random_left_canonical


class GaugeTests(unittest.TestCase):
    def test_finite_gauge_invariance(self) -> None:
        A, _ = random_left_canonical(d=2, D=2, seed=5)
        X_basis = antihermitian_basis(2)
        X = X_basis[0] + 0.3 * X_basis[2]
        rho = block_rdm(A, L=4)
        rho_virtual = block_rdm(finite_gauge_transform(A, X=X, epsilon=0.2), L=4)
        rho_phase = block_rdm(
            finite_gauge_transform(A, epsilon=0.2, alpha=1.0), L=4
        )
        self.assertLess(np.linalg.norm(rho_virtual - rho), 1e-12)
        self.assertLess(np.linalg.norm(rho_phase - rho), 1e-12)

    def test_gauge_derivative_jacobian_rank_zero(self) -> None:
        A, _ = random_left_canonical(d=2, D=2, seed=6)
        gauges = gauge_basis(A)
        J, _ = build_jacobian(A, gauges.tensors, L=4)
        info = singular_values_and_rank(J, tolerances=(1e-8, 1e-10))
        self.assertEqual(info["ranks_by_tol"][1e-8], 0)
        self.assertLess(np.linalg.norm(J), 1e-9)


if __name__ == "__main__":
    unittest.main()
