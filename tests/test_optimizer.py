from __future__ import annotations

from dataclasses import replace
import unittest

import numpy as np

from lomps.canonical import polar_retraction, random_left_canonical, stack_tensor, unstack_tensor
from lomps.differential import build_jacobian
from lomps.optimizer import (
    GaugeOrthogonalLM,
    LMOptions,
    batched_rdm_jacobian,
    gauge_orthogonal_basis,
)
from lomps.protocol import NONINTEGRABLE_ISING
from lomps.rdm import block_rdm
from lomps.transfer import right_fixed_point


class OptimizerTests(unittest.TestCase):
    def test_lm_recovers_nearby_reachable_rdm(self) -> None:
        A0, W0 = random_left_canonical(d=2, D=2, seed=72)
        A1, _ = random_left_canonical(d=2, D=2, seed=73)
        target_A = unstack_tensor(
            polar_retraction(W0 + 1e-3 * (stack_tensor(A1) - W0)),
            2,
            2,
        )
        target = block_rdm(target_A, 2)
        solver = GaugeOrthogonalLM(
            2,
            LMOptions(
                max_iterations=12,
                gradient_tolerance=1e-12,
                residual_tolerance=1e-10,
                verbose=False,
            ),
        )
        result = solver.optimize(W0, target)
        self.assertLess(result.residual_norm, 1e-9)

    def test_batched_jacobian_matches_columnwise_jacobian(self) -> None:
        A, W = random_left_canonical(d=2, D=2, seed=75)
        basis_W = gauge_orthogonal_basis(A, W, tolerance=1e-10)
        basis_A = [unstack_tensor(direction, 2, 2) for direction in basis_W]
        r, _ = right_fixed_point(A)
        expected, _ = build_jacobian(A, basis_A, 3, r=r)
        actual = batched_rdm_jacobian(A, basis_A, 3, r)
        np.testing.assert_allclose(actual, expected, atol=1e-11, rtol=1e-10)

    def test_zero_time_step_target_is_current_rdm(self) -> None:
        A, _ = random_left_canonical(d=2, D=2, seed=74)
        protocol = replace(NONINTEGRABLE_ISING, delta_t=0.0)
        target = protocol.target_rdm(A)
        current = block_rdm(A, protocol.block_length)
        np.testing.assert_allclose(target, current, atol=1e-12, rtol=0.0)


if __name__ == "__main__":
    unittest.main()
