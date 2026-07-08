from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import unittest

import numpy as np

from lomps.canonical import polar_retraction, random_left_canonical, stack_tensor, unstack_tensor
from lomps.differential import build_jacobian
from lomps.optimizer import (
    GaugeOrthogonalLM,
    LMOptions,
    batched_rdm_jacobian,
    gauge_orthogonal_basis,
    optimizer_right_fixed_point,
)
from lomps.protocol import NONINTEGRABLE_ISING
from lomps.rdm import block_rdm
from lomps.transfer import right_fixed_point


class OptimizerTests(unittest.TestCase):
    def test_dense_fixed_point_solver_is_default(self) -> None:
        self.assertEqual(LMOptions().fixed_point_solver, "dense")

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

    def test_optimizer_rejects_unknown_fixed_point_solver(self) -> None:
        with self.assertRaises(ValueError):
            GaugeOrthogonalLM(
                2,
                LMOptions(
                    fixed_point_solver="unknown",  # type: ignore[arg-type]
                    verbose=False,
                ),
            )

    def test_fixed_point_solver_policy_selects_dense_and_fast(self) -> None:
        A, _ = random_left_canonical(d=2, D=2, seed=78)
        dense, dense_info = optimizer_right_fixed_point(A, "dense")
        fast, fast_info = optimizer_right_fixed_point(A, "fast")
        np.testing.assert_allclose(dense, fast, atol=1e-10, rtol=1e-10)
        self.assertIn("residual", dense_info)
        self.assertIn("residual", fast_info)

    def test_dense_optimizer_evaluation_is_bitwise_reproducible(self) -> None:
        root = Path(__file__).resolve().parents[1]
        A = np.load(root / "data" / "nonintegrable_d12_t0p001.npy")
        target = NONINTEGRABLE_ISING.target_rdm(A)
        solver = GaugeOrthogonalLM(
            NONINTEGRABLE_ISING.block_length,
            LMOptions(
                cost_tolerance=3e-16,
                gradient_tolerance=1e-11,
                fixed_point_solver="dense",
                verbose=False,
            ),
        )
        first = solver.evaluate(stack_tensor(A), target)
        second = solver.evaluate(stack_tensor(A), target)
        self.assertTrue(np.array_equal(first.rho, second.rho))
        self.assertTrue(np.array_equal(first.jacobian, second.jacobian))
        self.assertTrue(np.array_equal(first.gradient, second.gradient))
        self.assertEqual(first.cost, second.cost)

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

    def test_zero_time_step_odd_target_is_current_rdm(self) -> None:
        A, _ = random_left_canonical(d=2, D=2, seed=76)
        protocol = replace(NONINTEGRABLE_ISING, block_length=3, delta_t=0.0)
        target = protocol.target_rdm(A)
        current = block_rdm(A, protocol.block_length)
        np.testing.assert_allclose(target, current, atol=1e-12, rtol=0.0)

    def test_odd_protocol_averages_asymmetric_reductions(self) -> None:
        A, _ = random_left_canonical(d=2, D=2, seed=77)
        protocol = replace(
            NONINTEGRABLE_ISING,
            block_length=3,
            odd_parity_warning_threshold=np.inf,
        )
        self.assertEqual(protocol.lightcone_sites, 8)
        self.assertEqual(protocol.target_margins, ((2, 3), (3, 2)))
        candidates = protocol.target_rdm_candidates(A)
        self.assertEqual(len(candidates), 2)
        expected = 0.5 * (candidates[0] + candidates[1])
        expected = 0.5 * (expected + expected.conj().T)
        expected = expected / np.trace(expected)
        np.testing.assert_allclose(protocol.target_rdm(A), expected, atol=1e-12, rtol=0.0)


if __name__ == "__main__":
    unittest.main()
