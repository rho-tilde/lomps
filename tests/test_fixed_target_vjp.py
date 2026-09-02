from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from lomps.canonical import (
    polar_retraction,
    random_left_canonical,
    stack_tensor,
    unstack_tensor,
)
from lomps.fixed_target_vjp import FixedTargetDirectGradient
from lomps.horizontal import GaugeHorizontalProjector
from lomps.optimizer import (
    GaugeOrthogonalLM,
    LMOptions,
    optimizer_right_fixed_point,
)
from lomps.protocol import NONINTEGRABLE_ISING
from lomps.rdm import block_rdm
from lomps.differential import real_vectorize_rho


def _metric(X: np.ndarray, Y: np.ndarray) -> float:
    return float(np.vdot(X, Y).real)


class FixedTargetDirectGradientTests(unittest.TestCase):
    def setUp(self) -> None:
        self.A, self.W = random_left_canonical(d=2, D=2, seed=1401)
        target_A, _ = random_left_canonical(d=2, D=2, seed=1402)
        self.target = block_rdm(target_A, 2)
        self.oracle = FixedTargetDirectGradient(
            self.target,
            local_dimension=2,
            block_length=2,
            fixed_point_solver="dense",
        )

    def test_cost_is_stable_half_squared_residual(self) -> None:
        evaluation = self.oracle.evaluate(self.W)
        self.assertAlmostEqual(
            evaluation.cost,
            0.5 * np.linalg.norm(evaluation.rho - self.target) ** 2,
            places=14,
        )

    def test_horizontal_gradient_matches_central_difference(self) -> None:
        evaluation = self.oracle.evaluate(self.W)
        projector = GaugeHorizontalProjector.from_tensor(self.A, self.W)
        rng = np.random.default_rng(1403)
        direction = projector.project(
            rng.normal(size=self.W.shape) + 1j * rng.normal(size=self.W.shape)
        )
        direction /= np.linalg.norm(direction)
        epsilon = 2e-6
        plus = polar_retraction(self.W + epsilon * direction)
        minus = polar_retraction(self.W - epsilon * direction)
        finite_difference = (
            self.oracle.evaluate(plus).cost - self.oracle.evaluate(minus).cost
        ) / (2.0 * epsilon)
        analytic = _metric(evaluation.gradient, direction)
        self.assertAlmostEqual(finite_difference, analytic, places=7)

    def test_grassmann_gradient_matches_central_difference(self) -> None:
        oracle = FixedTargetDirectGradient(
            self.target,
            local_dimension=2,
            block_length=2,
            fixed_point_solver="dense",
            tangent_slice="grassmann",
        )
        evaluation = oracle.evaluate(self.W)
        rng = np.random.default_rng(1413)
        direction = evaluation.projector.project(
            rng.normal(size=self.W.shape) + 1j * rng.normal(size=self.W.shape)
        )
        direction /= np.linalg.norm(direction)
        epsilon = 2e-6
        plus = polar_retraction(self.W + epsilon * direction)
        minus = polar_retraction(self.W - epsilon * direction)
        finite_difference = (
            oracle.evaluate(plus).cost - oracle.evaluate(minus).cost
        ) / (2.0 * epsilon)
        analytic = _metric(evaluation.gradient, direction)
        self.assertAlmostEqual(finite_difference, analytic, places=7)

    def test_grassmann_direct_gradient_matches_dense_jacobian_coordinates(self) -> None:
        direct = FixedTargetDirectGradient(
            self.target,
            local_dimension=2,
            block_length=2,
            fixed_point_solver="dense",
            tangent_slice="grassmann",
        ).evaluate(self.W)
        lm = GaugeOrthogonalLM(
            2,
            LMOptions(
                tangent_slice="grassmann",
                fixed_point_solver="dense",
                linear_solver="normal",
                verbose=False,
            ),
        ).evaluate(self.W, self.target)
        coordinates = np.array(
            [_metric(direction, direct.gradient) for direction in lm.basis]
        )
        np.testing.assert_allclose(
            coordinates, lm.gradient, atol=2e-10, rtol=2e-9
        )

    def test_direct_gradient_matches_full_jacobian_coordinates(self) -> None:
        direct = self.oracle.evaluate(self.W)
        lm = GaugeOrthogonalLM(
            2,
            LMOptions(
                fixed_point_solver="dense",
                linear_solver="normal",
                verbose=False,
            ),
        ).evaluate(self.W, self.target)
        coordinates = np.array(
            [_metric(direction, direct.gradient) for direction in lm.basis]
        )
        np.testing.assert_allclose(
            coordinates, lm.gradient, atol=2e-10, rtol=2e-9
        )

    def test_arbitrary_weight_vjp_matches_dense_jacobian(self) -> None:
        evaluation = self.oracle.evaluate(self.W)
        rng = np.random.default_rng(1404)
        weight = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
        weight = 0.5 * (weight + weight.conj().T)
        actual = self.oracle.horizontal_vjp(evaluation, weight)

        lm = GaugeOrthogonalLM(
            2,
            LMOptions(
                fixed_point_solver="dense",
                linear_solver="normal",
                verbose=False,
            ),
        ).evaluate(self.W, self.target)
        coordinates = lm.jacobian.T @ real_vectorize_rho(weight)
        expected = np.tensordot(coordinates, lm.basis, axes=(0, 0))
        np.testing.assert_allclose(actual, expected, atol=3e-10, rtol=3e-9)

    def test_exact_target_has_zero_cost(self) -> None:
        evaluation = FixedTargetDirectGradient(
            block_rdm(self.A, 2),
            local_dimension=2,
            block_length=2,
        ).evaluate(stack_tensor(unstack_tensor(self.W, 2, 2)))
        self.assertLess(evaluation.cost, 1e-28)
        self.assertLess(evaluation.residual_norm, 2e-14)

    def test_d12_gradient_uses_accurate_adjoint_fixed_point_solves(self) -> None:
        root = Path(__file__).resolve().parents[1]
        A = np.load(root / "data" / "nonintegrable_d12_t0p001.npy")
        W = stack_tensor(A)
        target = NONINTEGRABLE_ISING.target_rdm(A)
        direct = FixedTargetDirectGradient(
            target,
            local_dimension=2,
            block_length=4,
            adjoint_rtol=1e-12,
        ).evaluate(W)
        lm = GaugeOrthogonalLM(
            4,
            LMOptions(
                fixed_point_solver="dense",
                linear_solver="normal",
                verbose=False,
            ),
        ).evaluate(W, target)
        coordinates = np.array(
            [_metric(direction, direct.gradient) for direction in lm.basis]
        )
        np.testing.assert_allclose(
            coordinates, lm.gradient, atol=2e-11, rtol=2e-10
        )
        self.assertGreater(min(direct.adjoint_gmres_iterations), 0)

    def test_fast_fixed_point_is_bitwise_deterministic(self) -> None:
        first, first_info = optimizer_right_fixed_point(self.A, "fast")
        second, second_info = optimizer_right_fixed_point(self.A, "fast")
        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(first_info, second_info)


if __name__ == "__main__":
    unittest.main()
