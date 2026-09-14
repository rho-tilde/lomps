from __future__ import annotations

import unittest

import numpy as np

from lomps.buffered_purity import (
    BufferedPurityGradient,
    BufferedPurityOptions,
    minimize_buffered_purity,
)
from lomps.canonical import polar_retraction, random_left_canonical
from lomps.rdm import block_rdm


def _metric(X: np.ndarray, Y: np.ndarray) -> float:
    return float(np.vdot(X, Y).real)


class BufferedPurityTests(unittest.TestCase):
    def test_adjoint_purity_gradient_matches_central_difference(self) -> None:
        _, W = random_left_canonical(d=2, D=2, seed=2601)
        oracle = BufferedPurityGradient(2, buffer_sites=4)
        evaluation = oracle.evaluate(W)
        self.assertEqual(evaluation.extended_block_length, 6)

        rng = np.random.default_rng(2602)
        ambient = rng.normal(size=W.shape) + 1j * rng.normal(size=W.shape)
        direction = evaluation.gradient * 0.0 + ambient
        direction -= W @ (W.conj().T @ direction)
        direction /= np.linalg.norm(direction)
        epsilon = 1e-6
        plus = oracle.evaluate(
            polar_retraction(W + epsilon * direction)
        ).purity
        minus = oracle.evaluate(
            polar_retraction(W - epsilon * direction)
        ).purity
        finite_difference = (plus - minus) / (2.0 * epsilon)
        analytic = _metric(evaluation.gradient, direction)
        self.assertAlmostEqual(finite_difference, analytic, places=6)

    def test_projected_descent_lowers_purity_and_preserves_primary_cost(self) -> None:
        A, _ = random_left_canonical(d=2, D=3, seed=2611)
        target = block_rdm(A, 2)
        tolerance = 1e-10
        refined, result = minimize_buffered_purity(
            A,
            target,
            2,
            BufferedPurityOptions(
                primary_cost_tolerance=tolerance,
                max_iterations=4,
                initial_step=5e-2,
                minimum_step=1e-9,
                projection_max_iterations=6,
                verbose=False,
            ),
        )

        residual = block_rdm(refined, 2) - target
        exact_cost = 0.5 * float(np.vdot(residual, residual).real)
        self.assertEqual(result.accepted_steps, 4)
        self.assertLess(result.purity, 0.95 * result.initial_purity)
        self.assertLessEqual(exact_cost, tolerance)
        self.assertTrue(all(record.accepted for record in result.history))
        self.assertTrue(
            all(
                record.linearized_primary_change_norm < 1e-12
                for record in result.history
            )
        )

    def test_infeasible_initial_point_is_not_modified(self) -> None:
        A, _ = random_left_canonical(d=2, D=2, seed=2621)
        other, _ = random_left_canonical(d=2, D=2, seed=2622)
        _, result = minimize_buffered_purity(
            A,
            block_rdm(other, 2),
            2,
            BufferedPurityOptions(
                primary_cost_tolerance=1e-14,
                max_iterations=2,
                verbose=False,
            ),
        )
        self.assertEqual(result.status, "initial_point_infeasible")
        self.assertEqual(result.accepted_steps, 0)
        self.assertEqual(result.history, ())

    def test_reprojection_can_target_stricter_cost_than_acceptance(self) -> None:
        A, _ = random_left_canonical(d=2, D=3, seed=2631)
        target = block_rdm(A, 2)
        refined, result = minimize_buffered_purity(
            A,
            target,
            2,
            BufferedPurityOptions(
                primary_cost_tolerance=1e-8,
                projection_cost_tolerance=1e-12,
                max_iterations=2,
                initial_step=5e-2,
                minimum_step=1e-9,
                projection_max_iterations=8,
                verbose=False,
            ),
        )

        residual = block_rdm(refined, 2) - target
        exact_cost = 0.5 * float(np.vdot(residual, residual).real)
        self.assertEqual(result.accepted_steps, 2)
        self.assertLessEqual(exact_cost, 1e-12)

    def test_relative_plateau_can_be_classified_as_stall(self) -> None:
        A, _ = random_left_canonical(d=2, D=3, seed=2641)
        target = block_rdm(A, 2)
        _, result = minimize_buffered_purity(
            A,
            target,
            2,
            BufferedPurityOptions(
                primary_cost_tolerance=1e-10,
                max_iterations=8,
                initial_step=5e-2,
                minimum_step=1e-9,
                null_gradient_tolerance=0.0,
                relative_purity_tolerance=1.0,
                relative_purity_patience=2,
                relative_purity_is_convergence=False,
                projection_max_iterations=6,
                verbose=False,
            ),
        )

        self.assertEqual(result.status, "relative_purity_stall")
        self.assertEqual(result.accepted_steps, 2)

    def test_reprojection_is_skipped_when_raw_trial_is_already_strict(self) -> None:
        A, _ = random_left_canonical(d=2, D=3, seed=2651)
        target = block_rdm(A, 2)
        _, result = minimize_buffered_purity(
            A,
            target,
            2,
            BufferedPurityOptions(
                primary_cost_tolerance=1e-6,
                projection_cost_tolerance=1e-6,
                max_iterations=1,
                initial_step=1e-6,
                minimum_step=1e-12,
                verbose=False,
            ),
        )

        self.assertEqual(result.accepted_steps, 1)
        self.assertEqual(
            result.history[0].projection_status,
            "already_within_projection_tolerance",
        )
        self.assertEqual(result.history[0].projection_evaluations, 0)


if __name__ == "__main__":
    unittest.main()
