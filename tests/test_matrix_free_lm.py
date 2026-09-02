from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from lomps.canonical import polar_retraction, random_left_canonical, unstack_tensor
from lomps.differential import analytic_directional_rdm
from lomps.fixed_target_vjp import FixedTargetDirectGradient
from lomps.matrix_free_lm import (
    MatrixFreeLMOptions,
    optimize_fixed_target_matrix_free_lm,
)
from lomps.optimizer import GaugeOrthogonalLM, LMOptions
from lomps.rdm import block_rdm
from lomps.transfer import (
    DenseFixedPointResponseSolver,
    solve_delta_fixed_point_iterative,
)


class MatrixFreeLMTests(unittest.TestCase):
    def test_iterative_jvp_fixed_point_matches_dense_solve(self) -> None:
        A, W = random_left_canonical(d=2, D=3, seed=1891)
        dense = GaugeOrthogonalLM(
            2,
            LMOptions(
                linear_solver="normal",
                fixed_point_solver="dense",
                verbose=False,
            ),
        ).evaluate(W, block_rdm(A, 2))
        direction = dense.basis[2].reshape(A.shape)
        expected, _ = analytic_directional_rdm(A, direction, 2)
        actual, info = analytic_directional_rdm(
            A,
            direction,
            2,
            fixed_point_derivative_solver="iterative",
            fixed_point_derivative_rtol=1e-12,
        )
        np.testing.assert_allclose(actual, expected, atol=2e-10, rtol=2e-10)
        self.assertGreater(info["fixed_point_derivative"]["iterations"], 0)

    def test_cached_product_levels_leave_jvp_unchanged(self) -> None:
        A, W = random_left_canonical(d=2, D=3, seed=1893)
        dense = GaugeOrthogonalLM(
            3,
            LMOptions(
                linear_solver="normal",
                fixed_point_solver="dense",
                verbose=False,
            ),
        ).evaluate(W, block_rdm(A, 3))
        direction = dense.basis[1].reshape(A.shape)
        oracle = FixedTargetDirectGradient(
            block_rdm(A, 3),
            local_dimension=2,
            block_length=3,
            fixed_point_solver="dense",
        )
        evaluation = oracle.evaluate(W)
        expected, _ = analytic_directional_rdm(
            A, direction, 3, r=evaluation.r
        )
        actual, _ = analytic_directional_rdm(
            A,
            direction,
            3,
            r=evaluation.r,
            block_product_levels=evaluation.block_product_levels,
        )
        np.testing.assert_array_equal(actual, expected)

    def test_basis_free_normal_product_matches_dense_jacobian(self) -> None:
        A, W = random_left_canonical(d=2, D=2, seed=1901)
        target = block_rdm(A, 2)
        dense = GaugeOrthogonalLM(
            2,
            LMOptions(
                linear_solver="normal",
                fixed_point_solver="dense",
                verbose=False,
            ),
        ).evaluate(W, target)
        rng = np.random.default_rng(1902)
        coefficients = rng.normal(size=dense.basis.shape[0])
        direction = np.tensordot(coefficients, dense.basis, axes=(0, 0))

        oracle = FixedTargetDirectGradient(
            target,
            local_dimension=2,
            block_length=2,
            fixed_point_solver="dense",
        )
        evaluation = oracle.evaluate(W)
        delta_rho, _ = analytic_directional_rdm(
            A, direction.reshape(A.shape), 2, r=evaluation.r
        )
        damping = 3e-4
        actual = (
            oracle.horizontal_vjp(evaluation, delta_rho)
            + damping * direction
        )
        expected_coordinates = (
            dense.jacobian.T @ (dense.jacobian @ coefficients)
            + damping * coefficients
        )
        expected = np.tensordot(
            expected_coordinates, dense.basis, axes=(0, 0)
        )
        np.testing.assert_allclose(actual, expected, atol=5e-10, rtol=5e-9)

    def test_matrix_free_lm_reduces_nearby_reachable_cost(self) -> None:
        A0, W0 = random_left_canonical(d=2, D=2, seed=1911)
        _, W1 = random_left_canonical(d=2, D=2, seed=1912)
        target_A = unstack_tensor(
            polar_retraction(W0 + 2e-3 * (W1 - W0)), 2, 2
        )
        target = block_rdm(target_A, 2)
        initial = np.linalg.norm(block_rdm(A0, 2) - target)
        A, result = optimize_fixed_target_matrix_free_lm(
            A0,
            target,
            2,
            MatrixFreeLMOptions(
                max_iterations=10,
                cost_tolerance=0.0,
                gradient_tolerance=0.0,
                krylov_max_iterations=8,
                krylov_relative_tolerance=1e-4,
                gauge_projector="dense",
                fixed_point_solver="dense",
                verbose=False,
            ),
        )
        self.assertLess(result.residual_norm, initial * 1e-2)
        self.assertGreater(result.accepted_steps, 0)
        self.assertGreater(result.normal_matvecs, 0)
        self.assertLess(
            np.linalg.norm(block_rdm(A, 2) - target), initial * 1e-2
        )

    def test_stiefel_right_fixed_point_preconditioner_reduces_cost(self) -> None:
        A0, W0 = random_left_canonical(d=2, D=3, seed=1913)
        _, W1 = random_left_canonical(d=2, D=3, seed=1914)
        target_A = unstack_tensor(
            polar_retraction(W0 + 2e-3 * (W1 - W0)), 2, 3
        )
        target = block_rdm(target_A, 2)
        initial = np.linalg.norm(block_rdm(A0, 2) - target)
        A, result = optimize_fixed_target_matrix_free_lm(
            A0,
            target,
            2,
            MatrixFreeLMOptions(
                max_iterations=8,
                cost_tolerance=0.0,
                gradient_tolerance=0.0,
                krylov_max_iterations=16,
                krylov_relative_tolerance=1e-6,
                krylov_preconditioner="right_fixed_point_stiefel",
                gauge_projector="dense",
                fixed_point_solver="dense",
                verbose=False,
            ),
        )
        self.assertLess(result.residual_norm, initial * 1e-2)
        self.assertLess(np.linalg.norm(block_rdm(A, 2) - target), initial * 1e-2)

    def test_stiefel_preconditioner_preserves_converged_lm_step(self) -> None:
        A0, W0 = random_left_canonical(d=2, D=2, seed=1915)
        _, W1 = random_left_canonical(d=2, D=2, seed=1916)
        target_A = unstack_tensor(
            polar_retraction(W0 + 1e-3 * (W1 - W0)), 2, 2
        )
        target = block_rdm(target_A, 2)
        common = dict(
            max_iterations=1,
            cost_tolerance=0.0,
            gradient_tolerance=0.0,
            krylov_max_iterations=128,
            krylov_relative_tolerance=1e-12,
            krylov_minimum_relative_tolerance=1e-12,
            gauge_projector="dense",
            fixed_point_solver="dense",
            verbose=False,
        )
        plain_A, plain = optimize_fixed_target_matrix_free_lm(
            A0,
            target,
            2,
            MatrixFreeLMOptions(**common, krylov_preconditioner="none"),
        )
        preconditioned_A, preconditioned = optimize_fixed_target_matrix_free_lm(
            A0,
            target,
            2,
            MatrixFreeLMOptions(
                **common,
                krylov_preconditioner="right_fixed_point_stiefel",
            ),
        )
        np.testing.assert_allclose(
            block_rdm(preconditioned_A, 2),
            block_rdm(plain_A, 2),
            atol=2e-10,
            rtol=2e-9,
        )
        self.assertAlmostEqual(preconditioned.cost, plain.cost, places=13)

    def test_initial_fixed_point_hint_is_reused(self) -> None:
        A, W = random_left_canonical(d=2, D=2, seed=1921)
        target = block_rdm(A, 2)
        hint = np.eye(2, dtype=np.complex128) / 2.0
        # Use the actual fixed point as the hint; its value is immaterial to
        # this test, but it keeps the gradient evaluation physically valid.
        from lomps.optimizer import optimizer_right_fixed_point

        hint, _ = optimizer_right_fixed_point(A, "dense")
        oracle = FixedTargetDirectGradient(
            target,
            local_dimension=2,
            block_length=2,
            fixed_point_solver="dense",
            gauge_projector="dense",
        )
        with patch(
            "lomps.fixed_target_vjp.optimizer_right_fixed_point",
            side_effect=AssertionError("fixed-point hint was ignored"),
        ):
            evaluation = oracle.evaluate(W, hint)
        self.assertLess(evaluation.residual_norm, 1e-12)
        self.assertEqual(evaluation.fixed_point_info["provided_hint"], 1.0)

    def test_adaptive_krylov_budget_is_recorded_and_grows(self) -> None:
        A0, W0 = random_left_canonical(d=2, D=3, seed=1931)
        _, W1 = random_left_canonical(d=2, D=3, seed=1932)
        target_A = unstack_tensor(
            polar_retraction(W0 + 1e-2 * (W1 - W0)), 2, 3
        )
        target = block_rdm(target_A, 2)
        _, result = optimize_fixed_target_matrix_free_lm(
            A0,
            target,
            2,
            MatrixFreeLMOptions(
                max_iterations=2,
                cost_tolerance=0.0,
                gradient_tolerance=0.0,
                krylov_initial_iterations=1,
                krylov_max_iterations=4,
                krylov_growth_factor=2.0,
                krylov_relative_tolerance=1e-14,
                krylov_minimum_relative_tolerance=1e-14,
                gauge_projector="dense",
                fixed_point_solver="dense",
                verbose=False,
            ),
        )
        active = [record for record in result.history if record.accepted]
        self.assertGreaterEqual(len(active), 2)
        self.assertEqual(active[0].krylov_iteration_limit, 1)
        self.assertEqual(active[1].krylov_iteration_limit, 2)
        self.assertGreater(active[0].krylov_info, 0)

    def test_lsmr_reduces_reachable_cost_without_a_jacobian(self) -> None:
        A0, W0 = random_left_canonical(d=2, D=2, seed=1941)
        _, W1 = random_left_canonical(d=2, D=2, seed=1942)
        target_A = unstack_tensor(
            polar_retraction(W0 + 5e-3 * (W1 - W0)), 2, 2
        )
        target = block_rdm(target_A, 2)
        initial = np.linalg.norm(block_rdm(A0, 2) - target)
        _, result = optimize_fixed_target_matrix_free_lm(
            A0,
            target,
            2,
            MatrixFreeLMOptions(
                krylov_solver="lsmr",
                max_iterations=8,
                cost_tolerance=0.0,
                gradient_tolerance=0.0,
                krylov_max_iterations=32,
                krylov_relative_tolerance=1e-10,
                krylov_minimum_relative_tolerance=1e-10,
                gauge_projector="dense",
                fixed_point_solver="dense",
                verbose=False,
            ),
        )
        self.assertLess(result.residual_norm, initial * 1e-4)
        self.assertGreater(result.jvps, 0)
        self.assertGreater(result.vjps, 0)
        self.assertEqual(result.normal_matvecs, 0)
        self.assertTrue(
            any(record.krylov_info in (1, 2, 4, 5) for record in result.history)
        )

    def test_reusable_dense_fixed_point_response_matches_gmres(self) -> None:
        A, _ = random_left_canonical(d=2, D=3, seed=1951)
        from lomps.optimizer import optimizer_right_fixed_point

        r, _ = optimizer_right_fixed_point(A, "dense")
        rng = np.random.default_rng(1952)
        delta_A = rng.normal(size=A.shape) + 1j * rng.normal(size=A.shape)
        reusable = DenseFixedPointResponseSolver.from_tensor(A, r)
        expected, _ = solve_delta_fixed_point_iterative(
            A, delta_A, r, rtol=1e-12
        )
        actual, _ = reusable.solve_delta(delta_A)
        np.testing.assert_allclose(actual, expected, atol=2e-11, rtol=2e-11)

    def test_auto_response_solver_obeys_memory_limit(self) -> None:
        A, _ = random_left_canonical(d=2, D=2, seed=1961)
        target = block_rdm(A, 2)
        common = dict(
            max_iterations=0,
            fixed_point_response_solver="auto",
            gauge_projector="dense",
            fixed_point_solver="dense",
            verbose=False,
        )
        _, dense = optimize_fixed_target_matrix_free_lm(
            A,
            target,
            2,
            MatrixFreeLMOptions(
                **common,
                fixed_point_response_maximum_bytes=10_000,
            ),
        )
        _, iterative = optimize_fixed_target_matrix_free_lm(
            A,
            target,
            2,
            MatrixFreeLMOptions(
                **common,
                fixed_point_response_maximum_bytes=0,
            ),
        )
        self.assertEqual(dense.fixed_point_response_solver, "dense_lu")
        self.assertGreater(dense.fixed_point_response_storage_bytes, 0)
        self.assertEqual(iterative.fixed_point_response_solver, "iterative")
        self.assertEqual(iterative.fixed_point_response_storage_bytes, 0)


if __name__ == "__main__":
    unittest.main()
