from __future__ import annotations

import unittest

import numpy as np

from lomps.canonical import (
    canonical_errors,
    polar_retraction,
    random_left_canonical,
    stack_tensor,
    unstack_tensor,
)
from lomps.gauge import finite_virtual_gauge
from lomps.horizontal import GaugeHorizontalProjector
from lomps.horizontal_cg import (
    HorizontalCGOptions,
    optimize_fixed_target_horizontal_cg,
    polar_retraction_curve,
)
from lomps.rdm import block_rdm
from lomps.tangent import antihermitian_basis


class HorizontalCGTests(unittest.TestCase):
    def test_polar_curve_velocity_matches_finite_difference(self) -> None:
        A, W = random_left_canonical(d=2, D=3, seed=1501)
        projector = GaugeHorizontalProjector.from_tensor(A, W)
        rng = np.random.default_rng(1502)
        direction = projector.project(
            rng.normal(size=W.shape) + 1j * rng.normal(size=W.shape)
        )
        alpha = 0.17
        candidate, velocity = polar_retraction_curve(W, direction, alpha)
        epsilon = 1e-6
        plus, _ = polar_retraction_curve(W, direction, alpha + epsilon)
        minus, _ = polar_retraction_curve(W, direction, alpha - epsilon)
        np.testing.assert_allclose(
            velocity, (plus - minus) / (2 * epsilon), atol=2e-9, rtol=2e-9
        )
        np.testing.assert_allclose(
            candidate.conj().T @ candidate,
            np.eye(3),
            atol=2e-13,
            rtol=2e-13,
        )

    def test_horizontal_cg_reduces_nearby_reachable_cost(self) -> None:
        A0, W0 = random_left_canonical(d=2, D=2, seed=1511)
        _, W1 = random_left_canonical(d=2, D=2, seed=1512)
        target_A = unstack_tensor(
            polar_retraction(W0 + 2e-3 * (W1 - W0)), 2, 2
        )
        target = block_rdm(target_A, 2)
        initial_cost = 0.5 * np.linalg.norm(block_rdm(A0, 2) - target) ** 2
        A, result = optimize_fixed_target_horizontal_cg(
            A0,
            target,
            2,
            HorizontalCGOptions(
                max_iterations=12,
                cost_tolerance=0.0,
                gradient_tolerance=0.0,
                preconditioner="grassmann_slice",
                conjugacy_metric="horizontal",
                maximum_seconds=30.0,
                verbose=False,
            ),
        )
        self.assertLess(result.cost, initial_cost * 2e-3)
        self.assertGreater(result.accepted_steps, 0)
        self.assertLess(canonical_errors(A)["left_canonical_error"], 2e-12)

    def test_matched_cg_reduces_cost_in_both_tangent_slices(self) -> None:
        A0, W0 = random_left_canonical(d=2, D=2, seed=1515)
        _, W1 = random_left_canonical(d=2, D=2, seed=1516)
        target_A = unstack_tensor(
            polar_retraction(W0 + 2e-3 * (W1 - W0)), 2, 2
        )
        target = block_rdm(target_A, 2)
        initial_cost = 0.5 * np.linalg.norm(block_rdm(A0, 2) - target) ** 2
        results = []
        for tangent_slice in ("gauge_orthogonal", "grassmann"):
            _, result = optimize_fixed_target_horizontal_cg(
                A0,
                target,
                2,
                HorizontalCGOptions(
                    tangent_slice=tangent_slice,
                    max_iterations=8,
                    cost_tolerance=0.0,
                    gradient_tolerance=0.0,
                    preconditioner="none",
                    conjugacy_metric="horizontal",
                    fixed_point_solver="dense",
                    maximum_seconds=30.0,
                    verbose=False,
                ),
            )
            self.assertLess(result.cost, initial_cost)
            self.assertGreater(result.accepted_steps, 0)
            results.append(result)
        self.assertEqual(
            results[0].objective_evaluations,
            results[1].objective_evaluations,
        )

    def test_gauge_equivalent_starts_reach_same_local_accuracy(self) -> None:
        A0, W0 = random_left_canonical(d=2, D=2, seed=1521)
        _, W1 = random_left_canonical(d=2, D=2, seed=1522)
        target_A = unstack_tensor(
            polar_retraction(W0 + 1e-3 * (W1 - W0)), 2, 2
        )
        target = block_rdm(target_A, 2)
        generator = antihermitian_basis(2)[1]
        gauged_A0 = finite_virtual_gauge(A0, generator, epsilon=0.37)
        options = HorizontalCGOptions(
            max_iterations=10,
            cost_tolerance=1e-18,
            gradient_tolerance=0.0,
            maximum_seconds=30.0,
            verbose=False,
        )
        A_first, first = optimize_fixed_target_horizontal_cg(
            A0, target, 2, options
        )
        A_second, second = optimize_fixed_target_horizontal_cg(
            gauged_A0, target, 2, options
        )
        initial_residual = np.linalg.norm(block_rdm(A0, 2) - target)
        self.assertLess(first.residual_norm, initial_residual)
        self.assertLess(second.residual_norm, initial_residual)
        self.assertAlmostEqual(
            first.residual_norm, second.residual_norm, places=13
        )
        np.testing.assert_allclose(
            block_rdm(A_first, 2),
            block_rdm(A_second, 2),
            atol=2e-13,
            rtol=2e-13,
        )


if __name__ == "__main__":
    unittest.main()
