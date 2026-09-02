from __future__ import annotations

import unittest

import numpy as np

from lomps.canonical import random_left_canonical, stack_tensor
from lomps.horizontal import (
    GaugeHorizontalProjector,
    GrassmannHorizontalProjector,
    stiefel_project,
)
from lomps.optimizer import gauge_orthogonal_basis
from lomps.tangent import vectorize_complex_real


def _metric(X: np.ndarray, Y: np.ndarray) -> float:
    return float(np.vdot(X, Y).real)


class GaugeHorizontalProjectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.A, self.W = random_left_canonical(d=2, D=3, seed=1301)
        self.projector = GaugeHorizontalProjector.from_tensor(
            self.A, self.W, tolerance=1e-10
        )
        rng = np.random.default_rng(1302)
        self.X = rng.normal(size=self.W.shape) + 1j * rng.normal(
            size=self.W.shape
        )
        self.Y = rng.normal(size=self.W.shape) + 1j * rng.normal(
            size=self.W.shape
        )

    def test_expected_true_gauge_rank(self) -> None:
        self.assertEqual(self.projector.gauge_rank, 3**2)

    def test_output_is_tangent_and_gauge_orthogonal(self) -> None:
        horizontal = self.projector.project(self.X)
        self.assertLess(self.projector.tangency_error(horizontal), 2e-13)
        self.assertLess(self.projector.gauge_overlap_norm(horizontal), 2e-13)

    def test_projection_is_idempotent(self) -> None:
        once = self.projector.project(self.X)
        twice = self.projector.project(once)
        np.testing.assert_allclose(twice, once, atol=3e-14, rtol=3e-14)

    def test_projection_is_self_adjoint(self) -> None:
        left = _metric(self.projector.project(self.X), self.Y)
        right = _metric(self.X, self.projector.project(self.Y))
        self.assertAlmostEqual(left, right, places=12)

    def test_matches_full_horizontal_basis_reference(self) -> None:
        horizontal_basis = gauge_orthogonal_basis(
            self.A, self.W, tolerance=1e-10
        )
        tangent = stiefel_project(self.W, self.X)
        reference = sum(
            (_metric(Q, tangent) * Q for Q in horizontal_basis),
            np.zeros_like(self.W),
        )
        actual = self.projector.project(self.X)
        np.testing.assert_allclose(actual, reference, atol=2e-13, rtol=2e-13)

    def test_projector_does_not_use_finite_window_information(self) -> None:
        # Construction depends on A and the true gauge only: it has neither a
        # block length nor an RDM/Jacobian argument.  Its rank therefore stays
        # the same if a later optimizer changes the observed window length.
        self.assertFalse(hasattr(self.projector, "block_length"))
        self.assertEqual(
            self.projector.gauge_columns.shape,
            (vectorize_complex_real(stack_tensor(self.A)).size, 9),
        )

    def test_historical_representative_is_gauge_equivalent(self) -> None:
        horizontal = self.projector.project(self.X)
        historical = self.projector.historical_slice_representative(horizontal)
        self.assertLess(
            np.linalg.norm(self.W.conj().T @ historical), 2e-12
        )
        np.testing.assert_allclose(
            self.projector.project(historical),
            horizontal,
            atol=3e-13,
            rtol=3e-13,
        )


class GrassmannHorizontalProjectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.A, self.W = random_left_canonical(d=2, D=3, seed=1311)
        self.projector = GrassmannHorizontalProjector.from_tensor(
            self.A, self.W
        )
        rng = np.random.default_rng(1312)
        self.X = rng.normal(size=self.W.shape) + 1j * rng.normal(
            size=self.W.shape
        )
        self.Y = rng.normal(size=self.W.shape) + 1j * rng.normal(
            size=self.W.shape
        )

    def test_output_satisfies_tdvp_left_gauge_condition(self) -> None:
        horizontal = self.projector.project(self.X)
        self.assertLess(np.linalg.norm(self.W.conj().T @ horizontal), 2e-13)
        self.assertLess(self.projector.tangency_error(horizontal), 3e-13)

    def test_projection_is_idempotent_and_self_adjoint(self) -> None:
        once = self.projector.project(self.X)
        twice = self.projector.project(once)
        np.testing.assert_allclose(twice, once, atol=3e-14, rtol=3e-14)
        self.assertAlmostEqual(
            _metric(self.projector.project(self.X), self.Y),
            _metric(self.X, self.projector.project(self.Y)),
            places=12,
        )

    def test_matches_historical_representative_of_true_horizontal_vector(self) -> None:
        true_projector = GaugeHorizontalProjector.from_tensor(self.A, self.W)
        true_horizontal = true_projector.project(self.X)
        historical = true_projector.historical_slice_representative(
            true_horizontal
        )
        self.assertLess(np.linalg.norm(self.W.conj().T @ historical), 3e-12)
        np.testing.assert_allclose(
            self.projector.project(historical),
            historical,
            atol=3e-12,
            rtol=3e-12,
        )
        np.testing.assert_allclose(
            true_projector.project(historical),
            true_horizontal,
            atol=3e-12,
            rtol=3e-12,
        )
        # The true-horizontal representative is the orthogonal, hence
        # minimum-Frobenius-norm, member of its gauge-equivalence class.
        self.assertGreaterEqual(
            np.linalg.norm(historical) + 2e-13,
            np.linalg.norm(true_horizontal),
        )
        gauge_addition = historical - true_horizontal
        self.assertAlmostEqual(
            float(np.linalg.norm(historical) ** 2),
            float(
                np.linalg.norm(true_horizontal) ** 2
                + np.linalg.norm(gauge_addition) ** 2
            ),
            places=11,
        )
        # A naive orthogonal projection into the Grassmann slice generally
        # changes the represented physical tangent; the correct change of
        # slice must add a true uMPS gauge direction as above.
        self.assertGreater(
            np.linalg.norm(self.projector.project(true_horizontal) - historical),
            1e-6,
        )


if __name__ == "__main__":
    unittest.main()
