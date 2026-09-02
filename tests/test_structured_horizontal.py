from __future__ import annotations

import unittest

import numpy as np

from lomps.canonical import random_left_canonical
from lomps.horizontal import GaugeHorizontalProjector
from lomps.structured_horizontal import StructuredGaugeHorizontalProjector


class StructuredHorizontalTests(unittest.TestCase):
    def test_structured_projection_matches_dense_qr(self) -> None:
        A, W = random_left_canonical(d=2, D=3, seed=1801)
        rng = np.random.default_rng(1802)
        X = rng.normal(size=W.shape) + 1j * rng.normal(size=W.shape)
        dense = GaugeHorizontalProjector.from_tensor(A, W).project(X)
        structured_projector = StructuredGaugeHorizontalProjector(
            A, W, rtol=1e-12
        )
        structured = structured_projector.project(X)
        np.testing.assert_allclose(
            structured, dense, atol=2e-10, rtol=2e-10
        )
        self.assertLess(structured_projector.tangency_error(structured), 2e-12)
        self.assertLess(
            structured_projector.gauge_overlap_norm(structured), 2e-10
        )

    def test_structured_historical_representative(self) -> None:
        A, W = random_left_canonical(d=2, D=3, seed=1811)
        rng = np.random.default_rng(1812)
        X = rng.normal(size=W.shape) + 1j * rng.normal(size=W.shape)
        projector = StructuredGaugeHorizontalProjector(A, W, rtol=1e-12)
        horizontal = projector.project(X)
        historical = projector.historical_slice_representative(horizontal)
        self.assertLess(np.linalg.norm(W.conj().T @ historical), 2e-10)
        np.testing.assert_allclose(
            projector.project(historical),
            horizontal,
            atol=3e-10,
            rtol=3e-10,
        )


if __name__ == "__main__":
    unittest.main()
