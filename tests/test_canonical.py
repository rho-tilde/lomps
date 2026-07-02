from __future__ import annotations

import unittest

import numpy as np

from lomps.canonical import (
    canonical_errors,
    linearized_stiefel_error,
    orthonormal_complement,
    random_left_canonical,
)
from lomps.tangent import antihermitian_basis, tangent_bases


class CanonicalTests(unittest.TestCase):
    def test_random_left_canonical(self) -> None:
        A, W = random_left_canonical(d=2, D=2, seed=1)
        errors = canonical_errors(A)
        self.assertLess(errors["stiefel_error"], 1e-13)
        self.assertLess(errors["left_canonical_error"], 1e-13)
        self.assertLess(np.linalg.norm(W.conj().T @ W - np.eye(2)), 1e-13)

    def test_complement_and_tangent_bases(self) -> None:
        _, W = random_left_canonical(d=2, D=2, seed=2)
        W_perp = orthonormal_complement(W)
        self.assertEqual(W_perp.shape, (4, 2))
        self.assertLess(np.linalg.norm(W.conj().T @ W_perp), 1e-13)
        bases = tangent_bases(W, d=2, D=2)
        self.assertEqual(len(bases.parallel), 4)
        self.assertEqual(len(bases.perp), 8)
        self.assertEqual(len(bases.full), 12)
        for delta_W in bases.full:
            self.assertLess(linearized_stiefel_error(W, delta_W), 1e-12)

    def test_antihermitian_basis(self) -> None:
        basis = antihermitian_basis(3)
        self.assertEqual(len(basis), 9)
        for X in basis:
            self.assertLess(np.linalg.norm(X + X.conj().T), 1e-14)


if __name__ == "__main__":
    unittest.main()
