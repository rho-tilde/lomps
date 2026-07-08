from __future__ import annotations

import unittest

import numpy as np

from lomps.canonical import canonical_errors, random_left_canonical
from lomps.embedding import (
    coerce_initial_tensor,
    lift_left_canonical_seed,
    product_tensor,
)
from lomps.rdm import block_rdm


class InitialEmbeddingTests(unittest.TestCase):
    def test_product_vector_becomes_normalized_d1_tensor(self) -> None:
        tensor = product_tensor(np.array([2.0, 2.0j]))
        self.assertEqual(tensor.shape, (2, 1, 1))
        self.assertLess(canonical_errors(tensor)["left_canonical_error"], 1e-14)

    def test_coerce_initial_tensor_accepts_singleton_batch(self) -> None:
        A, _ = random_left_canonical(d=2, D=3, seed=91)
        actual = coerce_initial_tensor(A[None, ...])
        np.testing.assert_array_equal(actual, A)

    def test_lift_is_deterministic_and_left_canonical(self) -> None:
        source = product_tensor(np.array([1.0, 1.0j]))
        first, first_diag = lift_left_canonical_seed(
            source,
            4,
            noise_amplitude=1e-4,
            seed=92,
            diagnostic_block_length=2,
        )
        second, second_diag = lift_left_canonical_seed(
            source,
            4,
            noise_amplitude=1e-4,
            seed=92,
            diagnostic_block_length=2,
        )
        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(first.shape, (2, 4, 4))
        self.assertLess(canonical_errors(first)["left_canonical_error"], 1e-13)
        self.assertEqual(first_diag, second_diag)
        self.assertEqual(first_diag.source_bond_dimension, 1)
        self.assertEqual(first_diag.target_bond_dimension, 4)
        self.assertIn("2", first_diag.seed_rdm_frobenius_error_by_length)

    def test_equal_bond_dimension_returns_source_copy(self) -> None:
        source, _ = random_left_canonical(d=2, D=3, seed=93)
        lifted, diagnostics = lift_left_canonical_seed(
            source,
            3,
            diagnostic_block_length=2,
        )
        np.testing.assert_array_equal(lifted, source)
        self.assertEqual(diagnostics.noise_amplitude, 0.0)
        self.assertEqual(diagnostics.seed_rdm_frobenius_error_by_length["2"], 0.0)

    def test_lift_rejects_non_left_canonical_tensor(self) -> None:
        bad = np.ones((2, 2, 2), dtype=np.complex128)
        with self.assertRaises(ValueError):
            lift_left_canonical_seed(bad, 3)

    def test_lift_rejects_smaller_target_bond_dimension(self) -> None:
        source, _ = random_left_canonical(d=2, D=3, seed=94)
        with self.assertRaises(ValueError):
            lift_left_canonical_seed(source, 2)

    def test_lift_keeps_seed_close_to_source_locally(self) -> None:
        source = product_tensor(np.array([1.0, 1.0j]))
        lifted, _ = lift_left_canonical_seed(
            source,
            4,
            noise_amplitude=1e-6,
            seed=95,
            diagnostic_block_length=2,
        )
        self.assertLess(np.linalg.norm(block_rdm(lifted, 2) - block_rdm(source, 2)), 1e-4)


if __name__ == "__main__":
    unittest.main()
