from __future__ import annotations

import unittest

from lomps.dimensions import (
    minimum_bond_dimension_for_ti_rdm,
    quotient_tangent_dimension,
    translation_invariant_rdm_dimension,
)


class DimensionCountTests(unittest.TestCase):
    def test_translation_invariant_rdm_dimension(self) -> None:
        self.assertEqual(translation_invariant_rdm_dimension(2, 1), 3)
        self.assertEqual(translation_invariant_rdm_dimension(2, 3), 48)
        self.assertEqual(translation_invariant_rdm_dimension(2, 4), 192)
        self.assertEqual(translation_invariant_rdm_dimension(3, 2), 72)

    def test_quotient_tangent_dimension(self) -> None:
        self.assertEqual(quotient_tangent_dimension(2, 5), 50)
        self.assertEqual(quotient_tangent_dimension(2, 10), 200)
        self.assertEqual(quotient_tangent_dimension(3, 3), 36)

    def test_minimum_bond_dimension_for_ti_rdm(self) -> None:
        self.assertEqual(minimum_bond_dimension_for_ti_rdm(2, 1), 2)
        self.assertEqual(minimum_bond_dimension_for_ti_rdm(2, 3), 5)
        self.assertEqual(minimum_bond_dimension_for_ti_rdm(2, 4), 10)
        self.assertEqual(minimum_bond_dimension_for_ti_rdm(3, 2), 5)

    def test_invalid_dimensions_raise(self) -> None:
        with self.assertRaises(ValueError):
            translation_invariant_rdm_dimension(1, 3)
        with self.assertRaises(ValueError):
            translation_invariant_rdm_dimension(2, 0)
        with self.assertRaises(ValueError):
            quotient_tangent_dimension(2, 0)


if __name__ == "__main__":
    unittest.main()
