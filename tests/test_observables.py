from __future__ import annotations

import unittest

import numpy as np

from lomps.canonical import random_left_canonical
from lomps.gates import two_site_hamiltonian_tfim
from lomps.observables import (
    PAULI_X,
    PAULI_Y,
    PAULI_Z,
    local_expectations,
    one_site_rdm,
    trajectory_expectations,
)
from lomps.rdm import block_rdm
from lomps.transfer import right_fixed_point


class ObservableTests(unittest.TestCase):
    def test_one_site_rdm_matches_generic_block_rdm(self) -> None:
        A, _ = random_left_canonical(d=2, D=3, seed=101)
        r, _ = right_fixed_point(A)
        np.testing.assert_allclose(
            one_site_rdm(A, r),
            block_rdm(A, 1, r),
            atol=1e-12,
            rtol=1e-12,
        )

    def test_local_expectations_match_block_rdm_traces(self) -> None:
        A, _ = random_left_canonical(d=2, D=3, seed=102)
        H2 = two_site_hamiltonian_tfim(
            g=1.05,
            h=-0.5,
            J=-1.0,
            symmetric_transverse=False,
        )
        values = local_expectations(
            A,
            {
                "sx": PAULI_X,
                "sy": PAULI_Y,
                "sz": PAULI_Z,
                "energy": H2,
            },
            solver="dense",
        )
        rho1 = block_rdm(A, 1)
        rho2 = block_rdm(A, 2)
        self.assertAlmostEqual(values["sx"], np.trace(rho1 @ PAULI_X).real)
        self.assertAlmostEqual(values["sy"], np.trace(rho1 @ PAULI_Y).real)
        self.assertAlmostEqual(values["sz"], np.trace(rho1 @ PAULI_Z).real)
        self.assertAlmostEqual(values["energy"], np.trace(rho2 @ H2).real)

    def test_trajectory_expectations_shapes(self) -> None:
        states = [random_left_canonical(d=2, D=2, seed=seed)[0] for seed in (1, 2)]
        values = trajectory_expectations(states, {"sx": PAULI_X}, solver="dense")
        self.assertEqual(values["sx"].shape, (2,))


if __name__ == "__main__":
    unittest.main()
