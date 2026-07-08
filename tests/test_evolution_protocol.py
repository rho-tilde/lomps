from __future__ import annotations

from argparse import Namespace
from pathlib import Path
import unittest

import numpy as np

from lomps.canonical import random_left_canonical
from lomps.evolution import infer_block_length, protocol_from_args
from lomps.protocol import NONINTEGRABLE_ISING
from lomps.rdm import block_rdm


class EvolutionProtocolTests(unittest.TestCase):
    def test_configured_l4_protocol_matches_reference_protocol(self) -> None:
        A, _ = random_left_canonical(d=2, D=2, seed=81)
        protocol = protocol_from_args(
            Namespace(block_length=4, odd_parity_warning_threshold=1e-6)
        )
        self.assertEqual(protocol.block_length, NONINTEGRABLE_ISING.block_length)
        self.assertEqual(protocol.lightcone_sites, NONINTEGRABLE_ISING.lightcone_sites)
        self.assertEqual(protocol.target_margins, NONINTEGRABLE_ISING.target_margins)
        np.testing.assert_allclose(
            protocol.target_rdm(A),
            NONINTEGRABLE_ISING.target_rdm(A),
            atol=0.0,
            rtol=0.0,
        )

    def test_configured_l4_protocol_matches_d12_reference_prefix(self) -> None:
        root = Path(__file__).resolve().parents[1]
        initial = np.load(root / "data" / "nonintegrable_d12_t0p001.npy")
        trajectory = np.load(
            root / "data" / "nonintegrable_d12_trajectory" / "trajectory_states.npy",
            mmap_mode="r",
        )
        protocol = protocol_from_args(
            Namespace(block_length=4, odd_parity_warning_threshold=1e-6)
        )

        states = [initial] + [trajectory[index] for index in range(4)]
        for index, A in enumerate(states):
            with self.subTest(index=index):
                np.testing.assert_allclose(
                    protocol.target_rdm(A),
                    NONINTEGRABLE_ISING.target_rdm(A),
                    atol=0.0,
                    rtol=0.0,
                )

    def test_optimizer_block_length_is_inferred_from_target_shape(self) -> None:
        A, _ = random_left_canonical(d=2, D=2, seed=82)
        for block_length in (1, 2, 3, 4):
            with self.subTest(block_length=block_length):
                target = block_rdm(A, block_length)
                self.assertEqual(infer_block_length(A, target), block_length)


if __name__ == "__main__":
    unittest.main()
