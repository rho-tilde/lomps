from __future__ import annotations

from argparse import Namespace
import contextlib
import io
from pathlib import Path
import unittest

import numpy as np

from lomps.canonical import random_left_canonical
from lomps.evolution import (
    infer_block_length,
    options,
    parse_seed_list,
    protocol_from_args,
    select_initial_seed,
)
from lomps.embedding import product_tensor
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

    def test_optimizer_options_thread_fixed_point_policy(self) -> None:
        primary, strict = options(
            accept_cost=3e-16,
            rank_tolerance=1e-12,
            fixed_point_solver="fast",
        )
        self.assertEqual(primary.fixed_point_solver, "fast")
        self.assertEqual(strict.fixed_point_solver, "fast")

    def test_parse_seed_list_deduplicates_and_ignores_blanks(self) -> None:
        self.assertEqual(parse_seed_list(" 2, , 5,2,8 "), (2, 5, 8))
        self.assertEqual(parse_seed_list(""), ())

    def test_same_bond_initial_seed_skips_candidate_screening(self) -> None:
        source, _ = random_left_canonical(d=2, D=3, seed=83)
        primary, _ = options(
            accept_cost=1e-8,
            rank_tolerance=1e-12,
            fixed_point_solver="dense",
        )
        seed, diagnostics, screen = select_initial_seed(
            source,
            3,
            protocol=NONINTEGRABLE_ISING,
            primary=primary,
            embedding_noise_amplitude=1e-4,
            embedding_seed=11,
            embedding_candidate_seeds=(1, 2, 3),
            embedding_screen_max_iterations=0,
            embedding_screen_seconds=0.0,
            embedding_screen_fixed_point_solver="same",
            canonical_tolerance=1e-10,
        )
        np.testing.assert_array_equal(seed, source)
        self.assertEqual(screen, [])
        self.assertEqual(diagnostics.seed, 11)

    def test_low_bond_initial_seed_screening_selects_candidate(self) -> None:
        source = product_tensor(np.array([1.0, 1.0j]))
        primary, _ = options(
            accept_cost=1e-8,
            rank_tolerance=1e-12,
            fixed_point_solver="dense",
        )
        with contextlib.redirect_stdout(io.StringIO()):
            seed, diagnostics, screen = select_initial_seed(
                source,
                2,
                protocol=NONINTEGRABLE_ISING,
                primary=primary,
                embedding_noise_amplitude=1e-4,
                embedding_seed=11,
                embedding_candidate_seeds=(21, 22),
                embedding_screen_max_iterations=0,
                embedding_screen_seconds=0.0,
                embedding_screen_fixed_point_solver="same",
                canonical_tolerance=1e-10,
            )
        self.assertEqual(seed.shape, (2, 2, 2))
        self.assertEqual(len(screen), 2)
        self.assertIn(diagnostics.seed, {21, 22})
        self.assertEqual(sum(row["selected"] for row in screen), 1)


if __name__ == "__main__":
    unittest.main()
