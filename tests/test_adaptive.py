import argparse
import tempfile
import unittest
from pathlib import Path

import numpy as np

from lomps.adaptive import (
    child_command,
    extract_accepted_checkpoint,
    parse_bond_dimensions,
)


class AdaptiveBondDimensionTests(unittest.TestCase):
    def test_parse_bond_dimensions_requires_strict_increase(self) -> None:
        self.assertEqual(parse_bond_dimensions("4,6,8,12"), (4, 6, 8, 12))
        with self.assertRaises(ValueError):
            parse_bond_dimensions("4,4,8")
        with self.assertRaises(ValueError):
            parse_bond_dimensions("8,6")

    def test_extract_accepted_checkpoint_uses_completed_index(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            states = np.arange(3 * 2 * 2 * 2).reshape(3, 2, 2, 2)
            fixed_points = np.arange(3 * 2 * 2).reshape(3, 2, 2)
            np.save(root / "states.npy", states)
            np.save(root / "right_fixed_points.npy", fixed_points)
            tensor_path = root / "A.npy"
            fixed_point_path = root / "r.npy"
            extract_accepted_checkpoint(root, 2, tensor_path, fixed_point_path)
            np.testing.assert_array_equal(np.load(tensor_path), states[2])
            np.testing.assert_array_equal(np.load(fixed_point_path), fixed_points[2])

    def test_handoff_command_uses_separate_threshold(self) -> None:
        args = argparse.Namespace(
            protocol="nonintegrable-ising",
            block_length=6,
            delta_t=1e-3,
            accept_cost=1e-14,
            embedding_noise_amplitude=1e-6,
            embedding_seed=104729,
            first_step_optimizer="lm",
            first_step_cg_seconds=300.0,
            fixed_point_solver="dense",
            target_source_fixed_point_solver="dense",
            target_contraction="tensor",
            perturb_amplitudes="0.1,0.3",
            perturbations_per_amplitude=1,
            random_restarts=0,
            random_seed=2,
            checkpoint_every=25,
            strict_retry=True,
            initial_key=None,
            g=None,
            h=None,
            J=None,
            trotter_order=None,
            symmetric_transverse=None,
        )
        command = child_command(
            args,
            source=Path("A.npy"),
            source_layout="physical-left-right",
            segment_dir=Path("segment"),
            bond_dimension=12,
            base_time=0.5,
            steps=10,
            first_step_accept_cost=7e-15,
            include_initial_key=False,
        )
        threshold_index = command.index("--first-step-accept-cost") + 1
        self.assertEqual(float(command[threshold_index]), 7e-15)
        optimizer_index = command.index("--first-step-optimizer") + 1
        self.assertEqual(command[optimizer_index], "lm")
        self.assertIn("--strict-retry", command)


if __name__ == "__main__":
    unittest.main()
