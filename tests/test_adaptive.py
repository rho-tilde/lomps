import argparse
import tempfile
import unittest
from pathlib import Path

import numpy as np

from lomps.adaptive import (
    RESUMABLE_STATUSES,
    adaptive_policy,
    child_command,
    code_provenance,
    extract_accepted_checkpoint,
    optimizer_policy,
    parse_bond_dimensions,
    resume_context,
)


class AdaptiveBondDimensionTests(unittest.TestCase):
    def test_pause_statuses_are_resumable_not_promotable(self) -> None:
        self.assertIn("paused_by_PAUSE_file", RESUMABLE_STATUSES)
        self.assertIn("paused_by_step_runtime_limit", RESUMABLE_STATUSES)
        self.assertIn("paused_by_keyboard_interrupt", RESUMABLE_STATUSES)

    def test_runtime_request_is_promotable(self) -> None:
        from lomps.adaptive import PROMOTABLE_STATUSES

        self.assertIn("promote_requested_by_runtime", PROMOTABLE_STATUSES)

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
            time_predictor="secant",
            perturb_amplitudes="0.1,0.3",
            perturbations_per_amplitude=1,
            random_restarts=0,
            random_seed=2,
            checkpoint_every=25,
            strict_retry=True,
            initial_key=None,
            initial_seed_lift_noise_amplitude=1e-6,
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
        predictor_index = command.index("--time-predictor") + 1
        self.assertEqual(command[predictor_index], "secant")

        args.optimizer = "matrix-free-lm"
        args.matrix_free_krylov_initial_iterations = 64
        args.matrix_free_krylov_solver = "lsmr"
        args.matrix_free_krylov_max_iterations = 256
        args.matrix_free_krylov_preconditioner = (
            "right-fixed-point-stiefel"
        )
        args.matrix_free_krylov_relative_tolerance = 0.1
        args.matrix_free_krylov_minimum_relative_tolerance = 1e-6
        args.matrix_free_adaptive_krylov_tolerance = True
        args.matrix_free_recycle_krylov_solution = False
        args.matrix_free_lsmr_condition_limit = 1e12
        args.matrix_free_fixed_point_response_solver = "auto"
        args.matrix_free_fixed_point_response_max_mb = 128.0
        args.matrix_free_adjoint_rtol = 1e-8
        args.matrix_free_jvp_rtol = 1e-8
        args.matrix_free_verbose = True
        matrix_free_command = child_command(
            args,
            source=Path("A.npy"),
            source_layout="physical-left-right",
            segment_dir=Path("segment-mf"),
            bond_dimension=40,
            base_time=0.5,
            steps=10,
            first_step_accept_cost=7e-15,
            include_initial_key=False,
        )
        backend_index = matrix_free_command.index("--optimizer") + 1
        self.assertEqual(matrix_free_command[backend_index], "matrix-free-lm")
        maximum_index = (
            matrix_free_command.index(
                "--matrix-free-krylov-max-iterations"
            )
            + 1
        )
        self.assertEqual(matrix_free_command[maximum_index], "256")
        preconditioner_index = (
            matrix_free_command.index(
                "--matrix-free-krylov-preconditioner"
            )
            + 1
        )
        self.assertEqual(
            matrix_free_command[preconditioner_index],
            "right-fixed-point-stiefel",
        )
        self.assertIn(
            "--matrix-free-adaptive-krylov-tolerance", matrix_free_command
        )
        self.assertEqual(
            optimizer_policy(args),
            {
                "backend": "matrix-free-lm",
                "krylov_solver": "lsmr",
                "krylov_initial_iterations": 64,
                "krylov_max_iterations": 256,
                "krylov_preconditioner": "right-fixed-point-stiefel",
                "krylov_relative_tolerance": 0.1,
                "krylov_minimum_relative_tolerance": 1e-6,
                "adaptive_krylov_tolerance": True,
                "recycle_krylov_solution": False,
                "lsmr_condition_limit": 1e12,
                "fixed_point_response_solver": "auto",
                "fixed_point_response_max_mb": 128.0,
                "adjoint_rtol": 1e-8,
                "jvp_fixed_point_rtol": 1e-8,
                "verbose": True,
            },
        )

    def test_child_command_can_resume_existing_segment(self) -> None:
        args = argparse.Namespace(
            protocol="nonintegrable-ising",
            block_length=6,
            delta_t=1e-3,
            accept_cost=1e-12,
            embedding_noise_amplitude=1e-6,
            embedding_seed=104729,
            first_step_optimizer="lm",
            first_step_cg_seconds=900.0,
            fixed_point_solver="dense",
            target_source_fixed_point_solver="dense",
            target_contraction="tensor",
            time_predictor="secant",
            perturb_amplitudes="0.1,0.3,0.6,1.0",
            perturbations_per_amplitude=1,
            random_restarts=0,
            random_seed=20260702,
            checkpoint_every=25,
            strict_retry=False,
            initial_key=None,
            initial_seed_lift_noise_amplitude=1e-6,
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
            bond_dimension=40,
            base_time=4.144,
            steps=15856,
            first_step_accept_cost=1e-12,
            include_initial_key=False,
            initial_seed=Path("seed.npy"),
            resume=True,
        )
        self.assertIn("--resume", command)
        lift_noise_index = command.index("--initial-seed-lift-noise-amplitude") + 1
        self.assertEqual(command[lift_noise_index], "1e-06")

    def test_matrix_free_command_records_dense_handoff_and_rescue(self) -> None:
        from lomps.adaptive import build_parser

        args = build_parser().parse_args(
            [
                "--initial-A", "initial.npy",
                "--output-dir", "output",
                "--bond-dimensions", "30,32",
                "--block-length", "6",
                "--steps", "100",
                "--optimizer", "matrix-free-lm",
                "--matrix-free-dense-rescue",
                "--matrix-free-dense-rescue-seconds", "7200",
                "--matrix-free-dense-rescue-iteration-cap", "100",
                "--lifted-first-step-backend", "dense-lm",
            ]
        )
        command = child_command(
            args,
            source=Path("A.npy"),
            source_layout="physical-left-right",
            segment_dir=Path("segment"),
            bond_dimension=32,
            base_time=1.0,
            steps=10,
            first_step_accept_cost=1e-12,
            include_initial_key=False,
        )
        self.assertIn("--matrix-free-dense-rescue", command)
        self.assertEqual(
            command[command.index("--matrix-free-dense-rescue-seconds") + 1],
            "7200.0",
        )
        self.assertEqual(
            command[command.index("--lifted-first-step-backend") + 1],
            "dense-lm",
        )
        self.assertEqual(
            optimizer_policy(args)["dense_rescue"],
            {"maximum_seconds": 7200.0, "iteration_cap": 100},
        )
        self.assertEqual(
            adaptive_policy(args)["lifted_first_step_backend"], "dense-lm"
        )

    def test_dense_lm_production_policy_reaches_every_child(self) -> None:
        from lomps.adaptive import build_parser

        args = build_parser().parse_args(
            [
                "--initial-A", "initial.npy",
                "--output-dir", "output",
                "--bond-dimensions", "36,41",
                "--block-length", "6",
                "--steps", "100",
                "--fixed-point-solver", "fast",
                "--rank-tolerance", "2e-13",
                "--dense-verify-acceptance",
                "--lm-linear-solver", "normal",
                "--lm-tangent-slice", "grassmann",
                "--lm-initial-damping", "1e-10",
                "--lm-rdm-vectorization", "hermitian",
                "--lm-jacobian-response-solver", "dense-lu",
                "--lm-jacobian-workers", "4",
                "--maximum-step-seconds", "600",
            ]
        )
        command = child_command(
            args,
            source=Path("A.npy"),
            source_layout="physical-left-right",
            segment_dir=Path("segment"),
            bond_dimension=41,
            base_time=1.0,
            steps=10,
            first_step_accept_cost=1e-12,
            include_initial_key=False,
        )

        def option(name: str) -> str:
            return command[command.index(name) + 1]

        self.assertEqual(option("--fixed-point-solver"), "fast")
        self.assertEqual(option("--rank-tolerance"), "2e-13")
        self.assertIn("--dense-verify-acceptance", command)
        self.assertEqual(option("--lm-linear-solver"), "normal")
        self.assertEqual(option("--lm-tangent-slice"), "grassmann")
        self.assertEqual(option("--lm-initial-damping"), "1e-10")
        self.assertEqual(option("--lm-rdm-vectorization"), "hermitian")
        self.assertEqual(option("--lm-jacobian-response-solver"), "dense-lu")
        self.assertEqual(option("--lm-jacobian-workers"), "4")
        self.assertEqual(option("--maximum-step-seconds"), "600.0")

        policy = adaptive_policy(args)
        self.assertEqual(policy["rank_tolerance"], 2e-13)
        self.assertTrue(policy["dense_verify_acceptance"])
        self.assertEqual(policy["lm_tangent_slice"], "grassmann")
        self.assertEqual(policy["lm_initial_damping"], 1e-10)
        self.assertEqual(policy["lm_rdm_vectorization"], "hermitian")
        self.assertEqual(policy["lm_jacobian_response_solver"], "dense-lu")
        self.assertEqual(policy["lm_jacobian_workers"], 4)

    def test_fast_promotion_removes_below_threshold_retries_only(self) -> None:
        from lomps.adaptive import build_parser

        args = build_parser().parse_args(
            [
                "--initial-A", "initial.npy",
                "--output-dir", "output",
                "--bond-dimensions", "4,40",
                "--block-length", "6",
                "--steps", "100",
                "--time-predictor", "secant",
                "--perturb-amplitudes", "0.1,0.3",
                "--perturbations-per-amplitude", "2",
                "--random-restarts", "3",
                "--strict-retry",
                "--fast-promote-below-bond-dimension", "40",
                "--fast-promote-optimizer-seconds", "30",
                "--optimizer-max-seconds", "600",
                "--first-step-lm-seconds", "900",
                "--matrix-free-from-bond-dimension", "40",
            ]
        )
        common = {
            "source": Path("A.npy"),
            "source_layout": "physical-left-right",
            "base_time": 0.0,
            "steps": 10,
            "first_step_accept_cost": 1e-14,
            "include_initial_key": False,
        }
        below = child_command(
            args,
            segment_dir=Path("D4"),
            bond_dimension=4,
            **common,
        )
        at_threshold = child_command(
            args,
            segment_dir=Path("D40"),
            bond_dimension=40,
            **common,
        )

        def option(command: list[str], name: str) -> str:
            return command[command.index(name) + 1]

        self.assertEqual(option(below, "--time-predictor"), "warm")
        self.assertEqual(option(below, "--perturbations-per-amplitude"), "0")
        self.assertEqual(option(below, "--random-restarts"), "0")
        self.assertIn("--no-strict-retry", below)
        self.assertEqual(option(below, "--optimizer-max-seconds"), "30.0")
        self.assertEqual(option(below, "--first-step-lm-seconds"), "900.0")
        self.assertEqual(option(below, "--optimizer"), "dense-lm")
        self.assertEqual(option(at_threshold, "--time-predictor"), "secant")
        self.assertEqual(
            option(at_threshold, "--perturbations-per-amplitude"), "2"
        )
        self.assertEqual(option(at_threshold, "--random-restarts"), "3")
        self.assertIn("--strict-retry", at_threshold)
        self.assertEqual(option(at_threshold, "--optimizer-max-seconds"), "600.0")
        self.assertEqual(option(at_threshold, "--first-step-lm-seconds"), "900.0")
        self.assertEqual(option(at_threshold, "--optimizer"), "matrix-free-lm")
        self.assertEqual(
            adaptive_policy(args)["fast_promote_below_bond_dimension"], 40
        )
        self.assertEqual(
            optimizer_policy(args)["matrix_free_from_bond_dimension"], 40
        )

    def test_resume_context_finds_unrecorded_active_segment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            initial = root / "initial.npy"
            initial_seed = root / "seed.npy"
            np.save(initial, np.zeros((2, 36, 36)))
            np.save(initial_seed, np.zeros((2, 36, 36)))
            handoffs = root / "handoffs"
            handoffs.mkdir()
            handoff = handoffs / "handoff_000_D36.npy"
            np.save(handoff, np.zeros((2, 36, 36)))
            failed_seed = root / "segment_000_D36" / "failed_best_step_000011.npy"
            failed_seed.parent.mkdir()
            np.save(failed_seed, np.ones((2, 36, 36)))
            (root / "segment_001_D37").mkdir()
            args = argparse.Namespace(
                initial_A=initial,
                initial_seed_A=initial_seed,
                initial_layout="physical-left-right",
                block_length=6,
                delta_t=1e-3,
                steps=100,
                base_time=2.0,
            )
            manifest = {
                "initial_A": str(initial.resolve()),
                "initial_seed_A": str(initial_seed.resolve()),
                "bond_dimensions": [36, 37, 38],
                "block_length": 6,
                "delta_t": 1e-3,
                "requested_steps": 100,
                "base_time": 2.0,
                "segments": [
                    {
                        "index": 0,
                        "bond_dimension": 36,
                        "completed_steps": 10,
                        "handoff_tensor": "handoffs/handoff_000_D36.npy",
                        "promotion_seed_tensor": (
                            "segment_000_D36/failed_best_step_000011.npy"
                        ),
                    }
                ],
            }
            context = resume_context(args, (36, 37, 38), root, manifest)
            self.assertEqual(context[0], handoff)
            self.assertEqual(context[1], "physical-left-right")
            self.assertEqual(context[2], failed_seed)
            self.assertEqual(context[3], 10)
            self.assertAlmostEqual(context[4], 2.01)
            self.assertEqual(context[5], 1)
            self.assertTrue(context[6])

    def test_resume_context_rejects_changed_frozen_policy(self) -> None:
        parser_args = [
            "--initial-A", "initial.npy",
            "--output-dir", "output",
            "--bond-dimensions", "4,6",
            "--block-length", "6",
            "--steps", "100",
            "--optimizer", "matrix-free-lm",
        ]
        from lomps.adaptive import build_parser

        args = build_parser().parse_args(parser_args)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            initial = root / "initial.npy"
            np.save(initial, np.array([1.0, 0.0]))
            args.initial_A = initial
            manifest = {
                "initial_A": str(initial.resolve()),
                "initial_seed_A": None,
                "bond_dimensions": [4, 6],
                "block_length": 6,
                "delta_t": 1e-3,
                "requested_steps": 100,
                "base_time": 0.0,
                "segments": [],
                "optimizer": optimizer_policy(args),
                "adaptive_policy": adaptive_policy(args),
                "code_provenance": code_provenance(),
            }
            manifest["adaptive_policy"]["accept_cost"] = 1e-12
            with self.assertRaisesRegex(ValueError, "policy differs"):
                resume_context(args, (4, 6), root, manifest)


if __name__ == "__main__":
    unittest.main()
