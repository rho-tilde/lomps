from __future__ import annotations

from argparse import Namespace
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from lomps.canonical import random_left_canonical
from lomps.evolution import (
    effective_first_step_accept_cost,
    effective_initial_seed_lift_noise,
    first_step_cg_options,
    fit_fixed_target,
    horizontal_cg_options,
    infer_block_length,
    main,
    matrix_free_options,
    options,
    parse_args,
    parse_seed_list,
    protocol_from_args,
    runtime_promotion_diagnostic,
    run_fixed_target_optimizer,
    select_external_initial_seed,
    select_initial_seed,
    step_limit_reached,
    step_runtime_limit_exceeded,
)
from lomps.embedding import product_tensor
from lomps.horizontal_cg import HorizontalCGOptions
from lomps.protocol import INTEGRABLE_TFIM, NONINTEGRABLE_ISING
from lomps.rdm import block_rdm


def protocol_args(block_length: int = 4) -> Namespace:
    return Namespace(
        protocol="nonintegrable-ising",
        block_length=block_length,
        delta_t=NONINTEGRABLE_ISING.delta_t,
        trotter_order=None,
        g=None,
        h=None,
        J=None,
        symmetric_transverse=None,
        odd_parity_warning_threshold=1e-6,
        target_contraction=NONINTEGRABLE_ISING.target_contraction,
        target_source_fixed_point_solver=(
            NONINTEGRABLE_ISING.target_source_fixed_point_solver
        ),
    )


class EvolutionProtocolTests(unittest.TestCase):
    def test_runtime_promotion_uses_preceding_rolling_median(self) -> None:
        history = [20.0, 22.0, 21.0, 24.0, 23.0, 25.0, 22.0, 23.0]
        diagnostic = runtime_promotion_diagnostic(
            65.0,
            history,
            factor=2.5,
            window=8,
            minimum_seconds=60.0,
        )
        self.assertIsNotNone(diagnostic)
        assert diagnostic is not None
        self.assertTrue(diagnostic["triggered"])
        self.assertFalse(
            runtime_promotion_diagnostic(
                55.0,
                history,
                factor=2.5,
                window=8,
                minimum_seconds=60.0,
            )["triggered"]
        )
        self.assertIsNone(
            runtime_promotion_diagnostic(
                100.0,
                history[:7],
                factor=2.5,
                window=8,
                minimum_seconds=60.0,
            )
        )

    def test_same_dimension_external_seed_preserves_physical_predecessor(self) -> None:
        source, _ = random_left_canonical(d=2, D=2, seed=780)
        seed, _ = random_left_canonical(d=2, D=2, seed=781)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_path = root / "source.npy"
            seed_path = root / "seed.npy"
            output = root / "run"
            np.save(source_path, source)
            np.save(seed_path, seed)
            argv = [
                "lomps-evolve",
                "--initial-A", str(source_path),
                "--initial-layout", "physical-left-right",
                "--initial-seed-A", str(seed_path),
                "--initial-seed-layout", "physical-left-right",
                "--output-dir", str(output),
                "--steps", "1",
                "--base-time", "0.0",
                "--block-length", "1",
                "--accept-cost", "10",
                "--first-step-accept-cost", "10",
                "--perturbations-per-amplitude", "0",
                "--random-restarts", "0",
                "--no-strict-retry",
                "--checkpoint-every", "1",
            ]
            with patch("sys.argv", argv), contextlib.redirect_stdout(io.StringIO()):
                main()

            states = np.load(output / "states.npy")
            np.testing.assert_allclose(states[0], source, atol=0.0, rtol=0.0)
            metadata = json.loads((output / "metadata.json").read_text())
            self.assertEqual(metadata["states_zero_kind"], "initial_source")
            self.assertTrue(
                metadata["policy"][
                    "same_dimension_external_seed_preserves_source_history"
                ]
            )

    def test_strict_retry_continues_from_best_primary_iterate(self) -> None:
        seed = np.zeros((2, 1, 1), dtype=np.complex128)
        primary_A = np.ones_like(seed)
        strict_A = 2.0 * np.ones_like(seed)
        target = np.zeros((2, 2), dtype=np.complex128)
        primary = SimpleNamespace(cost_tolerance=1e-6, fixed_point_solver="dense")
        strict = SimpleNamespace(cost_tolerance=1e-6, fixed_point_solver="dense")
        primary_result = SimpleNamespace(cost=1e-4)
        strict_result = SimpleNamespace(cost=1e-8)

        with (
            patch(
                "lomps.evolution.run_fixed_target_optimizer",
                side_effect=((primary_A, primary_result), (strict_A, strict_result)),
            ) as optimize,
            patch(
                "lomps.evolution.optimizer_right_fixed_point",
                return_value=(np.ones((1, 1)), {}),
            ),
        ):
            best_A, best, used_strict, _ = fit_fixed_target(
                seed,
                target,
                primary,
                strict,
            )

        np.testing.assert_array_equal(optimize.call_args_list[0].args[0], seed)
        np.testing.assert_array_equal(
            optimize.call_args_list[1].args[0], primary_A
        )
        np.testing.assert_array_equal(best_A, strict_A)
        self.assertIs(best, strict_result)
        self.assertTrue(used_strict)

    def test_execution_step_limit_is_exact_and_optional(self) -> None:
        self.assertFalse(step_limit_reached(181, 0))
        self.assertFalse(step_limit_reached(180, 181))
        self.assertTrue(step_limit_reached(181, 181))
        self.assertTrue(step_limit_reached(182, 181))

    def test_step_runtime_limit_is_strict_and_optional(self) -> None:
        self.assertFalse(step_runtime_limit_exceeded(181.0, 0.0))
        self.assertFalse(step_runtime_limit_exceeded(180.0, 180.0))
        self.assertTrue(step_runtime_limit_exceeded(180.01, 180.0))

    def test_configured_l4_protocol_matches_reference_protocol(self) -> None:
        A, _ = random_left_canonical(d=2, D=2, seed=81)
        protocol = protocol_from_args(protocol_args())
        self.assertEqual(protocol.block_length, NONINTEGRABLE_ISING.block_length)
        self.assertEqual(protocol.lightcone_sites, NONINTEGRABLE_ISING.lightcone_sites)
        self.assertEqual(protocol.target_margins, NONINTEGRABLE_ISING.target_margins)
        self.assertEqual(protocol.target_contraction, "tensor")
        self.assertEqual(protocol.target_source_fixed_point_solver, "dense")
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
        protocol = protocol_from_args(protocol_args())

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
            linear_solver="normal",
        )
        self.assertEqual(primary.fixed_point_solver, "fast")
        self.assertEqual(strict.fixed_point_solver, "fast")
        self.assertEqual(primary.linear_solver, "normal")
        self.assertEqual(strict.linear_solver, "normal")

    def test_dense_lm_coordinate_controls_reach_both_option_sets(self) -> None:
        primary, strict = options(
            accept_cost=1e-12,
            rank_tolerance=1e-12,
            fixed_point_solver="fast",
            linear_solver="normal",
            tangent_slice="grassmann",
            initial_damping=1e-10,
            rdm_vectorization="hermitian",
        )
        for controls in (primary, strict):
            self.assertEqual(controls.tangent_slice, "grassmann")
            self.assertEqual(controls.initial_damping, 1e-10)
            self.assertEqual(controls.rdm_vectorization, "hermitian")

    def test_optimizer_options_use_separate_wall_time_caps(self) -> None:
        primary, strict = options(
            accept_cost=3e-16,
            rank_tolerance=1e-12,
            maximum_seconds=30.0,
        )
        self.assertEqual(primary.maximum_seconds, 30.0)
        self.assertEqual(strict.maximum_seconds, 30.0)
        matrix_free_primary, matrix_free_strict = matrix_free_options(
            3e-16,
            1e-12,
            "dense",
            krylov_initial_iterations=64,
            krylov_max_iterations=256,
            krylov_relative_tolerance=0.1,
            krylov_minimum_relative_tolerance=1e-6,
            adaptive_krylov_tolerance=True,
            adjoint_rtol=1e-8,
            jvp_fixed_point_rtol=1e-8,
            maximum_seconds=900.0,
        )
        self.assertEqual(matrix_free_primary.maximum_seconds, 900.0)
        self.assertEqual(matrix_free_strict.maximum_seconds, 900.0)
        horizontal_primary, horizontal_strict = horizontal_cg_options(
            3e-16,
            "dense",
            maximum_seconds=450.0,
        )
        self.assertEqual(horizontal_primary.maximum_seconds, 450.0)
        self.assertEqual(horizontal_strict.maximum_seconds, 450.0)
        self.assertEqual(
            horizontal_primary.preconditioner,
            "grassmann_right_fixed_point",
        )
        self.assertEqual(
            horizontal_primary.conjugacy_metric,
            "historical_slice",
        )
        self.assertEqual(horizontal_primary.line_search_interpolation, "secant")
        self.assertEqual(horizontal_strict.gradient_tolerance, 1e-13)

    def test_horizontal_cg_dispatch_uses_recurrent_optimizer(self) -> None:
        seed = np.zeros((2, 1, 1), dtype=np.complex128)
        target = np.zeros((2, 2), dtype=np.complex128)
        options = HorizontalCGOptions(cost_tolerance=1e-12, verbose=False)
        expected_A = np.ones_like(seed)
        expected_result = SimpleNamespace(cost=0.0)
        with patch(
            "lomps.evolution.optimize_fixed_target_horizontal_cg",
            return_value=(expected_A, expected_result),
        ) as optimize:
            actual_A, actual_result = run_fixed_target_optimizer(
                seed,
                target,
                1,
                options,
                initial_fixed_point=np.ones((1, 1)),
            )
        np.testing.assert_array_equal(actual_A, expected_A)
        self.assertIs(actual_result, expected_result)
        optimize.assert_called_once_with(seed, target, 1, options)

    def test_parse_seed_list_deduplicates_and_ignores_blanks(self) -> None:
        self.assertEqual(parse_seed_list(" 2, , 5,2,8 "), (2, 5, 8))
        self.assertEqual(parse_seed_list(""), ())

    def test_product_start_defaults_to_production_embedding_seed(self) -> None:
        argv = [
            "lomps-evolve",
            "--initial-A",
            "product.npy",
            "--output-dir",
            "out",
            "--steps",
            "1",
            "--base-time",
            "0.0",
        ]
        with patch("sys.argv", argv):
            args = parse_args()
        self.assertEqual(args.initial_seed_mode, "embedding")
        self.assertEqual(args.embedding_noise_amplitude, 1e-6)
        self.assertIsNone(args.initial_seed_A)
        self.assertIsNone(args.initial_seed_lift_noise_amplitude)
        self.assertEqual(args.target_contraction, "tensor")
        self.assertEqual(args.target_source_fixed_point_solver, "dense")
        self.assertEqual(args.protocol, "nonintegrable-ising")
        self.assertIsNone(args.block_length)
        self.assertIsNone(args.delta_t)
        self.assertIsNone(args.trotter_order)
        self.assertIsNone(args.g)
        self.assertIsNone(args.h)
        self.assertIsNone(args.J)
        self.assertIsNone(args.symmetric_transverse)
        self.assertEqual(args.lm_linear_solver, "normal")
        self.assertEqual(args.optimizer, "dense-lm")
        self.assertEqual(args.optimizer_max_seconds, 600.0)
        self.assertEqual(args.first_step_lm_seconds, 600.0)
        self.assertEqual(args.matrix_free_krylov_initial_iterations, 64)
        self.assertEqual(args.matrix_free_krylov_solver, "cg")
        self.assertEqual(args.matrix_free_krylov_max_iterations, 256)
        self.assertEqual(args.matrix_free_krylov_preconditioner, "none")
        self.assertFalse(args.matrix_free_verbose)
        self.assertFalse(args.matrix_free_recycle_krylov_solution)
        self.assertFalse(args.matrix_free_dense_rescue)
        self.assertEqual(args.matrix_free_dense_rescue_seconds, 3600.0)
        self.assertEqual(args.lifted_first_step_backend, "same")
        self.assertEqual(
            args.horizontal_cg_preconditioner,
            "grassmann-right-fixed-point",
        )
        self.assertEqual(
            args.horizontal_cg_conjugacy_metric,
            "historical-slice",
        )
        self.assertEqual(args.horizontal_cg_line_interpolation, "secant")
        self.assertFalse(args.horizontal_cg_verbose)
        self.assertEqual(
            args.matrix_free_fixed_point_response_solver, "auto"
        )
        self.assertEqual(args.time_predictor, "warm")
        self.assertEqual(args.accept_cost, 1e-14)
        self.assertEqual(args.first_step_accept_cost, 3e-16)
        self.assertEqual(args.first_step_optimizer, "cg-lm")
        self.assertTrue(args.first_step_cg_precondition)
        self.assertFalse(args.first_step_cg_verbose)
        self.assertTrue(args.strict_retry)

    def test_matrix_free_production_options_are_adaptive(self) -> None:
        primary, strict = matrix_free_options(
            1e-14,
            1e-12,
            "fast",
            krylov_initial_iterations=64,
            krylov_max_iterations=256,
            krylov_relative_tolerance=0.1,
            krylov_minimum_relative_tolerance=1e-6,
            adaptive_krylov_tolerance=True,
            adjoint_rtol=1e-8,
            jvp_fixed_point_rtol=1e-8,
        )
        self.assertEqual(primary.krylov_initial_iterations, 64)
        self.assertEqual(primary.krylov_max_iterations, 256)
        self.assertTrue(primary.adaptive_krylov_tolerance)
        self.assertFalse(primary.reduce_damping_only_on_krylov_convergence)
        self.assertEqual(primary.gauge_projector, "structured")
        self.assertEqual(primary.gauge_tolerance, 1e-10)
        self.assertEqual(primary.fixed_point_response_solver, "auto")
        self.assertEqual(strict.adjoint_rtol, 1e-10)
        self.assertEqual(strict.jvp_fixed_point_rtol, 1e-10)

    def test_delta_t_can_override_reference_protocol(self) -> None:
        args = protocol_args()
        args.delta_t = 5e-4
        protocol = protocol_from_args(args)
        self.assertEqual(protocol.delta_t, 5e-4)
        self.assertEqual(protocol.block_length, NONINTEGRABLE_ISING.block_length)

    def test_explicit_second_order_trotter_is_accepted(self) -> None:
        args = protocol_args()
        args.trotter_order = 2
        protocol = protocol_from_args(args)
        self.assertEqual(protocol.trotter_order, 2)

    def test_integrable_preset_and_hamiltonian_overrides(self) -> None:
        args = protocol_args()
        args.protocol = "integrable-tfim"
        args.block_length = None
        args.delta_t = None
        protocol = protocol_from_args(args)
        self.assertEqual(protocol, INTEGRABLE_TFIM)
        self.assertEqual(protocol.g, -0.2)
        self.assertEqual(protocol.h, 0.0)
        self.assertEqual(protocol.J, -1.0)
        self.assertFalse(protocol.symmetric_transverse)

        args.block_length = 5
        args.delta_t = 5e-4
        args.g = -0.3
        args.symmetric_transverse = True
        overridden = protocol_from_args(args)
        self.assertEqual(overridden.block_length, 5)
        self.assertEqual(overridden.delta_t, 5e-4)
        self.assertEqual(overridden.g, -0.3)
        self.assertTrue(overridden.symmetric_transverse)

    def test_strict_retry_can_be_disabled(self) -> None:
        argv = [
            "lomps-evolve",
            "--initial-A",
            "product.npy",
            "--output-dir",
            "out",
            "--steps",
            "1",
            "--base-time",
            "0.0",
            "--no-strict-retry",
        ]
        with patch("sys.argv", argv):
            args = parse_args()
        self.assertFalse(args.strict_retry)

    def test_effective_first_step_accept_cost_uses_global_cost_when_none(self) -> None:
        self.assertEqual(effective_first_step_accept_cost(None, 3e-16), 3e-16)
        self.assertEqual(effective_first_step_accept_cost(1e-15, 3e-16), 1e-15)

    def test_external_seed_lift_noise_default_is_larger_for_high_D(self) -> None:
        self.assertEqual(
            effective_initial_seed_lift_noise(
                requested=None,
                trajectory_bond_dimension=20,
                embedding_noise_amplitude=1e-8,
            ),
            1e-4,
        )
        self.assertEqual(
            effective_initial_seed_lift_noise(
                requested=None,
                trajectory_bond_dimension=19,
                embedding_noise_amplitude=3e-8,
            ),
            3e-8,
        )
        self.assertEqual(
            effective_initial_seed_lift_noise(
                requested=5e-5,
                trajectory_bond_dimension=20,
                embedding_noise_amplitude=1e-8,
            ),
            5e-5,
        )

    def test_first_step_cg_options_follow_main_solver_by_default(self) -> None:
        argv = [
            "lomps-evolve",
            "--initial-A",
            "product.npy",
            "--output-dir",
            "out",
            "--steps",
            "1",
            "--base-time",
            "0.0",
            "--fixed-point-solver",
            "fast",
        ]
        with patch("sys.argv", argv):
            args = parse_args()
        cg_options = first_step_cg_options(
            args,
            accept_cost=3e-16,
            rank_tolerance=1e-12,
            default_fixed_point_solver=args.fixed_point_solver,
        )
        self.assertEqual(cg_options.fixed_point_solver, "fast")
        self.assertEqual(cg_options.cost_tolerance, 3e-16)

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
            initial_seed_mode="embedding",
            embedding_noise_amplitude=1e-4,
            embedding_seed=11,
            embedding_candidate_seeds=(1, 2, 3),
            embedding_screen_max_iterations=0,
            embedding_screen_seconds=0.0,
            embedding_screen_fixed_point_solver="same",
            circuit_lift_mixing_amplitude=1e-3,
            circuit_lift_seed=2,
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
                initial_seed_mode="embedding",
                embedding_noise_amplitude=1e-4,
                embedding_seed=11,
                embedding_candidate_seeds=(21, 22),
                embedding_screen_max_iterations=0,
                embedding_screen_seconds=0.0,
                embedding_screen_fixed_point_solver="same",
                circuit_lift_mixing_amplitude=1e-3,
                circuit_lift_seed=2,
                canonical_tolerance=1e-10,
            )
        self.assertEqual(seed.shape, (2, 2, 2))
        self.assertEqual(len(screen), 2)
        self.assertIn(diagnostics.seed, {21, 22})
        self.assertEqual(sum(row["selected"] for row in screen), 1)

    def test_auto_initial_seed_uses_product_circuit_when_available(self) -> None:
        source = product_tensor(np.array([1.0, 1.0j]))
        primary, _ = options(
            accept_cost=1e-8,
            rank_tolerance=1e-12,
            fixed_point_solver="dense",
        )
        seed, diagnostics, screen = select_initial_seed(
            source,
            12,
            protocol=NONINTEGRABLE_ISING,
            primary=primary,
            initial_seed_mode="auto",
            embedding_noise_amplitude=1e-4,
            embedding_seed=11,
            embedding_candidate_seeds=(),
            embedding_screen_max_iterations=0,
            embedding_screen_seconds=0.0,
            embedding_screen_fixed_point_solver="same",
            circuit_lift_mixing_amplitude=0.0,
            circuit_lift_seed=2,
            canonical_tolerance=1e-10,
        )
        self.assertEqual(seed.shape, (2, 12, 12))
        self.assertEqual(screen, [])
        self.assertEqual(diagnostics.method, "product_circuit")

    def test_external_seed_can_be_lifted_independently_from_product_source(self) -> None:
        source_seed, _ = random_left_canonical(d=2, D=3, seed=84)
        seed, diagnostics = select_external_initial_seed(
            source_seed,
            5,
            protocol=NONINTEGRABLE_ISING,
            lift_noise_amplitude=1e-4,
            embedding_seed=104729,
            canonical_tolerance=1e-10,
        )
        self.assertEqual(seed.shape, (2, 5, 5))
        self.assertEqual(diagnostics.source_bond_dimension, 3)
        self.assertEqual(diagnostics.target_bond_dimension, 5)
        self.assertEqual(diagnostics.noise_amplitude, 1e-4)
        self.assertLess(diagnostics.seed_left_canonical_error, 1e-10)


if __name__ == "__main__":
    unittest.main()
