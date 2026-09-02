from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np

from lomps.canonical import polar_retraction, random_left_canonical, stack_tensor, unstack_tensor
from lomps.differential import (
    build_jacobian,
    hermitian_vectorize_rho,
    pack_hermitian_real_jacobian,
)
from lomps.fixed_target_cg import optimize_fixed_target_cg
from lomps.horizontal import GaugeHorizontalProjector
from lomps.optimizer import (
    CGOptions,
    GaugeOrthogonalLM,
    LMOptions,
    batched_rdm_jacobian,
    gauge_orthogonal_basis,
    optimizer_right_fixed_point,
)
from lomps.protocol import NONINTEGRABLE_ISING
from lomps.predictor import (
    align_virtual_gauge,
    secant_predictor,
    trajectory_secant_predictor,
)
from lomps.rdm import block_rdm
from lomps.transfer import right_fixed_point
from lomps.tangent import grassmann_tangent_basis


class OptimizerTests(unittest.TestCase):
    def test_dense_fixed_point_solver_is_default(self) -> None:
        self.assertEqual(LMOptions().fixed_point_solver, "dense")
        self.assertEqual(LMOptions().linear_solver, "svd")
        self.assertEqual(LMOptions().jacobian_response_solver, "lstsq")
        self.assertEqual(LMOptions().jacobian_workers, 1)

    def test_lm_recovers_nearby_reachable_rdm(self) -> None:
        A0, W0 = random_left_canonical(d=2, D=2, seed=72)
        A1, _ = random_left_canonical(d=2, D=2, seed=73)
        target_A = unstack_tensor(
            polar_retraction(W0 + 1e-3 * (stack_tensor(A1) - W0)),
            2,
            2,
        )
        target = block_rdm(target_A, 2)
        solver = GaugeOrthogonalLM(
            2,
            LMOptions(
                max_iterations=12,
                gradient_tolerance=1e-12,
                residual_tolerance=1e-10,
                verbose=False,
            ),
        )
        result = solver.optimize(W0, target)
        self.assertLess(result.residual_norm, 1e-9)

    def test_fixed_target_cg_reduces_nearby_reachable_rdm_cost(self) -> None:
        A0, W0 = random_left_canonical(d=2, D=2, seed=92)
        _, W1 = random_left_canonical(d=2, D=2, seed=93)
        target_A = unstack_tensor(
            polar_retraction(W0 + 1e-3 * (W1 - W0)),
            2,
            2,
        )
        target = block_rdm(target_A, 2)
        seed_cost = float(np.linalg.norm(block_rdm(A0, 2) - target) ** 2)
        _, result = optimize_fixed_target_cg(
            A0,
            target,
            2,
            CGOptions(
                max_iterations=5,
                gradient_tolerance=0.0,
                cost_tolerance=0.0,
                maximum_seconds=30.0,
                verbose=False,
            ),
        )
        self.assertLess(result.cost, seed_cost)

    def test_optimizer_rejects_unknown_fixed_point_solver(self) -> None:
        with self.assertRaises(ValueError):
            GaugeOrthogonalLM(
                2,
                LMOptions(
                    fixed_point_solver="unknown",  # type: ignore[arg-type]
                    verbose=False,
                ),
            )

    def test_optimizer_rejects_unknown_linear_solver(self) -> None:
        with self.assertRaises(ValueError):
            GaugeOrthogonalLM(
                2,
                LMOptions(
                    linear_solver="unknown",  # type: ignore[arg-type]
                    verbose=False,
                ),
            )

    def test_optimizer_rejects_invalid_jacobian_policy(self) -> None:
        with self.assertRaises(ValueError):
            GaugeOrthogonalLM(
                2,
                LMOptions(
                    jacobian_response_solver="unknown",  # type: ignore[arg-type]
                    verbose=False,
                ),
            )
        with self.assertRaises(ValueError):
            GaugeOrthogonalLM(
                2,
                LMOptions(jacobian_workers=0, verbose=False),
            )

    def test_optimizer_rejects_unknown_tangent_slice(self) -> None:
        with self.assertRaises(ValueError):
            GaugeOrthogonalLM(
                2,
                LMOptions(
                    tangent_slice="unknown",  # type: ignore[arg-type]
                    verbose=False,
                ),
            )

    def test_grassmann_basis_is_orthonormal_and_satisfies_left_gauge(self) -> None:
        _, W = random_left_canonical(d=2, D=3, seed=74)
        basis = grassmann_tangent_basis(W, 2, 3)
        self.assertEqual(len(basis), 2 * 3**2)
        gram = np.array(
            [
                [float(np.vdot(left, right).real) for right in basis]
                for left in basis
            ]
        )
        np.testing.assert_allclose(gram, np.eye(len(basis)), atol=2e-13)
        self.assertLess(
            max(np.linalg.norm(W.conj().T @ direction) for direction in basis),
            2e-13,
        )

    def test_two_dense_jacobian_slices_are_gauge_equivalent(self) -> None:
        A, W = random_left_canonical(d=2, D=2, seed=75)
        target = block_rdm(A, 3)
        true_evaluation = GaugeOrthogonalLM(
            3,
            LMOptions(
                tangent_slice="gauge_orthogonal",
                fixed_point_solver="dense",
                linear_solver="normal",
                verbose=False,
            ),
        ).evaluate(W, target)
        grassmann_evaluation = GaugeOrthogonalLM(
            3,
            LMOptions(
                tangent_slice="grassmann",
                fixed_point_solver="dense",
                linear_solver="normal",
                verbose=False,
            ),
        ).evaluate(W, target)
        grassmann_basis = grassmann_evaluation.basis
        true_projector = GaugeHorizontalProjector.from_tensor(A, W)
        mapped_true = np.array(
            [
                true_projector.historical_slice_representative(direction)
                for direction in true_evaluation.basis
            ]
        )
        coordinates = np.array(
            [
                [float(np.vdot(left, right).real) for right in mapped_true]
                for left in grassmann_basis
            ]
        )
        mapped_jacobian = grassmann_evaluation.jacobian @ coordinates
        # Coordinate changes from the intrinsic orthogonal representative to
        # an oblique Grassmann representative cannot contract the quotient
        # norm; this is the metric inflation measured by the audit.
        self.assertGreaterEqual(
            float(np.min(np.linalg.svd(coordinates, compute_uv=False))),
            1.0 - 2e-12,
        )
        np.testing.assert_allclose(
            mapped_jacobian,
            true_evaluation.jacobian,
            atol=3e-10,
            rtol=3e-9,
        )

    def test_hermitian_vectorization_preserves_frobenius_geometry(self) -> None:
        rng = np.random.default_rng(902)
        first = rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8))
        second = rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8))
        first = 0.5 * (first + first.conj().T)
        second = 0.5 * (second + second.conj().T)
        first_packed = hermitian_vectorize_rho(first)
        second_packed = hermitian_vectorize_rho(second)
        self.assertEqual(first_packed.size, 64)
        self.assertAlmostEqual(
            float(first_packed @ second_packed),
            float(np.vdot(first, second).real),
            places=12,
        )

    def test_hermitian_jacobian_preserves_gradient_and_gram(self) -> None:
        A, W = random_left_canonical(d=2, D=2, seed=903)
        target_A, _ = random_left_canonical(d=2, D=2, seed=904)
        target = block_rdm(target_A, 3)
        full = GaugeOrthogonalLM(
            3,
            LMOptions(linear_solver="normal", verbose=False),
        ).evaluate(W, target)
        packed_jacobian = pack_hermitian_real_jacobian(full.jacobian, 8)
        packed_residual = hermitian_vectorize_rho(full.residual)
        np.testing.assert_allclose(
            packed_jacobian.T @ packed_residual,
            full.gradient,
            atol=2e-12,
            rtol=2e-11,
        )
        np.testing.assert_allclose(
            packed_jacobian.T @ packed_jacobian,
            full.jacobian.T @ full.jacobian,
            atol=2e-11,
            rtol=2e-10,
        )

    def test_virtual_gauge_alignment_recovers_equivalent_tensor(self) -> None:
        A, _ = random_left_canonical(d=2, D=3, seed=905)
        rng = np.random.default_rng(906)
        unitary, _ = np.linalg.qr(
            rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
        )
        moving = np.einsum(
            "ab,sbc,cd->sad", unitary, A, unitary.conj().T, optimize=True
        )
        aligned, info = align_virtual_gauge(A, moving)
        np.testing.assert_allclose(aligned, A, atol=2e-10, rtol=2e-10)
        self.assertLess(info["distance_after"], 1e-9)

    def test_secant_predictor_is_left_canonical(self) -> None:
        previous, W = random_left_canonical(d=2, D=3, seed=907)
        direction, _ = random_left_canonical(d=2, D=3, seed=908)
        current = unstack_tensor(
            polar_retraction(W + 1e-3 * (stack_tensor(direction) - W)), 2, 3
        )
        predicted, _ = secant_predictor(previous, current)
        np.testing.assert_allclose(
            sum(tensor.conj().T @ tensor for tensor in predicted),
            np.eye(3),
            atol=2e-13,
            rtol=0.0,
        )
        trajectory_prediction = trajectory_secant_predictor(previous, current)
        np.testing.assert_allclose(
            sum(tensor.conj().T @ tensor for tensor in trajectory_prediction),
            np.eye(3),
            atol=2e-13,
            rtol=0.0,
        )

    def test_normal_linear_solver_matches_svd_direction(self) -> None:
        _, W = random_left_canonical(d=2, D=2, seed=507)
        target_A, _ = random_left_canonical(d=2, D=2, seed=508)
        target = block_rdm(target_A, 3)
        svd_solver = GaugeOrthogonalLM(
            3,
            LMOptions(
                rank_tolerance=1e-10,
                trust_radius=1e6,
                fixed_point_solver="dense",
                verbose=False,
            ),
        )
        normal_solver = GaugeOrthogonalLM(
            3,
            LMOptions(
                rank_tolerance=1e-10,
                trust_radius=1e6,
                fixed_point_solver="dense",
                linear_solver="normal",
                verbose=False,
            ),
        )
        damping = 1e-4
        svd_direction = svd_solver._lm_direction(
            svd_solver.evaluate(W, target),
            damping,
        )[0]
        normal_direction = normal_solver._lm_direction(
            normal_solver.evaluate(W, target),
            damping,
        )[0]
        np.testing.assert_allclose(
            normal_direction,
            svd_direction,
            atol=1e-10,
            rtol=1e-9,
        )

    def test_fixed_point_solver_policy_selects_dense_and_fast(self) -> None:
        A, _ = random_left_canonical(d=2, D=2, seed=78)
        dense, dense_info = optimizer_right_fixed_point(A, "dense")
        fast, fast_info = optimizer_right_fixed_point(A, "fast")
        np.testing.assert_allclose(dense, fast, atol=1e-10, rtol=1e-10)
        self.assertIn("residual", dense_info)
        self.assertIn("residual", fast_info)

    def test_dense_optimizer_evaluation_is_bitwise_reproducible(self) -> None:
        root = Path(__file__).resolve().parents[1]
        A = np.load(root / "data" / "nonintegrable_d12_t0p001.npy")
        target = NONINTEGRABLE_ISING.target_rdm(A)
        solver = GaugeOrthogonalLM(
            NONINTEGRABLE_ISING.block_length,
            LMOptions(
                cost_tolerance=3e-16,
                gradient_tolerance=1e-11,
                fixed_point_solver="dense",
                verbose=False,
            ),
        )
        first = solver.evaluate(stack_tensor(A), target)
        second = solver.evaluate(stack_tensor(A), target)
        self.assertTrue(np.array_equal(first.rho, second.rho))
        self.assertTrue(np.array_equal(first.jacobian, second.jacobian))
        self.assertTrue(np.array_equal(first.gradient, second.gradient))
        self.assertEqual(first.cost, second.cost)

    def test_batched_jacobian_matches_columnwise_jacobian(self) -> None:
        A, W = random_left_canonical(d=2, D=2, seed=75)
        basis_W = gauge_orthogonal_basis(A, W, tolerance=1e-10)
        basis_A = [unstack_tensor(direction, 2, 2) for direction in basis_W]
        r, _ = right_fixed_point(A)
        expected, _ = build_jacobian(A, basis_A, 3, r=r)
        actual = batched_rdm_jacobian(A, basis_A, 3, r)
        np.testing.assert_allclose(actual, expected, atol=1e-11, rtol=1e-10)

    def test_lu_threaded_jacobian_matches_legacy_jacobian(self) -> None:
        A, W = random_left_canonical(d=2, D=3, seed=751)
        basis_W = grassmann_tangent_basis(W, 2, 3)
        basis_A = [unstack_tensor(direction, 2, 3) for direction in basis_W]
        r, _ = right_fixed_point(A)
        legacy = batched_rdm_jacobian(A, basis_A, 3, r)
        threaded_lu = batched_rdm_jacobian(
            A,
            basis_A,
            3,
            r,
            response_solver="dense_lu",
            workers=4,
        )
        np.testing.assert_allclose(
            threaded_lu,
            legacy,
            atol=2e-11,
            rtol=2e-10,
        )

    def test_d23_l6_jacobian_avoids_oversized_complex_gemm(self) -> None:
        root = Path(__file__).resolve().parents[1]
        script = """
import numpy as np
from lomps.canonical import random_left_canonical
from lomps.optimizer import batched_rdm_jacobian

A, _ = random_left_canonical(2, 23, seed=8123)
tangents = np.zeros((1058, 2, 23, 23), dtype=np.complex128)
r = np.eye(23, dtype=np.complex128) / 23
jacobian = batched_rdm_jacobian(A, tangents, 6, r)
assert jacobian.shape == (8192, 1058)
assert np.count_nonzero(jacobian) == 0
"""
        environment = os.environ.copy()
        environment.update(
            {
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "VECLIB_MAXIMUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
            }
        )
        completed = subprocess.run(
            [sys.executable, "-B", "-c", script],
            cwd=root,
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
        )

    def test_zero_time_step_target_is_current_rdm(self) -> None:
        A, _ = random_left_canonical(d=2, D=2, seed=74)
        protocol = replace(NONINTEGRABLE_ISING, delta_t=0.0)
        target = protocol.target_rdm(A)
        current = block_rdm(A, protocol.block_length)
        np.testing.assert_allclose(target, current, atol=1e-12, rtol=0.0)

    def test_zero_time_step_odd_target_is_current_rdm(self) -> None:
        A, _ = random_left_canonical(d=2, D=2, seed=76)
        protocol = replace(NONINTEGRABLE_ISING, block_length=3, delta_t=0.0)
        target = protocol.target_rdm(A)
        current = block_rdm(A, protocol.block_length)
        np.testing.assert_allclose(target, current, atol=1e-12, rtol=0.0)

    def test_cached_target_source_fixed_point_reproduces_target(self) -> None:
        A, _ = random_left_canonical(d=2, D=2, seed=86)
        r, _ = optimizer_right_fixed_point(A, "dense")
        np.testing.assert_allclose(
            NONINTEGRABLE_ISING.target_rdm(A, r),
            NONINTEGRABLE_ISING.target_rdm(A),
            atol=1e-14,
            rtol=1e-13,
        )

    def test_tensor_target_contraction_matches_dense_even_protocol(self) -> None:
        A, _ = random_left_canonical(d=2, D=2, seed=79)
        tensor_protocol = replace(NONINTEGRABLE_ISING, target_contraction="tensor")
        dense_protocol = replace(NONINTEGRABLE_ISING, target_contraction="dense")
        np.testing.assert_allclose(
            tensor_protocol.target_rdm(A),
            dense_protocol.target_rdm(A),
            atol=2e-14,
            rtol=1e-13,
        )

    def test_tensor_target_contraction_matches_dense_odd_protocol(self) -> None:
        A, _ = random_left_canonical(d=2, D=2, seed=80)
        tensor_protocol = replace(
            NONINTEGRABLE_ISING,
            block_length=3,
            target_contraction="tensor",
            odd_parity_warning_threshold=np.inf,
        )
        dense_protocol = replace(
            NONINTEGRABLE_ISING,
            block_length=3,
            target_contraction="dense",
            odd_parity_warning_threshold=np.inf,
        )
        np.testing.assert_allclose(
            tensor_protocol.target_rdm(A),
            dense_protocol.target_rdm(A),
            atol=2e-14,
            rtol=1e-13,
        )

    def test_fast_target_source_fixed_point_matches_dense_target_source(self) -> None:
        A, _ = random_left_canonical(d=2, D=3, seed=85)
        dense_protocol = replace(
            NONINTEGRABLE_ISING,
            target_source_fixed_point_solver="dense",
        )
        fast_protocol = replace(
            NONINTEGRABLE_ISING,
            target_source_fixed_point_solver="fast",
        )
        np.testing.assert_allclose(
            fast_protocol.target_rdm(A),
            dense_protocol.target_rdm(A),
            atol=1e-10,
            rtol=1e-10,
        )

    def test_odd_protocol_averages_asymmetric_reductions(self) -> None:
        A, _ = random_left_canonical(d=2, D=2, seed=77)
        protocol = replace(
            NONINTEGRABLE_ISING,
            block_length=3,
            odd_parity_warning_threshold=np.inf,
        )
        self.assertEqual(protocol.lightcone_sites, 8)
        self.assertEqual(protocol.target_margins, ((2, 3), (3, 2)))
        candidates = protocol.target_rdm_candidates(A)
        self.assertEqual(len(candidates), 2)
        expected = 0.5 * (candidates[0] + candidates[1])
        expected = 0.5 * (expected + expected.conj().T)
        expected = expected / np.trace(expected)
        np.testing.assert_allclose(protocol.target_rdm(A), expected, atol=1e-12, rtol=0.0)


if __name__ == "__main__":
    unittest.main()
