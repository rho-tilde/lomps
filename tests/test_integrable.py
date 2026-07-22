from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from lomps.canonical import canonical_errors
from lomps.gates import partial_trace_sites
from lomps.integrable import exact_local_hs_rate, exact_transverse_magnetization
from lomps.observables import PAULI_X, one_site_rdm
from lomps.optimizer import optimizer_right_fixed_point
from lomps.protocol import INTEGRABLE_TFIM
from lomps.rdm import block_rdm


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "data" / "integrable_tfim_reference" / "tfim_g0_1p5_D12.npy"


class IntegrableBenchmarkTests(unittest.TestCase):
    def test_reference_tensor_is_left_canonical(self) -> None:
        A = np.load(REFERENCE)
        self.assertEqual(A.shape, (2, 12, 12))
        self.assertLess(canonical_errors(A)["left_canonical_error"], 1e-12)

    def test_tensor_target_matches_independent_dense_eight_site_oracle(self) -> None:
        A = np.load(REFERENCE)
        r, _ = optimizer_right_fixed_point(A, "dense")
        rho8 = block_rdm(A, 8, r)
        identity = np.eye(2, dtype=np.complex128)

        def kron_all(factors: list[np.ndarray]) -> np.ndarray:
            result = np.array([[1.0 + 0.0j]])
            for factor in factors:
                result = np.kron(result, factor)
            return result

        even = kron_all([INTEGRABLE_TFIM.half_gate] * 4)
        odd = kron_all(
            [identity]
            + [INTEGRABLE_TFIM.full_gate] * 3
            + [identity]
        )
        unitary = even @ odd @ even
        evolved = unitary @ rho8 @ unitary.conj().T
        oracle = partial_trace_sites(
            evolved,
            sites=8,
            traced_sites=(0, 1, 6, 7),
        )
        np.testing.assert_allclose(
            INTEGRABLE_TFIM.target_rdm(A, r),
            oracle,
            rtol=2e-13,
            atol=2e-14,
        )

    def test_exact_magnetization_is_vectorized_and_consistent_at_zero(self) -> None:
        scalar = exact_transverse_magnetization(0.0, quadrature_points=256)
        vector = exact_transverse_magnetization(
            np.array([0.0, 0.1]),
            quadrature_points=256,
        )
        self.assertAlmostEqual(scalar, vector[0], places=14)
        self.assertTrue(np.all(np.isfinite(vector)))

        A = np.load(REFERENCE)
        r, _ = optimizer_right_fixed_point(A, "dense")
        vumps_x = float(np.trace(one_site_rdm(A, r) @ PAULI_X).real)
        self.assertLess(abs(vumps_x - scalar), 2e-9)

    def test_exact_local_rate_is_vectorized_and_consistent_at_zero(self) -> None:
        scalar = exact_local_hs_rate(0.0, 4, ring_sites=64)
        vector = exact_local_hs_rate(
            np.array([0.0, 0.1]),
            4,
            ring_sites=64,
        )
        self.assertAlmostEqual(scalar, vector[0], places=14)
        self.assertTrue(np.all(np.isfinite(vector)))

        A = np.load(REFERENCE)
        r, _ = optimizer_right_fixed_point(A, "dense")
        rho4 = block_rdm(A, 4, r)
        vumps_rate = -np.log(float(np.trace(rho4 @ rho4).real)) / 4
        self.assertLess(abs(vumps_rate - scalar), 2e-9)

    def test_three_step_strict_cli_matches_local_free_fermion_curve(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_dir = Path(temporary_directory) / "integrable_smoke"
            command = [
                sys.executable,
                "-m",
                "lomps.evolution",
                "--protocol",
                "integrable-tfim",
                "--initial-A",
                str(REFERENCE),
                "--initial-layout",
                "physical-left-right",
                "--output-dir",
                str(run_dir),
                "--bond-dimension",
                "12",
                "--block-length",
                "4",
                "--delta-t",
                "1e-3",
                "--no-symmetric-transverse",
                "--fixed-point-solver",
                "dense",
                "--target-source-fixed-point-solver",
                "dense",
                "--target-contraction",
                "tensor",
                "--accept-cost",
                "1e-15",
                "--perturb-amplitudes",
                "0.1,0.3,0.6,1.0",
                "--perturbations-per-amplitude",
                "3",
                "--random-restarts",
                "0",
                "--no-strict-retry",
                "--checkpoint-every",
                "1",
                "--steps",
                "3",
                "--base-time",
                "0",
            ]
            result = subprocess.run(
                command,
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
                timeout=90,
            )
            self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)

            metadata = json.loads((run_dir / "metadata.json").read_text())
            self.assertEqual(metadata["status"], "completed")
            self.assertEqual(metadata["completed_steps"], 3)
            self.assertEqual(metadata["protocol_preset"], "integrable-tfim")
            with (run_dir / "steps.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 3)
            self.assertFalse(any(row["restart_used"] == "True" for row in rows))
            self.assertLessEqual(
                max(float(row["optimizer_cost"]) for row in rows),
                1.001e-15,
            )

            times = np.asarray(np.load(run_dir / "times.npy")[:4])
            states = np.load(run_dir / "states.npy", mmap_mode="r")[:4]
            fixed_points = np.load(
                run_dir / "right_fixed_points.npy", mmap_mode="r"
            )[:4]
            rho0 = block_rdm(states[0], 4, fixed_points[0])
            lomps_rate = np.asarray(
                [
                    -np.log(
                        float(
                            np.trace(
                                block_rdm(state, 4, fixed_point) @ rho0
                            ).real
                        )
                    )
                    / 4
                    for state, fixed_point in zip(states, fixed_points)
                ]
            )
            exact_rate = exact_local_hs_rate(times, 4, ring_sites=64)
            np.testing.assert_allclose(lomps_rate, exact_rate, rtol=0.0, atol=1e-8)


if __name__ == "__main__":
    unittest.main()
