from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_capped_itebd_yplus.py"


class CappedItebdReferenceTests(unittest.TestCase):
    def test_short_trajectory_is_physical(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "short_trajectory.npz"
            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--max-chi",
                    "4",
                    "--dt",
                    "0.01",
                    "--t-final",
                    "0.02",
                    "--sample-dt",
                    "0.01",
                    "--max-rdm-length",
                    "2",
                    "--svd-cutoff",
                    "1e-8",
                    "--output",
                    str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            with np.load(output, allow_pickle=False) as trajectory:
                np.testing.assert_allclose(trajectory["times"], [0.0, 0.01, 0.02])
                for key in ("rho_l1", "rho_l2"):
                    for rho in trajectory[key]:
                        np.testing.assert_allclose(rho, rho.conj().T, atol=1.0e-12)
                        self.assertAlmostEqual(float(np.trace(rho).real), 1.0, places=12)

            metadata = json.loads(output.with_suffix(".json").read_text())
            self.assertEqual(metadata["algorithm"], "two-site iTEBD in Vidal Gamma-Lambda form")
            self.assertEqual(metadata["max_chi_per_bond"], 4)


if __name__ == "__main__":
    unittest.main()
