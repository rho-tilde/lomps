from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from lomps.buffered_purity import BufferedPurityResult


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_buffered_purity_l4_production.py"
)
SPEC = importlib.util.spec_from_file_location("buffered_purity_production", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
PRODUCTION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PRODUCTION)


def _result(
    status: str,
    *,
    accepted_steps: int,
    gradient: float,
    relative_drop: float,
) -> BufferedPurityResult:
    return BufferedPurityResult(
        W=np.eye(1, dtype=np.complex128),
        primary_cost=1e-19,
        purity=0.2,
        initial_primary_cost=1e-19,
        initial_purity=0.3,
        status=status,
        accepted_steps=accepted_steps,
        final_null_gradient_norm=gradient,
        final_relative_purity_drop=relative_drop,
        history=(),
    )


class ProductionPurityClassificationTests(unittest.TestCase):
    def test_production_uses_a_strict_fixed_rho4_fibre(self) -> None:
        options = PRODUCTION.purity_options(
            SimpleNamespace(
                fibre_cost_target=1e-22,
                purity_max_iterations=64,
                purity_step=50.0,
                purity_minimum_step=1e-8,
                purity_gradient_tolerance=2e-8,
                purity_relative_tolerance=1e-12,
                purity_relative_patience=25,
                jacobian_workers=4,
                quiet_purity=True,
            )
        )

        self.assertEqual(options.primary_cost_tolerance, 1e-22)
        self.assertEqual(options.projection_cost_tolerance, 1e-22)
        self.assertTrue(options.warm_start_line_search)

    def test_deep_numerical_floor_can_use_iteration_evidence(self) -> None:
        result = _result(
            "line_search_failed",
            accepted_steps=250,
            gradient=2e-5,
            relative_drop=1e-14,
        )
        classification = PRODUCTION.classify_full_purity_search(
            result,
            minimum_iterations=200,
            relative_tolerance=1e-12,
            numerical_gradient_ceiling=1e-5,
            label="test",
        )
        self.assertEqual(classification, "numerical_line_search_floor")

    def test_deep_numerical_floor_can_use_gradient_evidence(self) -> None:
        result = _result(
            "line_search_failed",
            accepted_steps=12,
            gradient=9e-6,
            relative_drop=1e-14,
        )
        classification = PRODUCTION.classify_full_purity_search(
            result,
            minimum_iterations=200,
            relative_tolerance=1e-12,
            numerical_gradient_ceiling=1e-5,
            label="test",
        )
        self.assertEqual(classification, "numerical_line_search_floor")

    def test_tracking_floor_requires_small_drop_not_small_gradient(self) -> None:
        good = _result(
            "line_search_failed",
            accepted_steps=12,
            gradient=2e-5,
            relative_drop=1e-14,
        )
        self.assertEqual(
            PRODUCTION.require_purity_tracking(good, 64, 1e-12, "test"),
            "numerical_line_search_floor",
        )
        bad = _result(
            "line_search_failed",
            accepted_steps=12,
            gradient=2e-5,
            relative_drop=1e-8,
        )
        with self.assertRaises(RuntimeError):
            PRODUCTION.require_purity_tracking(bad, 64, 1e-12, "test")


if __name__ == "__main__":
    unittest.main()
