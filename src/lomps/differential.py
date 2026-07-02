"""Differentials and real Jacobians of the finite-block RDM map."""

from __future__ import annotations

import numpy as np

from .rdm import (
    block_rdm,
    directional_block_products,
    directional_rdm_from_products,
)
from .tangent import vectorize_complex_real
from .transfer import right_fixed_point, solve_delta_fixed_point


def analytic_directional_rdm(
    A: np.ndarray,
    delta_A: np.ndarray,
    L: int,
    r: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Analytic directional derivative of ``rho_L``.

    The derivative contains ket insertions, bra insertions, and the fixed-point
    derivative ``delta r`` satisfying
    ``(I - E) delta r = delta E(r)``, ``tr(delta r) = 0``.
    """

    if r is None:
        r, _ = right_fixed_point(A)
    products, d_products = directional_block_products(A, delta_A, L)
    delta_r, fixed_info = solve_delta_fixed_point(A, delta_A, r)
    delta_rho = directional_rdm_from_products(products, d_products, r, delta_r)
    return delta_rho, {"fixed_point_derivative": fixed_info}


def finite_difference_directional_rdm(
    A: np.ndarray,
    delta_A: np.ndarray,
    L: int,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """Central finite difference for the RDM derivative.

    ``A +/- epsilon delta_A`` is only left canonical up to second order for a
    left-canonical tangent, which is sufficient for a first derivative check.
    """

    eps = float(epsilon)
    return (block_rdm(A + eps * delta_A, L) - block_rdm(A - eps * delta_A, L)) / (
        2.0 * eps
    )


def real_vectorize_rho(delta_rho: np.ndarray) -> np.ndarray:
    """Real vectorization used for Jacobian columns."""

    return vectorize_complex_real(delta_rho)


def build_jacobian(
    A: np.ndarray,
    tangent_tensors: list[np.ndarray],
    L: int,
    method: str = "analytic",
    epsilon: float = 1e-6,
    r: np.ndarray | None = None,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Build a real Jacobian whose columns are vectorized ``D rho[delta A]``."""

    if not tangent_tensors:
        return np.zeros((0, 0), dtype=float), []
    columns: list[np.ndarray] = []
    derivatives: list[np.ndarray] = []
    if r is None:
        r, _ = right_fixed_point(A)
    for delta_A in tangent_tensors:
        if method == "analytic":
            delta_rho, _ = analytic_directional_rdm(A, delta_A, L, r=r)
        elif method == "finite_difference":
            delta_rho = finite_difference_directional_rdm(
                A, delta_A, L, epsilon=epsilon
            )
        else:
            raise ValueError("method must be 'analytic' or 'finite_difference'")
        derivatives.append(delta_rho)
        columns.append(real_vectorize_rho(delta_rho))
    return np.column_stack(columns), derivatives


def derivative_difference_norm(
    A: np.ndarray,
    delta_A: np.ndarray,
    L: int,
    epsilon: float = 1e-6,
) -> dict[str, float]:
    """Compare analytic and finite-difference directional derivatives."""

    analytic, _ = analytic_directional_rdm(A, delta_A, L)
    finite = finite_difference_directional_rdm(A, delta_A, L, epsilon=epsilon)
    diff = analytic - finite
    denom = max(np.linalg.norm(analytic), np.linalg.norm(finite), 1e-15)
    return {
        "absolute": float(np.linalg.norm(diff)),
        "relative": float(np.linalg.norm(diff) / denom),
    }
