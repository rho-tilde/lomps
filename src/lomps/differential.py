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
from .transfer import DenseFixedPointResponseSolver, solve_delta_fixed_point_iterative


def analytic_directional_rdm(
    A: np.ndarray,
    delta_A: np.ndarray,
    L: int,
    r: np.ndarray | None = None,
    *,
    fixed_point_derivative_solver: str = "dense",
    fixed_point_derivative_rtol: float = 1e-8,
    fixed_point_derivative_maximum_iterations: int | None = None,
    fixed_point_response_solver: DenseFixedPointResponseSolver | None = None,
    block_product_levels: tuple[np.ndarray, ...] | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Analytic directional derivative of ``rho_L``.

    The derivative contains ket insertions, bra insertions, and the fixed-point
    derivative ``delta r`` satisfying
    ``(I - E) delta r = delta E(r)``, ``tr(delta r) = 0``.
    """

    if r is None:
        r, _ = right_fixed_point(A)
    products, d_products = directional_block_products(
        A,
        delta_A,
        L,
        product_levels=block_product_levels,
    )
    if fixed_point_derivative_solver == "dense":
        delta_r, fixed_info = solve_delta_fixed_point(A, delta_A, r)
    elif fixed_point_derivative_solver == "iterative":
        if fixed_point_response_solver is None:
            delta_r, fixed_info = solve_delta_fixed_point_iterative(
                A,
                delta_A,
                r,
                rtol=fixed_point_derivative_rtol,
                maximum_iterations=fixed_point_derivative_maximum_iterations,
            )
        else:
            delta_r, fixed_info = fixed_point_response_solver.solve_delta(
                delta_A
            )
    else:
        raise ValueError(
            "fixed_point_derivative_solver must be 'dense' or 'iterative'"
        )
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


def hermitian_vectorize_rho(delta_rho: np.ndarray) -> np.ndarray:
    """Norm-preserving real coordinates for a Hermitian matrix.

    Diagonal entries are stored once. Upper-triangle real and imaginary
    entries are scaled by ``sqrt(2)`` to account for their conjugate partners,
    so Euclidean inner products in these coordinates equal Frobenius inner
    products of Hermitian matrices.
    """

    matrix = np.asarray(delta_rho, dtype=np.complex128)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("delta_rho must be a square matrix")
    matrix = 0.5 * (matrix + matrix.conj().T)
    upper = np.triu_indices(matrix.shape[0], k=1)
    scale = np.sqrt(2.0)
    return np.concatenate(
        (
            np.diag(matrix).real,
            scale * matrix[upper].real,
            scale * matrix[upper].imag,
        )
    )


def pack_hermitian_real_jacobian(jacobian: np.ndarray, matrix_size: int) -> np.ndarray:
    """Pack a full ``[Re, Im]`` Jacobian using Hermitian coordinates."""

    jacobian = np.asarray(jacobian, dtype=float)
    size = int(matrix_size)
    entries = size * size
    if jacobian.ndim != 2 or jacobian.shape[0] != 2 * entries:
        raise ValueError("jacobian row count is incompatible with matrix_size")
    diagonal = np.arange(size) * (size + 1)
    upper_row, upper_column = np.triu_indices(size, k=1)
    upper = upper_row + size * upper_column
    scale = np.sqrt(2.0)
    return np.vstack(
        (
            jacobian[diagonal],
            scale * jacobian[upper],
            scale * jacobian[entries + upper],
        )
    )


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
