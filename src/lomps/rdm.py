"""Finite-block reduced density matrices and their directional derivatives."""

from __future__ import annotations

import numpy as np
import scipy.linalg as la

from .transfer import right_fixed_point


def block_product_levels(A: np.ndarray, L: int) -> tuple[np.ndarray, ...]:
    """Return prefix-product batches for every length from zero through ``L``."""

    A = np.asarray(A, dtype=np.complex128)
    if L < 0:
        raise ValueError("L must be nonnegative")
    d, D, _ = A.shape
    levels = [np.eye(D, dtype=np.complex128)[None, :, :]]
    for _ in range(L):
        products = levels[-1]
        next_products = np.matmul(
            products[:, None, :, :], A[None, :, :, :]
        ).reshape(products.shape[0] * d, D, D)
        levels.append(next_products)
    return tuple(levels)


def block_products(A: np.ndarray, L: int) -> np.ndarray:
    """Return all products ``M_s = A[s_1] ... A[s_L]``.

    The output has shape ``(d**L, D, D)`` in lexicographic physical-index order.
    """

    return block_product_levels(A, L)[-1]


def directional_block_products(
    A: np.ndarray,
    delta_A: np.ndarray,
    L: int,
    *,
    product_levels: tuple[np.ndarray, ...] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``M_s`` and ``delta M_s`` for a tensor tangent ``delta_A``."""

    A = np.asarray(A, dtype=np.complex128)
    delta_A = np.asarray(delta_A, dtype=np.complex128)
    d, D, _ = A.shape
    levels = block_product_levels(A, L) if product_levels is None else product_levels
    if len(levels) != L + 1:
        raise ValueError("product_levels must contain levels zero through L")
    d_products = np.zeros_like(levels[0])
    for level in range(L):
        products = levels[level]
        if products.shape != (d**level, D, D):
            raise ValueError("product_levels contains an incompatible batch")
        next_d_products = np.matmul(
            d_products[:, None, :, :], A[None, :, :, :]
        ).reshape(products.shape[0] * d, D, D)
        next_d_products += np.matmul(
            products[:, None, :, :], delta_A[None, :, :, :]
        ).reshape(products.shape[0] * d, D, D)
        d_products = next_d_products
    products = levels[-1]
    if products.shape != (d**L, D, D):
        raise ValueError("product_levels contains an incompatible final batch")
    return products, d_products


def rdm_from_products(products: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Build ``rho[s,t] = tr(M_s r M_t^dagger)`` from precomputed products."""

    return np.einsum(
        "sab,bc,tac->st",
        products,
        r,
        products.conj(),
        optimize=True,
    ).astype(np.complex128)


def block_rdm(A: np.ndarray, L: int, r: np.ndarray | None = None) -> np.ndarray:
    """Construct the ``L``-site reduced density matrix."""

    if r is None:
        r, _ = right_fixed_point(A)
    return rdm_from_products(block_products(A, L), r)


def directional_rdm_from_products(
    products: np.ndarray,
    d_products: np.ndarray,
    r: np.ndarray,
    delta_r: np.ndarray,
) -> np.ndarray:
    """Directional derivative of ``rho_L`` from products and ``delta r``."""

    ket = np.einsum(
        "sab,bc,tac->st",
        d_products,
        r,
        products.conj(),
        optimize=True,
    )
    fixed_point = np.einsum(
        "sab,bc,tac->st",
        products,
        delta_r,
        products.conj(),
        optimize=True,
    )
    bra = np.einsum(
        "sab,bc,tac->st",
        products,
        r,
        d_products.conj(),
        optimize=True,
    )
    return (ket + fixed_point + bra).astype(np.complex128)


def rdm_diagnostics(rho: np.ndarray, tol: float = 1e-10) -> dict[str, object]:
    """Return trace, Hermiticity, positivity, and numerical-rank diagnostics."""

    rho = np.asarray(rho, dtype=np.complex128)
    hermitian = 0.5 * (rho + rho.conj().T)
    evals = np.linalg.eigvalsh(hermitian)
    rank = int(np.count_nonzero(evals > tol))
    return {
        "shape": rho.shape,
        "trace": complex(np.trace(rho)),
        "trace_error": float(abs(np.trace(rho) - 1.0)),
        "hermiticity_error": float(la.norm(rho - rho.conj().T)),
        "min_eigenvalue": float(np.min(evals).real),
        "eigenvalues": evals,
        "rank": rank,
    }
