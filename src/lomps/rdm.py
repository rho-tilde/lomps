"""Finite-block reduced density matrices and their directional derivatives."""

from __future__ import annotations

import numpy as np
import scipy.linalg as la

from .transfer import right_fixed_point


def block_products(A: np.ndarray, L: int) -> np.ndarray:
    """Return all products ``M_s = A[s_1] ... A[s_L]``.

    The output has shape ``(d**L, D, D)`` in lexicographic physical-index order.
    """

    A = np.asarray(A, dtype=np.complex128)
    if L < 0:
        raise ValueError("L must be nonnegative")
    d, D, _ = A.shape
    products = np.eye(D, dtype=np.complex128)[None, :, :]
    for _ in range(L):
        next_products = np.empty((products.shape[0] * d, D, D), dtype=np.complex128)
        out = 0
        for P in products:
            for i in range(d):
                next_products[out] = P @ A[i]
                out += 1
        products = next_products
    return products


def directional_block_products(
    A: np.ndarray,
    delta_A: np.ndarray,
    L: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``M_s`` and ``delta M_s`` for a tensor tangent ``delta_A``."""

    A = np.asarray(A, dtype=np.complex128)
    delta_A = np.asarray(delta_A, dtype=np.complex128)
    d, D, _ = A.shape
    products = np.eye(D, dtype=np.complex128)[None, :, :]
    d_products = np.zeros_like(products)
    for _ in range(L):
        next_products = np.empty((products.shape[0] * d, D, D), dtype=np.complex128)
        next_d_products = np.empty_like(next_products)
        out = 0
        for P, dP in zip(products, d_products):
            for i in range(d):
                next_products[out] = P @ A[i]
                next_d_products[out] = dP @ A[i] + P @ delta_A[i]
                out += 1
        products = next_products
        d_products = next_d_products
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
