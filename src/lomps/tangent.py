"""Real bases for Stiefel tangents and projections."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import scipy.linalg as la


TangentSlice = Literal["gauge_orthogonal", "grassmann"]

from .canonical import (
    linearized_stiefel_error,
    orthonormal_complement,
    project_parallel,
    project_perp,
)


def antihermitian_basis(D: int, normalize: bool = True) -> list[np.ndarray]:
    """Return a real basis of anti-Hermitian ``D x D`` matrices.

    The basis has ``D^2`` elements:
    diagonal matrices ``i E_aa`` and, for ``a < b``, the two off-diagonal
    directions ``E_ab - E_ba`` and ``i(E_ab + E_ba)``.
    """

    basis: list[np.ndarray] = []
    for a in range(D):
        X = np.zeros((D, D), dtype=np.complex128)
        X[a, a] = 1j
        basis.append(X)
    for a in range(D):
        for b in range(a + 1, D):
            X = np.zeros((D, D), dtype=np.complex128)
            X[a, b] = 1.0
            X[b, a] = -1.0
            basis.append(X)
            Y = np.zeros((D, D), dtype=np.complex128)
            Y[a, b] = 1j
            Y[b, a] = 1j
            basis.append(Y)
    if normalize:
        basis = [X / la.norm(X) for X in basis]
    return basis


def complex_matrix_real_basis(rows: int, cols: int) -> list[np.ndarray]:
    """Return the standard real basis of complex ``rows x cols`` matrices."""

    basis: list[np.ndarray] = []
    for a in range(rows):
        for b in range(cols):
            X = np.zeros((rows, cols), dtype=np.complex128)
            X[a, b] = 1.0
            basis.append(X)
            Y = np.zeros((rows, cols), dtype=np.complex128)
            Y[a, b] = 1j
            basis.append(Y)
    return basis


def vectorize_complex_real(X: np.ndarray) -> np.ndarray:
    """Vectorize a complex array as a real vector ``[Re X, Im X]``."""

    X = np.asarray(X, dtype=np.complex128)
    return np.concatenate(
        [
            X.real.reshape(-1, order="F"),
            X.imag.reshape(-1, order="F"),
        ]
    )


def as_real_columns(arrays: list[np.ndarray]) -> np.ndarray:
    """Stack complex arrays as columns of a real matrix."""

    if not arrays:
        return np.zeros((0, 0), dtype=float)
    return np.column_stack([vectorize_complex_real(X) for X in arrays])


def singular_values_and_rank(
    matrix: np.ndarray,
    tolerances: tuple[float, ...] = (1e-8, 1e-10, 1e-12),
) -> dict[str, object]:
    """Compute singular values and several numerical ranks."""

    matrix = np.asarray(matrix, dtype=float)
    if matrix.size == 0:
        svals = np.array([], dtype=float)
    else:
        svals = np.linalg.svd(matrix, compute_uv=False)
    if svals.size == 0:
        relative_tol = 0.0
        relative_rank = 0
        ranks = {tol: 0 for tol in tolerances}
    else:
        relative_tol = max(matrix.shape) * np.finfo(float).eps * svals[0]
        relative_rank = int(np.count_nonzero(svals > relative_tol))
        ranks = {tol: int(np.count_nonzero(svals > tol)) for tol in tolerances}
    return {
        "singular_values": svals,
        "relative_tol": float(relative_tol),
        "relative_rank": relative_rank,
        "ranks_by_tol": ranks,
    }


@dataclass(frozen=True)
class TangentBases:
    """Container for real tangent bases at ``W``."""

    W: np.ndarray
    W_perp: np.ndarray
    parallel: list[np.ndarray]
    perp: list[np.ndarray]
    full: list[np.ndarray]
    antihermitian: list[np.ndarray]
    B_basis: list[np.ndarray]


def tangent_bases(W: np.ndarray, d: int, D: int) -> TangentBases:
    """Construct real Stiefel tangent bases ``W Omega`` and ``W_perp B``."""

    W = np.asarray(W, dtype=np.complex128)
    if W.shape != (d * D, D):
        raise ValueError(f"W must have shape {(d * D, D)}")
    W_perp = orthonormal_complement(W)
    anti_basis = antihermitian_basis(D)
    B_basis = complex_matrix_real_basis((d - 1) * D, D)
    parallel = [W @ Omega for Omega in anti_basis]
    perp = [W_perp @ B for B in B_basis]
    return TangentBases(
        W=W,
        W_perp=W_perp,
        parallel=parallel,
        perp=perp,
        full=parallel + perp,
        antihermitian=anti_basis,
        B_basis=B_basis,
    )


def grassmann_tangent_basis(
    W: np.ndarray,
    d: int,
    D: int,
) -> list[np.ndarray]:
    """Return the TDVP/Grassmann slice ``W^dagger delta_W = 0``.

    For a left-canonical tensor stacked as the isometry ``W``, every member
    has the form ``W_perp X``.  The returned real Frobenius-orthonormal basis
    has ``2 (d - 1) D**2`` elements and is a direct alternative quotient
    representative to the true-gauge-orthogonal Stiefel slice.
    """

    return tangent_bases(W, d, D).perp


def max_linearized_stiefel_error(W: np.ndarray, basis: list[np.ndarray]) -> float:
    """Maximum linearized Stiefel residual over a tangent basis."""

    if not basis:
        return 0.0
    return max(linearized_stiefel_error(W, delta) for delta in basis)


def combine_real(coeffs: np.ndarray, basis: list[np.ndarray]) -> np.ndarray:
    """Combine complex basis arrays using real coefficients."""

    if len(coeffs) != len(basis):
        raise ValueError("number of coefficients must match basis size")
    out = np.zeros_like(basis[0], dtype=np.complex128)
    for c, B in zip(np.asarray(coeffs, dtype=float), basis):
        out += c * B
    return out


def project_parallel_list(W: np.ndarray, basis: list[np.ndarray]) -> list[np.ndarray]:
    """Apply ``P_parallel`` to each basis vector."""

    return [project_parallel(W, delta) for delta in basis]


def project_perp_list(W: np.ndarray, basis: list[np.ndarray]) -> list[np.ndarray]:
    """Apply ``P_perp`` to each basis vector."""

    return [project_perp(W, delta) for delta in basis]
