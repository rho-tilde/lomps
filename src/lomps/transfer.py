"""Transfer channel construction and fixed-point diagnostics."""

from __future__ import annotations

import numpy as np
import scipy.linalg as la


def vec(X: np.ndarray) -> np.ndarray:
    """Column-major vectorization."""

    return np.asarray(X, dtype=np.complex128).reshape(-1, order="F")


def unvec(v: np.ndarray, D: int) -> np.ndarray:
    """Inverse of :func:`vec` for a ``D x D`` matrix."""

    return np.asarray(v, dtype=np.complex128).reshape(D, D, order="F")


def transfer_matrix(A: np.ndarray) -> np.ndarray:
    """Return the matrix of ``E(X) = sum_i A[i] X A[i]^dagger``.

    With column-major vectorization,
    ``vec(A X A^dagger) = kron(conj(A), A) vec(X)``.
    """

    A = np.asarray(A, dtype=np.complex128)
    d, D, _ = A.shape
    T = np.zeros((D * D, D * D), dtype=np.complex128)
    for i in range(d):
        T += np.kron(A[i].conj(), A[i])
    return T


def apply_channel(A: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Apply the transfer channel to ``X``."""

    return sum(Ai @ X @ Ai.conj().T for Ai in np.asarray(A))


def transfer_eigenvalues(A: np.ndarray) -> np.ndarray:
    """Return transfer eigenvalues sorted by decreasing magnitude."""

    vals = np.linalg.eigvals(transfer_matrix(A))
    order = np.argsort(-np.abs(vals))
    return vals[order]


def injectivity_diagnostics(
    A: np.ndarray,
    unique_tol: float = 1e-8,
    gap_tol: float = 1e-8,
) -> dict[str, object]:
    """Compute finite-dimensional injectivity diagnostics.

    Numerically, an injective left-canonical tensor should have a unique
    transfer eigenvalue at one and all other eigenvalues strictly inside the
    unit disk.
    """

    vals = transfer_eigenvalues(A)
    close_to_one = np.abs(vals - 1.0) <= unique_tol
    lambda2 = vals[1] if len(vals) > 1 else 0.0
    gap = 1.0 - abs(lambda2)
    injective = bool(np.count_nonzero(close_to_one) == 1 and gap > gap_tol)
    return {
        "eigenvalues": vals,
        "lambda2": lambda2,
        "gap": float(np.real(gap)),
        "unique_unit_eigenvalue": bool(np.count_nonzero(close_to_one) == 1),
        "injective": injective,
    }


def right_fixed_point(A: np.ndarray) -> tuple[np.ndarray, dict[str, float | complex]]:
    """Compute the normalized right fixed point ``r`` of ``E(r) = r``.

    A dense eigensolver is used because the target dimensions in this project
    are small. The eigenvector with eigenvalue closest to one is hermitized and
    normalized to trace one.
    """

    A = np.asarray(A, dtype=np.complex128)
    D = A.shape[1]
    T = transfer_matrix(A)
    vals, vecs = np.linalg.eig(T)
    idx = int(np.argmin(np.abs(vals - 1.0)))
    lam = vals[idx]
    r = unvec(vecs[:, idx], D)
    r = 0.5 * (r + r.conj().T)
    tr = np.trace(r)
    if abs(tr) < 1e-14:
        raise RuntimeError("fixed-point eigenvector has nearly zero trace")
    r = r / tr
    r = 0.5 * (r + r.conj().T)
    r = r / np.trace(r)

    residual = apply_channel(A, r) - r
    evals = np.linalg.eigvalsh(r)
    info = {
        "eigenvalue": lam,
        "residual": float(la.norm(residual)),
        "trace": complex(np.trace(r)),
        "hermiticity_error": float(la.norm(r - r.conj().T)),
        "min_eigenvalue": float(np.min(evals).real),
    }
    return r.astype(np.complex128), info


def delta_channel(A: np.ndarray, delta_A: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Apply ``delta E`` to ``X`` for a real tangent direction ``delta_A``."""

    A = np.asarray(A, dtype=np.complex128)
    delta_A = np.asarray(delta_A, dtype=np.complex128)
    out = np.zeros_like(X, dtype=np.complex128)
    for Ai, dAi in zip(A, delta_A):
        out += dAi @ X @ Ai.conj().T + Ai @ X @ dAi.conj().T
    return out


def trace_row(D: int) -> np.ndarray:
    """Row vector implementing ``tr(unvec(v))`` for column-major ``v``."""

    row = np.zeros(D * D, dtype=np.complex128)
    for a in range(D):
        row[a + a * D] = 1.0
    return row


def solve_delta_fixed_point(
    A: np.ndarray,
    delta_A: np.ndarray,
    r: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Solve ``(I - E) delta_r = deltaE(r)`` with ``tr(delta_r)=0``.

    The singular fixed-point direction is removed by appending the trace-fixing
    equation and solving the overdetermined dense least-squares system.
    """

    A = np.asarray(A, dtype=np.complex128)
    D = A.shape[1]
    if r is None:
        r, _ = right_fixed_point(A)
    T = transfer_matrix(A)
    lhs = np.eye(D * D, dtype=np.complex128) - T
    rhs = vec(delta_channel(A, delta_A, r))
    augmented_lhs = np.vstack([lhs, trace_row(D)])
    augmented_rhs = np.concatenate([rhs, np.array([0.0], dtype=np.complex128)])
    sol, *_ = np.linalg.lstsq(augmented_lhs, augmented_rhs, rcond=None)
    delta_r = unvec(sol, D)
    delta_r = 0.5 * (delta_r + delta_r.conj().T)
    delta_r = delta_r - np.eye(D, dtype=np.complex128) * (np.trace(delta_r) / D)

    residual = lhs @ vec(delta_r) - rhs
    info = {
        "equation_residual": float(la.norm(residual)),
        "trace_abs": float(abs(np.trace(delta_r))),
        "hermiticity_error": float(la.norm(delta_r - delta_r.conj().T)),
    }
    return delta_r.astype(np.complex128), info
