"""Left-canonical MPS tensor utilities."""

from __future__ import annotations

import numpy as np
import scipy.linalg as la


def stack_tensor(A: np.ndarray) -> np.ndarray:
    """Stack ``A[i]`` vertically into ``W`` with shape ``(d * D, D)``."""

    A = np.asarray(A, dtype=np.complex128)
    if A.ndim != 3 or A.shape[1] != A.shape[2]:
        raise ValueError("A must have shape (d, D, D)")
    d, D, _ = A.shape
    return A.reshape(d * D, D)


def unstack_tensor(W: np.ndarray, d: int, D: int) -> np.ndarray:
    """Unstack ``W`` with shape ``(d * D, D)`` into ``A``."""

    W = np.asarray(W, dtype=np.complex128)
    if W.shape != (d * D, D):
        raise ValueError(f"W must have shape {(d * D, D)}, got {W.shape}")
    return W.reshape(d, D, D)


def _fix_qr_phases(Q: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Fix the arbitrary QR phases using the diagonal of ``R``."""

    diag = np.diag(R)
    phases = np.ones_like(diag, dtype=np.complex128)
    nonzero = np.abs(diag) > 0
    phases[nonzero] = diag[nonzero] / np.abs(diag[nonzero])
    return Q * phases.conj()[None, :]


def random_left_canonical(
    d: int = 2,
    D: int = 2,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a random left-canonical tensor by QR decomposition.

    A random complex matrix ``Z`` of shape ``(dD, D)`` is decomposed as
    ``Z = QR``. The first ``D`` columns of ``Q`` define the Stiefel point
    ``W`` and are reshaped into ``A[i]``.
    """

    if d < 1 or D < 1:
        raise ValueError("d and D must be positive")
    rng = np.random.default_rng(seed)
    Z = (
        rng.normal(size=(d * D, D))
        + 1j * rng.normal(size=(d * D, D))
    ).astype(np.complex128)
    Q, R = np.linalg.qr(Z, mode="reduced")
    W = _fix_qr_phases(Q, R)
    return unstack_tensor(W, d, D), W


def canonical_errors(A: np.ndarray) -> dict[str, float]:
    """Return equivalent left-canonical/Stiefel residuals."""

    A = np.asarray(A, dtype=np.complex128)
    d, D, _ = A.shape
    W = stack_tensor(A)
    eye = np.eye(D, dtype=np.complex128)
    stiefel = W.conj().T @ W - eye
    lc = sum(A[i].conj().T @ A[i] for i in range(d)) - eye
    return {
        "stiefel_error": float(la.norm(stiefel)),
        "left_canonical_error": float(la.norm(lc)),
    }


def orthonormal_complement(W: np.ndarray, rtol: float = 1e-12) -> np.ndarray:
    """Return an orthonormal basis for the complement of ``range(W)``.

    ``W`` is assumed to have orthonormal columns. The returned matrix has shape
    ``(dD, (d - 1)D)`` for an MPS Stiefel point.
    """

    W = np.asarray(W, dtype=np.complex128)
    W_perp = la.null_space(W.conj().T, rcond=rtol)
    expected = W.shape[0] - W.shape[1]
    if W_perp.shape != (W.shape[0], expected):
        raise RuntimeError(
            f"unexpected complement shape {W_perp.shape}; expected "
            f"{(W.shape[0], expected)}"
        )
    return W_perp.astype(np.complex128)


def project_parallel(W: np.ndarray, delta_W: np.ndarray) -> np.ndarray:
    """Project a Stiefel tangent onto ``range(W)``."""

    return W @ (W.conj().T @ delta_W)


def project_perp(W: np.ndarray, delta_W: np.ndarray) -> np.ndarray:
    """Project a Stiefel tangent onto the orthogonal complement of ``range(W)``."""

    return delta_W - project_parallel(W, delta_W)


def linearized_stiefel_error(W: np.ndarray, delta_W: np.ndarray) -> float:
    """Return ``||W^dagger delta_W + delta_W^dagger W||``."""

    residual = W.conj().T @ delta_W + delta_W.conj().T @ W
    return float(la.norm(residual))


def polar_retraction(W: np.ndarray) -> np.ndarray:
    """Retract a full-rank matrix to the Stiefel manifold by polar projection."""

    gram = W.conj().T @ W
    inv_sqrt = la.inv(la.sqrtm(gram))
    return (W @ inv_sqrt).astype(np.complex128)
