"""MPS gauge tangent and finite gauge transformations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg as la

from .canonical import stack_tensor
from .tangent import antihermitian_basis


def virtual_gauge_tangent(A: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Return ``delta A[i] = X A[i] - A[i] X``."""

    A = np.asarray(A, dtype=np.complex128)
    X = np.asarray(X, dtype=np.complex128)
    return np.array([X @ Ai - Ai @ X for Ai in A], dtype=np.complex128)


def phase_gauge_tangent(A: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Return the common global tensor phase tangent ``i alpha A[i]``."""

    return (1j * float(alpha) * np.asarray(A, dtype=np.complex128)).astype(
        np.complex128
    )


def full_gauge_tangent(
    A: np.ndarray,
    X: np.ndarray | None = None,
    alpha: float = 0.0,
) -> np.ndarray:
    """Return ``X A[i] - A[i] X + i alpha A[i]``."""

    out = np.zeros_like(A, dtype=np.complex128)
    if X is not None:
        out += virtual_gauge_tangent(A, X)
    if alpha != 0:
        out += phase_gauge_tangent(A, alpha)
    return out


@dataclass(frozen=True)
class GaugeBasis:
    """Raw gauge basis: ``D^2`` virtual generators plus one phase generator."""

    tensors: list[np.ndarray]
    matrices: list[np.ndarray | None]
    alphas: list[float]
    names: list[str]

    @property
    def stacked(self) -> list[np.ndarray]:
        return [stack_tensor(g) for g in self.tensors]


def gauge_basis(A: np.ndarray) -> GaugeBasis:
    """Construct the raw real gauge tangent list.

    One linear combination of the virtual directions, ``X = iI``, is zero.
    The effective real rank of the raw list is therefore ``D^2`` after adding
    the physical global phase.
    """

    A = np.asarray(A, dtype=np.complex128)
    D = A.shape[1]
    tensors: list[np.ndarray] = []
    matrices: list[np.ndarray | None] = []
    alphas: list[float] = []
    names: list[str] = []
    for k, X in enumerate(antihermitian_basis(D)):
        tensors.append(virtual_gauge_tangent(A, X))
        matrices.append(X)
        alphas.append(0.0)
        names.append(f"virtual_{k}")
    tensors.append(phase_gauge_tangent(A))
    matrices.append(None)
    alphas.append(1.0)
    names.append("global_phase")
    return GaugeBasis(tensors=tensors, matrices=matrices, alphas=alphas, names=names)


def finite_virtual_gauge(A: np.ndarray, X: np.ndarray, epsilon: float) -> np.ndarray:
    """Apply ``A[i] -> U A[i] U^dagger`` with ``U = exp(epsilon X)``."""

    U = la.expm(float(epsilon) * np.asarray(X, dtype=np.complex128))
    return np.array([U @ Ai @ U.conj().T for Ai in A], dtype=np.complex128)


def finite_phase_gauge(A: np.ndarray, epsilon: float, alpha: float = 1.0) -> np.ndarray:
    """Apply ``A[i] -> exp(i epsilon alpha) A[i]``."""

    return np.exp(1j * float(epsilon) * float(alpha)) * np.asarray(
        A, dtype=np.complex128
    )


def finite_gauge_transform(
    A: np.ndarray,
    X: np.ndarray | None = None,
    epsilon: float = 0.0,
    alpha: float = 0.0,
) -> np.ndarray:
    """Apply a finite virtual gauge and common tensor phase."""

    out = np.asarray(A, dtype=np.complex128)
    if X is not None:
        out = finite_virtual_gauge(out, X, epsilon)
    if alpha != 0.0:
        out = finite_phase_gauge(out, epsilon, alpha)
    return out.astype(np.complex128)
