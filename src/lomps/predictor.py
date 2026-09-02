"""Gauge-aware predictors for consecutive uniform-MPS tensors."""

from __future__ import annotations

import numpy as np
import scipy.linalg as la
from scipy.sparse.linalg import LinearOperator, eigs

from .canonical import polar_retraction, stack_tensor, unstack_tensor


Array = np.ndarray


def trajectory_secant_predictor(
    previous: Array, current: Array, scale: float = 1.0
) -> Array:
    """Extrapolate a gauge-continuous trajectory and restore left canonicity."""

    previous = np.asarray(previous, dtype=np.complex128)
    current = np.asarray(current, dtype=np.complex128)
    if previous.shape != current.shape or current.ndim != 3:
        raise ValueError("previous and current must have the same (d, D, D) shape")
    d, D, right_D = current.shape
    if D != right_D:
        raise ValueError("virtual dimensions must be square")
    predicted_W = polar_retraction(
        stack_tensor(current)
        + float(scale) * (stack_tensor(current) - stack_tensor(previous))
    )
    return unstack_tensor(predicted_W, d, D)


def align_virtual_gauge(reference: Array, moving: Array) -> tuple[Array, dict[str, float]]:
    """Align ``moving`` to ``reference`` using their dominant mixed channel."""

    reference = np.asarray(reference, dtype=np.complex128)
    moving = np.asarray(moving, dtype=np.complex128)
    if reference.shape != moving.shape or reference.ndim != 3:
        raise ValueError("reference and moving must have the same (d, D, D) shape")
    _, D, right_D = reference.shape
    if D != right_D:
        raise ValueError("virtual dimensions must be square")

    def action(vector: Array) -> Array:
        matrix = vector.reshape(D, D, order="F")
        value = sum(
            reference[physical].conj().T @ matrix @ moving[physical]
            for physical in range(reference.shape[0])
        )
        return value.reshape(-1, order="F")

    operator = LinearOperator(
        (D * D, D * D), matvec=action, dtype=np.complex128
    )
    values, vectors = eigs(
        operator,
        k=1,
        which="LM",
        v0=np.eye(D, dtype=np.complex128).reshape(-1, order="F"),
        tol=1e-12,
        maxiter=max(1000, 20 * D * D),
    )
    mixed_fixed_point = vectors[:, 0].reshape(D, D, order="F")
    left, _, right_h = la.svd(
        mixed_fixed_point,
        full_matrices=False,
        check_finite=False,
        lapack_driver="gesvd",
    )
    gauge = left @ right_h
    aligned = np.einsum(
        "ab,sbc,cd->sad", gauge, moving, gauge.conj().T, optimize=True
    )
    before = float(la.norm(reference - moving))
    after = float(la.norm(reference - aligned))
    return aligned, {
        "mixed_eigenvalue_abs": float(abs(values[0])),
        "distance_before": before,
        "distance_after": after,
    }


def secant_predictor(previous: Array, current: Array, scale: float = 1.0) -> tuple[Array, dict[str, float]]:
    """Extrapolate two consecutive tensors after virtual-gauge alignment."""

    aligned_previous, info = align_virtual_gauge(current, previous)
    d, D, _ = current.shape
    predicted_W = polar_retraction(
        stack_tensor(current)
        + float(scale) * (stack_tensor(current) - stack_tensor(aligned_previous))
    )
    predicted = unstack_tensor(predicted_W, d, D)
    return predicted, {**info, "scale": float(scale)}
