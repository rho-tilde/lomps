"""Matrix-free projection onto the true-gauge-horizontal uMPS tangent.

This module deliberately removes only redundancies of the infinite uniform
MPS representation: virtual unitary similarity transformations and the
common tensor phase.  It does *not* identify physical tensors that happen to
produce the same finite-window RDM.

The projector is built from the small raw gauge matrix (``D**2 + 1`` columns)
rather than from a basis of the entire Stiefel tangent.  This is important for
an analytic-gradient CG method, where constructing the full horizontal basis
would throw away much of the intended runtime advantage.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np
import scipy.linalg as la

from .canonical import stack_tensor, unstack_tensor
from .gauge import gauge_basis
from .tangent import as_real_columns, vectorize_complex_real


Array = np.ndarray


def real_vector_to_complex_matrix(vector: Array, shape: tuple[int, int]) -> Array:
    """Undo the package's ``[Re X, Im X]`` column-major vectorization."""

    vector = np.asarray(vector, dtype=float)
    entries = int(np.prod(shape))
    if vector.shape != (2 * entries,):
        raise ValueError(
            f"real vector must have shape {(2 * entries,)}, got {vector.shape}"
        )
    return (
        vector[:entries] + 1j * vector[entries:]
    ).reshape(shape, order="F").astype(np.complex128)


def stiefel_project(W: Array, X: Array) -> Array:
    """Euclidean orthogonal projection onto ``T_W St(n, p)``."""

    W = np.asarray(W, dtype=np.complex128)
    X = np.asarray(X, dtype=np.complex128)
    if X.shape != W.shape:
        raise ValueError(f"X must have shape {W.shape}, got {X.shape}")
    overlap = W.conj().T @ X
    hermitian_part = 0.5 * (overlap + overlap.conj().T)
    return (X - W @ hermitian_part).astype(np.complex128)


@dataclass(frozen=True)
class GaugeHorizontalProjector:
    """Orthogonal projector removing only the genuine uMPS gauge tangent."""

    W: Array
    gauge_columns: Array
    gauge_singular_values: Array
    gauge_rank: int
    tolerance: float

    @classmethod
    def from_tensor(
        cls,
        A: Array,
        W: Array | None = None,
        *,
        tolerance: float = 1e-10,
    ) -> "GaugeHorizontalProjector":
        A = np.asarray(A, dtype=np.complex128)
        if A.ndim != 3 or A.shape[1] != A.shape[2]:
            raise ValueError("A must have shape (d, D, D)")
        d, D, _ = A.shape
        if W is None:
            W = stack_tensor(A)
        W = np.asarray(W, dtype=np.complex128)
        if W.shape != (d * D, D):
            raise ValueError(f"W must have shape {(d * D, D)}, got {W.shape}")
        if tolerance < 0:
            raise ValueError("tolerance must be non-negative")

        # Numerical Stiefel projection makes the subset relation
        # G_A <= T_W St explicit even when the input is only canonical to
        # floating-point precision.
        raw_gauge = [stiefel_project(W, G) for G in gauge_basis(A).stacked]
        gauge_real = as_real_columns(raw_gauge)
        if gauge_real.size == 0:
            singular_values = np.array([], dtype=float)
            columns = np.zeros((2 * W.size, 0), dtype=float)
            rank = 0
        else:
            q, triangular, _ = la.qr(
                gauge_real,
                mode="economic",
                pivoting=True,
                check_finite=False,
            )
            singular_values = la.svdvals(triangular, check_finite=False)
            rank = int(np.count_nonzero(singular_values > tolerance))
            columns = np.asarray(q[:, :rank], dtype=float)
        return cls(
            W=W.copy(),
            gauge_columns=columns,
            gauge_singular_values=np.asarray(singular_values, dtype=float),
            gauge_rank=rank,
            tolerance=float(tolerance),
        )

    @classmethod
    def from_stiefel(
        cls,
        W: Array,
        *,
        local_dimension: int,
        tolerance: float = 1e-10,
    ) -> "GaugeHorizontalProjector":
        W = np.asarray(W, dtype=np.complex128)
        if W.ndim != 2:
            raise ValueError("W must be a matrix")
        D = W.shape[1]
        d = int(local_dimension)
        return cls.from_tensor(
            unstack_tensor(W, d, D), W, tolerance=tolerance
        )

    def project(self, X: Array) -> Array:
        """Project ``X`` onto the true-gauge-horizontal Stiefel tangent."""

        tangent = stiefel_project(self.W, X)
        vector = vectorize_complex_real(tangent)
        if self.gauge_rank:
            vector = vector - self.gauge_columns @ (
                self.gauge_columns.T @ vector
            )
        return real_vector_to_complex_matrix(vector, self.W.shape)

    def gauge_coordinates(self, X: Array) -> Array:
        """Coordinates of ``X`` along an orthonormal genuine-gauge basis."""

        vector = vectorize_complex_real(np.asarray(X, dtype=np.complex128))
        return self.gauge_columns.T @ vector

    def gauge_overlap_norm(self, X: Array) -> float:
        """Euclidean norm of the genuine-gauge component of ``X``."""

        return float(la.norm(self.gauge_coordinates(X)))

    def tangency_error(self, X: Array) -> float:
        """Linearized Stiefel constraint residual at this projector's point."""

        X = np.asarray(X, dtype=np.complex128)
        residual = self.W.conj().T @ X + X.conj().T @ self.W
        return float(la.norm(residual))

    def historical_slice_representative(self, X: Array) -> Array:
        """Return the gauge-equivalent tangent satisfying ``W^dagger X=0``.

        This is useful for applying the historical quotient preconditioner to
        an arbitrary true-horizontal vector.  Only a genuine uMPS gauge
        tangent is added; the represented physical tangent is unchanged.
        """

        horizontal = self.project(X)
        constraint = vectorize_complex_real(self.W.conj().T @ horizontal)
        coefficients = -self.historical_slice_inverse @ constraint
        gauge_vector = self.gauge_columns @ coefficients
        gauge = real_vector_to_complex_matrix(gauge_vector, self.W.shape)
        return (horizontal + gauge).astype(np.complex128)

    @cached_property
    def historical_slice_inverse(self) -> Array:
        """Cached gauge-to-historical-slice correction map."""

        if not self.gauge_rank:
            D = self.W.shape[1]
            return np.zeros((0, 2 * D * D), dtype=float)
        gauge_matrices = [
            real_vector_to_complex_matrix(
                self.gauge_columns[:, index], self.W.shape
            )
            for index in range(self.gauge_rank)
        ]
        constraints = as_real_columns(
            [self.W.conj().T @ gauge for gauge in gauge_matrices]
        )
        return np.asarray(
            np.linalg.pinv(
                constraints, rcond=max(self.tolerance, 1e-15)
            ),
            dtype=float,
        )


@dataclass(frozen=True)
class GrassmannHorizontalProjector:
    """Projector onto the TDVP left-gauge slice ``W^dagger X = 0``.

    This is the historical Grassmann representative of the uMPS quotient.
    It intentionally removes the entire component in the columns of ``W``
    rather than choosing the Euclidean complement of the true gauge orbit.
    """

    W: Array
    gauge_rank: int
    tolerance: float

    @classmethod
    def from_tensor(
        cls,
        A: Array,
        W: Array | None = None,
        *,
        tolerance: float = 1e-10,
    ) -> "GrassmannHorizontalProjector":
        A = np.asarray(A, dtype=np.complex128)
        if A.ndim != 3 or A.shape[1] != A.shape[2]:
            raise ValueError("A must have shape (d, D, D)")
        d, D, _ = A.shape
        if W is None:
            W = stack_tensor(A)
        W = np.asarray(W, dtype=np.complex128)
        if W.shape != (d * D, D):
            raise ValueError(f"W must have shape {(d * D, D)}, got {W.shape}")
        if tolerance < 0:
            raise ValueError("tolerance must be non-negative")
        return cls(W=W.copy(), gauge_rank=D * D, tolerance=float(tolerance))

    def project(self, X: Array) -> Array:
        """Return the orthogonal representative satisfying ``W^dagger X=0``."""

        X = np.asarray(X, dtype=np.complex128)
        if X.shape != self.W.shape:
            raise ValueError(f"X must have shape {self.W.shape}, got {X.shape}")
        return (X - self.W @ (self.W.conj().T @ X)).astype(np.complex128)

    def gauge_coordinates(self, X: Array) -> Array:
        """Return the matrix constraint ``W^dagger X`` as real coordinates."""

        return vectorize_complex_real(
            self.W.conj().T @ np.asarray(X, dtype=np.complex128)
        )

    def gauge_overlap_norm(self, X: Array) -> float:
        """Norm of the violation of the TDVP left-gauge condition."""

        return float(la.norm(self.W.conj().T @ X))

    def tangency_error(self, X: Array) -> float:
        """Linearized Stiefel residual at this projector's point."""

        X = np.asarray(X, dtype=np.complex128)
        residual = self.W.conj().T @ X + X.conj().T @ self.W
        return float(la.norm(residual))

    def historical_slice_representative(self, X: Array) -> Array:
        """Return ``X`` in this projector's own Grassmann slice."""

        return self.project(X)


def gauge_horizontal_project(
    A: Array,
    X: Array,
    *,
    tolerance: float = 1e-10,
) -> Array:
    """Convenience wrapper for one gauge-horizontal projection."""

    projector = GaugeHorizontalProjector.from_tensor(A, tolerance=tolerance)
    return projector.project(X)
