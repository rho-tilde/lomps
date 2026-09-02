"""Structured true-gauge projection without a dense gauge-column QR.

The gauge map is applied through tensor contractions and its normal equations
are solved iteratively in ``D**2`` real parameters.  This research projector
is aimed at the large-D regime where factorizing the explicit gauge matrix is
more expensive than the analytic RDM VJP itself.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg as la
from scipy.sparse.linalg import LinearOperator, cg

from .canonical import stack_tensor
from .horizontal import stiefel_project


Array = np.ndarray


def _metric(X: Array, Y: Array) -> float:
    return float(np.vdot(X, Y).real)


@dataclass(frozen=True)
class StructuredProjectionInfo:
    iterations: int
    solver_info: int
    adjoint_residual_norm: float


class StructuredGaugeHorizontalProjector:
    """Iterative projector using the structured uMPS gauge operator."""

    def __init__(
        self,
        A: Array,
        W: Array | None = None,
        *,
        rtol: float = 1e-10,
        atol: float = 0.0,
        maximum_iterations: int | None = 400,
    ):
        self.A = np.asarray(A, dtype=np.complex128)
        if self.A.ndim != 3 or self.A.shape[1] != self.A.shape[2]:
            raise ValueError("A must have shape (d, D, D)")
        self.d, self.D, _ = self.A.shape
        self.W = stack_tensor(self.A) if W is None else np.asarray(
            W, dtype=np.complex128
        )
        if self.W.shape != (self.d * self.D, self.D):
            raise ValueError("W shape does not match A")
        self.rtol = float(rtol)
        self.atol = float(atol)
        self.maximum_iterations = maximum_iterations
        if self.rtol <= 0 or self.atol < 0:
            raise ValueError("invalid iterative-solver tolerances")

        # The sum of all diagonal iE_aa generators is iI and maps to zero.
        # Dropping one diagonal generator leaves a basis of the virtual gauge
        # image; the separate common tensor phase supplies the final direction.
        self._upper_rows, self._upper_columns = np.triu_indices(self.D, k=1)
        self.parameter_count = self.D * self.D
        self.gauge_rank = self.parameter_count
        self.last_projection_info: StructuredProjectionInfo | None = None
        self.last_historical_info: StructuredProjectionInfo | None = None

    def _virtual_matrix(self, coefficients: Array) -> Array:
        coefficients = np.asarray(coefficients[:-1], dtype=float)
        expected = self.D * self.D - 1
        if coefficients.shape != (expected,):
            raise ValueError("virtual gauge coefficient shape mismatch")
        matrix = np.zeros((self.D, self.D), dtype=np.complex128)
        diagonal_count = self.D - 1
        diagonal = np.arange(diagonal_count)
        matrix[diagonal, diagonal] = 1j * coefficients[:diagonal_count]
        off_diagonal = coefficients[diagonal_count:]
        real_parts = off_diagonal[0::2] / np.sqrt(2.0)
        imaginary_parts = off_diagonal[1::2] / np.sqrt(2.0)
        rows = self._upper_rows
        columns = self._upper_columns
        matrix[rows, columns] = real_parts + 1j * imaginary_parts
        matrix[columns, rows] = -real_parts + 1j * imaginary_parts
        return matrix

    def _virtual_coordinates(self, matrix: Array) -> Array:
        """Real coordinates against the reduced anti-Hermitian basis."""

        matrix = np.asarray(matrix, dtype=np.complex128)
        diagonal = np.real(-1j * np.diag(matrix)[: self.D - 1])
        rows = self._upper_rows
        columns = self._upper_columns
        real_parts = np.real(
            (matrix[rows, columns] - matrix[columns, rows])
            / np.sqrt(2.0)
        )
        imaginary_parts = np.real(
            -1j
            * (matrix[rows, columns] + matrix[columns, rows])
            / np.sqrt(2.0)
        )
        off_diagonal = np.empty(2 * len(rows), dtype=float)
        off_diagonal[0::2] = real_parts
        off_diagonal[1::2] = imaginary_parts
        return np.concatenate([diagonal, off_diagonal])

    def _gauge(self, coefficients: Array) -> Array:
        coefficients = np.asarray(coefficients, dtype=float)
        if coefficients.shape != (self.parameter_count,):
            raise ValueError("gauge coefficient shape mismatch")
        K = self._virtual_matrix(coefficients)
        alpha = float(coefficients[-1])
        out = np.array(
            [K @ Ai - Ai @ K for Ai in self.A], dtype=np.complex128
        )
        out += 1j * alpha * self.A
        return stack_tensor(out)

    def _gauge_adjoint(self, X: Array) -> Array:
        tensor = np.asarray(X, dtype=np.complex128).reshape(
            self.d, self.D, self.D
        )
        virtual_adjoint = np.zeros(
            (self.D, self.D), dtype=np.complex128
        )
        for Ai, Xi in zip(self.A, tensor):
            virtual_adjoint += Xi @ Ai.conj().T - Ai.conj().T @ Xi
        virtual = self._virtual_coordinates(virtual_adjoint)
        phase = _metric(1j * self.W, X)
        return np.concatenate([virtual, np.array([phase])])

    def _constraint(self, coefficients: Array) -> Array:
        K = self._virtual_matrix(np.asarray(coefficients, dtype=float))
        alpha = float(coefficients[-1])
        value = -K + 1j * alpha * np.eye(
            self.D, dtype=np.complex128
        )
        for Ai in self.A:
            value += Ai.conj().T @ K @ Ai
        return value

    def _constraint_adjoint(self, Y: Array) -> Array:
        Y = np.asarray(Y, dtype=np.complex128)
        virtual_adjoint = -Y.copy()
        for Ai in self.A:
            virtual_adjoint += Ai @ Y @ Ai.conj().T
        virtual = self._virtual_coordinates(virtual_adjoint)
        phase = _metric(1j * np.eye(self.D), Y)
        return np.concatenate([virtual, np.array([phase])])

    def _normal_solve(
        self,
        rhs: Array,
        normal_matvec: object,
    ) -> tuple[Array, int, int]:
        iterations = 0

        def callback(_: Array) -> None:
            nonlocal iterations
            iterations += 1

        operator = LinearOperator(
            (self.parameter_count, self.parameter_count),
            matvec=normal_matvec,  # type: ignore[arg-type]
            dtype=float,
        )
        coefficients, info = cg(
            operator,
            np.asarray(rhs, dtype=float),
            rtol=self.rtol,
            atol=self.atol,
            maxiter=self.maximum_iterations,
            callback=callback,
        )
        return coefficients.astype(float), iterations, int(info)

    def project(self, X: Array) -> Array:
        tangent = stiefel_project(self.W, X)
        rhs = self._gauge_adjoint(tangent)
        coefficients, iterations, info = self._normal_solve(
            rhs,
            lambda value: self._gauge_adjoint(self._gauge(value)),
        )
        horizontal = tangent - self._gauge(coefficients)
        residual = float(la.norm(self._gauge_adjoint(horizontal)))
        self.last_projection_info = StructuredProjectionInfo(
            iterations=iterations,
            solver_info=info,
            adjoint_residual_norm=residual,
        )
        if info != 0:
            raise RuntimeError(
                f"structured gauge projection CG failed with info={info}"
            )
        return horizontal.astype(np.complex128)

    def historical_slice_representative(self, X: Array) -> Array:
        horizontal = self.project(X)
        constraint_rhs = -self.W.conj().T @ horizontal
        rhs = self._constraint_adjoint(constraint_rhs)
        coefficients, iterations, info = self._normal_solve(
            rhs,
            lambda value: self._constraint_adjoint(
                self._constraint(value)
            ),
        )
        historical = horizontal + self._gauge(coefficients)
        residual = float(la.norm(self.W.conj().T @ historical))
        self.last_historical_info = StructuredProjectionInfo(
            iterations=iterations,
            solver_info=info,
            adjoint_residual_norm=residual,
        )
        if info != 0:
            raise RuntimeError(
                f"structured historical-slice CG failed with info={info}"
            )
        return historical.astype(np.complex128)

    def tangency_error(self, X: Array) -> float:
        residual = self.W.conj().T @ X + X.conj().T @ self.W
        return float(la.norm(residual))

    def gauge_overlap_norm(self, X: Array) -> float:
        return float(la.norm(self._gauge_adjoint(X)))
