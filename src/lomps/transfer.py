"""Transfer channel construction and fixed-point diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg as la
from scipy.sparse.linalg import LinearOperator, gmres


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


def solve_delta_fixed_point_iterative(
    A: np.ndarray,
    delta_A: np.ndarray,
    r: np.ndarray,
    *,
    rtol: float = 1e-8,
    atol: float = 0.0,
    maximum_iterations: int | None = None,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Matrix-free constrained solve for the right-fixed-point derivative.

    The rank-one stabilized operator

    ``X -> X - E(X) + r trace(X)``

    is nonsingular for an injective left-canonical tensor.  Its solution for
    the traceless tangent source equals the constrained derivative.  Unlike
    :func:`solve_delta_fixed_point`, this routine never constructs the
    ``D**2`` transfer matrix or factorizes a dense augmented system.
    """

    A = np.asarray(A, dtype=np.complex128)
    delta_A = np.asarray(delta_A, dtype=np.complex128)
    r = np.asarray(r, dtype=np.complex128)
    D = A.shape[1]
    if delta_A.shape != A.shape or r.shape != (D, D):
        raise ValueError("delta_A or r shape does not match A")
    if rtol <= 0 or atol < 0:
        raise ValueError("invalid iterative-solver tolerances")

    source = delta_channel(A, delta_A, r)

    def matvec(vector: np.ndarray) -> np.ndarray:
        matrix = unvec(vector, D)
        value = matrix - apply_channel(A, matrix)
        value += r * np.trace(matrix)
        return vec(value)

    operator = LinearOperator(
        (D * D, D * D), matvec=matvec, dtype=np.complex128
    )
    iterations = 0

    def callback(_: object) -> None:
        nonlocal iterations
        iterations += 1

    solution, solver_info = gmres(
        operator,
        vec(source),
        rtol=float(rtol),
        atol=float(atol),
        maxiter=maximum_iterations,
        callback=callback,
        callback_type="pr_norm",
    )
    if solver_info != 0:
        raise RuntimeError(
            "iterative fixed-point derivative solve failed with "
            f"info={solver_info}"
        )
    delta_r = unvec(solution, D)
    delta_r = 0.5 * (delta_r + delta_r.conj().T)
    delta_r -= r * np.trace(delta_r)
    equation_residual = (
        delta_r - apply_channel(A, delta_r) - source
    )
    return delta_r.astype(np.complex128), {
        "equation_residual": float(la.norm(equation_residual)),
        "trace_abs": float(abs(np.trace(delta_r))),
        "hermiticity_error": float(la.norm(delta_r - delta_r.conj().T)),
        "iterations": iterations,
        "solver_info": int(solver_info),
    }


@dataclass
class DenseFixedPointResponseSolver:
    """Reusable LU solve for forward and adjoint fixed-point responses.

    For one outer LM iteration, every JVP and VJP uses the same stabilized
    transfer operator.  Factoring it once can therefore be cheaper than
    restarting hundreds of matrix-free GMRES solves.  This is an optional
    `O(D**4)` memory / `O(D**6)` setup tradeoff; it does not construct the RDM
    Jacobian or a doubled-layer ring.
    """

    A: np.ndarray
    r: np.ndarray
    lu_and_pivots: tuple[np.ndarray, np.ndarray]

    @classmethod
    def from_tensor(
        cls, A: np.ndarray, r: np.ndarray
    ) -> "DenseFixedPointResponseSolver":
        A = np.asarray(A, dtype=np.complex128)
        r = np.asarray(r, dtype=np.complex128)
        D = A.shape[1]
        if A.ndim != 3 or A.shape[2] != D or r.shape != (D, D):
            raise ValueError("A or r has an incompatible shape")
        stabilized = (
            np.eye(D * D, dtype=np.complex128)
            - transfer_matrix(A)
            + np.outer(vec(r), trace_row(D))
        )
        lu_and_pivots = la.lu_factor(
            stabilized,
            overwrite_a=True,
            check_finite=False,
        )
        return cls(A.copy(), r.copy(), lu_and_pivots)

    @property
    def storage_bytes(self) -> int:
        return int(
            self.lu_and_pivots[0].nbytes
            + self.lu_and_pivots[1].nbytes
        )

    def solve_delta(
        self, delta_A: np.ndarray
    ) -> tuple[np.ndarray, dict[str, float | int]]:
        delta_A = np.asarray(delta_A, dtype=np.complex128)
        if delta_A.shape != self.A.shape:
            raise ValueError("delta_A shape does not match A")
        D = self.A.shape[1]
        source = delta_channel(self.A, delta_A, self.r)
        solution = la.lu_solve(
            self.lu_and_pivots,
            vec(source),
            trans=0,
            check_finite=False,
        )
        delta_r = unvec(solution, D)
        delta_r = 0.5 * (delta_r + delta_r.conj().T)
        delta_r -= self.r * np.trace(delta_r)
        equation_residual = (
            delta_r - apply_channel(self.A, delta_r) - source
        )
        return delta_r.astype(np.complex128), {
            "equation_residual": float(la.norm(equation_residual)),
            "trace_abs": float(abs(np.trace(delta_r))),
            "hermiticity_error": float(la.norm(delta_r - delta_r.conj().T)),
            "iterations": 0,
            "solver_info": 0,
        }

    def solve_adjoint(self, source: np.ndarray) -> np.ndarray:
        source = np.asarray(source, dtype=np.complex128)
        D = self.A.shape[1]
        if source.shape != (D, D):
            raise ValueError("adjoint source shape does not match A")
        solution = la.lu_solve(
            self.lu_and_pivots,
            vec(source),
            trans=2,
            check_finite=False,
        )
        response = unvec(solution, D)
        return (0.5 * (response + response.conj().T)).astype(np.complex128)
