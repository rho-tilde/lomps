"""Gauge-orthogonal Levenberg--Marquardt optimization for MPS RDMs.

The optimization variable is a left-canonical tensor represented by the
Stiefel matrix ``W``.  At every iteration we construct the *true* infinitesimal
MPS gauge space, take its real Euclidean orthogonal complement inside the full
Stiefel tangent, restrict the RDM Jacobian to that quotient slice, and solve a
damped Gauss--Newton problem there.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Literal

import numpy as np
import scipy.linalg as la
from scipy.sparse.linalg import ArpackNoConvergence, eigs

from .canonical import polar_retraction, stack_tensor, unstack_tensor
from .differential import real_vectorize_rho
from .gauge import gauge_basis
from .rdm import block_rdm
from .tangent import as_real_columns, tangent_bases
from .transfer import (
    delta_channel,
    right_fixed_point,
    trace_row,
    transfer_matrix,
    unvec,
    vec,
)


Array = np.ndarray
FixedPointSolver = Literal["dense", "fast"]


@dataclass(frozen=True)
class LMOptions:
    """Numerical controls for gauge-orthogonal LM."""

    max_iterations: int = 40_000
    gradient_tolerance: float = 1e-11
    residual_tolerance: float = 0.0
    cost_tolerance: float = 0.0
    rank_tolerance: float = 1e-10
    initial_damping: float = 1e-4
    trust_radius: float = 1.0
    armijo_c1: float = 1e-4
    minimum_step: float = 1e-8
    maximum_damping_trials: int = 8
    plateau_window: int = 50
    plateau_relative_cost_drop: float = 1e-4
    plateau_absolute_cost_drop: float = 1e-20
    maximum_seconds: float = 600.0
    fixed_point_solver: FixedPointSolver = "dense"
    verbose: bool = True


@dataclass(frozen=True)
class LMEvaluation:
    W: Array
    A: Array
    rho: Array
    residual: Array
    residual_vector: Array
    cost: float
    basis: Array
    jacobian: Array
    gradient: Array
    svd_u: Array
    singular_values: Array
    svd_vh: Array
    visible_rank: int
    rank_tolerance: float


@dataclass(frozen=True)
class LMRecord:
    iteration: int
    cost: float
    residual_norm: float
    gradient_norm: float
    damping: float
    step_size: float
    step_norm: float
    raw_step_norm: float
    visible_rank: int
    basis_dimension: int
    accepted: bool
    elapsed_seconds: float


@dataclass(frozen=True)
class LMResult:
    W: Array
    cost: float
    residual_norm: float
    gradient_norm: float
    status: str
    history: tuple[LMRecord, ...]


@dataclass(frozen=True)
class CGOptions:
    """Numerical controls for Grassmann conjugate-gradient initialization."""

    max_iterations: int = 40_000
    gradient_tolerance: float = 1e-11
    cost_tolerance: float = 3e-16
    rank_tolerance: float = 1e-10
    initial_step: float | None = None
    precondition: bool = True
    restart: int = 100
    maximum_seconds: float = 900.0
    fixed_point_solver: FixedPointSolver = "dense"
    verbose: bool = True


def _safe_svd(matrix: Array, *, full_matrices: bool) -> tuple[Array, Array, Array]:
    try:
        return la.svd(
            matrix,
            full_matrices=full_matrices,
            check_finite=False,
            lapack_driver="gesdd",
        )
    except (la.LinAlgError, ValueError):
        return la.svd(
            matrix,
            full_matrices=full_matrices,
            check_finite=False,
            lapack_driver="gesvd",
        )


def gauge_orthogonal_basis(A: Array, W: Array, tolerance: float) -> list[Array]:
    """Orthonormal basis of the true-gauge-orthogonal Stiefel slice."""

    d, D, _ = A.shape
    full = tangent_bases(W, d, D).full
    full_real = as_real_columns(full)
    gauge_real = as_real_columns(gauge_basis(A).stacked)
    gauge_coordinates = full_real.T @ gauge_real
    _, singular_values, vh = _safe_svd(
        gauge_coordinates.T, full_matrices=True
    )
    gauge_rank = int(np.count_nonzero(singular_values > tolerance))
    null_coordinates = vh[gauge_rank:].T
    # ``full_real`` uses [Re, Im] column-major vectorization.  Multiplying all
    # null coordinates at once avoids 288 x 432 Python-level array additions
    # for D=12.
    horizontal_real = full_real @ null_coordinates
    entries = W.size
    basis = []
    for column in range(horizontal_real.shape[1]):
        vector = horizontal_real[:, column]
        direction = (
            vector[:entries] + 1j * vector[entries:]
        ).reshape(W.shape, order="F")
        basis.append(direction.astype(np.complex128))
    return basis


def _real_metric(X: Array, Y: Array) -> float:
    return float(np.real(np.trace(np.asarray(X).conj().T @ np.asarray(Y))))


def grassmann_project(W: Array, X: Array) -> Array:
    """Project a matrix onto the Grassmann-horizontal Stiefel tangent."""

    return X - W @ (W.conj().T @ X)


def batched_rdm_jacobian(
    A: Array,
    tangent_tensors: list[Array],
    block_length: int,
    r: Array,
) -> Array:
    """Build the real RDM Jacobian with one shared fixed-point solve.

    The conventional implementation solves the same augmented operator once
    per tangent.  Here all ``delta E(r)`` vectors are assembled as multiple
    right-hand sides, so the least-squares factorization is performed once.
    The directional block products and RDM contractions are batched as well.
    """

    if len(tangent_tensors) == 0:
        return np.zeros((0, 0), dtype=float)
    A = np.asarray(A, dtype=np.complex128)
    tangents = np.asarray(tangent_tensors, dtype=np.complex128)
    count, d, D, _ = tangents.shape

    lhs = np.eye(D * D, dtype=np.complex128) - transfer_matrix(A)
    augmented_lhs = np.vstack([lhs, trace_row(D)])
    right_sides = np.column_stack(
        [vec(delta_channel(A, delta_A, r)) for delta_A in tangents]
    )
    augmented_rhs = np.vstack(
        [right_sides, np.zeros((1, count), dtype=np.complex128)]
    )
    solutions, *_ = la.lstsq(
        augmented_lhs,
        augmented_rhs,
        cond=None,
        check_finite=False,
        lapack_driver="gelsy",
    )
    delta_r = np.empty((count, D, D), dtype=np.complex128)
    identity = np.eye(D, dtype=np.complex128)
    for index in range(count):
        value = solutions[:, index].reshape(D, D, order="F")
        value = 0.5 * (value + value.conj().T)
        delta_r[index] = value - identity * (np.trace(value) / D)

    products = np.eye(D, dtype=np.complex128)[None, :, :]
    delta_products = np.zeros((count, 1, D, D), dtype=np.complex128)
    for _ in range(block_length):
        next_products = np.empty(
            (products.shape[0] * d, D, D), dtype=np.complex128
        )
        next_delta = np.empty(
            (count, products.shape[0] * d, D, D), dtype=np.complex128
        )
        out = 0
        for sequence, product in enumerate(products):
            for physical in range(d):
                next_products[out] = product @ A[physical]
                next_delta[:, out] = (
                    delta_products[:, sequence] @ A[physical]
                    + product @ tangents[:, physical]
                )
                out += 1
        products = next_products
        delta_products = next_delta

    ket = np.einsum(
        "psab,bc,tac->pst",
        delta_products,
        r,
        products.conj(),
        optimize=True,
    )
    fixed = np.einsum(
        "sab,pbc,tac->pst",
        products,
        delta_r,
        products.conj(),
        optimize=True,
    )
    bra = np.einsum(
        "sab,bc,ptac->pst",
        products,
        r,
        delta_products.conj(),
        optimize=True,
    )
    derivatives = ket + fixed + bra
    return np.column_stack(
        [real_vectorize_rho(derivatives[index]) for index in range(count)]
    )


def fast_right_fixed_point(A: Array) -> tuple[Array, dict[str, float | complex]]:
    """Dominant transfer fixed point using ARPACK, with a dense fallback."""

    A = np.asarray(A, dtype=np.complex128)
    D = A.shape[1]
    transfer = transfer_matrix(A)
    try:
        values, vectors = eigs(
            transfer,
            k=1,
            which="LM",
            tol=1e-13,
            maxiter=max(1000, 10 * D * D),
        )
        eigenvalue = values[0]
        r = unvec(vectors[:, 0], D)
        r = 0.5 * (r + r.conj().T)
        trace = np.trace(r)
        if abs(trace) < 1e-14:
            raise RuntimeError("iterative fixed point has nearly zero trace")
        r /= trace
        r = 0.5 * (r + r.conj().T)
        r /= np.trace(r)
        residual = sum(Ai @ r @ Ai.conj().T for Ai in A) - r
        residual_norm = float(la.norm(residual))
        minimum_eigenvalue = float(np.min(la.eigvalsh(r)).real)
        if residual_norm > 1e-10 or minimum_eigenvalue < -1e-10:
            raise RuntimeError("iterative fixed point failed validation")
        return r.astype(np.complex128), {
            "eigenvalue": eigenvalue,
            "residual": residual_norm,
            "trace": complex(np.trace(r)),
            "hermiticity_error": float(la.norm(r - r.conj().T)),
            "min_eigenvalue": minimum_eigenvalue,
        }
    except (ArpackNoConvergence, la.LinAlgError, RuntimeError, ValueError):
        return right_fixed_point(A)


def optimizer_right_fixed_point(
    A: Array,
    solver: FixedPointSolver,
) -> tuple[Array, dict[str, float | complex]]:
    """Return the ansatz fixed point with the requested optimizer policy.

    ``dense`` uses the deterministic dense eigensolver. ``fast`` uses ARPACK
    first and falls back to dense if the iterative solve fails validation.
    """

    if solver == "dense":
        return right_fixed_point(A)
    if solver == "fast":
        return fast_right_fixed_point(A)
    raise ValueError("fixed_point_solver must be 'dense' or 'fast'")


class GaugeOrthogonalLM:
    """Damped Gauss--Newton/LM solver on the MPS quotient tangent."""

    def __init__(self, block_length: int, options: LMOptions | None = None):
        self.block_length = int(block_length)
        self.options = options or LMOptions()
        if self.options.fixed_point_solver not in ("dense", "fast"):
            raise ValueError("fixed_point_solver must be 'dense' or 'fast'")

    def _right_fixed_point(self, A: Array) -> tuple[Array, dict[str, float | complex]]:
        return optimizer_right_fixed_point(A, self.options.fixed_point_solver)

    def evaluate(self, W: Array, rho_target: Array) -> LMEvaluation:
        W = np.asarray(W, dtype=np.complex128)
        dD, D = W.shape
        if dD % D:
            raise ValueError("W must have shape (d*D, D)")
        d = dD // D
        A = unstack_tensor(W, d, D)
        r, _ = self._right_fixed_point(A)
        rho = block_rdm(A, self.block_length, r)
        residual = rho - np.asarray(rho_target, dtype=np.complex128)
        residual_vector = real_vectorize_rho(residual)
        basis_list = gauge_orthogonal_basis(
            A, W, tolerance=self.options.rank_tolerance
        )
        basis = np.asarray(basis_list, dtype=np.complex128)
        tangent_tensors = basis.reshape(len(basis), d, D, D)
        jacobian = batched_rdm_jacobian(
            A, tangent_tensors, self.block_length, r
        )
        gradient = jacobian.T @ residual_vector
        svd_u, singular_values, svd_vh = _safe_svd(
            jacobian, full_matrices=False
        )
        if singular_values.size:
            effective_tolerance = max(
                self.options.rank_tolerance,
                max(jacobian.shape) * np.finfo(float).eps * singular_values[0],
            )
            visible_rank = int(
                np.count_nonzero(singular_values > effective_tolerance)
            )
        else:
            effective_tolerance = self.options.rank_tolerance
            visible_rank = 0
        return LMEvaluation(
            W=W,
            A=A,
            rho=rho,
            residual=residual,
            residual_vector=residual_vector,
            cost=0.5 * float(np.dot(residual_vector, residual_vector)),
            basis=basis,
            jacobian=jacobian,
            gradient=gradient,
            svd_u=svd_u,
            singular_values=singular_values,
            svd_vh=svd_vh,
            visible_rank=visible_rank,
            rank_tolerance=effective_tolerance,
        )

    def _lm_direction(
        self, evaluation: LMEvaluation, damping: float
    ) -> tuple[Array, float, float, float]:
        coefficients = -evaluation.svd_vh.T @ (
            (
                evaluation.singular_values
                / (evaluation.singular_values**2 + damping)
            )
            * (evaluation.svd_u.T @ evaluation.residual_vector)
        )
        raw_norm = float(la.norm(coefficients))
        if self.options.trust_radius > 0 and raw_norm > self.options.trust_radius:
            coefficients *= self.options.trust_radius / raw_norm
        direction = np.tensordot(coefficients, evaluation.basis, axes=(0, 0))
        step_norm = float(la.norm(direction))
        slope = float(np.dot(evaluation.gradient, coefficients))
        return direction, slope, step_norm, raw_norm

    def _line_search(
        self,
        evaluation: LMEvaluation,
        direction: Array,
        slope: float,
        rho_target: Array,
    ) -> tuple[bool, float, Array, float]:
        alpha = 1.0
        while alpha >= self.options.minimum_step:
            candidate = polar_retraction(evaluation.W + alpha * direction)
            dD, D = candidate.shape
            A = unstack_tensor(candidate, dD // D, D)
            r, _ = self._right_fixed_point(A)
            residual = block_rdm(A, self.block_length, r) - rho_target
            residual_vector = real_vectorize_rho(residual)
            cost = 0.5 * float(np.dot(residual_vector, residual_vector))
            if np.isfinite(cost) and cost <= (
                evaluation.cost + self.options.armijo_c1 * alpha * slope
            ):
                return True, alpha, candidate, cost
            alpha *= 0.5
        return False, 0.0, evaluation.W, evaluation.cost

    def optimize(self, W0: Array, rho_target: Array) -> LMResult:
        W = np.asarray(W0, dtype=np.complex128).copy()
        rho_target = np.asarray(rho_target, dtype=np.complex128)
        damping = float(self.options.initial_damping)
        history: list[LMRecord] = []
        best_W = W.copy()
        best_cost = np.inf
        status = "maximum_iterations"
        final_evaluation: LMEvaluation | None = None
        best_evaluation: LMEvaluation | None = None
        optimization_started = time.perf_counter()

        for iteration in range(self.options.max_iterations + 1):
            started = time.perf_counter()
            evaluation = self.evaluate(W, rho_target)
            final_evaluation = evaluation
            residual_norm = float(np.sqrt(2.0 * evaluation.cost))
            gradient_norm = float(la.norm(evaluation.gradient))
            if evaluation.cost < best_cost:
                best_cost = evaluation.cost
                best_W = W.copy()
                best_evaluation = evaluation

            if self.options.verbose:
                print(
                    f"LM iter={iteration:4d} cost={evaluation.cost:.6e} "
                    f"residual={residual_norm:.6e} grad={gradient_norm:.6e} "
                    f"mu={damping:.3e}",
                    flush=True,
                )

            reached_residual = (
                self.options.residual_tolerance > 0
                and residual_norm <= self.options.residual_tolerance
            )
            reached_cost = (
                self.options.cost_tolerance > 0
                and evaluation.cost <= self.options.cost_tolerance
            )
            reached_gradient = gradient_norm <= self.options.gradient_tolerance
            recent_costs = [
                record.cost
                for record in history[-max(self.options.plateau_window - 1, 0) :]
            ] + [evaluation.cost]
            plateau = False
            if self.options.plateau_window > 1 and len(recent_costs) >= self.options.plateau_window:
                start_cost = recent_costs[0]
                best_recent = min(recent_costs)
                required_drop = self.options.plateau_absolute_cost_drop + (
                    self.options.plateau_relative_cost_drop
                    * max(abs(start_cost), 1e-300)
                )
                plateau = start_cost - best_recent <= required_drop
            time_limit = (
                self.options.maximum_seconds > 0
                and time.perf_counter() - optimization_started >= self.options.maximum_seconds
            )
            if (
                reached_residual
                or reached_cost
                or reached_gradient
                or plateau
                or time_limit
                or iteration == self.options.max_iterations
            ):
                if reached_residual:
                    status = "residual_tolerance"
                elif reached_cost:
                    status = "cost_tolerance"
                elif reached_gradient:
                    status = "gradient_tolerance"
                elif plateau:
                    status = "plateau"
                elif time_limit:
                    status = "time_limit"
                history.append(
                    LMRecord(
                        iteration=iteration,
                        cost=evaluation.cost,
                        residual_norm=residual_norm,
                        gradient_norm=gradient_norm,
                        damping=damping,
                        step_size=0.0,
                        step_norm=0.0,
                        raw_step_norm=0.0,
                        visible_rank=evaluation.visible_rank,
                        basis_dimension=evaluation.jacobian.shape[1],
                        accepted=False,
                        elapsed_seconds=time.perf_counter() - started,
                    )
                )
                break

            accepted = False
            alpha = 0.0
            step_norm = 0.0
            raw_step_norm = 0.0
            local_damping = damping
            candidate = W
            candidate_cost = evaluation.cost
            for _ in range(self.options.maximum_damping_trials):
                direction, slope, step_norm, raw_step_norm = self._lm_direction(
                    evaluation, local_damping
                )
                if not np.isfinite(slope) or slope >= 0:
                    local_damping *= 10.0
                    continue
                accepted, alpha, candidate, candidate_cost = self._line_search(
                    evaluation, direction, slope, rho_target
                )
                if accepted:
                    if alpha > 0.99 and raw_step_norm <= 1.05 * self.options.trust_radius:
                        damping = max(local_damping * 0.3, 1e-12)
                    elif alpha < 0.5:
                        damping = min(local_damping * 3.0, 1e12)
                    else:
                        damping = local_damping
                    W = candidate
                    break
                local_damping *= 10.0
            if not accepted:
                damping = local_damping
                status = "line_search_failed"

            history.append(
                LMRecord(
                    iteration=iteration,
                    cost=evaluation.cost,
                    residual_norm=residual_norm,
                    gradient_norm=gradient_norm,
                    damping=local_damping,
                    step_size=alpha,
                    step_norm=step_norm,
                    raw_step_norm=raw_step_norm,
                    visible_rank=evaluation.visible_rank,
                    basis_dimension=evaluation.jacobian.shape[1],
                    accepted=accepted,
                    elapsed_seconds=time.perf_counter() - started,
                )
            )
            if not accepted:
                break

        if final_evaluation is None:
            raise RuntimeError("optimizer performed no evaluation")
        if best_evaluation is None:
            raise RuntimeError("optimizer did not retain a best evaluation")
        return LMResult(
            W=best_W,
            cost=best_evaluation.cost,
            residual_norm=float(np.sqrt(2.0 * best_evaluation.cost)),
            gradient_norm=float(la.norm(best_evaluation.gradient)),
            status=status,
            history=tuple(history),
        )


def optimize_tensor(
    A0: Array,
    rho_target: Array,
    block_length: int,
    options: LMOptions | None = None,
) -> tuple[Array, LMResult]:
    """Convenience wrapper returning the optimized physical-first tensor."""

    W0 = stack_tensor(A0)
    result = GaugeOrthogonalLM(block_length, options).optimize(W0, rho_target)
    d, D, _ = A0.shape
    return unstack_tensor(result.W, d, D), result
