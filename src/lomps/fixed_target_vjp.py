"""Analytic fixed-target RDM gradient with true uMPS gauge removal.

The scalar objective is

    C(A) = 1/2 ||rho_L(A) - rho_target||_F^2.

Its gradient is evaluated as a vector--Jacobian product (``J^dagger r``)
using the established analytic contraction in :mod:`lomps.fixed_target_cg`.
No full RDM Jacobian and no finite-window-visible subspace are constructed.
Only the genuine infinite-uMPS gauge tangent is projected out.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Literal

import numpy as np
import scipy.linalg as la
from scipy.sparse.linalg import LinearOperator, gmres

from .canonical import stack_tensor, unstack_tensor
from .fixed_target_cg import FixedTargetHSCost, _ncon, _to_old_tensor
from .horizontal import (
    GaugeHorizontalProjector,
    GrassmannHorizontalProjector,
    stiefel_project,
)
from .optimizer import optimizer_right_fixed_point
from .rdm import block_product_levels
from .structured_horizontal import StructuredGaugeHorizontalProjector
from .tangent import TangentSlice
from .transfer import DenseFixedPointResponseSolver


Array = np.ndarray
FixedPointSolver = Literal["dense", "fast"]
GaugeProjector = Literal["dense", "structured"]
GradientProjection = Literal["exact", "stiefel"]


def _efficient_ambient_rdm_vjp(
    A: Array,
    r: Array,
    weight: Array,
    block_length: int,
    *,
    adjoint_rtol: float,
    adjoint_atol: float,
    adjoint_maximum_iterations: int | None,
    response_solver: DenseFixedPointResponseSolver | None = None,
    product_levels: tuple[Array, ...] | None = None,
) -> tuple[Array, int]:
    """Apply the finite-window RDM adjoint without an open derivative tensor.

    Prefix/suffix products fuse the finite-window weight into the ket and bra
    insertion contractions.  The right-fixed-point response is handled by one
    matrix-free adjoint solve.  Peak storage is therefore set by product
    levels and ``D x D`` batches, rather than by an object carrying all
    physical RDM indices plus open tensor indices.
    """

    A = np.asarray(A, dtype=np.complex128)
    r = np.asarray(r, dtype=np.complex128)
    weight = np.asarray(weight, dtype=np.complex128)
    d, D, _ = A.shape
    dimension = d**block_length
    if weight.shape != (dimension, dimension):
        raise ValueError(
            f"weight must have shape {(dimension, dimension)}, got "
            f"{weight.shape}"
        )
    # Only the Hermitian part couples to a physical RDM differential.
    weight = 0.5 * (weight + weight.conj().T)
    levels = (
        block_product_levels(A, block_length)
        if product_levels is None
        else product_levels
    )
    if len(levels) != block_length + 1:
        raise ValueError(
            "product_levels must contain levels zero through block_length"
        )
    products = levels[-1]
    product_daggers = products.conj().transpose(0, 2, 1)
    weighted_bra = np.einsum(
        "st,tij->sij", weight.conj(), product_daggers, optimize=True
    )

    gradient = np.zeros_like(A, dtype=np.complex128)
    for site in range(block_length):
        remaining = block_length - site - 1
        prefix_count = d**site
        suffix_count = d**remaining
        prefixes = levels[site][:, None, None, :, :]
        suffixes = levels[remaining][None, None, :, :, :]
        weighted_by_sequence = weighted_bra.reshape(
            prefix_count, d, suffix_count, D, D
        )
        insertion_coefficients = (
            suffixes @ r @ weighted_by_sequence @ prefixes
        )
        gradient += 2.0 * np.sum(
            insertion_coefficients.conj().transpose(0, 1, 2, 4, 3),
            axis=(0, 2),
        )

    # Gradient with respect to the normalized right fixed point.
    fixed_point_weight = np.einsum(
        "sab,tac,st->bc",
        products.conj(),
        products,
        weight,
        optimize=True,
    )
    if response_solver is None:
        iterations = 0

        def matvec(vector: Array) -> Array:
            matrix = np.asarray(vector, dtype=np.complex128).reshape(D, D)
            value = matrix.copy()
            for Ai in A:
                value -= Ai.conj().T @ matrix @ Ai
            value += np.eye(D, dtype=np.complex128) * np.trace(r @ matrix)
            return value.reshape(D * D)

        def callback(_: object) -> None:
            nonlocal iterations
            iterations += 1

        operator = LinearOperator(
            (D * D, D * D), matvec=matvec, dtype=np.complex128
        )
        response, info = gmres(
            operator,
            fixed_point_weight.reshape(D * D),
            rtol=float(adjoint_rtol),
            atol=float(adjoint_atol),
            maxiter=adjoint_maximum_iterations,
            callback=callback,
            callback_type="pr_norm",
        )
        if info != 0:
            raise RuntimeError(
                f"adjoint fixed-point GMRES failed with info={info}"
            )
        response = response.reshape(D, D)
        response = 0.5 * (response + response.conj().T)
    else:
        response = response_solver.solve_adjoint(fixed_point_weight)
        iterations = 0
    for physical in range(d):
        gradient[physical] += 2.0 * response @ A[physical] @ r
    return gradient.astype(np.complex128), iterations


def _accurate_right_fixed_point_gradient(
    A: Array,
    r: Array,
    v: Array,
    *,
    rtol: float,
    atol: float,
    maximum_iterations: int | None,
) -> tuple[Array, int, int]:
    """Accurate historical adjoint response solve with diagnostics."""

    D = A.shape[0]

    def matvec(vector: Array) -> Array:
        matrix = vector.reshape(D, D)
        transfer = _ncon(
            [matrix, A, A.conj()],
            [[1, 2], [2, 3, -2], [1, 3, -1]],
        )
        fixed = np.trace(matrix @ r) * np.eye(D, dtype=np.complex128)
        return (matrix - transfer + fixed).reshape(D * D)

    operator = LinearOperator(
        (D * D, D * D), matvec=matvec, dtype=np.complex128
    )
    iteration_count = 0

    def callback(_: object) -> None:
        nonlocal iteration_count
        iteration_count += 1

    solution, info = gmres(
        operator,
        np.asarray(v, dtype=np.complex128).reshape(D * D),
        rtol=float(rtol),
        atol=float(atol),
        maxiter=maximum_iterations,
        callback=callback,
        callback_type="pr_norm",
    )
    if info != 0:
        raise RuntimeError(f"adjoint fixed-point GMRES failed with info={info}")
    Rh = solution.reshape(D, D)
    contribution = _ncon([Rh, A, r], [[-1, 1], [1, -2, 2], [2, -3]])
    return contribution, iteration_count, int(info)


class _AccurateFixedTargetHSCost(FixedTargetHSCost):
    """Historical analytic contraction with controlled adjoint solves."""

    def __init__(
        self,
        rho_target: Array,
        *,
        local_dimension: int,
        block_length: int,
        adjoint_rtol: float,
        adjoint_atol: float,
        adjoint_maximum_iterations: int | None,
    ):
        super().__init__(
            rho_target,
            local_dimension=local_dimension,
            block_length=block_length,
        )
        self.adjoint_rtol = float(adjoint_rtol)
        self.adjoint_atol = float(adjoint_atol)
        self.adjoint_maximum_iterations = adjoint_maximum_iterations
        self.last_adjoint_iterations = (0, 0)

    def derivative(
        self,
        A: Array,
        r: Array,
        rho: Array | None = None,
    ) -> Array:
        if rho is None:
            rho = self._build_rho(A, r)
        residual = rho - self.rho_target
        return self.derivative_from_residual(A, r, residual)

    def derivative_from_residual(
        self,
        A: Array,
        r: Array,
        residual: Array,
    ) -> Array:
        """Apply the analytic RDM adjoint to an arbitrary residual tensor."""

        L = self.block_length
        expected_shape = (self.local_dimension,) * (2 * L)
        residual = np.asarray(residual, dtype=np.complex128)
        if residual.shape != expected_shape:
            raise ValueError(
                f"residual must have shape {expected_shape}, got "
                f"{residual.shape}"
            )
        gradient = np.zeros_like(A, dtype=np.complex128)
        for site in range(L):
            drho = self._drho(site, A, r)
            gradient += 2.0 * _ncon(
                [residual, drho],
                [
                    list(range(1, 2 * L + 1)),
                    list(range(L + 1, 2 * L + 1))
                    + list(range(1, L + 1))
                    + [-1, -2, -3],
                ],
            )

        rho_open = self._rho_open(A)
        response, response_iterations, _ = _accurate_right_fixed_point_gradient(
            A,
            r,
            self._contract_rho_open_with_rho(rho_open, residual).T,
            rtol=self.adjoint_rtol,
            atol=self.adjoint_atol,
            maximum_iterations=self.adjoint_maximum_iterations,
        )
        self.last_adjoint_iterations = (response_iterations,)
        gradient += 2.0 * response
        return gradient


@dataclass(frozen=True)
class DirectGradientTimings:
    fixed_point_seconds: float
    rho_seconds: float
    gradient_seconds: float
    projection_seconds: float
    total_seconds: float


@dataclass(frozen=True)
class DirectGradientEvaluation:
    W: Array
    A: Array
    r: Array
    rho: Array
    residual: Array
    cost: float
    residual_norm: float
    ambient_gradient: Array
    gradient: Array
    gradient_norm: float
    projector: (
        GaugeHorizontalProjector
        | GrassmannHorizontalProjector
        | StructuredGaugeHorizontalProjector
    )
    response_solver: DenseFixedPointResponseSolver | None
    block_product_levels: tuple[Array, ...]
    gauge_rank: int
    gauge_overlap_norm: float
    tangency_error: float
    fixed_point_info: dict[str, float | complex]
    adjoint_gmres_iterations: tuple[int, int]
    timings: DirectGradientTimings


class FixedTargetDirectGradient:
    """Cached-target analytic cost/gradient oracle for horizontal CG."""

    def __init__(
        self,
        rho_target: Array,
        *,
        local_dimension: int,
        block_length: int,
        gauge_tolerance: float = 1e-10,
        fixed_point_solver: FixedPointSolver = "dense",
        adjoint_rtol: float = 1e-12,
        adjoint_atol: float = 0.0,
        adjoint_maximum_iterations: int | None = None,
        gauge_projector: GaugeProjector = "dense",
        tangent_slice: TangentSlice = "gauge_orthogonal",
        gradient_projection: GradientProjection = "exact",
        structured_gauge_maximum_iterations: int | None = 800,
        fixed_point_response_solver: Literal["iterative", "dense_lu"] = "iterative",
    ):
        self.local_dimension = int(local_dimension)
        self.block_length = int(block_length)
        self.gauge_tolerance = float(gauge_tolerance)
        self.fixed_point_solver = fixed_point_solver
        self.adjoint_rtol = float(adjoint_rtol)
        self.adjoint_atol = float(adjoint_atol)
        self.adjoint_maximum_iterations = adjoint_maximum_iterations
        self.gauge_projector = gauge_projector
        self.tangent_slice = tangent_slice
        self.gradient_projection = gradient_projection
        self.structured_gauge_maximum_iterations = (
            structured_gauge_maximum_iterations
        )
        self.fixed_point_response_solver = fixed_point_response_solver
        if self.local_dimension < 1 or self.block_length < 1:
            raise ValueError("local_dimension and block_length must be positive")
        if self.gauge_tolerance < 0:
            raise ValueError("gauge_tolerance must be non-negative")
        if self.adjoint_rtol <= 0 or self.adjoint_atol < 0:
            raise ValueError("adjoint tolerances must be non-negative and rtol > 0")
        if fixed_point_solver not in ("dense", "fast"):
            raise ValueError("fixed_point_solver must be 'dense' or 'fast'")
        if gauge_projector not in ("dense", "structured"):
            raise ValueError("gauge_projector must be 'dense' or 'structured'")
        if tangent_slice not in ("gauge_orthogonal", "grassmann"):
            raise ValueError(
                "tangent_slice must be 'gauge_orthogonal' or 'grassmann'"
            )
        if gradient_projection not in ("exact", "stiefel"):
            raise ValueError(
                "gradient_projection must be 'exact' or 'stiefel'"
            )
        if fixed_point_response_solver not in ("iterative", "dense_lu"):
            raise ValueError(
                "fixed_point_response_solver must be 'iterative' or "
                "'dense_lu'"
            )
        dimension = self.local_dimension**self.block_length
        target = np.asarray(rho_target, dtype=np.complex128)
        if target.shape != (dimension, dimension):
            raise ValueError(
                f"rho_target must have shape {(dimension, dimension)}, "
                f"got {target.shape}"
            )
        self.rho_target = target.copy()
        self._old_model = _AccurateFixedTargetHSCost(
            self.rho_target,
            local_dimension=self.local_dimension,
            block_length=self.block_length,
            adjoint_rtol=self.adjoint_rtol,
            adjoint_atol=self.adjoint_atol,
            adjoint_maximum_iterations=self.adjoint_maximum_iterations,
        )
        self.evaluation_count = 0

    def evaluate(
        self,
        W: Array,
        right_fixed_point_hint: Array | None = None,
    ) -> DirectGradientEvaluation:
        """Return the half-squared cost and its gauge-horizontal gradient."""

        total_started = time.perf_counter()
        W = np.asarray(W, dtype=np.complex128)
        D = W.shape[1]
        expected_shape = (self.local_dimension * D, D)
        if W.shape != expected_shape:
            raise ValueError(f"W must have shape {expected_shape}, got {W.shape}")
        A = unstack_tensor(W, self.local_dimension, D)
        old_A = _to_old_tensor(A)

        started = time.perf_counter()
        if right_fixed_point_hint is None:
            r, fixed_point_info = optimizer_right_fixed_point(
                A, self.fixed_point_solver
            )
        else:
            r = np.asarray(right_fixed_point_hint, dtype=np.complex128)
            if r.shape != (D, D):
                raise ValueError(
                    "right_fixed_point_hint must have shape "
                    f"{(D, D)}, got {r.shape}"
                )
            fixed_point_residual = (
                sum(Ai @ r @ Ai.conj().T for Ai in A) - r
            )
            fixed_point_info = {
                "provided_hint": 1.0,
                "residual": float(la.norm(fixed_point_residual)),
                "trace": complex(np.trace(r)),
                "hermiticity_error": float(la.norm(r - r.conj().T)),
                "min_eigenvalue": float(
                    np.min(la.eigvalsh(0.5 * (r + r.conj().T))).real
                ),
            }
        response_solver = (
            DenseFixedPointResponseSolver.from_tensor(A, r)
            if self.fixed_point_response_solver == "dense_lu"
            else None
        )
        fixed_point_seconds = time.perf_counter() - started

        started = time.perf_counter()
        rho_tensor = self._old_model._build_rho(old_A, r)
        dimension = self.local_dimension**self.block_length
        rho = np.asarray(rho_tensor, dtype=np.complex128).reshape(
            dimension, dimension
        )
        residual = rho - self.rho_target
        residual_norm = float(la.norm(residual))
        # Computing the norm from the residual is stable close to a solution;
        # the expanded trace expression suffers cancellation there.
        cost = 0.5 * residual_norm**2
        rho_seconds = time.perf_counter() - started

        started = time.perf_counter()
        product_levels = block_product_levels(A, self.block_length)
        # ``FixedTargetHSCost.derivative`` is the Wirtinger derivative of the
        # full squared norm.  Under the real Frobenius metric it is therefore
        # exactly the gradient of the half-squared norm used here.
        ambient_tensor, adjoint_iterations = _efficient_ambient_rdm_vjp(
            A,
            r,
            residual,
            self.block_length,
            adjoint_rtol=self.adjoint_rtol,
            adjoint_atol=self.adjoint_atol,
            adjoint_maximum_iterations=self.adjoint_maximum_iterations,
            response_solver=response_solver,
            product_levels=product_levels,
        )
        ambient_gradient = stack_tensor(ambient_tensor)
        gradient_seconds = time.perf_counter() - started

        started = time.perf_counter()
        if self.tangent_slice == "grassmann":
            projector = GrassmannHorizontalProjector.from_tensor(
                A, W, tolerance=self.gauge_tolerance
            )
        elif self.gauge_projector == "structured":
            projector = StructuredGaugeHorizontalProjector(
                A,
                W,
                rtol=self.gauge_tolerance,
                maximum_iterations=self.structured_gauge_maximum_iterations,
            )
        else:
            projector = GaugeHorizontalProjector.from_tensor(
                A, W, tolerance=self.gauge_tolerance
            )
        if self.gradient_projection == "exact":
            gradient = projector.project(ambient_gradient)
        else:
            # RDM gauge invariance implies that the exact VJP is already
            # orthogonal to every gauge tangent.  This path removes only the
            # Stiefel normal component and avoids an unnecessary structured
            # gauge solve.  Consumers that take a step can still exactly
            # project the completed direction.
            gradient = stiefel_project(W, ambient_gradient)
        projection_seconds = time.perf_counter() - started
        gradient_norm = float(la.norm(gradient))
        self.evaluation_count += 1

        return DirectGradientEvaluation(
            W=W.copy(),
            A=A,
            r=r,
            rho=rho,
            residual=residual,
            cost=cost,
            residual_norm=residual_norm,
            ambient_gradient=ambient_gradient,
            gradient=gradient,
            gradient_norm=gradient_norm,
            projector=projector,
            response_solver=response_solver,
            block_product_levels=product_levels,
            gauge_rank=projector.gauge_rank,
            gauge_overlap_norm=projector.gauge_overlap_norm(gradient),
            tangency_error=projector.tangency_error(gradient),
            fixed_point_info=dict(fixed_point_info),
            adjoint_gmres_iterations=(adjoint_iterations,),
            timings=DirectGradientTimings(
                fixed_point_seconds=fixed_point_seconds,
                rho_seconds=rho_seconds,
                gradient_seconds=gradient_seconds,
                projection_seconds=projection_seconds,
                total_seconds=time.perf_counter() - total_started,
            ),
        )

    def ambient_vjp(
        self,
        evaluation: DirectGradientEvaluation,
        weight: Array,
    ) -> Array:
        """Return ``J^dagger weight`` without rebuilding a Jacobian or basis.

        The supplied evaluation fixes the tensor and its right fixed point.
        ``weight`` is a matrix in the same finite-window convention as
        ``evaluation.rho``.  The returned matrix has the stacked tensor shape
        and is not projected; callers can choose the appropriate quotient
        representative with ``evaluation.projector``.
        """

        weight = np.asarray(weight, dtype=np.complex128)
        if weight.shape != evaluation.rho.shape:
            raise ValueError(
                f"weight must have shape {evaluation.rho.shape}, got "
                f"{weight.shape}"
            )
        gradient, _ = _efficient_ambient_rdm_vjp(
            evaluation.A,
            evaluation.r,
            weight,
            self.block_length,
            adjoint_rtol=self.adjoint_rtol,
            adjoint_atol=self.adjoint_atol,
            adjoint_maximum_iterations=self.adjoint_maximum_iterations,
            response_solver=evaluation.response_solver,
            product_levels=evaluation.block_product_levels,
        )
        return stack_tensor(gradient)

    def horizontal_vjp(
        self,
        evaluation: DirectGradientEvaluation,
        weight: Array,
    ) -> Array:
        """Return the true-gauge-horizontal ``J^dagger weight``."""

        return evaluation.projector.project(
            self.ambient_vjp(evaluation, weight)
        )
