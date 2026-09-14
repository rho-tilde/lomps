"""Minimum-purity selection inside a finite-window LOMPS fibre.

The primary LOMPS constraint is kept explicit::

    C_L(A) = 1/2 ||rho_L(A) - rho_target||_F^2 <= epsilon.

Among feasible tensors, this module lowers the purity of the buffered block
``rho_{L+B}``.  Its gradient is evaluated by the finite-window RDM adjoint;
the full ``rho_{L+B}`` Jacobian is never constructed.  The ordinary, much
smaller ``rho_L`` Jacobian is used to remove all first-order visible motion,
and every trial point is checked against the nonlinear primary constraint.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Literal

import numpy as np
import scipy.linalg as la

from .canonical import polar_retraction, stack_tensor, unstack_tensor
from .fixed_target_vjp import FixedTargetDirectGradient
from .optimizer import GaugeOrthogonalLM, LMOptions, optimizer_right_fixed_point
from .rdm import block_rdm
from .tangent import TangentSlice


Array = np.ndarray
FixedPointSolver = Literal["dense", "fast"]
JacobianResponseSolver = Literal["lstsq", "dense_lu"]
PurityDirectionScaling = Literal["unit", "gradient"]


def density_matrix_purity(rho: Array) -> float:
    """Return ``Tr(rho**2)`` for a Hermitian density matrix."""

    rho = np.asarray(rho, dtype=np.complex128)
    if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
        raise ValueError("rho must be a square matrix")
    return float(np.vdot(rho, rho).real)


@dataclass(frozen=True)
class BufferedPurityGradientEvaluation:
    """Buffered purity and its horizontal gradient at one MPS tensor."""

    W: Array
    A: Array
    rho_buffered: Array
    purity: float
    gradient: Array
    gradient_norm: float
    extended_block_length: int
    elapsed_seconds: float


class BufferedPurityGradient:
    """Adjoint-gradient oracle for ``Tr(rho_{L+B}(A)**2)``.

    ``FixedTargetDirectGradient`` evaluates ``J^dagger weight`` directly by
    prefix/suffix contractions and one adjoint fixed-point response solve.
    Supplying the zero matrix as its formal target makes its half-squared RDM
    objective equal to one half of the purity.  Doubling that gradient gives
    the gradient of the physical purity itself, without building ``J_{L+B}``.
    """

    def __init__(
        self,
        block_length: int,
        *,
        buffer_sites: int = 4,
        local_dimension: int = 2,
        fixed_point_solver: FixedPointSolver = "dense",
        rank_tolerance: float = 1e-10,
        tangent_slice: TangentSlice = "grassmann",
        fixed_point_response_solver: Literal[
            "iterative", "dense_lu"
        ] = "dense_lu",
        adjoint_rtol: float = 1e-12,
    ):
        self.block_length = int(block_length)
        self.buffer_sites = int(buffer_sites)
        self.extended_block_length = self.block_length + self.buffer_sites
        self.local_dimension = int(local_dimension)
        if self.block_length < 1:
            raise ValueError("block_length must be positive")
        if self.buffer_sites < 1:
            raise ValueError("buffer_sites must be positive")
        dimension = self.local_dimension**self.extended_block_length
        self._oracle = FixedTargetDirectGradient(
            np.zeros((dimension, dimension), dtype=np.complex128),
            local_dimension=self.local_dimension,
            block_length=self.extended_block_length,
            gauge_tolerance=rank_tolerance,
            fixed_point_solver=fixed_point_solver,
            adjoint_rtol=adjoint_rtol,
            tangent_slice=tangent_slice,
            fixed_point_response_solver=fixed_point_response_solver,
        )

    def evaluate(self, W: Array) -> BufferedPurityGradientEvaluation:
        """Return buffered purity and its horizontal adjoint gradient."""

        started = time.perf_counter()
        evaluation = self._oracle.evaluate(W)
        purity = density_matrix_purity(evaluation.rho)
        # The wrapped objective is 1/2 ||rho||_F^2 = purity / 2.
        gradient = 2.0 * evaluation.gradient
        return BufferedPurityGradientEvaluation(
            W=evaluation.W,
            A=evaluation.A,
            rho_buffered=evaluation.rho,
            purity=purity,
            gradient=gradient,
            gradient_norm=float(la.norm(gradient)),
            extended_block_length=self.extended_block_length,
            elapsed_seconds=time.perf_counter() - started,
        )


@dataclass(frozen=True)
class BufferedPurityOptions:
    """Controls for exact-constraint, projected buffered-purity descent."""

    buffer_sites: int = 4
    primary_cost_tolerance: float = 1e-12
    max_iterations: int = 8
    initial_step: float = 1e-2
    direction_scaling: PurityDirectionScaling = "unit"
    minimum_step: float = 1e-8
    armijo_c1: float = 1e-4
    rank_tolerance: float = 1e-10
    null_gradient_tolerance: float = 1e-12
    relative_purity_tolerance: float = 1e-10
    relative_purity_patience: int = 1
    relative_purity_is_convergence: bool = True
    reproject_primary: bool = True
    projection_cost_tolerance: float | None = None
    projection_max_iterations: int = 6
    projection_initial_damping: float = 1e-10
    projection_maximum_seconds: float = 120.0
    tangent_slice: TangentSlice = "grassmann"
    fixed_point_solver: FixedPointSolver = "dense"
    jacobian_response_solver: JacobianResponseSolver = "dense_lu"
    jacobian_workers: int = 1
    adjoint_rtol: float = 1e-12
    verbose: bool = True


@dataclass(frozen=True)
class BufferedPurityRecord:
    iteration: int
    primary_cost: float
    purity: float
    purity_gradient_norm: float
    null_gradient_norm: float
    linearized_primary_change_norm: float
    visible_rank: int
    tangent_dimension: int
    trial_step: float
    accepted: bool
    candidate_primary_cost: float
    candidate_purity: float
    raw_candidate_primary_cost: float
    projection_status: str
    projection_evaluations: int
    elapsed_seconds: float


@dataclass(frozen=True)
class BufferedPurityResult:
    W: Array
    primary_cost: float
    purity: float
    initial_primary_cost: float
    initial_purity: float
    status: str
    accepted_steps: int
    final_null_gradient_norm: float
    final_relative_purity_drop: float
    history: tuple[BufferedPurityRecord, ...]


class BufferedPurityOptimizer:
    """Lower buffered purity while exactly guarding the LOMPS fit cost."""

    def __init__(
        self,
        block_length: int,
        options: BufferedPurityOptions | None = None,
    ):
        self.block_length = int(block_length)
        self.options = options or BufferedPurityOptions()
        if self.block_length < 1:
            raise ValueError("block_length must be positive")
        if self.options.buffer_sites < 1:
            raise ValueError("buffer_sites must be positive")
        if self.options.primary_cost_tolerance <= 0:
            raise ValueError("primary_cost_tolerance must be positive")
        if self.options.max_iterations < 0:
            raise ValueError("max_iterations must be non-negative")
        if self.options.initial_step <= 0 or self.options.minimum_step <= 0:
            raise ValueError("line-search steps must be positive")
        if self.options.minimum_step > self.options.initial_step:
            raise ValueError("minimum_step cannot exceed initial_step")
        if not 0 < self.options.armijo_c1 < 1:
            raise ValueError("armijo_c1 must lie strictly between zero and one")
        if self.options.direction_scaling not in {"unit", "gradient"}:
            raise ValueError("direction_scaling must be 'unit' or 'gradient'")
        if self.options.projection_max_iterations < 0:
            raise ValueError("projection_max_iterations must be non-negative")
        if self.options.relative_purity_tolerance < 0:
            raise ValueError("relative_purity_tolerance must be non-negative")
        if self.options.relative_purity_patience < 1:
            raise ValueError("relative_purity_patience must be positive")
        if (
            self.options.projection_cost_tolerance is not None
            and self.options.projection_cost_tolerance <= 0
        ):
            raise ValueError("projection_cost_tolerance must be positive")
        if (
            self.options.projection_cost_tolerance is not None
            and self.options.projection_cost_tolerance
            > self.options.primary_cost_tolerance
        ):
            raise ValueError(
                "projection_cost_tolerance cannot exceed "
                "primary_cost_tolerance"
            )
        if self.options.projection_initial_damping <= 0:
            raise ValueError("projection_initial_damping must be positive")
        if self.options.projection_maximum_seconds <= 0:
            raise ValueError("projection_maximum_seconds must be positive")

        primary_options = LMOptions(
            max_iterations=0,
            rank_tolerance=self.options.rank_tolerance,
            tangent_slice=self.options.tangent_slice,
            fixed_point_solver=self.options.fixed_point_solver,
            linear_solver="normal",
            rdm_vectorization="hermitian",
            jacobian_response_solver=self.options.jacobian_response_solver,
            jacobian_workers=self.options.jacobian_workers,
            verbose=False,
        )
        self._primary_oracle = GaugeOrthogonalLM(
            self.block_length, primary_options
        )
        self._purity_oracle = BufferedPurityGradient(
            self.block_length,
            buffer_sites=self.options.buffer_sites,
            fixed_point_solver=self.options.fixed_point_solver,
            rank_tolerance=self.options.rank_tolerance,
            tangent_slice=self.options.tangent_slice,
            fixed_point_response_solver="dense_lu",
            adjoint_rtol=self.options.adjoint_rtol,
        )

    def _reproject_primary(
        self,
        W: Array,
        rho_target: Array,
        known_primary_cost: float | None = None,
    ) -> tuple[Array, str, int]:
        """Return an LM-corrected point on the primary feasible set."""

        if not self.options.reproject_primary:
            return W, "disabled", 0
        projection_cost_tolerance = (
            self.options.primary_cost_tolerance
            if self.options.projection_cost_tolerance is None
            else self.options.projection_cost_tolerance
        )
        if (
            known_primary_cost is not None
            and np.isfinite(known_primary_cost)
            and known_primary_cost <= projection_cost_tolerance
        ):
            return W, "already_within_projection_tolerance", 0
        controls = LMOptions(
            max_iterations=self.options.projection_max_iterations,
            gradient_tolerance=0.0,
            cost_tolerance=projection_cost_tolerance,
            rank_tolerance=self.options.rank_tolerance,
            tangent_slice=self.options.tangent_slice,
            initial_damping=self.options.projection_initial_damping,
            maximum_seconds=self.options.projection_maximum_seconds,
            fixed_point_solver=self.options.fixed_point_solver,
            linear_solver="normal",
            rdm_vectorization="hermitian",
            jacobian_response_solver=self.options.jacobian_response_solver,
            jacobian_workers=self.options.jacobian_workers,
            verbose=False,
        )
        result = GaugeOrthogonalLM(self.block_length, controls).optimize(
            W, rho_target
        )
        return result.W, result.status, len(result.history)

    @staticmethod
    def _basis_coordinates(basis: Array, matrix: Array) -> Array:
        return np.asarray(
            [float(np.vdot(direction, matrix).real) for direction in basis],
            dtype=float,
        )

    def _null_project(
        self,
        jacobian: Array,
        coordinates: Array,
    ) -> tuple[Array, int]:
        """Remove the numerical row-space of ``jacobian`` from coordinates."""

        if jacobian.size == 0:
            return coordinates.copy(), 0
        _, singular_values, vh = la.svd(
            jacobian,
            full_matrices=False,
            check_finite=False,
            lapack_driver="gesdd",
        )
        if singular_values.size == 0:
            return coordinates.copy(), 0
        tolerance = max(
            self.options.rank_tolerance,
            max(jacobian.shape) * np.finfo(float).eps * singular_values[0],
        )
        rank = int(np.count_nonzero(singular_values > tolerance))
        row_basis = vh[:rank]
        projected = coordinates - row_basis.T @ (row_basis @ coordinates)
        return np.asarray(projected, dtype=float), rank

    def _trial_metrics(
        self,
        W: Array,
        rho_target: Array,
    ) -> tuple[float, float]:
        dD, D = W.shape
        A = unstack_tensor(W, dD // D, D)
        r, _ = optimizer_right_fixed_point(
            A, self.options.fixed_point_solver
        )
        residual = block_rdm(A, self.block_length, r) - rho_target
        primary_cost = 0.5 * float(np.vdot(residual, residual).real)
        rho_buffered = block_rdm(
            A, self.block_length + self.options.buffer_sites, r
        )
        return primary_cost, density_matrix_purity(rho_buffered)

    def _trial_primary_cost(self, W: Array, rho_target: Array) -> float:
        """Return only the exact dense primary cost for a trial point."""

        dD, D = W.shape
        A = unstack_tensor(W, dD // D, D)
        r, _ = optimizer_right_fixed_point(
            A, self.options.fixed_point_solver
        )
        residual = block_rdm(A, self.block_length, r) - rho_target
        return 0.5 * float(np.vdot(residual, residual).real)

    def optimize(self, W0: Array, rho_target: Array) -> BufferedPurityResult:
        """Run projected purity descent from an already feasible LOMPS fit."""

        W = np.asarray(W0, dtype=np.complex128).copy()
        rho_target = np.asarray(rho_target, dtype=np.complex128)
        history: list[BufferedPurityRecord] = []
        initial_primary_cost = np.nan
        initial_purity = np.nan
        final_primary_cost = np.nan
        final_purity = np.nan
        final_null_gradient_norm = np.nan
        final_relative_purity_drop = np.nan
        accepted_steps = 0
        small_relative_drop_streak = 0
        status = "maximum_iterations"

        for iteration in range(self.options.max_iterations + 1):
            started = time.perf_counter()
            primary = self._primary_oracle.evaluate(W, rho_target)
            purity = self._purity_oracle.evaluate(W)
            final_primary_cost = primary.cost
            final_purity = purity.purity
            if iteration == 0:
                initial_primary_cost = primary.cost
                initial_purity = purity.purity
                feasibility_slack = max(
                    1e-30, 1e-10 * self.options.primary_cost_tolerance
                )
                if primary.cost > (
                    self.options.primary_cost_tolerance + feasibility_slack
                ):
                    status = "initial_point_infeasible"
                    break
            gradient_coordinates = self._basis_coordinates(
                primary.basis, purity.gradient
            )
            null_gradient, visible_rank = self._null_project(
                primary.jacobian, gradient_coordinates
            )
            null_gradient_norm = float(la.norm(null_gradient))
            final_null_gradient_norm = null_gradient_norm
            linearized_change = float(
                la.norm(primary.jacobian @ null_gradient)
            )
            if null_gradient_norm <= self.options.null_gradient_tolerance:
                status = "null_gradient_tolerance"
                break
            if (
                small_relative_drop_streak
                >= self.options.relative_purity_patience
            ):
                status = (
                    "relative_purity_tolerance"
                    if self.options.relative_purity_is_convergence
                    else "relative_purity_stall"
                )
                break
            if iteration == self.options.max_iterations:
                status = "maximum_iterations"
                break

            if self.options.direction_scaling == "unit":
                direction_coefficients = -null_gradient / null_gradient_norm
                slope = -null_gradient_norm
            else:
                direction_coefficients = -null_gradient
                slope = -(null_gradient_norm**2)
            direction = np.tensordot(
                direction_coefficients, primary.basis, axes=(0, 0)
            )
            alpha = self.options.initial_step
            accepted = False
            candidate_primary_cost = primary.cost
            candidate_purity = purity.purity
            raw_candidate_primary_cost = primary.cost
            projection_status = "not_run"
            projection_evaluations = 0
            candidate = W
            while alpha >= self.options.minimum_step:
                trial = polar_retraction(W + alpha * direction)
                raw_trial_cost = self._trial_primary_cost(trial, rho_target)
                trial, trial_projection_status, trial_projection_evaluations = (
                    self._reproject_primary(
                        trial,
                        rho_target,
                        known_primary_cost=raw_trial_cost,
                    )
                )
                trial_cost, trial_purity = self._trial_metrics(trial, rho_target)
                feasible = (
                    np.isfinite(trial_cost)
                    and trial_cost <= self.options.primary_cost_tolerance
                )
                sufficient_decrease = (
                    np.isfinite(trial_purity)
                    and trial_purity
                    <= purity.purity
                    + self.options.armijo_c1 * alpha * slope
                )
                if feasible and sufficient_decrease:
                    accepted = True
                    candidate = trial
                    candidate_primary_cost = trial_cost
                    candidate_purity = trial_purity
                    raw_candidate_primary_cost = raw_trial_cost
                    projection_status = trial_projection_status
                    projection_evaluations = trial_projection_evaluations
                    break
                alpha *= 0.5

            history.append(
                BufferedPurityRecord(
                    iteration=iteration,
                    primary_cost=primary.cost,
                    purity=purity.purity,
                    purity_gradient_norm=purity.gradient_norm,
                    null_gradient_norm=null_gradient_norm,
                    linearized_primary_change_norm=linearized_change,
                    visible_rank=visible_rank,
                    tangent_dimension=primary.jacobian.shape[1],
                    trial_step=alpha if accepted else 0.0,
                    accepted=accepted,
                    candidate_primary_cost=candidate_primary_cost,
                    candidate_purity=candidate_purity,
                    raw_candidate_primary_cost=raw_candidate_primary_cost,
                    projection_status=projection_status,
                    projection_evaluations=projection_evaluations,
                    elapsed_seconds=time.perf_counter() - started,
                )
            )
            if self.options.verbose:
                print(
                    "purity "
                    f"iter={iteration:3d} C_L={primary.cost:.6e} "
                    f"P={purity.purity:.12e} "
                    f"|g_null|={null_gradient_norm:.3e} "
                    f"rank={visible_rank}/{primary.jacobian.shape[1]} "
                    f"alpha={alpha if accepted else 0.0:.3e} "
                    f"accepted={accepted}",
                    flush=True,
                )
            if not accepted:
                status = "line_search_failed"
                break

            relative_drop = (purity.purity - candidate_purity) / max(
                abs(purity.purity), 1e-300
            )
            final_relative_purity_drop = relative_drop
            if relative_drop <= self.options.relative_purity_tolerance:
                small_relative_drop_streak += 1
            else:
                small_relative_drop_streak = 0
            W = candidate
            final_primary_cost = candidate_primary_cost
            final_purity = candidate_purity
            accepted_steps += 1

        return BufferedPurityResult(
            W=W,
            primary_cost=float(final_primary_cost),
            purity=float(final_purity),
            initial_primary_cost=float(initial_primary_cost),
            initial_purity=float(initial_purity),
            status=status,
            accepted_steps=accepted_steps,
            final_null_gradient_norm=float(final_null_gradient_norm),
            final_relative_purity_drop=float(final_relative_purity_drop),
            history=tuple(history),
        )


def minimize_buffered_purity(
    A0: Array,
    rho_target: Array,
    block_length: int,
    options: BufferedPurityOptions | None = None,
) -> tuple[Array, BufferedPurityResult]:
    """Convenience wrapper returning a physical-first MPS tensor."""

    A0 = np.asarray(A0, dtype=np.complex128)
    result = BufferedPurityOptimizer(block_length, options).optimize(
        stack_tensor(A0), rho_target
    )
    d, D, _ = A0.shape
    return unstack_tensor(result.W, d, D), result
