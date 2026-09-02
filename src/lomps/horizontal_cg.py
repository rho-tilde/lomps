"""Nonlinear CG on the true-gauge quotient of left-canonical uMPS tensors.

Only the infinite-uMPS gauge tangent is removed.  In particular, directions
that are invisible to one chosen finite RDM window remain part of the search
space.  The objective and analytic gradient come from
:class:`lomps.fixed_target_vjp.FixedTargetDirectGradient`, so this optimizer
does not construct an RDM Jacobian.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Literal

import numpy as np
import scipy.linalg as la

from .canonical import unstack_tensor
from .fixed_target_vjp import (
    DirectGradientEvaluation,
    FixedTargetDirectGradient,
    GaugeProjector,
)
from .horizontal import GaugeHorizontalProjector, GrassmannHorizontalProjector
from .structured_horizontal import StructuredGaugeHorizontalProjector
from .tangent import TangentSlice


Array = np.ndarray
FixedPointSolver = Literal["dense", "fast"]
CGPreconditioner = Literal[
    "none",
    "right_fixed_point",
    "grassmann_slice",
    "grassmann_right_fixed_point",
]
CGConjugacyMetric = Literal["horizontal", "historical_slice"]
CGLineInterpolation = Literal["bisection", "secant"]


def _metric(X: Array, Y: Array) -> float:
    return float(np.vdot(X, Y).real)


def polar_retraction_curve(
    W: Array,
    direction: Array,
    alpha: float,
) -> tuple[Array, Array]:
    """Polar retraction and exact velocity along ``R_W(alpha direction)``.

    For a Stiefel tangent ``direction``, the Gram matrix on this curve is
    ``I + alpha**2 direction^dagger direction``.  Diagonalizing the latter
    gives both the polar factor and its derivative without differentiating a
    generic matrix square root.
    """

    W = np.asarray(W, dtype=np.complex128)
    direction = np.asarray(direction, dtype=np.complex128)
    if W.shape != direction.shape:
        raise ValueError("W and direction must have the same shape")
    alpha = float(alpha)
    gram_direction = direction.conj().T @ direction
    gram_direction = 0.5 * (gram_direction + gram_direction.conj().T)
    eigenvalues, eigenvectors = la.eigh(gram_direction, check_finite=False)
    eigenvalues = np.clip(eigenvalues.real, 0.0, None)
    scale = 1.0 + alpha * alpha * eigenvalues
    inverse_sqrt = (
        eigenvectors
        @ np.diag(scale ** -0.5)
        @ eigenvectors.conj().T
    )
    k_inverse_three_halves = (
        eigenvectors
        @ np.diag(eigenvalues * scale ** -1.5)
        @ eigenvectors.conj().T
    )
    affine = W + alpha * direction
    retracted = affine @ inverse_sqrt
    velocity = direction @ inverse_sqrt - alpha * affine @ k_inverse_three_halves
    return retracted.astype(np.complex128), velocity.astype(np.complex128)


@dataclass(frozen=True)
class HorizontalCGOptions:
    max_iterations: int = 40_000
    residual_tolerance: float = 0.0
    cost_tolerance: float = 3e-16
    gradient_tolerance: float = 1e-11
    gauge_tolerance: float = 1e-10
    tangent_slice: TangentSlice = "gauge_orthogonal"
    initial_step: float | None = None
    maximum_step: float = 1.0
    minimum_step: float = 1e-12
    line_search_growth: float = 2.0
    maximum_line_search_evaluations: int = 20
    line_search_interpolation: CGLineInterpolation = "secant"
    armijo_c1: float = 1e-4
    wolfe_c2: float = 0.9
    accept_armijo_fallback: bool = True
    preconditioner: CGPreconditioner = "grassmann_right_fixed_point"
    conjugacy_metric: CGConjugacyMetric = "historical_slice"
    minimum_preconditioner_shift: float = 1e-14
    adjoint_rtol: float = 1e-8
    gauge_projector: GaugeProjector = "dense"
    structured_gauge_maximum_iterations: int | None = 800
    restart: int = 100
    maximum_seconds: float = 900.0
    fixed_point_solver: FixedPointSolver = "fast"
    verbose: bool = True


@dataclass(frozen=True)
class HorizontalCGRecord:
    iteration: int
    cost: float
    residual_norm: float
    gradient_norm: float
    step_size: float
    slope: float
    beta: float
    line_search_evaluations: int
    accepted: bool
    wolfe_satisfied: bool
    restarted: bool
    elapsed_seconds: float


@dataclass(frozen=True)
class HorizontalCGResult:
    W: Array
    cost: float
    residual_norm: float
    gradient_norm: float
    status: str
    history: tuple[HorizontalCGRecord, ...]
    objective_evaluations: int
    accepted_steps: int
    restarts: int
    elapsed_seconds: float


@dataclass(frozen=True)
class _LinePoint:
    alpha: float
    W: Array
    velocity: Array
    evaluation: DirectGradientEvaluation
    slope: float


@dataclass(frozen=True)
class _LineSearchResult:
    point: _LinePoint | None
    evaluations: int
    wolfe_satisfied: bool


class HorizontalNonlinearCG:
    """Gauge-rectified Hager--Zhang nonlinear conjugate gradient."""

    theta = 1.0
    eta = 0.4

    def __init__(
        self,
        block_length: int,
        options: HorizontalCGOptions | None = None,
    ):
        self.block_length = int(block_length)
        self.options = options or HorizontalCGOptions()
        if self.block_length < 1:
            raise ValueError("block_length must be positive")
        if self.options.fixed_point_solver not in ("dense", "fast"):
            raise ValueError("fixed_point_solver must be 'dense' or 'fast'")
        if self.options.tangent_slice not in (
            "gauge_orthogonal",
            "grassmann",
        ):
            raise ValueError(
                "tangent_slice must be 'gauge_orthogonal' or 'grassmann'"
            )
        if self.options.preconditioner not in (
            "none",
            "right_fixed_point",
            "grassmann_slice",
            "grassmann_right_fixed_point",
        ):
            raise ValueError(
                "preconditioner must be 'none', 'right_fixed_point', or "
                "'grassmann_slice', or 'grassmann_right_fixed_point'"
            )
        if self.options.conjugacy_metric not in (
            "horizontal",
            "historical_slice",
        ):
            raise ValueError(
                "conjugacy_metric must be 'horizontal' or 'historical_slice'"
            )
        if (
            self.options.conjugacy_metric == "historical_slice"
            and self.options.preconditioner
            != "grassmann_right_fixed_point"
        ):
            raise ValueError(
                "historical_slice conjugacy requires the "
                "grassmann_right_fixed_point preconditioner"
            )
        if not (0.0 < self.options.armijo_c1 < self.options.wolfe_c2 < 1.0):
            raise ValueError("require 0 < armijo_c1 < wolfe_c2 < 1")
        if self.options.restart < 1:
            raise ValueError("restart must be positive")
        if self.options.maximum_line_search_evaluations < 1:
            raise ValueError("maximum_line_search_evaluations must be positive")
        if self.options.adjoint_rtol <= 0:
            raise ValueError("adjoint_rtol must be positive")
        if self.options.gauge_projector not in ("dense", "structured"):
            raise ValueError("gauge_projector must be 'dense' or 'structured'")
        if self.options.line_search_interpolation not in (
            "bisection",
            "secant",
        ):
            raise ValueError(
                "line_search_interpolation must be 'bisection' or 'secant'"
            )

    def _projector(
        self, evaluation: DirectGradientEvaluation
    ) -> (
        GaugeHorizontalProjector
        | GrassmannHorizontalProjector
        | StructuredGaugeHorizontalProjector
    ):
        return evaluation.projector

    def _precondition(
        self,
        evaluation: DirectGradientEvaluation,
        projector: (
            GaugeHorizontalProjector
            | GrassmannHorizontalProjector
            | StructuredGaugeHorizontalProjector
        ),
    ) -> Array:
        if self.options.preconditioner == "none":
            return evaluation.gradient
        if self.options.preconditioner in (
            "grassmann_slice",
            "grassmann_right_fixed_point",
        ):
            # The historical W^dagger X = 0 slice was an oblique complement
            # to the true MPS gauge orbit.  Mapping its representative back
            # into the orthogonal true-horizontal slice defines a cheap
            # quotient-space preconditioner.  The returned direction is still
            # genuinely gauge horizontal; no finite-window information is
            # used here.
            old_slice = evaluation.ambient_gradient - evaluation.W @ (
                evaluation.W.conj().T @ evaluation.ambient_gradient
            )
            if self.options.preconditioner == "grassmann_slice":
                return projector.project(old_slice)

            # Reproduce the useful order of operations in the historical CG:
            # first choose its W^dagger X = 0 representative, then apply the
            # right-fixed-point preconditioner.  Only after that do we choose
            # the orthogonal representative of the same quotient direction.
            # Applying r^{-1} to the orthogonal-horizontal gradient directly
            # is not equivalent and strongly amplifies poorly occupied bond
            # directions for the production D=12 tensor.
            shift = max(
                float(la.norm(old_slice) ** 2),
                self.options.minimum_preconditioner_shift,
            )
            matrix = evaluation.r + shift * np.eye(
                evaluation.r.shape[0], dtype=np.complex128
            )
            try:
                value = la.solve(
                    matrix.T,
                    old_slice.T,
                    assume_a="gen",
                    check_finite=False,
                ).T
            except la.LinAlgError:
                value = old_slice @ la.pinv(matrix, check_finite=False)
            return projector.project(value)
        shift = max(
            evaluation.gradient_norm**2,
            self.options.minimum_preconditioner_shift,
        )
        matrix = evaluation.r + shift * np.eye(
            evaluation.r.shape[0], dtype=np.complex128
        )
        try:
            value = la.solve(
                matrix.T,
                evaluation.gradient.T,
                assume_a="gen",
                check_finite=False,
            ).T
        except la.LinAlgError:
            value = evaluation.gradient @ la.pinv(matrix, check_finite=False)
        # Right preconditioning need not preserve either the Stiefel tangent
        # or its true-gauge-horizontal subspace.
        return projector.project(value)

    @staticmethod
    def _historical_slice_gradient(
        evaluation: DirectGradientEvaluation,
    ) -> Array:
        return evaluation.ambient_gradient - evaluation.W @ (
            evaluation.W.conj().T @ evaluation.ambient_gradient
        )

    def _historical_slice_precondition(
        self,
        evaluation: DirectGradientEvaluation,
        old_slice: Array,
    ) -> Array:
        shift = max(
            float(la.norm(old_slice) ** 2),
            self.options.minimum_preconditioner_shift,
        )
        matrix = evaluation.r + shift * np.eye(
            evaluation.r.shape[0], dtype=np.complex128
        )
        try:
            return la.solve(
                matrix.T,
                old_slice.T,
                assume_a="gen",
                check_finite=False,
            ).T
        except la.LinAlgError:
            return old_slice @ la.pinv(matrix, check_finite=False)

    @staticmethod
    def _historical_slice_transport(W: Array, X: Array) -> Array:
        """Projection transport into the historical ``W^dagger X=0`` slice."""

        return X - W @ (W.conj().T @ X)

    def _hager_zhang(
        self,
        gradient: Array,
        previous_gradient: Array,
        preconditioned: Array,
        previous_preconditioned: Array,
        previous_direction: Array,
    ) -> float:
        dd = _metric(previous_direction, previous_direction)
        dg = _metric(previous_direction, gradient)
        dg_previous = _metric(previous_direction, previous_gradient)
        dy = dg - dg_previous
        if abs(dy) < 1e-300 or abs(dd) < 1e-300:
            return 0.0
        g_pg = _metric(gradient, preconditioned)
        gp_pgp = _metric(previous_gradient, previous_preconditioned)
        g_pgp = _metric(gradient, previous_preconditioned)
        gp_pg = _metric(previous_gradient, preconditioned)
        g_py = g_pg - g_pgp
        y_py = g_pg + gp_pgp - g_pgp - gp_pg
        beta = (g_py - self.theta * (y_py / dy) * dg) / dy
        lower_bound = self.eta * dg_previous / dd
        return float(max(beta, lower_bound))

    def _line_search(
        self,
        oracle: FixedTargetDirectGradient,
        origin: DirectGradientEvaluation,
        direction: Array,
        alpha0: float,
    ) -> _LineSearchResult:
        slope0 = _metric(origin.gradient, direction)
        if not np.isfinite(slope0) or slope0 >= 0.0:
            return _LineSearchResult(None, 0, False)

        maximum = self.options.maximum_line_search_evaluations
        evaluations = 0
        cache: dict[float, _LinePoint] = {}
        best_armijo: _LinePoint | None = None

        def evaluate(alpha: float) -> _LinePoint:
            nonlocal evaluations, best_armijo
            alpha = float(alpha)
            if alpha in cache:
                return cache[alpha]
            candidate, velocity = polar_retraction_curve(
                origin.W, direction, alpha
            )
            evaluation = oracle.evaluate(candidate)
            point = _LinePoint(
                alpha=alpha,
                W=candidate,
                velocity=velocity,
                evaluation=evaluation,
                slope=_metric(evaluation.gradient, velocity),
            )
            cache[alpha] = point
            evaluations += 1
            armijo_bound = (
                origin.cost + self.options.armijo_c1 * alpha * slope0
            )
            if np.isfinite(evaluation.cost) and evaluation.cost <= armijo_bound:
                if best_armijo is None or evaluation.cost < best_armijo.evaluation.cost:
                    best_armijo = point
            return point

        def zoom(low: _LinePoint | None, high: _LinePoint) -> _LinePoint | None:
            nonlocal evaluations
            low_alpha = 0.0 if low is None else low.alpha
            low_cost = origin.cost if low is None else low.evaluation.cost
            low_slope = slope0 if low is None else low.slope
            high_alpha = high.alpha
            while evaluations < maximum:
                midpoint = 0.5 * (low_alpha + high_alpha)
                alpha = midpoint
                if self.options.line_search_interpolation == "secant":
                    denominator = high.slope - low_slope
                    if np.isfinite(denominator) and abs(denominator) > 1e-300:
                        secant = (
                            low_alpha * high.slope
                            - high_alpha * low_slope
                        ) / denominator
                        width = high_alpha - low_alpha
                        safeguard = 0.1 * width
                        if (
                            np.isfinite(secant)
                            and low_alpha + safeguard < secant
                            and secant < high_alpha - safeguard
                        ):
                            alpha = float(secant)
                if alpha < self.options.minimum_step:
                    break
                point = evaluate(alpha)
                armijo_bound = (
                    origin.cost + self.options.armijo_c1 * alpha * slope0
                )
                if (
                    not np.isfinite(point.evaluation.cost)
                    or point.evaluation.cost > armijo_bound
                    or point.evaluation.cost >= low_cost
                ):
                    high_alpha = alpha
                    high = point
                else:
                    if abs(point.slope) <= -self.options.wolfe_c2 * slope0:
                        return point
                    if point.slope >= 0.0:
                        high_alpha = alpha
                        high = point
                    else:
                        low_alpha = alpha
                        low_cost = point.evaluation.cost
                        low_slope = point.slope
                        low = point
                if abs(high_alpha - low_alpha) < self.options.minimum_step:
                    break
            return None

        alpha = min(
            max(float(alpha0), self.options.minimum_step),
            self.options.maximum_step,
        )
        previous: _LinePoint | None = None
        while evaluations < maximum:
            point = evaluate(alpha)
            armijo_bound = origin.cost + self.options.armijo_c1 * alpha * slope0
            if (
                not np.isfinite(point.evaluation.cost)
                or point.evaluation.cost > armijo_bound
                or (
                    previous is not None
                    and point.evaluation.cost >= previous.evaluation.cost
                )
            ):
                zoomed = zoom(previous, point)
                if zoomed is not None:
                    return _LineSearchResult(zoomed, evaluations, True)
                break
            if abs(point.slope) <= -self.options.wolfe_c2 * slope0:
                return _LineSearchResult(point, evaluations, True)
            if point.slope >= 0.0:
                zoomed = zoom(previous, point)
                if zoomed is not None:
                    return _LineSearchResult(zoomed, evaluations, True)
                break
            previous = point
            next_alpha = min(
                alpha * self.options.line_search_growth,
                self.options.maximum_step,
            )
            if next_alpha <= alpha:
                break
            alpha = next_alpha

        if self.options.accept_armijo_fallback and best_armijo is not None:
            return _LineSearchResult(best_armijo, evaluations, False)
        return _LineSearchResult(None, evaluations, False)

    def optimize(self, A0: Array, rho_target: Array) -> HorizontalCGResult:
        A0 = np.asarray(A0, dtype=np.complex128)
        if A0.ndim != 3 or A0.shape[1] != A0.shape[2]:
            raise ValueError("A0 must have shape (d, D, D)")
        d, D, _ = A0.shape
        W = A0.reshape(d * D, D).copy()
        oracle = FixedTargetDirectGradient(
            rho_target,
            local_dimension=d,
            block_length=self.block_length,
            gauge_tolerance=self.options.gauge_tolerance,
            fixed_point_solver=self.options.fixed_point_solver,
            adjoint_rtol=self.options.adjoint_rtol,
            gauge_projector=self.options.gauge_projector,
            tangent_slice=self.options.tangent_slice,
            structured_gauge_maximum_iterations=(
                self.options.structured_gauge_maximum_iterations
            ),
        )
        optimization_started = time.perf_counter()
        evaluation = oracle.evaluate(W)
        history: list[HorizontalCGRecord] = []
        best = evaluation
        status = "maximum_iterations"
        accepted_steps = 0
        restart_count = 0
        alpha = self.options.initial_step
        previous_gradient: Array | None = None
        previous_preconditioned: Array | None = None
        previous_direction: Array | None = None

        for iteration in range(self.options.max_iterations + 1):
            iteration_started = time.perf_counter()
            if evaluation.cost < best.cost:
                best = evaluation
            if self.options.verbose and iteration % 10 == 0:
                print(
                    f"horizontal CG iter={iteration:4d} "
                    f"cost={evaluation.cost:.6e} "
                    f"residual={evaluation.residual_norm:.6e} "
                    f"grad={evaluation.gradient_norm:.6e}",
                    flush=True,
                )

            reached_residual = (
                self.options.residual_tolerance > 0
                and evaluation.residual_norm <= self.options.residual_tolerance
            )
            reached_cost = (
                self.options.cost_tolerance > 0
                and evaluation.cost <= self.options.cost_tolerance
            )
            reached_gradient = (
                evaluation.gradient_norm <= self.options.gradient_tolerance
            )
            time_limit = (
                self.options.maximum_seconds > 0
                and time.perf_counter() - optimization_started
                >= self.options.maximum_seconds
            )
            if (
                reached_residual
                or reached_cost
                or reached_gradient
                or time_limit
                or iteration == self.options.max_iterations
            ):
                if reached_residual:
                    status = "residual_tolerance"
                elif reached_cost:
                    status = "cost_tolerance"
                elif reached_gradient:
                    status = "gradient_tolerance"
                elif time_limit:
                    status = "time_limit"
                history.append(
                    HorizontalCGRecord(
                        iteration=iteration,
                        cost=evaluation.cost,
                        residual_norm=evaluation.residual_norm,
                        gradient_norm=evaluation.gradient_norm,
                        step_size=0.0,
                        slope=0.0,
                        beta=0.0,
                        line_search_evaluations=0,
                        accepted=False,
                        wolfe_satisfied=False,
                        restarted=False,
                        elapsed_seconds=time.perf_counter() - iteration_started,
                    )
                )
                break

            projector = self._projector(evaluation)
            historical_conjugacy = (
                self.options.conjugacy_metric == "historical_slice"
            )
            if historical_conjugacy:
                metric_gradient = self._historical_slice_gradient(evaluation)
                metric_preconditioned = self._historical_slice_precondition(
                    evaluation, metric_gradient
                )
                preconditioned = projector.project(metric_preconditioned)
            else:
                preconditioned = self._precondition(evaluation, projector)
                metric_gradient = evaluation.gradient
                metric_preconditioned = preconditioned
            scheduled_restart = (
                previous_direction is None
                or iteration % self.options.restart == 0
            )
            restarted = bool(scheduled_restart)
            beta = 0.0
            if scheduled_restart:
                metric_direction = -metric_preconditioned
                direction = projector.project(metric_direction)
            else:
                assert previous_gradient is not None
                assert previous_preconditioned is not None
                if historical_conjugacy:
                    transported_gradient = self._historical_slice_transport(
                        evaluation.W, previous_gradient
                    )
                    transported_preconditioned = (
                        self._historical_slice_transport(
                            evaluation.W, previous_preconditioned
                        )
                    )
                    transported_direction = self._historical_slice_transport(
                        evaluation.W, previous_direction
                    )
                else:
                    transported_gradient = projector.project(
                        previous_gradient
                    )
                    transported_preconditioned = projector.project(
                        previous_preconditioned
                    )
                    transported_direction = projector.project(
                        previous_direction
                    )
                beta = self._hager_zhang(
                    metric_gradient,
                    transported_gradient,
                    metric_preconditioned,
                    transported_preconditioned,
                    transported_direction,
                )
                metric_direction = (
                    -metric_preconditioned + beta * transported_direction
                )
                direction = projector.project(metric_direction)

            slope = _metric(evaluation.gradient, direction)
            if (
                not np.isfinite(slope)
                or slope >= -1e-14
                * max(evaluation.gradient_norm * la.norm(direction), 1e-300)
            ):
                metric_direction = -metric_preconditioned
                direction = projector.project(metric_direction)
                slope = _metric(evaluation.gradient, direction)
                beta = 0.0
                if not restarted:
                    restart_count += 1
                restarted = True

            if alpha is None:
                alpha_trial = 1.0 / max(float(la.norm(direction)), 1e-300)
            else:
                alpha_trial = float(alpha)
            line = self._line_search(
                oracle, evaluation, direction, alpha_trial
            )
            if line.point is None:
                status = "line_search_failed"
                history.append(
                    HorizontalCGRecord(
                        iteration=iteration,
                        cost=evaluation.cost,
                        residual_norm=evaluation.residual_norm,
                        gradient_norm=evaluation.gradient_norm,
                        step_size=0.0,
                        slope=slope,
                        beta=beta,
                        line_search_evaluations=line.evaluations,
                        accepted=False,
                        wolfe_satisfied=False,
                        restarted=restarted,
                        elapsed_seconds=time.perf_counter() - iteration_started,
                    )
                )
                break

            point = line.point
            history.append(
                HorizontalCGRecord(
                    iteration=iteration,
                    cost=evaluation.cost,
                    residual_norm=evaluation.residual_norm,
                    gradient_norm=evaluation.gradient_norm,
                    step_size=point.alpha,
                    slope=slope,
                    beta=beta,
                    line_search_evaluations=line.evaluations,
                    accepted=True,
                    wolfe_satisfied=line.wolfe_satisfied,
                    restarted=restarted,
                    elapsed_seconds=time.perf_counter() - iteration_started,
                )
            )
            previous_gradient = metric_gradient.copy()
            previous_preconditioned = metric_preconditioned.copy()
            previous_direction = metric_direction.copy()
            W = point.W
            evaluation = point.evaluation
            accepted_steps += 1
            # Use the accepted step, not a post-grown trial step, for the next
            # line-search scale.  This avoids the historical accepted-alpha
            # bookkeeping ambiguity.
            alpha = min(
                point.alpha * 1.1,
                self.options.maximum_step,
            )

        elapsed = time.perf_counter() - optimization_started
        return HorizontalCGResult(
            W=best.W.copy(),
            cost=best.cost,
            residual_norm=best.residual_norm,
            gradient_norm=best.gradient_norm,
            status=status,
            history=tuple(history),
            objective_evaluations=oracle.evaluation_count,
            accepted_steps=accepted_steps,
            restarts=restart_count,
            elapsed_seconds=elapsed,
        )


def optimize_fixed_target_horizontal_cg(
    A0: Array,
    rho_target: Array,
    block_length: int,
    options: HorizontalCGOptions | None = None,
) -> tuple[Array, HorizontalCGResult]:
    """Optimize a fixed RDM target with gauge-only horizontal nonlinear CG."""

    result = HorizontalNonlinearCG(block_length, options).optimize(A0, rho_target)
    d, D, _ = np.asarray(A0).shape
    return unstack_tensor(result.W, d, D), result
