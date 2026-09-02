"""Basis-free, Jacobian-free LM for finite-window uMPS matching.

The damped Gauss--Newton problem is solved on the true-gauge-horizontal
tangent with either CG on the normal operator,

    (J^dagger J + mu I) p = -J^dagger r.

or rectangular LSMR applied directly to the damped least-squares problem.
Neither the dense RDM Jacobian nor a basis of the full horizontal tangent is
constructed. A Krylov product uses analytic RDM JVP/VJP contractions followed
by true-gauge projection.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Literal

import numpy as np
import scipy.linalg as la
from scipy.sparse.linalg import LinearOperator, cg, lsmr

from .canonical import polar_retraction, unstack_tensor
from .differential import analytic_directional_rdm, real_vectorize_rho
from .fixed_target_vjp import (
    DirectGradientEvaluation,
    FixedTargetDirectGradient,
    GaugeProjector,
)
from .horizontal import real_vector_to_complex_matrix, stiefel_project
from .optimizer import FixedPointSolver, optimizer_right_fixed_point
from .rdm import block_rdm
from .tangent import vectorize_complex_real


Array = np.ndarray
KrylovPreconditioner = Literal[
    "none",
    "right_fixed_point",
    "right_fixed_point_stiefel",
]
NormalMatvecProjection = Literal["exact", "stiefel"]
KrylovSolver = Literal["cg", "lsmr"]


def _metric(X: Array, Y: Array) -> float:
    return float(np.vdot(X, Y).real)


def _krylov_converged(solver: KrylovSolver, info: int) -> bool:
    if solver == "cg":
        return info == 0
    return info in (0, 1, 2, 4, 5)


@dataclass(frozen=True)
class MatrixFreeLMOptions:
    """Numerical controls for basis-free Gauss--Newton/LM."""

    max_iterations: int = 40
    residual_tolerance: float = 0.0
    cost_tolerance: float = 3e-16
    gradient_tolerance: float = 1e-11
    initial_damping: float = 1e-4
    trust_radius: float = 1.0
    krylov_solver: KrylovSolver = "cg"
    krylov_max_iterations: int = 8
    krylov_initial_iterations: int | None = None
    krylov_growth_factor: float = 2.0
    krylov_relative_tolerance: float = 0.1
    krylov_minimum_relative_tolerance: float = 1e-6
    adaptive_krylov_tolerance: bool = False
    recycle_krylov_solution: bool = False
    lsmr_condition_limit: float = 1e12
    krylov_absolute_tolerance: float = 0.0
    krylov_preconditioner: KrylovPreconditioner = "none"
    normal_matvec_projection: NormalMatvecProjection = "stiefel"
    minimum_preconditioner_shift: float = 1e-12
    maximum_damping_trials: int = 6
    reduce_damping_only_on_krylov_convergence: bool = True
    armijo_c1: float = 1e-4
    minimum_step: float = 1e-8
    gauge_tolerance: float = 1e-10
    gauge_projector: GaugeProjector = "structured"
    structured_gauge_maximum_iterations: int | None = 800
    fixed_point_solver: FixedPointSolver = "fast"
    fixed_point_response_solver: Literal[
        "iterative", "dense_lu", "auto"
    ] = "iterative"
    fixed_point_response_maximum_bytes: int = 128 * 1024**2
    adjoint_rtol: float = 1e-8
    jvp_fixed_point_rtol: float = 1e-8
    jvp_fixed_point_maximum_iterations: int | None = 800
    plateau_window: int = 50
    plateau_relative_cost_drop: float = 1e-4
    plateau_absolute_cost_drop: float = 1e-20
    maximum_seconds: float = 900.0
    verbose: bool = True


@dataclass(frozen=True)
class MatrixFreeLMRecord:
    iteration: int
    cost: float
    residual_norm: float
    gradient_norm: float
    damping: float
    step_size: float
    step_norm: float
    raw_step_norm: float
    krylov_iterations: int
    krylov_iteration_limit: int
    krylov_relative_tolerance: float
    normal_matvecs: int
    jvps: int
    vjps: int
    krylov_info: int
    krylov_residual_norm: float
    krylov_normal_residual_norm: float
    krylov_condition_estimate: float
    accepted: bool
    elapsed_seconds: float


@dataclass(frozen=True)
class MatrixFreeLMResult:
    W: Array
    cost: float
    residual_norm: float
    gradient_norm: float
    status: str
    history: tuple[MatrixFreeLMRecord, ...]
    objective_evaluations: int
    accepted_steps: int
    normal_matvecs: int
    jvps: int
    vjps: int
    krylov_solver: str
    fixed_point_response_solver: str
    fixed_point_response_storage_bytes: int
    elapsed_seconds: float


class MatrixFreeHorizontalLM:
    """Gauss--Newton/LM using only JVPs and VJPs."""

    def __init__(
        self,
        block_length: int,
        options: MatrixFreeLMOptions | None = None,
    ):
        self.block_length = int(block_length)
        self.options = options or MatrixFreeLMOptions()
        if self.block_length < 1:
            raise ValueError("block_length must be positive")
        if self.options.krylov_max_iterations < 1:
            raise ValueError("krylov_max_iterations must be positive")
        if self.options.krylov_solver not in ("cg", "lsmr"):
            raise ValueError("krylov_solver must be 'cg' or 'lsmr'")
        if (
            self.options.krylov_initial_iterations is not None
            and not 1
            <= self.options.krylov_initial_iterations
            <= self.options.krylov_max_iterations
        ):
            raise ValueError(
                "krylov_initial_iterations must lie between one and "
                "krylov_max_iterations"
            )
        if self.options.krylov_growth_factor <= 1:
            raise ValueError("krylov_growth_factor must be greater than one")
        if self.options.krylov_relative_tolerance <= 0:
            raise ValueError("krylov_relative_tolerance must be positive")
        if not (
            0 < self.options.krylov_minimum_relative_tolerance
            <= self.options.krylov_relative_tolerance
        ):
            raise ValueError(
                "krylov_minimum_relative_tolerance must be positive and no "
                "larger than krylov_relative_tolerance"
            )
        if self.options.krylov_absolute_tolerance < 0:
            raise ValueError("krylov_absolute_tolerance must be non-negative")
        if self.options.lsmr_condition_limit <= 1:
            raise ValueError("lsmr_condition_limit must be greater than one")
        if self.options.initial_damping <= 0:
            raise ValueError("initial_damping must be positive")
        if self.options.krylov_preconditioner not in (
            "none",
            "right_fixed_point",
            "right_fixed_point_stiefel",
        ):
            raise ValueError(
                "krylov_preconditioner must be 'none', 'right_fixed_point', "
                "or 'right_fixed_point_stiefel'"
            )
        if self.options.normal_matvec_projection not in (
            "exact",
            "stiefel",
        ):
            raise ValueError(
                "normal_matvec_projection must be 'exact' or 'stiefel'"
            )
        if self.options.gauge_projector not in ("dense", "structured"):
            raise ValueError("gauge_projector must be 'dense' or 'structured'")
        if self.options.fixed_point_solver not in ("dense", "fast"):
            raise ValueError("fixed_point_solver must be 'dense' or 'fast'")
        if self.options.fixed_point_response_solver not in (
            "iterative",
            "dense_lu",
            "auto",
        ):
            raise ValueError(
                "fixed_point_response_solver must be 'iterative', "
                "'dense_lu', or 'auto'"
            )
        if self.options.fixed_point_response_maximum_bytes < 0:
            raise ValueError(
                "fixed_point_response_maximum_bytes must be non-negative"
            )

    def _cg_normal_direction(
        self,
        oracle: FixedTargetDirectGradient,
        evaluation: DirectGradientEvaluation,
        damping: float,
        *,
        maximum_iterations: int,
        relative_tolerance: float,
        initial_guess: Array | None = None,
    ) -> tuple[Array, int, int, int]:
        shape = evaluation.W.shape
        size = 2 * evaluation.W.size
        normal_matvecs = 0

        def matvec(vector: Array) -> Array:
            nonlocal normal_matvecs
            normal_matvecs += 1
            # CG starts at zero with a horizontal right-hand side.  Because
            # this operator maps horizontal vectors to horizontal vectors,
            # all Krylov vectors remain in that subspace without an explicit
            # input projection.  The output VJP is projected exactly.
            direction = real_vector_to_complex_matrix(vector, shape)
            delta_A = unstack_tensor(
                direction,
                evaluation.A.shape[0],
                evaluation.A.shape[1],
            )
            delta_rho, _ = analytic_directional_rdm(
                evaluation.A,
                delta_A,
                self.block_length,
                r=evaluation.r,
                fixed_point_derivative_solver="iterative",
                fixed_point_derivative_rtol=(
                    self.options.jvp_fixed_point_rtol
                ),
                fixed_point_derivative_maximum_iterations=(
                    self.options.jvp_fixed_point_maximum_iterations
                ),
                fixed_point_response_solver=evaluation.response_solver,
                block_product_levels=evaluation.block_product_levels,
            )
            ambient_normal = oracle.ambient_vjp(evaluation, delta_rho)
            if self.options.normal_matvec_projection == "exact":
                normal = evaluation.projector.project(ambient_normal)
            else:
                # Gauge invariance gives J[gauge] = 0, hence J^dagger w is
                # already gauge orthogonal in exact arithmetic.  Stiefel
                # projection removes only the canonical normal component.
                # The completed Krylov direction is exactly gauge projected
                # once below before it can be used by the optimizer.
                normal = stiefel_project(evaluation.W, ambient_normal)
            normal += float(damping) * direction
            return vectorize_complex_real(normal)

        operator = LinearOperator(
            (size, size), matvec=matvec, dtype=float
        )
        preconditioner = None
        if self.options.krylov_preconditioner in (
            "right_fixed_point",
            "right_fixed_point_stiefel",
        ):
            matrix = evaluation.r + max(
                float(damping),
                self.options.minimum_preconditioner_shift,
            ) * np.eye(evaluation.r.shape[0], dtype=np.complex128)

            def precondition(vector: Array) -> Array:
                value = real_vector_to_complex_matrix(vector, shape)
                try:
                    value = la.solve(
                        matrix.T,
                        value.T,
                        assume_a="gen",
                        check_finite=False,
                    ).T
                except la.LinAlgError:
                    value = value @ la.pinv(matrix, check_finite=False)
                if (
                    self.options.krylov_preconditioner
                    == "right_fixed_point_stiefel"
                ):
                    # P_T R^{-1} P_T is positive definite on the Stiefel
                    # tangent and is therefore a valid PCG preconditioner.
                    # It deliberately leaves true-gauge directions in the
                    # inner solve; J annihilates them, damping controls them,
                    # and the completed LM direction is exactly gauge
                    # projected once below.  This avoids one structured gauge
                    # solve per Krylov iteration.
                    value = stiefel_project(evaluation.W, value)
                else:
                    value = evaluation.projector.project(value)
                return vectorize_complex_real(value)

            preconditioner = LinearOperator(
                (size, size), matvec=precondition, dtype=float
            )
        rhs = -vectorize_complex_real(evaluation.gradient)
        krylov_iterations = 0

        def callback(_: Array) -> None:
            nonlocal krylov_iterations
            krylov_iterations += 1

        coefficients, info = cg(
            operator,
            rhs,
            rtol=relative_tolerance,
            atol=self.options.krylov_absolute_tolerance,
            maxiter=maximum_iterations,
            M=preconditioner,
            x0=(
                None
                if initial_guess is None
                else vectorize_complex_real(initial_guess)
            ),
            callback=callback,
        )
        direction = real_vector_to_complex_matrix(coefficients, shape)
        # Remove accumulated roundoff and choose the current horizontal
        # representative before retraction.
        direction = evaluation.projector.project(direction)
        return direction, krylov_iterations, normal_matvecs, int(info)

    def _lsmr_direction(
        self,
        oracle: FixedTargetDirectGradient,
        evaluation: DirectGradientEvaluation,
        damping: float,
        *,
        maximum_iterations: int,
        relative_tolerance: float,
        initial_guess: Array | None = None,
    ) -> tuple[Array, int, int, int, int, float, float, float]:
        """Solve the damped rectangular least-squares problem with LSMR.

        LSMR sees the Jacobian only through ``Jv`` and ``J^dagger w``.  Its
        Krylov vectors start in the Stiefel tangent because every adjoint
        product is projected there.  The completed step is then projected
        onto the exact true-gauge-horizontal representative.
        """

        shape = evaluation.W.shape
        input_size = 2 * evaluation.W.size
        rdm_output_size = 2 * evaluation.rho.size
        output_size = rdm_output_size + input_size
        square_root_damping = float(np.sqrt(damping))
        jvps = 0
        vjps = 0

        def matvec(vector: Array) -> Array:
            nonlocal jvps
            jvps += 1
            direction = real_vector_to_complex_matrix(vector, shape)
            delta_A = unstack_tensor(
                direction,
                evaluation.A.shape[0],
                evaluation.A.shape[1],
            )
            delta_rho, _ = analytic_directional_rdm(
                evaluation.A,
                delta_A,
                self.block_length,
                r=evaluation.r,
                fixed_point_derivative_solver="iterative",
                fixed_point_derivative_rtol=(
                    self.options.jvp_fixed_point_rtol
                ),
                fixed_point_derivative_maximum_iterations=(
                    self.options.jvp_fixed_point_maximum_iterations
                ),
                fixed_point_response_solver=evaluation.response_solver,
                block_product_levels=evaluation.block_product_levels,
            )
            return np.concatenate(
                (
                    real_vectorize_rho(delta_rho),
                    square_root_damping * np.asarray(vector, dtype=float),
                )
            )

        def rmatvec(vector: Array) -> Array:
            nonlocal vjps
            vjps += 1
            vector = np.asarray(vector, dtype=float)
            weight = real_vector_to_complex_matrix(
                vector[:rdm_output_size], evaluation.rho.shape
            )
            ambient = oracle.ambient_vjp(evaluation, weight)
            regularization = real_vector_to_complex_matrix(
                vector[rdm_output_size:], shape
            )
            tangent = stiefel_project(
                evaluation.W,
                ambient + square_root_damping * regularization,
            )
            return vectorize_complex_real(tangent)

        operator = LinearOperator(
            (output_size, input_size),
            matvec=matvec,
            rmatvec=rmatvec,
            dtype=float,
        )
        rhs = np.concatenate(
            (
                -real_vectorize_rho(evaluation.residual),
                np.zeros(input_size, dtype=float),
            )
        )
        result = lsmr(
            operator,
            rhs,
            damp=0.0,
            atol=relative_tolerance,
            btol=relative_tolerance,
            conlim=self.options.lsmr_condition_limit,
            maxiter=maximum_iterations,
            show=False,
            x0=(
                None
                if initial_guess is None
                else vectorize_complex_real(initial_guess)
            ),
        )
        coefficients = result[0]
        direction = real_vector_to_complex_matrix(coefficients, shape)
        direction = evaluation.projector.project(direction)
        return (
            direction,
            int(result[2]),
            jvps,
            vjps,
            int(result[1]),
            float(result[3]),
            float(result[4]),
            float(result[6]),
        )

    def _line_search(
        self,
        evaluation: DirectGradientEvaluation,
        direction: Array,
        slope: float,
        rho_target: Array,
    ) -> tuple[bool, float, Array, Array | None, float]:
        alpha = 1.0
        while alpha >= self.options.minimum_step:
            candidate = polar_retraction(evaluation.W + alpha * direction)
            A = unstack_tensor(
                candidate,
                evaluation.A.shape[0],
                evaluation.A.shape[1],
            )
            r, _ = optimizer_right_fixed_point(
                A, self.options.fixed_point_solver
            )
            residual = block_rdm(A, self.block_length, r) - rho_target
            cost = 0.5 * float(la.norm(residual) ** 2)
            if np.isfinite(cost) and cost <= (
                evaluation.cost + self.options.armijo_c1 * alpha * slope
            ):
                return True, alpha, candidate, r, cost
            alpha *= 0.5
        return False, 0.0, evaluation.W, None, evaluation.cost

    def optimize(
        self,
        A0: Array,
        rho_target: Array,
        initial_fixed_point: Array | None = None,
    ) -> MatrixFreeLMResult:
        A0 = np.asarray(A0, dtype=np.complex128)
        if A0.ndim != 3 or A0.shape[1] != A0.shape[2]:
            raise ValueError("A0 must have shape (d, D, D)")
        d, D, _ = A0.shape
        W = A0.reshape(d * D, D).copy()
        response_solver = self.options.fixed_point_response_solver
        estimated_response_bytes = 16 * D**4 + 4 * D**2
        if response_solver == "auto":
            response_solver = (
                "dense_lu"
                if estimated_response_bytes
                <= self.options.fixed_point_response_maximum_bytes
                else "iterative"
            )
        oracle = FixedTargetDirectGradient(
            rho_target,
            local_dimension=d,
            block_length=self.block_length,
            gauge_tolerance=self.options.gauge_tolerance,
            fixed_point_solver=self.options.fixed_point_solver,
            fixed_point_response_solver=response_solver,
            adjoint_rtol=self.options.adjoint_rtol,
            gauge_projector=self.options.gauge_projector,
            # The analytic VJP is gauge orthogonal in exact arithmetic.  For
            # production, remove the small gauge component left by iterative
            # contraction tolerances before it becomes the CG right-hand side.
            gradient_projection="exact",
            structured_gauge_maximum_iterations=(
                self.options.structured_gauge_maximum_iterations
            ),
        )
        started = time.perf_counter()
        damping = float(self.options.initial_damping)
        history: list[MatrixFreeLMRecord] = []
        best = None
        accepted_steps = 0
        total_normal_matvecs = 0
        total_jvps = 0
        total_vjps = 0
        status = "maximum_iterations"
        next_fixed_point = (
            None
            if initial_fixed_point is None
            else np.asarray(initial_fixed_point, dtype=np.complex128)
        )
        krylov_iteration_limit = (
            self.options.krylov_max_iterations
            if self.options.krylov_initial_iterations is None
            else self.options.krylov_initial_iterations
        )
        initial_residual_norm: float | None = None
        recycled_direction: Array | None = None

        for iteration in range(self.options.max_iterations + 1):
            iteration_started = time.perf_counter()
            evaluation = oracle.evaluate(W, next_fixed_point)
            next_fixed_point = None
            if initial_residual_norm is None:
                initial_residual_norm = max(
                    evaluation.residual_norm, np.finfo(float).tiny
                )
            krylov_relative_tolerance = self.options.krylov_relative_tolerance
            if self.options.adaptive_krylov_tolerance:
                relative_residual = max(
                    evaluation.residual_norm / initial_residual_norm,
                    0.0,
                )
                krylov_relative_tolerance = max(
                    self.options.krylov_minimum_relative_tolerance,
                    min(
                        self.options.krylov_relative_tolerance,
                        self.options.krylov_relative_tolerance
                        * np.sqrt(relative_residual),
                    ),
                )
            if best is None or evaluation.cost < best.cost:
                best = evaluation
            if self.options.verbose:
                print(
                    f"matrix-free LM iter={iteration:3d} "
                    f"cost={evaluation.cost:.6e} "
                    f"residual={evaluation.residual_norm:.6e} "
                    f"grad={evaluation.gradient_norm:.6e} "
                    f"mu={damping:.3e} krylov_max={krylov_iteration_limit} "
                    f"krylov={self.options.krylov_solver} "
                    f"krylov_rtol={krylov_relative_tolerance:.1e}",
                    flush=True,
                )

            reached_residual = (
                self.options.residual_tolerance > 0
                and evaluation.residual_norm
                <= self.options.residual_tolerance
            )
            reached_cost = (
                self.options.cost_tolerance > 0
                and evaluation.cost <= self.options.cost_tolerance
            )
            reached_gradient = (
                evaluation.gradient_norm
                <= self.options.gradient_tolerance
            )
            time_limit = (
                self.options.maximum_seconds > 0
                and time.perf_counter() - started
                >= self.options.maximum_seconds
            )
            recent_costs = [
                record.cost
                for record in history[
                    -max(self.options.plateau_window - 1, 0) :
                ]
            ] + [evaluation.cost]
            plateau = False
            if (
                self.options.plateau_window > 1
                and len(recent_costs) >= self.options.plateau_window
            ):
                start_cost = recent_costs[0]
                best_recent = min(recent_costs)
                required_drop = (
                    self.options.plateau_absolute_cost_drop
                    + self.options.plateau_relative_cost_drop
                    * max(abs(start_cost), 1e-300)
                )
                plateau = start_cost - best_recent <= required_drop
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
                    MatrixFreeLMRecord(
                        iteration=iteration,
                        cost=evaluation.cost,
                        residual_norm=evaluation.residual_norm,
                        gradient_norm=evaluation.gradient_norm,
                        damping=damping,
                        step_size=0.0,
                        step_norm=0.0,
                        raw_step_norm=0.0,
                        krylov_iterations=0,
                        krylov_iteration_limit=krylov_iteration_limit,
                        krylov_relative_tolerance=krylov_relative_tolerance,
                        normal_matvecs=0,
                        jvps=0,
                        vjps=0,
                        krylov_info=0,
                        krylov_residual_norm=0.0,
                        krylov_normal_residual_norm=0.0,
                        krylov_condition_estimate=0.0,
                        accepted=False,
                        elapsed_seconds=time.perf_counter()
                        - iteration_started,
                    )
                )
                break

            accepted = False
            step_size = 0.0
            step_norm = 0.0
            raw_step_norm = 0.0
            krylov_iterations = 0
            normal_matvecs = 0
            jvps = 0
            vjps = 0
            krylov_info = 0
            krylov_residual_norm = 0.0
            krylov_normal_residual_norm = 0.0
            krylov_condition_estimate = 0.0
            local_damping = damping
            candidate = W
            candidate_fixed_point = None
            trial_iteration_limit = krylov_iteration_limit
            for _ in range(self.options.maximum_damping_trials):
                initial_guess = (
                    recycled_direction
                    if self.options.recycle_krylov_solution
                    else None
                )
                if self.options.krylov_solver == "lsmr":
                    (
                        direction,
                        solve_iterations,
                        solve_jvps,
                        solve_vjps,
                        solve_info,
                        solve_residual_norm,
                        solve_normal_residual_norm,
                        solve_condition_estimate,
                    ) = self._lsmr_direction(
                        oracle,
                        evaluation,
                        local_damping,
                        maximum_iterations=trial_iteration_limit,
                        relative_tolerance=krylov_relative_tolerance,
                        initial_guess=initial_guess,
                    )
                    solve_matvecs = 0
                else:
                    (
                        direction,
                        solve_iterations,
                        solve_matvecs,
                        solve_info,
                    ) = self._cg_normal_direction(
                        oracle,
                        evaluation,
                        local_damping,
                        maximum_iterations=trial_iteration_limit,
                        relative_tolerance=krylov_relative_tolerance,
                        initial_guess=initial_guess,
                    )
                    solve_jvps = solve_matvecs
                    solve_vjps = solve_matvecs
                    solve_residual_norm = float("nan")
                    solve_normal_residual_norm = float("nan")
                    solve_condition_estimate = float("nan")
                krylov_iterations += solve_iterations
                normal_matvecs += solve_matvecs
                jvps += solve_jvps
                vjps += solve_vjps
                krylov_info = solve_info
                krylov_residual_norm = solve_residual_norm
                krylov_normal_residual_norm = solve_normal_residual_norm
                krylov_condition_estimate = solve_condition_estimate
                raw_step_norm = float(la.norm(direction))
                if (
                    self.options.trust_radius > 0
                    and raw_step_norm > self.options.trust_radius
                ):
                    direction *= self.options.trust_radius / raw_step_norm
                step_norm = float(la.norm(direction))
                slope = _metric(evaluation.gradient, direction)
                if solve_info < 0 or not np.isfinite(slope) or slope >= 0:
                    local_damping *= 10.0
                    continue
                (
                    accepted,
                    step_size,
                    candidate,
                    candidate_fixed_point,
                    _,
                ) = self._line_search(evaluation, direction, slope, rho_target)
                if accepted:
                    resolved_step = (
                        _krylov_converged(
                            self.options.krylov_solver, solve_info
                        )
                        or not self.options.reduce_damping_only_on_krylov_convergence
                    )
                    if (
                        resolved_step
                        and step_size > 0.99
                        and raw_step_norm
                        <= 1.05 * self.options.trust_radius
                    ):
                        damping = max(local_damping * 0.3, 1e-12)
                    elif step_size < 0.5:
                        damping = min(local_damping * 3.0, 1e12)
                    else:
                        damping = local_damping
                    W = candidate
                    next_fixed_point = candidate_fixed_point
                    recycled_direction = step_size * direction
                    accepted_steps += 1
                    break
                if (
                    not _krylov_converged(
                        self.options.krylov_solver, solve_info
                    )
                    and trial_iteration_limit
                    < self.options.krylov_max_iterations
                ):
                    trial_iteration_limit = min(
                        self.options.krylov_max_iterations,
                        max(
                            trial_iteration_limit + 1,
                            int(
                                np.ceil(
                                    trial_iteration_limit
                                    * self.options.krylov_growth_factor
                                )
                            ),
                        ),
                    )
                    continue
                local_damping *= 10.0

            total_normal_matvecs += normal_matvecs
            total_jvps += jvps
            total_vjps += vjps
            if (
                not _krylov_converged(
                    self.options.krylov_solver, krylov_info
                )
                and krylov_iteration_limit
                < self.options.krylov_max_iterations
            ):
                krylov_iteration_limit = min(
                    self.options.krylov_max_iterations,
                    max(
                        krylov_iteration_limit + 1,
                        int(
                            np.ceil(
                                krylov_iteration_limit
                                * self.options.krylov_growth_factor
                            )
                        ),
                    ),
                )
            if not accepted:
                damping = local_damping
                status = "line_search_failed"
            history.append(
                MatrixFreeLMRecord(
                    iteration=iteration,
                    cost=evaluation.cost,
                    residual_norm=evaluation.residual_norm,
                    gradient_norm=evaluation.gradient_norm,
                    damping=local_damping,
                    step_size=step_size,
                    step_norm=step_norm,
                    raw_step_norm=raw_step_norm,
                    krylov_iterations=krylov_iterations,
                    krylov_iteration_limit=trial_iteration_limit,
                    krylov_relative_tolerance=krylov_relative_tolerance,
                    normal_matvecs=normal_matvecs,
                    jvps=jvps,
                    vjps=vjps,
                    krylov_info=krylov_info,
                    krylov_residual_norm=krylov_residual_norm,
                    krylov_normal_residual_norm=(
                        krylov_normal_residual_norm
                    ),
                    krylov_condition_estimate=krylov_condition_estimate,
                    accepted=accepted,
                    elapsed_seconds=time.perf_counter() - iteration_started,
                )
            )
            if not accepted:
                break

        if best is None:
            raise RuntimeError("optimizer performed no evaluation")
        return MatrixFreeLMResult(
            W=best.W.copy(),
            cost=best.cost,
            residual_norm=best.residual_norm,
            gradient_norm=best.gradient_norm,
            status=status,
            history=tuple(history),
            objective_evaluations=oracle.evaluation_count,
            accepted_steps=accepted_steps,
            normal_matvecs=total_normal_matvecs,
            jvps=total_jvps,
            vjps=total_vjps,
            krylov_solver=self.options.krylov_solver,
            fixed_point_response_solver=response_solver,
            fixed_point_response_storage_bytes=(
                0
                if best.response_solver is None
                else best.response_solver.storage_bytes
            ),
            elapsed_seconds=time.perf_counter() - started,
        )


def optimize_fixed_target_matrix_free_lm(
    A0: Array,
    rho_target: Array,
    block_length: int,
    options: MatrixFreeLMOptions | None = None,
    initial_fixed_point: Array | None = None,
) -> tuple[Array, MatrixFreeLMResult]:
    """Optimize a fixed target with basis-free Jacobian-free LM."""

    result = MatrixFreeHorizontalLM(block_length, options).optimize(
        A0,
        rho_target,
        initial_fixed_point=initial_fixed_point,
    )
    d, D, _ = np.asarray(A0).shape
    return unstack_tensor(result.W, d, D), result
