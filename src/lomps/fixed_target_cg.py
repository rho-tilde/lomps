"""Analytic fixed-target Grassmann CG for the first lifted QDMT step.

This module intentionally mirrors the historical QDMT conjugate-gradient
launch used to obtain the first nontrivial tensor from a product input.  The
objective is the fixed-target Hilbert-Schmidt distance

    Tr(rho(A)^2) - 2 Tr(rho_target rho(A)) + const,

with an analytic gradient rather than a directional RDM Jacobian.
"""

from __future__ import annotations

from dataclasses import dataclass
import string
import time
from typing import Callable

import numpy as np
import scipy.linalg as la
from scipy.sparse.linalg import LinearOperator, gmres

from .canonical import stack_tensor, unstack_tensor
from .optimizer import CGOptions, _real_metric, grassmann_project


Array = np.ndarray


@dataclass(frozen=True)
class FixedTargetCGRecord:
    iteration: int
    cost: float
    gradient_norm: float
    step_size: float
    beta: float
    accepted: bool
    elapsed_seconds: float


@dataclass(frozen=True)
class FixedTargetCGResult:
    W: Array
    cost: float
    residual_norm: float
    gradient_norm: float
    status: str
    history: tuple[FixedTargetCGRecord, ...]


@dataclass(frozen=True)
class LineSearchPoint:
    alpha: float
    cost: float
    slope: float


def _ncon(tensors: list[Array] | tuple[Array, ...], indices: list[list[int]] | tuple[list[int], ...]) -> Array:
    """Small ncon-compatible wrapper for the contractions used here."""

    labels: list[int] = []
    for item in indices:
        labels.extend(item)
    unique = list(dict.fromkeys(labels))
    if len(unique) > len(string.ascii_letters):
        raise ValueError("too many tensor labels for local ncon helper")
    label_to_char = dict(zip(unique, string.ascii_letters))
    inputs = ["".join(label_to_char[label] for label in item) for item in indices]
    outputs = [label for label in sorted(unique, key=abs) if label < 0]
    output = "".join(label_to_char[label] for label in outputs)
    expression = ",".join(inputs) + "->" + output
    return np.einsum(expression, *tensors, optimize=True)


def _to_old_tensor(A: Array) -> Array:
    return np.transpose(np.asarray(A, dtype=np.complex128), (1, 0, 2)).copy()


def _from_old_tensor(A: Array) -> Array:
    return np.transpose(np.asarray(A, dtype=np.complex128), (1, 0, 2)).copy()


def _old_matrix_to_lomps_matrix(W: Array, d: int, D: int) -> Array:
    old_tensor = np.asarray(W, dtype=np.complex128).reshape(D, d, D, order="C")
    return stack_tensor(_from_old_tensor(old_tensor))


def _lomps_matrix_to_old_matrix(W: Array, d: int, D: int) -> Array:
    tensor = unstack_tensor(np.asarray(W, dtype=np.complex128), d, D)
    return _to_old_tensor(tensor).reshape(d * D, D, order="C")


def _old_to_mps_chain(A: Array, length: int) -> Array:
    """Return the open-chain product for an old-convention tensor (D,d,D)."""

    if length < 0:
        raise ValueError("length must be non-negative")
    D = A.shape[0]
    if length == 0:
        return np.eye(D, dtype=np.complex128)
    if length == 1:
        return np.asarray(A, dtype=np.complex128)
    tensors = [A for _ in range(length)]
    indices = [
        [-(site), -(site + 1), site]
        if site == 1
        else [site - 1, -(site + 1), -(site + 2)]
        if site == length
        else [site - 1, -(site + 1), site]
        for site in range(1, length + 1)
    ]
    return _ncon(tensors, indices)


def _old_right_fixed_point(A: Array) -> Array:
    """Dense right fixed point for an old-convention tensor (D,d,D)."""

    A_lomps = _from_old_tensor(A)
    from .transfer import right_fixed_point

    r, _ = right_fixed_point(A_lomps)
    return r


def _right_fixed_point_gradient(A: Array, r: Array, v: Array) -> Array:
    """Historical adjoint fixed-point derivative contribution."""

    D = A.shape[0]

    def matvec(vector: Array) -> Array:
        matrix = vector.reshape(D, D)
        transfer = _ncon(
            [matrix, A, A.conj()],
            [[1, 2], [2, 3, -2], [1, 3, -1]],
        )
        fixed = np.trace(matrix @ r) * np.eye(D, dtype=np.complex128)
        return (matrix - transfer + fixed).reshape(D * D)

    operator = LinearOperator((D * D, D * D), matvec=matvec, dtype=np.complex128)
    solution, info = gmres(operator, np.asarray(v, dtype=np.complex128).reshape(D * D))
    if info != 0:
        # The historical code accepted the GMRES output. Keep that behavior,
        # but surface a warning-like print because this affects only launch CG.
        print(f"Warning: fixed-point derivative GMRES info={info}", flush=True)
    Rh = solution.reshape(D, D)
    return _ncon([Rh, A, r], [[-1, 1], [1, -2, 2], [2, -3]])


class FixedTargetHSCost:
    """Fixed-target HS cost and analytic derivative in old tensor convention."""

    def __init__(self, rho_target: Array, *, local_dimension: int, block_length: int):
        self.local_dimension = int(local_dimension)
        self.block_length = int(block_length)
        dimension = self.local_dimension ** self.block_length
        rho_target = np.asarray(rho_target, dtype=np.complex128)
        if rho_target.shape != (dimension, dimension):
            raise ValueError("rho_target shape does not match block length")
        self.rho_target = rho_target.reshape(
            (self.local_dimension,) * (2 * self.block_length)
        )
        self.target_norm = self._contract_rhos(self.rho_target, self.rho_target)

    def _build_rho(self, A: Array, r: Array) -> Array:
        L = self.block_length
        chain = _old_to_mps_chain(A, L)
        return _ncon(
            [chain, chain.conj(), r],
            [
                [1] + [-(i + 1) for i in range(L)] + [2],
                [1] + [-(L + i + 1) for i in range(L)] + [3],
                [2, 3],
            ],
        )

    def _contract_rhos(self, rho_a: Array, rho_b: Array) -> complex:
        L = self.block_length
        return _ncon(
            [rho_a, rho_b],
            [
                list(range(1, 2 * L + 1)),
                list(range(L + 1, 2 * L + 1)) + list(range(1, L + 1)),
            ],
        )

    def cost(self, A: Array, r: Array) -> complex:
        rho = self._build_rho(A, r)
        return (
            self.target_norm
            + self._contract_rhos(rho, rho)
            - 2.0 * self._contract_rhos(self.rho_target, rho)
        )

    def _drho(self, site: int, A: Array, r: Array) -> Array:
        L = self.block_length
        d = A.shape[1]
        chain = _old_to_mps_chain(A, L)
        left_bra = _old_to_mps_chain(A, site).conj()
        right_bra = _old_to_mps_chain(A, L - site - 1).conj()
        return _ncon(
            [chain, left_bra, right_bra, np.eye(d, dtype=np.complex128), r],
            [
                [1] + [-i for i in range(1, L + 1)] + [2],
                [1] + [-i for i in range(L + 1, L + 1 + site)] + [-2 * L - 1],
                [-2 * L - 3] + [-i for i in range(L + 2 + site, 2 * L + 1)] + [3],
                [-L - 1 - site, -2 * L - 2],
                [2, 3],
            ],
        )

    def _rho_open(self, A: Array) -> Array:
        L = self.block_length
        chain = _old_to_mps_chain(A, L)
        return _ncon(
            [chain, chain.conj()],
            [
                [1] + [-(k) for k in range(1, L + 1)] + [-(2 * L + 1)],
                [1] + [-(k) for k in range(L + 1, 2 * L + 1)] + [-(2 * L + 2)],
            ],
        )

    def _contract_rho_open_with_rho(self, rho_open: Array, rho_closed: Array) -> Array:
        L = self.block_length
        return _ncon(
            [rho_open, rho_closed],
            [
                list(range(1, 2 * L + 1)) + [-1, -2],
                list(range(L + 1, 2 * L + 1)) + list(range(1, L + 1)),
            ],
        )

    def derivative(self, A: Array, r: Array) -> Array:
        L = self.block_length
        rho = self._build_rho(A, r)
        gradient = np.zeros_like(A, dtype=np.complex128)

        target_part = np.zeros_like(A, dtype=np.complex128)
        rho_part = np.zeros_like(A, dtype=np.complex128)
        for site in range(L):
            drho = self._drho(site, A, r)
            target_part += _ncon(
                [self.rho_target, drho],
                [
                    list(range(1, 2 * L + 1)),
                    list(range(L + 1, 2 * L + 1))
                    + list(range(1, L + 1))
                    + [-1, -2, -3],
                ],
            )
            rho_part += _ncon(
                [rho, drho],
                [
                    list(range(1, 2 * L + 1)),
                    list(range(L + 1, 2 * L + 1))
                    + list(range(1, L + 1))
                    + [-1, -2, -3],
                ],
            )

        gradient -= 2.0 * target_part
        gradient += 2.0 * rho_part

        rho_open = self._rho_open(A)
        gradient += 2.0 * _right_fixed_point_gradient(
            A, r, self._contract_rho_open_with_rho(rho_open, rho).T
        )
        gradient -= 2.0 * _right_fixed_point_gradient(
            A, r, self._contract_rho_open_with_rho(rho_open, self.rho_target).T
        )
        return gradient

    def fg(self, W: Array) -> tuple[complex, Array, Array]:
        dD, D = W.shape
        d = dD // D
        A = np.asarray(W, dtype=np.complex128).reshape(D, d, D, order="C")
        r = _old_right_fixed_point(A)
        cost = self.cost(A, r)
        derivative = self.derivative(A, r).reshape(d * D, D, order="C")
        return cost, grassmann_project(W, derivative), r


def _precondition(gradient: Array, r: Array) -> Array:
    delta = float(la.norm(gradient) ** 2)
    return gradient @ la.inv(r + np.eye(r.shape[0], dtype=np.complex128) * delta)


def _retract(W: Array, X: Array, alpha: float) -> tuple[Array, Array]:
    U, singular_values, Vh = la.svd(X, full_matrices=False, check_finite=False)
    cos_values = np.diag(np.cos(singular_values * alpha))
    sin_values = np.diag(np.sin(singular_values * alpha))
    W_new = W @ (Vh.conj().T @ cos_values @ Vh) + U @ sin_values @ Vh
    sin_scaled = np.diag(np.sin(singular_values * alpha) * singular_values)
    cos_scaled = np.diag(np.cos(singular_values * alpha) * singular_values)
    X_new = -W @ (Vh.conj().T @ sin_scaled @ Vh) + U @ cos_scaled @ Vh

    gram = W_new.conj().T @ W_new
    evals, evecs = la.eigh(gram, check_finite=False)
    evals = np.clip(evals, 1e-15, None)
    W_new = W_new @ (evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.conj().T)
    X_new = grassmann_project(W_new, X_new)
    return W_new.astype(np.complex128), X_new.astype(np.complex128)


def _transport(Y: Array, W: Array, X: Array, alpha: float, W_new: Array) -> Array:
    U, singular_values, Vh = la.svd(X, full_matrices=False, check_finite=False)
    U_dagger_Y = U.conj().T @ Y
    W_V = W @ Vh.conj().T
    cos_values = np.diag(np.cos(singular_values * alpha))
    sin_values = np.diag(np.sin(singular_values * alpha))
    transported = Y + U @ ((cos_values - np.eye(len(singular_values))) @ U_dagger_Y)
    transported -= W_V @ (sin_values @ U_dagger_Y)
    return grassmann_project(W_new, transported).astype(np.complex128)


def _satisfies_wolfe(
    point: LineSearchPoint,
    origin: LineSearchPoint,
    *,
    c1: float,
    c2: float,
    epsilon: float,
) -> bool:
    exact = (
        point.cost <= origin.cost + c1 * point.alpha * origin.slope
        and point.slope >= c2 * origin.slope
    )
    approximate = point.cost <= origin.cost + epsilon and (
        (2.0 * c1 - 1.0) * origin.slope >= point.slope >= c2 * origin.slope
    )
    return bool(exact or approximate)


def _line_search(
    fg: Callable[[Array], tuple[complex, Array, Array]],
    W: Array,
    X: Array,
    cost: complex,
    gradient: Array,
    alpha0: float,
    *,
    c1: float = 1e-4,
    c2: float = 0.9,
    epsilon: float = 1e-6,
    rho: float = 5.0,
    gamma: float = 0.66,
    max_iter: int = 20,
) -> tuple[float, Array, complex, Array, Array, bool]:
    def take_step(alpha: float) -> tuple[LineSearchPoint, Array, Array, Array]:
        W_candidate, transported = _retract(W, X, alpha)
        candidate_cost, candidate_gradient, candidate_r = fg(W_candidate)
        slope = _real_metric(candidate_gradient, transported)
        point = LineSearchPoint(alpha, float(np.real(candidate_cost)), slope)
        return point, W_candidate, candidate_gradient, candidate_r

    initial_slope = _real_metric(gradient, X)
    if initial_slope >= 0:
        return alpha0, W, cost, gradient, np.eye(W.shape[1]), False

    origin = LineSearchPoint(0.0, float(np.real(cost)), initial_slope)
    initial, initial_W, initial_gradient, initial_r = take_step(alpha0)
    if _satisfies_wolfe(initial, origin, c1=c1, c2=c2, epsilon=epsilon):
        return initial.alpha, initial_W, initial.cost, initial_gradient, initial_r, True

    alpha = alpha0
    lower = origin
    upper: LineSearchPoint | None = None
    for _ in range(max_iter):
        point, _, _, _ = take_step(alpha)
        if not np.isfinite(point.cost) or not np.isfinite(point.slope):
            alpha = 0.5 * (lower.alpha + alpha)
            continue
        if point.slope >= 0 or point.cost > origin.cost + epsilon:
            upper = point
            break
        lower = point
        alpha *= rho
    if upper is None:
        return alpha0, W, cost, gradient, np.eye(W.shape[1]), False

    previous_width = upper.alpha - lower.alpha
    for _ in range(max_iter):
        if abs(upper.slope - lower.slope) > 1e-300:
            alpha = (
                lower.alpha * upper.slope - upper.alpha * lower.slope
            ) / (upper.slope - lower.slope)
        else:
            alpha = 0.5 * (lower.alpha + upper.alpha)
        if upper.alpha - lower.alpha > gamma * previous_width or not (
            lower.alpha < alpha < upper.alpha
        ):
            alpha = 0.5 * (lower.alpha + upper.alpha)
        previous_width = upper.alpha - lower.alpha
        point, W_candidate, gradient_candidate, r_candidate = take_step(alpha)
        if _satisfies_wolfe(point, origin, c1=c1, c2=c2, epsilon=epsilon):
            return (
                point.alpha,
                W_candidate,
                point.cost,
                gradient_candidate,
                r_candidate,
                True,
            )
        if point.slope >= 0 or point.cost > origin.cost + epsilon:
            upper = point
        else:
            lower = point
        if abs(upper.alpha - lower.alpha) < 1e-10:
            break
    return alpha0, W, cost, gradient, np.eye(W.shape[1]), False


class FixedTargetGrassmannCG:
    """Historical-style analytic CG for a frozen RDM target."""

    theta = 1.0
    eta = 0.4

    def __init__(self, block_length: int, options: CGOptions | None = None):
        self.block_length = int(block_length)
        self.options = options or CGOptions()

    def _hager_zhang(
        self,
        gradient: Array,
        previous_gradient: Array,
        preconditioned: Array,
        previous_preconditioned: Array,
        previous_direction: Array,
    ) -> float:
        dd = _real_metric(previous_direction, previous_direction)
        dg = _real_metric(previous_direction, gradient)
        dg_previous = _real_metric(previous_direction, previous_gradient)
        dy = dg - dg_previous
        if abs(dy) < 1e-300 or abs(dd) < 1e-300:
            return 0.0
        g_pg = _real_metric(gradient, preconditioned)
        g_previous_pg_previous = _real_metric(
            previous_gradient, previous_preconditioned
        )
        g_pg_previous = _real_metric(gradient, previous_preconditioned)
        g_previous_pg = _real_metric(previous_gradient, preconditioned)
        g_py = g_pg - g_pg_previous
        y_py = g_pg + g_previous_pg_previous - g_pg_previous - g_previous_pg
        beta = (g_py - self.theta * (y_py / dy) * dg) / dy
        eta = self.eta * dg_previous / dd
        return float(max(beta, eta))

    def optimize(self, A0: Array, rho_target: Array) -> FixedTargetCGResult:
        A0 = np.asarray(A0, dtype=np.complex128)
        d, D, _ = A0.shape
        W = _lomps_matrix_to_old_matrix(stack_tensor(A0), d, D)
        cost_model = FixedTargetHSCost(
            rho_target,
            local_dimension=d,
            block_length=self.block_length,
        )
        started = time.perf_counter()
        cost, gradient, r = cost_model.fg(W)
        gradient_norm = float(np.sqrt(max(_real_metric(gradient, gradient), 0.0)))
        history: list[FixedTargetCGRecord] = []
        best_W = W.copy()
        best_cost = float(np.real(cost))
        best_gradient_norm = gradient_norm
        status = "maximum_iterations"

        preconditioned = _precondition(gradient, r) if self.options.precondition else gradient
        alpha = self.options.initial_step
        if alpha is None:
            alpha = 1.0 / np.sqrt(max(_real_metric(preconditioned, preconditioned), 1e-300))

        previous_gradient = gradient
        previous_preconditioned = preconditioned
        previous_direction = -preconditioned
        beta = 0.0

        for iteration in range(self.options.max_iterations + 1):
            iteration_started = time.perf_counter()
            real_cost = float(np.real(cost))
            if self.options.verbose and iteration % 10 == 0:
                print(
                    f"fixed-target CG iter={iteration:4d} "
                    f"cost={real_cost:.6e} grad={gradient_norm:.6e} "
                    f"alpha={float(alpha):.3e}",
                    flush=True,
                )
            if real_cost < best_cost:
                best_W = W.copy()
                best_cost = real_cost
                best_gradient_norm = gradient_norm

            reached_cost = self.options.cost_tolerance > 0 and real_cost <= self.options.cost_tolerance
            reached_gradient = gradient_norm <= self.options.gradient_tolerance
            reached_old_tiny_cost = abs(cost) <= 2e-16
            time_limit = (
                self.options.maximum_seconds > 0
                and time.perf_counter() - started >= self.options.maximum_seconds
            )
            if (
                reached_cost
                or reached_gradient
                or reached_old_tiny_cost
                or time_limit
                or iteration == self.options.max_iterations
            ):
                if reached_cost:
                    status = "cost_tolerance"
                elif reached_gradient:
                    status = "gradient_tolerance"
                elif reached_old_tiny_cost:
                    status = "tiny_cost"
                elif time_limit:
                    status = "time_limit"
                history.append(
                    FixedTargetCGRecord(
                        iteration=iteration,
                        cost=real_cost,
                        gradient_norm=gradient_norm,
                        step_size=0.0,
                        beta=beta,
                        accepted=False,
                        elapsed_seconds=time.perf_counter() - iteration_started,
                    )
                )
                break

            if self.options.precondition:
                preconditioned = _precondition(gradient, r)
            else:
                preconditioned = gradient
            direction = -preconditioned
            if iteration % self.options.restart:
                beta = self._hager_zhang(
                    gradient,
                    previous_gradient,
                    preconditioned,
                    previous_preconditioned,
                    previous_direction,
                )
                direction = direction + previous_direction * beta
            else:
                beta = 0.0

            W_previous = W
            direction_previous = direction
            gradient_previous = gradient
            preconditioned_previous = preconditioned
            step_alpha, W_new, cost_new, gradient_new, r_new, accepted = _line_search(
                cost_model.fg,
                W,
                direction,
                cost,
                gradient,
                float(alpha),
            )
            if not accepted:
                step_alpha = min(float(alpha), 0.1)
                W_new, _ = _retract(W, direction, step_alpha)
                cost_new, gradient_new, r_new = cost_model.fg(W_new)
            else:
                alpha = min(step_alpha * 1.1, 1.0)

            history.append(
                FixedTargetCGRecord(
                    iteration=iteration,
                    cost=real_cost,
                    gradient_norm=gradient_norm,
                    step_size=step_alpha,
                    beta=beta,
                    accepted=accepted,
                    elapsed_seconds=time.perf_counter() - iteration_started,
                )
            )

            W = W_new
            cost = cost_new
            gradient = gradient_new
            r = r_new
            gradient_norm = float(np.sqrt(max(_real_metric(gradient, gradient), 0.0)))
            previous_gradient = _transport(
                gradient_previous,
                W_previous,
                direction_previous,
                step_alpha,
                W,
            )
            previous_preconditioned = _transport(
                preconditioned_previous,
                W_previous,
                direction_previous,
                step_alpha,
                W,
            )
            previous_direction = _transport(
                direction_previous,
                W_previous,
                direction_previous,
                step_alpha,
                W,
            )

        return FixedTargetCGResult(
            W=_old_matrix_to_lomps_matrix(best_W, d, D),
            cost=best_cost,
            residual_norm=float(np.sqrt(max(best_cost, 0.0))),
            gradient_norm=best_gradient_norm,
            status=status,
            history=tuple(history),
        )


def optimize_fixed_target_cg(
    A0: Array,
    rho_target: Array,
    block_length: int,
    options: CGOptions | None = None,
) -> tuple[Array, FixedTargetCGResult]:
    result = FixedTargetGrassmannCG(block_length, options).optimize(A0, rho_target)
    d, D, _ = A0.shape
    return unstack_tensor(result.W, d, D), result
