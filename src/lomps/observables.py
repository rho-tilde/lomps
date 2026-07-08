"""Efficient local observable evaluation for left-canonical uMPS tensors."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np

from .optimizer import FixedPointSolver, optimizer_right_fixed_point
from .rdm import block_products, rdm_from_products


Array = np.ndarray


PAULI_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
PAULI_Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
PAULI_Z = np.diag([1.0, -1.0]).astype(np.complex128)


def one_site_rdm(A: Array, r: Array | None = None, *, solver: FixedPointSolver = "fast") -> Array:
    """Return the one-site RDM, solving the fixed point only if needed."""

    A = np.asarray(A, dtype=np.complex128)
    if r is None:
        r, _ = optimizer_right_fixed_point(A, solver)
    return np.einsum("sab,bc,tac->st", A, r, A.conj(), optimize=True).astype(
        np.complex128
    )


def local_rdm(
    A: Array,
    length: int,
    r: Array | None = None,
    *,
    solver: FixedPointSolver = "fast",
) -> Array:
    """Return a short local RDM with a cached fixed point when available."""

    A = np.asarray(A, dtype=np.complex128)
    if length == 1:
        return one_site_rdm(A, r, solver=solver)
    if r is None:
        r, _ = optimizer_right_fixed_point(A, solver)
    return rdm_from_products(block_products(A, length), r)


def expectation_from_rdm(rho: Array, operator: Array) -> complex:
    """Return ``Tr(rho operator)``."""

    return complex(np.trace(np.asarray(rho) @ np.asarray(operator)))


def local_expectations(
    A: Array,
    operators: Mapping[str, Array],
    *,
    solver: FixedPointSolver = "fast",
    real_if_close: bool = True,
) -> dict[str, complex | float]:
    """Evaluate local operators while reusing one fixed-point solve.

    Operators are grouped by their support length inferred from their square
    matrix dimension. For example, all Pauli one-site expectations share one
    one-site RDM, and a two-site energy operator uses only ``rho_2``.
    """

    A = np.asarray(A, dtype=np.complex128)
    d = A.shape[0]
    r, _ = optimizer_right_fixed_point(A, solver)
    rdms: dict[int, Array] = {}
    values: dict[str, complex | float] = {}
    for name, operator in operators.items():
        operator = np.asarray(operator, dtype=np.complex128)
        if operator.ndim != 2 or operator.shape[0] != operator.shape[1]:
            raise ValueError(f"operator {name!r} must be a square matrix")
        length = _operator_support_length(operator.shape[0], d)
        if length not in rdms:
            rdms[length] = local_rdm(A, length, r, solver=solver)
        value = expectation_from_rdm(rdms[length], operator)
        if real_if_close and abs(value.imag) <= 1e-12:
            values[name] = float(value.real)
        else:
            values[name] = value
    return values


def trajectory_expectations(
    states: Iterable[Array],
    operators: Mapping[str, Array],
    *,
    solver: FixedPointSolver = "fast",
    real_if_close: bool = True,
) -> dict[str, Array]:
    """Evaluate local observables for each tensor in a trajectory."""

    rows = [
        local_expectations(
            A,
            operators,
            solver=solver,
            real_if_close=real_if_close,
        )
        for A in states
    ]
    if not rows:
        return {name: np.array([], dtype=float) for name in operators}
    return {
        name: np.asarray([row[name] for row in rows])
        for name in operators
    }


def _operator_support_length(dimension: int, local_dimension: int) -> int:
    length = 0
    power = 1
    while power < dimension:
        power *= local_dimension
        length += 1
    if power != dimension:
        raise ValueError(
            "operator dimension is not a power of the local physical dimension"
        )
    return length
